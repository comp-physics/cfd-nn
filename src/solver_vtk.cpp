/// @file solver_vtk.cpp
/// @brief VTK output implementation for RANSSolver
///
/// This file contains the VTK file writing functionality, separated from
/// the main solver implementation for better code organization.

#include "solver.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdint>

namespace nncfd {

/// Helper to write a double as big-endian binary
static void write_double_binary(std::ostream& os, double val) {
    // VTK binary format requires big-endian
    // Most systems are little-endian, so we need to swap bytes
    union {
        double d;
        uint64_t u;
    } conv;
    conv.d = val;

    // Check if system is little-endian and swap if needed
    uint32_t test = 1;
    bool is_little_endian = (*reinterpret_cast<char*>(&test) == 1);

    if (is_little_endian) {
        // Swap bytes for big-endian output
        uint64_t swapped = ((conv.u & 0x00000000000000FFULL) << 56) |
                          ((conv.u & 0x000000000000FF00ULL) << 40) |
                          ((conv.u & 0x0000000000FF0000ULL) << 24) |
                          ((conv.u & 0x00000000FF000000ULL) << 8)  |
                          ((conv.u & 0x000000FF00000000ULL) >> 8)  |
                          ((conv.u & 0x0000FF0000000000ULL) >> 24) |
                          ((conv.u & 0x00FF000000000000ULL) >> 40) |
                          ((conv.u & 0xFF00000000000000ULL) >> 56);
        os.write(reinterpret_cast<const char*>(&swapped), sizeof(double));
    } else {
        os.write(reinterpret_cast<const char*>(&conv.d), sizeof(double));
    }
}

void RANSSolver::write_vtk(const std::string& filename) const {
    // NaN/Inf GUARD: Check before writing output
    // Catch NaNs before they're written to files
    check_for_nan_inf(step_count_);

#ifdef USE_GPU_OFFLOAD
    // Download solution fields from GPU for I/O (only what's needed!)
    const_cast<RANSSolver*>(this)->sync_solution_from_gpu();
    // Transport fields only if they'll be written (turbulence model active)
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        const_cast<RANSSolver*>(this)->sync_transport_from_gpu();
    }
#endif

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const bool is_2d = mesh_->is2D();
    const bool use_binary = config_.vtk_binary && !is_2d;  // Binary only for 3D (spectral analysis)

    // Open file in appropriate mode
    std::ofstream file;
    if (use_binary) {
        file.open(filename, std::ios::binary);
    } else {
        file.open(filename);
    }

    if (!file) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return;
    }

    // VTK header (always ASCII)
    file << "# vtk DataFile Version 3.0\n";
    file << "RANS simulation output\n";
    file << (use_binary ? "BINARY\n" : "ASCII\n");
    file << "DATASET STRUCTURED_POINTS\n";

    // Binary 3D output - streamlined for spectral analysis
    if (use_binary) {
        file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
        file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " " << mesh_->z_min << "\n";
        file << "SPACING " << mesh_->dx << " " << mesh_->dy << " " << mesh_->dz << "\n";
        file << "POINT_DATA " << Nx * Ny * Nz << "\n";

        // Velocity vector field (main data for spectral analysis)
        file << "VECTORS velocity double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    write_double_binary(file, velocity_.u_center(i, j, k));
                    write_double_binary(file, velocity_.v_center(i, j, k));
                    write_double_binary(file, velocity_.w_center(i, j, k));
                }
            }
        }

        // Pressure scalar field
        file << "\nSCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    write_double_binary(file, pressure_(i, j, k));
                }
            }
        }

        file.close();
        return;
    }

    // ASCII output (original code for 2D or when --vtk_ascii is specified)

    if (is_2d) {
        file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
        file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " 0\n";
        file << "SPACING " << mesh_->dx << " " << mesh_->dy << " 1\n";
        file << "POINT_DATA " << Nx * Ny << "\n";

        // Velocity vector field (interpolated from staggered grid to cell centers)
        // Use 2-component SCALARS for 2D (VTK VECTORS requires 3 components)
        file << "SCALARS velocity double 2\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double u_center = velocity_.u_center(i, j);
                double v_center = velocity_.v_center(i, j);
                file << u_center << " " << v_center << "\n";
            }
        }

        // Pressure scalar field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << pressure_(i, j) << "\n";
            }
        }

        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.magnitude(i, j) << "\n";
            }
        }

        // Eddy viscosity (if turbulence model is active)
        if (turb_model_) {
            file << "SCALARS nu_t double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << nu_t_(i, j) << "\n";
                }
            }
        }

        // Individual velocity components as scalars
        file << "SCALARS u double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.u_center(i, j) << "\n";
            }
        }

        file << "SCALARS v double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.v_center(i, j) << "\n";
            }
        }

        // Pressure gradients using central differences
        // For periodic BCs: wrap indices; for non-periodic: one-sided at boundaries
        const bool periodic_x = (poisson_bc_x_lo_ == PoissonBC::Periodic);
        const bool periodic_y = (poisson_bc_y_lo_ == PoissonBC::Periodic);

        // Helper lambdas for pressure gradient computation
        auto compute_dpdx_2d = [&](int i, int j) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (pressure_(ip, j) - pressure_(im, j)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (pressure_(i + 1, j) - pressure_(i, j)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (pressure_(i, j) - pressure_(i - 1, j)) / mesh_->dx;
                } else {
                    return (pressure_(i + 1, j) - pressure_(i - 1, j)) / (2.0 * mesh_->dx);
                }
            }
        };

        auto compute_dpdy_2d = [&](int i, int j) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (pressure_(i, jp) - pressure_(i, jm)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (pressure_(i, j + 1) - pressure_(i, j)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (pressure_(i, j) - pressure_(i, j - 1)) / mesh_->dy;
                } else {
                    return (pressure_(i, j + 1) - pressure_(i, j - 1)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Pressure gradient as 2-component vector
        file << "SCALARS pressure_gradient double 2\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdx_2d(i, j) << " " << compute_dpdy_2d(i, j) << "\n";
            }
        }

        // Pressure gradient components as scalars
        file << "SCALARS dP_dx double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdx_2d(i, j) << "\n";
            }
        }

        file << "SCALARS dP_dy double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdy_2d(i, j) << "\n";
            }
        }

        // Vorticity (omega_z = dv/dx - du/dy) - scalar in 2D
        // NOTE: Uses uniform dx/dy spacing. On stretched meshes, this is an approximation
        // suitable for visualization but not metrically consistent with the solver discretization.
        // Guard: skip vorticity output for degenerate meshes (need >= 2 cells per direction)
        const int nx_2d = mesh_->i_end() - mesh_->i_begin();
        const int ny_2d = mesh_->j_end() - mesh_->j_begin();
        if (nx_2d >= 2 && ny_2d >= 2) {
        auto compute_vorticity_2d = [&](int i, int j) -> double {
            // dv/dx
            double dvdx;
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                dvdx = (velocity_.v_center(ip, j) - velocity_.v_center(im, j)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    dvdx = (velocity_.v_center(i + 1, j) - velocity_.v_center(i, j)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    dvdx = (velocity_.v_center(i, j) - velocity_.v_center(i - 1, j)) / mesh_->dx;
                } else {
                    dvdx = (velocity_.v_center(i + 1, j) - velocity_.v_center(i - 1, j)) / (2.0 * mesh_->dx);
                }
            }

            // du/dy
            double dudy;
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                dudy = (velocity_.u_center(i, jp) - velocity_.u_center(i, jm)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    dudy = (velocity_.u_center(i, j + 1) - velocity_.u_center(i, j)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    dudy = (velocity_.u_center(i, j) - velocity_.u_center(i, j - 1)) / mesh_->dy;
                } else {
                    dudy = (velocity_.u_center(i, j + 1) - velocity_.u_center(i, j - 1)) / (2.0 * mesh_->dy);
                }
            }

            return dvdx - dudy;
        };

        file << "SCALARS vorticity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_vorticity_2d(i, j) << "\n";
            }
        }

        // Vorticity magnitude (same as |omega_z| in 2D)
        file << "SCALARS vorticity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << std::abs(compute_vorticity_2d(i, j)) << "\n";
            }
        }
        } // end degenerate mesh guard for 2D vorticity
    } else {
        // 3D output
        file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
        file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " " << mesh_->z_min << "\n";
        file << "SPACING " << mesh_->dx << " " << mesh_->dy << " " << mesh_->dz << "\n";
        file << "POINT_DATA " << Nx * Ny * Nz << "\n";

        // Velocity vector field (interpolated from staggered grid to cell centers)
        file << "VECTORS velocity double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double u_center = velocity_.u_center(i, j, k);
                    double v_center = velocity_.v_center(i, j, k);
                    double w_center = velocity_.w_center(i, j, k);
                    file << u_center << " " << v_center << " " << w_center << "\n";
                }
            }
        }

        // Pressure scalar field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << pressure_(i, j, k) << "\n";
                }
            }
        }

        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.magnitude(i, j, k) << "\n";
                }
            }
        }

        // Eddy viscosity (if turbulence model is active)
        if (turb_model_) {
            file << "SCALARS nu_t double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        file << nu_t_(i, j, k) << "\n";
                    }
                }
            }
        }

        // Individual velocity components as scalars
        file << "SCALARS u double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.u_center(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS v double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.v_center(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS w double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.w_center(i, j, k) << "\n";
                }
            }
        }

        // Pressure gradients using central differences
        // For periodic BCs: wrap indices; for non-periodic: one-sided at boundaries
        const bool periodic_x = (poisson_bc_x_lo_ == PoissonBC::Periodic);
        const bool periodic_y = (poisson_bc_y_lo_ == PoissonBC::Periodic);
        const bool periodic_z = (poisson_bc_z_lo_ == PoissonBC::Periodic);

        // Helper lambdas for pressure gradient computation
        auto compute_dpdx_3d = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (pressure_(ip, j, k) - pressure_(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (pressure_(i + 1, j, k) - pressure_(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (pressure_(i + 1, j, k) - pressure_(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        auto compute_dpdy_3d = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (pressure_(i, jp, k) - pressure_(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (pressure_(i, j + 1, k) - pressure_(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (pressure_(i, j + 1, k) - pressure_(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        auto compute_dpdz_3d = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (pressure_(i, j, kp) - pressure_(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (pressure_(i, j, k + 1) - pressure_(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (pressure_(i, j, k + 1) - pressure_(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Pressure gradient as vector
        file << "VECTORS pressure_gradient double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdx_3d(i, j, k) << " "
                         << compute_dpdy_3d(i, j, k) << " "
                         << compute_dpdz_3d(i, j, k) << "\n";
                }
            }
        }

        // Pressure gradient components as scalars
        file << "SCALARS dP_dx double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdx_3d(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS dP_dy double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdy_3d(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS dP_dz double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdz_3d(i, j, k) << "\n";
                }
            }
        }

        // Vorticity (vector in 3D):
        //   omega_x = dw/dy - dv/dz
        //   omega_y = du/dz - dw/dx
        //   omega_z = dv/dx - du/dy
        // NOTE: Uses uniform dx/dy/dz spacing. On stretched meshes, this is an approximation
        // suitable for visualization but not metrically consistent with the solver discretization.
        // Guard: skip vorticity/Q-criterion output for degenerate meshes (need >= 2 cells per direction)
        const int nx_3d = mesh_->i_end() - mesh_->i_begin();
        const int ny_3d = mesh_->j_end() - mesh_->j_begin();
        const int nz_3d = mesh_->k_end() - mesh_->k_begin();
        if (nx_3d >= 2 && ny_3d >= 2 && nz_3d >= 2) {

        // Helper lambda for dw/dy
        auto compute_dwdy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.w_center(i, jp, k) - velocity_.w_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.w_center(i, j + 1, k) - velocity_.w_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.w_center(i, j + 1, k) - velocity_.w_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Helper lambda for dv/dz
        auto compute_dvdz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.v_center(i, j, kp) - velocity_.v_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.v_center(i, j, k + 1) - velocity_.v_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.v_center(i, j, k + 1) - velocity_.v_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Helper lambda for du/dz
        auto compute_dudz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.u_center(i, j, kp) - velocity_.u_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.u_center(i, j, k + 1) - velocity_.u_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.u_center(i, j, k + 1) - velocity_.u_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Helper lambda for dw/dx
        auto compute_dwdx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.w_center(ip, j, k) - velocity_.w_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.w_center(i + 1, j, k) - velocity_.w_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.w_center(i + 1, j, k) - velocity_.w_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for dv/dx
        auto compute_dvdx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.v_center(ip, j, k) - velocity_.v_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.v_center(i + 1, j, k) - velocity_.v_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.v_center(i + 1, j, k) - velocity_.v_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for du/dy
        auto compute_dudy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.u_center(i, jp, k) - velocity_.u_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.u_center(i, j + 1, k) - velocity_.u_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.u_center(i, j + 1, k) - velocity_.u_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Vorticity components
        auto compute_omega_x = [&](int i, int j, int k) -> double {
            return compute_dwdy(i, j, k) - compute_dvdz(i, j, k);
        };

        auto compute_omega_y = [&](int i, int j, int k) -> double {
            return compute_dudz(i, j, k) - compute_dwdx(i, j, k);
        };

        auto compute_omega_z = [&](int i, int j, int k) -> double {
            return compute_dvdx(i, j, k) - compute_dudy(i, j, k);
        };

        // Vorticity vector field
        file << "VECTORS vorticity double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_omega_x(i, j, k) << " "
                         << compute_omega_y(i, j, k) << " "
                         << compute_omega_z(i, j, k) << "\n";
                }
            }
        }

        // Vorticity magnitude
        file << "SCALARS vorticity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double ox = compute_omega_x(i, j, k);
                    double oy = compute_omega_y(i, j, k);
                    double oz = compute_omega_z(i, j, k);
                    file << std::sqrt(ox*ox + oy*oy + oz*oz) << "\n";
                }
            }
        }

        // Q-criterion: Q = 0.5 * (||Omega||^2 - ||S||^2)
        // Positive Q indicates vortex cores (rotation dominates strain)
        // Requires diagonal derivatives (dudx, dvdy, dwdz) in addition to existing off-diagonals
        // NOTE: Uses uniform dx/dy/dz spacing (same assumption as vorticity above).
        // TODO: For large grids, consider precomputing the 3x3 gradient tensor per cell
        //       to avoid redundant lambda calls in the output loop.

        // Helper lambda for du/dx
        auto compute_dudx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.u_center(ip, j, k) - velocity_.u_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.u_center(i + 1, j, k) - velocity_.u_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.u_center(i + 1, j, k) - velocity_.u_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for dv/dy
        auto compute_dvdy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.v_center(i, jp, k) - velocity_.v_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.v_center(i, j + 1, k) - velocity_.v_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.v_center(i, j + 1, k) - velocity_.v_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Helper lambda for dw/dz
        auto compute_dwdz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.w_center(i, j, kp) - velocity_.w_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.w_center(i, j, k + 1) - velocity_.w_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.w_center(i, j, k + 1) - velocity_.w_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        file << "SCALARS Q_criterion double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    // Velocity gradients (reuse existing lambdas + new diagonal ones)
                    double dudx = compute_dudx(i, j, k);
                    double dudy = compute_dudy(i, j, k);
                    double dudz = compute_dudz(i, j, k);
                    double dvdx = compute_dvdx(i, j, k);
                    double dvdy = compute_dvdy(i, j, k);
                    double dvdz = compute_dvdz(i, j, k);
                    double dwdx = compute_dwdx(i, j, k);
                    double dwdy = compute_dwdy(i, j, k);
                    double dwdz = compute_dwdz(i, j, k);

                    // Strain rate tensor components: S_ij = 0.5*(du_i/dx_j + du_j/dx_i)
                    double Sxx = dudx;
                    double Syy = dvdy;
                    double Szz = dwdz;
                    double Sxy = 0.5 * (dudy + dvdx);
                    double Sxz = 0.5 * (dudz + dwdx);
                    double Syz = 0.5 * (dvdz + dwdy);

                    // Rotation rate tensor components: Omega_ij = 0.5*(du_i/dx_j - du_j/dx_i)
                    double Oxy = 0.5 * (dudy - dvdx);
                    double Oxz = 0.5 * (dudz - dwdx);
                    double Oyz = 0.5 * (dvdz - dwdy);

                    // Squared Frobenius norms
                    double S_sq = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);
                    double O_sq = 2.0 * (Oxy*Oxy + Oxz*Oxz + Oyz*Oyz);

                    // Q-criterion
                    double Q = 0.5 * (O_sq - S_sq);
                    file << Q << "\n";
                }
            }
        }
        } // end degenerate mesh guard for 3D vorticity/Q-criterion
    }

    file.close();
}

}  // namespace nncfd
