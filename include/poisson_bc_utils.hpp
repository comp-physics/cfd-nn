/// @file poisson_bc_utils.hpp
/// @brief Unified boundary condition utilities for Poisson solvers
///
/// Provides common BC configuration and application logic shared by:
/// - PoissonSolver (SOR-based)
/// - MultigridPoissonSolver
/// - HyprePoissonSolver
/// - FFTPoissonSolver variants
///
/// This reduces duplication and ensures consistent BC handling across all backends.

#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"  // For PoissonBC enum

namespace nncfd {

/// Unified configuration for Poisson boundary conditions
struct PoissonBCConfig {
    PoissonBC x_lo = PoissonBC::Periodic;
    PoissonBC x_hi = PoissonBC::Periodic;
    PoissonBC y_lo = PoissonBC::Neumann;
    PoissonBC y_hi = PoissonBC::Neumann;
    PoissonBC z_lo = PoissonBC::Periodic;
    PoissonBC z_hi = PoissonBC::Periodic;
    double dirichlet_val = 0.0;

    /// Factory: all periodic BCs
    static PoissonBCConfig all_periodic() {
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Periodic, PoissonBC::Periodic, 0.0};
    }

    /// Factory: channel flow (periodic x/z, Neumann y)
    static PoissonBCConfig channel() {
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Periodic, PoissonBC::Periodic, 0.0};
    }

    /// Factory: duct flow (periodic x, Neumann y/z)
    static PoissonBCConfig duct() {
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Neumann, PoissonBC::Neumann, 0.0};
    }

    /// Factory: lid-driven cavity (all Neumann)
    static PoissonBCConfig cavity() {
        return {PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Neumann, PoissonBC::Neumann, 0.0};
    }

    /// Check if all BCs are Neumann (pressure needs gauge fixing)
    bool all_neumann() const {
        return x_lo == PoissonBC::Neumann && x_hi == PoissonBC::Neumann &&
               y_lo == PoissonBC::Neumann && y_hi == PoissonBC::Neumann &&
               z_lo == PoissonBC::Neumann && z_hi == PoissonBC::Neumann;
    }

    /// Check if any BC is periodic (affects FFT solver compatibility)
    bool has_periodic() const {
        return x_lo == PoissonBC::Periodic || x_hi == PoissonBC::Periodic ||
               y_lo == PoissonBC::Periodic || y_hi == PoissonBC::Periodic ||
               z_lo == PoissonBC::Periodic || z_hi == PoissonBC::Periodic;
    }
};

namespace poisson_bc {

/// Apply ghost cell values for a single direction
/// @param bc_type  BC type (Periodic, Neumann, Dirichlet)
/// @param p_ghost  Value at ghost cell
/// @param p_int    Value at adjacent interior cell
/// @param p_per    Value at periodic partner (for Periodic BC)
/// @param dir_val  Dirichlet value
/// @return Value to assign to ghost cell
inline double apply_1d(PoissonBC bc_type, double p_int, double p_per, double dir_val) {
    switch (bc_type) {
        case PoissonBC::Periodic:
            return p_per;
        case PoissonBC::Neumann:
            return p_int;
        case PoissonBC::Dirichlet:
            return 2.0 * dir_val - p_int;
    }
    return p_int;  // Default fallback
}

/// Apply all Poisson BCs to a scalar field (2D/3D unified)
/// @param p      Pressure field (in/out)
/// @param mesh   Computational mesh
/// @param bc     BC configuration
inline void apply_all(ScalarField& p, const Mesh& mesh, const PoissonBCConfig& bc) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;
    const bool is_2d = mesh.is2D();

    const int k_start = 0;
    const int k_stop = mesh.total_Nz();

    // X-direction boundaries
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int g = 0; g < Ng; ++g) {
                // Left (x_lo)
                int i_ghost = g;
                int i_interior = Ng;
                int i_periodic = Nx + Ng - 1 - g;
                p(i_ghost, j, k) = apply_1d(bc.x_lo, p(i_interior, j, k),
                                             p(i_periodic, j, k), bc.dirichlet_val);

                // Right (x_hi)
                i_ghost = Nx + Ng + g;
                i_interior = Nx + Ng - 1;
                i_periodic = Ng + g;
                p(i_ghost, j, k) = apply_1d(bc.x_hi, p(i_interior, j, k),
                                             p(i_periodic, j, k), bc.dirichlet_val);
            }
        }
    }

    // Y-direction boundaries
    for (int k = k_start; k < k_stop; ++k) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            for (int g = 0; g < Ng; ++g) {
                // Bottom (y_lo)
                int j_ghost = g;
                int j_interior = Ng;
                int j_periodic = Ny + Ng - 1 - g;
                p(i, j_ghost, k) = apply_1d(bc.y_lo, p(i, j_interior, k),
                                             p(i, j_periodic, k), bc.dirichlet_val);

                // Top (y_hi)
                j_ghost = Ny + Ng + g;
                j_interior = Ny + Ng - 1;
                j_periodic = Ng + g;
                p(i, j_ghost, k) = apply_1d(bc.y_hi, p(i, j_interior, k),
                                             p(i, j_periodic, k), bc.dirichlet_val);
            }
        }
    }

    // Z-direction boundaries (3D only)
    if (!is_2d) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                for (int g = 0; g < Ng; ++g) {
                    // Front (z_lo)
                    int k_ghost = g;
                    int k_interior = Ng;
                    int k_periodic = Nz + Ng - 1 - g;
                    p(i, j, k_ghost) = apply_1d(bc.z_lo, p(i, j, k_interior),
                                                 p(i, j, k_periodic), bc.dirichlet_val);

                    // Back (z_hi)
                    k_ghost = Nz + Ng + g;
                    k_interior = Nz + Ng - 1;
                    k_periodic = Ng + g;
                    p(i, j, k_ghost) = apply_1d(bc.z_hi, p(i, j, k_interior),
                                                 p(i, j, k_periodic), bc.dirichlet_val);
                }
            }
        }
    }
}

/// Subtract mean from field (for all-Neumann pressure gauge fixing)
inline void subtract_mean(ScalarField& p, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += p(i, j, k);
                ++count;
            }
        }
    }

    if (count > 0) {
        double mean = sum / count;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    p(i, j, k) -= mean;
                }
            }
        }
    }
}

} // namespace poisson_bc
} // namespace nncfd
