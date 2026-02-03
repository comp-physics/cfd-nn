// Core operators for RANSSolver (BC, convection, diffusion, divergence, projection)
// Split from solver.cpp to avoid nvc++ compiler crash on large files

#include "solver.hpp"
#include "gpu_utils.hpp"
#include "profiling.hpp"
#include "stencil_operators.hpp"
#include "solver_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

// NVTX macros for profiling
#ifdef GPU_PROFILE_KERNELS
#if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#elif __has_include(<nvToolsExt.h>)
    #include <nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif

namespace nncfd {

// Import kernel functions from solver_kernels.hpp
using namespace nncfd::kernels;

void RANSSolver::apply_velocity_bc() {
    NVTX_PUSH("apply_velocity_bc");
    
    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();
    
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int u_total_Ny = Ny + 2 * Ng;
    const int v_total_Nx = Nx + 2 * Ng;

    const bool x_lo_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool x_lo_noslip   = (velocity_bc_.x_lo == VelocityBC::NoSlip);
    const bool x_lo_inflow   = (velocity_bc_.x_lo == VelocityBC::Inflow);
    const bool x_hi_periodic = (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool x_hi_noslip   = (velocity_bc_.x_hi == VelocityBC::NoSlip);
    const bool x_hi_outflow  = (velocity_bc_.x_hi == VelocityBC::Outflow);

    const bool y_lo_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool y_lo_noslip   = (velocity_bc_.y_lo == VelocityBC::NoSlip);
    const bool y_hi_periodic = (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool y_hi_noslip   = (velocity_bc_.y_hi == VelocityBC::NoSlip);

    // Validate that all BCs are supported
    if (!x_lo_periodic && !x_lo_noslip && !x_lo_inflow) {
        throw std::runtime_error("Unsupported velocity BC type for x_lo");
    }
    if (!x_hi_periodic && !x_hi_noslip && !x_hi_outflow) {
        throw std::runtime_error("Unsupported velocity BC type for x_hi");
    }
    if (!y_lo_periodic && !y_lo_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_lo (only Periodic and NoSlip are implemented)");
    }
    if (!y_hi_periodic && !y_hi_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_hi (only Periodic and NoSlip are implemented)");
    }

    double* u_ptr = v.u_face;
    double* v_ptr = v.v_face;
    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();

    // For 3D, apply x/y BCs to ALL z-planes (interior and ghost)
    // This is necessary because the z-BC code assumes x/y BCs are already applied
    // Nz_total = Nz + 2*Ng for 3D, 1 for 2D
    const int Nz_total = mesh_->is2D() ? 1 : (v.Nz + 2*Ng);
    const int u_plane_stride = mesh_->is2D() ? 0 : v.u_plane_stride;
    const int v_plane_stride = mesh_->is2D() ? 0 : v.v_plane_stride;

    // Apply u BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_u_x_bc = u_total_Ny * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_u_x_bc; ++idx) {
        int j = idx % u_total_Ny;
        int g = (idx / u_total_Ny) % Ng;
        int k = idx / (u_total_Ny * Ng);  // k = 0 to Nz_total-1 covers all planes
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, u_plane_ptr);
    }

    // Apply u BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_u_y_bc = (Nx + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_u_y_bc; ++idx) {
        int u_x_size = Nx + 1 + 2 * Ng;
        int i = idx % u_x_size;
        int g = (idx / u_x_size) % Ng;
        int k = idx / (u_x_size * Ng);
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, u_plane_ptr);
    }

    // Apply v BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_v_x_bc = (Ny + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_v_x_bc; ++idx) {
        int v_y_size = Ny + 1 + 2 * Ng;
        int j = idx % v_y_size;
        int g = (idx / v_y_size) % Ng;
        int k = idx / (v_y_size * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, v_plane_ptr);
    }

    // Apply v BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_v_y_bc = v_total_Nx * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_v_y_bc; ++idx) {
        int i = idx % v_total_Nx;
        int g = (idx / v_total_Nx) % Ng;
        int k = idx / (v_total_Nx * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, v_plane_ptr);
    }

    // CORNER FIX: For fully periodic domains, apply x-direction BCs again
    // to ensure corner ghosts are correctly wrapped after y-direction BCs modified them
    if (x_lo_periodic && x_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_u_x_bc; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, u_plane_ptr);
        }
    }

    if (y_lo_periodic && y_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_v_y_bc; ++idx) {
            int i = idx % v_total_Nx;
            int g = (idx / v_total_Nx) % Ng;
            int k = idx / (v_total_Nx * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, v_plane_ptr);
        }
    }

    // Outflow BC at x_hi: zero-gradient (Neumann) for ghost cells
    // Copy interior values to ghost cells: ghost[Ng+Nx+1+g] = interior[Ng+Nx-1-g]
    // NOTE: Loop bounds precomputed to avoid nvc++ 25.5 compiler bug (signal 11)
    if (x_hi_outflow) {
        // u outflow (normal velocity): apply to all j, all k-planes
        const int n_u_outflow = u_total_Ny * Ng * Nz_total;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ng, u_stride, u_plane_stride, u_total_Ny, n_u_outflow)
        for (int idx = 0; idx < n_u_outflow; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            int i_ghost = Ng + Nx + 1 + g;     // Ghost face beyond outflow
            int i_interior = Ng + Nx - 1 - g;  // Interior face
            u_plane_ptr[j * u_stride + i_ghost] = u_plane_ptr[j * u_stride + i_interior];
        }

        // v outflow (tangential velocity): apply to all j, all k-planes
        const int v_y_total = Ny + 1 + 2 * Ng;
        const int n_v_outflow = v_y_total * Ng * Nz_total;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ng, v_stride, v_plane_stride, v_y_total, n_v_outflow)
        for (int idx = 0; idx < n_v_outflow; ++idx) {
            int j = idx % v_y_total;
            int g = (idx / v_y_total) % Ng;
            int k = idx / (v_y_total * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            int i_ghost = Ng + Nx + g;         // Ghost cell beyond outflow
            int i_interior = Ng + Nx - 1 - g;  // Interior cell
            v_plane_ptr[j * v_stride + i_ghost] = v_plane_ptr[j * v_stride + i_interior];
        }
    }

    // Inflow BC at x_lo: zero-gradient for ghost cells (inlet face set by recycling)
    // NOTE: Loop bounds precomputed to avoid nvc++ 25.5 compiler bug (signal 11)
    if (x_lo_inflow) {
        // u inflow ghost cells: ghost[Ng-1-g] = interior[Ng+g]
        const int n_u_inflow = u_total_Ny * Ng * Nz_total;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Ng, u_stride, u_plane_stride, u_total_Ny, n_u_inflow)
        for (int idx = 0; idx < n_u_inflow; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            int i_ghost = Ng - 1 - g;     // Ghost face before inlet
            int i_interior = Ng + g;      // Interior face
            u_plane_ptr[j * u_stride + i_ghost] = u_plane_ptr[j * u_stride + i_interior];
        }

        // v inflow ghost cells
        const int v_y_total = Ny + 1 + 2 * Ng;
        const int n_v_inflow = v_y_total * Ng * Nz_total;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Ng, v_stride, v_plane_stride, v_y_total, n_v_inflow)
        for (int idx = 0; idx < n_v_inflow; ++idx) {
            int j = idx % v_y_total;
            int g = (idx / v_y_total) % Ng;
            int k = idx / (v_y_total * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            int i_ghost = Ng - 1 - g;     // Ghost cell before inlet
            int i_interior = Ng + g;      // Interior cell
            v_plane_ptr[j * v_stride + i_ghost] = v_plane_ptr[j * v_stride + i_interior];
        }
    }

    // 3D z-direction boundary conditions
    if (!mesh_->is2D()) {
        const int Nz = v.Nz;
        // u_plane_stride and v_plane_stride already defined in outer scope
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        double* w_ptr = v.w_face;
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

        const bool z_lo_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic);
        const bool z_hi_periodic = (velocity_bc_.z_hi == VelocityBC::Periodic);
        const bool z_lo_noslip = (velocity_bc_.z_lo == VelocityBC::NoSlip);
        const bool z_hi_noslip = (velocity_bc_.z_hi == VelocityBC::NoSlip);

        // Validate that z-direction BCs are supported (Inflow/Outflow not implemented)
        if (!z_lo_periodic && !z_lo_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_lo (only Periodic and NoSlip are implemented)");
        }
        if (!z_hi_periodic && !z_hi_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_hi (only Periodic and NoSlip are implemented)");
        }

        // Apply u BCs in z-direction (for all x-faces, all y rows)
        // Each x-face: (Nx+1) i-values, (Ny) j-values, Ng ghost layers at each z-end
        const int n_u_z_bc = (Nx + 1 + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, u_stride, u_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_u_z_bc; ++idx) {
            int i = idx % (Nx + 1 + 2*Ng);
            int j = (idx / (Nx + 1 + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 1 + 2*Ng) * (Ny + 2*Ng));
            // z-lo ghost: k = Ng-1-g = Ng-1, Ng-2, ... for g=0,1,...
            // z-hi ghost: k = Ng+Nz+g
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;        // First interior k
            int src_hi = Ng + Nz - 1;  // Last interior k
            int idx_lo = k_lo * u_plane_stride + j * u_stride + i;
            int idx_hi = k_hi * u_plane_stride + j * u_stride + i;
            int idx_src_lo = src_lo * u_plane_stride + j * u_stride + i;
            int idx_src_hi = src_hi * u_plane_stride + j * u_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // Periodic: copy from opposite interior boundary
                u_ptr[idx_lo] = u_ptr[(Ng + Nz - 1 - g) * u_plane_stride + j * u_stride + i];
                u_ptr[idx_hi] = u_ptr[(Ng + g) * u_plane_stride + j * u_stride + i];
            } else {
                if (z_lo_noslip) u_ptr[idx_lo] = -u_ptr[idx_src_lo];
                if (z_hi_noslip) u_ptr[idx_hi] = -u_ptr[idx_src_hi];
            }
        }

        // Apply v BCs in z-direction
        const int n_v_z_bc = (Nx + 2*Ng) * (Ny + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, v_stride, v_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_v_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 1 + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;
            int src_hi = Ng + Nz - 1;
            int idx_lo = k_lo * v_plane_stride + j * v_stride + i;
            int idx_hi = k_hi * v_plane_stride + j * v_stride + i;
            int idx_src_lo = src_lo * v_plane_stride + j * v_stride + i;
            int idx_src_hi = src_hi * v_plane_stride + j * v_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                v_ptr[idx_lo] = v_ptr[(Ng + Nz - 1 - g) * v_plane_stride + j * v_stride + i];
                v_ptr[idx_hi] = v_ptr[(Ng + g) * v_plane_stride + j * v_stride + i];
            } else {
                if (z_lo_noslip) v_ptr[idx_lo] = -v_ptr[idx_src_lo];
                if (z_hi_noslip) v_ptr[idx_hi] = -v_ptr[idx_src_hi];
            }
        }

        // Apply w BCs in z-direction (w is at z-faces, so different treatment)
        // For periodic: w at k=Ng and k=Ng+Nz should be same (wrap around)
        const int n_w_z_bc = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, w_stride, w_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_w_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + 1 + g;  // Note: Nz+1 z-faces for w
            int idx_lo = k_lo * w_plane_stride + j * w_stride + i;
            int idx_hi = k_hi * w_plane_stride + j * w_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // CRITICAL for staggered grid with periodic BCs:
                // The topmost interior face (Ng+Nz) IS the bottommost interior face (Ng)
                // They represent the same physical location in a periodic domain
                if (g == 0) {
                    w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] =
                        w_ptr[Ng * w_plane_stride + j * w_stride + i];
                }
                // For w at z-faces with periodic BC:
                // Ghost at k=Ng-1-g gets value from k=Ng+Nz-1-g (interior near hi)
                // Ghost at k=Ng+Nz+1+g gets value from k=Ng+1+g (interior near lo)
                w_ptr[idx_lo] = w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                w_ptr[idx_hi] = w_ptr[(Ng + 1 + g) * w_plane_stride + j * w_stride + i];
            } else {
                // For no-slip: w at boundaries should be zero (normal velocity)
                if (z_lo_noslip) {
                    // w at k=Ng (first interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_lo] = -w_ptr[(Ng + g + 1) * w_plane_stride + j * w_stride + i];
                }
                if (z_hi_noslip) {
                    // w at k=Ng+Nz (last interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_hi] = -w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                }
            }
        }

        // Apply w BCs in x and y directions
        // w in x-direction
        const int n_w_x_bc = (Ny + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ng, w_stride, w_plane_stride, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_w_x_bc; ++idx) {
            int j = idx % (Ny + 2*Ng);
            int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
            int i_lo = Ng - 1 - g;
            int i_hi = Ng + Nx + g;
            int idx_lo = k * w_plane_stride + j * w_stride + i_lo;
            int idx_hi = k * w_plane_stride + j * w_stride + i_hi;

            if (x_lo_periodic && x_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
            } else {
                if (x_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
                if (x_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
            }
        }

        // w inflow/outflow in x-direction (zero-gradient)
        if (x_lo_inflow) {
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane_stride)
            for (int idx = 0; idx < n_w_x_bc; ++idx) {
                int j = idx % (Ny + 2*Ng);
                int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
                int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
                int i_ghost = Ng - 1 - g;
                int i_interior = Ng + g;
                w_ptr[k * w_plane_stride + j * w_stride + i_ghost] =
                    w_ptr[k * w_plane_stride + j * w_stride + i_interior];
            }
        }
        if (x_hi_outflow) {
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane_stride)
            for (int idx = 0; idx < n_w_x_bc; ++idx) {
                int j = idx % (Ny + 2*Ng);
                int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
                int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
                int i_ghost = Ng + Nx + g;
                int i_interior = Ng + Nx - 1 - g;
                w_ptr[k * w_plane_stride + j * w_stride + i_ghost] =
                    w_ptr[k * w_plane_stride + j * w_stride + i_interior];
            }
        }

        // w in y-direction
        const int n_w_y_bc = (Nx + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Ny, Ng, w_stride, w_plane_stride, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_w_y_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int k = (idx / (Nx + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Nz + 1 + 2*Ng));
            int j_lo = Ng - 1 - g;
            int j_hi = Ng + Ny + g;
            int idx_lo = k * w_plane_stride + j_lo * w_stride + i;
            int idx_hi = k * w_plane_stride + j_hi * w_stride + i;

            if (y_lo_periodic && y_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
            } else {
                if (y_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
                if (y_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
            }
        }
        // NOTE: x/y BCs for u and v across all k-planes are now handled
        // above (lines 1268-1360) using the staggered kernel functions
        // which properly handle staggered grid periodic BCs.
    }

    NVTX_POP();  // End apply_velocity_bc
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    NVTX_SCOPE_CONVECT("solver:convective_term");

    (void)vel;   // Unused - always operates on velocity_ via view
    (void)conv;  // Unused - always operates on conv_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);
    const bool use_skew = (config_.convective_scheme == ConvectiveScheme::Skew);
    const bool use_upwind2 = (config_.convective_scheme == ConvectiveScheme::Upwind2);
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary safety checks
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    double*       conv_u_ptr = v.conv_u;
    double*       conv_v_ptr = v.conv_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const double* w_ptr = v.w_face;
        double*       conv_w_ptr = v.conv_w;

        const int n_u_faces = (Nx + 1) * Ny * Nz;
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        const int n_w_faces = Nx * Ny * (Nz + 1);
        const int conv_u_stride = u_stride;
        const int conv_u_plane_stride = u_plane_stride;
        const int conv_v_stride = v_stride;
        const int conv_v_plane_stride = v_plane_stride;
        const int conv_w_stride = w_stride;
        const int conv_w_plane_stride = w_plane_stride;

        if (use_O4 && use_skew) {
            // O4 Skew-symmetric advection (energy-conserving with 4th-order advective derivatives)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_skew) {
            // O2 Skew-symmetric (energy-conserving) advection
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_upwind2) {
            // 2nd-order upwind with minmod limiter
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_O4 && use_central) {
            // O4 Central advection (4th-order derivatives, hybrid O4/O2 near boundaries)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else {
            // Central or 1st-order upwind (O2 path)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        }
        return;
    }

    // 2D path
    const int n_u_faces = (Nx + 1) * Ny;
    const int n_v_faces = Nx * (Ny + 1);
    const int conv_u_stride = u_stride;
    const int conv_v_stride = v_stride;

    // Warn once if O4 requested but not implemented for 2D advection
    if (use_O4 && (use_skew || use_central)) {
        static bool warned_o4_2d = false;
        if (!warned_o4_2d) {
            std::cerr << "[Solver] WARNING: space_order=4 requested but O4 advection kernels "
                      << "are not implemented for 2D. Using O2 advection.\n";
            warned_o4_2d = true;
        }
    }

    if (use_skew) {
        // Skew-symmetric (energy-conserving) advection - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_u_stride,
                                            dx, dy, u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_v_stride,
                                            dx, dy, u_ptr, v_ptr, conv_v_ptr);
        }
    } else if (use_upwind2) {
        // 2nd-order upwind with minmod limiter - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_u_stride,
                                               dx, dy, u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_v_stride,
                                               dx, dy, u_ptr, v_ptr, conv_v_ptr);
        }
    } else {
        // Central or 1st-order upwind (original path) - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i_local = idx % (Nx + 1);
            int j_local = idx / (Nx + 1);
            int i = i_local + Ng;
            int j = j_local + Ng;

            convective_u_face_kernel_staggered(i, j, u_stride, v_stride, u_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i_local = idx % Nx;
            int j_local = idx / Nx;
            int i = i_local + Ng;
            int j = j_local + Ng;

            convective_v_face_kernel_staggered(i, j, u_stride, v_stride, v_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_v_ptr);
        }
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff,
                                        VectorField& diff) {
    NVTX_SCOPE_DIFFUSE("solver:diffusive_term");

    (void)vel;     // Unused - always operates on velocity_ via view
    (void)nu_eff;  // Unused - always operates on nu_eff_ via view
    (void)diff;    // Unused - always operates on diff_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int nu_stride = v.cell_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t nu_total_size = field_total_size_;

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    const double* nu_ptr     = v.nu_eff;
    double*       diff_u_ptr = v.diff_u;
    double*       diff_v_ptr = v.diff_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int nu_plane_stride = v.cell_plane_stride;
        const double* w_ptr = v.w_face;
        double*       diff_w_ptr = v.diff_w;

        // Compute u-momentum diffusion at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, dz, u_stride, u_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            diffusive_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, nu_stride, nu_plane_stride, u_stride, u_plane_stride,
                dx, dy, dz, u_ptr, nu_ptr, diff_u_ptr);
        }

        // Compute v-momentum diffusion at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, dz, v_stride, v_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            diffusive_v_face_kernel_staggered_3d(i, j, k,
                v_stride, v_plane_stride, nu_stride, nu_plane_stride, v_stride, v_plane_stride,
                dx, dy, dz, v_ptr, nu_ptr, diff_v_ptr);
        }

        // Compute w-momentum diffusion at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size], nu_ptr[0:nu_total_size], diff_w_ptr[0:w_total_size]) \
            firstprivate(dx, dy, dz, w_stride, w_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            diffusive_w_face_kernel_staggered_3d(i, j, k,
                w_stride, w_plane_stride, nu_stride, nu_plane_stride, w_stride, w_plane_stride,
                dx, dy, dz, w_ptr, nu_ptr, diff_w_ptr);
        }
        return;
    }

    // 2D path
    // Compute u-momentum diffusion at x-faces
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
        firstprivate(dx, dy, u_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_u_face_kernel_staggered(i, j, u_stride, nu_stride, u_stride, dx, dy,
                                         u_ptr, nu_ptr, diff_u_ptr);
    }

    // Compute v-momentum diffusion at y-faces
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
        firstprivate(dx, dy, v_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_v_face_kernel_staggered(i, j, v_stride, nu_stride, v_stride, dx, dy,
                                         v_ptr, nu_ptr, diff_v_ptr);
    }
}

void RANSSolver::compute_divergence(VelocityWhich which, ScalarField& div) {
    (void)div;  // Unused - always operates on div_velocity_ via view

    // Get unified view
    auto v = get_solver_view();
    auto vel_ptrs = select_face_velocity(v, which);

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int div_stride = v.cell_stride;

    // O4 spatial discretization for divergence (Dfc_O4)
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary handling
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    const int u_stride = vel_ptrs.u_stride;
    const int v_stride = vel_ptrs.v_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t div_total_size = field_total_size_;

    const double* u_ptr = vel_ptrs.u;
    const double* v_ptr = vel_ptrs.v;
    double* div_ptr = v.div;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = vel_ptrs.u_plane_stride;
        const int v_plane_stride = vel_ptrs.v_plane_stride;
        const int w_stride = vel_ptrs.w_stride;
        const int w_plane_stride = vel_ptrs.w_plane_stride;
        const int div_plane_stride = v.cell_plane_stride;
        const double* w_ptr = vel_ptrs.w;

        const int n_cells = Nx * Ny * Nz;
        if (use_O4) {
            // O4 divergence with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], div_ptr[0:div_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Nz, Ng, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_cells; ++idx) {
                const int i = idx % Nx + Ng;
                const int j = (idx / Nx) % Ny + Ng;
                const int k = idx / (Nx * Ny) + Ng;

                divergence_cell_kernel_staggered_O4_3d(i, j, k, Ng, Nx, Ny, Nz,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, div_stride, div_plane_stride,
                    dx, dy, dz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, div_ptr);
            }
        } else {
            // O2 divergence
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], div_ptr[0:div_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_cells; ++idx) {
                const int i = idx % Nx + Ng;
                const int j = (idx / Nx) % Ny + Ng;
                const int k = idx / (Nx * Ny) + Ng;

                divergence_cell_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, div_stride, div_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, div_ptr);
            }
        }
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Use target data for scalar parameters (NVHPC workaround)
    #pragma omp target data map(to: dx, dy, u_stride, v_stride, div_stride, Nx, Ng)
    {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], div_ptr[0:div_total_size])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;  // Cell center i index (with ghosts)
            const int j = idx / Nx + Ng;  // Cell center j index (with ghosts)

            // Fully inlined divergence computation
            const int u_right = j * u_stride + (i + 1);
            const int u_left = j * u_stride + i;
            const int v_top = (j + 1) * v_stride + i;
            const int v_bottom = j * v_stride + i;
            const int div_idx = j * div_stride + i;

            const double dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
            const double dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
            div_ptr[div_idx] = dudx + dvdy;
        }
    }
}

void RANSSolver::compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            dp_dx(i, j) = (pressure_(i+1, j) - pressure_(i-1, j)) / (2.0 * dx);
            dp_dy(i, j) = (pressure_(i, j+1) - pressure_(i, j-1)) / (2.0 * dy);
        }
    }
}

void RANSSolver::correct_velocity() {
    NVTX_PUSH("correct_velocity");

    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const double dt = v.dt;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int p_stride = v.cell_stride;

    // O4 spatial discretization for pressure gradient (Dcf_O4)
    const bool use_O4 = (config_.space_order == 4);

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t p_total_size = field_total_size_;

    const double* u_star_ptr = v.u_star_face;
    const double* v_star_ptr = v.v_star_face;
    const double* p_corr_ptr = v.p_corr;
    double*       u_ptr      = v.u_face;
    double*       v_ptr      = v.v_face;
    double*       p_ptr      = v.p;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int p_plane_stride = v.cell_plane_stride;
        const double* w_star_ptr = v.w_star_face;
        double*       w_ptr      = v.w_face;

        // Correct u-velocities at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng, x_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                correct_u_face_kernel_staggered_O4_3d(i, j, k, Ng, Nx,
                    u_stride, u_plane_stride, p_stride, p_plane_stride,
                    dx, dt, x_periodic, u_star_ptr, p_corr_ptr, u_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                correct_u_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, p_stride, p_plane_stride,
                    dx, dt, u_star_ptr, p_corr_ptr, u_ptr);
            }
        }

        // Enforce x-periodicity (3D)
        if (x_periodic) {
            const int n_u_periodic = Ny * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size]) \
                firstprivate(u_stride, u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_periodic; ++idx) {
                int j = idx % Ny + Ng;
                int k = idx / Ny + Ng;
                int i_left = Ng;
                int i_right = Ng + Nx;
                int idx_left = k * u_plane_stride + j * u_stride + i_left;
                int idx_right = k * u_plane_stride + j * u_stride + i_right;
                double u_avg = 0.5 * (u_ptr[idx_left] + u_ptr[idx_right]);
                u_ptr[idx_left] = u_avg;
                u_ptr[idx_right] = u_avg;
            }
        }

        // Correct v-velocities at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng, y_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                correct_v_face_kernel_staggered_O4_3d(i, j, k, Ng, Ny,
                    v_stride, v_plane_stride, p_stride, p_plane_stride,
                    dy, dt, y_periodic, v_star_ptr, p_corr_ptr, v_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                correct_v_face_kernel_staggered_3d(i, j, k,
                    v_stride, v_plane_stride, p_stride, p_plane_stride,
                    dy, dt, v_star_ptr, p_corr_ptr, v_ptr);
            }
        }

        // Enforce y-periodicity (3D)
        if (y_periodic) {
            const int n_v_periodic = Nx * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size]) \
                firstprivate(v_stride, v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int k = idx / Nx + Ng;
                int j_bottom = Ng;
                int j_top = Ng + Ny;
                int idx_bottom = k * v_plane_stride + j_bottom * v_stride + i;
                int idx_top = k * v_plane_stride + j_top * v_stride + i;
                double v_avg = 0.5 * (v_ptr[idx_bottom] + v_ptr[idx_top]);
                v_ptr[idx_bottom] = v_avg;
                v_ptr[idx_top] = v_avg;
            }
        }

        // Correct w-velocities at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size], w_star_ptr[0:w_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dz, dt, w_stride, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Nz, Ng, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                correct_w_face_kernel_staggered_O4_3d(i, j, k, Ng, Nz,
                    w_stride, w_plane_stride, p_stride, p_plane_stride,
                    dz, dt, z_periodic, w_star_ptr, p_corr_ptr, w_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size], w_star_ptr[0:w_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dz, dt, w_stride, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                correct_w_face_kernel_staggered_3d(i, j, k,
                    w_stride, w_plane_stride, p_stride, p_plane_stride,
                    dz, dt, w_star_ptr, p_corr_ptr, w_ptr);
            }
        }

        // Enforce z-periodicity (3D)
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(w_stride, w_plane_stride, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int k_front = Ng;
                int k_back = Ng + Nz;
                int idx_front = k_front * w_plane_stride + j * w_stride + i;
                int idx_back = k_back * w_plane_stride + j * w_stride + i;
                double w_avg = 0.5 * (w_ptr[idx_front] + w_ptr[idx_back]);
                w_ptr[idx_front] = w_avg;
                w_ptr[idx_back] = w_avg;
            }
        }

        // Update pressure at cell centers (3D)
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            int p_idx = k * p_plane_stride + j * p_stride + i;
            p_ptr[p_idx] += p_corr_ptr[p_idx];
        }

        NVTX_POP();
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Correct ALL u-velocities at x-faces (including redundant face if periodic)
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dx, dt, u_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_u_face_kernel_staggered(i, j, u_stride, p_stride, dx, dt,
                                       u_star_ptr, p_corr_ptr, u_ptr);
    }

    // Enforce exact x-periodicity: average the left and right edge values
    if (x_periodic) {
        const int n_u_periodic = Ny;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(u_stride, Nx, Ng)
        for (int j_local = 0; j_local < n_u_periodic; ++j_local) {
            int j = j_local + Ng;
            int i_left = Ng;
            int i_right = Ng + Nx;
            double u_avg = 0.5 * (u_ptr[j * u_stride + i_left] + u_ptr[j * u_stride + i_right]);
            u_ptr[j * u_stride + i_left] = u_avg;
            u_ptr[j * u_stride + i_right] = u_avg;
        }
    }

    // Correct ALL v-velocities at y-faces (including redundant face if periodic)
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dy, dt, v_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_v_face_kernel_staggered(i, j, v_stride, p_stride, dy, dt,
                                       v_star_ptr, p_corr_ptr, v_ptr);
    }

    // Enforce exact y-periodicity: average the bottom and top edge values
    if (y_periodic) {
        const int n_v_periodic = Nx;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(v_stride, Ny, Ng)
        for (int i_local = 0; i_local < n_v_periodic; ++i_local) {
            int i = i_local + Ng;
            int j_bottom = Ng;
            int j_top = Ng + Ny;
            double v_avg = 0.5 * (v_ptr[j_bottom * v_stride + i] + v_ptr[j_top * v_stride + i]);
            v_ptr[j_bottom * v_stride + i] = v_avg;
            v_ptr[j_top * v_stride + i] = v_avg;
        }
    }

    // Update pressure at cell centers
    #pragma omp target teams distribute parallel for \
        map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(p_stride, Nx)
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = idx / Nx + Ng;

        update_pressure_kernel(i, j, p_stride, p_corr_ptr, p_ptr);
    }

    NVTX_POP();  // End correct_velocity
}


} // namespace nncfd
