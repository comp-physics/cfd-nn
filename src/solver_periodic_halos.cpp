/// @file solver_periodic_halos.cpp
/// @brief Periodic halo fill kernels for GPU-resident stepping
/// Split from solver.cpp to work around nvc++ compiler crash with large TUs

#include "solver.hpp"
#include "gpu_utils.hpp"

namespace nncfd {

void RANSSolver::enforce_periodic_halos_device(double* u_ptr, double* v_ptr, double* w_ptr) {
    // Minimal periodic halo fill for GPU-resident stepping (no swaps, no host transfers)
    // This function fills Ng ghost layers and enforces seam face averaging for periodic BCs.

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);

    // Early exit if no periodic boundaries
    if (!x_periodic && !y_periodic) {
        if (mesh_->is2D()) return;
        const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                (velocity_bc_.z_hi == VelocityBC::Periodic);
        if (!z_periodic) return;
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;

    // Strides match VectorField layout
    const int u_stride = Nx + 1 + 2 * Ng;
    const int v_stride = Nx + 2 * Ng;

    // Total extents including ghosts
    const int u_Ny_total = Ny + 2 * Ng;
    const int v_Ny_total = Ny + 1 + 2 * Ng;
    const int u_Nx_total = Nx + 1 + 2 * Ng;
    const int v_Nx_total = Nx + 2 * Ng;

    // =========================================================================
    // 2D Path
    // =========================================================================
    if (mesh_->is2D()) {
        // X-direction: u is NORMAL (seam avg + ghost fill), v is TANGENTIAL
        if (x_periodic) {
            // u: Seam face averaging
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
            for (int j = 0; j < u_Ny_total; ++j) {
                u_ptr[j * u_stride + (Ng + Nx)] = u_ptr[j * u_stride + Ng];
            }

            // u: Ghost fill
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
            for (int t = 0; t < Ng * u_Ny_total; ++t) {
                int j = t / Ng;
                int g = t % Ng;
                u_ptr[j * u_stride + (Ng - 1 - g)] = u_ptr[j * u_stride + (Ng + Nx - 1 - g)];
                u_ptr[j * u_stride + (Ng + Nx + 1 + g)] = u_ptr[j * u_stride + (Ng + 1 + g)];
            }

            // v: Ghost fill only (tangential)
            #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
            for (int t = 0; t < Ng * v_Ny_total; ++t) {
                int j = t / Ng;
                int g = t % Ng;
                v_ptr[j * v_stride + (Ng - 1 - g)] = v_ptr[j * v_stride + (Ng + Nx - 1 - g)];
                v_ptr[j * v_stride + (Ng + Nx + g)] = v_ptr[j * v_stride + (Ng + g)];
            }
        }

        // Y-direction: v is NORMAL, u is TANGENTIAL
        if (y_periodic) {
            // v: Seam face averaging
            #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
            for (int i = 0; i < v_Nx_total; ++i) {
                v_ptr[(Ng + Ny) * v_stride + i] = v_ptr[Ng * v_stride + i];
            }

            // v: Ghost fill
            #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
            for (int t = 0; t < Ng * v_Nx_total; ++t) {
                int i = t / Ng;
                int g = t % Ng;
                v_ptr[(Ng - 1 - g) * v_stride + i] = v_ptr[(Ng + Ny - 1 - g) * v_stride + i];
                v_ptr[(Ng + Ny + 1 + g) * v_stride + i] = v_ptr[(Ng + 1 + g) * v_stride + i];
            }

            // u: Ghost fill only (tangential)
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
            for (int t = 0; t < Ng * u_Nx_total; ++t) {
                int i = t / Ng;
                int g = t % Ng;
                u_ptr[(Ng - 1 - g) * u_stride + i] = u_ptr[(Ng + Ny - 1 - g) * u_stride + i];
                u_ptr[(Ng + Ny + g) * u_stride + i] = u_ptr[(Ng + g) * u_stride + i];
            }
        }

        // Corner fix for fully periodic 2D
        if (x_periodic && y_periodic) {
            // Re-apply X for u after Y modified corners
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
            for (int j = 0; j < u_Ny_total; ++j) {
                u_ptr[j * u_stride + (Ng + Nx)] = u_ptr[j * u_stride + Ng];
            }
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
            for (int t = 0; t < Ng * u_Ny_total; ++t) {
                int j = t / Ng;
                int g = t % Ng;
                u_ptr[j * u_stride + (Ng - 1 - g)] = u_ptr[j * u_stride + (Ng + Nx - 1 - g)];
                u_ptr[j * u_stride + (Ng + Nx + 1 + g)] = u_ptr[j * u_stride + (Ng + 1 + g)];
            }

            // Re-apply Y for v after X modified corners
            #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
            for (int i = 0; i < v_Nx_total; ++i) {
                v_ptr[(Ng + Ny) * v_stride + i] = v_ptr[Ng * v_stride + i];
            }
            #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
            for (int t = 0; t < Ng * v_Nx_total; ++t) {
                int i = t / Ng;
                int g = t % Ng;
                v_ptr[(Ng - 1 - g) * v_stride + i] = v_ptr[(Ng + Ny - 1 - g) * v_stride + i];
                v_ptr[(Ng + Ny + 1 + g) * v_stride + i] = v_ptr[(Ng + 1 + g) * v_stride + i];
            }
        }
        return;
    }

    // =========================================================================
    // 3D Path
    // =========================================================================
    const int Nz = mesh_->Nz;
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    const int u_plane = u_stride * (Ny + 2 * Ng);
    const int v_plane = v_stride * (Ny + 1 + 2 * Ng);
    const int w_stride = Nx + 2 * Ng;
    const int w_plane = w_stride * (Ny + 2 * Ng);
    const int Nz_total = Nz + 2 * Ng;

    // X-direction
    if (x_periodic) {
        // u: seam + ghost
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
        for (int t = 0; t < u_Ny_total * Nz_total; ++t) {
            int j = t % u_Ny_total;
            int k = t / u_Ny_total;
            u_ptr[k * u_plane + j * u_stride + (Ng + Nx)] = u_ptr[k * u_plane + j * u_stride + Ng];
        }
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
        for (int t = 0; t < Ng * u_Ny_total * Nz_total; ++t) {
            int g = t % Ng;
            int j = (t / Ng) % u_Ny_total;
            int k = t / (Ng * u_Ny_total);
            int base = k * u_plane + j * u_stride;
            u_ptr[base + (Ng - 1 - g)] = u_ptr[base + (Ng + Nx - 1 - g)];
            u_ptr[base + (Ng + Nx + 1 + g)] = u_ptr[base + (Ng + 1 + g)];
        }

        // v: tangential ghost
        #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
        for (int t = 0; t < Ng * v_Ny_total * Nz_total; ++t) {
            int g = t % Ng;
            int j = (t / Ng) % v_Ny_total;
            int k = t / (Ng * v_Ny_total);
            int base = k * v_plane + j * v_stride;
            v_ptr[base + (Ng - 1 - g)] = v_ptr[base + (Ng + Nx - 1 - g)];
            v_ptr[base + (Ng + Nx + g)] = v_ptr[base + (Ng + g)];
        }

        // w: tangential ghost
        if (w_ptr) {
            const int w_Ny_total = Ny + 2 * Ng;
            const int w_Nz_total = Nz + 1 + 2 * Ng;
            #pragma omp target teams distribute parallel for is_device_ptr(w_ptr)
            for (int t = 0; t < Ng * w_Ny_total * w_Nz_total; ++t) {
                int g = t % Ng;
                int j = (t / Ng) % w_Ny_total;
                int k = t / (Ng * w_Ny_total);
                int base = k * w_plane + j * w_stride;
                w_ptr[base + (Ng - 1 - g)] = w_ptr[base + (Ng + Nx - 1 - g)];
                w_ptr[base + (Ng + Nx + g)] = w_ptr[base + (Ng + g)];
            }
        }
    }

    // Y-direction
    if (y_periodic) {
        // v: seam + ghost
        #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
        for (int t = 0; t < v_Nx_total * Nz_total; ++t) {
            int i = t % v_Nx_total;
            int k = t / v_Nx_total;
            v_ptr[k * v_plane + (Ng + Ny) * v_stride + i] = v_ptr[k * v_plane + Ng * v_stride + i];
        }
        #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
        for (int t = 0; t < Ng * v_Nx_total * Nz_total; ++t) {
            int g = t % Ng;
            int i = (t / Ng) % v_Nx_total;
            int k = t / (Ng * v_Nx_total);
            int base = k * v_plane + i;
            v_ptr[base + (Ng - 1 - g) * v_stride] = v_ptr[base + (Ng + Ny - 1 - g) * v_stride];
            v_ptr[base + (Ng + Ny + 1 + g) * v_stride] = v_ptr[base + (Ng + 1 + g) * v_stride];
        }

        // u: tangential ghost
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
        for (int t = 0; t < Ng * u_Nx_total * Nz_total; ++t) {
            int g = t % Ng;
            int i = (t / Ng) % u_Nx_total;
            int k = t / (Ng * u_Nx_total);
            int base = k * u_plane + i;
            u_ptr[base + (Ng - 1 - g) * u_stride] = u_ptr[base + (Ng + Ny - 1 - g) * u_stride];
            u_ptr[base + (Ng + Ny + g) * u_stride] = u_ptr[base + (Ng + g) * u_stride];
        }

        // w: tangential ghost
        if (w_ptr) {
            const int w_Nx_total = Nx + 2 * Ng;
            const int w_Nz_total = Nz + 1 + 2 * Ng;
            #pragma omp target teams distribute parallel for is_device_ptr(w_ptr)
            for (int t = 0; t < Ng * w_Nx_total * w_Nz_total; ++t) {
                int g = t % Ng;
                int i = (t / Ng) % w_Nx_total;
                int k = t / (Ng * w_Nx_total);
                int base = k * w_plane + i;
                w_ptr[base + (Ng - 1 - g) * w_stride] = w_ptr[base + (Ng + Ny - 1 - g) * w_stride];
                w_ptr[base + (Ng + Ny + g) * w_stride] = w_ptr[base + (Ng + g) * w_stride];
            }
        }
    }

    // Z-direction
    if (z_periodic && w_ptr) {
        const int w_Nx_total = Nx + 2 * Ng;
        const int w_Ny_total = Ny + 2 * Ng;

        // w: seam + ghost
        #pragma omp target teams distribute parallel for is_device_ptr(w_ptr)
        for (int t = 0; t < w_Nx_total * w_Ny_total; ++t) {
            int i = t % w_Nx_total;
            int j = t / w_Nx_total;
            w_ptr[(Ng + Nz) * w_plane + j * w_stride + i] = w_ptr[Ng * w_plane + j * w_stride + i];
        }
        #pragma omp target teams distribute parallel for is_device_ptr(w_ptr)
        for (int t = 0; t < Ng * w_Nx_total * w_Ny_total; ++t) {
            int g = t % Ng;
            int i = (t / Ng) % w_Nx_total;
            int j = t / (Ng * w_Nx_total);
            int base = j * w_stride + i;
            w_ptr[(Ng - 1 - g) * w_plane + base] = w_ptr[(Ng + Nz - 1 - g) * w_plane + base];
            w_ptr[(Ng + Nz + 1 + g) * w_plane + base] = w_ptr[(Ng + 1 + g) * w_plane + base];
        }

        // u: tangential ghost in z
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
        for (int t = 0; t < Ng * u_Nx_total * u_Ny_total; ++t) {
            int g = t % Ng;
            int i = (t / Ng) % u_Nx_total;
            int j = t / (Ng * u_Nx_total);
            int base = j * u_stride + i;
            u_ptr[(Ng - 1 - g) * u_plane + base] = u_ptr[(Ng + Nz - 1 - g) * u_plane + base];
            u_ptr[(Ng + Nz + g) * u_plane + base] = u_ptr[(Ng + g) * u_plane + base];
        }

        // v: tangential ghost in z
        #pragma omp target teams distribute parallel for is_device_ptr(v_ptr)
        for (int t = 0; t < Ng * v_Nx_total * v_Ny_total; ++t) {
            int g = t % Ng;
            int i = (t / Ng) % v_Nx_total;
            int j = t / (Ng * v_Nx_total);
            int base = j * v_stride + i;
            v_ptr[(Ng - 1 - g) * v_plane + base] = v_ptr[(Ng + Nz - 1 - g) * v_plane + base];
            v_ptr[(Ng + Nz + g) * v_plane + base] = v_ptr[(Ng + g) * v_plane + base];
        }
    }
}

} // namespace nncfd
