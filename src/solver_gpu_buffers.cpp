// GPU buffer management for RANSSolver
// Split from solver.cpp to avoid nvc++ compiler crash on large files

#include "solver.hpp"
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include "gpu_utils.hpp"
#endif

namespace nncfd {

// ============================================================================
// Shared pointer extraction (used by both CPU and GPU paths)
// ============================================================================

void RANSSolver::extract_field_pointers() {
    field_total_size_ = mesh_->total_cells();

    // Staggered grid velocity fields
    velocity_u_ptr_ = velocity_.u_data().data();
    velocity_v_ptr_ = velocity_.v_data().data();
    velocity_star_u_ptr_ = velocity_star_.u_data().data();
    velocity_star_v_ptr_ = velocity_star_.v_data().data();
    velocity_old_u_ptr_ = velocity_old_.u_data().data();
    velocity_old_v_ptr_ = velocity_old_.v_data().data();
    velocity_rk_u_ptr_ = velocity_rk_.u_data().data();
    velocity_rk_v_ptr_ = velocity_rk_.v_data().data();

    // Cell-centered fields
    pressure_ptr_ = pressure_.data().data();
    pressure_corr_ptr_ = pressure_correction_.data().data();
    nu_t_ptr_ = nu_t_.data().data();
    nu_eff_ptr_ = nu_eff_.data().data();
    rhs_poisson_ptr_ = rhs_poisson_.data().data();
    div_velocity_ptr_ = div_velocity_.data().data();

    // Work arrays
    conv_u_ptr_ = conv_.u_data().data();
    conv_v_ptr_ = conv_.v_data().data();
    diff_u_ptr_ = diff_.u_data().data();
    diff_v_ptr_ = diff_.v_data().data();

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        velocity_w_ptr_ = velocity_.w_data().data();
        velocity_star_w_ptr_ = velocity_star_.w_data().data();
        velocity_old_w_ptr_ = velocity_old_.w_data().data();
        velocity_rk_w_ptr_ = velocity_rk_.w_data().data();
        conv_w_ptr_ = conv_.w_data().data();
        diff_w_ptr_ = diff_.w_data().data();
    }

    // Turbulence transport fields
    k_ptr_ = k_.data().data();
    omega_ptr_ = omega_.data().data();

    // Reynolds stress tensor components (for EARSM/TBNN)
    tau_xx_ptr_ = tau_ij_.xx_data().data();
    tau_xy_ptr_ = tau_ij_.xy_data().data();
    tau_yy_ptr_ = tau_ij_.yy_data().data();

    // Gradient scratch buffers for turbulence models
    dudx_ptr_ = dudx_.data().data();
    dudy_ptr_ = dudy_.data().data();
    dvdx_ptr_ = dvdx_.data().data();
    dvdy_ptr_ = dvdy_.data().data();
    wall_distance_ptr_ = wall_distance_.data().data();
}

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();

    // Extract all raw pointers (shared with CPU path)
    extract_field_pointers();
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_start = std::chrono::steady_clock::now();
#endif
    
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    // Map all arrays to GPU device and copy initial values
    // Using map(to:) to transfer initialized data, map(alloc:) for device-only buffers
    // Data will persist on GPU for entire solver lifetime
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    // Consolidated GPU buffer mapping - grouping arrays by size and transfer type
    // Group 1: u-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_u_ptr_[0:u_total_size], \
                velocity_star_u_ptr_[0:u_total_size], \
                conv_u_ptr_[0:u_total_size], \
                diff_u_ptr_[0:u_total_size])

    // Group 2: v-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_v_ptr_[0:v_total_size], \
                velocity_star_v_ptr_[0:v_total_size], \
                conv_v_ptr_[0:v_total_size], \
                diff_v_ptr_[0:v_total_size])

    // Group 3: field-sized arrays with initial data (to: transfer)
    #pragma omp target enter data \
        map(to: pressure_ptr_[0:field_total_size_], \
                pressure_corr_ptr_[0:field_total_size_], \
                nu_t_ptr_[0:field_total_size_], \
                nu_eff_ptr_[0:field_total_size_], \
                rhs_poisson_ptr_[0:field_total_size_], \
                div_velocity_ptr_[0:field_total_size_], \
                k_ptr_[0:field_total_size_], \
                omega_ptr_[0:field_total_size_])

    // Group 4: gradient buffers (to: need zero init to prevent NaN in EARSM)
    #pragma omp target enter data \
        map(to: dudx_ptr_[0:field_total_size_], \
                dudy_ptr_[0:field_total_size_], \
                dvdx_ptr_[0:field_total_size_], \
                dvdy_ptr_[0:field_total_size_], \
                wall_distance_ptr_[0:field_total_size_])

    // Group 5: device-only arrays (alloc: will be computed on GPU)
    // velocity_old: device-resident for residual computation (host never used)
    // velocity_rk: work buffer for RK time integration stages
    // tau_*: Reynolds stress components computed by EARSM/TBNN
    #pragma omp target enter data \
        map(alloc: velocity_old_u_ptr_[0:u_total_size], \
                   velocity_old_v_ptr_[0:v_total_size], \
                   velocity_rk_u_ptr_[0:u_total_size], \
                   velocity_rk_v_ptr_[0:v_total_size], \
                   tau_xx_ptr_[0:field_total_size_], \
                   tau_xy_ptr_[0:field_total_size_], \
                   tau_yy_ptr_[0:field_total_size_])

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target enter data \
            map(to: velocity_w_ptr_[0:w_total_size], \
                    velocity_star_w_ptr_[0:w_total_size], \
                    conv_w_ptr_[0:w_total_size], \
                    diff_w_ptr_[0:w_total_size]) \
            map(alloc: velocity_old_w_ptr_[0:w_total_size], \
                       velocity_rk_w_ptr_[0:w_total_size])
    }

    // Zero-initialize device-only arrays to prevent garbage in first residual computation
    // Arrays allocated with map(alloc:) contain garbage until explicitly written
    #pragma omp target teams distribute parallel for map(present: velocity_old_u_ptr_[0:u_total_size])
    for (size_t i = 0; i < u_total_size; ++i) velocity_old_u_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_old_v_ptr_[0:v_total_size])
    for (size_t i = 0; i < v_total_size; ++i) velocity_old_v_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_rk_u_ptr_[0:u_total_size])
    for (size_t i = 0; i < u_total_size; ++i) velocity_rk_u_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_rk_v_ptr_[0:v_total_size])
    for (size_t i = 0; i < v_total_size; ++i) velocity_rk_v_ptr_[i] = 0.0;

    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target teams distribute parallel for map(present: velocity_old_w_ptr_[0:w_total_size])
        for (size_t i = 0; i < w_total_size; ++i) velocity_old_w_ptr_[i] = 0.0;
        #pragma omp target teams distribute parallel for map(present: velocity_rk_w_ptr_[0:w_total_size])
        for (size_t i = 0; i < w_total_size; ++i) velocity_rk_w_ptr_[i] = 0.0;
    }

    // Zero-initialize Reynolds stress tensor components
    #pragma omp target teams distribute parallel for map(present: tau_xx_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xx_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_xy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xy_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_yy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_yy_ptr_[i] = 0.0;

    // Verify mappings succeeded (fail fast if GPU unavailable despite num_devices>0)
    if (!gpu::is_pointer_present(velocity_u_ptr_)) {
        throw std::runtime_error("GPU mapping failed despite device availability");
    }
    
    gpu_ready_ = true;
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> transfer_time = transfer_end - transfer_start;
    double mb_transferred = 16 * field_total_size_ * sizeof(double) / 1024.0 / 1024.0;
    double bandwidth = mb_transferred / transfer_time.count();
    (void)mb_transferred;
    (void)bandwidth;
#endif
}

void RANSSolver::cleanup_gpu_buffers() {
    assert(gpu_ready_ && "GPU must be initialized before cleanup");
    
    // Staggered grid sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();
    
    // Copy final results back from GPU, then free device memory
    // Using map(from:) to get final state back to host
    #pragma omp target exit data map(from: velocity_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(from: velocity_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(from: pressure_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity results
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(from: velocity_w_ptr_[0:w_total_size])
    }

    // Delete temporary/work arrays without copying back
    #pragma omp target exit data map(delete: velocity_star_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_star_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_old_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_rk_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_rk_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: conv_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: conv_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: diff_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: diff_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: div_velocity_ptr_[0:field_total_size_])

    // 3D temporary arrays
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(delete: velocity_star_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: velocity_old_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: velocity_rk_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: conv_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: diff_w_ptr_[0:w_total_size])
    }
    
    // Delete gradient scratch buffers
    #pragma omp target exit data map(delete: dudx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dudy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: wall_distance_ptr_[0:field_total_size_])
    
    // Delete transport fields
    #pragma omp target exit data map(delete: k_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: omega_ptr_[0:field_total_size_])
    
    // Delete Reynolds stress tensor buffers
    #pragma omp target exit data map(delete: tau_xx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_xy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_yy_ptr_[0:field_total_size_])

    // Free recycling inflow buffers (allocated with omp_target_alloc)
    if (use_recycling_) {
        int device_id = omp_get_default_device();
        if (recycle_u_ptr_) { omp_target_free(recycle_u_ptr_, device_id); recycle_u_ptr_ = nullptr; }
        if (recycle_v_ptr_) { omp_target_free(recycle_v_ptr_, device_id); recycle_v_ptr_ = nullptr; }
        if (recycle_w_ptr_) { omp_target_free(recycle_w_ptr_, device_id); recycle_w_ptr_ = nullptr; }
        if (inlet_u_ptr_) { omp_target_free(inlet_u_ptr_, device_id); inlet_u_ptr_ = nullptr; }
        if (inlet_v_ptr_) { omp_target_free(inlet_v_ptr_, device_id); inlet_v_ptr_ = nullptr; }
        if (inlet_w_ptr_) { omp_target_free(inlet_w_ptr_, device_id); inlet_w_ptr_ = nullptr; }
    }

    gpu_ready_ = false;
}

void RANSSolver::sync_to_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Update GPU with changed fields (typically after CPU-side modifications)
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update to(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update to(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update to(pressure_ptr_[0:field_total_size_])
    #pragma omp target update to(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update to(velocity_w_ptr_[0:w_total_size])
    }

    // Upload k and omega if turbulence model uses transport equations
    // These are initialized by RANSSolver::initialize() after GPU buffers are allocated
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update to(k_ptr_[0:field_total_size_])
        #pragma omp target update to(omega_ptr_[0:field_total_size_])
    }
}

void RANSSolver::sync_from_gpu() {
    // Legacy sync for backward compatibility - downloads everything
    // Prefer using sync_solution_from_gpu() and sync_transport_from_gpu() selectively
    sync_solution_from_gpu();
    sync_transport_from_gpu();
}

void RANSSolver::sync_solution_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download only primary solution fields needed for I/O/analysis
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update from(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update from(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update from(pressure_ptr_[0:field_total_size_])
    #pragma omp target update from(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update from(velocity_w_ptr_[0:w_total_size])
    }
}

void RANSSolver::sync_transport_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download transport equation fields (k, omega) only if turbulence model uses them
    // For laminar runs (turb_model = none), this saves hundreds of MB on large grids!
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update from(k_ptr_[0:field_total_size_])
        #pragma omp target update from(omega_ptr_[0:field_total_size_])
    }
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    assert(gpu_ready_ && "GPU must be initialized to get device view");
    
    TurbulenceDeviceView view;
    
    // Velocity field (staggered, solver-owned, persistent on GPU)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();  // Stride for cell-centered fields
    
    // Reynolds stress tensor
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;
    
    // Gradient scratch buffers
    view.dudx = dudx_ptr_;
    view.dudy = dudy_ptr_;
    view.dvdx = dvdx_ptr_;
    view.dvdy = dvdy_ptr_;
    
    // Wall distance
    view.wall_distance = wall_distance_ptr_;
    
    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);
    
    return view;
}

// ============================================================================
// Device-side diagnostic functions (GPU QOI computation)
// ============================================================================

bool RANSSolver::verify_gpu_field_presence() const {
    if (!gpu_ready_) return false;

    int device = omp_get_default_device();
    bool all_present = true;

    // Helper lambda to check and report
    auto check_field = [&](double* ptr, const char* name) {
        if (!omp_target_is_present(ptr, device)) {
            std::fprintf(stderr, "[verify_gpu_field_presence] MISSING: %s (ptr=%p)\n",
                         name, static_cast<void*>(ptr));
            all_present = false;
        }
    };

    // Check all velocity fields (critical for RK stepping)
    check_field(velocity_u_ptr_, "velocity_u");
    check_field(velocity_v_ptr_, "velocity_v");
    check_field(velocity_star_u_ptr_, "velocity_star_u");
    check_field(velocity_star_v_ptr_, "velocity_star_v");
    check_field(velocity_old_u_ptr_, "velocity_old_u");
    check_field(velocity_old_v_ptr_, "velocity_old_v");
    check_field(velocity_rk_u_ptr_, "velocity_rk_u");
    check_field(velocity_rk_v_ptr_, "velocity_rk_v");

    // Check work arrays
    check_field(conv_u_ptr_, "conv_u");
    check_field(conv_v_ptr_, "conv_v");
    check_field(diff_u_ptr_, "diff_u");
    check_field(diff_v_ptr_, "diff_v");

    // Check scalar fields (critical for projection)
    check_field(pressure_ptr_, "pressure");
    check_field(pressure_corr_ptr_, "pressure_correction");
    check_field(rhs_poisson_ptr_, "rhs_poisson");
    check_field(div_velocity_ptr_, "div_velocity");
    check_field(nu_eff_ptr_, "nu_eff");

    // 3D fields
    if (!mesh_->is2D()) {
        check_field(velocity_w_ptr_, "velocity_w");
        check_field(velocity_star_w_ptr_, "velocity_star_w");
        check_field(velocity_old_w_ptr_, "velocity_old_w");
        check_field(velocity_rk_w_ptr_, "velocity_rk_w");
        check_field(conv_w_ptr_, "conv_w");
        check_field(diff_w_ptr_, "diff_w");
    }

    // If presence checks passed, do a write/read sanity check on a critical field
    if (all_present) {
        const int Ng = mesh_->Nghost;
        const int test_idx = Ng * (mesh_->Nx + 2 * Ng) + Ng;
        const double sentinel = 314159.265358979;

        double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_poisson_ptr_, device));

        double original = 0.0;
        double readback = 0.0;

        #pragma omp target is_device_ptr(rhs_dev) map(from: original)
        {
            original = rhs_dev[test_idx];
        }

        #pragma omp target is_device_ptr(rhs_dev)
        {
            rhs_dev[test_idx] = sentinel;
        }

        #pragma omp target is_device_ptr(rhs_dev) map(from: readback)
        {
            readback = rhs_dev[test_idx];
        }

        #pragma omp target is_device_ptr(rhs_dev) firstprivate(original)
        {
            rhs_dev[test_idx] = original;
        }

        if (std::abs(readback - sentinel) > 1e-10) {
            std::fprintf(stderr, "[verify_gpu_field_presence] WRITE/READ FAILED: "
                         "wrote %.6f, read %.6f (expected sentinel)\n", sentinel, readback);
            all_present = false;
        }
    }

    return all_present;
}

double RANSSolver::compute_kinetic_energy_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;
    const double dV = dx * dy * dz;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    double ke = 0.0;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u = 0.5 * (u_dev[j * u_stride + i] + u_dev[j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[j * v_stride + i] + v_dev[(j + 1) * v_stride + i]);
            ke += 0.5 * (u * u + v * v) * dV;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const int n_cells = Nx * Ny * Nz;

        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u = 0.5 * (u_dev[k * u_plane + j * u_stride + i] + u_dev[k * u_plane + j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[k * v_plane + j * v_stride + i] + v_dev[k * v_plane + (j + 1) * v_stride + i]);
            double w = 0.5 * (w_dev[k * w_plane + j * w_stride + i] + w_dev[(k + 1) * w_plane + j * w_stride + i]);
            ke += 0.5 * (u * u + v * v + w * w) * dV;
        }
    }

    return ke;
}

double RANSSolver::compute_max_velocity_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    double max_vel = 0.0;

    const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
    const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

    if (mesh_->is2D()) {
        const int n_u = (Nx + 1) * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, u_stride)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;
            double val = u_dev[j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_v = Nx * (Ny + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, v_stride)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = v_dev[j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

        const int n_u = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, u_plane)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;
            double val = u_dev[k * u_plane + j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_v = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, v_stride, v_plane)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;
            double val = v_dev[k * v_plane + j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_w = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane)
        for (int idx = 0; idx < n_w; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = w_dev[k * w_plane + j * w_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    }

    return max_vel;
}

double RANSSolver::compute_divergence_linf_device() const {
    auto* self = const_cast<RANSSolver*>(this);
    self->compute_divergence(VelocityWhich::Current, self->div_velocity_);

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int stride = mesh_->total_Nx();
    const int plane_stride = stride * mesh_->total_Ny();

    double max_div = 0.0;
    const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Ng, stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = div_dev[j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Nz, Ng, stride, plane_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = div_dev[k * plane_stride + j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    }

    return max_div;
}

double RANSSolver::compute_max_conv_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    double max_conv = 0.0;

    const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
    const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);

    if (mesh_->is2D()) {
        const int u_stride = conv_.u_stride();
        const int v_stride = conv_.v_stride();
        const int n_cells = Nx * Ny;

        #pragma omp target teams distribute parallel for is_device_ptr(conv_u_dev, conv_v_dev) \
            reduction(max:max_conv) firstprivate(Nx, Ny, Ng, u_stride, v_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double cu = conv_u_dev[j * u_stride + i];
            double cv = conv_v_dev[j * v_stride + i];
            if (cu < 0) cu = -cu;
            if (cv < 0) cv = -cv;
            double val = (cu > cv) ? cu : cv;
            if (val > max_conv) max_conv = val;
        }
    } else {
        const int u_stride = conv_.u_stride();
        const int v_stride = conv_.v_stride();
        const int w_stride = conv_.w_stride();
        const int u_plane = conv_.u_plane_stride();
        const int v_plane = conv_.v_plane_stride();
        const int w_plane = conv_.w_plane_stride();
        const int n_cells = Nx * Ny * Nz;

        const double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(conv_u_dev, conv_v_dev, conv_w_dev) \
            reduction(max:max_conv) firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double cu = conv_u_dev[k * u_plane + j * u_stride + i];
            double cv = conv_v_dev[k * v_plane + j * v_stride + i];
            double cw = conv_w_dev[k * w_plane + j * w_stride + i];
            if (cu < 0) cu = -cu;
            if (cv < 0) cv = -cv;
            if (cw < 0) cw = -cw;
            double val = cu;
            if (cv > val) val = cv;
            if (cw > val) val = cw;
            if (val > max_conv) max_conv = val;
        }
    }

    return max_conv;
}


SolverDeviceView RANSSolver::get_solver_view() const {
    SolverDeviceView view;

#ifdef USE_GPU_OFFLOAD
    assert(gpu_ready_ && "GPU must be initialized to get solver view");

    // GPU path: return device-present pointers
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_star_face = velocity_star_u_ptr_;
    view.v_star_face = velocity_star_v_ptr_;
    view.u_old_face = velocity_old_u_ptr_;
    view.v_old_face = velocity_old_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = velocity_w_ptr_;
        view.w_star_face = velocity_star_w_ptr_;
        view.w_old_face = velocity_old_w_ptr_;
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = pressure_ptr_;
    view.p_corr = pressure_corr_ptr_;
    view.nu_t = nu_t_ptr_;
    view.nu_eff = nu_eff_ptr_;
    view.rhs = rhs_poisson_ptr_;
    view.div = div_velocity_ptr_;
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = conv_u_ptr_;
    view.conv_v = conv_v_ptr_;
    view.diff_u = diff_u_ptr_;
    view.diff_v = diff_v_ptr_;

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = conv_w_ptr_;
        view.diff_w = diff_w_ptr_;
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;
#else
    // CPU build: always return host pointers
    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    
    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());
    
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dt = current_dt_;
#endif
    
    return view;
}
#else
// CPU: Set raw pointers for unified code paths (no GPU mapping)
//
// This function enables the same loop code to work on both CPU and GPU builds.
// In GPU builds, these pointers are mapped to device memory with OpenMP target pragmas.
// In CPU builds, the loops simply use these raw pointers directly (no pragmas applied).
// This unification eliminates divergent CPU/GPU arithmetic and reduces code duplication.
void RANSSolver::initialize_gpu_buffers() {
    extract_field_pointers();
    gpu_ready_ = false;
}

void RANSSolver::cleanup_gpu_buffers() {
    // No-op
}

void RANSSolver::sync_to_gpu() {
    // No-op
}

void RANSSolver::sync_from_gpu() {
    // No-op
}

void RANSSolver::sync_solution_from_gpu() {
    // No-op
}

void RANSSolver::sync_transport_from_gpu() {
    // No-op
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    // CPU build: return host pointers (same pattern as GPU version)
    TurbulenceDeviceView view;

    // Velocity field (staggered)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();

    // Reynolds stress tensor
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;

    // Gradient scratch buffers
    view.dudx = dudx_ptr_;
    view.dudy = dudy_ptr_;
    view.dvdx = dvdx_ptr_;
    view.dvdy = dvdy_ptr_;

    // Wall distance
    view.wall_distance = wall_distance_ptr_;

    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);

    return view;
}

SolverDeviceView RANSSolver::get_solver_view() const {
    // CPU build: always return host pointers
    SolverDeviceView view;

    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = const_cast<double*>(velocity_.w_data().data());
        view.w_star_face = const_cast<double*>(velocity_star_.w_data().data());
        view.w_old_face = const_cast<double*>(velocity_old_.w_data().data());
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = const_cast<double*>(conv_.w_data().data());
        view.diff_w = const_cast<double*>(diff_.w_data().data());
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;

    return view;
}
#endif
} // namespace nncfd
