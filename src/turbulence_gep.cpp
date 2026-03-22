#include "turbulence_gep.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include "features.hpp"
#include "numerics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// Unified GEP kernel - compiles for both CPU and GPU
// ============================================================================
#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Compute GEP eddy viscosity for a single cell
/// @param cell_idx    Linear index into cell-centered arrays
/// @param variant     GEP variant (0=WS2016_Channel, 1=WS2016_PeriodicHill, 2=Simple)
/// @param nu          Molecular viscosity
/// @param kappa       von Karman constant
/// @param A_plus      van Driest constant
/// @param dudx_ptr    Cell-centered du/dx
/// @param dudy_ptr    Cell-centered du/dy
/// @param dvdx_ptr    Cell-centered dv/dx
/// @param dvdy_ptr    Cell-centered dv/dy
/// @param wall_dist_ptr  Wall distance array
/// @param nu_t_ptr    [out] Eddy viscosity array
inline void gep_cell_kernel(
    int cell_idx,
    int variant,
    double nu,
    double kappa,
    double A_plus,
    const double* dudx_ptr,
    const double* dudy_ptr,
    const double* dvdx_ptr,
    const double* dvdy_ptr,
    const double* wall_dist_ptr,
    double* nu_t_ptr)
{
    // Get gradients
    const double dudx_v = dudx_ptr[cell_idx];
    const double dudy_v = dudy_ptr[cell_idx];
    const double dvdx_v = dvdx_ptr[cell_idx];
    const double dvdy_v = dvdy_ptr[cell_idx];

    // Strain and rotation tensors
    const double Sxx = dudx_v;
    const double Syy = dvdy_v;
    const double Sxy = 0.5 * (dudy_v + dvdx_v);
    const double Oxy = 0.5 * (dudy_v - dvdx_v);

    const double S_mag_sq = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
    const double S_mag = std::sqrt(S_mag_sq);
    const double Omega_mag = std::sqrt(2.0 * Oxy * Oxy);

    // Wall distance and y+
    double y_wall = wall_dist_ptr[cell_idx];
    y_wall = (y_wall > 1e-10) ? y_wall : 1e-10;

    // Simple mixing length with van Driest damping
    const double y_plus = S_mag * y_wall / (nu + 1e-20);  // Approximation
    double f_damp = 1.0 - std::exp(-y_plus / A_plus);
    f_damp = f_damp * f_damp;

    // Compute GEP correction factor based on variant
    double f_gep = 1.0;
    if (variant == 0) {  // WS2016_Channel
        const double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
        const double f_rot = 1.0 / (1.0 + 0.1 * ratio * ratio);
        f_gep = f_damp * f_rot;
    } else if (variant == 1) {  // WS2016_PeriodicHill
        const double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
        const double f_sep = 1.0 / (1.0 + 0.2 * ratio * ratio);
        const double f_wall = std::tanh(y_plus / 50.0);
        f_gep = f_wall * f_sep;
    } else {  // Simple
        f_gep = f_damp;
    }

    // Mixing length: l = kappa * y * f_gep
    const double l = kappa * y_wall * f_gep;

    // Eddy viscosity: nu_t = l^2 * |S|
    double nu_t_val = l * l * S_mag;

    // Clipping
    nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
    const double max_nu_t = 1000.0 * nu;
    nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;

    nu_t_ptr[cell_idx] = nu_t_val;
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif
// ============================================================================

TurbulenceGEP::TurbulenceGEP()
    : feature_computer_(Mesh()) {
}

TurbulenceGEP::~TurbulenceGEP() {
    cleanup_gpu_buffers();
}

void TurbulenceGEP::initialize(const Mesh& mesh, const VectorField& velocity) {
    (void)velocity;  // Reserved for future use
    ensure_initialized(mesh);
}

void TurbulenceGEP::initialize_gpu_buffers(const Mesh& mesh) {
    // GEP uses device_view for GPU execution, no internal GPU state needed
    (void)mesh;
}

void TurbulenceGEP::cleanup_gpu_buffers() {
    // GEP uses device_view for GPU execution, no internal GPU state to clean up
}

void TurbulenceGEP::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        initialized_ = true;
    }
}

void TurbulenceGEP::update(const Mesh& mesh,
                           const VectorField& velocity,
                           const ScalarField& k,
                           const ScalarField& omega,
                           ScalarField& nu_t,
                           TensorField* tau_ij,
                           const TurbulenceDeviceView* device_view) {
    (void)tau_ij;  // GEP provides eddy viscosity only, not explicit Reynolds stresses
    (void)k;       // Not used by algebraic GEP model
    (void)omega;   // Not used by algebraic GEP model
    TIMED_SCOPE("gep_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, delta_, u_ref_);
    
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view (like Baseline/EARSM)
    if (device_view && device_view->is_valid()) {
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int Ng = mesh.Nghost;
        const int cell_stride = mesh.total_Nx();
        const int cell_total_size = device_view->cell_total;
        // First compute gradients on GPU
        gpu_kernels::compute_gradients_from_mac_gpu(
            device_view->u_face, device_view->v_face, device_view->w_face,
            device_view->dudx, device_view->dudy,
            device_view->dvdx, device_view->dvdy,
            Nx, Ny, device_view->Nz, Ng,
            mesh.dx, mesh.dy, mesh.dz,
            device_view->u_stride, device_view->v_stride, cell_stride,
            device_view->u_plane_stride, device_view->v_plane_stride,
            device_view->w_stride, device_view->w_plane_stride,
            device_view->cell_plane_stride,
            device_view->u_total, device_view->v_total, device_view->w_total,
            device_view->cell_total,
            device_view->dyc,
            device_view->dyc_size
        );
        
        // Then run GEP algebraic model on GPU
        const double kappa = numerics::KAPPA;
        const double A_plus = numerics::A_PLUS;
        const double nu_val = nu_;
        const int variant_val = static_cast<int>(variant_);
        
        const double* dudx_ptr = device_view->dudx;
        const double* dudy_ptr = device_view->dudy;
        const double* dvdx_ptr = device_view->dvdx;
        const double* dvdy_ptr = device_view->dvdy;
        const double* wall_dist_ptr = device_view->wall_distance;
        double* nu_t_ptr = device_view->nu_t;
        
        // GPU kernel: compute GEP eddy viscosity using unified kernel
        #pragma omp target teams distribute parallel for \
            map(present: dudx_ptr[0:cell_total_size], dudy_ptr[0:cell_total_size], \
                         dvdx_ptr[0:cell_total_size], dvdy_ptr[0:cell_total_size], \
                         wall_dist_ptr[0:cell_total_size], nu_t_ptr[0:cell_total_size])
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            const int i = idx % Nx + Ng;
            const int j = idx / Nx + Ng;
            const int cell_idx = j * cell_stride + i;

            // Call unified kernel (same code path as CPU)
            gep_cell_kernel(cell_idx, variant_val, nu_val, kappa, A_plus,
                            dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
                            wall_dist_ptr, nu_t_ptr);
        }
        
        // Done - nu_t computed entirely on GPU
        return;
    }
#else
    (void)device_view;
#endif
    
    // Host path - use same unified kernel as GPU
    ScalarField dudx_field(mesh), dudy_field(mesh), dvdx_field(mesh), dvdy_field(mesh);
    compute_gradients_from_mac(mesh, velocity, dudx_field, dudy_field, dvdx_field, dvdy_field);

    constexpr double kappa = numerics::KAPPA;
    constexpr double A_plus = numerics::A_PLUS;
    const int variant_val = static_cast<int>(variant_);
    const int cell_stride = mesh.total_Nx();

    // Get raw pointers for unified kernel
    const double* dudx_ptr = dudx_field.data().data();
    const double* dudy_ptr = dudy_field.data().data();
    const double* dvdx_ptr = dvdx_field.data().data();
    const double* dvdy_ptr = dvdy_field.data().data();
    double* nu_t_ptr = nu_t.data().data();

    // Gradient and nu_t fields use 3D indexing; wall distance is y-only
    // Create 3D wall distance buffer so the unified kernel can use 3D cell_idx
    const int plane_stride = mesh.total_Nx() * mesh.total_Ny();
    const size_t total_cells_3d = (size_t)mesh.total_cells();
    std::vector<double> wall_dist_buf(total_cells_3d, 0.0);
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                const int idx = k * plane_stride + j * cell_stride + i;
                wall_dist_buf[idx] = mesh.wall_distance(i, j);
            }
        }
    }
    const double* wall_dist_ptr = wall_dist_buf.data();

    const int k_offset = mesh.Nghost * plane_stride;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            const int cell_idx = k_offset + j * cell_stride + i;

            gep_cell_kernel(cell_idx, variant_val, nu_, kappa, A_plus,
                            dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
                            wall_dist_ptr, nu_t_ptr);
        }
    }
}

} // namespace nncfd
