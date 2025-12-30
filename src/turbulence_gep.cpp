#include "turbulence_gep.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include "features.hpp"
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

double TurbulenceGEP::compute_gep_factor(double I1_S, double I2_S, 
                                          double I1_Omega, double I2_Omega,
                                          double y_plus, double Re_d) const {
    // GEP-discovered algebraic expressions
    // These are simplified versions inspired by Weatheritt & Sandberg (2016)
    // The actual GEP expressions are more complex and case-specific
    
    // Suppress warnings for parameters reserved for future GEP variants
    (void)I2_S; (void)I2_Omega; (void)Re_d;
    
    switch (variant_) {
        case Variant::WS2016_Channel: {
            // Channel flow model: accounts for wall effects
            // Based on the structure of WS2016 results
            double S_mag = std::sqrt(std::max(0.0, I1_S));
            double Omega_mag = std::sqrt(std::max(0.0, -I1_Omega));
            
            // Strain-rotation ratio
            double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
            
            // Wall damping similar to van Driest
            double A_plus = 26.0;
            double f_wall = 1.0 - std::exp(-y_plus / A_plus);
            f_wall = f_wall * f_wall;
            
            // GEP-style correction: reduce nu_t in regions of high rotation
            double f_rot = 1.0 / (1.0 + 0.1 * ratio * ratio);
            
            return f_wall * f_rot;
        }
        
        case Variant::WS2016_PeriodicHill: {
            // Periodic hill model: accounts for separation
            double S_mag = std::sqrt(std::max(0.0, I1_S));
            double Omega_mag = std::sqrt(std::max(0.0, -I1_Omega));
            
            // Curvature/rotation effects are stronger for separated flows
            double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
            
            // Reduce production in separated regions (high rotation)
            double f_sep = 1.0 / (1.0 + 0.2 * ratio * ratio);
            
            // Wall proximity effect
            double f_wall = std::tanh(y_plus / 50.0);
            
            return f_wall * f_sep;
        }
        
        case Variant::Simple:
        default: {
            // Simple algebraic model: mixing length with corrections
            double S_mag = std::sqrt(std::max(0.0, I1_S));
            (void)S_mag;  // Reserved for future use
            
            // Basic mixing length: l = kappa * y * (1 - exp(-y+/A+))
            double kappa = 0.41;
            double A_plus = 26.0;
            
            // van Driest damping
            double f_damp = 1.0 - std::exp(-y_plus / A_plus);
            f_damp = f_damp * f_damp;
            
            // Mixing length squared (normalized)
            double l_plus = kappa * y_plus * f_damp;
            (void)l_plus;  // Computed for documentation; nu_t uses f_damp directly
            
            // nu_t = l^2 * |S|
            // Return correction factor relative to base model
            return f_damp;
        }
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
        const size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
        const size_t u_total_size = (size_t)mesh.total_Ny() * (mesh.total_Nx() + 1);
        const size_t v_total_size = (size_t)(mesh.total_Ny() + 1) * mesh.total_Nx();
        
        // First compute gradients on GPU
        gpu_kernels::compute_gradients_from_mac_gpu(
            device_view->u_face, device_view->v_face,
            device_view->dudx, device_view->dudy,
            device_view->dvdx, device_view->dvdy,
            Nx, Ny, Ng,
            mesh.dx, mesh.dy,
            device_view->u_stride, device_view->v_stride, cell_stride,
            u_total_size, v_total_size, cell_total_size
        );
        
        // Then run GEP algebraic model on GPU
        const double kappa = 0.41;
        const double A_plus = 26.0;
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

    constexpr double kappa = 0.41;
    constexpr double A_plus = 26.0;
    const int variant_val = static_cast<int>(variant_);
    const int cell_stride = mesh.total_Nx();

    // Get raw pointers for unified kernel
    const double* dudx_ptr = dudx_field.data().data();
    const double* dudy_ptr = dudy_field.data().data();
    const double* dvdx_ptr = dvdx_field.data().data();
    const double* dvdy_ptr = dvdy_field.data().data();
    double* nu_t_ptr = nu_t.data().data();

    // Create local buffer for wall distance (needed for unified kernel)
    const size_t total_cells = (size_t)mesh.total_Nx() * mesh.total_Ny();
    std::vector<double> wall_dist_buf(total_cells, 0.0);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            const int cell_idx = j * cell_stride + i;
            wall_dist_buf[cell_idx] = mesh.wall_distance(i, j);
        }
    }
    const double* wall_dist_ptr = wall_dist_buf.data();

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            const int cell_idx = j * cell_stride + i;

            // Call unified kernel (same code path as GPU)
            gep_cell_kernel(cell_idx, variant_val, nu_, kappa, A_plus,
                            dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
                            wall_dist_ptr, nu_t_ptr);
        }
    }
}

} // namespace nncfd
