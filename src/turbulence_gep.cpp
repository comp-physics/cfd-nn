#include "turbulence_gep.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include "features.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

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
        feature_computer_.set_reference(nu_, u_ref_, delta_);
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
    feature_computer_.set_reference(nu_, u_ref_, delta_);
    
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
        
        #pragma omp target teams distribute parallel for \
            map(present: dudx_ptr[0:cell_total_size], dudy_ptr[0:cell_total_size], \
                         dvdx_ptr[0:cell_total_size], dvdy_ptr[0:cell_total_size], \
                         wall_dist_ptr[0:cell_total_size], nu_t_ptr[0:cell_total_size])
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            int cell_idx = j * cell_stride + i;
            
            // Get gradients
            double dudx_v = dudx_ptr[cell_idx];
            double dudy_v = dudy_ptr[cell_idx];
            double dvdx_v = dvdx_ptr[cell_idx];
            double dvdy_v = dvdy_ptr[cell_idx];
            
            // Strain and rotation tensors
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double Oxy = 0.5 * (dudy_v - dvdx_v);
            
            double S_mag_sq = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            double S_mag = sqrt(S_mag_sq);
            double Omega_mag = sqrt(2.0 * Oxy * Oxy);
            
            // Wall distance and y+
            double y_wall = wall_dist_ptr[cell_idx];
            y_wall = (y_wall > 1e-10) ? y_wall : 1e-10;
            
            // Simple mixing length with van Driest damping
            double y_plus = S_mag * y_wall / (nu_val + 1e-20);  // Approximation
            double f_damp = 1.0 - exp(-y_plus / A_plus);
            f_damp = f_damp * f_damp;
            
            // Compute GEP correction factor based on variant
            double f_gep = 1.0;
            if (variant_val == 0) {  // WS2016_Channel
                double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                double f_rot = 1.0 / (1.0 + 0.1 * ratio * ratio);
                f_gep = f_damp * f_rot;
            } else if (variant_val == 1) {  // WS2016_PeriodicHill
                double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                double f_sep = 1.0 / (1.0 + 0.2 * ratio * ratio);
                double f_wall = tanh(y_plus / 50.0);
                f_gep = f_wall * f_sep;
            } else {  // Simple
                f_gep = f_damp;
            }
            
            // Mixing length: l = kappa * y * f_damp
            double l = kappa * y_wall * f_gep;
            
            // Eddy viscosity: nu_t = l^2 * |S|
            double nu_t_val = l * l * S_mag;
            
            // Clipping
            nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
            double max_nu_t = 1000.0 * nu_val;
            nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
            
            nu_t_ptr[cell_idx] = nu_t_val;
        }
        
        // Done - nu_t computed entirely on GPU
        return;
    }
#else
    (void)device_view;
#endif
    
    // CPU fallback path - match GPU implementation exactly
    // Use same MAC gradient computation as GPU for consistency
    ScalarField dudx_field(mesh), dudy_field(mesh), dvdx_field(mesh), dvdy_field(mesh);
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_field, dudy_field, dvdx_field, dvdy_field);
    
    constexpr double kappa = 0.41;
    constexpr double A_plus = 26.0;
    const int variant_val = static_cast<int>(variant_);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Get precomputed gradients (matching GPU kernel)
            double Sxx = dudx_field(i, j);
            double Syy = dvdy_field(i, j);
            double Sxy = 0.5 * (dudy_field(i, j) + dvdx_field(i, j));
            double Oxy = 0.5 * (dudy_field(i, j) - dvdx_field(i, j));
            
            double S_mag_sq = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            double S_mag = std::sqrt(S_mag_sq);
            double Omega_mag = std::sqrt(2.0 * Oxy * Oxy);
            
            // Wall distance and y+
            double y_wall = mesh.wall_distance(i, j);
            y_wall = std::max(y_wall, 1e-10);
            
            // Approximate y+ from local strain rate (same as GPU)
            double y_plus = S_mag * y_wall / (nu_ + 1e-20);
            double f_damp = 1.0 - std::exp(-y_plus / A_plus);
            f_damp = f_damp * f_damp;
            
            // Compute GEP correction factor based on variant (same logic as GPU)
            double f_gep = 1.0;
            if (variant_val == 0) {  // WS2016_Channel
                double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                double f_rot = 1.0 / (1.0 + 0.1 * ratio * ratio);
                f_gep = f_damp * f_rot;
            } else if (variant_val == 1) {  // WS2016_PeriodicHill
                double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                double f_sep = 1.0 / (1.0 + 0.2 * ratio * ratio);
                double f_wall = std::tanh(y_plus / 50.0);
                f_gep = f_wall * f_sep;
            } else {  // Simple
                f_gep = f_damp;
            }
            
            // Mixing length: l = kappa * y * f_gep
            double l = kappa * y_wall * f_gep;
            
            // Eddy viscosity: nu_t = l^2 * |S|
            double nu_t_val = l * l * S_mag;
            
            // Clipping (same as GPU)
            nu_t_val = std::max(0.0, nu_t_val);
            double max_nu_t = 1000.0 * nu_;
            nu_t_val = std::min(nu_t_val, max_nu_t);
            
            nu_t(i, j) = nu_t_val;
        }
    }
}

} // namespace nncfd
