#include "turbulence_gep.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
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
#ifdef USE_GPU_OFFLOAD
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    const int n_cells = mesh.Nx * mesh.Ny;
    
    // Check if already allocated for this mesh size
    if (buffers_on_gpu_ && cached_n_cells_ == n_cells) {
        gpu_ready_ = true;
        return;
    }
    
    // Free old buffers if they exist
    free_gpu_arrays();
    
    // Allocate GPU buffers
    allocate_gpu_arrays(mesh);
    
    cached_n_cells_ = n_cells;
    buffers_on_gpu_ = true;
    gpu_ready_ = true;
#else
    (void)mesh;
    gpu_ready_ = false;
#endif
}

#ifdef USE_GPU_OFFLOAD
void TurbulenceGEP::allocate_gpu_arrays(const Mesh& mesh) {
    const int n_cells = mesh.Nx * mesh.Ny;
    const int total_size = mesh.total_cells();
    const int stride = mesh.total_Nx();
    
    // Allocate flat arrays
    S_mag_flat_.resize(n_cells, 0.0);
    Omega_mag_flat_.resize(n_cells, 0.0);
    wall_dist_flat_.resize(n_cells, 0.0);
    u_mag_flat_.resize(n_cells, 0.0);
    nu_t_gpu_flat_.resize(total_size, 0.0);
    
    // Precompute wall distances
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            wall_dist_flat_[idx++] = mesh.wall_distance(i, j);
        }
    }
    
    // Get pointers
    double* S_mag_ptr = S_mag_flat_.data();
    double* Omega_mag_ptr = Omega_mag_flat_.data();
    double* wall_dist_ptr = wall_dist_flat_.data();
    double* u_mag_ptr = u_mag_flat_.data();
    double* nu_t_ptr = nu_t_gpu_flat_.data();
    
    // Map to GPU persistently
    #pragma omp target enter data map(alloc: S_mag_ptr[0:n_cells])
    #pragma omp target enter data map(alloc: Omega_mag_ptr[0:n_cells])
    #pragma omp target enter data map(to: wall_dist_ptr[0:n_cells])
    #pragma omp target enter data map(alloc: u_mag_ptr[0:n_cells])
    #pragma omp target enter data map(alloc: nu_t_ptr[0:total_size])
}

void TurbulenceGEP::free_gpu_arrays() {
    if (!buffers_on_gpu_) return;
    
    const int n_cells = cached_n_cells_;
    if (n_cells == 0) return;
    
    // Need to compute total_size for nu_t (which has ghost cells)
    const int total_size = nu_t_gpu_flat_.size();
    
    double* S_mag_ptr = S_mag_flat_.data();
    double* Omega_mag_ptr = Omega_mag_flat_.data();
    double* wall_dist_ptr = wall_dist_flat_.data();
    double* u_mag_ptr = u_mag_flat_.data();
    double* nu_t_ptr = nu_t_gpu_flat_.data();
    
    #pragma omp target exit data map(delete: S_mag_ptr[0:n_cells])
    #pragma omp target exit data map(delete: Omega_mag_ptr[0:n_cells])
    #pragma omp target exit data map(delete: wall_dist_ptr[0:n_cells])
    #pragma omp target exit data map(delete: u_mag_ptr[0:n_cells])
    #pragma omp target exit data map(delete: nu_t_ptr[0:total_size])
    
    buffers_on_gpu_ = false;
}
#endif

void TurbulenceGEP::cleanup_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    free_gpu_arrays();
#endif
    gpu_ready_ = false;
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
    TIMED_SCOPE("gep_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, u_ref_, delta_);
    
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view
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
    
    // CPU fallback path
    [[maybe_unused]] double delta = delta_;
    constexpr double kappa = 0.41;
    constexpr double A_plus = 26.0;
    
    // Compute velocity gradients for all cells
    std::vector<Features> features;
    {
        TIMED_SCOPE("gep_features");
        feature_computer_.compute_scalar_features(velocity, k, omega, features);
    }
    
    {
        TIMED_SCOPE("gep_compute");
        
#ifdef USE_GPU_OFFLOAD
        // GPU path using persistent mapping
        if (gpu_ready_) {
            const int n_cells = mesh.Nx * mesh.Ny;
            
            // Flatten features for GPU into our persistent arrays
            for (int idx = 0; idx < n_cells; ++idx) {
                if (idx < static_cast<int>(features.size()) && features[idx].size() > 0) {
                    S_mag_flat_[idx] = features[idx][0];
                    Omega_mag_flat_[idx] = (features[idx].size() > 1) ? features[idx][1] : 0.0;
                } else {
                    S_mag_flat_[idx] = 0.0;
                    Omega_mag_flat_[idx] = 0.0;
                }
            }
            
            // Flatten velocity magnitudes into our persistent array
            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u_val = velocity.u(i, j);
                    double v_val = velocity.v(i, j);
                    u_mag_flat_[idx++] = std::sqrt(u_val * u_val + v_val * v_val);
                }
            }
            
            // Get pointers to persistent GPU buffers
            const double* S_mag_ptr = S_mag_flat_.data();
            const double* Omega_mag_ptr = Omega_mag_flat_.data();
            const double* wall_dist_ptr = wall_dist_flat_.data();
            const double* u_mag_ptr = u_mag_flat_.data();
            double* nu_t_gpu_ptr = nu_t_gpu_flat_.data();
            
            const int Nx = mesh.Nx;
            const int variant_int = static_cast<int>(variant_);
            const double nu = nu_;
            const double nu_t_max = nu_t_max_;
            const int stride = mesh.total_Nx();
            const size_t total_size = nu_t_gpu_flat_.size();
            const double kappa = 0.41;
            const double A_plus = 26.0;
            
            // Upload data to GPU
            #pragma omp target update to(S_mag_ptr[0:n_cells])
            #pragma omp target update to(Omega_mag_ptr[0:n_cells])
            #pragma omp target update to(u_mag_ptr[0:n_cells])
            
            // GPU kernel - NO map() clauses (all arrays already persistent!)
            #pragma omp target teams distribute parallel for firstprivate(Nx, stride)
            for (int idx = 0; idx < n_cells; ++idx) {
                // Convert flat index to (i,j) including ghost cells
                const int i = idx % Nx + 1;  // +1 to skip ghost cells
                const int j = idx / Nx + 1;
                const int cell_idx = j * stride + i;  // Stride-based index
                
                // Get wall distance (flat array, no ghosts)
                double y_wall = wall_dist_ptr[idx];
                
                // Estimate u_tau from local velocity magnitude
                double u_local = u_mag_ptr[idx];
                double u_tau_est = sqrt(nu * u_local / fmax(y_wall, 1e-10));
                u_tau_est = fmax(u_tau_est, 1e-6);
                double y_plus = y_wall * u_tau_est / nu;
                
                // Get strain/rotation from features (flat arrays, no ghosts)
                double S_mag = S_mag_ptr[idx];
                double Omega_mag = Omega_mag_ptr[idx];
                
                // Invariants
                double I1_S = S_mag * S_mag;
                double I1_Omega = -Omega_mag * Omega_mag;
                
                // GEP correction factor (inline simplified version)
                double f_gep;
                if (variant_int == 0) {  // WS2016_Channel
                    double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                    double f_wall = 1.0 - exp(-y_plus / 26.0);
                    f_wall = f_wall * f_wall;
                    double f_rot = 1.0 / (1.0 + 0.1 * ratio * ratio);
                    f_gep = f_wall * f_rot;
                } else if (variant_int == 1) {  // WS2016_PeriodicHill
                    double ratio = (S_mag > 1e-10) ? Omega_mag / S_mag : 1.0;
                    double f_sep = 1.0 / (1.0 + 0.2 * ratio * ratio);
                    double f_wall = tanh(y_plus / 50.0);
                    f_gep = f_wall * f_sep;
                } else {  // Simple
                    double f_damp = 1.0 - exp(-y_plus / 26.0);
                    f_gep = f_damp * f_damp;
                }
                
                // Mixing length with van Driest damping
                double f_damp = 1.0 - exp(-y_plus / A_plus);
                double l_mix = kappa * y_wall * f_damp;
                
                // Eddy viscosity
                double nu_t_val = l_mix * l_mix * S_mag * f_gep;
                
                // Clipping
                if (nu_t_val < 0.0) nu_t_val = 0.0;
                if (nu_t_val > nu_t_max) nu_t_val = nu_t_max;
                
                nu_t_gpu_ptr[cell_idx] = nu_t_val;  // Use stride-based index
            }
            
            // Download result from GPU
            #pragma omp target update from(nu_t_gpu_ptr[0:total_size])
            
            // Copy back to provided nu_t field
            std::copy(nu_t_gpu_flat_.begin(), nu_t_gpu_flat_.end(), nu_t.data().begin());
            
            return;
        }
#endif
        
        // CPU path
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                int idx = (j - mesh.j_begin()) * mesh.Nx + (i - mesh.i_begin());
                
                // Get wall distance
                double y_wall = mesh.wall_distance(i, j);
                
                // Compute y+ (wall units)
                double u_local = std::sqrt(velocity.u(i, j) * velocity.u(i, j) + 
                                          velocity.v(i, j) * velocity.v(i, j));
                double u_tau_est = std::sqrt(nu_ * u_local / std::max(y_wall, 1e-10));
                u_tau_est = std::max(u_tau_est, 1e-6);
                double y_plus = y_wall * u_tau_est / nu_;
                
                // Get strain rate magnitude from features
                double S_mag = 0.0;
                if (idx < static_cast<int>(features.size()) && features[idx].size() > 0) {
                    S_mag = features[idx][0];
                }
                
                // Compute invariants for GEP correction
                double I1_S = S_mag * S_mag;
                double Omega_mag = (idx < static_cast<int>(features.size()) && features[idx].size() > 1) 
                                   ? features[idx][1] : 0.0;
                double I1_Omega = -Omega_mag * Omega_mag;
                double I2_S = 0.0;
                double I2_Omega = 0.0;
                
                double Re_d = u_ref_ * delta / nu_;
                
                // Compute GEP correction factor
                double f_gep = compute_gep_factor(I1_S, I2_S, I1_Omega, I2_Omega, y_plus, Re_d);
                
                // Mixing length model with GEP correction
                double l_mix = kappa * y_wall;
                double f_damp = 1.0 - std::exp(-y_plus / A_plus);
                l_mix *= f_damp;
                
                // Eddy viscosity
                double nu_t_val = l_mix * l_mix * S_mag * f_gep;
                
                // Clipping
                nu_t_val = std::max(0.0, nu_t_val);
                nu_t_val = std::min(nu_t_val, nu_t_max_);
                
                nu_t(i, j) = nu_t_val;
            }
        }
    }
}

} // namespace nncfd

