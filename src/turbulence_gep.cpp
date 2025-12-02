#include "turbulence_gep.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

TurbulenceGEP::TurbulenceGEP()
    : feature_computer_(Mesh()) {
}

void TurbulenceGEP::initialize(const Mesh& mesh, const VectorField& velocity) {
    (void)velocity;  // Reserved for future use
    ensure_initialized(mesh);
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
                           TensorField* tau_ij) {
    (void)tau_ij;  // GEP provides eddy viscosity only, not explicit Reynolds stresses
    TIMED_SCOPE("gep_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, u_ref_, delta_);
    
    [[maybe_unused]] double delta = delta_;  // Used in CPU path below
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
        // GPU path if available
        if (omp_get_num_devices() > 0) {
            const int n_cells = mesh.Nx * mesh.Ny;
            
            // Flatten features for GPU
            std::vector<double> S_mag_flat(n_cells);
            std::vector<double> Omega_mag_flat(n_cells);
            for (int idx = 0; idx < n_cells; ++idx) {
                if (idx < static_cast<int>(features.size()) && features[idx].size() > 0) {
                    S_mag_flat[idx] = features[idx][0];
                    Omega_mag_flat[idx] = (features[idx].size() > 1) ? features[idx][1] : 0.0;
                } else {
                    S_mag_flat[idx] = 0.0;
                    Omega_mag_flat[idx] = 0.0;
                }
            }
            
            // Precompute wall distances
            std::vector<double> wall_dist_flat(n_cells);
            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    wall_dist_flat[idx++] = mesh.wall_distance(i, j);
                }
            }
            
            // Get pointers
            const double* S_mag_ptr = S_mag_flat.data();
            const double* Omega_mag_ptr = Omega_mag_flat.data();
            const double* wall_dist_ptr = wall_dist_flat.data();
            const double* u_ptr = velocity.u_field().data().data();
            const double* v_ptr = velocity.v_field().data().data();
            double* nu_t_ptr = nu_t.data().data();
            
            const int variant_int = static_cast<int>(variant_);
            const double nu = nu_;
            const double nu_t_max = nu_t_max_;
            
            // Flatten velocity magnitudes (avoid complex indexing on GPU)
            std::vector<double> u_mag_flat(n_cells);
            idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u_val = velocity.u(i, j);
                    double v_val = velocity.v(i, j);
                    u_mag_flat[idx++] = std::sqrt(u_val * u_val + v_val * v_val);
                }
            }
            const double* u_mag_ptr = u_mag_flat.data();
            
            // GPU kernel
            #pragma omp target teams distribute parallel for \
                map(to: S_mag_ptr[0:n_cells], Omega_mag_ptr[0:n_cells], \
                        wall_dist_ptr[0:n_cells], u_mag_ptr[0:n_cells]) \
                map(from: nu_t_ptr[0:n_cells])
            for (int idx = 0; idx < n_cells; ++idx) {
                // Get wall distance
                double y_wall = wall_dist_ptr[idx];
                
                // Estimate u_tau from local velocity magnitude
                double u_local = u_mag_ptr[idx];
                double u_tau_est = sqrt(nu * u_local / fmax(y_wall, 1e-10));
                u_tau_est = fmax(u_tau_est, 1e-6);
                double y_plus = y_wall * u_tau_est / nu;
                
                // Get strain/rotation from features
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
                
                nu_t_ptr[idx] = nu_t_val;
            }
            
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

