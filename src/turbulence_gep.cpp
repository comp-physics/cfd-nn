#include "turbulence_gep.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>

namespace nncfd {

TurbulenceGEP::TurbulenceGEP()
    : feature_computer_(Mesh()) {
}

void TurbulenceGEP::initialize(const Mesh& mesh, const VectorField& velocity) {
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
            
            // Basic mixing length: l = kappa * y * (1 - exp(-y+/A+))
            double kappa = 0.41;
            double A_plus = 26.0;
            
            // van Driest damping
            double f_damp = 1.0 - std::exp(-y_plus / A_plus);
            f_damp = f_damp * f_damp;
            
            // Mixing length squared (normalized)
            double l_plus = kappa * y_plus * f_damp;
            
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
    TIMED_SCOPE("gep_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, u_ref_, delta_);
    
    double delta = delta_;
    double kappa = 0.41;
    double A_plus = 26.0;
    
    // Compute velocity gradients for all cells
    std::vector<Features> features;
    {
        TIMED_SCOPE("gep_features");
        feature_computer_.compute_scalar_features(velocity, k, omega, features);
    }
    
    {
        TIMED_SCOPE("gep_compute");
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                int idx = (j - mesh.j_begin()) * mesh.Nx + (i - mesh.i_begin());
                
                // Get wall distance
                double y_wall = mesh.wall_distance(i, j);
                
                // Compute y+ (wall units)
                // Estimate friction velocity from local velocity gradient
                double u_local = std::sqrt(velocity.u(i, j) * velocity.u(i, j) + 
                                          velocity.v(i, j) * velocity.v(i, j));
                double u_tau_est = std::sqrt(nu_ * u_local / std::max(y_wall, 1e-10));
                u_tau_est = std::max(u_tau_est, 1e-6);
                double y_plus = y_wall * u_tau_est / nu_;
                
                // Get strain rate magnitude from features
                double S_mag = 0.0;
                if (idx < static_cast<int>(features.size()) && features[idx].size() > 0) {
                    S_mag = features[idx][0];  // First feature is strain magnitude
                }
                
                // Compute invariants for GEP correction
                // I1_S = tr(S^2) ~ S_mag^2
                // I1_Omega = tr(Omega^2) ~ -Omega_mag^2
                double I1_S = S_mag * S_mag;
                double Omega_mag = (idx < static_cast<int>(features.size()) && features[idx].size() > 1) 
                                   ? features[idx][1] : 0.0;
                double I1_Omega = -Omega_mag * Omega_mag;
                double I2_S = 0.0;  // Higher invariants (not used in simple model)
                double I2_Omega = 0.0;
                
                double Re_d = u_ref_ * delta / nu_;
                
                // Compute GEP correction factor
                double f_gep = compute_gep_factor(I1_S, I2_S, I1_Omega, I2_Omega, y_plus, Re_d);
                
                // Mixing length model with GEP correction
                double l_mix = kappa * y_wall;
                
                // van Driest damping
                double f_damp = 1.0 - std::exp(-y_plus / A_plus);
                l_mix *= f_damp;
                
                // Eddy viscosity: nu_t = l^2 * |S| * f_gep
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

