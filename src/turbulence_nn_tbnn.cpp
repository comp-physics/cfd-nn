#include "turbulence_nn_tbnn.hpp"
#include "timing.hpp"
#include <algorithm>
#include <cmath>

namespace nncfd {

TurbulenceNNTBNN::TurbulenceNNTBNN()
    : feature_computer_(Mesh()) {}

void TurbulenceNNTBNN::load(const std::string& weights_dir, const std::string& scaling_dir) {
    mlp_.load_weights(weights_dir);
    
    try {
        mlp_.load_scaling(scaling_dir + "/input_means.txt",
                         scaling_dir + "/input_stds.txt");
    } catch (const std::exception& e) {
        // Scaling files optional
    }
}

void TurbulenceNNTBNN::ensure_initialized(const Mesh& mesh) {
    if (features_.empty()) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        
        int n_interior = mesh.Nx * mesh.Ny;
        features_.resize(n_interior);
        basis_.resize(n_interior);
    }
}

void TurbulenceNNTBNN::estimate_k(const Mesh& mesh, const VectorField& velocity, 
                                  ScalarField& k) {
    // Simple algebraic estimate of k from velocity gradient
    // k ~ (nu_t * |S|^2) / C_mu  or  k ~ 0.1 * U^2 near walls
    
    const double C_mu = 0.09;
    
    // First estimate friction velocity
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_avg = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (j+1 < mesh.j_end() && j-1 >= mesh.j_begin()) {
                double dudy = (velocity.u(i, j+1) - velocity.u(i, j-1)) / (2.0 * mesh.dy);
                dudy_avg += std::abs(dudy);
                ++count;
            }
        }
        if (count > 0) {
            dudy_avg /= count;
            u_tau = std::sqrt(nu_ * dudy_avg);
        }
    }
    
    u_tau = std::max(u_tau, 1e-6);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / (nu_ + 1e-20);
            
            // van Driest-like damping
            double f_mu = 1.0 - std::exp(-std::min(y_plus / 26.0, 20.0));
            
            // k in log layer ~ u_tau^2 / sqrt(C_mu)
            double k_est = (u_tau * u_tau / std::sqrt(C_mu)) * f_mu * f_mu;
            k(i, j) = std::max(k_min_, std::min(k_est, 10.0 * u_tau * u_tau));
        }
    }
}

void TurbulenceNNTBNN::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k_in,
    const ScalarField& omega_in,
    ScalarField& nu_t,
    TensorField* tau_ij) {
    
    TIMED_SCOPE("nn_tbnn_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, delta_, u_ref_);
    
    // Use provided k/omega or estimate
    ScalarField k_local(mesh);
    ScalarField omega_local(mesh);
    
    // Check if k is provided (non-zero values)
    bool k_provided = false;
    for (int j = mesh.j_begin(); j < mesh.j_end() && !k_provided; ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end() && !k_provided; ++i) {
            if (k_in(i, j) > k_min_) {
                k_provided = true;
            }
        }
    }
    
    if (k_provided) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                k_local(i, j) = k_in(i, j);
                omega_local(i, j) = omega_in(i, j);
            }
        }
    } else {
        estimate_k(mesh, velocity, k_local);
        // Estimate omega from k
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double y_wall = mesh.wall_distance(i, j);
                // omega ~ k / (C_mu * nu_t) or ~ 1/(kappa * y)
                omega_local(i, j) = std::sqrt(k_local(i, j)) / (0.41 * std::max(y_wall, 1e-10));
            }
        }
    }
    
    // Compute features and tensor basis
    {
        TIMED_SCOPE("nn_tbnn_features");
        feature_computer_.compute_tbnn_features(velocity, k_local, omega_local, 
                                                features_, basis_);
    }
    
    // NN inference to get G coefficients, construct anisotropy and Reynolds stresses
    {
        TIMED_SCOPE("nn_tbnn_inference");
        
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // NN forward pass
                std::vector<double> output = mlp_.forward(features_[idx].values);
                
                // Output should be G coefficients (NUM_BASIS values)
                std::array<double, TensorBasis::NUM_BASIS> G;
                for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
                    G[n] = (n < static_cast<int>(output.size())) ? output[n] : 0.0;
                }
                
                // Construct anisotropy tensor
                double b_xx, b_xy, b_yy;
                TensorBasis::construct_anisotropy(G, basis_[idx], b_xx, b_xy, b_yy);
                
                // Convert to Reynolds stresses if requested
                if (tau_ij) {
                    double k_val = k_local(i, j);
                    double tau_xx, tau_xy, tau_yy;
                    TensorBasis::anisotropy_to_reynolds_stress(b_xx, b_xy, b_yy, k_val,
                                                              tau_xx, tau_xy, tau_yy);
                    tau_ij->xx(i, j) = tau_xx;
                    tau_ij->xy(i, j) = tau_xy;
                    tau_ij->yy(i, j) = tau_yy;
                }
                
                // Also compute equivalent eddy viscosity
                // From b_ij = -nu_t * S_ij / k (Boussinesq), approximate:
                // nu_t ~ -b_xy * k / S_xy (if S_xy != 0)
                auto grad = compute_velocity_gradient(mesh, velocity, i, j);
                double Sxy = grad.Sxy();
                double k_val = k_local(i, j);
                
                if (std::abs(Sxy) > 1e-10) {
                    nu_t(i, j) = std::abs(-b_xy * k_val / Sxy);
                } else {
                    // Fallback: use trace relation
                    double S_mag = grad.S_mag();
                    if (S_mag > 1e-10) {
                        // nu_t ~ k * |b| / |S|
                        double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                        nu_t(i, j) = k_val * b_mag / S_mag;
                    } else {
                        nu_t(i, j) = 0.0;
                    }
                }
                
                // Ensure positivity and clip to reasonable bounds
                nu_t(i, j) = std::max(0.0, std::min(nu_t(i, j), 10.0 * nu_));
                
                // Debug: Check for problematic values
                if (std::isnan(nu_t(i, j)) || std::isinf(nu_t(i, j))) {
                    nu_t(i, j) = 0.0;  // Fallback to zero
                }
                
                ++idx;
            }
        }
    }
}

} // namespace nncfd


