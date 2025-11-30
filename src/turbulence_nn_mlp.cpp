#include "turbulence_nn_mlp.hpp"
#include "turbulence_baseline.hpp"
#include "timing.hpp"
#include <algorithm>

namespace nncfd {

TurbulenceNNMLP::TurbulenceNNMLP()
    : feature_computer_(Mesh()) {  // Will be re-initialized
}

void TurbulenceNNMLP::load(const std::string& weights_dir, const std::string& scaling_dir) {
    mlp_.load_weights(weights_dir);
    
    // Load scaling if available
    try {
        mlp_.load_scaling(scaling_dir + "/input_means.txt", 
                         scaling_dir + "/input_stds.txt");
    } catch (const std::exception& e) {
        // Scaling files optional
    }
}

void TurbulenceNNMLP::ensure_initialized(const Mesh& mesh) {
    if (features_.empty()) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        
        // Allocate work buffers
        int n_interior = mesh.Nx * mesh.Ny;
        features_.resize(n_interior);
        
        if (blend_with_baseline_ && !baseline_) {
            baseline_ = std::make_unique<MixingLengthModel>();
            baseline_->set_nu(nu_);
            auto* ml = dynamic_cast<MixingLengthModel*>(baseline_.get());
            if (ml) {
                ml->set_delta(delta_);
            }
            baseline_nu_t_ = ScalarField(mesh);
        }
    }
}

void TurbulenceNNMLP::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij) {
    
    TIMED_SCOPE("nn_mlp_update");
    
    (void)tau_ij;
    
    ensure_initialized(mesh);
    
    // Update reference quantities
    feature_computer_.set_reference(nu_, delta_, u_ref_);
    
    // Compute features for all cells
    {
        TIMED_SCOPE("nn_mlp_features");
        feature_computer_.compute_scalar_features(velocity, k, omega, features_);
    }
    
    // Compute baseline if blending
    if (blend_with_baseline_ && baseline_) {
        TIMED_SCOPE("nn_mlp_baseline");
        baseline_->update(mesh, velocity, k, omega, baseline_nu_t_);
    }
    
    // NN inference for each cell
    {
        TIMED_SCOPE("nn_mlp_inference");
        
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Forward pass
                std::vector<double> output = mlp_.forward(features_[idx].values);
                
                // Output is raw nu_t prediction
                double nu_t_nn = output.empty() ? 0.0 : output[0];
                
                // Ensure positivity and apply clipping
                nu_t_nn = std::max(0.0, nu_t_nn);
                nu_t_nn = std::min(nu_t_nn, nu_t_max_);
                
                // Apply blending with baseline if enabled
                if (blend_with_baseline_ && baseline_) {
                    nu_t(i, j) = (1.0 - blend_alpha_) * baseline_nu_t_(i, j) 
                               + blend_alpha_ * nu_t_nn;
                } else {
                    nu_t(i, j) = nu_t_nn;
                }
                
                ++idx;
            }
        }
    }
}

} // namespace nncfd


