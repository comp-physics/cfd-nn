#include "turbulence_nn_mlp.hpp"
#include "turbulence_baseline.hpp"
#include "timing.hpp"
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

TurbulenceNNMLP::TurbulenceNNMLP()
    : feature_computer_(Mesh()) {}

TurbulenceNNMLP::~TurbulenceNNMLP() {
    free_gpu_buffers();
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

void TurbulenceNNMLP::upload_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (!gpu_ready_) {
        mlp_.upload_to_gpu();
        gpu_ready_ = mlp_.is_on_gpu();
    }
#endif
}

void TurbulenceNNMLP::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    if (!gpu_available()) {
        gpu_ready_ = false;
        return;
    }
    
    const int n_cells = mesh.Nx * mesh.Ny;
    upload_to_gpu();  // Upload MLP weights if not already done
    allocate_gpu_buffers(n_cells);
#else
    (void)mesh;
    gpu_ready_ = false;
#endif
}

void TurbulenceNNMLP::cleanup_gpu_buffers() {
    free_gpu_buffers();
    gpu_ready_ = false;
}

void TurbulenceNNMLP::allocate_gpu_buffers(int n_cells) {
#ifdef USE_GPU_OFFLOAD
    if (n_cells == cached_n_cells_ && !features_flat_.empty() && buffers_on_gpu_) {
        return;  // Already allocated and mapped
    }
    
    free_gpu_buffers();
    
    int feature_dim = mlp_.input_dim();
    int output_dim = mlp_.output_dim();
    size_t workspace_size = mlp_.workspace_size(n_cells);
    
    // Allocate CPU buffers
    features_flat_.resize(n_cells * feature_dim);
    outputs_flat_.resize(n_cells * output_dim);
    workspace_.resize(workspace_size);
    
    // Map to GPU only if we have valid data
    if (!features_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
        double* feat_ptr = features_flat_.data();
        double* out_ptr = outputs_flat_.data();
        double* work_ptr = workspace_.data();
        size_t feat_size = features_flat_.size();
        size_t out_size = outputs_flat_.size();
        size_t work_size = workspace_.size();
        
        #pragma omp target enter data \
            map(alloc: feat_ptr[0:feat_size]) \
            map(alloc: out_ptr[0:out_size]) \
            map(alloc: work_ptr[0:work_size])
        
        buffers_on_gpu_ = true;
    }
    
    cached_n_cells_ = n_cells;
#else
    (void)n_cells;
#endif
}

void TurbulenceNNMLP::free_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    // Only free GPU buffers if they were actually mapped to GPU
    if (buffers_on_gpu_) {
        // Check vectors are non-empty before unmapping
        if (!features_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
            // Set flag FIRST to prevent re-entry
            buffers_on_gpu_ = false;
            
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            size_t feat_size = features_flat_.size();
            size_t out_size = outputs_flat_.size();
            size_t work_size = workspace_.size();
            
            #pragma omp target exit data \
                map(delete: feat_ptr[0:feat_size]) \
                map(delete: out_ptr[0:out_size]) \
                map(delete: work_ptr[0:work_size])
        } else {
            buffers_on_gpu_ = false;  // Clear flag even if vectors are empty
        }
    }
#endif
    features_flat_.clear();
    outputs_flat_.clear();
    workspace_.clear();
    cached_n_cells_ = 0;
}

void TurbulenceNNMLP::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
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
        
        initialized_ = true;
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
    
    // Compute features for all cells (CPU)
    {
        TIMED_SCOPE("nn_mlp_features");
        feature_computer_.compute_scalar_features(velocity, k, omega, features_);
    }
    
    // Compute baseline if blending
    if (blend_with_baseline_ && baseline_) {
        TIMED_SCOPE("nn_mlp_baseline");
        baseline_->update(mesh, velocity, k, omega, baseline_nu_t_);
    }
    
    [[maybe_unused]] int n_cells = mesh.Nx * mesh.Ny;
    [[maybe_unused]] int feature_dim = mlp_.input_dim();
    [[maybe_unused]] int output_dim = mlp_.output_dim();
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: batched inference
    if (gpu_ready_) {
        TIMED_SCOPE("nn_mlp_inference_gpu");
        
        // Ensure GPU buffers are allocated
        allocate_gpu_buffers(n_cells);
        
        // Flatten features for GPU (CPU side)
        {
            TIMED_SCOPE("nn_mlp_flatten");
            for (int idx = 0; idx < n_cells; ++idx) {
                for (int f = 0; f < feature_dim; ++f) {
                    features_flat_[idx * feature_dim + f] = features_[idx].values[f];
                }
            }
        }
        
        // Upload features to GPU
        {
            TIMED_SCOPE("nn_mlp_upload");
            double* feat_ptr = features_flat_.data();
            size_t feat_size = features_flat_.size();
            #pragma omp target update to(feat_ptr[0:feat_size])
        }
        
        // Run batched NN inference on GPU
        {
            TIMED_SCOPE("nn_mlp_kernel");
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            mlp_.forward_batch_gpu(feat_ptr, out_ptr, n_cells, work_ptr);
        }
        
        // Download outputs from GPU
        {
            TIMED_SCOPE("nn_mlp_download");
            double* out_ptr = outputs_flat_.data();
            size_t out_size = outputs_flat_.size();
            #pragma omp target update from(out_ptr[0:out_size])
        }
        
        // Post-process: apply clipping and blending (CPU)
        {
            TIMED_SCOPE("nn_mlp_postprocess");
            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    // Output is raw nu_t prediction
                    double nu_t_nn = outputs_flat_[idx * output_dim];
                    
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
    } else
#endif
    {
        // CPU fallback: sequential inference
        TIMED_SCOPE("nn_mlp_inference_cpu");
        
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
