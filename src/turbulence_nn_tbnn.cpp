#include "turbulence_nn_tbnn.hpp"
#include "timing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

TurbulenceNNTBNN::TurbulenceNNTBNN()
    : feature_computer_(Mesh()) {}

TurbulenceNNTBNN::~TurbulenceNNTBNN() {
    free_gpu_buffers();
}

void TurbulenceNNTBNN::load(const std::string& weights_dir, const std::string& scaling_dir) {
    mlp_.load_weights(weights_dir);
    
    try {
        mlp_.load_scaling(scaling_dir + "/input_means.txt",
                         scaling_dir + "/input_stds.txt");
    } catch (const std::exception& e) {
        // Scaling files optional
    }
}

void TurbulenceNNTBNN::upload_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    
    if (num_devices > 0) {
        mlp_.upload_to_gpu();
        gpu_ready_ = mlp_.is_on_gpu();
    } else {
        gpu_ready_ = false;
    }
#endif
}

void TurbulenceNNTBNN::allocate_gpu_buffers(int n_cells) {
#ifdef USE_GPU_OFFLOAD
    if (n_cells == cached_n_cells_ && !features_flat_.empty()) {
        return;  // Already allocated
    }
    
    free_gpu_buffers();
    
    int feature_dim = mlp_.input_dim();
    int output_dim = mlp_.output_dim();
    size_t workspace_size = mlp_.workspace_size(n_cells);
    
    // Allocate CPU buffers
    features_flat_.resize(n_cells * feature_dim);
    outputs_flat_.resize(n_cells * output_dim);
    workspace_.resize(workspace_size);
    
    // Map to GPU
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
    
    cached_n_cells_ = n_cells;
#else
    (void)n_cells;
#endif
}

void TurbulenceNNTBNN::free_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    if (!features_flat_.empty()) {
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
    }
#endif
    features_flat_.clear();
    outputs_flat_.clear();
    workspace_.clear();
    cached_n_cells_ = 0;
}

void TurbulenceNNTBNN::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        
        int n_interior = mesh.Nx * mesh.Ny;
        features_.resize(n_interior);
        basis_.resize(n_interior);
        
        initialized_ = true;
        
        initialized_ = true;
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
    
    // Compute features and tensor basis (CPU)
    {
        TIMED_SCOPE("nn_tbnn_features");
        feature_computer_.compute_tbnn_features(velocity, k_local, omega_local, 
                                                features_, basis_);
    }
    
    int n_cells = mesh.Nx * mesh.Ny;
    int feature_dim = mlp_.input_dim();
    int output_dim = mlp_.output_dim();
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: batched inference
    if (gpu_ready_) {
        TIMED_SCOPE("nn_tbnn_inference_gpu");
        
        // Ensure GPU buffers are allocated
        allocate_gpu_buffers(n_cells);
        
        // Flatten features for GPU (CPU side)
        {
            TIMED_SCOPE("nn_tbnn_flatten");
            for (int idx = 0; idx < n_cells; ++idx) {
                for (int f = 0; f < feature_dim; ++f) {
                    features_flat_[idx * feature_dim + f] = features_[idx].values[f];
                }
            }
        }
        
        // Upload features to GPU
        {
            TIMED_SCOPE("nn_tbnn_upload");
            double* feat_ptr = features_flat_.data();
            size_t feat_size = features_flat_.size();
            #pragma omp target update to(feat_ptr[0:feat_size])
        }
        
        // Run batched NN inference on GPU
        {
            TIMED_SCOPE("nn_tbnn_kernel");
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            mlp_.forward_batch_gpu(feat_ptr, out_ptr, n_cells, work_ptr);
        }
        
        // Download outputs from GPU
        {
            TIMED_SCOPE("nn_tbnn_download");
            double* out_ptr = outputs_flat_.data();
            size_t out_size = outputs_flat_.size();
            #pragma omp target update from(out_ptr[0:out_size])
        }
        
        // Post-process: construct anisotropy and compute nu_t (CPU)
        {
            TIMED_SCOPE("nn_tbnn_postprocess");
            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    // Extract G coefficients from NN output
                    std::array<double, TensorBasis::NUM_BASIS> G;
                    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
                        G[n] = (n < output_dim) ? outputs_flat_[idx * output_dim + n] : 0.0;
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
                    
                    // Compute equivalent eddy viscosity
                    auto grad = compute_velocity_gradient(mesh, velocity, i, j);
                    double Sxy = grad.Sxy();
                    double k_val = k_local(i, j);
                    
                    if (std::abs(Sxy) > 1e-10) {
                        nu_t(i, j) = std::abs(-b_xy * k_val / Sxy);
                    } else {
                        double S_mag = grad.S_mag();
                        if (S_mag > 1e-10) {
                            double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                            nu_t(i, j) = k_val * b_mag / S_mag;
                        } else {
                            nu_t(i, j) = 0.0;
                        }
                    }
                    
                    // Ensure positivity and clip to reasonable bounds
                    nu_t(i, j) = std::max(0.0, std::min(nu_t(i, j), 10.0 * nu_));
                    
                    if (std::isnan(nu_t(i, j)) || std::isinf(nu_t(i, j))) {
                        nu_t(i, j) = 0.0;
                    }
                    
                    ++idx;
                }
            }
        }
    } else
#endif
    {
        // CPU fallback path: sequential inference
        TIMED_SCOPE("nn_tbnn_inference_cpu");
        
        std::vector<double> buffer1, buffer2;
        
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
                auto grad = compute_velocity_gradient(mesh, velocity, i, j);
                double Sxy = grad.Sxy();
                double k_val = k_local(i, j);
                
                if (std::abs(Sxy) > 1e-10) {
                    nu_t(i, j) = std::abs(-b_xy * k_val / Sxy);
                } else {
                    double S_mag = grad.S_mag();
                    if (S_mag > 1e-10) {
                        double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                        nu_t(i, j) = k_val * b_mag / S_mag;
                    } else {
                        nu_t(i, j) = 0.0;
                    }
                }
                
                // Ensure positivity and clip to reasonable bounds
                nu_t(i, j) = std::max(0.0, std::min(nu_t(i, j), 10.0 * nu_));
                
                if (std::isnan(nu_t(i, j)) || std::isinf(nu_t(i, j))) {
                    nu_t(i, j) = 0.0;
                }
                
                ++idx;
            }
        }
    }
}

} // namespace nncfd
