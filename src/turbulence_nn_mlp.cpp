#include "turbulence_nn_mlp.hpp"
#include "turbulence_baseline.hpp"
#include "timing.hpp"
#include "gpu_kernels.hpp"
#include <algorithm>

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
        mlp_.upload_to_gpu();  // Will throw if no GPU available
        gpu_ready_ = mlp_.is_on_gpu();
    }
#endif
}

void TurbulenceNNMLP::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    const int n_cells = mesh.Nx * mesh.Ny;
    upload_to_gpu();  // Upload MLP weights if not already done
    allocate_gpu_buffers(n_cells);
    gpu_ready_ = (mlp_.is_on_gpu() && buffers_on_gpu_);  // Set gpu_ready after successful allocation
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
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    TIMED_SCOPE("nn_mlp_update");
    
    (void)tau_ij;       // MLP doesn't compute anisotropic stresses
    (void)device_view;  // avoid -Wunused-parameter in CPU builds
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, delta_, u_ref_);
    
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    [[maybe_unused]] const int Ng = mesh.Nghost;
    [[maybe_unused]] const int n_cells = Nx * Ny;
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: require device_view and gpu_ready (no CPU fallback)
    if (!device_view || !gpu_ready_) {
        throw std::runtime_error("NN-MLP GPU pipeline requires device_view and GPU buffers initialized");
    }
    
    // Validate device_view has all required buffers
    if (!device_view->u_face || !device_view->v_face ||
        !device_view->k || !device_view->omega ||
        !device_view->dudx || !device_view->dudy || !device_view->dvdx || !device_view->dvdy ||
        !device_view->wall_distance ||
        !device_view->nu_t) {
        throw std::runtime_error("NN-MLP GPU pipeline: device_view missing required buffers");
    }
    
    {  // GPU pipeline scope
    
    // Ensure GPU buffers are allocated
    allocate_gpu_buffers(n_cells);
        
        const int total_cells = (Nx + 2*Ng) * (Ny + 2*Ng);
        const int u_total = velocity.u_total_size();
        const int v_total = velocity.v_total_size();
        const int cell_stride = Nx + 2*Ng;
        const int u_stride = Nx + 2*Ng + 1;
        const int v_stride = Nx + 2*Ng;
        
        // Step 1: Compute gradients on GPU (using solver-owned buffers)
        {
            TIMED_SCOPE("nn_mlp_gradients_gpu");
            gpu_kernels::compute_gradients_from_mac_gpu(
                device_view->u_face, device_view->v_face,
                device_view->dudx, device_view->dudy,
                device_view->dvdx, device_view->dvdy,
                Nx, Ny, Ng,
                mesh.dx, mesh.dy,
                u_stride, v_stride, cell_stride,
                u_total, v_total, total_cells
            );
        }
        
        // Step 2: Compute features on GPU
        {
            TIMED_SCOPE("nn_mlp_features_gpu");
            double* feat_ptr = features_flat_.data();
            gpu_kernels::compute_mlp_scalar_features_gpu(
                device_view->dudx, device_view->dudy,
                device_view->dvdx, device_view->dvdy,
                device_view->k, device_view->omega,
                device_view->wall_distance,
                device_view->u_face, device_view->v_face,
                feat_ptr,  // Output: n_cells * 6 (already on GPU)
                Nx, Ny, Ng,
                cell_stride, u_stride, v_stride,
                total_cells, u_total, v_total,
                nu_, delta_, u_ref_
            );
        }
        
        // Step 3: Run NN inference on GPU
        {
            TIMED_SCOPE("nn_mlp_inference_gpu");
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            mlp_.forward_batch_gpu(feat_ptr, out_ptr, n_cells, work_ptr);
        }
        
        // Step 4: Postprocess outputs and write to nu_t field on GPU
        {
            TIMED_SCOPE("nn_mlp_postprocess_gpu");
            double* out_ptr = outputs_flat_.data();
            double* nu_t_ptr = nu_t.data().data();
            
            gpu_kernels::postprocess_mlp_outputs_gpu(
                out_ptr,      // NN outputs (n_cells * 1)
                nu_t_ptr,     // nu_t field with ghosts
                Nx, Ny, Ng,
                cell_stride,
                nu_t_max_
            );
        }
    }  // End GPU pipeline scope
    
#else
    // CPU path (only for CPU builds)
    ensure_initialized(mesh);
    
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
    
    // CPU-only sequential inference path
    {
        TIMED_SCOPE("nn_mlp_inference_cpu");
        
        int idx = 0;
        int nan_count = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Forward pass
                std::vector<double> output = mlp_.forward(features_[idx].values);
                
                // Output is raw nu_t prediction
                double nu_t_nn = output.empty() ? 0.0 : output[0];
                
                // Check for NaN/Inf and replace with safe value
                if (!std::isfinite(nu_t_nn)) {
                    ++nan_count;
                    nu_t_nn = (blend_with_baseline_ && baseline_) ? baseline_nu_t_(i, j) : 0.0;
                } else {
                    // Ensure positivity and apply clipping
                    nu_t_nn = std::max(0.0, nu_t_nn);
                    nu_t_nn = std::min(nu_t_nn, nu_t_max_);
                }
                
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
        
        // Warn if NaN/Inf detected
        if (nan_count > 0) {
            std::cerr << "[WARNING] NN-MLP (CPU) produced " << nan_count 
                      << " NaN/Inf values (replaced with fallback)" << std::endl;
        }
    }
#endif  // USE_GPU_OFFLOAD
}

} // namespace nncfd
