#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// Activation function types
enum class Activation {
    Linear,
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU
};

/// Dense (fully connected) layer with GPU support
/// Weights are stored contiguously for efficient GPU access
struct DenseLayer {
    int in_dim;
    int out_dim;
    std::vector<double> W;   ///< Weights: row-major, shape (out_dim, in_dim)
    std::vector<double> b;   ///< Biases: shape (out_dim)
    
    DenseLayer() : in_dim(0), out_dim(0) {}
    DenseLayer(int in, int out);
    
    /// Forward pass: y = Wx + b (CPU)
    void forward(const double* x, double* y) const;
    
    /// Forward pass with std::vector
    std::vector<double> forward(const std::vector<double>& x) const;
    
    /// Load weights from files
    void load_weights(const std::string& W_file, const std::string& b_file);
};

/// Multi-layer perceptron with GPU acceleration
/// 
/// GPU Strategy:
/// - Weights are uploaded to GPU once via sync_weights_to_gpu()
/// - Weights remain on GPU for entire simulation
/// - forward_batch_gpu() processes all grid cells in parallel
/// - Input/output buffers are managed externally for zero-copy
class MLP {
public:
    MLP() = default;
    ~MLP();
    
    // Prevent copying (GPU resources)
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    
    // Allow moving
    MLP(MLP&& other) noexcept;
    MLP& operator=(MLP&& other) noexcept;
    
    /// Construct from layer dimensions (e.g., {10, 64, 64, 1})
    /// All hidden layers use the specified activation; output is linear
    MLP(const std::vector<int>& dims, Activation hidden_act = Activation::Tanh);
    
    /// Forward pass (CPU, single sample)
    std::vector<double> forward(const std::vector<double>& x) const;
    
    /// Forward pass with pre-allocated buffers (CPU, avoids allocation)
    void forward(const double* x, double* output, 
                 std::vector<double>& buffer1, std::vector<double>& buffer2) const;
    
    /// GPU-accelerated batched forward pass
    /// All pointers must already be on GPU (use with target data regions)
    /// @param x_batch Input features [batch_size * input_dim], device pointer
    /// @param y_batch Output [batch_size * output_dim], device pointer
    /// @param batch_size Number of samples to process
    /// @param workspace Temporary buffer [batch_size * max_layer_dim * 2], device pointer
    void forward_batch_gpu(double* x_batch, double* y_batch, 
                           int batch_size, double* workspace) const;
    
    /// Load all weights from directory
    /// Expects files: layer0_W.txt, layer0_b.txt, layer1_W.txt, ...
    void load_weights(const std::string& dir);
    
    /// Load scaling parameters (input normalization)
    void load_scaling(const std::string& means_file, const std::string& stds_file);
    
    /// Apply input scaling (CPU)
    void scale_input(std::vector<double>& x) const;
    
    /// Upload all weights and scaling to GPU (call once after loading)
    void sync_weights_to_gpu();
    
    /// Check if weights are on GPU
    bool is_on_gpu() const { return gpu_ready_; }
    
    /// Free GPU memory
    void free_gpu();
    
    /// Get dimensions
    int input_dim() const { return layers_.empty() ? 0 : layers_[0].in_dim; }
    int output_dim() const { return layers_.empty() ? 0 : layers_.back().out_dim; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    int max_layer_dim() const;
    
    /// Get workspace size needed for batched GPU forward
    size_t workspace_size(int batch_size) const;
    
    /// Access layers (for testing)
    const std::vector<DenseLayer>& layers() const { return layers_; }
    const std::vector<Activation>& activations() const { return activations_; }
    
    /// Set layer and activation manually
    void add_layer(const DenseLayer& layer, Activation act);
    
    /// Access scaling for GPU kernels
    bool has_scaling() const { return has_scaling_; }
    const double* input_means_ptr() const { return input_means_.data(); }
    const double* input_stds_ptr() const { return input_stds_.data(); }
    int scaling_size() const { return static_cast<int>(input_means_.size()); }
    
    /// Access layer info
    const DenseLayer& layer(int i) const { return layers_[i]; }
    
    /// GPU pointer accessors (for full GPU pipeline)
    /// These return device pointers when GPU is ready, nullptr otherwise
    const double* weights_gpu() const { return gpu_ready_ ? all_weights_.data() : nullptr; }
    const double* biases_gpu() const { return gpu_ready_ ? all_biases_.data() : nullptr; }
    const int* weight_offsets_gpu() const { return gpu_ready_ ? weight_offsets_.data() : nullptr; }
    const int* bias_offsets_gpu() const { return gpu_ready_ ? bias_offsets_.data() : nullptr; }
    const int* layer_dims_gpu() const { return gpu_ready_ ? layer_dims_.data() : nullptr; }
    const int* activation_types_gpu() const { return gpu_ready_ ? activation_types_.data() : nullptr; }
    const double* input_means_gpu() const { return (gpu_ready_ && has_scaling_) ? input_means_.data() : nullptr; }
    const double* input_stds_gpu() const { return (gpu_ready_ && has_scaling_) ? input_stds_.data() : nullptr; }
    int scale_size() const { return has_scaling_ ? static_cast<int>(input_means_.size()) : 0; }
    
private:
    std::vector<DenseLayer> layers_;
    std::vector<Activation> activations_;  ///< One per layer (after the layer)
    
    // Input scaling parameters
    std::vector<double> input_means_;
    std::vector<double> input_stds_;
    bool has_scaling_ = false;
    
    // GPU state
    bool gpu_ready_ = false;
    
    // Flattened weight arrays for GPU (contiguous memory)
    std::vector<double> all_weights_;  ///< All W matrices concatenated
    std::vector<double> all_biases_;   ///< All b vectors concatenated
    std::vector<int> weight_offsets_;  ///< Offset into all_weights_ for each layer
    std::vector<int> bias_offsets_;    ///< Offset into all_biases_ for each layer
    std::vector<int> layer_dims_;      ///< [in0, out0, in1, out1, ...]
    std::vector<int> activation_types_; ///< Activation enum as int for GPU
    
    /// Flatten weights for GPU upload
    void flatten_weights();
};

/// Apply activation function in-place (CPU)
void apply_activation(double* x, int n, Activation act);

/// Apply activation function element-wise
inline double activate(double x, Activation act) {
    switch (act) {
        case Activation::Linear: return x;
        case Activation::ReLU: return x > 0 ? x : 0;
        case Activation::Tanh: return std::tanh(x);
        case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case Activation::Swish: return x / (1.0 + std::exp(-x));
        case Activation::GELU: {
            // Approximate GELU
            constexpr double c = 0.044715;
            double x3 = x * x * x;
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0/M_PI) * (x + c * x3)));
        }
        default: return x;
    }
}

/// Load a vector from text file (one value per line)
std::vector<double> load_vector(const std::string& filename);

/// Load a matrix from text file (space-separated, row-major)
std::vector<double> load_matrix(const std::string& filename, int& rows, int& cols);

} // namespace nncfd
