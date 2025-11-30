#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

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

/// Dense (fully connected) layer
struct DenseLayer {
    int in_dim;
    int out_dim;
    std::vector<double> W;   ///< Weights: row-major, shape (out_dim, in_dim)
    std::vector<double> b;   ///< Biases: shape (out_dim)
    
    DenseLayer() = default;
    DenseLayer(int in, int out);
    
    /// Forward pass: y = Wx + b
    void forward(const double* x, double* y) const;
    
    /// Forward pass with std::vector
    std::vector<double> forward(const std::vector<double>& x) const;
    
    /// Load weights from files
    void load_weights(const std::string& W_file, const std::string& b_file);
};

/// Multi-layer perceptron
class MLP {
public:
    MLP() = default;
    
    /// Construct from layer dimensions (e.g., {10, 64, 64, 1})
    /// All hidden layers use the specified activation; output is linear
    MLP(const std::vector<int>& dims, Activation hidden_act = Activation::Tanh);
    
    /// Forward pass
    std::vector<double> forward(const std::vector<double>& x) const;
    
    /// Forward pass with pre-allocated buffers (avoids allocation)
    void forward(const double* x, double* output, 
                 std::vector<double>& buffer1, std::vector<double>& buffer2) const;
    
    /// Load all weights from directory
    /// Expects files: layer0_W.txt, layer0_b.txt, layer1_W.txt, ...
    void load_weights(const std::string& dir);
    
    /// Load scaling parameters (input normalization)
    void load_scaling(const std::string& means_file, const std::string& stds_file);
    
    /// Apply input scaling
    void scale_input(std::vector<double>& x) const;
    
    /// Get dimensions
    int input_dim() const { return layers_.empty() ? 0 : layers_[0].in_dim; }
    int output_dim() const { return layers_.empty() ? 0 : layers_.back().out_dim; }
    int num_layers() const { return static_cast<int>(layers_.size()); }
    
    /// Access layers
    const std::vector<DenseLayer>& layers() const { return layers_; }
    const std::vector<Activation>& activations() const { return activations_; }
    
    /// Set layer and activation manually
    void add_layer(const DenseLayer& layer, Activation act);
    
private:
    std::vector<DenseLayer> layers_;
    std::vector<Activation> activations_;  ///< One per layer (after the layer)
    
    // Input scaling parameters
    std::vector<double> input_means_;
    std::vector<double> input_stds_;
    bool has_scaling_ = false;
};

/// Apply activation function in-place
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


