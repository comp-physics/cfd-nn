#include "nn_core.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// DenseLayer implementation
// ============================================================================

DenseLayer::DenseLayer(int in, int out)
    : in_dim(in), out_dim(out), W(out * in, 0.0), b(out, 0.0) {}

void DenseLayer::forward(const double* x, double* y) const {
    // y = W * x + b (CPU version)
    // W is stored row-major: W[i * in_dim + j] = W_ij
    for (int i = 0; i < out_dim; ++i) {
        double sum = b[i];
        const double* W_row = &W[i * in_dim];
        for (int j = 0; j < in_dim; ++j) {
            sum += W_row[j] * x[j];
        }
        y[i] = sum;
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& x) const {
    std::vector<double> y(out_dim);
    forward(x.data(), y.data());
    return y;
}

void DenseLayer::load_weights(const std::string& W_file, const std::string& b_file) {
    // Load weights
    std::ifstream wf(W_file);
    if (!wf) {
        throw std::runtime_error("Cannot open weights file: " + W_file);
    }
    
    W.resize(out_dim * in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        if (!(wf >> W[i])) {
            throw std::runtime_error("Error reading weights from: " + W_file);
        }
    }
    
    // Load biases
    std::ifstream bf(b_file);
    if (!bf) {
        throw std::runtime_error("Cannot open bias file: " + b_file);
    }
    
    b.resize(out_dim);
    for (int i = 0; i < out_dim; ++i) {
        if (!(bf >> b[i])) {
            throw std::runtime_error("Error reading biases from: " + b_file);
        }
    }
}

// ============================================================================
// MLP implementation
// ============================================================================

MLP::~MLP() {
    free_gpu();
}

MLP::MLP(MLP&& other) noexcept
    : layers_(std::move(other.layers_))
    , activations_(std::move(other.activations_))
    , input_means_(std::move(other.input_means_))
    , input_stds_(std::move(other.input_stds_))
    , has_scaling_(other.has_scaling_)
    , gpu_ready_(other.gpu_ready_)
    , all_weights_(std::move(other.all_weights_))
    , all_biases_(std::move(other.all_biases_))
    , weight_offsets_(std::move(other.weight_offsets_))
    , bias_offsets_(std::move(other.bias_offsets_))
    , layer_dims_(std::move(other.layer_dims_))
    , activation_types_(std::move(other.activation_types_))
{
    other.gpu_ready_ = false;
}

MLP& MLP::operator=(MLP&& other) noexcept {
    if (this != &other) {
        free_gpu();
        layers_ = std::move(other.layers_);
        activations_ = std::move(other.activations_);
        input_means_ = std::move(other.input_means_);
        input_stds_ = std::move(other.input_stds_);
        has_scaling_ = other.has_scaling_;
        gpu_ready_ = other.gpu_ready_;
        all_weights_ = std::move(other.all_weights_);
        all_biases_ = std::move(other.all_biases_);
        weight_offsets_ = std::move(other.weight_offsets_);
        bias_offsets_ = std::move(other.bias_offsets_);
        layer_dims_ = std::move(other.layer_dims_);
        activation_types_ = std::move(other.activation_types_);
        other.gpu_ready_ = false;
    }
    return *this;
}

MLP::MLP(const std::vector<int>& dims, Activation hidden_act) {
    if (dims.size() < 2) {
        throw std::invalid_argument("MLP requires at least 2 dimensions (input, output)");
    }
    
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        layers_.emplace_back(dims[i], dims[i + 1]);
        // Use hidden activation for all but last layer
        if (i < dims.size() - 2) {
            activations_.push_back(hidden_act);
        } else {
            activations_.push_back(Activation::Linear);
        }
    }
}

std::vector<double> MLP::forward(const std::vector<double>& x) const {
    if (layers_.empty()) {
        return x;
    }
    
    std::vector<double> current = x;
    
    // Apply scaling if available
    if (has_scaling_) {
        for (size_t i = 0; i < current.size() && i < input_means_.size(); ++i) {
            current[i] = (current[i] - input_means_[i]) / input_stds_[i];
        }
    }
    
    // Forward through layers
    for (size_t l = 0; l < layers_.size(); ++l) {
        current = layers_[l].forward(current);
        apply_activation(current.data(), static_cast<int>(current.size()), activations_[l]);
    }
    
    return current;
}

void MLP::forward(const double* x, double* output, 
                  std::vector<double>& buffer1, std::vector<double>& buffer2) const {
    if (layers_.empty()) {
        return;
    }
    
    // Ensure buffers are large enough
    int max_dim = max_layer_dim();
    buffer1.resize(max_dim);
    buffer2.resize(max_dim);
    
    // Copy input to buffer1
    double* current = buffer1.data();
    double* next = buffer2.data();
    
    for (int i = 0; i < layers_[0].in_dim; ++i) {
        current[i] = x[i];
    }
    
    // Apply scaling if available
    if (has_scaling_) {
        for (int i = 0; i < layers_[0].in_dim && i < static_cast<int>(input_means_.size()); ++i) {
            current[i] = (current[i] - input_means_[i]) / input_stds_[i];
        }
    }
    
    // Forward through layers
    for (size_t l = 0; l < layers_.size(); ++l) {
        layers_[l].forward(current, next);
        apply_activation(next, layers_[l].out_dim, activations_[l]);
        std::swap(current, next);
    }
    
    // Copy to output
    for (int i = 0; i < layers_.back().out_dim; ++i) {
        output[i] = current[i];
    }
}

int MLP::max_layer_dim() const {
    int max_dim = 0;
    for (const auto& layer : layers_) {
        max_dim = std::max(max_dim, std::max(layer.in_dim, layer.out_dim));
    }
    return max_dim;
}

size_t MLP::workspace_size(int batch_size) const {
    // Need 2 buffers per sample for ping-pong, each of size max_layer_dim
    return static_cast<size_t>(batch_size) * max_layer_dim() * 2;
}

void MLP::flatten_weights() {
    // Compute total sizes
    size_t total_weights = 0;
    size_t total_biases = 0;
    
    for (const auto& layer : layers_) {
        total_weights += layer.W.size();
        total_biases += layer.b.size();
    }
    
    // Allocate
    all_weights_.resize(total_weights);
    all_biases_.resize(total_biases);
    weight_offsets_.resize(layers_.size());
    bias_offsets_.resize(layers_.size());
    layer_dims_.resize(layers_.size() * 2);
    activation_types_.resize(layers_.size());
    
    // Copy weights and biases contiguously
    size_t w_offset = 0;
    size_t b_offset = 0;
    
    for (size_t l = 0; l < layers_.size(); ++l) {
        const auto& layer = layers_[l];
        
        weight_offsets_[l] = static_cast<int>(w_offset);
        bias_offsets_[l] = static_cast<int>(b_offset);
        layer_dims_[l * 2] = layer.in_dim;
        layer_dims_[l * 2 + 1] = layer.out_dim;
        activation_types_[l] = static_cast<int>(activations_[l]);
        
        std::copy(layer.W.begin(), layer.W.end(), all_weights_.begin() + w_offset);
        std::copy(layer.b.begin(), layer.b.end(), all_biases_.begin() + b_offset);
        
        w_offset += layer.W.size();
        b_offset += layer.b.size();
    }
}

void MLP::upload_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        return;
    }
    if (layers_.empty()) {
        return;
    }
    
    // Flatten weights for contiguous GPU memory
    flatten_weights();
    
    // Get raw pointers
    double* weights_ptr = all_weights_.data();
    double* biases_ptr = all_biases_.data();
    int* w_offsets_ptr = weight_offsets_.data();
    int* b_offsets_ptr = bias_offsets_.data();
    int* dims_ptr = layer_dims_.data();
    int* act_ptr = activation_types_.data();
    
    size_t w_size = all_weights_.size();
    size_t b_size = all_biases_.size();
    size_t n_layers = layers_.size();
    
    // Upload to GPU
    #pragma omp target enter data \
        map(to: weights_ptr[0:w_size]) \
        map(to: biases_ptr[0:b_size]) \
        map(to: w_offsets_ptr[0:n_layers]) \
        map(to: b_offsets_ptr[0:n_layers]) \
        map(to: dims_ptr[0:n_layers*2]) \
        map(to: act_ptr[0:n_layers])
    
    // Upload scaling if available
    if (has_scaling_) {
        double* means_ptr = input_means_.data();
        double* stds_ptr = input_stds_.data();
        size_t scale_size = input_means_.size();
        
        #pragma omp target enter data \
            map(to: means_ptr[0:scale_size]) \
            map(to: stds_ptr[0:scale_size])
    }
    
    gpu_ready_ = true;
#endif
}

void MLP::free_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (!gpu_ready_) return;
    
    double* weights_ptr = all_weights_.data();
    double* biases_ptr = all_biases_.data();
    int* w_offsets_ptr = weight_offsets_.data();
    int* b_offsets_ptr = bias_offsets_.data();
    int* dims_ptr = layer_dims_.data();
    int* act_ptr = activation_types_.data();
    
    size_t w_size = all_weights_.size();
    size_t b_size = all_biases_.size();
    size_t n_layers = layers_.size();
    
    #pragma omp target exit data \
        map(delete: weights_ptr[0:w_size]) \
        map(delete: biases_ptr[0:b_size]) \
        map(delete: w_offsets_ptr[0:n_layers]) \
        map(delete: b_offsets_ptr[0:n_layers]) \
        map(delete: dims_ptr[0:n_layers*2]) \
        map(delete: act_ptr[0:n_layers])
    
    if (has_scaling_) {
        double* means_ptr = input_means_.data();
        double* stds_ptr = input_stds_.data();
        size_t scale_size = input_means_.size();
        
        #pragma omp target exit data \
            map(delete: means_ptr[0:scale_size]) \
            map(delete: stds_ptr[0:scale_size])
    }
    
    gpu_ready_ = false;
#endif
}

void MLP::forward_batch_gpu(double* x_batch, double* y_batch, 
                            int batch_size, double* workspace) const {
#ifdef USE_GPU_OFFLOAD
    if (!gpu_ready_ || layers_.empty()) {
        // Fallback to CPU
        std::vector<double> buf1, buf2;
        int in_dim = input_dim();
        int out_dim = output_dim();
        for (int b = 0; b < batch_size; ++b) {
            forward(x_batch + b * in_dim, y_batch + b * out_dim, buf1, buf2);
        }
        return;
    }
    
    // Get device pointers
    const double* weights_ptr = all_weights_.data();
    const double* biases_ptr = all_biases_.data();
    const int* w_offsets_ptr = weight_offsets_.data();
    const int* b_offsets_ptr = bias_offsets_.data();
    const int* dims_ptr = layer_dims_.data();
    const int* act_ptr = activation_types_.data();
    const double* means_ptr = has_scaling_ ? input_means_.data() : nullptr;
    const double* stds_ptr = has_scaling_ ? input_stds_.data() : nullptr;
    
    const int n_layers = static_cast<int>(layers_.size());
    const int max_dim = max_layer_dim();
    const int in_dim = input_dim();
    const int out_dim_final = output_dim();
    const bool do_scaling = has_scaling_;
    const int scale_size = has_scaling_ ? static_cast<int>(input_means_.size()) : 0;
    
    // Process all samples in parallel on GPU
    // Each sample gets its own workspace slice
    // All pointers (x_batch, y_batch, workspace, weights_ptr, etc.) are already mapped via target enter data
    // The target region will use the device copies automatically
    #pragma omp target teams distribute parallel for
    for (int b = 0; b < batch_size; ++b) {
        // Each thread works on one sample
        // Workspace layout: [sample0_buf1][sample0_buf2][sample1_buf1][sample1_buf2]...
        double* buf1 = workspace + b * max_dim * 2;
        double* buf2 = buf1 + max_dim;
        
        // Copy input to buf1 with scaling
        const double* x = x_batch + b * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            if (do_scaling && i < scale_size) {
                buf1[i] = (x[i] - means_ptr[i]) / stds_ptr[i];
            } else {
                buf1[i] = x[i];
            }
        }
        
        // Ping-pong buffers through layers
        double* current = buf1;
        double* next = buf2;
        
        for (int l = 0; l < n_layers; ++l) {
            int layer_in = dims_ptr[l * 2];
            int layer_out = dims_ptr[l * 2 + 1];
            int w_off = w_offsets_ptr[l];
            int b_off = b_offsets_ptr[l];
            int act_type = act_ptr[l];
            
            // Matrix-vector multiply: next = W * current + b
            for (int i = 0; i < layer_out; ++i) {
                double sum = biases_ptr[b_off + i];
                const double* W_row = weights_ptr + w_off + i * layer_in;
                for (int j = 0; j < layer_in; ++j) {
                    sum += W_row[j] * current[j];
                }
                
                // Apply activation inline
                switch (act_type) {
                    case 0: // Linear
                        next[i] = sum;
                        break;
                    case 1: // ReLU
                        next[i] = sum > 0.0 ? sum : 0.0;
                        break;
                    case 2: // Tanh
                        next[i] = tanh(sum);
                        break;
                    case 3: // Sigmoid
                        next[i] = 1.0 / (1.0 + exp(-sum));
                        break;
                    case 4: // Swish
                        next[i] = sum / (1.0 + exp(-sum));
                        break;
                    case 5: { // GELU
                        double c = 0.044715;
                        double s3 = sum * sum * sum;
                        next[i] = 0.5 * sum * (1.0 + tanh(sqrt(2.0/3.14159265358979323846) * (sum + c * s3)));
                        break;
                    }
                    default:
                        next[i] = sum;
                }
            }
            
            // Swap buffers
            double* tmp = current;
            current = next;
            next = tmp;
        }
        
        // Copy output
        double* y = y_batch + b * out_dim_final;
        for (int i = 0; i < out_dim_final; ++i) {
            y[i] = current[i];
        }
    }
#else
    // CPU fallback
    std::vector<double> buf1, buf2;
    int in_dim = input_dim();
    int out_dim = output_dim();
    for (int b = 0; b < batch_size; ++b) {
        forward(x_batch + b * in_dim, y_batch + b * out_dim, buf1, buf2);
    }
#endif
}

void MLP::load_weights(const std::string& dir) {
    // Auto-detect layers from files: layer0_W.txt, layer0_b.txt, layer1_W.txt, ...
    layers_.clear();
    activations_.clear();
    
    int layer_idx = 0;
    while (true) {
        std::string W_file = dir + "/layer" + std::to_string(layer_idx) + "_W.txt";
        std::string b_file = dir + "/layer" + std::to_string(layer_idx) + "_b.txt";
        
        // Check if files exist
        std::ifstream wf(W_file);
        std::ifstream bf(b_file);
        
        if (!wf || !bf) {
            // No more layers
            break;
        }
        
        // Load the weight matrix to determine dimensions
        int rows, cols;
        std::vector<double> W_data = load_matrix(W_file, rows, cols);
        std::vector<double> b_data = load_vector(b_file);
        
        // Create layer
        DenseLayer layer(cols, rows);  // (in_dim, out_dim)
        layer.W = W_data;
        layer.b = b_data;
        
        layers_.push_back(layer);
        
        // Use Tanh for hidden layers, Linear for output layer
        // (Will be updated if this is the last layer)
        activations_.push_back(Activation::Tanh);
        
        layer_idx++;
    }
    
    // Set last layer activation to Linear
    if (!activations_.empty()) {
        activations_.back() = Activation::Linear;
    }
}

void MLP::load_scaling(const std::string& means_file, const std::string& stds_file) {
    input_means_ = load_vector(means_file);
    input_stds_ = load_vector(stds_file);
    has_scaling_ = true;
    
    // Replace zero stds with 1 to avoid division by zero
    for (auto& s : input_stds_) {
        if (std::abs(s) < 1e-10) {
            s = 1.0;
        }
    }
}

void MLP::scale_input(std::vector<double>& x) const {
    if (!has_scaling_) return;
    
    for (size_t i = 0; i < x.size() && i < input_means_.size(); ++i) {
        x[i] = (x[i] - input_means_[i]) / input_stds_[i];
    }
}

void MLP::add_layer(const DenseLayer& layer, Activation act) {
    layers_.push_back(layer);
    activations_.push_back(act);
}

// ============================================================================
// Standalone functions
// ============================================================================

void apply_activation(double* x, int n, Activation act) {
    for (int i = 0; i < n; ++i) {
        x[i] = activate(x[i], act);
    }
}

std::vector<double> load_vector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<double> result;
    double val;
    while (file >> val) {
        result.push_back(val);
    }
    
    return result;
}

std::vector<double> load_matrix(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    rows = static_cast<int>(data.size());
    cols = rows > 0 ? static_cast<int>(data[0].size()) : 0;
    
    // Flatten to row-major
    std::vector<double> result;
    result.reserve(rows * cols);
    for (const auto& row : data) {
        for (double v : row) {
            result.push_back(v);
        }
    }
    
    return result;
}

} // namespace nncfd
