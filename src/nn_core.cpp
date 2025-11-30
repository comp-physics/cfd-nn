#include "nn_core.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nncfd {

DenseLayer::DenseLayer(int in, int out)
    : in_dim(in), out_dim(out), W(out * in, 0.0), b(out, 0.0) {}

void DenseLayer::forward(const double* x, double* y) const {
    // y = W * x + b
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
    int max_dim = 0;
    for (const auto& layer : layers_) {
        max_dim = std::max(max_dim, std::max(layer.in_dim, layer.out_dim));
    }
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

void MLP::load_weights(const std::string& dir) {
    // Expects files: layer0_W.txt, layer0_b.txt, layer1_W.txt, ...
    for (size_t i = 0; i < layers_.size(); ++i) {
        std::string W_file = dir + "/layer" + std::to_string(i) + "_W.txt";
        std::string b_file = dir + "/layer" + std::to_string(i) + "_b.txt";
        layers_[i].load_weights(W_file, b_file);
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


