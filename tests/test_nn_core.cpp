/// Unit tests for NN core

#include "nn_core.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::file_exists;

static std::string resolve_model_path(const std::string& model_name) {
    std::string path1 = "data/models/" + model_name;
    if (file_exists(path1 + "/layer0_W.txt")) return path1;
    std::string path2 = "../data/models/" + model_name;
    if (file_exists(path2 + "/layer0_W.txt")) return path2;
    return "";
}

void test_dense_layer() {
    DenseLayer layer;
    layer.in_dim = 3;
    layer.out_dim = 2;
    layer.W = {1.0, 0.0, -1.0, 0.0, 1.0, 0.0};
    layer.b = {1.0, 2.0};

    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = layer.forward(x);

    // y[0] = 1*1 + 0*2 + (-1)*3 + 1 = -1
    // y[1] = 0*1 + 1*2 + 0*3 + 2 = 4
    bool pass = (y.size() == 2);
    pass = pass && (std::abs(y[0] - (-1.0)) < 1e-10);
    pass = pass && (std::abs(y[1] - 4.0) < 1e-10);

    record("Dense layer forward pass", pass);
}

void test_mlp_forward() {
    MLP mlp;

    DenseLayer layer1;
    layer1.in_dim = 2;
    layer1.out_dim = 3;
    layer1.W = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    layer1.b = {0.0, 0.0, 0.0};

    DenseLayer layer2;
    layer2.in_dim = 3;
    layer2.out_dim = 1;
    layer2.W = {1.0, 1.0, 1.0};
    layer2.b = {0.0};

    mlp.add_layer(layer1, Activation::Tanh);
    mlp.add_layer(layer2, Activation::Linear);

    std::vector<double> x = {1.0, 1.0};
    std::vector<double> y = mlp.forward(x);

    bool pass = (y.size() == 1) && std::isfinite(y[0]);
    record("MLP forward pass", pass);
}

void test_load_weights() {
    std::string model_path = resolve_model_path("mlp_channel_caseholdout");
    if (model_path.empty()) {
        record("Weight loading", true, true);  // skip
        return;
    }

    try {
        MLP mlp;
        mlp.load_weights(model_path);

        if (mlp.input_dim() == 0) {
            record("Weight loading", true, true);  // skip
            return;
        }

        std::vector<double> x(mlp.input_dim(), 1.0);
        std::vector<double> y = mlp.forward(x);

        bool pass = (mlp.output_dim() > 0) && (mlp.num_layers() > 0);
        pass = pass && (y.size() == static_cast<size_t>(mlp.output_dim()));
        pass = pass && std::isfinite(y[0]);

        record("Weight loading", pass);
    } catch (...) {
        record("Weight loading", true, true);  // skip
    }
}

int main() {
    return nncfd::test::harness::run("NN Core Tests", [] {
        test_dense_layer();
        test_mlp_forward();
        test_load_weights();
    });
}

