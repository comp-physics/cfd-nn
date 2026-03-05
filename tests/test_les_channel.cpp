/// @file test_les_channel.cpp
/// @brief LES channel flow integration test
///
/// Runs a short channel flow simulation with each LES SGS model to verify:
///   1. Stable (no blow-up)
///   2. Bulk velocity reasonable (model integrates with solver pipeline)
///   Note: nu_sgs positivity is tested in test_les_sgs.cpp

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_les.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <memory>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

void test_les_model_channel(const std::string& model_name, TurbulenceModelType model_type) {
    const int Nx = 16, Ny = 16;
    const double nu = 0.01;
    const double dp_dx = -1.0;
    const int n_steps = 50;

    Config config;
    config.Nx = Nx;
    config.Ny = Ny;
    config.Nz = 1;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.nu = nu;
    config.dp_dx = dp_dx;
    config.dt = 0.001;
    config.max_steps = n_steps;
    config.turb_model = model_type;
    config.poisson_tol = 1e-6;
    config.poisson_max_vcycles = 20;
    config.verbose = false;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, config.x_min, config.x_max, config.y_min, config.y_max);

    Decomposition decomp(config.Nz);
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-dp_dx, 0.0);

    auto turb_model = create_turbulence_model(model_type);
    CHECK(turb_model != nullptr, (model_name + " factory returned null").c_str());
    turb_model->set_nu(nu);
    solver.set_turbulence_model(std::move(turb_model));

    solver.initialize_uniform(0.1, 0.0);

    double final_residual = 0.0;
    bool stable = true;
    for (int step = 0; step < n_steps; ++step) {
        double res = solver.step();
        if (std::isnan(res) || std::isinf(res)) {
            stable = false;
            break;
        }
        final_residual = res;
    }

    CHECK(stable, (model_name + " channel simulation blew up").c_str());
    CHECK(final_residual < 1.0, (model_name + " residual too large").c_str());

    // Check that bulk velocity is positive (flow in +x direction)
    double U_b = solver.bulk_velocity();
    CHECK(U_b > 0.0, (model_name + " bulk velocity should be positive").c_str());

    std::cout << "PASS: " << model_name << " channel (U_b=" << U_b
              << ", residual=" << final_residual << ")" << std::endl;
}

int main() {
    test_les_model_channel("Smagorinsky", TurbulenceModelType::Smagorinsky);
    test_les_model_channel("WALE", TurbulenceModelType::WALE);
    test_les_model_channel("Vreman", TurbulenceModelType::Vreman);
    test_les_model_channel("Sigma", TurbulenceModelType::Sigma);
    test_les_model_channel("DynamicSmagorinsky", TurbulenceModelType::DynamicSmagorinsky);

    std::cout << "\nAll LES channel integration tests PASSED" << std::endl;
    return 0;
}
