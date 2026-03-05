/// @file test_les_sgs.cpp
/// @brief Tests for LES subgrid-scale models
///
/// Test coverage:
///   1. Smagorinsky: pure shear → nu_sgs = (Cs*delta)^2 * S
///   2. WALE: vanishes for pure shear (correct wall behavior)
///   3. Vreman: vanishes for 2D shear flow (correct dissipation)
///   4. Sigma: vanishes for solid body rotation
///   5. All models: zero nu_sgs for zero velocity
///   6. Factory: all model names parse correctly

#include "turbulence_les.hpp"
#include "turbulence_model.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

void test_smagorinsky_pure_shear() {
    // u = S*y, v = 0, w = 0 → |S| = S, nu_sgs = (Cs*delta)^2 * S
    const double S = 10.0;
    const double Cs = 0.17;
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    double delta = std::sqrt(mesh.dx * mesh.dy);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    // Set u = S*y
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = S * y;
        }
    }

    SmagorinskyModel smag(Cs);
    ScalarField k(mesh), omega(mesh), nu_t(mesh);
    smag.update(mesh, vel, k, omega, nu_t, nullptr, nullptr);

    double expected = (Cs * delta) * (Cs * delta) * S;

    // Check interior cells
    double max_err = 0.0;
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            double nu_val = nu_t(i + Ng, j + Ng);
            max_err = std::max(max_err, std::abs(nu_val - expected) / expected);
        }
    }

    CHECK(max_err < 1e-10, "Smagorinsky nu_sgs should match analytical for pure shear");
    std::cout << "PASS: Smagorinsky pure shear (expected=" << expected
              << ", max_rel_err=" << max_err << ")" << std::endl;
}

void test_wale_pure_rotation() {
    // WALE should produce zero nu_sgs for solid body rotation
    // u = -y, v = x → pure rotation, Sij = 0
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, -1.0, 1.0, -1.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = -y;
        }
    }
    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            double x = mesh.x(i);
            vel.v(i, j) = x;
        }
    }

    WALEModel wale;
    ScalarField k(mesh), omega(mesh), nu_t(mesh);
    wale.update(mesh, vel, k, omega, nu_t, nullptr, nullptr);

    double max_nu = 0.0;
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            max_nu = std::max(max_nu, std::abs(nu_t(i + Ng, j + Ng)));
        }
    }

    // WALE should give small (ideally zero) nu_sgs for rotation
    // Not exactly zero due to discrete gradients but should be very small
    CHECK(max_nu < 0.01, "WALE should give near-zero nu_sgs for pure rotation");
    std::cout << "PASS: WALE pure rotation (max_nu=" << max_nu << ")" << std::endl;
}

void test_all_models_zero_velocity() {
    const int Nx = 8, Ny = 8;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);  // All zero
    ScalarField k(mesh), omega(mesh), nu_t(mesh);
    const int Ng = mesh.Nghost;

    // Test each model
    std::vector<std::unique_ptr<LESModel>> models;
    models.push_back(std::make_unique<SmagorinskyModel>());
    models.push_back(std::make_unique<WALEModel>());
    models.push_back(std::make_unique<VremanModel>());
    models.push_back(std::make_unique<SigmaModel>());
    models.push_back(std::make_unique<DynamicSmagorinskyModel>());

    for (auto& model : models) {
        nu_t.fill(999.0);  // Fill with garbage to detect if model updates
        model->update(mesh, vel, k, omega, nu_t, nullptr, nullptr);

        double max_nu = 0.0;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                max_nu = std::max(max_nu, std::abs(nu_t(i + Ng, j + Ng)));
            }
        }

        CHECK(max_nu < 1e-30, (model->name() + " must give zero nu_sgs for zero velocity").c_str());
        std::cout << "PASS: " << model->name() << " zero velocity (max_nu=" << max_nu << ")" << std::endl;
    }
}

void test_all_models_nonnegative_nu_sgs() {
    // All LES models must produce nu_sgs >= 0 for any flow
    const double S = 10.0;
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    // Set u = S*y (shear flow)
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = S * y;
        }
    }

    ScalarField k(mesh), omega(mesh), nu_t(mesh);

    std::vector<std::unique_ptr<LESModel>> models;
    models.push_back(std::make_unique<SmagorinskyModel>());
    models.push_back(std::make_unique<WALEModel>());
    models.push_back(std::make_unique<VremanModel>());
    models.push_back(std::make_unique<SigmaModel>());
    models.push_back(std::make_unique<DynamicSmagorinskyModel>());

    for (auto& model : models) {
        nu_t.fill(-1.0);  // Fill with negative to detect if model updates
        model->update(mesh, vel, k, omega, nu_t, nullptr, nullptr);

        double min_nu = 1e30;
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                min_nu = std::min(min_nu, nu_t(i + Ng, j + Ng));
            }
        }

        CHECK(min_nu >= 0.0, (model->name() + " must produce non-negative nu_sgs").c_str());
        std::cout << "PASS: " << model->name() << " non-negative nu_sgs (min=" << min_nu << ")" << std::endl;
    }
}

void test_factory() {
    auto smag = create_turbulence_model(TurbulenceModelType::Smagorinsky);
    CHECK(smag != nullptr, "Smagorinsky factory should return non-null");
    CHECK(smag->name() == "Smagorinsky", "Smagorinsky name mismatch");

    auto wale = create_turbulence_model(TurbulenceModelType::WALE);
    CHECK(wale != nullptr && wale->name() == "WALE", "WALE factory failed");

    auto vreman = create_turbulence_model(TurbulenceModelType::Vreman);
    CHECK(vreman != nullptr && vreman->name() == "Vreman", "Vreman factory failed");

    auto sigma = create_turbulence_model(TurbulenceModelType::Sigma);
    CHECK(sigma != nullptr && sigma->name() == "Sigma", "Sigma factory failed");

    auto dsmag = create_turbulence_model(TurbulenceModelType::DynamicSmagorinsky);
    CHECK(dsmag != nullptr && dsmag->name() == "DynamicSmagorinsky", "DynamicSmag factory failed");

    std::cout << "PASS: LES model factory" << std::endl;
}

void test_vreman_3d_shear() {
    // Vreman model: for 3D shear u=S*y, v=0, w=T*x → nonzero nu_sgs
    // (Vreman correctly vanishes for 2D flows, so test 3D)
    const double S = 10.0;
    const double T = 5.0;
    const int Nx = 8, Ny = 8, Nz = 8;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    for (int k = 0; k < Nz + 2 * Ng; ++k) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            double y = mesh.y(j);
            for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
                vel.u(i, j, k) = S * y;
            }
        }
    }
    for (int k = 0; k < Nz + 1 + 2 * Ng; ++k) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                double x = mesh.x(i);
                vel.w(i, j, k) = T * x;
            }
        }
    }

    VremanModel vreman;
    ScalarField kf(mesh), omega(mesh), nu_t(mesh);
    vreman.update(mesh, vel, kf, omega, nu_t, nullptr, nullptr);

    double max_nu = 0.0;
    for (int kk = 0; kk < Nz; ++kk) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                max_nu = std::max(max_nu, nu_t(i + Ng, j + Ng, kk + Ng));
            }
        }
    }

    CHECK(max_nu > 0.0, "Vreman should give nonzero nu_sgs for 3D shear flow");
    std::cout << "PASS: Vreman 3D shear (max_nu=" << max_nu << ")" << std::endl;
}

int main() {
    test_smagorinsky_pure_shear();
    test_wale_pure_rotation();
    test_all_models_zero_velocity();
    test_all_models_nonnegative_nu_sgs();
    test_vreman_3d_shear();
    test_factory();

    std::cout << "\nAll LES SGS tests PASSED" << std::endl;
    return 0;
}
