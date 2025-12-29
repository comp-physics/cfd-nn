/// EARSM Trace-Free Constraint Test
/// Verifies that the anisotropy tensor b_ij computed by EARSM models
/// satisfies the trace-free constraint: b_xx + b_yy = 0 (2D)
///
/// This is a fundamental constraint from incompressibility:
///   b_ij = (u'_i u'_j)/(2k) - (1/3) delta_ij
///   => trace(b_ij) = (u'_i u'_i)/(2k) - 1 = k/(2k) - 1 = 0 (when properly normalized)
///
/// Tests:
/// 1. Tensor basis functions are individually trace-free
/// 2. Anisotropy construction preserves trace-free property
/// 3. EARSM models produce trace-free anisotropy in channel flow

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "features.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_earsm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>

using namespace nncfd;

//=============================================================================
// Helper: Compute max trace error for anisotropy tensor b_ij
// In 2D: tau_ij = 2k * (b_ij + (1/3)*delta_ij)
// trace(tau) = 2k * (trace(b) + 2/3), so for trace(b)=0: trace(tau) = 4k/3
// b_trace = trace(tau)/(2k) - 2/3 should be 0
//=============================================================================
double compute_max_trace_error(const Mesh& mesh, const ScalarField& k,
                                const TensorField& tau_ij) {
    double max_error = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_val = k(i, j);
            if (k_val < 1e-10) continue;

            double tau_trace = tau_ij.trace(i, j);
            double b_trace = tau_trace / (2.0 * k_val) - 2.0/3.0;  // 2D: trace(delta)=2
            max_error = std::max(max_error, std::abs(b_trace));
        }
    }
    return max_error;
}

//=============================================================================
// Test 1: Each tensor basis function should be trace-free
//=============================================================================
bool test_tensor_basis_trace_free() {
    std::cout << "Test 1: Tensor basis trace-free property... ";

    // Test with various velocity gradient configurations
    std::vector<VelocityGradient> test_cases = {
        // Pure shear
        {0.0, 1.0, 0.0, 0.0},
        // Strain + rotation
        {0.5, 0.5, -0.5, -0.5},
        // Asymmetric case
        {0.3, 0.7, -0.2, -0.3},
        // High strain
        {2.0, 0.0, 0.0, -2.0}
    };

    const double tol = 1e-10;
    bool all_passed = true;

    for (const auto& grad : test_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        double k = 0.1, epsilon = 0.01;

        TensorBasis::compute(grad, k, epsilon, basis);

        // Check each basis tensor is trace-free
        for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
            double trace = basis[n][0] + basis[n][2];  // T_xx + T_yy
            if (std::abs(trace) > tol) {
                std::cout << "FAILED\n";
                std::cout << "  Tensor basis T^(" << n+1 << ") has trace = " << trace
                          << " (expected 0)\n";
                all_passed = false;
            }
        }
    }

    if (all_passed) {
        std::cout << "PASSED (all " << TensorBasis::NUM_BASIS << " basis tensors trace-free)\n";
    }

    return all_passed;
}

//=============================================================================
// Test 2: Anisotropy construction preserves trace-free property
//=============================================================================
bool test_anisotropy_construction_trace_free() {
    std::cout << "Test 2: Anisotropy construction trace-free... ";

    const double tol = 1e-10;
    bool all_passed = true;

    // Test with various G coefficients
    std::vector<std::array<double, TensorBasis::NUM_BASIS>> G_cases = {
        {-0.1, 0.0, 0.0, 0.0},    // Only linear term
        {-0.1, 0.05, 0.0, 0.0},   // Linear + commutator
        {-0.1, 0.05, 0.02, 0.0},  // All non-zero
        {-0.3, 0.1, 0.08, 0.0}    // Larger coefficients
    };

    // Test with various velocity gradients
    std::vector<VelocityGradient> grad_cases = {
        {0.0, 1.0, 0.0, 0.0},      // Pure shear
        {0.5, 0.5, -0.5, -0.5},    // Strain + rotation
        {1.0, 0.5, -0.3, -1.0}     // Mixed case
    };

    for (const auto& grad : grad_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        double k = 0.1, epsilon = 0.01;

        TensorBasis::compute(grad, k, epsilon, basis);

        for (const auto& G : G_cases) {
            double b_xx, b_xy, b_yy;
            TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_yy);

            double trace = b_xx + b_yy;
            if (std::abs(trace) > tol) {
                std::cout << "FAILED\n";
                std::cout << "  Anisotropy trace = " << trace << " (expected 0)\n";
                std::cout << "  b_xx=" << b_xx << ", b_yy=" << b_yy << "\n";
                all_passed = false;
            }
        }
    }

    if (all_passed) {
        std::cout << "PASSED (trace = 0 for all test cases)\n";
    }

    return all_passed;
}

//=============================================================================
// Test 3: EARSM closures with varying flow conditions
//=============================================================================
bool test_earsm_varying_conditions() {
    std::cout << "Test 3: EARSM closures under varying flow conditions... ";

    const double tol = 1e-10;
    bool all_passed = true;

    // Create mesh with varying wall distances
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);

    // Test with different velocity profiles
    std::vector<std::string> profile_names = {"linear", "parabolic", "shear"};

    for (const auto& profile_name : profile_names) {
        VectorField vel(mesh);
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.y(j);
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                if (profile_name == "linear") {
                    vel.u(i, j) = y;
                    vel.v(i, j) = 0.0;
                } else if (profile_name == "parabolic") {
                    vel.u(i, j) = 1.0 - y * y;
                    vel.v(i, j) = 0.0;
                } else {  // shear
                    vel.u(i, j) = 0.5 * (y + 1.0);
                    vel.v(i, j) = 0.0;
                }
            }
        }

        ScalarField k(mesh, 0.1);
        ScalarField omega(mesh, 10.0);
        ScalarField nu_t(mesh, 0.0);
        TensorField tau_ij(mesh);

        // Test each closure type
        std::vector<EARSMType> types = {
            EARSMType::WallinJohansson2000,
            EARSMType::GatskiSpeziale1993,
            EARSMType::Pope1975
        };

        for (auto type : types) {
            SSTWithEARSM model(type);
            model.set_nu(0.001);
            model.set_delta(1.0);
            model.initialize(mesh, vel);

            model.update(mesh, vel, k, omega, nu_t, &tau_ij);

            double max_trace_error = compute_max_trace_error(mesh, k, tau_ij);
            if (max_trace_error > tol) {
                std::cout << "\n  Profile=" << profile_name
                          << " has max b_trace=" << max_trace_error;
                all_passed = false;
            }
        }
    }

    if (all_passed) {
        std::cout << "PASSED (trace-free for all profiles and closures)\n";
    } else {
        std::cout << "\n  FAILED\n";
    }

    return all_passed;
}

//=============================================================================
// Test 4: Direct EARSM closure test (bypass solver)
//=============================================================================
bool test_earsm_direct_trace_free() {
    std::cout << "Test 4: Direct EARSM closure trace-free... ";

    const double tol = 1e-10;
    bool all_passed = true;

    // Create simple shear flow conditions
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = mesh.y(j);  // Linear shear
            vel.v(i, j) = 0.0;
        }
    }

    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh, 0.0);
    TensorField tau_ij(mesh);

    // Test each EARSM closure type
    std::vector<EARSMType> types = {
        EARSMType::WallinJohansson2000,
        EARSMType::GatskiSpeziale1993,
        EARSMType::Pope1975
    };

    std::vector<std::string> type_names = {
        "WallinJohansson2000",
        "GatskiSpeziale1993",
        "Pope1975"
    };

    for (size_t t = 0; t < types.size(); ++t) {
        SSTWithEARSM model(types[t]);
        model.set_nu(0.001);
        model.set_delta(1.0);
        model.initialize(mesh, vel);

        // Compute anisotropy via update with tau_ij output
        model.update(mesh, vel, k, omega, nu_t, &tau_ij);

        double max_trace_error = compute_max_trace_error(mesh, k, tau_ij);
        if (max_trace_error > tol) {
            std::cout << "\n  " << type_names[t] << ": max b_trace = "
                      << std::scientific << max_trace_error;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "PASSED (all closures produce trace-free b_ij)\n";
    } else {
        std::cout << "\n  FAILED\n";
    }

    return all_passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  EARSM TRACE-FREE CONSTRAINT TEST\n";
    std::cout << "================================================================\n";
    std::cout << "Verifies anisotropy tensor b_ij satisfies: b_xx + b_yy = 0\n";
    std::cout << "This is required by incompressibility constraint\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_tensor_basis_trace_free()) passed++;
    total++; if (test_anisotropy_construction_trace_free()) passed++;
    total++; if (test_earsm_varying_conditions()) passed++;
    total++; if (test_earsm_direct_trace_free()) passed++;

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "Passed: " << passed << "/" << total << " tests\n\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All trace-free constraint tests passed!\n";
        std::cout << "================================================================\n\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        std::cout << "================================================================\n\n";
        return 1;
    }
}
