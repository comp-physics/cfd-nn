/// @file test_operator_convergence.cpp
/// @brief Convergence tests for O2 and O4 spatial derivative operators
///
/// Tests:
/// 1. Derivative accuracy on smooth periodic functions (sin/cos)
/// 2. Convergence rate verification (O2 → 2, O4 → 4)
/// 3. Adjoint identity: <f, Dfc(g)> = -<Dcf(f), g> for periodic BCs

#include "test_harness.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace nncfd::test::harness;

// Constants
constexpr double PI = 3.14159265358979323846;

//=============================================================================
// O2 Derivative Operators (reference implementations)
//=============================================================================

/// O2 center→face derivative: D^cf = (p[i+1] - p[i]) / h
/// Maps from cell-centers to x-faces
inline double Dcf_O2(double p_i, double p_ip1, double h) {
    return (p_ip1 - p_i) / h;
}

/// O2 face→center derivative: D^fc = (u[i+1] - u[i]) / h
/// Maps from x-faces to cell-centers
inline double Dfc_O2(double u_i, double u_ip1, double h) {
    return (u_ip1 - u_i) / h;
}

/// O2 same-stagger derivative: D_same = (f[i+1] - f[i-1]) / (2h)
inline double D_same_O2(double f_im1, double f_ip1, double h) {
    return (f_ip1 - f_im1) / (2.0 * h);
}

//=============================================================================
// O4 Derivative Operators (4th-order accurate)
//=============================================================================

/// O4 center→face derivative: D^cf_O4 = (p[i-1] - 27*p[i] + 27*p[i+1] - p[i+2]) / (24*h)
/// Derivative located at x_{i+1/2} (face between cells i and i+1)
inline double Dcf_O4(double p_im1, double p_i, double p_ip1, double p_ip2, double h) {
    return (p_im1 - 27.0*p_i + 27.0*p_ip1 - p_ip2) / (24.0 * h);
}

/// O4 face→center derivative: D^fc_O4 = (u[i-1] - 27*u[i] + 27*u[i+1] - u[i+2]) / (24*h)
/// Derivative located at x_i (center of cell i, using faces i-1/2 through i+3/2)
inline double Dfc_O4(double u_im1, double u_i, double u_ip1, double u_ip2, double h) {
    return (u_im1 - 27.0*u_i + 27.0*u_ip1 - u_ip2) / (24.0 * h);
}

/// O4 same-stagger derivative: D_same_O4 = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
inline double D_same_O4(double f_im2, double f_im1, double f_ip1, double f_ip2, double h) {
    return (-f_ip2 + 8.0*f_ip1 - 8.0*f_im1 + f_im2) / (12.0 * h);
}

//=============================================================================
// Test Functions
//=============================================================================

/// Smooth periodic test function: f(x) = sin(2*pi*k*x/L) for x in [0, L]
struct SinFunction {
    double L;      // Domain length
    int k;         // Wave number

    double value(double x) const {
        return std::sin(2.0 * PI * k * x / L);
    }

    double deriv(double x) const {
        return (2.0 * PI * k / L) * std::cos(2.0 * PI * k * x / L);
    }
};

/// Cosine test function: f(x) = cos(2*pi*k*x/L)
struct CosFunction {
    double L;
    int k;

    double value(double x) const {
        return std::cos(2.0 * PI * k * x / L);
    }

    double deriv(double x) const {
        return -(2.0 * PI * k / L) * std::sin(2.0 * PI * k * x / L);
    }
};

//=============================================================================
// Same-Stagger Derivative Convergence Test
//=============================================================================

/// Test same-stagger derivative (e.g., du/dx at u-faces) convergence
/// Uses periodic domain [0, L] with N grid points
static void test_same_stagger_convergence() {
    std::cout << "Testing same-stagger derivative D_same convergence...\n\n";

    const double L = 2.0 * PI;  // Domain length
    SinFunction f{L, 1};        // sin(x) on [0, 2*pi]

    std::vector<int> Ns = {16, 32, 64, 128, 256};
    std::vector<double> err_O2, err_O4;

    for (int N : Ns) {
        double h = L / N;

        // Allocate periodic array with ghost cells
        int Ng = 2;  // 2 ghost cells for O4
        std::vector<double> phi(N + 2*Ng);

        // Fill periodic function (cell-centered values)
        for (int i = 0; i < N; ++i) {
            double x = (i + 0.5) * h;  // Cell center
            phi[Ng + i] = f.value(x);
        }

        // Fill ghost cells (periodic wrap)
        phi[0] = phi[N];
        phi[1] = phi[N + 1];
        phi[N + Ng] = phi[Ng];
        phi[N + Ng + 1] = phi[Ng + 1];

        // Compute errors
        double max_err_O2 = 0.0;
        double max_err_O4 = 0.0;

        for (int i = 0; i < N; ++i) {
            double x = (i + 0.5) * h;
            double exact = f.deriv(x);

            // O2 derivative
            int idx = Ng + i;
            double d_O2 = D_same_O2(phi[idx - 1], phi[idx + 1], h);
            max_err_O2 = std::max(max_err_O2, std::abs(d_O2 - exact));

            // O4 derivative
            double d_O4 = D_same_O4(phi[idx - 2], phi[idx - 1], phi[idx + 1], phi[idx + 2], h);
            max_err_O4 = std::max(max_err_O4, std::abs(d_O4 - exact));
        }

        err_O2.push_back(max_err_O2);
        err_O4.push_back(max_err_O4);

        std::cout << "  N=" << std::setw(4) << N
                  << "  h=" << std::scientific << std::setprecision(4) << h
                  << "  O2 err=" << max_err_O2
                  << "  O4 err=" << max_err_O4 << "\n";
    }

    // Compute convergence rates
    std::cout << "\n  Convergence rates:\n";
    for (size_t i = 1; i < Ns.size(); ++i) {
        double rate_O2 = std::log(err_O2[i-1] / err_O2[i]) / std::log(2.0);
        double rate_O4 = std::log(err_O4[i-1] / err_O4[i]) / std::log(2.0);
        std::cout << "  N=" << std::setw(4) << Ns[i-1] << " -> " << std::setw(4) << Ns[i]
                  << ":  O2 rate=" << std::fixed << std::setprecision(2) << rate_O2
                  << "  O4 rate=" << rate_O4 << "\n";
    }

    // Check final rates (should approach 2 for O2, 4 for O4)
    double final_rate_O2 = std::log(err_O2[err_O2.size()-2] / err_O2.back()) / std::log(2.0);
    double final_rate_O4 = std::log(err_O4[err_O4.size()-2] / err_O4.back()) / std::log(2.0);

    record("D_same O2 convergence rate >= 1.9", final_rate_O2 >= 1.9);
    record("D_same O4 convergence rate >= 3.8", final_rate_O4 >= 3.8);
}

//=============================================================================
// Center-Face / Face-Center Derivative Convergence Test
//=============================================================================

/// Test Dcf (center→face) derivative convergence
static void test_Dcf_convergence() {
    std::cout << "\nTesting center-to-face derivative D^cf convergence...\n\n";

    const double L = 2.0 * PI;
    SinFunction f{L, 1};

    std::vector<int> Ns = {16, 32, 64, 128, 256};
    std::vector<double> err_O2, err_O4;

    for (int N : Ns) {
        double h = L / N;

        // Cell-centered values with ghost cells
        int Ng = 2;
        std::vector<double> phi_c(N + 2*Ng);

        for (int i = 0; i < N; ++i) {
            double x = (i + 0.5) * h;
            phi_c[Ng + i] = f.value(x);
        }

        // Periodic ghost cells
        phi_c[0] = phi_c[N];
        phi_c[1] = phi_c[N + 1];
        phi_c[N + Ng] = phi_c[Ng];
        phi_c[N + Ng + 1] = phi_c[Ng + 1];

        double max_err_O2 = 0.0;
        double max_err_O4 = 0.0;

        // Dcf: derivative at face i+1/2, located at x = (i+1)*h
        // For N cells, there are N faces (periodic)
        for (int i = 0; i < N; ++i) {
            double x_face = (i + 1) * h;  // Face position
            double exact = f.deriv(x_face);

            int idx = Ng + i;  // Cell i

            // O2: (phi[i+1] - phi[i]) / h
            double d_O2 = Dcf_O2(phi_c[idx], phi_c[idx + 1], h);
            max_err_O2 = std::max(max_err_O2, std::abs(d_O2 - exact));

            // O4: (phi[i-1] - 27*phi[i] + 27*phi[i+1] - phi[i+2]) / (24h)
            double d_O4 = Dcf_O4(phi_c[idx - 1], phi_c[idx], phi_c[idx + 1], phi_c[idx + 2], h);
            max_err_O4 = std::max(max_err_O4, std::abs(d_O4 - exact));
        }

        err_O2.push_back(max_err_O2);
        err_O4.push_back(max_err_O4);

        std::cout << "  N=" << std::setw(4) << N
                  << "  h=" << std::scientific << std::setprecision(4) << h
                  << "  O2 err=" << max_err_O2
                  << "  O4 err=" << max_err_O4 << "\n";
    }

    std::cout << "\n  Convergence rates:\n";
    for (size_t i = 1; i < Ns.size(); ++i) {
        double rate_O2 = std::log(err_O2[i-1] / err_O2[i]) / std::log(2.0);
        double rate_O4 = std::log(err_O4[i-1] / err_O4[i]) / std::log(2.0);
        std::cout << "  N=" << std::setw(4) << Ns[i-1] << " -> " << std::setw(4) << Ns[i]
                  << ":  O2 rate=" << std::fixed << std::setprecision(2) << rate_O2
                  << "  O4 rate=" << rate_O4 << "\n";
    }

    double final_rate_O2 = std::log(err_O2[err_O2.size()-2] / err_O2.back()) / std::log(2.0);
    double final_rate_O4 = std::log(err_O4[err_O4.size()-2] / err_O4.back()) / std::log(2.0);

    record("D^cf O2 convergence rate >= 1.9", final_rate_O2 >= 1.9);
    record("D^cf O4 convergence rate >= 3.8", final_rate_O4 >= 3.8);
}

/// Test Dfc (face→center) derivative convergence
static void test_Dfc_convergence() {
    std::cout << "\nTesting face-to-center derivative D^fc convergence...\n\n";

    const double L = 2.0 * PI;
    CosFunction f{L, 1};  // cos(x) so derivative is -sin(x)

    std::vector<int> Ns = {16, 32, 64, 128, 256};
    std::vector<double> err_O2, err_O4;

    for (int N : Ns) {
        double h = L / N;

        // Face-centered values with ghost cells
        // Face i is at x = i*h
        int Ng = 2;
        std::vector<double> phi_f(N + 1 + 2*Ng);  // N+1 faces for N cells, plus ghosts

        for (int i = 0; i <= N; ++i) {
            double x = i * h;
            phi_f[Ng + i] = f.value(x);
        }

        // Periodic ghost cells (face 0 = face N for periodic)
        phi_f[0] = phi_f[N];
        phi_f[1] = phi_f[N + 1];
        phi_f[N + Ng + 1] = phi_f[Ng + 1];
        phi_f[N + Ng + 2] = phi_f[Ng + 2];

        double max_err_O2 = 0.0;
        double max_err_O4 = 0.0;

        // Dfc: derivative at cell center i, located at x = (i+0.5)*h
        for (int i = 0; i < N; ++i) {
            double x_cell = (i + 0.5) * h;
            double exact = f.deriv(x_cell);

            int idx = Ng + i;  // Face i

            // O2: (phi_f[i+1] - phi_f[i]) / h
            double d_O2 = Dfc_O2(phi_f[idx], phi_f[idx + 1], h);
            max_err_O2 = std::max(max_err_O2, std::abs(d_O2 - exact));

            // O4: (phi_f[i-1] - 27*phi_f[i] + 27*phi_f[i+1] - phi_f[i+2]) / (24h)
            double d_O4 = Dfc_O4(phi_f[idx - 1], phi_f[idx], phi_f[idx + 1], phi_f[idx + 2], h);
            max_err_O4 = std::max(max_err_O4, std::abs(d_O4 - exact));
        }

        err_O2.push_back(max_err_O2);
        err_O4.push_back(max_err_O4);

        std::cout << "  N=" << std::setw(4) << N
                  << "  h=" << std::scientific << std::setprecision(4) << h
                  << "  O2 err=" << max_err_O2
                  << "  O4 err=" << max_err_O4 << "\n";
    }

    std::cout << "\n  Convergence rates:\n";
    for (size_t i = 1; i < Ns.size(); ++i) {
        double rate_O2 = std::log(err_O2[i-1] / err_O2[i]) / std::log(2.0);
        double rate_O4 = std::log(err_O4[i-1] / err_O4[i]) / std::log(2.0);
        std::cout << "  N=" << std::setw(4) << Ns[i-1] << " -> " << std::setw(4) << Ns[i]
                  << ":  O2 rate=" << std::fixed << std::setprecision(2) << rate_O2
                  << "  O4 rate=" << rate_O4 << "\n";
    }

    double final_rate_O2 = std::log(err_O2[err_O2.size()-2] / err_O2.back()) / std::log(2.0);
    double final_rate_O4 = std::log(err_O4[err_O4.size()-2] / err_O4.back()) / std::log(2.0);

    record("D^fc O2 convergence rate >= 1.9", final_rate_O2 >= 1.9);
    record("D^fc O4 convergence rate >= 3.8", final_rate_O4 >= 3.8);
}

//=============================================================================
// Adjoint Identity Test
//=============================================================================

/// Test that Dcf and Dfc are negative adjoints: <f, Dfc(g)> = -<Dcf(f), g>
/// This is the discrete analog of integration by parts for periodic BCs
static void test_adjoint_identity() {
    std::cout << "\nTesting adjoint identity: <f, D^fc(g)> = -<D^cf(f), g>...\n\n";

    const double L = 2.0 * PI;
    const int N = 64;
    const double h = L / N;
    const int Ng = 2;

    // Use non-orthogonal functions: sin(x) + 0.5 and cos(x) + 0.3
    // These produce non-zero inner products suitable for relative error testing
    auto phi_func = [&](double x) { return std::sin(x) + 0.5; };
    auto psi_func = [&](double x) { return std::cos(x) + 0.3; };

    // Create cell-centered array
    std::vector<double> phi_c(N + 2*Ng);
    for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        phi_c[Ng + i] = phi_func(x);
    }
    phi_c[0] = phi_c[N]; phi_c[1] = phi_c[N + 1];
    phi_c[N + Ng] = phi_c[Ng]; phi_c[N + Ng + 1] = phi_c[Ng + 1];

    // Create face-centered array
    std::vector<double> psi_f(N + 1 + 2*Ng);
    for (int i = 0; i <= N; ++i) {
        double x = i * h;
        psi_f[Ng + i] = psi_func(x);
    }
    psi_f[0] = psi_f[N]; psi_f[1] = psi_f[N + 1];
    psi_f[N + Ng + 1] = psi_f[Ng + 1]; psi_f[N + Ng + 2] = psi_f[Ng + 2];

    // Test O2 adjoint identity
    // LHS: sum_cells phi_c[i] * Dfc(psi_f)[i] * h
    // RHS: -sum_faces Dcf(phi_c)[i] * psi_f[i] * h
    double lhs_O2 = 0.0, rhs_O2 = 0.0;

    for (int i = 0; i < N; ++i) {
        int idx_c = Ng + i;
        int idx_f = Ng + i;

        // LHS: phi at cell i times Dfc(psi) at cell i
        double Dfc_psi = Dfc_O2(psi_f[idx_f], psi_f[idx_f + 1], h);
        lhs_O2 += phi_c[idx_c] * Dfc_psi * h;

        // RHS: Dcf(phi) at face i times psi at face i
        double Dcf_phi = Dcf_O2(phi_c[idx_c], phi_c[idx_c + 1], h);
        rhs_O2 -= Dcf_phi * psi_f[idx_f + 1] * h;
    }

    double err_O2 = std::abs(lhs_O2 - rhs_O2) / (std::abs(lhs_O2) + 1e-15);
    std::cout << "  O2: <phi, D^fc(psi)> = " << std::scientific << lhs_O2
              << "  -<D^cf(phi), psi> = " << rhs_O2
              << "  rel_err = " << err_O2 << "\n";

    // Test O4 adjoint identity
    double lhs_O4 = 0.0, rhs_O4 = 0.0;

    for (int i = 0; i < N; ++i) {
        int idx_c = Ng + i;
        int idx_f = Ng + i;

        // LHS: phi at cell i times Dfc_O4(psi) at cell i
        double Dfc_psi = Dfc_O4(psi_f[idx_f - 1], psi_f[idx_f], psi_f[idx_f + 1], psi_f[idx_f + 2], h);
        lhs_O4 += phi_c[idx_c] * Dfc_psi * h;

        // RHS: Dcf_O4(phi) at face i times psi at face i
        double Dcf_phi = Dcf_O4(phi_c[idx_c - 1], phi_c[idx_c], phi_c[idx_c + 1], phi_c[idx_c + 2], h);
        rhs_O4 -= Dcf_phi * psi_f[idx_f + 1] * h;
    }

    double err_O4 = std::abs(lhs_O4 - rhs_O4) / (std::abs(lhs_O4) + 1e-15);
    std::cout << "  O4: <phi, D^fc(psi)> = " << std::scientific << lhs_O4
              << "  -<D^cf(phi), psi> = " << rhs_O4
              << "  rel_err = " << err_O4 << "\n";

    // The adjoint identity should be satisfied to machine precision for O2
    // For O4, there may be small truncation differences but should still be very good
    record("O2 adjoint identity rel_err < 1e-12", err_O2 < 1e-12);
    record("O4 adjoint identity rel_err < 1e-10", err_O4 < 1e-10);
}

//=============================================================================
// Divergence-Gradient Adjoint Test
//=============================================================================

/// Test that div and -grad are adjoints: <p, div(u)> = -<grad(p), u>
/// This is crucial for pressure projection energy conservation
static void test_div_grad_adjoint() {
    std::cout << "\nTesting div-grad adjoint: <p, div(u)> = -<grad(p), u>...\n\n";

    const double L = 2.0 * PI;
    const int N = 64;
    const double h = L / N;
    const int Ng = 2;

    // Create smooth periodic pressure field (cell-centered)
    std::vector<double> p(N + 2*Ng);
    for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        p[Ng + i] = std::sin(x) + 0.5 * std::cos(2*x);
    }
    p[0] = p[N]; p[1] = p[N + 1];
    p[N + Ng] = p[Ng]; p[N + Ng + 1] = p[Ng + 1];

    // Create smooth periodic velocity field (face-centered)
    std::vector<double> u(N + 1 + 2*Ng);
    for (int i = 0; i <= N; ++i) {
        double x = i * h;
        u[Ng + i] = std::cos(x) - 0.3 * std::sin(3*x);
    }
    u[0] = u[N]; u[1] = u[N + 1];
    u[N + Ng + 1] = u[Ng + 1]; u[N + Ng + 2] = u[Ng + 2];

    // O2: div(u) = Dfc(u), grad(p) = Dcf(p)
    // LHS: sum_cells p[i] * div(u)[i] * h
    // RHS: -sum_faces grad(p)[i] * u[i] * h
    double lhs_O2 = 0.0, rhs_O2 = 0.0;

    for (int i = 0; i < N; ++i) {
        int idx = Ng + i;

        // div(u) at cell center
        double div_u = Dfc_O2(u[idx], u[idx + 1], h);
        lhs_O2 += p[idx] * div_u * h;

        // -grad(p) * u at face
        double grad_p = Dcf_O2(p[idx], p[idx + 1], h);
        rhs_O2 -= grad_p * u[idx + 1] * h;
    }

    double err_O2 = std::abs(lhs_O2 - rhs_O2) / (std::abs(lhs_O2) + 1e-15);
    std::cout << "  O2: <p, div(u)> = " << std::scientific << lhs_O2
              << "  -<grad(p), u> = " << rhs_O2
              << "  rel_err = " << err_O2 << "\n";

    // O4 version
    double lhs_O4 = 0.0, rhs_O4 = 0.0;

    for (int i = 0; i < N; ++i) {
        int idx = Ng + i;

        double div_u = Dfc_O4(u[idx - 1], u[idx], u[idx + 1], u[idx + 2], h);
        lhs_O4 += p[idx] * div_u * h;

        double grad_p = Dcf_O4(p[idx - 1], p[idx], p[idx + 1], p[idx + 2], h);
        rhs_O4 -= grad_p * u[idx + 1] * h;
    }

    double err_O4 = std::abs(lhs_O4 - rhs_O4) / (std::abs(lhs_O4) + 1e-15);
    std::cout << "  O4: <p, div(u)> = " << std::scientific << lhs_O4
              << "  -<grad(p), u> = " << rhs_O4
              << "  rel_err = " << err_O4 << "\n";

    record("O2 div-grad adjoint rel_err < 1e-12", err_O2 < 1e-12);
    record("O4 div-grad adjoint rel_err < 1e-10", err_O4 < 1e-10);
}

//=============================================================================
// Higher Wave Number Test
//=============================================================================

/// Test convergence with higher wave numbers to stress-test the operators
static void test_high_wavenumber() {
    std::cout << "\nTesting with higher wave number (k=4)...\n\n";

    const double L = 2.0 * PI;
    SinFunction f{L, 4};  // sin(4x) - higher frequency

    std::vector<int> Ns = {32, 64, 128, 256};
    std::vector<double> err_O2, err_O4;

    for (int N : Ns) {
        double h = L / N;
        int Ng = 2;
        std::vector<double> phi(N + 2*Ng);

        for (int i = 0; i < N; ++i) {
            double x = (i + 0.5) * h;
            phi[Ng + i] = f.value(x);
        }
        phi[0] = phi[N]; phi[1] = phi[N + 1];
        phi[N + Ng] = phi[Ng]; phi[N + Ng + 1] = phi[Ng + 1];

        double max_err_O2 = 0.0, max_err_O4 = 0.0;

        for (int i = 0; i < N; ++i) {
            double x = (i + 0.5) * h;
            double exact = f.deriv(x);
            int idx = Ng + i;

            double d_O2 = D_same_O2(phi[idx - 1], phi[idx + 1], h);
            double d_O4 = D_same_O4(phi[idx - 2], phi[idx - 1], phi[idx + 1], phi[idx + 2], h);

            max_err_O2 = std::max(max_err_O2, std::abs(d_O2 - exact));
            max_err_O4 = std::max(max_err_O4, std::abs(d_O4 - exact));
        }

        err_O2.push_back(max_err_O2);
        err_O4.push_back(max_err_O4);

        std::cout << "  N=" << std::setw(4) << N
                  << "  O2 err=" << std::scientific << max_err_O2
                  << "  O4 err=" << max_err_O4 << "\n";
    }

    double final_rate_O2 = std::log(err_O2[err_O2.size()-2] / err_O2.back()) / std::log(2.0);
    double final_rate_O4 = std::log(err_O4[err_O4.size()-2] / err_O4.back()) / std::log(2.0);

    std::cout << "  Final rates: O2=" << std::fixed << std::setprecision(2) << final_rate_O2
              << "  O4=" << final_rate_O4 << "\n";

    record("High-k O2 convergence rate >= 1.9", final_rate_O2 >= 1.9);
    record("High-k O4 convergence rate >= 3.8", final_rate_O4 >= 3.8);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    return run_sections("Operator Convergence Tests", {
        {"D_same Convergence", test_same_stagger_convergence},
        {"D^cf Convergence", test_Dcf_convergence},
        {"D^fc Convergence", test_Dfc_convergence},
        {"Adjoint Identity", test_adjoint_identity},
        {"Div-Grad Adjoint", test_div_grad_adjoint},
        {"High Wave Number", test_high_wavenumber}
    });
}
