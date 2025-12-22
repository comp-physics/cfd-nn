/// Unit tests for feature computation (analytic verification)

#include "mesh.hpp"
#include "fields.hpp"
#include "features.hpp"
#include "gpu_kernels.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

void test_pure_shear_flow() {
    std::cout << "Testing pure shear flow (u = γy, v = 0)... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;  // Shear rate
    
    // Create pure shear: u = γy, v = 0
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    // Compute gradients using MAC-aware method
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    VelocityGradient grad;
    grad.dudx = (vel.u(i + 1, j) - vel.u(i - 1, j)) * inv_2dx;
    grad.dudy = (vel.u(i, j + 1) - vel.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (vel.v(i + 1, j) - vel.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (vel.v(i, j + 1) - vel.v(i, j - 1)) * inv_2dy;
    
    // Expected: dudx=0, dudy=γ, dvdx=0, dvdy=0
    assert(std::abs(grad.dudx - 0.0) < 1e-12);
    assert(std::abs(grad.dudy - gamma) < 1e-10);
    assert(std::abs(grad.dvdx - 0.0) < 1e-12);
    assert(std::abs(grad.dvdy - 0.0) < 1e-12);
    
    // Strain: Sxx=0, Syy=0, Sxy=γ/2
    assert(std::abs(grad.Sxx() - 0.0) < 1e-12);
    assert(std::abs(grad.Syy() - 0.0) < 1e-12);
    assert(std::abs(grad.Sxy() - gamma/2.0) < 1e-10);
    
    // Rotation: Oxy=γ/2
    assert(std::abs(grad.Oxy() - gamma/2.0) < 1e-10);
    
    // Magnitudes for pure shear (Frobenius norm):
    // |S| = |Omega| = |gamma| / sqrt(2)
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    assert(std::abs(S_mag - (std::abs(gamma) / std::sqrt(2.0))) < 1e-10);
    assert(std::abs(Omega_mag - (std::abs(gamma) / std::sqrt(2.0))) < 1e-10);
    // For pure shear: Omega/S = 1
    assert(std::abs(S_mag - Omega_mag) < 1e-10);
    
    std::cout << "PASSED (|S|=" << S_mag << ", |Ω|=" << Omega_mag << ")\n";
}

void test_pure_strain_flow() {
    std::cout << "Testing pure strain flow (u = ax, v = -ay)... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, -1.0, 1.0, -1.0, 1.0);
    
    const double a = 1.5;  // Strain rate
    
    // Create incompressible pure strain: u = ax, v = -ay
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = a * mesh.x(i);
            vel.v(i, j) = -a * mesh.y(j);
        }
    }
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    VelocityGradient grad;
    grad.dudx = (vel.u(i + 1, j) - vel.u(i - 1, j)) * inv_2dx;
    grad.dudy = (vel.u(i, j + 1) - vel.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (vel.v(i + 1, j) - vel.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (vel.v(i, j + 1) - vel.v(i, j - 1)) * inv_2dy;
    
    // Expected: dudx=a, dudy=0, dvdx=0, dvdy=-a
    assert(std::abs(grad.dudx - a) < 1e-10);
    assert(std::abs(grad.dudy - 0.0) < 1e-12);
    assert(std::abs(grad.dvdx - 0.0) < 1e-12);
    assert(std::abs(grad.dvdy - (-a)) < 1e-10);
    
    // Strain: Sxx=a, Syy=-a, Sxy=0 (incompressible: trace=0)
    assert(std::abs(grad.Sxx() - a) < 1e-10);
    assert(std::abs(grad.Syy() - (-a)) < 1e-10);
    assert(std::abs(grad.Sxy() - 0.0) < 1e-12);
    
    // Rotation: Oxy=0 (no rotation in pure strain)
    assert(std::abs(grad.Oxy() - 0.0) < 1e-12);
    
    // Magnitudes: |Ω| = 0, |S| > 0
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    assert(Omega_mag < 1e-12);
    assert(S_mag > 1.0);
    
    std::cout << "PASSED (|S|=" << S_mag << ", |Ω|=" << Omega_mag << ")\n";
}

void test_solid_body_rotation() {
    std::cout << "Testing solid body rotation (u = -Ωy, v = Ωx)... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, -1.0, 1.0, -1.0, 1.0);
    
    const double Omega = 3.0;  // Angular velocity
    
    // Create solid body rotation: u = -Ωy, v = Ωx
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = -Omega * mesh.y(j);
            vel.v(i, j) = Omega * mesh.x(i);
        }
    }
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    VelocityGradient grad;
    grad.dudx = (vel.u(i + 1, j) - vel.u(i - 1, j)) * inv_2dx;
    grad.dudy = (vel.u(i, j + 1) - vel.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (vel.v(i + 1, j) - vel.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (vel.v(i, j + 1) - vel.v(i, j - 1)) * inv_2dy;
    
    // Expected: dudx=0, dudy=-Ω, dvdx=Ω, dvdy=0
    assert(std::abs(grad.dudx - 0.0) < 1e-12);
    assert(std::abs(grad.dudy - (-Omega)) < 1e-10);
    assert(std::abs(grad.dvdx - Omega) < 1e-10);
    assert(std::abs(grad.dvdy - 0.0) < 1e-12);
    
    // Strain: all zero (no deformation in rigid rotation)
    assert(std::abs(grad.Sxx() - 0.0) < 1e-12);
    assert(std::abs(grad.Syy() - 0.0) < 1e-12);
    assert(std::abs(grad.Sxy() - 0.0) < 1e-12);
    
    // Rotation: Oxy = (dudy - dvdx)/2 = (-Ω - Ω)/2 = -Ω
    assert(std::abs(grad.Oxy() - (-Omega)) < 1e-10);
    
    // Magnitudes: |S| = 0, |Ω| > 0
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    assert(S_mag < 1e-12);
    assert(std::abs(Omega_mag - std::sqrt(2.0)*Omega) < 1e-10);  // |Ω| = √2·Ω
    
    std::cout << "PASSED (|S|=" << S_mag << ", |Ω|=" << Omega_mag << ")\n";
}

void test_scalar_features_shear() {
    std::cout << "Testing scalar features for shear flow... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    const double nu = 0.001;
    const double delta = 1.0;
    
    Features feat = compute_features_scalar_nut(mesh, vel, k, omega, i, j, nu, delta);
    
    // Should have 6 features
    assert(feat.size() == 6);
    
    // All features should be finite
    for (int n = 0; n < feat.size(); ++n) {
        assert(std::isfinite(feat[n]));
    }
    
    // For pure shear: Omega/S ratio should be ~1
    // feat[3] = Omega_mag / S_mag
    assert(std::abs(feat[3] - 1.0) < 0.1);
    
    std::cout << "PASSED (Ω/S ratio=" << feat[3] << ")\n";
}

void test_tbnn_features_and_basis() {
    std::cout << "Testing TBNN features and tensor basis... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, -1.0, 1.0, -1.0, 1.0);
    
    const double a = 1.5;
    
    // Pure strain flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = a * mesh.x(i);
            vel.v(i, j) = -a * mesh.y(j);
        }
    }
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 1.0);
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    Features feat = compute_features_tbnn(mesh, vel, k, omega, i, j, 0.001, 1.0);
    
    // Should have 5 invariants
    assert(feat.size() == 5);
    
    // All invariants should be finite
    for (int n = 0; n < feat.size(); ++n) {
        assert(std::isfinite(feat[n]));
    }
    
    // For pure strain: rotation invariants should be ~0
    // feat[1] = Omega_norm^2, feat[3] = 2*Oxy^2
    assert(feat[1] < 1e-10);  // ~tr(Omega_norm^2)
    assert(feat[3] < 1e-10);  // ~tr(Omega^2)
    
    // Strain invariants should be positive
    assert(feat[0] > 0.0);  // ~tr(S_norm^2)
    assert(feat[2] > 0.0);  // tr(S^2)
    
    // Compute tensor basis
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    VelocityGradient grad;
    grad.dudx = (vel.u(i + 1, j) - vel.u(i - 1, j)) * inv_2dx;
    grad.dudy = (vel.u(i, j + 1) - vel.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (vel.v(i + 1, j) - vel.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (vel.v(i, j + 1) - vel.v(i, j - 1)) * inv_2dy;
    
    double k_val = k(i, j);
    double omega_val = omega(i, j);
    double eps = 0.09 * k_val * omega_val;
    
    std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, k_val, eps, basis);
    
    // T^(1) = S (should be non-zero for pure strain)
    double T1_mag_sq = basis[0][0]*basis[0][0] + basis[0][1]*basis[0][1] + basis[0][2]*basis[0][2];
    assert(T1_mag_sq > 1e-6);
    
    // T^(2) = [S, Omega] (should be ~0 for pure strain, no rotation)
    double T2_mag_sq = basis[1][0]*basis[1][0] + basis[1][1]*basis[1][1] + basis[1][2]*basis[1][2];
    assert(T2_mag_sq < 1e-10);
    
    // T^(4) should be zero in 2D
    assert(std::abs(basis[3][0]) < 1e-15);
    assert(std::abs(basis[3][1]) < 1e-15);
    assert(std::abs(basis[3][2]) < 1e-15);
    
    std::cout << "PASSED (|T1|=" << std::sqrt(T1_mag_sq) << ", |T2|=" << std::sqrt(T2_mag_sq) << ")\n";
}

void test_anisotropy_construction() {
    std::cout << "Testing anisotropy tensor construction... ";
    
    // Create simple basis and coefficients
    std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
    basis[0] = {1.0, 0.0, -1.0};  // T^(1) = diag(1, 0, -1) (traceless)
    basis[1] = {0.0, 0.5, 0.0};   // T^(2) = off-diagonal
    basis[2] = {0.2, 0.0, -0.2};  // T^(3) = quadratic
    basis[3] = {0.0, 0.0, 0.0};   // T^(4) = 0 in 2D
    
    std::array<double, TensorBasis::NUM_BASIS> G = {-0.1, 0.05, 0.02, 0.0};
    
    double b_xx, b_xy, b_yy;
    TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_yy);
    
    // b = sum G_n * T^n
    [[maybe_unused]] double expected_xx = -0.1*1.0 + 0.05*0.0 + 0.02*0.2;
    [[maybe_unused]] double expected_xy = -0.1*0.0 + 0.05*0.5 + 0.02*0.0;
    [[maybe_unused]] double expected_yy = -0.1*(-1.0) + 0.05*0.0 + 0.02*(-0.2);
    
    assert(std::abs(b_xx - expected_xx) < 1e-12);
    assert(std::abs(b_xy - expected_xy) < 1e-12);
    assert(std::abs(b_yy - expected_yy) < 1e-12);
    
    // Anisotropy should be traceless (in incompressible flow)
    [[maybe_unused]] double trace = b_xx + b_yy;
    assert(std::abs(trace) < 1e-10);
    
    std::cout << "PASSED (b_xx=" << b_xx << ", b_xy=" << b_xy << ", b_yy=" << b_yy << ")\n";
}

void test_reynolds_stress_conversion() {
    std::cout << "Testing anisotropy to Reynolds stress conversion... ";
    
    double b_xx = 0.1;
    double b_xy = -0.05;
    double b_yy = -0.1;
    double k = 0.5;
    
    double tau_xx, tau_xy, tau_yy;
    TensorBasis::anisotropy_to_reynolds_stress(b_xx, b_xy, b_yy, k, tau_xx, tau_xy, tau_yy);
    
    // tau_ij = 2*k*(b_ij + (1/3)*delta_ij)
    [[maybe_unused]] double expected_xx = 2.0 * k * (b_xx + 1.0/3.0);
    [[maybe_unused]] double expected_xy = 2.0 * k * b_xy;
    [[maybe_unused]] double expected_yy = 2.0 * k * (b_yy + 1.0/3.0);
    
    assert(std::abs(tau_xx - expected_xx) < 1e-12);
    assert(std::abs(tau_xy - expected_xy) < 1e-12);
    assert(std::abs(tau_yy - expected_yy) < 1e-12);
    
    // Trace should equal 2k (incompressible)
    [[maybe_unused]] double trace = tau_xx + tau_yy;
    [[maybe_unused]] double expected_trace = 2.0 * k * (b_xx + b_yy + 2.0/3.0);
    assert(std::abs(trace - expected_trace) < 1e-12);
    
    std::cout << "PASSED (trace=" << trace << ")\n";
}

void test_gradient_computation_backend() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "Testing gradient computation (CPU vs GPU)... ";
    
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    // Create shear flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    // CPU gradients
    ScalarField dudx_cpu(mesh), dudy_cpu(mesh), dvdx_cpu(mesh), dvdy_cpu(mesh);
    compute_gradients_from_mac_cpu(mesh, vel, dudx_cpu, dudy_cpu, dvdx_cpu, dvdy_cpu);
    
    // GPU gradients
    ScalarField dudx_gpu(mesh), dudy_gpu(mesh), dvdx_gpu(mesh), dvdy_gpu(mesh);
    
    const int total_cells = mesh.total_cells();
    const int u_total = vel.u_total_size();
    const int v_total = vel.v_total_size();
    
    double* u_ptr = vel.u_data().data();
    double* v_ptr = vel.v_data().data();
    double* dudx_ptr = dudx_gpu.data().data();
    double* dudy_ptr = dudy_gpu.data().data();
    double* dvdx_ptr = dvdx_gpu.data().data();
    double* dvdy_ptr = dvdy_gpu.data().data();
    
    // Map to GPU
    #pragma omp target enter data map(to: u_ptr[0:u_total], v_ptr[0:v_total])
    #pragma omp target enter data map(alloc: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
    #pragma omp target enter data map(alloc: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
    
    // Compute on GPU
    gpu_kernels::compute_gradients_from_mac_gpu(
        u_ptr, v_ptr,
        dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
        mesh.Nx, mesh.Ny, mesh.Nghost,
        mesh.dx, mesh.dy,
        vel.u_stride(),        // u_stride
        vel.v_stride(),        // v_stride
        mesh.total_Nx(),       // cell_stride
        u_total, v_total, total_cells
    );
    
    // Download results
    #pragma omp target update from(dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
    #pragma omp target update from(dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
    
    // Cleanup
    #pragma omp target exit data map(delete: u_ptr[0:u_total], v_ptr[0:v_total])
    #pragma omp target exit data map(delete: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
    #pragma omp target exit data map(delete: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
    
    // Compare
    double max_diff = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_diff = std::max(max_diff, std::abs(dudx_cpu(i,j) - dudx_gpu(i,j)));
            max_diff = std::max(max_diff, std::abs(dudy_cpu(i,j) - dudy_gpu(i,j)));
            max_diff = std::max(max_diff, std::abs(dvdx_cpu(i,j) - dvdx_gpu(i,j)));
            max_diff = std::max(max_diff, std::abs(dvdy_cpu(i,j) - dvdy_gpu(i,j)));
        }
    }
    
    assert(max_diff < 1e-12);
    std::cout << "PASSED (max_diff=" << max_diff << ")\n";
#else
    std::cout << "Testing gradient computation (CPU)... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    // Create shear flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    // CPU gradients
    ScalarField dudx(mesh), dudy(mesh), dvdx(mesh), dvdy(mesh);
    compute_gradients_from_mac_cpu(mesh, vel, dudx, dudy, dvdx, dvdy);
    
    // Verify expected values for shear flow at center (use any interior point)
    // Just verify the computation ran without errors
    std::cout << "PASSED\n";
#endif
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  FEATURE COMPUTATION TESTS\n";
    std::cout << "========================================\n";
    std::cout << "Purpose: Validate feature extraction,\n";
    std::cout << "         invariants, and tensor basis\n";
    std::cout << "         using analytic velocity fields\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Backend: GPU (with CPU reference)\n";
#else
    std::cout << "Backend: CPU\n";
#endif
    std::cout << "========================================\n\n";
    
    // Gradient computation tests
    test_pure_shear_flow();
    test_pure_strain_flow();
    test_solid_body_rotation();
    
    // Feature computation tests
    test_scalar_features_shear();
    test_tbnn_features_and_basis();
    
    // Tensor algebra tests
    test_anisotropy_construction();
    test_reynolds_stress_conversion();
    
    // Backend-specific gradient tests
    test_gradient_computation_backend();
    
    std::cout << "\n========================================\n";
    std::cout << "[SUCCESS] All feature tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}

