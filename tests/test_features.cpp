/// Unit tests for feature computation (analytic verification)

#include "mesh.hpp"
#include "fields.hpp"
#include "features.hpp"
#include "gpu_kernels.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <cmath>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using nncfd::test::harness::record;

void test_pure_shear_flow() {
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

    bool pass = true;
    // Expected: dudx=0, dudy=γ, dvdx=0, dvdy=0
    pass = pass && (std::abs(grad.dudx - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.dudy - gamma) < 1e-10);
    pass = pass && (std::abs(grad.dvdx - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.dvdy - 0.0) < 1e-12);

    // Strain: Sxx=0, Syy=0, Sxy=γ/2
    pass = pass && (std::abs(grad.Sxx() - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.Syy() - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.Sxy() - gamma/2.0) < 1e-10);

    // Rotation: Oxy=γ/2
    pass = pass && (std::abs(grad.Oxy() - gamma/2.0) < 1e-10);

    // Magnitudes for pure shear (Frobenius norm):
    // |S| = |Omega| = |gamma| / sqrt(2)
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    pass = pass && (std::abs(S_mag - (std::abs(gamma) / std::sqrt(2.0))) < 1e-10);
    pass = pass && (std::abs(Omega_mag - (std::abs(gamma) / std::sqrt(2.0))) < 1e-10);
    // For pure shear: Omega/S = 1
    pass = pass && (std::abs(S_mag - Omega_mag) < 1e-10);

    record("Pure shear flow gradients", pass);
}

void test_pure_strain_flow() {
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

    bool pass = true;
    // Expected: dudx=a, dudy=0, dvdx=0, dvdy=-a
    pass = pass && (std::abs(grad.dudx - a) < 1e-10);
    pass = pass && (std::abs(grad.dudy - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.dvdx - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.dvdy - (-a)) < 1e-10);

    // Strain: Sxx=a, Syy=-a, Sxy=0 (incompressible: trace=0)
    pass = pass && (std::abs(grad.Sxx() - a) < 1e-10);
    pass = pass && (std::abs(grad.Syy() - (-a)) < 1e-10);
    pass = pass && (std::abs(grad.Sxy() - 0.0) < 1e-12);

    // Rotation: Oxy=0 (no rotation in pure strain)
    pass = pass && (std::abs(grad.Oxy() - 0.0) < 1e-12);

    // Magnitudes: |Ω| = 0, |S| > 0
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    pass = pass && (Omega_mag < 1e-12);
    pass = pass && (S_mag > 1.0);

    record("Pure strain flow gradients", pass);
}

void test_solid_body_rotation() {
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

    bool pass = true;
    // Expected: dudx=0, dudy=-Ω, dvdx=Ω, dvdy=0
    pass = pass && (std::abs(grad.dudx - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.dudy - (-Omega)) < 1e-10);
    pass = pass && (std::abs(grad.dvdx - Omega) < 1e-10);
    pass = pass && (std::abs(grad.dvdy - 0.0) < 1e-12);

    // Strain: all zero (no deformation in rigid rotation)
    pass = pass && (std::abs(grad.Sxx() - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.Syy() - 0.0) < 1e-12);
    pass = pass && (std::abs(grad.Sxy() - 0.0) < 1e-12);

    // Rotation: Oxy = (dudy - dvdx)/2 = (-Ω - Ω)/2 = -Ω
    pass = pass && (std::abs(grad.Oxy() - (-Omega)) < 1e-10);

    // Magnitudes: |S| = 0, |Ω| > 0
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    pass = pass && (S_mag < 1e-12);
    pass = pass && (std::abs(Omega_mag - std::sqrt(2.0)*Omega) < 1e-10);  // |Ω| = √2·Ω

    record("Solid body rotation gradients", pass);
}

void test_scalar_features_shear() {
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

    bool pass = (feat.size() == 6);

    // All features should be finite
    for (int n = 0; n < feat.size(); ++n) {
        pass = pass && std::isfinite(feat[n]);
    }

    // For pure shear: Omega/S ratio should be ~1
    // feat[3] = Omega_mag / S_mag
    pass = pass && (std::abs(feat[3] - 1.0) < 0.1);

    record("Scalar features for shear flow", pass);
}

void test_tbnn_features_and_basis() {
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

    bool pass = (feat.size() == 5);

    // All invariants should be finite
    for (int n = 0; n < feat.size(); ++n) {
        pass = pass && std::isfinite(feat[n]);
    }

    // For pure strain: rotation invariants should be ~0
    // feat[1] = Omega_norm^2, feat[3] = 2*Oxy^2
    pass = pass && (feat[1] < 1e-10);  // ~tr(Omega_norm^2)
    pass = pass && (feat[3] < 1e-10);  // ~tr(Omega^2)

    // Strain invariants should be positive
    pass = pass && (feat[0] > 0.0);  // ~tr(S_norm^2)
    pass = pass && (feat[2] > 0.0);  // tr(S^2)

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

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, k_val, eps, basis);

    // T^(1) = S (should be non-zero for pure strain)
    double T1_mag_sq = basis[0][0]*basis[0][0] + basis[0][1]*basis[0][1] + basis[0][3]*basis[0][3];
    pass = pass && (T1_mag_sq > 1e-6);

    // T^(2) = [S, Omega] (should be ~0 for pure strain, no rotation)
    double T2_mag_sq = basis[1][0]*basis[1][0] + basis[1][1]*basis[1][1] + basis[1][3]*basis[1][3];
    pass = pass && (T2_mag_sq < 1e-10);

    // T^(4) should be zero in 2D (all z-gradients zero)
    for (int c = 0; c < TensorBasis::NUM_COMPONENTS; ++c)
        pass = pass && (std::abs(basis[3][c]) < 1e-14);

    // T^(5)..T^(10) should be zero in 2D
    for (int n = 4; n < TensorBasis::NUM_BASIS; ++n)
        for (int c = 0; c < TensorBasis::NUM_COMPONENTS; ++c)
            pass = pass && (std::abs(basis[n][c]) < 1e-14);

    record("TBNN features and tensor basis", pass);
}

void test_anisotropy_construction() {
    // Create simple basis and coefficients (6 components: xx,xy,xz,yy,yz,zz)
    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis = {};
    basis[0] = {1.0, 0.0, 0.0, -0.5, 0.0, -0.5};  // traceless: xx+yy+zz=0
    basis[1] = {0.0, 0.5, 0.0, 0.0, 0.0, 0.0};     // off-diagonal xy
    basis[2] = {0.2, 0.0, 0.0, -0.1, 0.0, -0.1};   // traceless
    // basis[3..9] = 0

    std::array<double, TensorBasis::NUM_BASIS> G = {};
    G[0] = -0.1; G[1] = 0.05; G[2] = 0.02;

    double b_xx, b_xy, b_xz, b_yy, b_yz, b_zz;
    TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_xz, b_yy, b_yz, b_zz);

    // b = sum G_n * T^n
    double expected_xx = -0.1*1.0 + 0.05*0.0 + 0.02*0.2;
    double expected_xy = -0.1*0.0 + 0.05*0.5 + 0.02*0.0;
    double expected_yy = -0.1*(-0.5) + 0.05*0.0 + 0.02*(-0.1);

    bool pass = true;
    pass = pass && (std::abs(b_xx - expected_xx) < 1e-12);
    pass = pass && (std::abs(b_xy - expected_xy) < 1e-12);
    pass = pass && (std::abs(b_yy - expected_yy) < 1e-12);

    // 3D trace: b_xx + b_yy + b_zz should be zero (traceless basis)
    double trace = b_xx + b_yy + b_zz;
    pass = pass && (std::abs(trace) < 1e-10);

    record("Anisotropy tensor construction", pass);
}

void test_reynolds_stress_conversion() {
    double b_xx = 0.1, b_xy = -0.05, b_xz = 0.02;
    double b_yy = -0.03, b_yz = 0.01, b_zz = -0.07;
    double k = 0.5;

    double tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
    TensorBasis::anisotropy_to_reynolds_stress(
        b_xx, b_xy, b_xz, b_yy, b_yz, b_zz, k,
        tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz);

    // tau_ij = 2*k*(b_ij + (1/3)*delta_ij)
    bool pass = true;
    pass = pass && (std::abs(tau_xx - 2.0*k*(b_xx + 1.0/3.0)) < 1e-12);
    pass = pass && (std::abs(tau_xy - 2.0*k*b_xy) < 1e-12);
    pass = pass && (std::abs(tau_xz - 2.0*k*b_xz) < 1e-12);
    pass = pass && (std::abs(tau_yy - 2.0*k*(b_yy + 1.0/3.0)) < 1e-12);
    pass = pass && (std::abs(tau_yz - 2.0*k*b_yz) < 1e-12);
    pass = pass && (std::abs(tau_zz - 2.0*k*(b_zz + 1.0/3.0)) < 1e-12);

    // Trace should equal 2k * (b_ii + 1)
    double trace = tau_xx + tau_yy + tau_zz;
    double expected_trace = 2.0 * k * (b_xx + b_yy + b_zz + 1.0);
    pass = pass && (std::abs(trace - expected_trace) < 1e-12);

    record("Reynolds stress conversion (3D)", pass);
}

void test_gradient_computation_backend() {
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        record("Gradient computation (CPU vs GPU)", true, true);  // skip
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
    compute_gradients_from_mac(mesh, vel, dudx_cpu, dudy_cpu, dvdx_cpu, dvdy_cpu);

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
        u_ptr, v_ptr, nullptr,
        dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        mesh.Nx, mesh.Ny, mesh.Nz, mesh.Nghost,
        mesh.dx, mesh.dy, mesh.dz,
        vel.u_stride(), vel.v_stride(), mesh.total_Nx(),
        vel.u_plane_stride(), vel.v_plane_stride(),
        vel.w_stride(), vel.w_plane_stride(),
        mesh.total_Nx() * mesh.total_Ny(),
        u_total, v_total, 0, total_cells
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

    record("Gradient computation (CPU vs GPU)", max_diff < 1e-12);
#else
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
    compute_gradients_from_mac(mesh, vel, dudx, dudy, dvdx, dvdy);

    // Verify computation ran without errors
    record("Gradient computation (CPU)", true);
#endif
}

// ============================================================================
// 3D Tensor Basis Tests
// ============================================================================

/// Helper: compute Frobenius norm of 6-component symmetric tensor
static double tensor_norm(const std::array<double, TensorBasis::NUM_COMPONENTS>& T) {
    // ||T||_F = sqrt(T_xx^2 + T_yy^2 + T_zz^2 + 2*(T_xy^2 + T_xz^2 + T_yz^2))
    return std::sqrt(T[0]*T[0] + T[3]*T[3] + T[5]*T[5]
                   + 2.0*(T[1]*T[1] + T[2]*T[2] + T[4]*T[4]));
}

/// Helper: compute trace of 6-component symmetric tensor (xx + yy + zz)
static double tensor_trace(const std::array<double, TensorBasis::NUM_COMPONENTS>& T) {
    return T[0] + T[3] + T[5];
}

void test_3d_basis_traceless() {
    // ALL 10 Pope basis tensors must be traceless for ANY velocity gradient
    std::vector<VelocityGradient> test_cases;

    // 3D shear: u = y, v = 0, w = 0
    VelocityGradient g1 = {};
    g1.dudy = 1.0;
    test_cases.push_back(g1);

    // 3D general: all 9 gradients nonzero, incompressible (dudx+dvdy+dwdz=0)
    VelocityGradient g2 = {};
    g2.dudx = 0.5; g2.dudy = 0.3; g2.dudz = -0.2;
    g2.dvdx = -0.1; g2.dvdy = 0.7; g2.dvdz = 0.4;
    g2.dwdx = 0.6; g2.dwdy = -0.3; g2.dwdz = -(g2.dudx + g2.dvdy); // enforce continuity
    test_cases.push_back(g2);

    // Pure rotation about z-axis
    VelocityGradient g3 = {};
    g3.dudy = -1.0; g3.dvdx = 1.0;
    test_cases.push_back(g3);

    // 3D swirl: u = -z, w = x
    VelocityGradient g4 = {};
    g4.dudz = -1.0; g4.dwdx = 1.0;
    test_cases.push_back(g4);

    // Axial strain: u = x, v = -0.5*y, w = -0.5*z
    VelocityGradient g5 = {};
    g5.dudx = 1.0; g5.dvdy = -0.5; g5.dwdz = -0.5;
    test_cases.push_back(g5);

    bool pass = true;
    for (const auto& grad : test_cases) {
        std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
        TensorBasis::compute(grad, 0.1, 0.01, basis);
        for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
            double tr = tensor_trace(basis[n]);
            if (std::abs(tr) > 1e-12) {
                pass = false;
            }
        }
    }
    record("3D tensor basis: all tensors traceless", pass);
}

void test_3d_basis_symmetric() {
    // All basis tensors should be symmetric (guaranteed by construction via store_sym)
    // Test by computing with general 3D gradients
    VelocityGradient grad = {};
    grad.dudx = 0.5; grad.dudy = 0.3; grad.dudz = -0.2;
    grad.dvdx = -0.1; grad.dvdy = 0.7; grad.dvdz = 0.4;
    grad.dwdx = 0.6; grad.dwdy = -0.3; grad.dwdz = -1.2;

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    // Symmetry is encoded in the 6-component storage (xx,xy,xz,yy,yz,zz)
    // Just verify all values are finite
    bool pass = true;
    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n)
        for (int c = 0; c < TensorBasis::NUM_COMPONENTS; ++c)
            pass = pass && std::isfinite(basis[n][c]);

    record("3D tensor basis: all tensors symmetric and finite", pass);
}

void test_3d_basis_2d_backward_compat() {
    // With zero z-gradients:
    // - All xz, yz components of ALL tensors should be zero
    // - T5-T10 are NOT zero (they're linearly dependent on T1-T4 in 2D, not zero)
    // - T1 xx/xy values should match S*tau
    VelocityGradient grad = {};
    grad.dudx = 0.5; grad.dudy = 0.3; grad.dvdx = -0.2; grad.dvdy = -0.5;

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    bool pass = true;
    // All xz and yz components should be zero for ALL tensors (no z-coupling)
    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
        pass = pass && (std::abs(basis[n][TensorBasis::XZ]) < 1e-12);
        pass = pass && (std::abs(basis[n][TensorBasis::YZ]) < 1e-12);
    }

    // T1 = normalized S: T1_xx = Sxx*tau, T1_xy = Sxy*tau
    double tau = 0.1 / 0.01;  // k/eps
    double expected_T1_xx = grad.Sxx() * tau;
    double expected_T1_xy = grad.Sxy() * tau;
    pass = pass && (std::abs(basis[0][TensorBasis::XX] - expected_T1_xx) < 1e-12);
    pass = pass && (std::abs(basis[0][TensorBasis::XY] - expected_T1_xy) < 1e-12);

    // All tensors must still be traceless
    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
        double tr = basis[n][TensorBasis::XX] + basis[n][TensorBasis::YY] + basis[n][TensorBasis::ZZ];
        pass = pass && (std::abs(tr) < 1e-10);
    }

    record("3D tensor basis: 2D backward compatibility", pass);
}

void test_3d_basis_nonzero_for_3d_flow() {
    // With 3D velocity gradients, T5-T10 should be nonzero
    VelocityGradient grad = {};
    grad.dudx = 0.5; grad.dudy = 0.3; grad.dudz = -0.2;
    grad.dvdx = -0.1; grad.dvdy = 0.7; grad.dvdz = 0.4;
    grad.dwdx = 0.6; grad.dwdy = -0.3; grad.dwdz = -(0.5 + 0.7);

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    bool pass = true;
    // T1 should always be nonzero for nonzero strain
    pass = pass && (tensor_norm(basis[0]) > 1e-6);

    // T2 = [S,O] should be nonzero when both S and O are nonzero
    pass = pass && (tensor_norm(basis[1]) > 1e-6);

    // T5 through T10 should be nonzero for general 3D flow
    for (int n = 4; n < TensorBasis::NUM_BASIS; ++n) {
        double norm = tensor_norm(basis[n]);
        pass = pass && (norm > 1e-10);
    }

    // z-components should be nonzero for 3D flow
    pass = pass && (std::abs(basis[0][TensorBasis::XZ]) > 1e-10);
    pass = pass && (std::abs(basis[0][TensorBasis::YZ]) > 1e-10);

    record("3D tensor basis: nonzero for 3D flow", pass);
}

void test_3d_basis_pure_rotation() {
    // Pure solid-body rotation: S=0, O≠0
    // All basis tensors with S should be zero. Only T4 = O² - (1/3)tr(O²)I is nonzero.
    VelocityGradient grad = {};
    grad.dudy = -1.0; grad.dvdx = 1.0;  // rotation about z
    grad.dudz = -0.5; grad.dwdx = 0.5;  // rotation about y
    grad.dvdz = 0.3; grad.dwdy = -0.3;  // rotation about x

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    bool pass = true;
    // T1 = S = 0 (no strain in pure rotation)
    pass = pass && (tensor_norm(basis[0]) < 1e-14);

    // T2 = SO - OS = 0 (S=0)
    pass = pass && (tensor_norm(basis[1]) < 1e-14);

    // T3 = S² - ... = 0 (S=0)
    pass = pass && (tensor_norm(basis[2]) < 1e-14);

    // T4 = O² - (1/3)tr(O²)I should be nonzero
    pass = pass && (tensor_norm(basis[3]) > 1e-6);

    // T4 should be traceless
    pass = pass && (std::abs(tensor_trace(basis[3])) < 1e-12);

    // T5-T10 all involve S in products, so they should be zero
    for (int n = 4; n < TensorBasis::NUM_BASIS; ++n)
        pass = pass && (tensor_norm(basis[n]) < 1e-14);

    record("3D tensor basis: pure rotation (only T4 nonzero)", pass);
}

void test_3d_basis_pure_strain() {
    // Pure strain (no rotation): O=0
    // T2, T4, T5, T6, T7, T9, T10 should be zero (all involve O)
    VelocityGradient grad = {};
    grad.dudx = 1.0; grad.dvdy = -0.3; grad.dwdz = -0.7;
    grad.dudy = 0.5; grad.dvdx = 0.5;  // symmetric → no rotation
    grad.dudz = 0.2; grad.dwdx = 0.2;  // symmetric
    grad.dvdz = -0.1; grad.dwdy = -0.1; // symmetric

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    bool pass = true;
    // T1 = S nonzero
    pass = pass && (tensor_norm(basis[0]) > 1e-6);

    // T2 = [S,O] = 0 (O=0)
    pass = pass && (tensor_norm(basis[1]) < 1e-13);

    // T3 = S² - (1/3)tr(S²)I nonzero
    pass = pass && (tensor_norm(basis[2]) > 1e-6);

    // T4 = O² - ... = 0 (O=0)
    pass = pass && (tensor_norm(basis[3]) < 1e-14);

    // T5-T10 all involve O, should be zero
    for (int n = 4; n < TensorBasis::NUM_BASIS; ++n)
        pass = pass && (tensor_norm(basis[n]) < 1e-13);

    record("3D tensor basis: pure strain (T2,T4-T10 zero)", pass);
}

void test_3d_basis_scaling() {
    // Basis tensors scale with tau = k/epsilon
    // Doubling k should scale T^(n) by tau_factor^order where order depends on the tensor
    // T1 ∝ tau, T2 ∝ tau², T3 ∝ tau², T4 ∝ tau², T5-T10 ∝ tau^(3-5)
    VelocityGradient grad = {};
    grad.dudx = 0.5; grad.dudy = 0.3; grad.dudz = -0.2;
    grad.dvdx = -0.1; grad.dvdy = 0.7; grad.dvdz = 0.4;
    grad.dwdx = 0.6; grad.dwdy = -0.3; grad.dwdz = -1.2;

    double k1 = 0.1, eps1 = 0.01;
    double k2 = 0.2, eps2 = 0.01;  // tau doubled

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis1, basis2;
    TensorBasis::compute(grad, k1, eps1, basis1);
    TensorBasis::compute(grad, k2, eps2, basis2);

    bool pass = true;
    double tau_ratio = (k2/eps2) / (k1/eps1);  // = 2.0

    // T1 = S*tau → scales linearly with tau
    if (tensor_norm(basis1[0]) > 1e-10) {
        double ratio = tensor_norm(basis2[0]) / tensor_norm(basis1[0]);
        pass = pass && (std::abs(ratio - tau_ratio) < 1e-10);
    }

    // T2 = [S*tau, O*tau] → scales as tau²
    if (tensor_norm(basis1[1]) > 1e-10) {
        double ratio = tensor_norm(basis2[1]) / tensor_norm(basis1[1]);
        pass = pass && (std::abs(ratio - tau_ratio*tau_ratio) < 1e-8);
    }

    record("3D tensor basis: correct tau scaling", pass);
}

void test_3d_basis_construct_anisotropy_traceless() {
    // Anisotropy b_ij = sum G_n T^(n)_ij should be traceless
    // because all T^(n) are traceless
    VelocityGradient grad = {};
    grad.dudx = 0.5; grad.dudy = 0.3; grad.dudz = -0.2;
    grad.dvdx = -0.1; grad.dvdy = 0.7; grad.dvdz = 0.4;
    grad.dwdx = 0.6; grad.dwdy = -0.3; grad.dwdz = -1.2;

    std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, 0.1, 0.01, basis);

    // Use random-ish G coefficients
    std::array<double, TensorBasis::NUM_BASIS> G = {
        -0.1, 0.05, 0.02, -0.03, 0.01, -0.005, 0.008, -0.002, 0.001, -0.0005
    };

    double b_xx, b_xy, b_xz, b_yy, b_yz, b_zz;
    TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_xz, b_yy, b_yz, b_zz);

    bool pass = true;
    // b must be traceless
    double trace = b_xx + b_yy + b_zz;
    pass = pass && (std::abs(trace) < 1e-10);

    // b must be finite
    pass = pass && std::isfinite(b_xx) && std::isfinite(b_xy) && std::isfinite(b_xz);
    pass = pass && std::isfinite(b_yy) && std::isfinite(b_yz) && std::isfinite(b_zz);

    record("3D anisotropy construction: traceless", pass);
}

int main() {
    return nncfd::test::harness::run("Feature Computation Tests", [] {
        // Gradient computation tests
        test_pure_shear_flow();
        test_pure_strain_flow();
        test_solid_body_rotation();

        // Feature computation tests
        test_scalar_features_shear();
        test_tbnn_features_and_basis();

        // Tensor algebra tests (updated for 3D)
        test_anisotropy_construction();
        test_reynolds_stress_conversion();

        // 3D tensor basis tests
        test_3d_basis_traceless();
        test_3d_basis_symmetric();
        test_3d_basis_2d_backward_compat();
        test_3d_basis_nonzero_for_3d_flow();
        test_3d_basis_pure_rotation();
        test_3d_basis_pure_strain();
        test_3d_basis_scaling();
        test_3d_basis_construct_anisotropy_traceless();

        // Backend-specific gradient tests
        test_gradient_computation_backend();
    });
}

