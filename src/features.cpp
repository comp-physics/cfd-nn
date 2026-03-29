/// @file features.cpp
/// @brief Feature computation for data-driven turbulence models
///
/// This file implements computation of turbulence invariants and tensor basis
/// functions used by neural network and EARSM models. Key features:
/// - Velocity gradient computation from MAC staggered grid
/// - Galilean/rotation invariants of strain and rotation tensors
/// - Tensor basis functions (Pope 1975 integrity basis)
/// - Feature normalization for neural network inputs
/// - Anisotropy tensor reconstruction from basis coefficients
///
/// The invariants computed here are frame-independent (Galilean invariant) and
/// form the inputs to data-driven closures like TBNN and EARSM.

#include "features.hpp"
#include "gpu_kernels.hpp"
#include "numerics.hpp"
#include <cmath>
#include <algorithm>

namespace nncfd {

/// Compute gradients from MAC staggered grid (wrapper for abstraction types)
/// This thin wrapper extracts raw pointers from Mesh/VectorField/ScalarField
/// and calls the unified implementation in gpu_kernels::compute_gradients_from_mac_gpu.
/// The unified implementation handles both CPU and GPU paths via conditional compilation.
void compute_gradients_from_mac(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy) {

    // Extract raw pointers from abstractions
    const double* u_face = velocity.u_data().data();
    const double* v_face = velocity.v_data().data();
    const double* w_face = velocity.w_data().empty() ? nullptr : velocity.w_data().data();
    double* dudx_cell = dudx.data().data();
    double* dudy_cell = dudy.data().data();
    double* dvdx_cell = dvdx.data().data();
    double* dvdy_cell = dvdy.data().data();

    // Get dimensions and strides
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;
    const int u_stride = velocity.u_stride();
    const int v_stride = velocity.v_stride();
    const int cell_stride = mesh.total_Nx();
    const int u_plane_stride = velocity.u_plane_stride();
    const int v_plane_stride = velocity.v_plane_stride();
    const int w_stride = velocity.w_stride();
    const int w_plane_stride = velocity.w_plane_stride();
    const int cell_plane_stride = mesh.total_Nx() * mesh.total_Ny();

    // Pass dyc for stretched grids (nullptr for uniform)
    const double* dyc_ptr = mesh.dyc.empty() ? nullptr : mesh.dyc.data();
    int dyc_size = static_cast<int>(mesh.dyc.size());

    // Call the unified implementation (uses CPU path when USE_GPU_OFFLOAD not defined)
    // 2D wrapper: pass nullptr for 3D gradient outputs
    gpu_kernels::compute_gradients_from_mac_gpu(
        u_face, v_face, w_face,
        dudx_cell, dudy_cell, dvdx_cell, dvdy_cell,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        Nx, Ny, Nz, Ng,
        mesh.dx, mesh.dy, mesh.dz,
        u_stride, v_stride, cell_stride,
        u_plane_stride, v_plane_stride,
        w_stride, w_plane_stride, cell_plane_stride,
        velocity.u_total_size(),
        velocity.v_total_size(),
        velocity.w_total_size(),
        static_cast<int>(dudx.data().size()),
        dyc_ptr, dyc_size
    );
}

Features compute_features_scalar_nut(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    int i, int j,
    double nu,
    double delta) {
    
    Features feat(6);  // 6 features for scalar nu_t model
    
    // Compute gradients using MAC-aware method (matches GPU)
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    
    VelocityGradient grad;
    grad.dudx = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
    grad.dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
    
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    double y_wall = mesh.wall_distance(i, j);
    double u_mag = velocity.magnitude(i, j);
    
    // Reference velocity (use bulk or local)
    double u_ref = std::max(u_mag, 1e-10);
    
    // Feature 0: Normalized strain rate
    feat[0] = S_mag * delta / u_ref;
    
    // Feature 1: Normalized rotation rate
    feat[1] = Omega_mag * delta / u_ref;
    
    // Feature 2: Normalized wall distance
    feat[2] = y_wall / delta;
    
    // Feature 3: Strain-rotation ratio
    feat[3] = (S_mag > 1e-10) ? Omega_mag / S_mag : 0.0;
    
    // Feature 4: Local Reynolds number based on strain
    feat[4] = S_mag * delta * delta / nu;
    
    // Feature 5: Normalized velocity magnitude
    feat[5] = u_mag / u_ref;
    
    // Optional: include k and omega if available
    (void)k;
    (void)omega;
    
    return feat;
}

Features compute_features_tbnn(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    int i, int j,
    double nu,
    double delta) {
    
    Features feat(5);  // Invariants for TBNN
    
    // Compute gradients using MAC-aware method (matches GPU)
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    
    VelocityGradient grad;
    grad.dudx = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
    grad.dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
    
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    
    // Get k and epsilon
    double k_val = k(i, j);
    double omega_val = omega(i, j);
    double eps = numerics::C_MU * k_val * omega_val;  // epsilon = C_mu * k * omega

    // Avoid division by zero
    double k_safe = std::max(k_val, numerics::K_FLOOR);
    double eps_safe = std::max(eps, 1e-20);
    
    // Time scale for normalization
    double tau = k_safe / eps_safe;
    
    // Normalized strain and rotation tensors
    // S_norm = S * tau, Omega_norm = Omega * tau
    double S_norm = S_mag * tau;
    double Omega_norm = Omega_mag * tau;
    
    // Invariants (2D case, simplified)
    // Lambda1 = tr(S^2) ~ S_mag^2
    // Lambda2 = tr(Omega^2) ~ Omega_mag^2
    // Lambda3 = tr(S^3) = 0 in 2D trace-free case
    // Lambda4 = tr(S * Omega^2) 
    // Lambda5 = tr(S^2 * Omega^2)
    
    feat[0] = S_norm * S_norm;      // ~tr(S_norm^2)
    feat[1] = Omega_norm * Omega_norm;  // ~tr(Omega_norm^2)
    
    // Higher order invariants (simplified)
    double Sxx = grad.Sxx() * tau;
    double Syy = grad.Syy() * tau;
    double Sxy = grad.Sxy() * tau;
    double Oxy = grad.Oxy() * tau;
    
    // tr(S^2) = Sxx^2 + Syy^2 + 2*Sxy^2
    feat[2] = Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy;
    
    // tr(Omega^2) = -2*Oxy^2 (antisymmetric)
    feat[3] = 2.0 * Oxy * Oxy;
    
    // Wall distance (normalized)
    feat[4] = mesh.wall_distance(i, j) / delta;
    
    (void)nu;
    
    return feat;
}

void TensorBasis::compute(
    const VelocityGradient& grad,
    double k, double epsilon,
    std::array<std::array<double, NUM_COMPONENTS>, NUM_BASIS>& basis) {

    // Avoid division by zero
    double k_safe = std::max(k, 1e-10);
    double eps_safe = std::max(epsilon, 1e-20);
    double tau = k_safe / eps_safe;

    // Normalized strain rate (symmetric, traceless) — 6 independent components
    double S[3][3] = {
        {grad.Sxx() * tau,  grad.Sxy() * tau,  grad.Sxz() * tau},
        {grad.Sxy() * tau,  grad.Syy() * tau,  grad.Syz() * tau},
        {grad.Sxz() * tau,  grad.Syz() * tau,  grad.Szz() * tau}
    };

    // Normalized rotation rate (antisymmetric) — 3 independent components
    double O[3][3] = {
        { 0.0,              grad.Oxy() * tau,  grad.Oxz() * tau},
        {-grad.Oxy() * tau, 0.0,               grad.Oyz() * tau},
        {-grad.Oxz() * tau,-grad.Oyz() * tau,  0.0}
    };

    // Helper: 3x3 matrix multiply C = A * B
    auto mat_mul = [](const double A[3][3], const double B[3][3], double C[3][3]) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                C[i][j] = 0.0;
                for (int m = 0; m < 3; ++m)
                    C[i][j] += A[i][m] * B[m][j];
            }
    };

    // Helper: trace of a 3x3 matrix
    auto tr = [](const double A[3][3]) { return A[0][0] + A[1][1] + A[2][2]; };

    // Helper: extract symmetric tensor components (xx,xy,xz,yy,yz,zz)
    auto store_sym = [](const double A[3][3], std::array<double, NUM_COMPONENTS>& out) {
        out[XX] = A[0][0]; out[XY] = 0.5*(A[0][1]+A[1][0]); out[XZ] = 0.5*(A[0][2]+A[2][0]);
        out[YY] = A[1][1]; out[YZ] = 0.5*(A[1][2]+A[2][1]); out[ZZ] = A[2][2];
    };

    // Helper: store symmetric deviatoric (subtract (1/3)*trace*I)
    auto store_sym_dev = [&tr](const double A[3][3], std::array<double, NUM_COMPONENTS>& out) {
        double t = tr(A) / 3.0;
        out[XX] = 0.5*(A[0][0]+A[0][0]) - t;
        out[XY] = 0.5*(A[0][1]+A[1][0]);
        out[XZ] = 0.5*(A[0][2]+A[2][0]);
        out[YY] = 0.5*(A[1][1]+A[1][1]) - t;
        out[YZ] = 0.5*(A[1][2]+A[2][1]);
        out[ZZ] = 0.5*(A[2][2]+A[2][2]) - t;
    };

    // Compute intermediate matrix products
    double S2[3][3], O2[3][3];
    double SO[3][3], OS[3][3];
    mat_mul(S, S, S2);
    mat_mul(O, O, O2);
    mat_mul(S, O, SO);
    mat_mul(O, S, OS);

    // T^(1) = S
    store_sym(S, basis[0]);

    // T^(2) = SO - OS
    double T2[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T2[i][j] = SO[i][j] - OS[i][j];
    store_sym(T2, basis[1]);

    // T^(3) = S^2 - (1/3)*tr(S^2)*I
    store_sym_dev(S2, basis[2]);

    // T^(4) = O^2 - (1/3)*tr(O^2)*I
    store_sym_dev(O2, basis[3]);

    // T^(5) = OS^2 - S^2O
    double OS2[3][3], S2O[3][3];
    mat_mul(O, S2, OS2);
    mat_mul(S2, O, S2O);
    double T5[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T5[i][j] = OS2[i][j] - S2O[i][j];
    store_sym(T5, basis[4]);

    // T^(6) = O^2S + SO^2 - (2/3)*tr(SO^2)*I
    double O2S[3][3], SO2[3][3];
    mat_mul(O2, S, O2S);
    mat_mul(S, O2, SO2);
    double T6[3][3];
    double tr_SO2 = tr(SO2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T6[i][j] = O2S[i][j] + SO2[i][j] - (2.0/3.0) * tr_SO2 * (i == j ? 1.0 : 0.0);
    store_sym(T6, basis[5]);

    // T^(7) = O*S*O^2 - O^2*S*O
    double OSO2[3][3], O2SO[3][3];
    mat_mul(O, SO2, OSO2);   // O*(S*O^2)
    mat_mul(O2, SO, O2SO);   // O^2*(S*O)
    double T7[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T7[i][j] = OSO2[i][j] - O2SO[i][j];
    store_sym(T7, basis[6]);

    // T^(8) = S*O*S^2 - S^2*O*S
    double SOS2[3][3], S2OS[3][3];
    mat_mul(SO, S2, SOS2);   // (S*O)*S^2
    mat_mul(S2O, S, S2OS);   // (S^2*O)*S  (reuses S2O from T5)
    double T8[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T8[i][j] = SOS2[i][j] - S2OS[i][j];
    store_sym(T8, basis[7]);

    // T^(9) = O^2*S^2 + S^2*O^2 - (2/3)*tr(S^2*O^2)*I
    double O2S2[3][3], S2O2[3][3];
    mat_mul(O2, S2, O2S2);
    mat_mul(S2, O2, S2O2);
    double tr_S2O2 = tr(S2O2);
    double T9[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T9[i][j] = O2S2[i][j] + S2O2[i][j] - (2.0/3.0) * tr_S2O2 * (i == j ? 1.0 : 0.0);
    store_sym(T9, basis[8]);

    // T^(10) = O*S^2*O^2 - O^2*S^2*O
    double OS2O2[3][3], O2S2O[3][3];
    mat_mul(O, S2O2, OS2O2);  // O*(S^2*O^2)
    mat_mul(O2, S2O, O2S2O);  // O^2*(S^2*O)  (reuses S2O from T5)
    double T10[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T10[i][j] = OS2O2[i][j] - O2S2O[i][j];
    store_sym(T10, basis[9]);
}

void TensorBasis::construct_anisotropy(
    const std::array<double, NUM_BASIS>& G,
    const std::array<std::array<double, NUM_COMPONENTS>, NUM_BASIS>& basis,
    double& b_xx, double& b_xy, double& b_xz,
    double& b_yy, double& b_yz, double& b_zz) {

    b_xx = 0.0; b_xy = 0.0; b_xz = 0.0;
    b_yy = 0.0; b_yz = 0.0; b_zz = 0.0;

    for (int n = 0; n < NUM_BASIS; ++n) {
        b_xx += G[n] * basis[n][XX];
        b_xy += G[n] * basis[n][XY];
        b_xz += G[n] * basis[n][XZ];
        b_yy += G[n] * basis[n][YY];
        b_yz += G[n] * basis[n][YZ];
        b_zz += G[n] * basis[n][ZZ];
    }
}

void TensorBasis::anisotropy_to_reynolds_stress(
    double b_xx, double b_xy, double b_xz,
    double b_yy, double b_yz, double b_zz,
    double k,
    double& tau_xx, double& tau_xy, double& tau_xz,
    double& tau_yy, double& tau_yz, double& tau_zz) {

    // tau_ij = 2*k*(b_ij + (1/3)*delta_ij)
    double k_safe = std::max(k, 0.0);
    tau_xx = 2.0 * k_safe * (b_xx + 1.0/3.0);
    tau_xy = 2.0 * k_safe * b_xy;
    tau_xz = 2.0 * k_safe * b_xz;
    tau_yy = 2.0 * k_safe * (b_yy + 1.0/3.0);
    tau_yz = 2.0 * k_safe * b_yz;
    tau_zz = 2.0 * k_safe * (b_zz + 1.0/3.0);
}

FeatureComputer::FeatureComputer(const Mesh& mesh)
    : mesh_(&mesh)
    , dudx_(mesh), dudy_(mesh), dvdx_(mesh), dvdy_(mesh) {}

void FeatureComputer::set_reference(double nu, double delta, double u_ref) {
    nu_ = nu;
    delta_ = delta;
    u_ref_ = u_ref;
}

void FeatureComputer::compute_scalar_features(
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    std::vector<Features>& features) {
    
    // Compute all gradients first (MAC-aware for CPU/GPU consistency)
    compute_gradients_from_mac(*mesh_, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    // Resize output
    int n_interior = mesh_->Nx * mesh_->Ny;
    features.resize(n_interior);
    
    int idx = 0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            features[idx] = compute_features_scalar_nut(*mesh_, velocity, k, omega, 
                                                        i, j, nu_, delta_);
            ++idx;
        }
    }
}

void FeatureComputer::compute_tbnn_features(
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    std::vector<Features>& features,
    std::vector<std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS>>& basis) {
    
    compute_gradients_from_mac(*mesh_, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    int n_interior = mesh_->Nx * mesh_->Ny;
    features.resize(n_interior);
    basis.resize(n_interior);
    
    int idx = 0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            features[idx] = compute_features_tbnn(*mesh_, velocity, k, omega,
                                                  i, j, nu_, delta_);
            
            // Compute tensor basis
            VelocityGradient grad;
            grad.dudx = dudx_(i, j);
            grad.dudy = dudy_(i, j);
            grad.dvdx = dvdx_(i, j);
            grad.dvdy = dvdy_(i, j);
            
            double k_val = k(i, j);
            double omega_val = omega(i, j);
            double eps = numerics::C_MU * k_val * omega_val;

            TensorBasis::compute(grad, k_val, eps, basis[idx]);
            
            ++idx;
        }
    }
}

} // namespace nncfd


