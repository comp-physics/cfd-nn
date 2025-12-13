#include "features.hpp"
#include <cmath>
#include <algorithm>

namespace nncfd {

VelocityGradient compute_velocity_gradient(
    const Mesh& mesh,
    const VectorField& velocity,
    int i, int j) {
    
    double dx = mesh.dx;
    double dy = mesh.dy;
    
    VelocityGradient grad;
    
    // Central differences
    grad.dudx = (velocity.u(i+1, j) - velocity.u(i-1, j)) / (2.0 * dx);
    grad.dudy = (velocity.u(i, j+1) - velocity.u(i, j-1)) / (2.0 * dy);
    grad.dvdx = (velocity.v(i+1, j) - velocity.v(i-1, j)) / (2.0 * dx);
    grad.dvdy = (velocity.v(i, j+1) - velocity.v(i, j-1)) / (2.0 * dy);
    
    return grad;
}

void compute_all_velocity_gradients(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy) {
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            auto grad = compute_velocity_gradient(mesh, velocity, i, j);
            dudx(i, j) = grad.dudx;
            dudy(i, j) = grad.dudy;
            dvdx(i, j) = grad.dvdx;
            dvdy(i, j) = grad.dvdy;
        }
    }
}

/// Compute gradients from MAC staggered grid (CPU version matching GPU kernel)
/// This mirrors compute_gradients_from_mac_gpu for CPU/GPU consistency
void compute_gradients_from_mac_cpu(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy) {
    
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    
    // Loop over interior cells
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // For gradients at cell (i,j), sample neighboring face values
            // This matches the GPU kernel's indexing exactly
            
            // dudx: central difference of u at x-faces
            dudx(i, j) = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
            
            // dudy: central difference of u at x-faces in y-direction
            dudy(i, j) = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
            
            // dvdx: central difference of v at y-faces in x-direction
            dvdx(i, j) = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
            
            // dvdy: central difference of v at y-faces
            dvdy(i, j) = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
        }
    }
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
    
    auto grad = compute_velocity_gradient(mesh, velocity, i, j);
    
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
    
    auto grad = compute_velocity_gradient(mesh, velocity, i, j);
    
    double S_mag = grad.S_mag();
    double Omega_mag = grad.Omega_mag();
    
    // Get k and epsilon
    double k_val = k(i, j);
    double omega_val = omega(i, j);
    double eps = 0.09 * k_val * omega_val;  // epsilon = C_mu * k * omega
    
    // Avoid division by zero
    double k_safe = std::max(k_val, 1e-10);
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
    std::array<std::array<double, 3>, NUM_BASIS>& basis) {
    
    // Avoid division by zero
    double k_safe = std::max(k, 1e-10);
    double eps_safe = std::max(epsilon, 1e-20);
    double tau = k_safe / eps_safe;
    
    // Normalized strain and rotation (2D)
    double Sxx = grad.Sxx() * tau;
    double Syy = grad.Syy() * tau;
    double Sxy = grad.Sxy() * tau;
    double Oxy = grad.Oxy() * tau;
    
    // T^(1) = S (normalized)
    basis[0][0] = Sxx;  // xx
    basis[0][1] = Sxy;  // xy
    basis[0][2] = Syy;  // yy
    
    // T^(2) = S*Omega - Omega*S
    // In 2D: [S, Omega] = S*Omega - Omega*S
    // (S*Omega)_xx = Sxx*0 + Sxy*(-Oxy) = -Sxy*Oxy
    // (S*Omega)_xy = Sxx*Oxy + Sxy*0 = Sxx*Oxy
    // etc. (Omega is antisymmetric: Omega_xx = Omega_yy = 0, Omega_xy = Oxy, Omega_yx = -Oxy)
    
    // S*Omega:
    double SOxx = Sxy * (-Oxy);  // = -Sxy*Oxy
    double SOxy = Sxx * Oxy;      // = Sxx*Oxy
    double SOyx = Syy * (-Oxy);   // = -Syy*Oxy
    double SOyy = Sxy * Oxy;      // = Sxy*Oxy
    
    // Omega*S:
    double OSxx = (-Oxy) * Sxy;   // = -Oxy*Sxy
    double OSxy = (-Oxy) * Syy;   // = -Oxy*Syy
    double OSyx = Oxy * Sxx;      // = Oxy*Sxx
    double OSyy = Oxy * Sxy;      // = Oxy*Sxy
    
    // Suppress warnings for intermediate calculations
    (void)SOyx; (void)OSyx;
    
    // T^(2) = S*Omega - Omega*S
    basis[1][0] = SOxx - OSxx;  // = 0
    basis[1][1] = SOxy - OSxy;  // = Sxx*Oxy + Oxy*Syy = Oxy*(Sxx + Syy)
    basis[1][2] = SOyy - OSyy;  // = 0
    
    // For 2D incompressible: Sxx + Syy = 0 (traceless)
    // So T^(2)_xy = Oxy * 0 = 0 in this case... Let's recalculate properly
    
    // Actually for 2D: S*Omega - Omega*S simplifies
    // Let's use component form directly
    // [S, Omega]_ij = S_ik * Omega_kj - Omega_ik * S_kj
    // In matrix form with S = [[Sxx, Sxy], [Sxy, Syy]] and Omega = [[0, Oxy], [-Oxy, 0]]
    
    // Recalculate more carefully:
    // S*Omega = [[Sxx, Sxy], [Sxy, Syy]] * [[0, Oxy], [-Oxy, 0]]
    //         = [[-Sxy*Oxy, Sxx*Oxy], [-Syy*Oxy, Sxy*Oxy]]
    
    // Omega*S = [[0, Oxy], [-Oxy, 0]] * [[Sxx, Sxy], [Sxy, Syy]]
    //         = [[Oxy*Sxy, Oxy*Syy], [-Oxy*Sxx, -Oxy*Sxy]]
    
    // T^(2) = S*Omega - Omega*S = 
    //   [[-Sxy*Oxy - Oxy*Sxy, Sxx*Oxy - Oxy*Syy], 
    //    [-Syy*Oxy + Oxy*Sxx, Sxy*Oxy + Oxy*Sxy]]
    // = [[-2*Sxy*Oxy, (Sxx-Syy)*Oxy],
    //    [(Sxx-Syy)*Oxy, 2*Sxy*Oxy]]
    
    basis[1][0] = -2.0 * Sxy * Oxy;
    basis[1][1] = (Sxx - Syy) * Oxy;
    basis[1][2] = 2.0 * Sxy * Oxy;
    
    // T^(3) = S^2 - (1/3)*tr(S^2)*I  (deviatoric part of S^2)
    // S^2 = [[Sxx^2 + Sxy^2, Sxy*(Sxx+Syy)], [Sxy*(Sxx+Syy), Sxy^2 + Syy^2]]
    // For incompressible 2D: Sxx + Syy = 0, so Sxy*(Sxx+Syy) = 0
    // S^2_xx = Sxx^2 + Sxy^2
    // S^2_yy = Sxy^2 + Syy^2 = Sxy^2 + Sxx^2 (since Syy = -Sxx)
    // S^2_xy = 0
    // tr(S^2) = 2*(Sxx^2 + Sxy^2)
    
    double S2xx = Sxx*Sxx + Sxy*Sxy;
    double S2yy = Sxy*Sxy + Syy*Syy;
    double S2xy = Sxy * (Sxx + Syy);
    double trS2 = S2xx + S2yy;
    
    // In 2D, deviatoric: subtract (1/2)*tr(S^2)*I (not 1/3)
    basis[2][0] = S2xx - 0.5 * trS2;
    basis[2][1] = S2xy;
    basis[2][2] = S2yy - 0.5 * trS2;
    
    // T^(4) = Omega^2 - (1/3)*tr(Omega^2)*I
    // Omega^2 = [[0, Oxy], [-Oxy, 0]] * [[0, Oxy], [-Oxy, 0]]
    //         = [[-Oxy^2, 0], [0, -Oxy^2]]
    // tr(Omega^2) = -2*Oxy^2
    // Deviatoric: Omega^2 - (1/2)*tr(Omega^2)*I = [[-Oxy^2 + Oxy^2, 0], [0, -Oxy^2 + Oxy^2]] = 0
    // So T^(4) = 0 in 2D
    
    basis[3][0] = 0.0;
    basis[3][1] = 0.0;
    basis[3][2] = 0.0;
}

void TensorBasis::construct_anisotropy(
    const std::array<double, NUM_BASIS>& G,
    const std::array<std::array<double, 3>, NUM_BASIS>& basis,
    double& b_xx, double& b_xy, double& b_yy) {
    
    b_xx = 0.0;
    b_xy = 0.0;
    b_yy = 0.0;
    
    for (int n = 0; n < NUM_BASIS; ++n) {
        b_xx += G[n] * basis[n][0];
        b_xy += G[n] * basis[n][1];
        b_yy += G[n] * basis[n][2];
    }
}

void TensorBasis::anisotropy_to_reynolds_stress(
    double b_xx, double b_xy, double b_yy,
    double k,
    double& tau_xx, double& tau_xy, double& tau_yy) {
    
    // R_ij = 2*k*(b_ij + (1/3)*delta_ij)
    // In 2D: delta_ij contribution is (1/2) for consistency
    // Actually for the Reynolds stress tensor:
    // u'_i u'_j = 2*k*(b_ij + (1/3)*delta_ij)
    // For 2D approximation we use:
    // tau_ij = -rho * u'_i u'_j (Reynolds stress tensor as appears in RANS)
    
    double k_safe = std::max(k, 0.0);
    tau_xx = 2.0 * k_safe * (b_xx + 1.0/3.0);
    tau_xy = 2.0 * k_safe * b_xy;
    tau_yy = 2.0 * k_safe * (b_yy + 1.0/3.0);
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
    
    // Compute all gradients first
    compute_all_velocity_gradients(*mesh_, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
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
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>>& basis) {
    
    compute_all_velocity_gradients(*mesh_, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
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
            double eps = 0.09 * k_val * omega_val;
            
            TensorBasis::compute(grad, k_val, eps, basis[idx]);
            
            ++idx;
        }
    }
}

} // namespace nncfd


