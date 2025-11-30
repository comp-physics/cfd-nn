#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include <vector>
#include <array>

namespace nncfd {

/// Feature vector container
struct Features {
    std::vector<double> values;
    
    Features() = default;
    explicit Features(int n) : values(n, 0.0) {}
    
    double& operator[](int i) { return values[i]; }
    double operator[](int i) const { return values[i]; }
    int size() const { return static_cast<int>(values.size()); }
};

/// Velocity gradient tensor at a cell
struct VelocityGradient {
    double dudx, dudy;
    double dvdx, dvdy;
    
    /// Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    double Sxx() const { return dudx; }
    double Syy() const { return dvdy; }
    double Sxy() const { return 0.5 * (dudy + dvdx); }
    
    /// Rotation tensor Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
    double Oxy() const { return 0.5 * (dudy - dvdx); }
    
    /// Strain rate magnitude |S| = sqrt(2 * S_ij * S_ij)
    double S_mag() const {
        return std::sqrt(2.0 * (Sxx()*Sxx() + Syy()*Syy() + 2.0*Sxy()*Sxy()));
    }
    
    /// Rotation rate magnitude |Omega|
    double Omega_mag() const {
        return std::sqrt(2.0 * Oxy() * Oxy());
    }
};

/// Compute velocity gradients at cell (i, j)
VelocityGradient compute_velocity_gradient(
    const Mesh& mesh,
    const VectorField& velocity,
    int i, int j
);

/// Compute all velocity gradients for the mesh
void compute_all_velocity_gradients(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy
);

/// Feature computation for scalar eddy viscosity NN
/// Standard features include:
///   0: normalized strain rate magnitude (S * delta / u_ref)
///   1: normalized rotation rate magnitude (Omega * delta / u_ref)  
///   2: normalized wall distance (y / delta)
///   3: strain-rotation ratio (Omega / S)
///   4: local Reynolds number (S * delta^2 / nu)
///   5: normalized velocity magnitude
///   etc.
Features compute_features_scalar_nut(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    int i, int j,
    double nu,
    double delta  ///< Reference length (e.g., channel half-height)
);

/// Feature computation for TBNN model
/// Includes invariants of normalized S and Omega tensors
Features compute_features_tbnn(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    int i, int j,
    double nu,
    double delta
);

/// Tensor basis for TBNN (2D case)
/// T^(1) = S
/// T^(2) = S*Omega - Omega*S
/// T^(3) = S^2 - (1/2)*tr(S^2)*I
/// etc.
/// Returns basis tensors as [T1_xx, T1_xy, T1_yy, T2_xx, ...]
class TensorBasis {
public:
    static constexpr int NUM_BASIS = 4;  // Reduced set for 2D
    
    /// Compute tensor basis at a cell given velocity gradients
    /// Output: basis[n][0] = T^n_xx, basis[n][1] = T^n_xy, basis[n][2] = T^n_yy
    static void compute(
        const VelocityGradient& grad,
        double k, double epsilon,
        std::array<std::array<double, 3>, NUM_BASIS>& basis
    );
    
    /// Construct anisotropy tensor from coefficients
    /// b_ij = sum_n G_n * T^n_ij
    static void construct_anisotropy(
        const std::array<double, NUM_BASIS>& G,
        const std::array<std::array<double, 3>, NUM_BASIS>& basis,
        double& b_xx, double& b_xy, double& b_yy
    );
    
    /// Convert anisotropy to Reynolds stress
    /// tau_ij = 2*k*(b_ij + (1/3)*delta_ij)
    static void anisotropy_to_reynolds_stress(
        double b_xx, double b_xy, double b_yy,
        double k,
        double& tau_xx, double& tau_xy, double& tau_yy
    );
};

/// Batch feature computation for all interior cells
class FeatureComputer {
public:
    explicit FeatureComputer(const Mesh& mesh);
    
    /// Set reference quantities
    void set_reference(double nu, double delta, double u_ref);
    
    /// Compute features for scalar nu_t model
    void compute_scalar_features(
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        std::vector<Features>& features  ///< One per interior cell
    );
    
    /// Compute features for TBNN model
    void compute_tbnn_features(
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        std::vector<Features>& features,
        std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>>& basis
    );
    
private:
    const Mesh* mesh_;
    double nu_ = 0.001;
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    
    // Cached gradient fields
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
};

} // namespace nncfd


