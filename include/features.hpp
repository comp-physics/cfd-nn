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

/// Velocity gradient tensor at a cell (full 3D: 9 components)
struct VelocityGradient {
    double dudx, dudy;
    double dvdx, dvdy;
    // 3D gradient components (zero-initialized for backward compatibility)
    double dudz = 0.0;
    double dvdz = 0.0;
    double dwdx = 0.0, dwdy = 0.0, dwdz = 0.0;

    /// Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    double Sxx() const { return dudx; }
    double Syy() const { return dvdy; }
    double Szz() const { return dwdz; }
    double Sxy() const { return 0.5 * (dudy + dvdx); }
    double Sxz() const { return 0.5 * (dudz + dwdx); }
    double Syz() const { return 0.5 * (dvdz + dwdy); }

    /// Rotation tensor Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
    double Oxy() const { return 0.5 * (dudy - dvdx); }
    double Oxz() const { return 0.5 * (dudz - dwdx); }
    double Oyz() const { return 0.5 * (dvdz - dwdy); }

    /// Strain rate magnitude |S| = sqrt(S_ij * S_ij) (Frobenius norm, full 3D)
    double S_mag() const {
        return std::sqrt(Sxx()*Sxx() + Syy()*Syy() + Szz()*Szz()
                       + 2.0*(Sxy()*Sxy() + Sxz()*Sxz() + Syz()*Syz()));
    }

    /// Rotation rate magnitude |Omega| = sqrt(2 * Omega_ij * Omega_ij) (full 3D)
    double Omega_mag() const {
        return std::sqrt(2.0 * (Oxy()*Oxy() + Oxz()*Oxz() + Oyz()*Oyz()));
    }
};

/// Compute gradients from MAC staggered grid (abstraction wrapper)
/// This extracts raw pointers and calls the unified implementation in
/// gpu_kernels::compute_gradients_from_mac_gpu, which handles both CPU and
/// GPU paths via conditional compilation. This ensures CPU/GPU consistency
/// by having a single source of truth for the gradient computation.
void compute_gradients_from_mac(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy
);

/// Feature computation for scalar eddy viscosity NN
/// Computes 6 features:
///   0: normalized strain rate magnitude (S * delta / u_ref)
///   1: normalized rotation rate magnitude (Omega * delta / u_ref)
///   2: normalized wall distance (y / delta)
///   3: strain-rotation ratio (Omega / S)
///   4: local Reynolds number (S * delta^2 / nu)
///   5: normalized velocity magnitude
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
/// Computes 5 scalar invariants of normalized S and Omega tensors
Features compute_features_tbnn(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    int i, int j,
    double nu,
    double delta
);

/// Tensor basis for TBNN — Pope (1975) integrity basis
/// Always 10 tensors × 6 components (xx, xy, xz, yy, yz, zz).
/// In 2D (z-gradients=0): xz/yz are zero; T5-T10 are linearly dependent on T1-T4.
///
/// T^(1) = S
/// T^(2) = SO - OS
/// T^(3) = S^2 - (1/3)*tr(S^2)*I
/// T^(4) = O^2 - (1/3)*tr(O^2)*I
/// T^(5) = OS^2 - S^2O
/// T^(6) = O^2S + SO^2 - (2/3)*tr(SO^2)*I
/// T^(7) = OSO^2 - O^2SO
/// T^(8) = SOS^2 - S^2OS
/// T^(9) = O^2S^2 + S^2O^2 - (2/3)*tr(S^2O^2)*I
/// T^(10) = OS^2O^2 - O^2S^2O
class TensorBasis {
public:
    static constexpr int NUM_BASIS = 10;       // Pope (1975) full 3D basis
    static constexpr int NUM_COMPONENTS = 6;   // xx, xy, xz, yy, yz, zz

    /// Component indices for symmetric tensor storage
    static constexpr int XX = 0, XY = 1, XZ = 2, YY = 3, YZ = 4, ZZ = 5;

    /// Compute full 3D tensor basis at a cell given velocity gradients
    /// Output: basis[n][c] where n=0..9 (basis index), c=0..5 (component)
    /// In 2D (all z-gradients zero), T5-T10 are identically zero
    static void compute(
        const VelocityGradient& grad,
        double k, double epsilon,
        std::array<std::array<double, NUM_COMPONENTS>, NUM_BASIS>& basis
    );

    /// Construct anisotropy tensor from coefficients (full 3D)
    /// b_ij = sum_n G_n * T^n_ij
    static void construct_anisotropy(
        const std::array<double, NUM_BASIS>& G,
        const std::array<std::array<double, NUM_COMPONENTS>, NUM_BASIS>& basis,
        double& b_xx, double& b_xy, double& b_xz,
        double& b_yy, double& b_yz, double& b_zz
    );

    /// Convert anisotropy to Reynolds stress (full 3D)
    /// tau_ij = 2*k*(b_ij + (1/3)*delta_ij)
    static void anisotropy_to_reynolds_stress(
        double b_xx, double b_xy, double b_xz,
        double b_yy, double b_yz, double b_zz,
        double k,
        double& tau_xx, double& tau_xy, double& tau_xz,
        double& tau_yy, double& tau_yz, double& tau_zz
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
        std::vector<std::array<std::array<double, TensorBasis::NUM_COMPONENTS>, TensorBasis::NUM_BASIS>>& basis
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


