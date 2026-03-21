#pragma once

/// @file turbulence_les.hpp
/// @brief LES subgrid-scale turbulence models
///
/// Five SGS models for Large Eddy Simulation:
///   1. Smagorinsky — classical constant-coefficient model
///   2. WALE — Wall-Adapting Local Eddy-viscosity
///   3. Vreman — based on algebraic invariants of gradient tensor
///   4. Sigma — based on singular values of gradient tensor
///   5. Dynamic Smagorinsky — Germano procedure with test filter
///
/// All models compute nu_sgs from the velocity gradient tensor.

#include "turbulence_model.hpp"
#include "velocity_gradient.hpp"

#include <vector>

namespace nncfd {

/// Base class for LES SGS models
/// Computes velocity gradient tensor, then delegates to subclass for nu_sgs
class LESModel : public TurbulenceModel {
public:
    void update(const Mesh& mesh, const VectorField& velocity,
                const ScalarField& k, const ScalarField& omega,
                ScalarField& nu_t, TensorField* tau_ij,
                const TurbulenceDeviceView* device_view) override;

    bool uses_transport_equations() const override { return false; }
    bool is_gpu_ready() const override { return true; }

protected:
    /// Compute local filter width from mesh spacing at cell j
    /// Uses actual cell height for stretched grid support
    double filter_width(const Mesh& mesh, int jg) const;

    /// Override in subclass: compute nu_sgs from gradient tensor at a single cell
    /// @param g  Gradient tensor components (row-major: g[0..8] = g11,g12,g13,g21,...)
    /// @param delta  Filter width
    /// @return nu_sgs for this cell
    virtual double compute_nu_sgs_cell(const double g[9], double delta) const = 0;

    /// Override in subclass: GPU kernel for fused gradient + nu_sgs computation
    virtual void update_gpu(const TurbulenceDeviceView* dv) = 0;

    GradientComputer grad_computer_;
    GradientTensor3D grad_;
};

/// Static Smagorinsky model: nu_sgs = (Cs * delta)^2 * |S|
/// where |S| = sqrt(2 * Sij * Sij), Sij = 0.5*(dui/dxj + duj/dxi)
class SmagorinskyModel : public LESModel {
public:
    explicit SmagorinskyModel(double Cs = 0.17) : Cs_(Cs) {}
    std::string name() const override { return "Smagorinsky"; }

protected:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;

private:
    double Cs_;
};

/// WALE model: Wall-Adapting Local Eddy-viscosity (Nicoud & Ducros 1999)
/// nu_sgs = (Cw * delta)^2 * (Sd_ij * Sd_ij)^(3/2) /
///          ((Sij*Sij)^(5/2) + (Sd_ij*Sd_ij)^(5/4))
/// where Sd_ij = 0.5*(gik*gkj + gjk*gki) - (1/3)*delta_ij*(gkk*gkk)
class WALEModel : public LESModel {
public:
    explicit WALEModel(double Cw = 0.325) : Cw_(Cw) {}
    std::string name() const override { return "WALE"; }

protected:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;

private:
    double Cw_;
};

/// Vreman model (Vreman 2004)
/// nu_sgs = Cv * sqrt(B_beta / (alpha_ij * alpha_ij))
/// where alpha_ij = duj/dxi, B_beta = det(beta_ij), beta_ij = sum_m(alpha_mi * alpha_mj)
class VremanModel : public LESModel {
public:
    explicit VremanModel(double Cv = 0.07) : Cv_(Cv) {}
    std::string name() const override { return "Vreman"; }

protected:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;

private:
    double Cv_;
};

/// Sigma model (Nicoud et al. 2011)
/// nu_sgs = (Cs * delta)^2 * sigma3 * (sigma1 - sigma2) * (sigma2 - sigma3) / sigma1^2
/// where sigma1 >= sigma2 >= sigma3 are singular values of the gradient tensor
class SigmaModel : public LESModel {
public:
    explicit SigmaModel(double Cs = 1.35) : Cs_(Cs) {}
    std::string name() const override { return "Sigma"; }

protected:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;

private:
    double Cs_;
};

/// Dynamic Smagorinsky model (Germano et al. 1991, Lilly 1992)
/// Cs^2 computed dynamically via Germano identity with test filter at 2*delta.
/// Uses plane-averaged Cs^2 in x-z (homogeneous directions).
class Decomposition;  // Forward declaration

class DynamicSmagorinskyModel : public LESModel {
public:
    ~DynamicSmagorinskyModel() override;
    std::string name() const override { return "DynamicSmagorinsky"; }

    /// Set MPI decomposition for plane-averaged Cs² allreduce
    void set_decomposition(Decomposition* decomp) { decomp_ = decomp; }

    void update(const Mesh& mesh, const VectorField& velocity,
                const ScalarField& k, const ScalarField& omega,
                ScalarField& nu_t, TensorField* tau_ij,
                const TurbulenceDeviceView* device_view) override;

protected:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;

private:
    void init_dynamic_gpu(const TurbulenceDeviceView* dv);
    void cleanup_dynamic_gpu();

    // Cell-centered velocity scratch (GPU-mapped)
    double* u_cc_ = nullptr;
    double* v_cc_ = nullptr;
    double* w_cc_ = nullptr;
    // Per-y-plane Germano sums (GPU-mapped)
    double* LM_plane_ = nullptr;  // sum of L_ij * M_ij per plane
    double* MM_plane_ = nullptr;  // sum of M_ij * M_ij per plane
    double* Cs2_plane_ = nullptr; // resulting Cs^2(y)
    int Ny_ = 0;
    int cell_total_ = 0;
    bool dyn_gpu_ready_ = false;
    Decomposition* decomp_ = nullptr;  // For MPI allreduce of plane sums
};

} // namespace nncfd
