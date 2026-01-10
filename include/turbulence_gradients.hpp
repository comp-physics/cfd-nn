/// @file turbulence_gradients.hpp
/// @brief Unified velocity gradient computation for turbulence models
///
/// Provides a single interface for computing velocity gradients from MAC-staggered
/// grids, used by all turbulence closures (mixing length, EARSM, k-omega, NN).
///
/// Key features:
/// - Works on both CPU and GPU via gpu_kernels
/// - Handles MAC grid staggering correctly
/// - Provides both raw pointer and ScalarField interfaces
/// - Thread-safe and reentrant

#pragma once

#include "mesh.hpp"
#include "fields.hpp"

namespace nncfd {

/// Velocity gradient components at a cell center
struct VelocityGradient {
    double dudx = 0.0;
    double dudy = 0.0;
    double dvdx = 0.0;
    double dvdy = 0.0;

    /// Strain rate tensor magnitude: |S| = sqrt(2 * S_ij * S_ij)
    double S_mag() const {
        double Sxx = dudx;
        double Syy = dvdy;
        double Sxy = 0.5 * (dudy + dvdx);
        return std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
    }

    /// Rotation rate tensor magnitude: |Omega| = sqrt(2 * Omega_ij * Omega_ij)
    double Omega_mag() const {
        double Omega_xy = 0.5 * (dudy - dvdx);
        return std::sqrt(2.0 * Omega_xy * Omega_xy);
    }

    /// Strain-rotation parameter: 0 = pure strain, 1 = pure rotation
    double strain_rotation_ratio() const {
        double s = S_mag();
        double o = Omega_mag();
        return (s > 1e-10) ? o / s : 0.0;
    }
};

/// Compute velocity gradients from MAC staggered grid
/// Wrapper around gpu_kernels implementation for use with ScalarField outputs
///
/// @param mesh     Computational mesh
/// @param velocity MAC-staggered velocity field
/// @param dudx     [out] du/dx at cell centers
/// @param dudy     [out] du/dy at cell centers
/// @param dvdx     [out] dv/dx at cell centers
/// @param dvdy     [out] dv/dy at cell centers
void compute_gradients_from_mac(
    const Mesh& mesh,
    const VectorField& velocity,
    ScalarField& dudx, ScalarField& dudy,
    ScalarField& dvdx, ScalarField& dvdy);

/// Compute velocity gradient at a single cell (CPU only, for scalar feature computation)
/// Uses central differencing from neighboring MAC faces
///
/// @param mesh     Computational mesh
/// @param velocity MAC-staggered velocity field
/// @param i, j     Cell indices
/// @return Gradient components at cell center
inline VelocityGradient compute_gradient_at_cell(
    const Mesh& mesh,
    const VectorField& velocity,
    int i, int j)
{
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);

    VelocityGradient grad;
    grad.dudx = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
    grad.dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
    return grad;
}

/// Compute velocity gradient at a single cell (3D)
inline VelocityGradient compute_gradient_at_cell_3d(
    const Mesh& mesh,
    const VectorField& velocity,
    int i, int j, int k)
{
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);

    VelocityGradient grad;
    grad.dudx = (velocity.u(i + 1, j, k) - velocity.u(i - 1, j, k)) * inv_2dx;
    grad.dudy = (velocity.u(i, j + 1, k) - velocity.u(i, j - 1, k)) * inv_2dy;
    grad.dvdx = (velocity.v(i + 1, j, k) - velocity.v(i - 1, j, k)) * inv_2dx;
    grad.dvdy = (velocity.v(i, j + 1, k) - velocity.v(i, j - 1, k)) * inv_2dy;
    return grad;
}

/// Helper class for managing gradient computation with automatic memory
/// Useful when turbulence model doesn't own gradient storage
class GradientComputer {
public:
    explicit GradientComputer(const Mesh& mesh)
        : dudx_(mesh), dudy_(mesh), dvdx_(mesh), dvdy_(mesh) {}

    /// Compute all gradients from velocity field
    void compute(const Mesh& mesh, const VectorField& velocity) {
        compute_gradients_from_mac(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    }

    /// Access computed gradients
    const ScalarField& dudx() const { return dudx_; }
    const ScalarField& dudy() const { return dudy_; }
    const ScalarField& dvdx() const { return dvdx_; }
    const ScalarField& dvdy() const { return dvdy_; }

    ScalarField& dudx() { return dudx_; }
    ScalarField& dudy() { return dudy_; }
    ScalarField& dvdx() { return dvdx_; }
    ScalarField& dvdy() { return dvdy_; }

    /// Get gradient at a specific cell
    VelocityGradient at(int i, int j) const {
        VelocityGradient g;
        g.dudx = dudx_(i, j);
        g.dudy = dudy_(i, j);
        g.dvdx = dvdx_(i, j);
        g.dvdy = dvdy_(i, j);
        return g;
    }

private:
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
};

} // namespace nncfd
