#pragma once

/// @file velocity_gradient.hpp
/// @brief Velocity gradient tensor computation on staggered MAC grid
///
/// Computes the 9-component velocity gradient tensor (dui/dxj) at cell centers
/// from staggered velocity components. Uses 2nd-order central differences.
///
/// Staggered → cell-center interpolation:
///   u_center(i,j,k) = 0.5 * (u(i,j,k) + u(i+1,j,k))
///   v_center(i,j,k) = 0.5 * (v(i,j,k) + v(i,j+1,k))
///   w_center(i,j,k) = 0.5 * (w(i,j,k) + w(i,j,k+1))

#include "mesh.hpp"
#include "fields.hpp"
#include <vector>

namespace nncfd {

/// 3D velocity gradient tensor stored at cell centers
/// Components: g[i][j] = du_i/dx_j
struct GradientTensor3D {
    /// 9 components, each sized Nx*Ny*Nz (interior cells only)
    std::vector<double> g11, g12, g13;  // du/dx, du/dy, du/dz
    std::vector<double> g21, g22, g23;  // dv/dx, dv/dy, dv/dz
    std::vector<double> g31, g32, g33;  // dw/dx, dw/dy, dw/dz

    int Nx = 0, Ny = 0, Nz = 0;

    void resize(int nx, int ny, int nz);
    int index(int i, int j, int k) const { return k * Nx * Ny + j * Nx + i; }
};

/// Compute velocity gradient tensor from staggered velocity field
class GradientComputer {
public:
    /// Compute all 9 gradient components at cell centers
    /// @param mesh     Computational mesh
    /// @param vel      Staggered velocity field
    /// @param grad     Output gradient tensor (resized internally)
    void compute(const Mesh& mesh, const VectorField& vel,
                 GradientTensor3D& grad) const;
};

} // namespace nncfd
