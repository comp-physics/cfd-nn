#pragma once

#include "mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>

namespace nncfd {

/// Scalar field on structured 2D/3D mesh
class ScalarField {
public:
    ScalarField() = default;
    explicit ScalarField(const Mesh& mesh, double init_val = 0.0);

    /// Access by (i, j) - 2D backward compatible (uses k=0 plane for flat array compatibility)
    double& operator()(int i, int j) { return data_[mesh_->index(i, j)]; }
    double operator()(int i, int j) const { return data_[mesh_->index(i, j)]; }

    /// Access by (i, j, k) - 3D
    double& operator()(int i, int j, int k) { return data_[mesh_->index(i, j, k)]; }
    double operator()(int i, int j, int k) const { return data_[mesh_->index(i, j, k)]; }

    /// Access by flat index
    double& operator[](int idx) { return data_[idx]; }
    double operator[](int idx) const { return data_[idx]; }

    /// Fill with constant value
    void fill(double val);

    /// Get max value in interior
    double max_interior() const;

    /// L2 norm of interior values
    double norm_L2() const;

    /// Linf norm of interior values
    double norm_Linf() const;

    /// Raw data access
    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }

    const Mesh* mesh() const { return mesh_; }

    /// Write to file (simple text format)
    void write(const std::string& filename) const;

private:
    const Mesh* mesh_ = nullptr;
    std::vector<double> data_;
};

/// 2D/3D Vector field (u, v, w components) on staggered MAC grid
/// u is stored at x-faces (i+1/2, j, k), v at y-faces (i, j+1/2, k), w at z-faces (i, j, k+1/2)
class VectorField {
public:
    VectorField() = default;
    explicit VectorField(const Mesh& mesh, double init_u = 0.0, double init_v = 0.0, double init_w = 0.0);

    /// Access u component at x-face (i+1/2, j) - 2D backward compatible (uses k=0 plane for flat array compatibility)
    double& u(int i, int j) { return u_data_[u_index(i, j, 0)]; }
    double u(int i, int j) const { return u_data_[u_index(i, j, 0)]; }

    /// Access u component at x-face (i+1/2, j, k) - 3D
    double& u(int i, int j, int k) { return u_data_[u_index(i, j, k)]; }
    double u(int i, int j, int k) const { return u_data_[u_index(i, j, k)]; }

    /// Access v component at y-face (i, j+1/2) - 2D backward compatible (uses k=0 plane for flat array compatibility)
    double& v(int i, int j) { return v_data_[v_index(i, j, 0)]; }
    double v(int i, int j) const { return v_data_[v_index(i, j, 0)]; }

    /// Access v component at y-face (i, j+1/2, k) - 3D
    double& v(int i, int j, int k) { return v_data_[v_index(i, j, k)]; }
    double v(int i, int j, int k) const { return v_data_[v_index(i, j, k)]; }

    /// Access w component at z-face (i, j, k+1/2) - 3D only
    double& w(int i, int j, int k) { return w_data_[w_index(i, j, k)]; }
    double w(int i, int j, int k) const { return w_data_[w_index(i, j, k)]; }

    /// Raw data access for GPU and other kernels
    std::vector<double>& u_data() { return u_data_; }
    std::vector<double>& v_data() { return v_data_; }
    std::vector<double>& w_data() { return w_data_; }
    const std::vector<double>& u_data() const { return u_data_; }
    const std::vector<double>& v_data() const { return v_data_; }
    const std::vector<double>& w_data() const { return w_data_; }

    /// Get staggered field dimensions
    int u_stride() const { return u_stride_; }
    int v_stride() const { return v_stride_; }
    int w_stride() const { return w_stride_; }
    int u_plane_stride() const { return u_plane_stride_; }
    int v_plane_stride() const { return v_plane_stride_; }
    int w_plane_stride() const { return w_plane_stride_; }
    int u_total_size() const { return u_data_.size(); }
    int v_total_size() const { return v_data_.size(); }
    int w_total_size() const { return w_data_.size(); }

    /// Fill with constant values
    void fill(double u_val, double v_val, double w_val = 0.0);

    /// Compute velocity magnitude at cell center (i, j) by interpolation - 2D
    double magnitude(int i, int j) const;

    /// Compute velocity magnitude at cell center (i, j, k) by interpolation - 3D
    double magnitude(int i, int j, int k) const;

    /// Max velocity magnitude in interior (at cell centers)
    double max_magnitude() const;

    /// L2 norm of velocity field
    double norm_L2() const;

    /// Get velocity at cell center by interpolation - 2D backward compatible
    double u_center(int i, int j) const;
    double v_center(int i, int j) const;

    /// Get velocity at cell center by interpolation - 3D
    double u_center(int i, int j, int k) const;
    double v_center(int i, int j, int k) const;
    double w_center(int i, int j, int k) const;

    const Mesh* mesh() const { return mesh_; }

    /// Write to file
    void write(const std::string& filename) const;

private:
    const Mesh* mesh_ = nullptr;

    // Staggered storage for 3D (Ng = ghost layers):
    // u at x-faces: ((Nx+1)+2*Ng) × (Ny+2*Ng) × (Nz+2*Ng)
    // v at y-faces: (Nx+2*Ng) × ((Ny+1)+2*Ng) × (Nz+2*Ng)
    // w at z-faces: (Nx+2*Ng) × (Ny+2*Ng) × ((Nz+1)+2*Ng)
    std::vector<double> u_data_;
    std::vector<double> v_data_;
    std::vector<double> w_data_;

    int u_stride_;        // u row stride = Nx+1+2*Ng
    int v_stride_;        // v row stride = Nx+2*Ng
    int w_stride_;        // w row stride = Nx+2*Ng
    int u_plane_stride_;  // u plane stride = u_stride * (Ny+2*Ng)
    int v_plane_stride_;  // v plane stride = v_stride * (Ny+1+2*Ng)
    int w_plane_stride_;  // w plane stride = w_stride * (Ny+2*Ng)

    // Index calculations for staggered locations (3D)
    int u_index(int i, int j, int k) const { return k * u_plane_stride_ + j * u_stride_ + i; }
    int v_index(int i, int j, int k) const { return k * v_plane_stride_ + j * v_stride_ + i; }
    int w_index(int i, int j, int k) const { return k * w_plane_stride_ + j * w_stride_ + i; }
};

/// Symmetric 2D/3D tensor field (for Reynolds stresses, strain rate, etc.)
/// 2D stores: xx, xy, yy (symmetric, so xy=yx)
/// 3D stores: xx, xy, xz, yy, yz, zz (symmetric)
class TensorField {
public:
    TensorField() = default;
    explicit TensorField(const Mesh& mesh);

    /// Access components - 2D backward compatible
    double& xx(int i, int j) { return xx_(i, j); }
    double& xy(int i, int j) { return xy_(i, j); }
    double& yy(int i, int j) { return yy_(i, j); }

    double xx(int i, int j) const { return xx_(i, j); }
    double xy(int i, int j) const { return xy_(i, j); }
    double yy(int i, int j) const { return yy_(i, j); }

    /// Access components - 3D
    double& xx(int i, int j, int k) { return xx_(i, j, k); }
    double& xy(int i, int j, int k) { return xy_(i, j, k); }
    double& xz(int i, int j, int k) { return xz_(i, j, k); }
    double& yy(int i, int j, int k) { return yy_(i, j, k); }
    double& yz(int i, int j, int k) { return yz_(i, j, k); }
    double& zz(int i, int j, int k) { return zz_(i, j, k); }

    double xx(int i, int j, int k) const { return xx_(i, j, k); }
    double xy(int i, int j, int k) const { return xy_(i, j, k); }
    double xz(int i, int j, int k) const { return xz_(i, j, k); }
    double yy(int i, int j, int k) const { return yy_(i, j, k); }
    double yz(int i, int j, int k) const { return yz_(i, j, k); }
    double zz(int i, int j, int k) const { return zz_(i, j, k); }

    /// Get trace - 2D
    double trace(int i, int j) const { return xx_(i, j) + yy_(i, j); }

    /// Get trace - 3D
    double trace(int i, int j, int k) const { return xx_(i, j, k) + yy_(i, j, k) + zz_(i, j, k); }

    /// Get component fields
    ScalarField& xx_field() { return xx_; }
    ScalarField& xy_field() { return xy_; }
    ScalarField& xz_field() { return xz_; }
    ScalarField& yy_field() { return yy_; }
    ScalarField& yz_field() { return yz_; }
    ScalarField& zz_field() { return zz_; }

    const ScalarField& xx_field() const { return xx_; }
    const ScalarField& xy_field() const { return xy_; }
    const ScalarField& xz_field() const { return xz_; }
    const ScalarField& yy_field() const { return yy_; }
    const ScalarField& yz_field() const { return yz_; }
    const ScalarField& zz_field() const { return zz_; }

    /// Get raw data vectors (for GPU mapping)
    std::vector<double>& xx_data() { return xx_.data(); }
    std::vector<double>& xy_data() { return xy_.data(); }
    std::vector<double>& xz_data() { return xz_.data(); }
    std::vector<double>& yy_data() { return yy_.data(); }
    std::vector<double>& yz_data() { return yz_.data(); }
    std::vector<double>& zz_data() { return zz_.data(); }

    const std::vector<double>& xx_data() const { return xx_.data(); }
    const std::vector<double>& xy_data() const { return xy_.data(); }
    const std::vector<double>& xz_data() const { return xz_.data(); }
    const std::vector<double>& yy_data() const { return yy_.data(); }
    const std::vector<double>& yz_data() const { return yz_.data(); }
    const std::vector<double>& zz_data() const { return zz_.data(); }

    const Mesh* mesh() const { return xx_.mesh(); }

private:
    ScalarField xx_, xy_, xz_;
    ScalarField yy_, yz_, zz_;
};

} // namespace nncfd


