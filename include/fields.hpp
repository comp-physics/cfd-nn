#pragma once

#include "mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>

namespace nncfd {

/// Scalar field on structured mesh
class ScalarField {
public:
    ScalarField() = default;
    explicit ScalarField(const Mesh& mesh, double init_val = 0.0);
    
    /// Access by (i, j)
    double& operator()(int i, int j) { return data_[mesh_->index(i, j)]; }
    double operator()(int i, int j) const { return data_[mesh_->index(i, j)]; }
    
    /// Access by flat index
    double& operator[](int idx) { return data_[idx]; }
    double operator[](int idx) const { return data_[idx]; }
    
    /// Fill with constant value
    void fill(double val);
    
    /// Get min/max values in interior
    double max_interior() const;
    double min_interior() const;
    
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

/// 2D Vector field (u, v components) on staggered MAC grid
/// u is stored at x-faces (i+1/2, j), v at y-faces (i, j+1/2)
class VectorField {
public:
    VectorField() = default;
    explicit VectorField(const Mesh& mesh, double init_u = 0.0, double init_v = 0.0);
    
    /// Access u component at x-face (i+1/2, j)
    /// For interior: i in [Ng, Ng+Nx], j in [Ng, Ng+Ny-1]
    double& u(int i, int j) { return u_data_[u_index(i, j)]; }
    double u(int i, int j) const { return u_data_[u_index(i, j)]; }
    
    /// Access v component at y-face (i, j+1/2)
    /// For interior: i in [Ng, Ng+Nx-1], j in [Ng, Ng+Ny]
    double& v(int i, int j) { return v_data_[v_index(i, j)]; }
    double v(int i, int j) const { return v_data_[v_index(i, j)]; }
    
    /// Raw data access for GPU and other kernels
    std::vector<double>& u_data() { return u_data_; }
    std::vector<double>& v_data() { return v_data_; }
    const std::vector<double>& u_data() const { return u_data_; }
    const std::vector<double>& v_data() const { return v_data_; }
    
    /// Get staggered field dimensions
    int u_stride() const { return u_stride_; }
    int v_stride() const { return v_stride_; }
    int u_total_size() const { return u_data_.size(); }
    int v_total_size() const { return v_data_.size(); }
    
    /// Fill with constant values
    void fill(double u_val, double v_val);
    
    /// Compute velocity magnitude at cell center (i, j) by interpolation
    double magnitude(int i, int j) const;
    
    /// Max velocity magnitude in interior (at cell centers)
    double max_magnitude() const;
    
    /// L2 norm of velocity field
    double norm_L2() const;
    
    /// Get velocity at cell center by interpolation
    double u_center(int i, int j) const;
    double v_center(int i, int j) const;
    
    const Mesh* mesh() const { return mesh_; }
    
    /// Write to file
    void write(const std::string& filename) const;
    
private:
    const Mesh* mesh_ = nullptr;
    
    // Staggered storage:
    // u at x-faces: (Nx+1+2*Ng) × (Ny+2*Ng)
    // v at y-faces: (Nx+2*Ng) × (Ny+1+2*Ng)
    std::vector<double> u_data_;
    std::vector<double> v_data_;
    
    int u_stride_;  // u row stride = Nx+1+2*Ng
    int v_stride_;  // v row stride = Nx+2*Ng
    
    // Index calculations for staggered locations
    int u_index(int i, int j) const { return j * u_stride_ + i; }
    int v_index(int i, int j) const { return j * v_stride_ + i; }
};

/// Symmetric 2D tensor field (for Reynolds stresses, strain rate, etc.)
/// Stores: [0]=xx, [1]=xy, [2]=yy (symmetric, so xy=yx)
class TensorField {
public:
    TensorField() = default;
    explicit TensorField(const Mesh& mesh);
    
    /// Access components
    double& xx(int i, int j) { return xx_(i, j); }
    double& xy(int i, int j) { return xy_(i, j); }
    double& yy(int i, int j) { return yy_(i, j); }
    
    double xx(int i, int j) const { return xx_(i, j); }
    double xy(int i, int j) const { return xy_(i, j); }
    double yy(int i, int j) const { return yy_(i, j); }
    
    /// Get trace
    double trace(int i, int j) const { return xx_(i, j) + yy_(i, j); }
    
    /// Get component fields
    ScalarField& xx_field() { return xx_; }
    ScalarField& xy_field() { return xy_; }
    ScalarField& yy_field() { return yy_; }
    
    const ScalarField& xx_field() const { return xx_; }
    const ScalarField& xy_field() const { return xy_; }
    const ScalarField& yy_field() const { return yy_; }
    
    /// Fill all components
    void fill(double xx_val, double xy_val, double yy_val);
    
    const Mesh* mesh() const { return xx_.mesh(); }
    
private:
    ScalarField xx_, xy_, yy_;
};

} // namespace nncfd


