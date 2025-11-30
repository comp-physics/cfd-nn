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

/// 2D Vector field (u, v components)
class VectorField {
public:
    VectorField() = default;
    explicit VectorField(const Mesh& mesh, double init_u = 0.0, double init_v = 0.0);
    
    /// Access u component
    double& u(int i, int j) { return u_(i, j); }
    double u(int i, int j) const { return u_(i, j); }
    
    /// Access v component
    double& v(int i, int j) { return v_(i, j); }
    double v(int i, int j) const { return v_(i, j); }
    
    /// Get component fields
    ScalarField& u_field() { return u_; }
    ScalarField& v_field() { return v_; }
    const ScalarField& u_field() const { return u_; }
    const ScalarField& v_field() const { return v_; }
    
    /// Fill with constant values
    void fill(double u_val, double v_val);
    
    /// Compute velocity magnitude at (i, j)
    double magnitude(int i, int j) const {
        double uu = u_(i, j);
        double vv = v_(i, j);
        return std::sqrt(uu * uu + vv * vv);
    }
    
    /// Max velocity magnitude in interior
    double max_magnitude() const;
    
    /// L2 norm of velocity field
    double norm_L2() const;
    
    const Mesh* mesh() const { return u_.mesh(); }
    
    /// Write to file
    void write(const std::string& filename) const;
    
private:
    ScalarField u_, v_;
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


