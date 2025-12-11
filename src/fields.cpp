#include "fields.hpp"
#include <cmath>
#include <limits>

namespace nncfd {

// ScalarField implementation

ScalarField::ScalarField(const Mesh& mesh, double init_val)
    : mesh_(&mesh), data_(mesh.total_cells(), init_val) {}

void ScalarField::fill(double val) {
    std::fill(data_.begin(), data_.end(), val);
}

double ScalarField::max_interior() const {
    double max_val = -std::numeric_limits<double>::max();
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            max_val = std::max(max_val, (*this)(i, j));
        }
    }
    return max_val;
}

double ScalarField::min_interior() const {
    double min_val = std::numeric_limits<double>::max();
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            min_val = std::min(min_val, (*this)(i, j));
        }
    }
    return min_val;
}

double ScalarField::norm_L2() const {
    double sum = 0.0;
    int count = 0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double val = (*this)(i, j);
            sum += val * val;
            ++count;
        }
    }
    return std::sqrt(sum / count);
}

double ScalarField::norm_Linf() const {
    double max_abs = 0.0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            max_abs = std::max(max_abs, std::abs((*this)(i, j)));
        }
    }
    return max_abs;
}

void ScalarField::write(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << "# Nx=" << mesh_->Nx << " Ny=" << mesh_->Ny << "\n";
    file << "# x y value\n";
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << mesh_->x(i) << " " << mesh_->y(j) << " " << (*this)(i, j) << "\n";
        }
        file << "\n";  // Blank line between rows (for gnuplot splot)
    }
}

// VectorField implementation (staggered MAC grid)

VectorField::VectorField(const Mesh& mesh, double init_u, double init_v)
    : mesh_(&mesh)
{
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;
    
    // u at x-faces: (Nx+1+2*Ng) × (Ny+2*Ng)
    u_stride_ = Nx + 1 + 2 * Ng;
    const int u_total = u_stride_ * (Ny + 2 * Ng);
    u_data_.resize(u_total, init_u);
    
    // v at y-faces: (Nx+2*Ng) × (Ny+1+2*Ng)
    v_stride_ = Nx + 2 * Ng;
    const int v_total = v_stride_ * (Ny + 1 + 2 * Ng);
    v_data_.resize(v_total, init_v);
}

void VectorField::fill(double u_val, double v_val) {
    std::fill(u_data_.begin(), u_data_.end(), u_val);
    std::fill(v_data_.begin(), v_data_.end(), v_val);
}

double VectorField::u_center(int i, int j) const {
    // Interpolate from x-faces to cell center
    // u(i,j) at center ≈ 0.5 * (u_face(i,j) + u_face(i+1,j))
    return 0.5 * (u(i, j) + u(i + 1, j));
}

double VectorField::v_center(int i, int j) const {
    // Interpolate from y-faces to cell center
    // v(i,j) at center ≈ 0.5 * (v_face(i,j) + v_face(i,j+1))
    return 0.5 * (v(i, j) + v(i, j + 1));
}

double VectorField::magnitude(int i, int j) const {
    double uu = u_center(i, j);
    double vv = v_center(i, j);
    return std::sqrt(uu * uu + vv * vv);
}

double VectorField::max_magnitude() const {
    double max_mag = 0.0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            max_mag = std::max(max_mag, magnitude(i, j));
        }
    }
    return max_mag;
}

double VectorField::norm_L2() const {
    double sum = 0.0;
    int count = 0;
    // Compute at cell centers for consistency
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double uu = u_center(i, j);
            double vv = v_center(i, j);
            sum += uu * uu + vv * vv;
            ++count;
        }
    }
    return std::sqrt(sum / count);
}

void VectorField::write(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << "# Nx=" << mesh_->Nx << " Ny=" << mesh_->Ny << "\n";
    file << "# x y u v (interpolated to cell centers)\n";
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << mesh_->x(i) << " " << mesh_->y(j) << " " 
                 << u_center(i, j) << " " << v_center(i, j) << "\n";
        }
        file << "\n";
    }
}

// TensorField implementation

TensorField::TensorField(const Mesh& mesh)
    : xx_(mesh), xy_(mesh), yy_(mesh) {}

void TensorField::fill(double xx_val, double xy_val, double yy_val) {
    xx_.fill(xx_val);
    xy_.fill(xy_val);
    yy_.fill(yy_val);
}

} // namespace nncfd


