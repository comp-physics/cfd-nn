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

// VectorField implementation

VectorField::VectorField(const Mesh& mesh, double init_u, double init_v)
    : u_(mesh, init_u), v_(mesh, init_v) {}

void VectorField::fill(double u_val, double v_val) {
    u_.fill(u_val);
    v_.fill(v_val);
}

double VectorField::max_magnitude() const {
    double max_mag = 0.0;
    const Mesh* mesh = u_.mesh();
    for (int j = mesh->j_begin(); j < mesh->j_end(); ++j) {
        for (int i = mesh->i_begin(); i < mesh->i_end(); ++i) {
            max_mag = std::max(max_mag, magnitude(i, j));
        }
    }
    return max_mag;
}

double VectorField::norm_L2() const {
    double sum = 0.0;
    int count = 0;
    const Mesh* mesh = u_.mesh();
    for (int j = mesh->j_begin(); j < mesh->j_end(); ++j) {
        for (int i = mesh->i_begin(); i < mesh->i_end(); ++i) {
            double uu = u_(i, j);
            double vv = v_(i, j);
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
    
    const Mesh* mesh = u_.mesh();
    file << "# Nx=" << mesh->Nx << " Ny=" << mesh->Ny << "\n";
    file << "# x y u v\n";
    
    for (int j = mesh->j_begin(); j < mesh->j_end(); ++j) {
        for (int i = mesh->i_begin(); i < mesh->i_end(); ++i) {
            file << mesh->x(i) << " " << mesh->y(j) << " " 
                 << u_(i, j) << " " << v_(i, j) << "\n";
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


