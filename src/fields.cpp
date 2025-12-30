#include "fields.hpp"
#include <cmath>
#include <limits>

namespace nncfd {

// ScalarField implementation
// Note: For 2D meshes, data is stored at k=0 plane for backward compatibility.
// All interior loop methods use k_start=0, k_stop=1 for 2D, and k_begin/k_end for 3D.

ScalarField::ScalarField(const Mesh& mesh, double init_val)
    : mesh_(&mesh), data_(mesh.total_cells(), init_val) {}

void ScalarField::fill(double val) {
    std::fill(data_.begin(), data_.end(), val);
}

double ScalarField::max_interior() const {
    double max_val = -std::numeric_limits<double>::max();
    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop  = mesh_->is2D() ? 1 : mesh_->k_end();
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                max_val = std::max(max_val, (*this)(i, j, k));
            }
        }
    }
    return max_val;
}

double ScalarField::norm_L2() const {
    double sum = 0.0;
    int count = 0;
    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop  = mesh_->is2D() ? 1 : mesh_->k_end();
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double val = (*this)(i, j, k);
                sum += val * val;
                ++count;
            }
        }
    }
    return (count > 0) ? std::sqrt(sum / count) : 0.0;
}

double ScalarField::norm_Linf() const {
    double max_abs = 0.0;
    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop  = mesh_->is2D() ? 1 : mesh_->k_end();
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                max_abs = std::max(max_abs, std::abs((*this)(i, j, k)));
            }
        }
    }
    return max_abs;
}

void ScalarField::write(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << "# Nx=" << mesh_->Nx << " Ny=" << mesh_->Ny << " Nz=" << mesh_->Nz << "\n";
    file << "# x y z value\n";

    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop  = mesh_->is2D() ? 1 : mesh_->k_end();

    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << mesh_->x(i) << " " << mesh_->y(j) << " " << mesh_->z(k) << " "
                     << (*this)(i, j, k) << "\n";
            }
            file << "\n";  // Blank line between rows (for gnuplot splot)
        }
        if (!mesh_->is2D()) {
            file << "\n";  // Extra blank line between z-planes
        }
    }
}

// VectorField implementation (staggered MAC grid)

VectorField::VectorField(const Mesh& mesh, double init_u, double init_v, double init_w)
    : mesh_(&mesh)
{
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;

    // u at x-faces: (Nx+1+2*Ng) × (Ny+2*Ng) × (Nz+2*Ng)
    u_stride_ = Nx + 1 + 2 * Ng;
    u_plane_stride_ = u_stride_ * (Ny + 2 * Ng);
    const int u_total = u_plane_stride_ * (Nz + 2 * Ng);
    u_data_.resize(u_total, init_u);

    // v at y-faces: (Nx+2*Ng) × (Ny+1+2*Ng) × (Nz+2*Ng)
    v_stride_ = Nx + 2 * Ng;
    v_plane_stride_ = v_stride_ * (Ny + 1 + 2 * Ng);
    const int v_total = v_plane_stride_ * (Nz + 2 * Ng);
    v_data_.resize(v_total, init_v);

    // w at z-faces: (Nx+2*Ng) × (Ny+2*Ng) × (Nz+1+2*Ng)
    w_stride_ = Nx + 2 * Ng;
    w_plane_stride_ = w_stride_ * (Ny + 2 * Ng);
    const int w_total = w_plane_stride_ * (Nz + 1 + 2 * Ng);
    w_data_.resize(w_total, init_w);
}

void VectorField::fill(double u_val, double v_val, double w_val) {
    std::fill(u_data_.begin(), u_data_.end(), u_val);
    std::fill(v_data_.begin(), v_data_.end(), v_val);
    std::fill(w_data_.begin(), w_data_.end(), w_val);
}

double VectorField::u_center(int i, int j) const {
    // 2D backward compatible - use k = 0 to match 2D u(i,j) accessor
    return u_center(i, j, 0);
}

double VectorField::v_center(int i, int j) const {
    // 2D backward compatible - use k = 0 to match 2D v(i,j) accessor
    return v_center(i, j, 0);
}

double VectorField::u_center(int i, int j, int k) const {
    // Interpolate from x-faces to cell center
    return 0.5 * (u(i, j, k) + u(i + 1, j, k));
}

double VectorField::v_center(int i, int j, int k) const {
    // Interpolate from y-faces to cell center
    return 0.5 * (v(i, j, k) + v(i, j + 1, k));
}

double VectorField::w_center(int i, int j, int k) const {
    // Interpolate from z-faces to cell center
    return 0.5 * (w(i, j, k) + w(i, j, k + 1));
}

double VectorField::magnitude(int i, int j) const {
    // 2D backward compatible
    double uu = u_center(i, j);
    double vv = v_center(i, j);
    return std::sqrt(uu * uu + vv * vv);
}

double VectorField::magnitude(int i, int j, int k) const {
    double uu = u_center(i, j, k);
    double vv = v_center(i, j, k);
    double ww = w_center(i, j, k);
    return std::sqrt(uu * uu + vv * vv + ww * ww);
}

double VectorField::max_magnitude() const {
    double max_mag = 0.0;
    // For 2D, data lives at k=0 plane for backward compatibility
    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop = mesh_->is2D() ? 1 : mesh_->k_end();
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                max_mag = std::max(max_mag, magnitude(i, j, k));
            }
        }
    }
    return max_mag;
}

double VectorField::norm_L2() const {
    double sum = 0.0;
    int count = 0;
    // For 2D, data lives at k=0 plane for backward compatibility
    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop = mesh_->is2D() ? 1 : mesh_->k_end();
    // Compute at cell centers for consistency
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double uu = u_center(i, j, k);
                double vv = v_center(i, j, k);
                double ww = w_center(i, j, k);
                sum += uu * uu + vv * vv + ww * ww;
                ++count;
            }
        }
    }
    return (count > 0) ? std::sqrt(sum / count) : 0.0;
}

void VectorField::write(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << "# Nx=" << mesh_->Nx << " Ny=" << mesh_->Ny << " Nz=" << mesh_->Nz << "\n";
    file << "# x y z u v w (interpolated to cell centers)\n";

    const int k_start = mesh_->is2D() ? 0 : mesh_->k_begin();
    const int k_stop  = mesh_->is2D() ? 1 : mesh_->k_end();

    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << mesh_->x(i) << " " << mesh_->y(j) << " " << mesh_->z(k) << " "
                     << u_center(i, j, k) << " " << v_center(i, j, k) << " "
                     << w_center(i, j, k) << "\n";
            }
            file << "\n";
        }
        if (!mesh_->is2D()) {
            file << "\n";
        }
    }
}

// TensorField implementation

TensorField::TensorField(const Mesh& mesh)
    : xx_(mesh), xy_(mesh), xz_(mesh),
      yy_(mesh), yz_(mesh), zz_(mesh) {}

} // namespace nncfd


