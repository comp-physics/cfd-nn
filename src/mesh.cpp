#include "mesh.hpp"
#include <algorithm>

namespace nncfd {

double Mesh::wall_distance(int i, int j) const {
    (void)i;  // Unused for channel flow
    double y = yc[j];
    double dist_lo = std::abs(y - y_min);
    double dist_hi = std::abs(y - y_max);
    return std::min(dist_lo, dist_hi);
}

double Mesh::y_plus(int i, int j, double u_tau, double nu) const {
    return wall_distance(i, j) * u_tau / nu;
}

void Mesh::init_uniform(int nx, int ny, 
                        double xmin, double xmax,
                        double ymin, double ymax,
                        int nghost) {
    Nx = nx;
    Ny = ny;
    Nghost = nghost;
    x_min = xmin;
    x_max = xmax;
    y_min = ymin;
    y_max = ymax;
    
    dx = (x_max - x_min) / Nx;
    dy = (y_max - y_min) / Ny;
    
    int total_nx = total_Nx();
    int total_ny = total_Ny();
    
    // Cell centers (including ghost cells)
    xc.resize(total_nx);
    yc.resize(total_ny);
    
    for (int i = 0; i < total_nx; ++i) {
        // Ghost cells extend beyond domain
        xc[i] = x_min + (i - Nghost + 0.5) * dx;
    }
    
    for (int j = 0; j < total_ny; ++j) {
        yc[j] = y_min + (j - Nghost + 0.5) * dy;
    }
    
    // Face coordinates
    xf.resize(total_nx + 1);
    yf.resize(total_ny + 1);
    
    for (int i = 0; i <= total_nx; ++i) {
        xf[i] = x_min + (i - Nghost) * dx;
    }
    
    for (int j = 0; j <= total_ny; ++j) {
        yf[j] = y_min + (j - Nghost) * dy;
    }
    
    // Clear stretching vectors (uniform grid)
    dxv.clear();
    dyv.clear();
}

void Mesh::init_stretched_y(int nx, int ny,
                            double xmin, double xmax,
                            double ymin, double ymax,
                            StretchingFunc stretch,
                            int nghost) {
    Nx = nx;
    Ny = ny;
    Nghost = nghost;
    x_min = xmin;
    x_max = xmax;
    y_min = ymin;
    y_max = ymax;
    
    // Uniform in x
    dx = (x_max - x_min) / Nx;
    dy = (y_max - y_min) / Ny;  // Average dy (not used for stretched)
    
    int total_nx = total_Nx();
    int total_ny = total_Ny();
    
    // x coordinates (uniform)
    xc.resize(total_nx);
    xf.resize(total_nx + 1);
    
    for (int i = 0; i < total_nx; ++i) {
        xc[i] = x_min + (i - Nghost + 0.5) * dx;
    }
    for (int i = 0; i <= total_nx; ++i) {
        xf[i] = x_min + (i - Nghost) * dx;
    }
    
    // y coordinates (stretched)
    // First compute face locations using stretching function
    yf.resize(total_ny + 1);
    double Ly = y_max - y_min;
    
    for (int j = 0; j <= total_ny; ++j) {
        double eta = static_cast<double>(j - Nghost) / Ny;  // [0, 1] for interior
        // Clamp eta for ghost cells
        if (eta < 0) {
            // Extrapolate below domain
            double eta0 = 0.0;
            double eta1 = 1.0 / Ny;
            double y0 = y_min + stretch(eta0) * Ly;
            double y1 = y_min + stretch(eta1) * Ly;
            double dy0 = y1 - y0;
            yf[j] = y0 + eta * Ny * dy0;
        } else if (eta > 1) {
            // Extrapolate above domain
            double eta0 = 1.0 - 1.0 / Ny;
            double eta1 = 1.0;
            double y0 = y_min + stretch(eta0) * Ly;
            double y1 = y_min + stretch(eta1) * Ly;
            double dy1 = y1 - y0;
            yf[j] = y1 + (eta - 1.0) * Ny * dy1;
        } else {
            yf[j] = y_min + stretch(eta) * Ly;
        }
    }
    
    // Cell centers and variable dy
    yc.resize(total_ny);
    dyv.resize(total_ny);
    
    for (int j = 0; j < total_ny; ++j) {
        yc[j] = 0.5 * (yf[j] + yf[j + 1]);
        dyv[j] = yf[j + 1] - yf[j];
    }
    
    dxv.clear();  // Uniform in x
}

StretchingFunc Mesh::tanh_stretching(double beta) {
    return [beta](double eta) -> double {
        // Maps [0, 1] -> [0, 1] with clustering near 0 and 1
        // Uses symmetric tanh distribution
        return 0.5 * (1.0 + std::tanh(beta * (2.0 * eta - 1.0)) / std::tanh(beta));
    };
}

Mesh Mesh::create_uniform(int nx, int ny, int nghost) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, 0.0, 1.0, 0.0, 1.0, nghost);
    return mesh;
}

} // namespace nncfd


