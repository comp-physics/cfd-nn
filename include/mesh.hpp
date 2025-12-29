#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace nncfd {

/// Stretching function type for non-uniform grids
using StretchingFunc = std::function<double(double)>;

/// Structured 2D/3D mesh with cell-centered variables
/// Supports ghost cells for boundary condition implementation
/// For 2D backward compatibility, set Nz=1
struct Mesh {
    int Nx;         ///< Number of interior cells in x
    int Ny;         ///< Number of interior cells in y
    int Nz = 1;     ///< Number of interior cells in z (default 1 for 2D)
    int Nghost;     ///< Number of ghost cell layers (default 1)

    double x_min, x_max;
    double y_min, y_max;
    double z_min = 0.0, z_max = 1.0;

    // Cell sizes (uniform case)
    double dx, dy;
    double dz = 1.0;

    // Coordinate arrays (cell centers, including ghost cells)
    std::vector<double> xc;  ///< x-coordinates of cell centers
    std::vector<double> yc;  ///< y-coordinates of cell centers
    std::vector<double> zc;  ///< z-coordinates of cell centers

    // Face coordinates
    std::vector<double> xf;  ///< x-coordinates of cell faces
    std::vector<double> yf;  ///< y-coordinates of cell faces
    std::vector<double> zf;  ///< z-coordinates of cell faces

    // Variable cell sizes for stretched grids
    std::vector<double> dxv; ///< dx for each cell column
    std::vector<double> dyv; ///< dy for each cell row
    std::vector<double> dzv; ///< dz for each cell layer

    /// Total number of cells including ghosts
    int total_Nx() const { return Nx + 2 * Nghost; }
    int total_Ny() const { return Ny + 2 * Nghost; }
    int total_Nz() const { return Nz + 2 * Nghost; }
    int total_cells() const { return total_Nx() * total_Ny() * total_Nz(); }

    /// Check if this is a 2D mesh (Nz == 1)
    bool is2D() const { return Nz == 1; }

    /// Convert (i, j) to flat index for 2D (backward compatible)
    /// i in [0, total_Nx), j in [0, total_Ny)
    int index(int i, int j) const {
        return j * total_Nx() + i;
    }

    /// Convert (i, j, k) to flat index for 3D
    /// k-major ordering: idx = k * (Nx_tot * Ny_tot) + j * Nx_tot + i
    int index(int i, int j, int k) const {
        return k * total_Nx() * total_Ny() + j * total_Nx() + i;
    }

    /// Convert flat index back to (i, j) for 2D
    void inv_index(int idx, int& i, int& j) const {
        j = idx / total_Nx();
        i = idx % total_Nx();
    }

    /// Convert flat index back to (i, j, k) for 3D
    void inv_index(int idx, int& i, int& j, int& k) const {
        int plane_size = total_Nx() * total_Ny();
        k = idx / plane_size;
        int remainder = idx % plane_size;
        j = remainder / total_Nx();
        i = remainder % total_Nx();
    }

    /// Check if cell (i, j) is in the interior domain (2D)
    bool isInterior(int i, int j) const {
        return i >= Nghost && i < Nx + Nghost &&
               j >= Nghost && j < Ny + Nghost;
    }

    /// Check if cell (i, j, k) is in the interior domain (3D)
    bool isInterior(int i, int j, int k) const {
        return i >= Nghost && i < Nx + Nghost &&
               j >= Nghost && j < Ny + Nghost &&
               k >= Nghost && k < Nz + Nghost;
    }

    /// Check if cell is a ghost cell (2D)
    bool isGhost(int i, int j) const {
        return !isInterior(i, j);
    }

    /// Check if cell is a ghost cell (3D)
    bool isGhost(int i, int j, int k) const {
        return !isInterior(i, j, k);
    }

    /// Interior index ranges
    int i_begin() const { return Nghost; }
    int i_end() const { return Nx + Nghost; }
    int j_begin() const { return Nghost; }
    int j_end() const { return Ny + Nghost; }
    int k_begin() const { return Nghost; }
    int k_end() const { return Nz + Nghost; }

    /// Get cell center coordinates
    double x(int i) const { return xc[i]; }
    double y(int j) const { return yc[j]; }
    double z(int k) const { return zc[k]; }

    /// Get cell size at position
    double dx_at(int i) const { return dxv.empty() ? dx : dxv[i]; }
    double dy_at(int j) const { return dyv.empty() ? dy : dyv[j]; }
    double dz_at(int k) const { return dzv.empty() ? dz : dzv[k]; }
    
    /// Wall distance (for channel: distance to nearest wall in y)
    double wall_distance(int i, int j) const;
    double wall_distance(int i, int j, int k) const;

    /// Normalized wall distance y+ (requires friction velocity)
    double y_plus(int i, int j, double u_tau, double nu) const;
    double y_plus(int i, int j, int k, double u_tau, double nu) const;

    /// Initialize uniform 2D grid (backward compatible)
    void init_uniform(int nx, int ny,
                      double xmin, double xmax,
                      double ymin, double ymax,
                      int nghost = 1);

    /// Initialize uniform 3D grid
    void init_uniform(int nx, int ny, int nz,
                      double xmin, double xmax,
                      double ymin, double ymax,
                      double zmin, double zmax,
                      int nghost = 1);

    /// Initialize with stretching in y (e.g., for wall-resolved simulations)
    void init_stretched_y(int nx, int ny,
                          double xmin, double xmax,
                          double ymin, double ymax,
                          StretchingFunc stretch,
                          int nghost = 1);

    /// Initialize 3D with stretching in y
    void init_stretched_y(int nx, int ny, int nz,
                          double xmin, double xmax,
                          double ymin, double ymax,
                          double zmin, double zmax,
                          StretchingFunc stretch,
                          int nghost = 1);

    /// Tanh stretching function for wall clustering
    /// Returns a function that maps [0,1] -> [0,1] with clustering near 0 and 1
    static StretchingFunc tanh_stretching(double beta);

    /// Create a simple uniform mesh (for multigrid levels)
    static Mesh create_uniform(int nx, int ny, int nghost = 1);
    static Mesh create_uniform(int nx, int ny, int nz, int nghost = 1);
};

} // namespace nncfd


