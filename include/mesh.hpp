#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace nncfd {

/// Stretching function type for non-uniform grids
using StretchingFunc = std::function<double(double)>;

/// Structured 2D mesh with cell-centered variables
/// Supports ghost cells for boundary condition implementation
struct Mesh {
    int Nx;         ///< Number of interior cells in x
    int Ny;         ///< Number of interior cells in y
    int Nghost;     ///< Number of ghost cell layers (default 1)
    
    double x_min, x_max;
    double y_min, y_max;
    
    // Cell sizes (uniform case)
    double dx, dy;
    
    // Coordinate arrays (cell centers, including ghost cells)
    std::vector<double> xc;  ///< x-coordinates of cell centers
    std::vector<double> yc;  ///< y-coordinates of cell centers
    
    // Face coordinates
    std::vector<double> xf;  ///< x-coordinates of cell faces
    std::vector<double> yf;  ///< y-coordinates of cell faces
    
    // Variable cell sizes for stretched grids
    std::vector<double> dxv; ///< dx for each cell column
    std::vector<double> dyv; ///< dy for each cell row
    
    /// Total number of cells including ghosts
    int total_Nx() const { return Nx + 2 * Nghost; }
    int total_Ny() const { return Ny + 2 * Nghost; }
    int total_cells() const { return total_Nx() * total_Ny(); }
    
    /// Convert (i, j) to flat index (row-major with ghost cells)
    /// i in [0, total_Nx), j in [0, total_Ny)
    int index(int i, int j) const {
        return j * total_Nx() + i;
    }
    
    /// Convert flat index back to (i, j)
    void inv_index(int idx, int& i, int& j) const {
        j = idx / total_Nx();
        i = idx % total_Nx();
    }
    
    /// Check if cell (i, j) is in the interior domain
    bool isInterior(int i, int j) const {
        return i >= Nghost && i < Nx + Nghost &&
               j >= Nghost && j < Ny + Nghost;
    }
    
    /// Check if cell is a ghost cell
    bool isGhost(int i, int j) const {
        return !isInterior(i, j);
    }
    
    /// Interior index ranges
    int i_begin() const { return Nghost; }
    int i_end() const { return Nx + Nghost; }
    int j_begin() const { return Nghost; }
    int j_end() const { return Ny + Nghost; }
    
    /// Get cell center coordinates
    double x(int i) const { return xc[i]; }
    double y(int j) const { return yc[j]; }
    
    /// Get cell size at position
    double dx_at(int i) const { return dxv.empty() ? dx : dxv[i]; }
    double dy_at(int j) const { return dyv.empty() ? dy : dyv[j]; }
    
    /// Wall distance (for channel: distance to nearest wall in y)
    double wall_distance(int i, int j) const;
    
    /// Normalized wall distance y+ (requires friction velocity)
    double y_plus(int i, int j, double u_tau, double nu) const;
    
    /// Initialize uniform grid
    void init_uniform(int nx, int ny, 
                      double xmin, double xmax,
                      double ymin, double ymax,
                      int nghost = 1);
    
    /// Initialize with stretching in y (e.g., for wall-resolved simulations)
    void init_stretched_y(int nx, int ny,
                          double xmin, double xmax,
                          double ymin, double ymax,
                          StretchingFunc stretch,
                          int nghost = 1);
    
    /// Tanh stretching function for wall clustering
    /// Returns a function that maps [0,1] -> [0,1] with clustering near 0 and 1
    static StretchingFunc tanh_stretching(double beta);
    
    /// Create a simple uniform mesh (for multigrid levels)
    static Mesh create_uniform(int nx, int ny, int nghost = 1);
};

} // namespace nncfd


