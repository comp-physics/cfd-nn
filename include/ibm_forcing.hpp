#pragma once

/// @file ibm_forcing.hpp
/// @brief Direct-forcing immersed boundary method
///
/// Classifies cells as Fluid, Solid, or Forcing (interface) based on the signed
/// distance function. Forcing cells receive a direct-forcing term that drives
/// velocity to zero (no-slip) in a single time step.
///
/// Cell classification:
///   - Fluid: phi > 0 (outside body)
///   - Solid: phi < -band (deep inside body, set u=0 directly)
///   - Forcing: -band <= phi <= 0 (near-surface, apply interpolated forcing)
///
/// Integration: call compute_forcing() after predictor, before Poisson solve.
/// The forcing modifies u*, v*, w* to enforce no-slip on the IBM surface.

#include "ibm_geometry.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <vector>
#include <memory>
#include <cstddef>

namespace nncfd {

/// Cell type classification for IBM
enum class IBMCellType : int {
    Fluid = 0,    ///< Normal fluid cell
    Forcing = 1,  ///< Interface cell (receives direct forcing)
    Solid = 2     ///< Deep inside body (velocity set to zero)
};

/// IBM forcing manager
class IBMForcing {
public:
    /// @param mesh  Computational mesh
    /// @param body  Immersed body geometry (ownership shared)
    IBMForcing(const Mesh& mesh, std::shared_ptr<IBMBody> body);

    /// Classify all velocity face locations as Fluid/Forcing/Solid
    /// Call once at initialization or when body moves
    void classify_cells();

    /// Compute and apply direct forcing to predicted velocity
    /// GPU-accelerated: uses pre-computed weight arrays on device
    /// @param vel Velocity field (modified in place)
    /// @param dt  Time step size
    void apply_forcing(VectorField& vel, double dt);

    /// Apply forcing directly to GPU-resident velocity data (no CPU sync)
    /// @param u_ptr  Device-resident u-velocity pointer
    /// @param v_ptr  Device-resident v-velocity pointer
    /// @param w_ptr  Device-resident w-velocity pointer (nullptr for 2D)
    void apply_forcing_device(double* u_ptr, double* v_ptr, double* w_ptr);

    /// Map IBM data to GPU (call after classify_cells)
    void map_to_gpu();

    /// Unmap IBM data from GPU
    void unmap_from_gpu();

    /// Whether GPU data is mapped
    bool is_gpu_ready() const { return gpu_mapped_; }

    /// Compute drag and lift forces on the body
    /// @param vel  Current velocity field
    /// @param dt   Time step size
    /// @return {Fx, Fy, Fz} force components
    std::tuple<double, double, double> compute_forces(const VectorField& vel, double dt) const;

    /// Get cell type at u-face (i,j,k)
    IBMCellType cell_type_u(int i, int j, int k = 0) const;

    /// Get cell type at v-face (i,j,k)
    IBMCellType cell_type_v(int i, int j, int k = 0) const;

    /// Get cell type at w-face (i,j,k) (3D only)
    IBMCellType cell_type_w(int i, int j, int k) const;

    /// Number of forcing cells (for diagnostics)
    int num_forcing_cells() const { return n_forcing_; }

    /// Number of solid cells
    int num_solid_cells() const { return n_solid_; }

    /// Get the body
    const IBMBody& body() const { return *body_; }

private:
    const Mesh* mesh_;
    std::shared_ptr<IBMBody> body_;

    // Band width for forcing region (cells with -band < phi < 0)
    double band_width_;

    // Cell classification arrays (stored at velocity face locations)
    std::vector<IBMCellType> cell_type_u_;
    std::vector<IBMCellType> cell_type_v_;
    std::vector<IBMCellType> cell_type_w_;

    // Pre-computed weight arrays for GPU forcing (0=solid, weight for forcing, 1=fluid)
    std::vector<double> weight_u_;
    std::vector<double> weight_v_;
    std::vector<double> weight_w_;

    // Raw pointers for GPU mapping
    double* weight_u_ptr_ = nullptr;
    double* weight_v_ptr_ = nullptr;
    double* weight_w_ptr_ = nullptr;
    size_t u_total_ = 0, v_total_ = 0, w_total_ = 0;
    bool gpu_mapped_ = false;

    int n_forcing_ = 0;
    int n_solid_ = 0;

    // Classify a single point
    IBMCellType classify_point(double phi) const;

    // Compute weight arrays from cell types and geometry
    void compute_weights();

    // Strides for indexing
    int u_stride_, u_plane_stride_;
    int v_stride_, v_plane_stride_;
    int w_stride_, w_plane_stride_;
};

} // namespace nncfd
