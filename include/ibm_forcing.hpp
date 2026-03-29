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
/// Integration: call apply_forcing() after predictor, before Poisson solve.
/// The forcing modifies u*, v*, w* to enforce no-slip on the IBM surface.

#include "ibm_geometry.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <vector>
#include <memory>
#include <tuple>
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

    /// Set volume penalization timescale (0 = hard forcing, >0 = smooth)
    void set_penalization_eta(double eta) { penalization_eta_ = eta; }

    /// Enable ghost-cell IBM (Fadlun et al. 2000) for accurate no-slip.
    /// Call recompute_weights() after this to rebuild interpolation stencils.
    void set_ghost_cell_ibm(bool enable) { ghost_cell_ibm_ = enable; }
    bool is_ghost_cell_ibm() const { return ghost_cell_ibm_; }

    // Accessors for testing
    int n_ghost_u() const { return n_ghost_u_; }
    int n_ghost_v() const { return n_ghost_v_; }
    int n_ghost_w() const { return n_ghost_w_; }
    double ghost_alpha_u(int g) const { return ghost_alpha_u_[g]; }
    double ghost_alpha_v(int g) const { return ghost_alpha_v_[g]; }
    size_t weight_u_size() const { return weight_u_.size(); }
    double weight_u(int i) const { return weight_u_[i]; }
    IBMCellType cell_type_u(int i) const { return cell_type_u_[i]; }

    /// Apply ghost-cell no-slip after pressure correction (mirror condition)
    /// Single code path: pragmas ignored on CPU builds
    void apply_ghost_cell(double* u_ptr, double* v_ptr, double* w_ptr);

    /// Recompute weights (call after changing penalization_eta or ghost_cell_ibm)
    void recompute_weights() { compute_weights(); }

    /// Recompute weights and re-upload to GPU (for mid-simulation IBM changes)
    void recompute_and_remap();

    /// Exclude wall-adjacent cells from IBM forcing (set weight=1.0)
    /// Call after classify_cells() + compute_weights() if no-slip BCs overlap with IBM body
    void exclude_wall_cells(bool y_lo_wall, bool y_hi_wall,
                            bool z_lo_wall = false, bool z_hi_wall = false);

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
    /// @param dt     Time step size; when >0 accumulates predictor forces into last_Fx_/Fy_/Fz_
    void apply_forcing_device(double* u_ptr, double* v_ptr, double* w_ptr, double dt = 0.0);

    /// Enable/disable force accumulation during apply_forcing_device.
    /// When disabled (default after construction), the force reductions are skipped
    /// and apply_forcing_device only does the cheap weight multiply.
    /// Enable at output intervals when you need Cd/Cl values.
    void set_accumulate_forces(bool enable) { accumulate_forces_ = enable; }

    /// Query whether force accumulation is enabled
    bool accumulate_forces() const { return accumulate_forces_; }

    /// Zero out Poisson RHS at solid cell centers (GPU, no CPU sync)
    /// @param rhs_ptr  Device-resident RHS pointer (cell-centered)
    void mask_rhs_device(double* rhs_ptr);

    /// Map IBM data to GPU (call after classify_cells)
    void map_to_gpu();

    /// Unmap IBM data from GPU
    void unmap_from_gpu();

    /// Whether GPU data is mapped
    bool is_gpu_ready() const { return gpu_mapped_; }

    /// Reset force accumulator to zero; call once at the start of each time step,
    /// before the first apply_forcing call. Both IBM calls (predictor + corrected
    /// velocity) then ADD their contributions so the total captures pressure drag too.
    void reset_force_accumulator();

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

    // Volume penalization timescale (0 = hard forcing, >0 = semi-implicit)
    double penalization_eta_ = 0.0;

    // Ghost-cell IBM (Fadlun et al. 2000): sparse interpolation at forcing cells
    bool ghost_cell_ibm_ = false;
    std::vector<int>    ghost_self_u_, ghost_self_v_, ghost_self_w_;
    std::vector<int>    ghost_nbr_u_,  ghost_nbr_v_,  ghost_nbr_w_;
    std::vector<double> ghost_alpha_u_, ghost_alpha_v_, ghost_alpha_w_;
    int*    ghost_self_u_ptr_ = nullptr;
    int*    ghost_nbr_u_ptr_ = nullptr;
    double* ghost_alpha_u_ptr_ = nullptr;
    int*    ghost_self_v_ptr_ = nullptr;
    int*    ghost_nbr_v_ptr_ = nullptr;
    double* ghost_alpha_v_ptr_ = nullptr;
    int*    ghost_self_w_ptr_ = nullptr;
    int*    ghost_nbr_w_ptr_ = nullptr;
    double* ghost_alpha_w_ptr_ = nullptr;
    int n_ghost_u_ = 0, n_ghost_v_ = 0, n_ghost_w_ = 0;

    // Second-order ghost-cell (Tseng-Ferziger 2003): image-point bilinear interpolation
    // For each ghost cell: 4 fluid neighbor indices + 4 bilinear weights (2D)
    // u_ghost = -sum(w_k * u[nbr_k]) for no-slip (mirror reflection)
    static constexpr int GC_STENCIL_SIZE = 4;  // 2x2 bilinear in 2D
    std::vector<int>    gc2_nbr_u_;   // [n_ghost_u_ * GC_STENCIL_SIZE] flat
    std::vector<double> gc2_wt_u_;    // [n_ghost_u_ * GC_STENCIL_SIZE] flat
    std::vector<int>    gc2_nbr_v_;
    std::vector<double> gc2_wt_v_;
    std::vector<int>    gc2_nbr_w_;
    std::vector<double> gc2_wt_w_;
    // GPU pointers for second-order arrays
    int*    gc2_nbr_u_ptr_ = nullptr;
    double* gc2_wt_u_ptr_  = nullptr;
    int*    gc2_nbr_v_ptr_ = nullptr;
    double* gc2_wt_v_ptr_  = nullptr;
    int*    gc2_nbr_w_ptr_ = nullptr;
    double* gc2_wt_w_ptr_  = nullptr;

    /// Precompute ghost-cell interpolation stencils
    void compute_ghost_cell_interp();

    /// Precompute second-order ghost-cell stencils (Tseng-Ferziger image points)
    void compute_ghost_cell_interp_2nd();

    // Cell classification arrays (stored at velocity face locations)
    std::vector<IBMCellType> cell_type_u_;
    std::vector<IBMCellType> cell_type_v_;
    std::vector<IBMCellType> cell_type_w_;

    // Pre-computed weight arrays for GPU forcing (0=solid, weight for forcing, 1=fluid)
    std::vector<double> weight_u_;
    std::vector<double> weight_v_;
    std::vector<double> weight_w_;

    // Cell-centered solid mask (0=solid/forcing inside body, 1=fluid)
    std::vector<double> solid_mask_cell_;
    double* solid_mask_cell_ptr_ = nullptr;
    size_t cell_total_ = 0;

    // Raw pointers for GPU mapping
    double* weight_u_ptr_ = nullptr;
    double* weight_v_ptr_ = nullptr;
    double* weight_w_ptr_ = nullptr;
    size_t u_total_ = 0, v_total_ = 0, w_total_ = 0;
    bool gpu_mapped_ = false;
    bool accumulate_forces_ = false;

    int n_forcing_ = 0;
    int n_solid_ = 0;

    // Cached force on body from last apply_forcing / apply_forcing_device call with dt > 0
    double last_Fx_ = 0.0;
    double last_Fy_ = 0.0;
    double last_Fz_ = 0.0;
    double current_dt_ = 0.0;  // stored for post-correction force accumulation

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
