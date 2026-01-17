#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#ifdef USE_FFT_POISSON
#include "poisson_solver_fft.hpp"
#include "poisson_solver_fft1d.hpp"
#include "poisson_solver_fft2d.hpp"
#endif
#include "turbulence_model.hpp"
#include "config.hpp"
#include <memory>
#include <functional>

namespace nncfd {

/// Selector for which velocity field to use in unified GPU/CPU routines
enum class VelocityWhich {
    Current,  ///< Current velocity (velocity_)
    Star,     ///< Predictor velocity (velocity_star_)
    Old       ///< Previous time step (velocity_old_)
};

/// Helper struct for velocity pointer pairs from SolverDeviceView
struct FaceVelPtrs {
    const double* u;
    const double* v;
    const double* w;
    int u_stride;
    int v_stride;
    int w_stride;
    int u_plane_stride;
    int v_plane_stride;
    int w_plane_stride;
};

/// Select velocity pointers from SolverDeviceView based on which field
inline FaceVelPtrs select_face_velocity(const SolverDeviceView& v, VelocityWhich which) {
    switch (which) {
    case VelocityWhich::Current: return {v.u_face, v.v_face, v.w_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    case VelocityWhich::Star:    return {v.u_star_face, v.v_star_face, v.w_star_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    case VelocityWhich::Old:     return {v.u_old_face, v.v_old_face, v.w_old_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    }
    return {nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0};
}

/// Boundary condition specification for velocity
struct VelocityBC {
    enum Type { NoSlip, Periodic, Inflow, Outflow };
    Type x_lo = Periodic;
    Type x_hi = Periodic;
    Type y_lo = NoSlip;
    Type y_hi = NoSlip;
    Type z_lo = Periodic;  // Default periodic for spanwise direction
    Type z_hi = Periodic;

    // Inflow values (if applicable) - 2D backward compatible
    std::function<double(double y)> u_inflow = [](double) { return 0.0; };
    std::function<double(double y)> v_inflow = [](double) { return 0.0; };

    // 3D inflow values (if applicable)
    std::function<double(double y, double z)> u_inflow_3d = [](double, double) { return 0.0; };
    std::function<double(double y, double z)> v_inflow_3d = [](double, double) { return 0.0; };
    std::function<double(double y, double z)> w_inflow_3d = [](double, double) { return 0.0; };
};

/// Incompressible RANS solver using projection method
class RANSSolver {
public:
    explicit RANSSolver(const Mesh& mesh, const Config& config);
    ~RANSSolver();  // Clean up GPU resources
    
    /// Set turbulence model (takes ownership)
    void set_turbulence_model(std::unique_ptr<TurbulenceModel> model);
    
    /// Set velocity boundary conditions
    void set_velocity_bc(const VelocityBC& bc);
    
    /// Set body force (pressure gradient equivalent)
    void set_body_force(double fx, double fy, double fz = 0.0);

    /// Get the selected Poisson solver type (after auto-selection)
    PoissonSolverType poisson_solver_type() const { return selected_solver_; }

    /// Get the reason for Poisson solver selection (for debugging/observability)
    /// Returns a human-readable string explaining why the solver was chosen
    const std::string& selection_reason() const { return selection_reason_; }

    /// Poisson solve statistics from last step (for testing/diagnostics)
    struct PoissonStats {
        int cycles = 0;           ///< V-cycles (or iterations) performed
        double rhs_norm_l2 = 0.0; ///< ||b||_L2 at start of solve
        double rhs_norm_inf = 0.0;///< ||b||_∞ at start of solve
        double res_norm_l2 = 0.0; ///< ||r||_L2 after solve
        double res_norm_inf = 0.0;///< ||r||_∞ after solve
        double res_over_rhs = 0.0;///< ||r||/||b|| (relative residual)
    };

    /// Get statistics from last Poisson solve (updated after each step())
    const PoissonStats& poisson_stats() const { return poisson_stats_; }

    /// Print solver configuration (call after set_velocity_bc() for final info)
    /// Prints: selected solver, selection reason, solver parameters, build info
    void print_solver_info() const;

#ifdef USE_HYPRE
    /// Enable/disable HYPRE PFMG Poisson solver (legacy API, prefer --poisson=hypre)
    void set_use_hypre(bool use) { if (use) selected_solver_ = PoissonSolverType::HYPRE; }
    bool using_hypre() const { return selected_solver_ == PoissonSolverType::HYPRE; }
#endif

    /// Initialize velocity field
    void initialize(const VectorField& initial_velocity);
    void initialize_uniform(double u0, double v0);
    
    /// Advance one pseudo-time step
    /// Returns velocity residual (for convergence check)
    double step();
    
    /// Run to steady state
    /// Returns final residual and number of iterations
    std::pair<double, int> solve_steady();
    
    /// Run to steady state with optional VTK snapshots
    /// @param output_prefix Base name for VTK files (e.g., "output/flow")
    /// @param num_snapshots Number of snapshots to write during simulation (0 = final only)
    /// @param snapshot_freq Override automatic snapshot frequency (optional, -1 = auto)
    /// @return Final residual and number of iterations
    std::pair<double, int> solve_steady_with_snapshots(
        const std::string& output_prefix = "",
        int num_snapshots = 0,
        int snapshot_freq = -1
    );
    
    /// Advance unsteady simulation for multiple time steps
    /// @param dt Time step size
    /// @param nsteps Number of time steps to take
    void advance_unsteady(double dt, int nsteps);
    
    /// Compute adaptive time step based on CFL and diffusion stability
    double compute_adaptive_dt() const;
    
    /// Get current time step
    double current_dt() const { return current_dt_; }
    
    /// Access fields
    const VectorField& velocity() const { return velocity_; }
    const ScalarField& pressure() const { return pressure_; }
    const ScalarField& nu_t() const { return nu_t_; }
    const ScalarField& k() const { return k_; }
    const ScalarField& omega() const { return omega_; }
    const ScalarField& nu_eff() const { return nu_eff_; }
    const ScalarField& div_velocity() const { return div_velocity_; }  ///< For diagnostics
    const ScalarField& rhs_poisson() const { return rhs_poisson_; }    ///< For diagnostics
    const ScalarField& pressure_correction() const { return pressure_correction_; }  ///< For Poisson residual checks

    VectorField& velocity() { return velocity_; }
    ScalarField& pressure() { return pressure_; }
    
    /// Get current iteration
    int iteration() const { return iter_; }
    
    /// Compute bulk velocity (area-averaged)
    double bulk_velocity() const;
    
    /// Compute wall shear stress (at y_min wall)
    double wall_shear_stress() const;
    
    /// Compute friction velocity u_tau
    double friction_velocity() const;
    
    /// Compute friction Reynolds number Re_tau
    double Re_tau() const;

    /// Compute convective KE production rate: <u, conv(u)>
    /// Returns the rate of KE change due to advection (should be ~0 for skew-symmetric form)
    /// Call this after compute_convective_term() to get meaningful results
    double compute_convective_ke_production() const;
    
    /// Print velocity profile at x = x_loc
    void print_velocity_profile(double x_loc = 0.0) const;
    
    /// Write fields to files
    void write_fields(const std::string& prefix) const;
    
    /// Write VTK output for ParaView visualization
    void write_vtk(const std::string& filename) const;
    
    /// GPU buffer management (public for testing and initialization)
    void sync_to_gpu();                 // Update GPU after CPU-side modifications (e.g., after initialization)
    void sync_from_gpu();               // Update CPU copy for I/O (data stays on GPU) - downloads all fields
    void sync_solution_from_gpu();      // Sync only solution fields (u,v,p,nu_t) - use for diagnostics/I/O
    void sync_transport_from_gpu();     // Sync transport fields (k,omega) only if needed - guards laminar runs

    /// Check for NaN/Inf in solution fields and abort if detected
    /// @param step Current step number (used for guard interval checking)
    /// @throws std::runtime_error if NaN/Inf detected and guard is enabled
    /// @note Checks velocity, pressure, nu_t, and transport fields (k, omega)
    void check_for_nan_inf(int step) const;
    
private:
    const Mesh* mesh_;
    Config config_;
    
    // Solution fields
    VectorField velocity_;
    VectorField velocity_star_;  // Provisional velocity
    ScalarField pressure_;
    ScalarField pressure_correction_;
    ScalarField nu_t_;           // Eddy viscosity
    ScalarField k_;              // TKE (for transport models)
    ScalarField omega_;          // Specific dissipation (for transport models)
    TensorField tau_ij_;         // Reynolds stresses (for TBNN)
    
    // Auxiliary fields
    ScalarField rhs_poisson_;    // RHS for pressure Poisson equation
    ScalarField div_velocity_;   // Velocity divergence
    
    // Persistent work fields (avoid reallocation every step)
    ScalarField nu_eff_;         // Effective viscosity nu + nu_t
    VectorField conv_;           // Convective term
    VectorField diff_;           // Diffusive term
    VectorField velocity_old_;   // Previous velocity for residual (GPU-resident when offload enabled)
    VectorField velocity_rk_;    // Work buffer for RK stages (stores u^n during multi-stage update)
    
    // Gradient scratch buffers for turbulence models (GPU-resident)
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    ScalarField wall_distance_;  // Precomputed wall distance field
    
    // Solvers
    PoissonSolver poisson_solver_;
    MultigridPoissonSolver mg_poisson_solver_;
#ifdef USE_HYPRE
    std::unique_ptr<HyprePoissonSolver> hypre_poisson_solver_;
#endif
#ifdef USE_FFT_POISSON
    std::unique_ptr<FFTPoissonSolver> fft_poisson_solver_;      // 2D FFT (periodic x AND z) - 3D only
    std::unique_ptr<FFT1DPoissonSolver> fft1d_poisson_solver_;  // 1D FFT (periodic x OR z) - 3D only
    std::unique_ptr<FFT2DPoissonSolver> fft2d_poisson_solver_;  // 2D mesh FFT (periodic x, walls y)
#endif
    PoissonSolverType selected_solver_ = PoissonSolverType::MG;  // Actually selected solver (after auto)
    std::string selection_reason_;  // Human-readable reason for solver selection (observability)
    PoissonStats poisson_stats_;    // Statistics from last Poisson solve (for testing/diagnostics)
    std::unique_ptr<TurbulenceModel> turb_model_;
    bool use_multigrid_ = true;  // Use multigrid by default for speed
    
    // Boundary conditions
    VelocityBC velocity_bc_;
    PoissonBC poisson_bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC poisson_bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC poisson_bc_z_lo_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_z_hi_ = PoissonBC::Periodic;
    double fx_ = 0.0, fy_ = 0.0, fz_ = 0.0;  // Body force (3D)
    
    // State
    int iter_ = 0;
    int step_count_ = 0;  // Track current step for guard
    double current_dt_;
    
    // Original internal methods
    void apply_velocity_bc();
    void compute_convective_term(const VectorField& vel, VectorField& conv);
    void compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, VectorField& diff);
    void compute_divergence(VelocityWhich which, ScalarField& div);
    void correct_velocity();
    double compute_residual();
    
    // IMEX methods
    void solve_implicit_diffusion(const VectorField& u_explicit, const ScalarField& nu_eff, 
                                  VectorField& u_new, double dt);
    
    // Time integration methods
    void euler_substep(VectorField& vel_in, VectorField& vel_out, double dt);
    void project_velocity(VectorField& vel, double dt);
    void ssprk2_step(double dt);
    void ssprk3_step(double dt);
    
    // Gradient computations
    void compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy);
    
    // GPU buffers (always present for ABI stability)
    // Simplified GPU strategy: Data mapped once at initialization, stays resident on GPU
    // All kernels use is_device_ptr to access already-mapped data
    // No temporary map(to:/from:) clauses in kernels - eliminates mapping conflicts
    bool gpu_ready_ = false;
    
    // Persistent pointers to GPU-resident arrays (mapped for solver lifetime)
    double* velocity_u_ptr_ = nullptr;
    double* velocity_v_ptr_ = nullptr;
    double* velocity_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_star_u_ptr_ = nullptr;
    double* velocity_star_v_ptr_ = nullptr;
    double* velocity_star_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_old_u_ptr_ = nullptr;  // Device-resident old velocity for residual
    double* velocity_old_v_ptr_ = nullptr;  // Device-resident old velocity for residual
    double* velocity_old_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_rk_u_ptr_ = nullptr;   // RK work buffer for u
    double* velocity_rk_v_ptr_ = nullptr;   // RK work buffer for v
    double* velocity_rk_w_ptr_ = nullptr;   // RK work buffer for w (3D)
    double* pressure_ptr_ = nullptr;
    double* pressure_corr_ptr_ = nullptr;
    double* nu_t_ptr_ = nullptr;
    double* nu_eff_ptr_ = nullptr;
    double* conv_u_ptr_ = nullptr;
    double* conv_v_ptr_ = nullptr;
    double* conv_w_ptr_ = nullptr;  // 3D convective term
    double* diff_u_ptr_ = nullptr;
    double* diff_v_ptr_ = nullptr;
    double* diff_w_ptr_ = nullptr;  // 3D diffusive term
    double* rhs_poisson_ptr_ = nullptr;
    double* div_velocity_ptr_ = nullptr;
    double* k_ptr_ = nullptr;
    double* omega_ptr_ = nullptr;
    
    // Reynolds stress tensor pointers (for EARSM/TBNN models)
    double* tau_xx_ptr_ = nullptr;
    double* tau_xy_ptr_ = nullptr;
    double* tau_xz_ptr_ = nullptr;  // 3D
    double* tau_yy_ptr_ = nullptr;
    double* tau_yz_ptr_ = nullptr;  // 3D
    double* tau_zz_ptr_ = nullptr;  // 3D

    // Gradient scratch buffers (cell-centered, for turbulence models)
    double* dudx_ptr_ = nullptr;
    double* dudy_ptr_ = nullptr;
    double* dudz_ptr_ = nullptr;  // 3D
    double* dvdx_ptr_ = nullptr;
    double* dvdy_ptr_ = nullptr;
    double* dvdz_ptr_ = nullptr;  // 3D
    double* dwdx_ptr_ = nullptr;  // 3D
    double* dwdy_ptr_ = nullptr;  // 3D
    double* dwdz_ptr_ = nullptr;  // 3D
    double* wall_distance_ptr_ = nullptr;
    
    size_t field_total_size_ = 0;  // (Nx+2)*(Ny+2) for fields with ghost cells

    void extract_field_pointers();  // Set raw pointers to field data (shared by CPU/GPU paths)
    void initialize_gpu_buffers();  // Map data to GPU (called once in constructor)
    void cleanup_gpu_buffers();     // Unmap and copy results back (called in destructor)
    
public:
    /// Get device view for turbulence models (GPU-resident pointers)
    /// Returns a view with pointers to solver-owned GPU data.
    /// Only valid if gpu_ready_ == true.
    TurbulenceDeviceView get_device_view() const;
    
    /// Get solver view for core NS/projection kernels
    /// Returns GPU-resident pointers if gpu_ready_, else host pointers
    SolverDeviceView get_solver_view() const;
};

} // namespace nncfd


