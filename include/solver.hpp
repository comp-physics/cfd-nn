#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#include "turbulence_model.hpp"
#include "config.hpp"
#include <memory>
#include <functional>

namespace nncfd {

/// Boundary condition specification for velocity
struct VelocityBC {
    enum Type { NoSlip, Periodic, Inflow, Outflow };
    Type x_lo = Periodic;
    Type x_hi = Periodic;
    Type y_lo = NoSlip;
    Type y_hi = NoSlip;
    
    // Inflow values (if applicable)
    std::function<double(double y)> u_inflow = [](double) { return 0.0; };
    std::function<double(double y)> v_inflow = [](double) { return 0.0; };
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
    void set_body_force(double fx, double fy);
    
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
    
    /// Print velocity profile at x = x_loc
    void print_velocity_profile(double x_loc = 0.0) const;
    
    /// Write fields to files
    void write_fields(const std::string& prefix) const;
    
    /// Write VTK output for ParaView visualization
    void write_vtk(const std::string& filename) const;
    
    /// GPU buffer management (public for testing and initialization)
    void sync_to_gpu();              // Update GPU after CPU-side modifications (e.g., after initialization)
    void sync_from_gpu();            // Update CPU copy for I/O (data stays on GPU)
    
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
    
    // Solvers
    PoissonSolver poisson_solver_;
    MultigridPoissonSolver mg_poisson_solver_;
    std::unique_ptr<TurbulenceModel> turb_model_;
    bool use_multigrid_ = true;  // Use multigrid by default for speed
    
    // Boundary conditions
    VelocityBC velocity_bc_;
    PoissonBC poisson_bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC poisson_bc_y_hi_ = PoissonBC::Neumann;
    double fx_ = 0.0, fy_ = 0.0;  // Body force
    
    // State
    int iter_ = 0;
    double current_dt_;
    
    // Internal methods
    void apply_velocity_bc();
    void compute_convective_term(const VectorField& vel, VectorField& conv);
    void compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, VectorField& diff);
    void compute_divergence(const VectorField& vel, ScalarField& div);
    void correct_velocity();
    double compute_residual();
    
    // IMEX methods
    void solve_implicit_diffusion(const VectorField& u_explicit, const ScalarField& nu_eff, 
                                  VectorField& u_new, double dt);
    
    // Time integration methods
    void project_velocity(VectorField& vel_star, double dt);
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
    double* velocity_star_u_ptr_ = nullptr;
    double* velocity_star_v_ptr_ = nullptr;
    double* pressure_ptr_ = nullptr;
    double* pressure_corr_ptr_ = nullptr;
    double* nu_t_ptr_ = nullptr;
    double* nu_eff_ptr_ = nullptr;
    double* conv_u_ptr_ = nullptr;
    double* conv_v_ptr_ = nullptr;
    double* diff_u_ptr_ = nullptr;
    double* diff_v_ptr_ = nullptr;
    double* rhs_poisson_ptr_ = nullptr;
    double* div_velocity_ptr_ = nullptr;
    double* k_ptr_ = nullptr;
    double* omega_ptr_ = nullptr;
    size_t field_total_size_ = 0;  // (Nx+2)*(Ny+2) for fields with ghost cells
    
    void initialize_gpu_buffers();  // Map data to GPU (called once in constructor)
    void cleanup_gpu_buffers();     // Unmap and copy results back (called in destructor)
};

} // namespace nncfd


