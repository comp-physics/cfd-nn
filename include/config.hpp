#pragma once

#include <string>
#include <map>
#include <sstream>
#include <stdexcept>

namespace nncfd {

/// Turbulence model selection
enum class TurbulenceModelType {
    None,           ///< Laminar (no turbulence model)
    Baseline,       ///< Simple algebraic mixing length model
    GEP,            ///< Gene Expression Programming algebraic model (Weatheritt-Sandberg 2016)
    NNMLP,          ///< Neural network scalar eddy viscosity
    NNTBNN,         ///< TBNN-style anisotropy model
    // Transport equation models
    SSTKOmega,      ///< SST k-ω transport with linear Boussinesq closure
    KOmega,         ///< Standard k-ω (Wilcox 1988)
    // EARSM models
    EARSM_WJ,       ///< SST k-ω + Wallin-Johansson EARSM
    EARSM_GS,       ///< SST k-ω + Gatski-Speziale EARSM
    EARSM_Pope      ///< SST k-ω + Pope quadratic model
};

/// Convective/advection scheme selection
enum class ConvectiveScheme {
    Central,    ///< 2nd-order central (advective form) - original default
    Upwind,     ///< 1st-order upwind (advective form) - dissipative, stable
    Skew,       ///< Skew-symmetric/split form - energy conserving (DNS/LES)
    Upwind2     ///< 2nd-order upwind (advective form) - less dissipative than 1st-order
};

/// Simulation mode selection
enum class SimulationMode {
    Steady,     ///< Solve to steady state (convergence-based)
    Unsteady    ///< Time-accurate integration (fixed number of steps)
};

/// Poisson solver selection
enum class PoissonSolverType {
    Auto,       ///< Auto-select: FFT → FFT2D → FFT1D → HYPRE → MG
    FFT,        ///< 2D FFT solver (requires periodic x AND z, uniform dx/dz) - 3D only
    FFT2D,      ///< 2D mesh FFT solver (periodic x, walls y, Nz=1)
    FFT1D,      ///< 1D FFT + 2D Helmholtz solver (requires periodic x OR z) - 3D only
    HYPRE,      ///< HYPRE PFMG solver (requires USE_HYPRE build)
    MG          ///< Native geometric multigrid
};

/// Time integration scheme selection
enum class TimeIntegrator {
    Euler,      ///< Forward Euler (1st order) - current default, for debugging/RANS
    RK2,        ///< SSP-RK2 (2nd order) - solid baseline
    RK3         ///< SSP-RK3 (3rd order) - recommended for LES/DNS
};

/// Simulation configuration
struct Config {
    // Domain and mesh
    int Nx = 64;
    int Ny = 64;
    double x_min = 0.0;
    double x_max = 2.0 * 3.14159265358979;
    double y_min = -1.0;
    double y_max = 1.0;
    bool stretch_y = false;
    double stretch_beta = 2.0;  ///< Stretching parameter for tanh stretching

    // Z-direction parameters (3D)
    int Nz = 1;                 ///< Grid cells in z (1 = 2D simulation)
    double z_min = 0.0;
    double z_max = 1.0;
    bool stretch_z = false;
    double stretch_beta_z = 2.0;  ///< Stretching parameter for z-direction

    // Physical parameters
    double Re = 1000.0;         ///< Reynolds number (based on channel half-height and bulk velocity)
    double nu = 0.001;          ///< Kinematic viscosity (computed from Re)
    double rho = 1.0;           ///< Density
    double dp_dx = -1.0;        ///< Pressure gradient (or body force) driving the flow
    
    // Control flags for Re-based setup
    bool Re_specified = false;
    bool nu_specified = false;
    bool dp_dx_specified = false;
    
    // Time stepping
    double dt = 0.001;          ///< Time step
    double CFL_max = 0.5;       ///< Maximum CFL for adaptive dt
    bool adaptive_dt = true;    ///< Use adaptive time stepping based on CFL
    int max_steps = 10000;      ///< Maximum time steps for simulation
    double T_final = -1.0;      ///< Final time for unsteady simulations (-1 = not set, use max_steps)
    double tol = 1e-6;          ///< Convergence tolerance for steady-state
    TimeIntegrator time_integrator = TimeIntegrator::Euler;  ///< Time integration scheme
    
    // Numerical schemes
    ConvectiveScheme convective_scheme = ConvectiveScheme::Central;
    int space_order = 2;        ///< Spatial discretization order (2 or 4)

    // Simulation mode
    SimulationMode simulation_mode = SimulationMode::Steady;

    // Initial conditions (for unsteady/DNS mode)
    double perturbation_amplitude = 1e-2;  ///< Amplitude of initial perturbations for DNS

    // Turbulence model
    TurbulenceModelType turb_model = TurbulenceModelType::None;
    double nu_t_max = 1.0;      ///< Maximum eddy viscosity (clipping)
    
    // NN model paths (must be explicitly specified - no legacy fallback)
    std::string nn_weights_path;
    std::string nn_scaling_path;
    std::string nn_preset;      ///< Preset model name (e.g., "tbnn_channel_caseholdout")
    
    // Output
    std::string output_dir = "output/";
    int output_freq = 100;      ///< Console output frequency (iterations)
    int num_snapshots = 10;     ///< Number of VTK snapshots during simulation
    bool verbose = true;
    
    // Benchmark / postprocessing controls
    // - postprocess: Poiseuille table + L2 error + velocity_profile.dat
    // - write_fields: VTK/field output via solver.write_fields(...) and snapshots
    bool postprocess = true;
    bool write_fields = true;

    // Performance benchmarking
    int warmup_steps = 0;           ///< Steps to run before resetting timers (excluded from timing)
    
    // Poisson solver
    double poisson_tol = 1e-6;       ///< Legacy absolute tolerance (deprecated)
    int poisson_max_vcycles = 20;    ///< Max MG V-cycles per Poisson solve
    double poisson_omega = 1.8;      ///< SOR relaxation parameter
    double poisson_abs_tol_floor = 1e-8; ///< Absolute tolerance floor to prevent over-solving near steady state
    PoissonSolverType poisson_solver = PoissonSolverType::Auto;  ///< Poisson solver selection

    // Robust MG convergence criteria (recommended for projection)
    double poisson_tol_abs = 0.0;    ///< Absolute tolerance on ||r||_∞ (0 = disabled)
    double poisson_tol_rhs = 1e-6;   ///< RHS-relative: ||r||/||b|| (tight for Galilean invariance)
    double poisson_tol_rel = 1e-3;   ///< Initial-residual relative: ||r||/||r0||
    int poisson_check_interval = 1;  ///< Check convergence every N V-cycles (fused norms are cheap)
    bool poisson_use_l2_norm = true; ///< Use L2 norm for convergence (smoother than L∞, less hot-cell sensitive)
    double poisson_linf_safety = 10.0; ///< L∞ safety cap multiplier (prevent L2 from hiding bad cells)
    int poisson_fixed_cycles = 8;    ///< Fixed V-cycle count (optimal: 8 cycles with nu1=2,nu2=1)

    // Adaptive fixed-cycle mode: run check_after cycles, check residual, add 2 more if needed
    // DEFAULT ON: ensures Galilean invariance by checking convergence, adds ~1% overhead
    // The solver runs check_after cycles fast, then checks ||r||/||b|| < tol_rhs, adds more if needed
    bool poisson_adaptive_cycles = true;   ///< Enable adaptive checking within fixed-cycle mode (default: ON for robustness)
    int poisson_check_after = 4;           ///< Check residual after this many cycles

    // MG smoother tuning parameters
    // Optimal at 128³ with walls: nu1=3, nu2=1 (more pre-smooth for wall BCs)
    int poisson_nu1 = 0;             ///< Pre-smoothing sweeps (0 = auto: 3 for wall BCs)
    int poisson_nu2 = 0;             ///< Post-smoothing sweeps (0 = auto: 1)
    int poisson_chebyshev_degree = 4; ///< Chebyshev polynomial degree (3-4 typical)

    // CUDA Graph acceleration (GPU only)
    // Captures entire V-cycle as single graph for massive kernel launch reduction
    bool poisson_use_vcycle_graph = true;  ///< Enable V-cycle CUDA Graph (default: ON)

    // Turbulence guard (abort on NaN/Inf)
    bool turb_guard_enabled = true;         ///< Enable NaN/Inf guard checks
    int turb_guard_interval = 5;            ///< Check every N steps (performance)

    // Benchmark mode
    bool benchmark = false;                 ///< Enable benchmark mode (optimized for timing)
    
    /// Load configuration from file
    void load(const std::string& filename);
    
    /// Parse command line arguments
    void parse_args(int argc, char** argv);
    
    /// Print configuration
    void print() const;
    
    /// Compute derived quantities (e.g., nu from Re)
    void finalize();
};

/// Parse a key-value config file
std::map<std::string, std::string> parse_config_file(const std::string& filename);

} // namespace nncfd


