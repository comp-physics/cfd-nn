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

/// Convective scheme selection
enum class ConvectiveScheme {
    Central,
    Upwind
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
    int max_iter = 10000;       ///< Maximum iterations for steady-state convergence
    double tol = 1e-6;          ///< Convergence tolerance for steady-state
    
    // Numerical schemes
    ConvectiveScheme convective_scheme = ConvectiveScheme::Central;
    
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
    double poisson_tol = 1e-6;
    int poisson_max_iter = 5;  ///< MG V-cycles per solve (5 for projection, increase for accurate solve)
    double poisson_omega = 1.8; ///< SOR relaxation parameter
    double poisson_abs_tol_floor = 1e-8; ///< Absolute tolerance floor to prevent over-solving near steady state
    PoissonSolverType poisson_solver = PoissonSolverType::Auto;  ///< Poisson solver selection

    // Turbulence guard (abort on NaN/Inf)
    bool turb_guard_enabled = true;         ///< Enable NaN/Inf guard checks
    int turb_guard_interval = 5;            ///< Check every N steps (performance)
    
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


