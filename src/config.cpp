/// @file config.cpp
/// @brief Configuration file parsing and command-line argument processing
///
/// Implements the Config class for managing simulation parameters. Supports:
/// - Key-value config file parsing (.cfg format)
/// - Command-line argument processing (--flag value syntax)
/// - Automatic Reynolds number calculations from specified parameters
/// - Parameter validation and consistency checking

#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace nncfd {

static std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        ++start;
    }
    auto end = s.end();
    while (end != start && std::isspace(*(end - 1))) {
        --end;
    }
    return std::string(start, end);
}

/// Parse convective scheme from string. Returns Central if unknown.
static ConvectiveScheme parse_convective_scheme(const std::string& val) {
    if (val == "upwind") {
        return ConvectiveScheme::Upwind;
    } else if (val == "skew" || val == "skew_symmetric" || val == "skewsymmetric") {
        return ConvectiveScheme::Skew;
    } else if (val == "upwind2") {
        return ConvectiveScheme::Upwind2;
    } else if (val == "conservative" || val == "cons") {
        std::cerr << "Warning: 'conservative' (flux-form) scheme was removed; mapping to 'skew' (not flux-form).\n";
        std::cerr << "         Note: skew is energy-conserving but NOT equivalent to flux-form conservative.\n";
        std::cerr << "         Available schemes: central, upwind, skew, upwind2\n";
        return ConvectiveScheme::Skew;
    } else if (val == "central") {
        return ConvectiveScheme::Central;
    } else {
        std::cerr << "Warning: Unknown convective_scheme '" << val << "'. Using 'central'.\n";
        std::cerr << "         Available schemes: central, upwind, skew, upwind2\n";
        return ConvectiveScheme::Central;
    }
}

/// Parse time integrator from string. Returns Euler if unknown.
static TimeIntegrator parse_time_integrator(const std::string& val) {
    if (val == "rk2") {
        return TimeIntegrator::RK2;
    } else if (val == "rk3") {
        return TimeIntegrator::RK3;
    } else if (val == "euler") {
        return TimeIntegrator::Euler;
    } else {
        std::cerr << "Warning: Unknown time_integrator '" << val << "'. Using 'euler'.\n";
        return TimeIntegrator::Euler;
    }
}

/// Parse and validate space_order. Returns 2 if invalid.
static int parse_space_order(int val) {
    if (val != 2 && val != 4) {
        std::cerr << "Warning: space_order must be 2 or 4, got " << val << ". Using 2.\n";
        return 2;
    }
    return val;
}

std::map<std::string, std::string> parse_config_file(const std::string& filename) {
    std::map<std::string, std::string> result;
    std::ifstream file(filename);
    
    if (!file) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find '=' separator
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        
        result[key] = value;
    }
    
    return result;
}

void Config::load(const std::string& filename) {
    auto params = parse_config_file(filename);
    
    auto get_int = [&](const std::string& key, int def) {
        auto it = params.find(key);
        return it != params.end() ? std::stoi(it->second) : def;
    };
    
    auto get_double = [&](const std::string& key, double def) {
        auto it = params.find(key);
        return it != params.end() ? std::stod(it->second) : def;
    };
    
    auto get_bool = [&](const std::string& key, bool def) {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second == "true" || it->second == "1";
        }
        return def;
    };
    
    auto get_string = [&](const std::string& key, const std::string& def) {
        auto it = params.find(key);
        return it != params.end() ? it->second : def;
    };
    
    // Mesh
    Nx = get_int("Nx", Nx);
    Ny = get_int("Ny", Ny);
    x_min = get_double("x_min", x_min);
    x_max = get_double("x_max", x_max);
    y_min = get_double("y_min", y_min);
    y_max = get_double("y_max", y_max);
    stretch_y = get_bool("stretch_y", stretch_y);
    stretch_beta = get_double("stretch_beta", stretch_beta);

    // Z-direction (3D)
    Nz = get_int("Nz", Nz);
    z_min = get_double("z_min", z_min);
    z_max = get_double("z_max", z_max);
    stretch_z = get_bool("stretch_z", stretch_z);
    stretch_beta_z = get_double("stretch_beta_z", stretch_beta_z);

    // Physical
    auto params_map = params; // Save for checking if key exists
    Re = get_double("Re", Re);
    nu = get_double("nu", nu);
    rho = get_double("rho", rho);
    dp_dx = get_double("dp_dx", dp_dx);
    
    // Track which parameters were explicitly set in config file
    if (params_map.find("Re") != params_map.end()) Re_specified = true;
    if (params_map.find("nu") != params_map.end()) nu_specified = true;
    if (params_map.find("dp_dx") != params_map.end()) dp_dx_specified = true;
    
    // Time stepping
    dt = get_double("dt", dt);
    CFL_max = get_double("CFL_max", CFL_max);
    adaptive_dt = get_bool("adaptive_dt", adaptive_dt);
    max_steps = get_int("max_steps", max_steps);
    T_final = get_double("T_final", T_final);
    tol = get_double("tol", tol);
    
    // Numerical scheme (advection)
    convective_scheme = parse_convective_scheme(get_string("convective_scheme", "central"));

    // Spatial order (2 or 4)
    space_order = parse_space_order(get_int("space_order", 2));

    // Time integrator
    time_integrator = parse_time_integrator(get_string("time_integrator", "euler"));
    
    // Simulation mode
    auto mode_str = get_string("simulation_mode", "steady");
    if (mode_str == "unsteady") {
        simulation_mode = SimulationMode::Unsteady;
    } else {
        simulation_mode = SimulationMode::Steady;
    }

    // Initial conditions
    perturbation_amplitude = get_double("perturbation_amplitude", perturbation_amplitude);

    // Turbulence
    auto model_str = get_string("turb_model", "none");
    if (model_str == "baseline") {
        turb_model = TurbulenceModelType::Baseline;
    } else if (model_str == "gep") {
        turb_model = TurbulenceModelType::GEP;
    } else if (model_str == "nn_mlp") {
        turb_model = TurbulenceModelType::NNMLP;
    } else if (model_str == "nn_tbnn") {
        turb_model = TurbulenceModelType::NNTBNN;
    } else if (model_str == "sst" || model_str == "sst_komega") {
        turb_model = TurbulenceModelType::SSTKOmega;
    } else if (model_str == "komega" || model_str == "k-omega") {
        turb_model = TurbulenceModelType::KOmega;
    } else if (model_str == "earsm_wj" || model_str == "wallin_johansson") {
        turb_model = TurbulenceModelType::EARSM_WJ;
    } else if (model_str == "earsm_gs" || model_str == "gatski_speziale") {
        turb_model = TurbulenceModelType::EARSM_GS;
    } else if (model_str == "earsm_pope" || model_str == "pope") {
        turb_model = TurbulenceModelType::EARSM_Pope;
    } else {
        turb_model = TurbulenceModelType::None;
    }
    
    nu_t_max = get_double("nu_t_max", nu_t_max);
    nn_weights_path = get_string("nn_weights_path", nn_weights_path);
    nn_scaling_path = get_string("nn_scaling_path", nn_scaling_path);
    nn_preset = get_string("nn_preset", nn_preset);
    
    // Output
    output_dir = get_string("output_dir", output_dir);
    output_freq = get_int("output_freq", output_freq);
    num_snapshots = get_int("num_snapshots", num_snapshots);
    verbose = get_bool("verbose", verbose);
    postprocess = get_bool("postprocess", postprocess);
    write_fields = get_bool("write_fields", write_fields);
    vtk_binary = get_bool("vtk_binary", vtk_binary);

    // Poisson
    poisson_tol = get_double("poisson_tol", poisson_tol);
    poisson_max_vcycles = get_int("poisson_max_vcycles", poisson_max_vcycles);
    poisson_omega = get_double("poisson_omega", poisson_omega);

    // Parse poisson_solver: auto, fft, fft2d, fft1d, hypre, mg
    std::string solver_str = get_string("poisson_solver", "auto");
    if (solver_str == "auto") {
        poisson_solver = PoissonSolverType::Auto;
    } else if (solver_str == "fft") {
        poisson_solver = PoissonSolverType::FFT;
    } else if (solver_str == "fft2d") {
        poisson_solver = PoissonSolverType::FFT2D;
    } else if (solver_str == "fft1d") {
        poisson_solver = PoissonSolverType::FFT1D;
    } else if (solver_str == "hypre") {
        poisson_solver = PoissonSolverType::HYPRE;
    } else if (solver_str == "mg" || solver_str == "multigrid") {
        poisson_solver = PoissonSolverType::MG;
    } else {
        std::cerr << "ERROR: Unknown poisson_solver='" << solver_str << "'.\n"
                  << "Valid options: auto, fft, fft2d, fft1d, hypre, mg\n";
        std::exit(1);
    }

    // Poisson solver tuning
    poisson_abs_tol_floor = get_double("poisson_abs_tol_floor", poisson_abs_tol_floor);

    // Robust MG convergence criteria
    poisson_tol_abs = get_double("poisson_tol_abs", poisson_tol_abs);
    poisson_tol_rhs = get_double("poisson_tol_rhs", poisson_tol_rhs);
    poisson_tol_rel = get_double("poisson_tol_rel", poisson_tol_rel);
    poisson_check_interval = get_int("poisson_check_interval", poisson_check_interval);
    poisson_use_l2_norm = get_bool("poisson_use_l2_norm", poisson_use_l2_norm);
    poisson_linf_safety = get_double("poisson_linf_safety", poisson_linf_safety);
    poisson_fixed_cycles = get_int("poisson_fixed_cycles", poisson_fixed_cycles);

    // Turbulence guard settings
    turb_guard_enabled = get_bool("turb_guard_enabled", turb_guard_enabled);
    turb_guard_interval = get_int("turb_guard_interval", turb_guard_interval);

    // Legacy flags (for backward compatibility)
    if (get_bool("use_hypre", false)) {
        poisson_solver = PoissonSolverType::HYPRE;
    }
    if (get_bool("use_fft", false)) {
        poisson_solver = PoissonSolverType::FFT;
    }

    // Recycling inflow parameters
    recycling_inflow = get_bool("recycling_inflow", recycling_inflow);
    recycle_x = get_double("recycle_x", recycle_x);
    recycle_shift_z = get_int("recycle_shift_z", recycle_shift_z);
    recycle_shift_interval = get_int("recycle_shift_interval", recycle_shift_interval);
    recycle_filter_tau = get_double("recycle_filter_tau", recycle_filter_tau);
    recycle_fringe_length = get_double("recycle_fringe_length", recycle_fringe_length);
    recycle_target_bulk_u = get_double("recycle_target_bulk_u", recycle_target_bulk_u);
    recycle_remove_transverse_mean = get_bool("recycle_remove_transverse_mean", recycle_remove_transverse_mean);

    // Projection health watchdog
    div_threshold = get_double("div_threshold", div_threshold);
    div_tol_acceptable = get_double("div_tol_acceptable", div_tol_acceptable);
    projection_watchdog = get_bool("projection_watchdog", projection_watchdog);
    gpu_only_mode = get_bool("gpu_only_mode", gpu_only_mode);

    finalize();
}

void Config::parse_args(int argc, char** argv) {
    // Helper to get argument value (handles both --key=value and --key value)
    auto get_value = [&](int& i, const std::string& arg, const std::string& key) -> std::string {
        // Check for --key=value format
        if (arg.rfind(key + "=", 0) == 0) {
            return arg.substr(key.length() + 1);
        }
        // Check for --key value format
        if (arg == key && i + 1 < argc) {
            return argv[++i];
        }
        return "";
    };

    // Helper to check for flag-style arguments (--flag)
    auto is_flag = [](const std::string& arg, const std::string& key) -> bool {
        return arg == key;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string val;

        if ((val = get_value(i, arg, "--config")) != "") {
            load(val);
        } else if ((val = get_value(i, arg, "--Nx")) != "") {
            Nx = std::stoi(val);
        } else if ((val = get_value(i, arg, "--Ny")) != "") {
            Ny = std::stoi(val);
        } else if ((val = get_value(i, arg, "--Nz")) != "") {
            Nz = std::stoi(val);
        } else if ((val = get_value(i, arg, "--z_min")) != "") {
            z_min = std::stod(val);
        } else if ((val = get_value(i, arg, "--z_max")) != "") {
            z_max = std::stod(val);
        } else if (is_flag(arg, "--stretch_z")) {
            stretch_z = true;
        } else if ((val = get_value(i, arg, "--stretch_beta_z")) != "") {
            stretch_beta_z = std::stod(val);
        } else if ((val = get_value(i, arg, "--Re")) != "") {
            Re = std::stod(val);
            Re_specified = true;
        } else if ((val = get_value(i, arg, "--nu")) != "") {
            nu = std::stod(val);
            nu_specified = true;
        } else if ((val = get_value(i, arg, "--dp_dx")) != "") {
            dp_dx = std::stod(val);
            dp_dx_specified = true;
        } else if ((val = get_value(i, arg, "--dt")) != "") {
            dt = std::stod(val);
        } else if ((val = get_value(i, arg, "--max_steps")) != "") {
            max_steps = std::stoi(val);
        } else if ((val = get_value(i, arg, "--tol")) != "") {
            tol = std::stod(val);
        } else if ((val = get_value(i, arg, "--poisson_tol")) != "") {
            poisson_tol = std::stod(val);
        } else if ((val = get_value(i, arg, "--poisson_max_vcycles")) != "") {
            poisson_max_vcycles = std::stoi(val);
        } else if ((val = get_value(i, arg, "--poisson")) != "") {
            if (val == "auto") {
                poisson_solver = PoissonSolverType::Auto;
            } else if (val == "fft") {
                poisson_solver = PoissonSolverType::FFT;
            } else if (val == "fft2d") {
                poisson_solver = PoissonSolverType::FFT2D;
            } else if (val == "fft1d") {
                poisson_solver = PoissonSolverType::FFT1D;
            } else if (val == "hypre") {
                poisson_solver = PoissonSolverType::HYPRE;
            } else if (val == "mg" || val == "multigrid") {
                poisson_solver = PoissonSolverType::MG;
            } else {
                std::cerr << "Warning: Unknown --poisson value '" << val << "', using auto\n";
                poisson_solver = PoissonSolverType::Auto;
            }
        } else if (is_flag(arg, "--use_hypre")) {
            // Legacy flag (backward compatibility)
            poisson_solver = PoissonSolverType::HYPRE;
        } else if (is_flag(arg, "--use_fft")) {
            // Legacy flag (backward compatibility)
            poisson_solver = PoissonSolverType::FFT;
        } else if ((val = get_value(i, arg, "--poisson_abs_tol_floor")) != "") {
            poisson_abs_tol_floor = std::stod(val);
        } else if (is_flag(arg, "--no_poisson_vcycle_graph")) {
            poisson_use_vcycle_graph = false;
        } else if ((val = get_value(i, arg, "--turb_guard_enabled")) != "") {
            turb_guard_enabled = (val == "true" || val == "1" || val == "yes");
        } else if (is_flag(arg, "--turb_guard_enabled")) {
            turb_guard_enabled = true;
        } else if ((val = get_value(i, arg, "--turb_guard_interval")) != "") {
            turb_guard_interval = std::stoi(val);
        } else if (is_flag(arg, "--recycling_inflow")) {
            recycling_inflow = true;
        } else if ((val = get_value(i, arg, "--recycle_x")) != "") {
            recycle_x = std::stod(val);
        } else if ((val = get_value(i, arg, "--recycle_shift_z")) != "") {
            recycle_shift_z = std::stoi(val);
        } else if ((val = get_value(i, arg, "--recycle_shift_interval")) != "") {
            recycle_shift_interval = std::stoi(val);
        } else if ((val = get_value(i, arg, "--recycle_filter_tau")) != "") {
            recycle_filter_tau = std::stod(val);
        } else if ((val = get_value(i, arg, "--recycle_fringe_length")) != "") {
            recycle_fringe_length = std::stod(val);
        } else if ((val = get_value(i, arg, "--recycle_target_bulk_u")) != "") {
            recycle_target_bulk_u = std::stod(val);
        } else if ((val = get_value(i, arg, "--model")) != "") {
            std::string model = val;
            if (model == "none" || model == "laminar") {
                turb_model = TurbulenceModelType::None;
            } else if (model == "baseline") {
                turb_model = TurbulenceModelType::Baseline;
            } else if (model == "gep") {
                turb_model = TurbulenceModelType::GEP;
            } else if (model == "nn_mlp") {
                turb_model = TurbulenceModelType::NNMLP;
            } else if (model == "nn_tbnn") {
                turb_model = TurbulenceModelType::NNTBNN;
            } else if (model == "sst" || model == "sst_komega") {
                turb_model = TurbulenceModelType::SSTKOmega;
            } else if (model == "komega" || model == "k-omega") {
                turb_model = TurbulenceModelType::KOmega;
            } else if (model == "earsm_wj" || model == "wallin_johansson") {
                turb_model = TurbulenceModelType::EARSM_WJ;
            } else if (model == "earsm_gs" || model == "gatski_speziale") {
                turb_model = TurbulenceModelType::EARSM_GS;
            } else if (model == "earsm_pope" || model == "pope") {
                turb_model = TurbulenceModelType::EARSM_Pope;
            }
        } else if ((val = get_value(i, arg, "--weights")) != "") {
            nn_weights_path = val;
        } else if ((val = get_value(i, arg, "--scaling")) != "") {
            nn_scaling_path = val;
        } else if ((val = get_value(i, arg, "--nn_preset")) != "") {
            nn_preset = val;
        } else if ((val = get_value(i, arg, "--output")) != "") {
            output_dir = val;
        } else if ((val = get_value(i, arg, "--num_snapshots")) != "") {
            num_snapshots = std::stoi(val);
        } else if (is_flag(arg, "--verbose")) {
            verbose = true;
        } else if (is_flag(arg, "--quiet")) {
            verbose = false;
        } else if (is_flag(arg, "--no_postprocess")) {
            postprocess = false;
        } else if (is_flag(arg, "--no_write_fields")) {
            write_fields = false;
        } else if (is_flag(arg, "--vtk_ascii")) {
            vtk_binary = false;
        } else if (is_flag(arg, "--stretch")) {
            stretch_y = true;
        } else if (is_flag(arg, "--adaptive_dt")) {
            adaptive_dt = true;
        } else if ((val = get_value(i, arg, "--CFL")) != "") {
            CFL_max = std::stod(val);
        } else if ((val = get_value(i, arg, "--scheme")) != "") {
            convective_scheme = parse_convective_scheme(val);
        } else if ((val = get_value(i, arg, "--space-order")) != "") {
            space_order = parse_space_order(std::stoi(val));
        } else if ((val = get_value(i, arg, "--integrator")) != "") {
            time_integrator = parse_time_integrator(val);
        } else if ((val = get_value(i, arg, "--simulation_mode")) != "") {
            if (val == "unsteady") {
                simulation_mode = SimulationMode::Unsteady;
            } else {
                simulation_mode = SimulationMode::Steady;
            }
        } else if ((val = get_value(i, arg, "--perturbation_amplitude")) != "") {
            perturbation_amplitude = std::stod(val);
        } else if ((val = get_value(i, arg, "--warmup_steps")) != "") {
            warmup_steps = std::stoi(val);
        } else if (is_flag(arg, "--benchmark")) {
            // Benchmark mode: optimized for timing 3D duct flow
            benchmark = true;
            // Grid: 192^3 (can be overridden by subsequent --Nx, --Ny, --Nz)
            Nx = 192;
            Ny = 192;
            Nz = 192;
            // Domain: 3D duct (periodic x, walls y/z)
            x_min = 0.0;
            x_max = 2.0 * 3.14159265358979;
            y_min = -1.0;
            y_max = 1.0;
            z_min = -1.0;
            z_max = 1.0;
            // Disable all output for clean timing
            verbose = false;
            postprocess = false;
            write_fields = false;
            num_snapshots = 0;
            // Numerics: upwind convection, 1 Poisson cycle per step
            convective_scheme = ConvectiveScheme::Upwind;
            poisson_fixed_cycles = 1;
            poisson_adaptive_cycles = false;
            // No turbulence model
            turb_model = TurbulenceModelType::None;
            // Default to 20 steps (can be overridden)
            max_steps = 20;
            // Fixed time step (no adaptive dt for consistent timing)
            adaptive_dt = false;
            dt = 0.001;
        } else if (is_flag(arg, "--perf_mode") || is_flag(arg, "--perf")) {
            // Performance mode: reduce GPU sync overhead for production turbulence runs
            // Sets diag_interval=50, poisson_check_interval=5 in finalize()
            perf_mode = true;
        } else if (is_flag(arg, "--help") || is_flag(arg, "-h")) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --config FILE     Load config file\n"
                      << "  --Nx N            Grid cells in x\n"
                      << "  --Ny N            Grid cells in y\n"
                      << "  --Nz N            Grid cells in z (1 = 2D)\n"
                      << "  --z_min V         Domain z minimum\n"
                      << "  --z_max V         Domain z maximum\n"
                      << "  --stretch_z       Use stretched mesh in z\n"
                      << "  --stretch_beta_z B  Z-stretching parameter (default 2.0)\n"
                      << "  --Re R            Reynolds number (auto-computes nu or dp_dx)\n"
                      << "  --nu V            Kinematic viscosity\n"
                      << "  --dp_dx D         Pressure gradient (driving force)\n"
                      << "  --dt T            Time step\n"
                      << "  --max_steps N     Maximum time steps\n"
                      << "  --tol T           Convergence tolerance for steady solve\n"
                      << "  --poisson_tol T   Poisson solver tolerance (per solve)\n"
                      << "  --poisson_max_vcycles N  Max MG V-cycles per Poisson solve\n"
                      << "  --poisson S       Poisson solver (auto, fft, fft2d, fft1d, hypre, mg)\n"
                      << "                      auto: FFT -> FFT2D -> FFT1D -> HYPRE -> MG\n"
                      << "                      fft: 2D FFT (3D only, requires periodic x AND z, uniform dx/dz)\n"
                      << "                      fft2d: 2D mesh FFT (2D only, periodic x, walls y, Nz=1)\n"
                      << "                      fft1d: 1D FFT + 2D Helmholtz (3D only, periodic x OR z)\n"
                      << "                      hypre: HYPRE PFMG (requires USE_HYPRE build)\n"
                      << "                      mg: native geometric multigrid (always available)\n"
                      << "  --use_hypre       [deprecated] Same as --poisson=hypre\n"
                      << "  --use_fft         [deprecated] Same as --poisson=fft\n"
                      << "  --poisson_abs_tol_floor V  Absolute tolerance floor for Poisson (default 0)\n"
                      << "  --turb_guard_enabled  Enable turbulence guard (NaN/Inf checks)\n"
                      << "  --turb_guard_interval N  Check interval for turb guard (default 5)\n"
                      << "\n"
                      << "Recycling inflow (turbulent inlet BC for DNS/LES):\n"
                      << "  --recycling_inflow  Enable recycling inflow at x_lo boundary\n"
                      << "  --recycle_x V     x-location of recycle plane (-1 = auto: 10*delta)\n"
                      << "  --recycle_shift_z N  Spanwise shift for decorrelation (-1 = auto: Nz/4)\n"
                      << "  --recycle_shift_interval N  Timesteps between shift updates (default 100)\n"
                      << "  --recycle_filter_tau V  Temporal filter timescale (-1 = disabled)\n"
                      << "  --recycle_fringe_length V  Fringe zone length (-1 = auto: 2*delta)\n"
                      << "  --recycle_target_bulk_u V  Target bulk velocity (-1 = from IC)\n"
                      << "\n"
                      << "  --model M         Turbulence model:\n"
                      << "                      none, baseline, gep, nn_mlp, nn_tbnn\n"
                      << "                      sst, komega (transport models)\n"
                      << "                      earsm_wj, earsm_gs, earsm_pope (EARSM)\n"
                      << "  --nn_preset NAME  Use preset model from data/models/<NAME>\n"
                      << "  --weights DIR     NN weights directory (overrides preset)\n"
                      << "  --scaling DIR     NN scaling directory (overrides preset)\n"
                      << "  --output DIR      Output directory\n"
                      << "  --num_snapshots N Number of VTK snapshots (default 10)\n"
                      << "  --no_postprocess  Skip Poiseuille table + profile output\n"
                      << "  --no_write_fields Skip VTK/field output (snapshots + final)\n"
                      << "  --vtk_ascii       Use ASCII VTK format (default: binary)\n"
                      << "  --stretch         Use stretched mesh in y\n"
                      << "  --adaptive_dt     Enable adaptive time stepping\n"
                      << "  --CFL VALUE       Max CFL number for adaptive dt (default 0.5)\n"
                      << "  --scheme SCHEME   Advection scheme: central (default), upwind, skew, upwind2\n"
                      << "  --integrator I    Time integrator: euler (default), rk2, rk3\n"
                      << "  --space-order N   Spatial discretization order: 2 (default) or 4\n"
                      << "  --simulation_mode MODE  Simulation mode: steady (default), unsteady\n"
                      << "  --perturbation_amplitude A  Initial perturbation amplitude for DNS (default 1e-2)\n"
                      << "  --warmup_steps N  Warmup steps excluded from timing (default 0)\n"
                      << "  --benchmark       Benchmark mode: 192^3 3D duct, upwind, 1 Poisson cycle,\n"
                      << "                      no output/turbulence (all overridable by other flags)\n"
                      << "  --perf_mode       Performance mode: reduce GPU sync overhead for production runs\n"
                      << "                      (sets diag_interval=50, poisson_check_interval=5)\n"
                      << "  --verbose/--quiet Print progress\n"
                      << "  --help            Show this message\n"
                      << "\nPhysical Parameter Coupling:\n"
                      << "  Only TWO of {Re, nu, dp_dx} can be specified independently.\n"
                      << "  The third is computed from Re = sqrt(|dp_dx|) * H / nu.\n"
                      << "  Default: nu = 1.5e-5 (air viscosity), dp_dx = -1.0\n";
            std::exit(0);
        }
    }

    finalize();
}

void Config::finalize() {
    // Ensure output_dir ends with a path separator so it behaves as a directory
    if (!output_dir.empty()) {
        char back = output_dir.back();
        if (back != '/' && back != '\\') {
            output_dir.push_back('/');
        }
    }
    
    // Validate NN model configuration - require explicit model selection
    const bool using_nn = 
        (turb_model == TurbulenceModelType::NNMLP || turb_model == TurbulenceModelType::NNTBNN);
    
    if (using_nn) {
        const bool has_preset = !nn_preset.empty();
        const bool has_weights = !nn_weights_path.empty();
        const bool has_scaling = !nn_scaling_path.empty();
        
        // Require either preset OR explicit paths
        if (!has_preset && !has_weights && !has_scaling) {
            std::cerr << "ERROR: NN turbulence model selected but no model specified.\n"
                      << "Please provide either:\n"
                      << "  --nn_preset <NAME>            (loads from data/models/<NAME>/)\n"
                      << "or:\n"
                      << "  --weights <DIR> --scaling <DIR>\n"
                      << "\nAvailable presets:\n"
                      << "  - tbnn_channel_caseholdout\n"
                      << "  - tbnn_phll_caseholdout\n"
                      << "  - example_tbnn (demo only)\n"
                      << "  - example_scalar_nut (demo only)\n";
            std::exit(1);
        }
        
        // Map preset to paths if not explicitly overridden
        if (has_preset) {
            if (!has_weights) nn_weights_path = "data/models/" + nn_preset;
            if (!has_scaling) nn_scaling_path = "data/models/" + nn_preset;
        } else {
            // Mirror weights/scaling if only one provided (common usage, only when no preset)
            if (has_weights && !has_scaling) nn_scaling_path = nn_weights_path;
            if (!has_weights && has_scaling) nn_weights_path = nn_scaling_path;
        }
    }
    
    // =========================================================================
    // CONFIGURATION VALIDATION
    // =========================================================================

    // Validate 3D-only options when Nz=1 (2D simulation)
    if (Nz == 1) {
        if (stretch_z) {
            std::cerr << "ERROR: Invalid configuration: stretch_z=true with Nz=1 (2D simulation).\n"
                      << "\n"
                      << "The stretch_z option only applies to 3D simulations (Nz > 1).\n"
                      << "For a 2D simulation, remove 'stretch_z' from your configuration.\n";
            std::exit(1);
        }

        // FFT (doubly-periodic) requires 3D
        if (poisson_solver == PoissonSolverType::FFT) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft with Nz=1 (2D simulation).\n"
                      << "\n"
                      << "The FFT solver requires a 3D mesh with periodic boundaries in both x and z.\n"
                      << "For 2D simulations, use one of:\n"
                      << "  --poisson auto    (recommended, auto-selects best solver)\n"
                      << "  --poisson fft2d   (FFT for 2D meshes with periodic x)\n"
                      << "  --poisson hypre   (HYPRE PFMG, requires USE_HYPRE build)\n"
                      << "  --poisson mg      (native multigrid, always available)\n";
            std::exit(1);
        }

        // FFT1D requires 3D
        if (poisson_solver == PoissonSolverType::FFT1D) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft1d with Nz=1 (2D simulation).\n"
                      << "\n"
                      << "The FFT1D solver requires a 3D mesh with periodic boundary in x or z.\n"
                      << "For 2D simulations, use one of:\n"
                      << "  --poisson auto    (recommended, auto-selects best solver)\n"
                      << "  --poisson fft2d   (FFT for 2D meshes with periodic x)\n"
                      << "  --poisson hypre   (HYPRE PFMG, requires USE_HYPRE build)\n"
                      << "  --poisson mg      (native multigrid, always available)\n";
            std::exit(1);
        }
    }

    // Validate 2D-only options when Nz>1 (3D simulation)
    if (Nz > 1) {
        if (poisson_solver == PoissonSolverType::FFT2D) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft2d with Nz=" << Nz << " (3D simulation).\n"
                      << "\n"
                      << "The FFT2D solver is only for 2D meshes (Nz=1).\n"
                      << "For 3D simulations, use one of:\n"
                      << "  --poisson auto    (recommended, auto-selects best solver)\n"
                      << "  --poisson fft     (2D FFT, requires periodic x AND z)\n"
                      << "  --poisson fft1d   (1D FFT, requires periodic x OR z)\n"
                      << "  --poisson hypre   (HYPRE PFMG, requires USE_HYPRE build)\n"
                      << "  --poisson mg      (native multigrid, always available)\n";
            std::exit(1);
        }
    }

    // Validate FFT solvers with stretched meshes
    // FFT solvers require uniform grid spacing in their periodic directions
    if (poisson_solver == PoissonSolverType::FFT) {
        if (stretch_y) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft with stretch_y=true.\n"
                      << "\n"
                      << "The FFT solver requires uniform grid spacing in all directions.\n"
                      << "Y-direction stretching creates non-uniform spacing that is incompatible\n"
                      << "with the FFT algorithm's spectral decomposition.\n"
                      << "\n"
                      << "Options:\n"
                      << "  1. Use --poisson mg (multigrid supports stretched meshes)\n"
                      << "  2. Use --poisson hypre (HYPRE supports stretched meshes)\n"
                      << "  3. Use --poisson auto (will select a compatible solver)\n"
                      << "  4. Disable mesh stretching with stretch_y=false\n";
            std::exit(1);
        }
        if (stretch_z) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft with stretch_z=true.\n"
                      << "\n"
                      << "The FFT solver requires uniform grid spacing in all periodic directions.\n"
                      << "Z-direction stretching is incompatible with the doubly-periodic FFT solver.\n"
                      << "\n"
                      << "Options:\n"
                      << "  1. Use --poisson mg (multigrid supports stretched meshes)\n"
                      << "  2. Use --poisson hypre (HYPRE supports stretched meshes)\n"
                      << "  3. Use --poisson auto (will select a compatible solver)\n"
                      << "  4. Use --poisson fft1d (if only one direction is periodic)\n"
                      << "  5. Disable z-stretching with stretch_z=false\n";
            std::exit(1);
        }
    }

    if (poisson_solver == PoissonSolverType::FFT2D && stretch_y) {
        std::cerr << "ERROR: Invalid configuration: poisson_solver=fft2d with stretch_y=true.\n"
                  << "\n"
                  << "The FFT2D solver requires uniform grid spacing. Y-direction stretching\n"
                  << "is incompatible with the FFT-based pressure solver.\n"
                  << "\n"
                  << "Options:\n"
                  << "  1. Use --poisson mg (multigrid supports stretched meshes)\n"
                  << "  2. Use --poisson hypre (HYPRE supports stretched meshes)\n"
                  << "  3. Use --poisson auto (will select a compatible solver)\n"
                  << "  4. Disable mesh stretching with stretch_y=false\n";
        std::exit(1);
    }

    if (poisson_solver == PoissonSolverType::FFT1D) {
        if (stretch_y) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft1d with stretch_y=true.\n"
                      << "\n"
                      << "The FFT1D solver performs a 1D FFT followed by 2D Helmholtz solves.\n"
                      << "Y-direction stretching would require non-uniform tridiagonal systems\n"
                      << "in the Helmholtz step, which is not currently supported.\n"
                      << "\n"
                      << "Options:\n"
                      << "  1. Use --poisson mg (multigrid supports stretched meshes)\n"
                      << "  2. Use --poisson hypre (HYPRE supports stretched meshes)\n"
                      << "  3. Use --poisson auto (will select a compatible solver)\n"
                      << "  4. Disable mesh stretching with stretch_y=false\n";
            std::exit(1);
        }
        if (stretch_z) {
            std::cerr << "ERROR: Invalid configuration: poisson_solver=fft1d with stretch_z=true.\n"
                      << "\n"
                      << "The FFT1D solver performs a 1D FFT followed by 2D Helmholtz solves.\n"
                      << "Z-direction stretching would require non-uniform systems in the Helmholtz\n"
                      << "step, which is not currently supported.\n"
                      << "\n"
                      << "Options:\n"
                      << "  1. Use --poisson mg (multigrid supports stretched meshes)\n"
                      << "  2. Use --poisson hypre (HYPRE supports stretched meshes)\n"
                      << "  3. Use --poisson auto (will select a compatible solver)\n"
                      << "  4. Disable z-stretching with stretch_z=false\n";
            std::exit(1);
        }
    }

    // Validate turbulence model configuration
    const bool is_transport_model =
        (turb_model == TurbulenceModelType::SSTKOmega ||
         turb_model == TurbulenceModelType::KOmega ||
         turb_model == TurbulenceModelType::EARSM_WJ ||
         turb_model == TurbulenceModelType::EARSM_GS ||
         turb_model == TurbulenceModelType::EARSM_Pope);

    // NOTE: Transport model Re check is done AFTER Reynolds number computation below,
    // so it catches low-Re cases regardless of whether Re was specified directly
    // or computed from (nu, dp_dx).

    // Validate mesh resolution for turbulence models
    // Transport models need reasonable resolution to resolve boundary layers
    if (is_transport_model && Ny < 32) {
        std::cerr << "ERROR: Invalid configuration: transport turbulence model with Ny=" << Ny << ".\n"
                  << "\n"
                  << "Transport-equation turbulence models require adequate wall-normal resolution\n"
                  << "to properly resolve the boundary layer and compute wall-bounded quantities\n"
                  << "like y+ and the turbulent kinetic energy production.\n"
                  << "\n"
                  << "Recommended: Ny >= 64 for RANS turbulence modeling.\n"
                  << "Minimum: Ny >= 32.\n"
                  << "\n"
                  << "If this is intentional (e.g., debugging), use a simpler model:\n"
                  << "  --model none      (laminar)\n"
                  << "  --model baseline  (algebraic mixing length)\n";
        std::exit(1);
    }

    // Validate nu_t_max is positive
    if (nu_t_max <= 0.0) {
        std::cerr << "ERROR: Invalid configuration: nu_t_max=" << nu_t_max << ".\n"
                  << "\n"
                  << "The maximum eddy viscosity clipping value must be positive.\n"
                  << "Typical values are O(1) for channel flows. Default is 1.0.\n";
        std::exit(1);
    }

    // Validate dt is positive (prevents invalid time integration)
    if (dt <= 0.0) {
        std::cerr << "ERROR: Invalid configuration: dt=" << dt << ".\n"
                  << "\n"
                  << "The time step must be a positive value.\n";
        std::exit(1);
    }

    // Validate CFL_max is positive (unconditional - prevents division by zero)
    if (CFL_max <= 0.0) {
        std::cerr << "ERROR: Invalid configuration: CFL_max=" << CFL_max << ".\n"
                  << "\n"
                  << "The maximum CFL number must be a positive value.\n"
                  << "Typical values: 0.5 (conservative) to 1.0 (maximum theoretical limit).\n";
        std::exit(1);
    }

    // Validate CFL_max <= 1 for adaptive time stepping
    if (adaptive_dt && CFL_max > 1.0) {
        std::cerr << "ERROR: Invalid configuration: CFL_max=" << CFL_max << " with adaptive_dt=true.\n"
                  << "\n"
                  << "For stable explicit time integration, CFL must be in (0, 1].\n"
                  << "Recommended values:\n"
                  << "  CFL_max = 0.5  (default, conservative)\n"
                  << "  CFL_max = 0.8  (aggressive but usually stable)\n"
                  << "  CFL_max = 1.0  (maximum theoretical limit)\n";
        std::exit(1);
    }

    // Validate turb_guard_interval
    if (turb_guard_enabled && turb_guard_interval < 1) {
        std::cerr << "ERROR: Invalid configuration: turb_guard_interval=" << turb_guard_interval << ".\n"
                  << "\n"
                  << "The turbulence guard check interval must be >= 1.\n"
                  << "Recommended: 5-10 (checks every N time steps for NaN/Inf values).\n";
        std::exit(1);
    }

    // Validate poisson_abs_tol_floor is non-negative
    if (poisson_abs_tol_floor < 0.0) {
        std::cerr << "ERROR: Invalid configuration: poisson_abs_tol_floor=" << poisson_abs_tol_floor << ".\n"
                  << "\n"
                  << "The absolute tolerance floor must be non-negative.\n"
                  << "This sets a minimum threshold for the Poisson solver convergence check.\n"
                  << "Typical values: 0.0 (disabled) or 1e-12 to 1e-10.\n";
        std::exit(1);
    }

    // Validate poisson_abs_tol_floor <= poisson_tol (floor should not override tolerance)
    if (poisson_abs_tol_floor > poisson_tol) {
        std::cerr << "ERROR: Invalid configuration: poisson_abs_tol_floor=" << poisson_abs_tol_floor
                  << " exceeds poisson_tol=" << poisson_tol << ".\n"
                  << "\n"
                  << "The absolute tolerance floor must be <= poisson_tol, otherwise it becomes\n"
                  << "the effective stopping criterion for the Poisson solver.\n";
        std::exit(1);
    }

    // =========================================================================

    // Reynolds number coupling for channel flow
    // For laminar Poiseuille: U_bulk = -dp_dx * delta^2 / (3*nu)
    // Re = U_bulk * delta / nu = -dp_dx * delta^3 / (3*nu^2)
    // Therefore: dp_dx = -3 * Re * nu^2 / delta^3
    //        or: nu = sqrt(-dp_dx * delta^3 / (3 * Re))
    
    double delta = (y_max - y_min) / 2.0;
    
    // Check for over-constrained input (all three specified)
    if (Re_specified && nu_specified && dp_dx_specified) {
        // Verify consistency: compute what Re would be from nu and dp_dx
        double Re_check = -dp_dx * delta * delta * delta / (3.0 * nu * nu);
        double relative_error = std::abs(Re_check - Re) / Re;
        
        if (relative_error > 0.01) { // 1% tolerance
            std::cerr << "ERROR: Over-constrained input! You specified all three:\n"
                      << "  --Re " << Re << "\n"
                      << "  --nu " << nu << "\n"
                      << "  --dp_dx " << dp_dx << "\n"
                      << "But these are inconsistent (computed Re = " << Re_check << ").\n"
                      << "\n"
                      << "Please specify only TWO of (Re, nu, dp_dx):\n"
                      << "  --Re --nu        → code computes dp_dx\n"
                      << "  --Re --dp_dx     → code computes nu\n"
                      << "  --nu --dp_dx     → code computes Re\n";
            std::exit(1);
        } else {
            // All three specified and consistent - just use them
            if (verbose) {
                std::cout << "Note: All three (Re, nu, dp_dx) specified and are consistent.\n";
            }
        }
    }
    // Case 1: User specified Re but not nu → compute nu from Re and dp_dx
    else if (Re_specified && !nu_specified) {
        if (!dp_dx_specified || dp_dx >= 0.0) {
            if (verbose) {
                std::cout << "Re specified without nu or dp_dx. Using default dp_dx = -1.0\n";
            }
            if (dp_dx >= 0.0) {
                dp_dx = -1.0;
            }
        }
        // nu = sqrt(-dp_dx * delta^3 / (3 * Re))
        nu = std::sqrt(-dp_dx * delta * delta * delta / (3.0 * Re));
        if (verbose) {
            std::cout << "Computing nu from Re and dp_dx: nu = " << std::scientific << nu 
                      << " (Re = " << Re << ", dp_dx = " << dp_dx << ")\n" << std::defaultfloat;
        }
    }
    // Case 2: User specified Re and nu → compute dp_dx to achieve desired Re
    else if (Re_specified && nu_specified && !dp_dx_specified) {
        // dp_dx = -3 * Re * nu^2 / delta^3
        dp_dx = -3.0 * Re * nu * nu / (delta * delta * delta);
        if (verbose) {
            std::cout << "Computing dp_dx from Re and nu: dp_dx = " << dp_dx 
                      << " (Re = " << Re << ", nu = " << std::scientific << nu << ")\n" << std::defaultfloat;
        }
    }
    // Case 3: User specified nu and dp_dx → compute Re from these
    else if (nu_specified && dp_dx_specified && !Re_specified) {
        // Re = -dp_dx * delta^3 / (3 * nu^2)
        if (dp_dx < 0.0) {
            Re = -dp_dx * delta * delta * delta / (3.0 * nu * nu);
            if (verbose) {
                std::cout << "Computing Re from nu and dp_dx: Re = " << Re 
                          << " (nu = " << std::scientific << nu << ", dp_dx = " << dp_dx << ")\n" << std::defaultfloat;
            }
        }
    }
    // Case 4: Only one or none specified - use defaults and compute Re
    else {
        // Compute Re from defaults
        if (dp_dx < 0.0) {
            Re = -dp_dx * delta * delta * delta / (3.0 * nu * nu);
            if (verbose) {
                std::cout << "Using default parameters: nu = " << std::scientific << nu
                          << " (air at 20°C), dp_dx = " << dp_dx << "\n"
                          << "Computed Re = " << Re << "\n" << std::defaultfloat;
            }
        }
    }

    // =========================================================================
    // POST-COMPUTATION VALIDATION
    // =========================================================================

    // Transport models with very low Reynolds numbers may have numerical issues
    // The k-omega models are designed for turbulent flows (Re > ~2000 for channels)
    // This check uses the COMPUTED Re, so it catches low-Re cases regardless of
    // whether Re was specified directly or computed from (nu, dp_dx).
    if (is_transport_model && Re < 500) {
        std::cerr << "ERROR: Invalid configuration: transport turbulence model with Re=" << Re << ".\n"
                  << "\n"
                  << "Transport-equation turbulence models (k-omega, SST, EARSM) are designed\n"
                  << "for turbulent flows. At Re=" << Re << ", the flow is likely laminar or\n"
                  << "transitional, and these models may produce numerical instabilities.\n"
                  << "\n"
                  << "For low Reynolds number flows, use one of:\n"
                  << "  --model none      (laminar, no turbulence model)\n"
                  << "  --model baseline  (simple mixing length, if mild turbulence expected)\n";
        std::exit(1);
    }

    // =========================================================================
    // PERFORMANCE MODE - reduce GPU sync overhead for production runs
    // =========================================================================
    if (perf_mode) {
        // Only override if user didn't explicitly set these values
        if (diag_interval == 1) {
            diag_interval = 50;  // Reduce div norm computation frequency
        }
        if (poisson_check_interval == 3) {  // 3 is the new default
            poisson_check_interval = 5;  // Further reduce MG sync frequency
        }
        // Note: NaN/Inf guard still runs at turb_guard_interval (default=5)
    }
}

void Config::print() const {
    double delta = (y_max - y_min) / 2.0;
    double Re_actual = -dp_dx * delta * delta * delta / (3.0 * nu * nu);
    
    std::cout << "=== Configuration ===\n";
    if (Nz > 1) {
        std::cout << "Mesh: " << Nx << " x " << Ny << " x " << Nz << " (3D)\n"
                  << "Domain: [" << x_min << ", " << x_max << "] x ["
                  << y_min << ", " << y_max << "] x ["
                  << z_min << ", " << z_max << "]\n"
                  << "Stretched y: " << (stretch_y ? "yes" : "no")
                  << ", z: " << (stretch_z ? "yes" : "no") << "\n";
    } else {
        std::cout << "Mesh: " << Nx << " x " << Ny << "\n"
                  << "Domain: [" << x_min << ", " << x_max << "] x ["
                  << y_min << ", " << y_max << "]\n"
                  << "Stretched y: " << (stretch_y ? "yes" : "no") << "\n";
    }
    std::cout << "Physical: Re = " << Re << " (actual: " << Re_actual << "), nu = " << nu << "\n"
              << "dp/dx: " << dp_dx << "\n"
              << "Time integrator: ";
    switch (time_integrator) {
        case TimeIntegrator::Euler: std::cout << "Euler (1st order)"; break;
        case TimeIntegrator::RK2: std::cout << "SSP-RK2 (2nd order)"; break;
        case TimeIntegrator::RK3: std::cout << "SSP-RK3 (3rd order)"; break;
    }
    std::cout << " + Projection\n"
              << "Poisson solver: ";
    switch (poisson_solver) {
        case PoissonSolverType::Auto: std::cout << "Auto"; break;
        case PoissonSolverType::FFT: std::cout << "FFT (doubly-periodic)"; break;
        case PoissonSolverType::FFT2D: std::cout << "FFT2D (2D mesh)"; break;
        case PoissonSolverType::FFT1D: std::cout << "FFT1D (singly-periodic)"; break;
        case PoissonSolverType::HYPRE: std::cout << "HYPRE PFMG"; break;
        case PoissonSolverType::MG: std::cout << "Multigrid"; break;
    }
    std::cout << "\n"
              << "Advection scheme: ";
    switch (convective_scheme) {
        case ConvectiveScheme::Central: std::cout << "Central (2nd-order)"; break;
        case ConvectiveScheme::Upwind: std::cout << "Upwind (1st-order)"; break;
        case ConvectiveScheme::Skew: std::cout << "Skew-symmetric (energy-conserving)"; break;
        case ConvectiveScheme::Upwind2: std::cout << "Upwind (2nd-order)"; break;
    }
    std::cout << "\n"
              << "Spatial order: O" << space_order << "\n"
              << "dt: " << dt << ", max_steps: " << max_steps << ", tol: " << tol << "\n"
              << "Turbulence model: ";
    
    switch (turb_model) {
        case TurbulenceModelType::None: std::cout << "None (laminar)"; break;
        case TurbulenceModelType::Baseline: std::cout << "Baseline"; break;
        case TurbulenceModelType::GEP: std::cout << "GEP (Weatheritt-Sandberg)"; break;
        case TurbulenceModelType::NNMLP: std::cout << "NN-MLP"; break;
        case TurbulenceModelType::NNTBNN: std::cout << "NN-TBNN"; break;
        case TurbulenceModelType::SSTKOmega: std::cout << "SST k-omega"; break;
        case TurbulenceModelType::KOmega: std::cout << "k-omega (Wilcox)"; break;
        case TurbulenceModelType::EARSM_WJ: std::cout << "SST + Wallin-Johansson EARSM"; break;
        case TurbulenceModelType::EARSM_GS: std::cout << "SST + Gatski-Speziale EARSM"; break;
        case TurbulenceModelType::EARSM_Pope: std::cout << "SST + Pope Quadratic EARSM"; break;
    }
    std::cout << "\n";
    
    if (turb_model == TurbulenceModelType::NNMLP || turb_model == TurbulenceModelType::NNTBNN) {
        if (!nn_preset.empty()) {
            std::cout << "NN preset: " << nn_preset << "\n";
        }
        std::cout << "NN weights: " << nn_weights_path << "\n"
                  << "NN scaling: " << nn_scaling_path << "\n";
    }
    
    std::cout << "=====================\n";
}

} // namespace nncfd


