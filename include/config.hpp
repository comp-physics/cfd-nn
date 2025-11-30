#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nncfd {

/// Turbulence model selection
enum class TurbulenceModelType {
    None,       ///< Laminar (no turbulence model)
    Baseline,   ///< Simple algebraic or k-omega like model
    GEP,        ///< Gene Expression Programming algebraic model (Weatheritt-Sandberg 2016)
    NNMLP,      ///< Neural network scalar eddy viscosity
    NNTBNN      ///< TBNN-style anisotropy model
};

/// Convective scheme selection
enum class ConvectiveScheme {
    Central,
    Upwind,
    QUICK
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
    
    // Physical parameters
    double Re = 1000.0;         ///< Reynolds number (based on channel half-height and bulk velocity)
    double nu = 0.001;          ///< Kinematic viscosity (computed from Re)
    double rho = 1.0;           ///< Density
    double dp_dx = -1.0;        ///< Pressure gradient (or body force) driving the flow
    
    // Time stepping
    double dt = 0.001;          ///< Time step
    double CFL_max = 0.5;       ///< Maximum CFL number for adaptive dt
    bool adaptive_dt = false;   ///< Use adaptive time stepping
    int max_iter = 10000;       ///< Maximum pseudo-time iterations
    double tol = 1e-6;          ///< Convergence tolerance
    
    // Numerical schemes
    ConvectiveScheme convective_scheme = ConvectiveScheme::Central;
    
    // Turbulence model
    TurbulenceModelType turb_model = TurbulenceModelType::None;
    double nu_t_max = 1.0;      ///< Maximum eddy viscosity (clipping)
    double blend_alpha = 1.0;   ///< Blending factor for NN (0=baseline, 1=NN)
    
    // NN model paths
    std::string nn_weights_path = "data/";
    std::string nn_scaling_path = "data/";
    std::string nn_preset;      ///< Optional preset model name (e.g., "ling_tbnn_2016")
    
    // Output
    std::string output_dir = "output/";
    int output_freq = 100;      ///< Output frequency (iterations)
    bool verbose = true;
    
    // Poisson solver
    double poisson_tol = 1e-6;
    int poisson_max_iter = 10000;
    double poisson_omega = 1.8; ///< SOR relaxation parameter
    
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


