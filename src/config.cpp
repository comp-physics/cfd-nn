#include "config.hpp"
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
    max_iter = get_int("max_iter", max_iter);
    tol = get_double("tol", tol);
    
    // Numerical scheme
    auto scheme_str = get_string("convective_scheme", "central");
    if (scheme_str == "upwind") {
        convective_scheme = ConvectiveScheme::Upwind;
    } else {
        convective_scheme = ConvectiveScheme::Central;
    }
    
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
    blend_alpha = get_double("blend_alpha", blend_alpha);
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
    
    // Poisson
    poisson_tol = get_double("poisson_tol", poisson_tol);
    poisson_max_iter = get_int("poisson_max_iter", poisson_max_iter);
    poisson_omega = get_double("poisson_omega", poisson_omega);
    
    finalize();
}

void Config::parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--config" && i + 1 < argc) {
            load(argv[++i]);
        } else if (arg == "--Nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        } else if (arg == "--Ny" && i + 1 < argc) {
            Ny = std::stoi(argv[++i]);
        } else if (arg == "--Re" && i + 1 < argc) {
            Re = std::stod(argv[++i]);
            Re_specified = true;
        } else if (arg == "--nu" && i + 1 < argc) {
            nu = std::stod(argv[++i]);
            nu_specified = true;
        } else if (arg == "--dp_dx" && i + 1 < argc) {
            dp_dx = std::stod(argv[++i]);
            dp_dx_specified = true;
        } else if (arg == "--dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        } else if (arg == "--max_iter" && i + 1 < argc) {
            max_iter = std::stoi(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            tol = std::stod(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            std::string model = argv[++i];
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
        } else if (arg == "--weights" && i + 1 < argc) {
            nn_weights_path = argv[++i];
        } else if (arg == "--scaling" && i + 1 < argc) {
            nn_scaling_path = argv[++i];
        } else if (arg == "--nn_preset" && i + 1 < argc) {
            nn_preset = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--num_snapshots" && i + 1 < argc) {
            num_snapshots = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            verbose = false;
        } else if (arg == "--no_postprocess") {
            postprocess = false;
        } else if (arg == "--no_write_fields") {
            write_fields = false;
        } else if (arg == "--stretch") {
            stretch_y = true;
        } else if (arg == "--adaptive_dt") {
            adaptive_dt = true;
        } else if (arg == "--CFL" && i + 1 < argc) {
            CFL_max = std::stod(argv[++i]);
        } else if (arg == "--scheme" && i + 1 < argc) {
            std::string scheme = argv[++i];
            if (scheme == "upwind") {
                convective_scheme = ConvectiveScheme::Upwind;
            } else {
                convective_scheme = ConvectiveScheme::Central;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --config FILE     Load config file\n"
                      << "  --Nx N            Grid cells in x\n"
                      << "  --Ny N            Grid cells in y\n"
                      << "  --Re R            Reynolds number (auto-computes nu or dp_dx)\n"
                      << "  --nu V            Kinematic viscosity\n"
                      << "  --dp_dx D         Pressure gradient (driving force)\n"
                      << "  --dt T            Time step\n"
                      << "  --max_iter N      Maximum iterations\n"
                      << "  --tol T           Convergence tolerance\n"
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
                      << "  --stretch         Use stretched mesh in y\n"
                      << "  --adaptive_dt     Enable adaptive time stepping\n"
                      << "  --CFL VALUE       Max CFL number for adaptive dt (default 0.5)\n"
                      << "  --scheme SCHEME   Convective scheme: central (default), upwind\n"
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
    // Map nn_preset to weights/scaling paths if provided
    // Convention: data/models/<preset>/ contains the weights
    if (!nn_preset.empty()) {
        // Check if weights_path was explicitly set via --weights
        // If not, use the preset path
        if (nn_weights_path == "data/") {
            nn_weights_path = "data/models/" + nn_preset;
        }
        if (nn_scaling_path == "data/") {
            nn_scaling_path = "data/models/" + nn_preset;
        }
    }
    
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
}

void Config::print() const {
    double delta = (y_max - y_min) / 2.0;
    double Re_actual = -dp_dx * delta * delta * delta / (3.0 * nu * nu);
    
    std::cout << "=== Configuration ===\n"
              << "Mesh: " << Nx << " x " << Ny << "\n"
              << "Domain: [" << x_min << ", " << x_max << "] x [" << y_min << ", " << y_max << "]\n"
              << "Stretched y: " << (stretch_y ? "yes" : "no") << "\n"
              << "Physical: Re = " << Re << " (actual: " << Re_actual << "), nu = " << nu << "\n"
              << "dp/dx: " << dp_dx << "\n"
              << "Time stepping: Explicit Euler + Projection\n"
              << "Poisson solver: Multigrid (warm-start enabled)\n"
              << "Convective scheme: " 
              << (convective_scheme == ConvectiveScheme::Upwind ? "Upwind" : "Central") << "\n"
              << "dt: " << dt << ", max_iter: " << max_iter << ", tol: " << tol << "\n"
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
                  << "NN scaling: " << nn_scaling_path << "\n"
                  << "Blend alpha: " << blend_alpha << "\n";
    }
    
    std::cout << "=====================\n";
}

} // namespace nncfd


