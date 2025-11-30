#include "config.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>

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
    Re = get_double("Re", Re);
    nu = get_double("nu", nu);
    rho = get_double("rho", rho);
    dp_dx = get_double("dp_dx", dp_dx);
    
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
    } else if (scheme_str == "quick") {
        convective_scheme = ConvectiveScheme::QUICK;
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
    verbose = get_bool("verbose", verbose);
    
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
        } else if (arg == "--nu" && i + 1 < argc) {
            nu = std::stod(argv[++i]);
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
            }
        } else if (arg == "--weights" && i + 1 < argc) {
            nn_weights_path = argv[++i];
        } else if (arg == "--scaling" && i + 1 < argc) {
            nn_scaling_path = argv[++i];
        } else if (arg == "--nn_preset" && i + 1 < argc) {
            nn_preset = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quiet") {
            verbose = false;
        } else if (arg == "--stretch") {
            stretch_y = true;
        } else if (arg == "--adaptive_dt") {
            adaptive_dt = true;
        } else if (arg == "--CFL" && i + 1 < argc) {
            CFL_max = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --config FILE     Load config file\n"
                      << "  --Nx N            Grid cells in x\n"
                      << "  --Ny N            Grid cells in y\n"
                      << "  --Re R            Reynolds number\n"
                      << "  --nu V            Kinematic viscosity\n"
                      << "  --dt T            Time step\n"
                      << "  --max_iter N      Maximum iterations\n"
                      << "  --tol T           Convergence tolerance\n"
                      << "  --model M         Turbulence model (none, baseline, gep, nn_mlp, nn_tbnn)\n"
                      << "  --nn_preset NAME  Use preset model from data/models/<NAME>\n"
                      << "  --weights DIR     NN weights directory (overrides preset)\n"
                      << "  --scaling DIR     NN scaling directory (overrides preset)\n"
                      << "  --output DIR      Output directory\n"
                      << "  --stretch         Use stretched mesh in y\n"
                      << "  --adaptive_dt     Enable adaptive time stepping\n"
                      << "  --CFL VALUE       Max CFL number for adaptive dt (default 0.5)\n"
                      << "  --verbose/--quiet Print progress\n"
                      << "  --help            Show this message\n";
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
    
    // Compute nu from Re if Re is specified and nu is default
    // Convention: Re = U_bulk * delta / nu, where delta = (y_max - y_min) / 2
    double delta = (y_max - y_min) / 2.0;
    (void)delta;
    
    // If user specifies Re and default nu, compute nu
    // Otherwise keep nu as specified
    // For simplicity, we'll use: nu = U_bulk * delta / Re
    // Assume U_bulk will be determined from dp_dx and nu for Poiseuille
}

void Config::print() const {
    std::cout << "=== Configuration ===\n"
              << "Mesh: " << Nx << " x " << Ny << "\n"
              << "Domain: [" << x_min << ", " << x_max << "] x [" << y_min << ", " << y_max << "]\n"
              << "Stretched y: " << (stretch_y ? "yes" : "no") << "\n"
              << "Re: " << Re << ", nu: " << nu << "\n"
              << "dp/dx: " << dp_dx << "\n"
              << "dt: " << dt << ", max_iter: " << max_iter << ", tol: " << tol << "\n"
              << "Turbulence model: ";
    
    switch (turb_model) {
        case TurbulenceModelType::None: std::cout << "None (laminar)"; break;
        case TurbulenceModelType::Baseline: std::cout << "Baseline"; break;
        case TurbulenceModelType::GEP: std::cout << "GEP (Weatheritt-Sandberg)"; break;
        case TurbulenceModelType::NNMLP: std::cout << "NN-MLP"; break;
        case TurbulenceModelType::NNTBNN: std::cout << "NN-TBNN"; break;
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


