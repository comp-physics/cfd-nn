/// Unit tests for configuration parsing and validation
///
/// Tests the Config class (config.cpp) which handles:
/// - Config file parsing (.cfg format)
/// - Type conversion (int, double, bool, enum)
/// - Validation rules (mesh-solver compatibility, time stepping, turbulence)
/// - Reynolds number coupling (Re, nu, dp_dx)
/// - Command-line argument parsing

#include "config.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

using namespace nncfd;
using nncfd::test::harness::record;

namespace {
// Helper to create temporary config files
std::string create_temp_config(const std::string& content) {
    static int counter = 0;
    std::string filename = "/tmp/test_config_" + std::to_string(counter++) + ".cfg";
    std::ofstream file(filename);
    file << content;
    file.close();
    return filename;
}

// Helper to clean up temp files
void remove_temp_file(const std::string& filename) {
    std::remove(filename.c_str());
}
} // namespace

// ============================================================================
// Config File Parsing Tests
// ============================================================================

void test_parse_basic_key_value() {
    std::string cfg = R"(
Nx = 128
Ny = 64
nu = 0.005
verbose = true
)";
    std::string filename = create_temp_config(cfg);

    auto params = parse_config_file(filename);

    bool pass = (params["Nx"] == "128");
    pass = pass && (params["Ny"] == "64");
    pass = pass && (params["nu"] == "0.005");
    pass = pass && (params["verbose"] == "true");

    remove_temp_file(filename);
    record("Basic key=value parsing", pass);
}

void test_parse_whitespace_handling() {
    std::string cfg = R"(
  Nx   =   128
Ny=64
   nu =0.005
)";
    std::string filename = create_temp_config(cfg);

    auto params = parse_config_file(filename);

    bool pass = (params["Nx"] == "128");
    pass = pass && (params["Ny"] == "64");
    pass = pass && (params["nu"] == "0.005");

    remove_temp_file(filename);
    record("Whitespace handling", pass);
}

void test_parse_comments() {
    std::string cfg = R"(
# This is a comment
Nx = 32
# Another comment
Ny = 64  # Inline comments are NOT supported (value includes the #)
)";
    std::string filename = create_temp_config(cfg);

    auto params = parse_config_file(filename);

    bool pass = (params["Nx"] == "32");
    // Note: inline comments become part of value (current behavior)
    pass = pass && (params.find("Ny") != params.end());

    remove_temp_file(filename);
    record("Comment handling", pass);
}

void test_parse_empty_file() {
    std::string cfg = "";
    std::string filename = create_temp_config(cfg);

    auto params = parse_config_file(filename);

    bool pass = params.empty();

    remove_temp_file(filename);
    record("Empty file parsing", pass);
}

void test_parse_missing_equals() {
    std::string cfg = R"(
Nx = 32
invalid line without equals
Ny = 64
)";
    std::string filename = create_temp_config(cfg);

    auto params = parse_config_file(filename);

    // Lines without = are silently skipped
    bool pass = (params["Nx"] == "32");
    pass = pass && (params["Ny"] == "64");
    pass = pass && (params.size() == 2);

    remove_temp_file(filename);
    record("Lines without equals sign", pass);
}

void test_parse_nonexistent_file() {
    bool threw = false;
    try {
        parse_config_file("/nonexistent/path/config.cfg");
    } catch (const std::runtime_error&) {
        threw = true;
    }

    record("Nonexistent file error", threw);
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

void test_load_integers() {
    std::string cfg = R"(
Nx = 128
Ny = 256
Nz = 32
max_iter = 5000
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.load(filename);

    bool pass = (config.Nx == 128);
    pass = pass && (config.Ny == 256);
    pass = pass && (config.Nz == 32);
    pass = pass && (config.max_iter == 5000);

    remove_temp_file(filename);
    record("Integer loading", pass);
}

void test_load_doubles() {
    std::string cfg = R"(
nu = 1.5e-5
dt = 0.001
CFL_max = 0.8
poisson_tol = 1e-8
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.load(filename);

    bool pass = (std::abs(config.nu - 1.5e-5) < 1e-10);
    pass = pass && (std::abs(config.dt - 0.001) < 1e-10);
    pass = pass && (std::abs(config.CFL_max - 0.8) < 1e-10);
    pass = pass && (std::abs(config.poisson_tol - 1e-8) < 1e-12);

    remove_temp_file(filename);
    record("Double loading", pass);
}

void test_load_booleans() {
    // Test "true" string
    std::string cfg1 = "verbose = true\nadaptive_dt = 1\nstretch_y = false\n";
    std::string filename1 = create_temp_config(cfg1);

    Config config1;
    config1.load(filename1);

    bool pass = (config1.verbose == true);
    pass = pass && (config1.adaptive_dt == true);
    pass = pass && (config1.stretch_y == false);

    remove_temp_file(filename1);
    record("Boolean loading", pass);
}

void test_load_enum_turb_model() {
    // Test various turbulence model strings
    std::vector<std::pair<std::string, TurbulenceModelType>> test_cases = {
        {"none", TurbulenceModelType::None},
        {"baseline", TurbulenceModelType::Baseline},
        {"gep", TurbulenceModelType::GEP},
        {"sst", TurbulenceModelType::SSTKOmega},
        {"sst_komega", TurbulenceModelType::SSTKOmega},
        {"komega", TurbulenceModelType::KOmega},
        {"k-omega", TurbulenceModelType::KOmega},
        {"earsm_wj", TurbulenceModelType::EARSM_WJ},
        {"wallin_johansson", TurbulenceModelType::EARSM_WJ},
        {"earsm_gs", TurbulenceModelType::EARSM_GS},
        {"gatski_speziale", TurbulenceModelType::EARSM_GS},
        {"earsm_pope", TurbulenceModelType::EARSM_Pope},
        {"pope", TurbulenceModelType::EARSM_Pope},
    };

    bool pass = true;
    for (const auto& [str, expected] : test_cases) {
        std::string cfg = "turb_model = " + str + "\n";
        // For transport models, ensure sufficient resolution and Re
        cfg += "Ny = 64\nRe = 1000\nnu = 0.001\n";
        std::string filename = create_temp_config(cfg);

        Config config;
        config.load(filename);

        if (config.turb_model != expected) {
            pass = false;
        }

        remove_temp_file(filename);
    }

    record("Turbulence model enum loading", pass);
}

void test_load_enum_poisson_solver() {
    std::vector<std::pair<std::string, PoissonSolverType>> test_cases = {
        {"auto", PoissonSolverType::Auto},
        {"mg", PoissonSolverType::MG},
        {"multigrid", PoissonSolverType::MG},
        {"hypre", PoissonSolverType::HYPRE},
        {"fft2d", PoissonSolverType::FFT2D},
    };

    bool pass = true;
    for (const auto& [str, expected] : test_cases) {
        std::string cfg = "poisson_solver = " + str + "\n";
        std::string filename = create_temp_config(cfg);

        Config config;
        config.load(filename);

        if (config.poisson_solver != expected) {
            pass = false;
        }

        remove_temp_file(filename);
    }

    record("Poisson solver enum loading", pass);
}

void test_load_enum_convective_scheme() {
    std::string cfg1 = "convective_scheme = central\n";
    std::string filename1 = create_temp_config(cfg1);
    Config config1;
    config1.load(filename1);
    bool pass = (config1.convective_scheme == ConvectiveScheme::Central);
    remove_temp_file(filename1);

    std::string cfg2 = "convective_scheme = upwind\n";
    std::string filename2 = create_temp_config(cfg2);
    Config config2;
    config2.load(filename2);
    pass = pass && (config2.convective_scheme == ConvectiveScheme::Upwind);
    remove_temp_file(filename2);

    record("Convective scheme enum loading", pass);
}

void test_load_enum_simulation_mode() {
    std::string cfg1 = "simulation_mode = steady\n";
    std::string filename1 = create_temp_config(cfg1);
    Config config1;
    config1.load(filename1);
    bool pass = (config1.simulation_mode == SimulationMode::Steady);
    remove_temp_file(filename1);

    std::string cfg2 = "simulation_mode = unsteady\n";
    std::string filename2 = create_temp_config(cfg2);
    Config config2;
    config2.load(filename2);
    pass = pass && (config2.simulation_mode == SimulationMode::Unsteady);
    remove_temp_file(filename2);

    record("Simulation mode enum loading", pass);
}

// ============================================================================
// Reynolds Number Coupling Tests
// ============================================================================

void test_re_nu_coupling() {
    // Specify Re and nu, code should compute dp_dx
    std::string cfg = R"(
Re = 1000
nu = 0.001
y_min = -1.0
y_max = 1.0
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    // dp_dx = -3 * Re * nu² / delta³
    // delta = (y_max - y_min) / 2 = 1.0
    // dp_dx = -3 * 1000 * 0.001² / 1³ = -0.003
    double expected_dp_dx = -3.0 * 1000.0 * 0.001 * 0.001 / 1.0;

    bool pass = (std::abs(config.dp_dx - expected_dp_dx) < 1e-10);

    remove_temp_file(filename);
    record("Re+nu to dp_dx coupling", pass);
}

void test_re_dpdx_coupling() {
    std::string cfg = R"(
Re = 500
dp_dx = -1.0
y_min = -1.0
y_max = 1.0
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    // nu = sqrt(-dp_dx * delta³ / (3 * Re))
    // delta = 1.0
    // nu = sqrt(1.0 * 1.0 / (3 * 500)) = sqrt(1/1500) ≈ 0.0258
    double expected_nu = std::sqrt(1.0 / (3.0 * 500.0));

    bool pass = (std::abs(config.nu - expected_nu) < 1e-10);

    remove_temp_file(filename);
    record("Re+dp_dx to nu coupling", pass);
}

void test_nu_dpdx_coupling() {
    std::string cfg = R"(
nu = 0.01
dp_dx = -0.5
y_min = -1.0
y_max = 1.0
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    // Re = -dp_dx * delta³ / (3 * nu²)
    // delta = 1.0
    // Re = 0.5 * 1.0 / (3 * 0.0001) = 0.5 / 0.0003 ≈ 1666.67
    double expected_Re = 0.5 / (3.0 * 0.01 * 0.01);

    bool pass = (std::abs(config.Re - expected_Re) < 1e-6);

    remove_temp_file(filename);
    record("nu+dp_dx to Re coupling", pass);
}

// ============================================================================
// Validation Tests (these should NOT exit, just verify the logic works)
// ============================================================================

void test_valid_config_basic() {
    std::string cfg = R"(
Nx = 64
Ny = 64
nu = 0.001
dt = 0.001
CFL_max = 0.5
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);  // Should not throw or exit

    bool pass = (config.Nx == 64);
    pass = pass && (config.Ny == 64);

    remove_temp_file(filename);
    record("Valid basic configuration", pass);
}

void test_valid_config_3d() {
    std::string cfg = R"(
Nx = 32
Ny = 64
Nz = 16
poisson_solver = mg
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    bool pass = (config.Nz == 16);
    pass = pass && (config.poisson_solver == PoissonSolverType::MG);

    remove_temp_file(filename);
    record("Valid 3D configuration", pass);
}

void test_valid_config_stretched_mg() {
    // Stretched mesh is valid with MG solver
    std::string cfg = R"(
Nx = 32
Ny = 64
stretch_y = true
stretch_beta = 2.5
poisson_solver = mg
)";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    bool pass = (config.stretch_y == true);
    pass = pass && (std::abs(config.stretch_beta - 2.5) < 1e-10);

    remove_temp_file(filename);
    record("Stretched mesh with multigrid", pass);
}

void test_output_dir_normalization() {
    // Without trailing slash
    std::string cfg1 = "output_dir = results\n";
    std::string filename1 = create_temp_config(cfg1);
    Config config1;
    config1.verbose = false;
    config1.load(filename1);
    bool pass = (config1.output_dir == "results/");
    remove_temp_file(filename1);

    // With trailing slash
    std::string cfg2 = "output_dir = results/\n";
    std::string filename2 = create_temp_config(cfg2);
    Config config2;
    config2.verbose = false;
    config2.load(filename2);
    pass = pass && (config2.output_dir == "results/");
    remove_temp_file(filename2);

    record("Output directory normalization", pass);
}

void test_default_values() {
    // Empty config should use all defaults
    std::string cfg = "";
    std::string filename = create_temp_config(cfg);

    Config config;
    config.verbose = false;
    config.load(filename);

    // Check some defaults
    bool pass = (config.Nx == 64);
    pass = pass && (config.Ny == 64);
    pass = pass && (config.Nz == 1);
    pass = pass && (std::abs(config.CFL_max - 0.5) < 1e-10);
    pass = pass && (config.adaptive_dt == true);
    pass = pass && (config.turb_model == TurbulenceModelType::None);
    pass = pass && (config.poisson_solver == PoissonSolverType::Auto);
    pass = pass && (config.convective_scheme == ConvectiveScheme::Central);
    pass = pass && (config.simulation_mode == SimulationMode::Steady);

    remove_temp_file(filename);
    record("Default values", pass);
}

void test_legacy_flags() {
    // use_hypre should map to poisson_solver = HYPRE
    std::string cfg1 = "use_hypre = true\n";
    std::string filename1 = create_temp_config(cfg1);
    Config config1;
    config1.verbose = false;
    config1.load(filename1);
    bool pass = (config1.poisson_solver == PoissonSolverType::HYPRE);
    remove_temp_file(filename1);

    // use_fft should map to poisson_solver = FFT
    // Note: FFT requires 3D, so we need Nz > 1
    std::string cfg2 = "use_fft = true\nNz = 16\n";
    std::string filename2 = create_temp_config(cfg2);
    Config config2;
    config2.verbose = false;
    config2.load(filename2);
    pass = pass && (config2.poisson_solver == PoissonSolverType::FFT);
    remove_temp_file(filename2);

    record("Legacy flags", pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Configuration Module Tests", [] {
        // Parsing tests
        test_parse_basic_key_value();
        test_parse_whitespace_handling();
        test_parse_comments();
        test_parse_empty_file();
        test_parse_missing_equals();
        test_parse_nonexistent_file();

        // Type conversion tests
        test_load_integers();
        test_load_doubles();
        test_load_booleans();
        test_load_enum_turb_model();
        test_load_enum_poisson_solver();
        test_load_enum_convective_scheme();
        test_load_enum_simulation_mode();

        // Reynolds number coupling tests
        test_re_nu_coupling();
        test_re_dpdx_coupling();
        test_nu_dpdx_coupling();

        // Validation tests (valid configs)
        test_valid_config_basic();
        test_valid_config_3d();
        test_valid_config_stretched_mg();
        test_output_dir_normalization();
        test_default_values();
        test_legacy_flags();
    });
}
