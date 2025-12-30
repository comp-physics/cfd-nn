/// Golden File Regression Tests
/// Tests that solver outputs match pre-computed reference files
/// Usage: test_golden <case_name> [--regenerate]
///
/// Cases:
///   channel_komega    - 2D channel with SST k-omega
///   channel_earsm     - 2D channel with EARSM-WJ
///   channel_mlp       - 2D channel with NN-MLP
///   channel_tbnn      - 2D channel with NN-TBNN
///   mixing_length     - 2D channel with mixing length model
///   laminar_3d        - 3D duct with laminar flow

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>

using namespace nncfd;

// Tolerance for golden file comparison (1e-10 = near-bitwise)
constexpr double GOLDEN_TOL = 1e-10;

// Get path relative to executable (handles running from build dir)
std::string get_golden_dir() {
    // Try relative paths from common build locations
    std::vector<std::string> paths = {
        "../tests/golden/",
        "../../tests/golden/",
        "tests/golden/",
        "../../../tests/golden/"
    };
    for (const auto& p : paths) {
        std::ifstream test(p + "test_golden.cpp");
        if (test.good()) return p;
    }
    return "../tests/golden/";  // Default
}

// Helper to check file existence
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Write a ScalarField to file (2D or 3D)
void write_scalar_field(const ScalarField& field, const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    f << std::setprecision(17) << std::scientific;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f << field(i, j) << "\n";
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    f << field(i, j, k) << "\n";
                }
            }
        }
    }
}

// Write velocity component (u, v, or w) to file
// Note: VectorField uses staggered grid, so we interpolate to cell centers
void write_velocity_u(const VectorField& vel, const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    f << std::setprecision(17) << std::scientific;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f << vel.u_center(i, j) << "\n";
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    f << vel.u_center(i, j, k) << "\n";
                }
            }
        }
    }
}

void write_velocity_v(const VectorField& vel, const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    f << std::setprecision(17) << std::scientific;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f << vel.v_center(i, j) << "\n";
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    f << vel.v_center(i, j, k) << "\n";
                }
            }
        }
    }
}

void write_velocity_w(const VectorField& vel, const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    f << std::setprecision(17) << std::scientific;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f << vel.w_center(i, j, k) << "\n";
            }
        }
    }
}

// Compare a ScalarField against golden file
bool compare_scalar_field(const ScalarField& field, const Mesh& mesh,
                          const std::string& golden_path, const std::string& name,
                          double tol = GOLDEN_TOL) {
    std::ifstream f(golden_path);
    if (!f) {
        std::cerr << "  [FAIL] Missing golden file: " << golden_path << "\n";
        return false;
    }

    double max_diff = 0.0;
    int max_i = 0, max_j = 0, max_k = 0;
    double computed_val = 0.0, golden_val = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double gval;
                if (!(f >> gval)) {
                    std::cerr << "  [FAIL] " << name << ": Unexpected EOF at ("
                              << i << "," << j << ")\n";
                    return false;
                }
                double diff = std::abs(field(i, j) - gval);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                    computed_val = field(i, j);
                    golden_val = gval;
                }
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double gval;
                    if (!(f >> gval)) {
                        std::cerr << "  [FAIL] " << name << ": Unexpected EOF at ("
                                  << i << "," << j << "," << k << ")\n";
                        return false;
                    }
                    double diff = std::abs(field(i, j, k) - gval);
                    if (diff > max_diff) {
                        max_diff = diff;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                        computed_val = field(i, j, k);
                        golden_val = gval;
                    }
                }
            }
        }
    }

    bool pass = (max_diff < tol);
    std::cout << "  " << std::left << std::setw(12) << name
              << ": max_diff = " << std::scientific << std::setprecision(3) << max_diff;
    if (pass) {
        std::cout << " [OK]\n";
    } else {
        if (mesh.is2D()) {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << ")";
        } else {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << "," << max_k << ")";
        }
        std::cout << " computed=" << computed_val << " golden=" << golden_val << "\n";
    }
    return pass;
}

// Compare velocity component (u or v) at cell centers against golden file
bool compare_velocity_u(const VectorField& vel, const Mesh& mesh,
                        const std::string& golden_path, const std::string& name,
                        double tol = GOLDEN_TOL) {
    std::ifstream f(golden_path);
    if (!f) {
        std::cerr << "  [FAIL] Missing golden file: " << golden_path << "\n";
        return false;
    }

    double max_diff = 0.0;
    int max_i = 0, max_j = 0, max_k = 0;
    double computed_val = 0.0, golden_val = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double gval;
                if (!(f >> gval)) {
                    std::cerr << "  [FAIL] " << name << ": Unexpected EOF\n";
                    return false;
                }
                double val = vel.u_center(i, j);
                double diff = std::abs(val - gval);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                    computed_val = val;
                    golden_val = gval;
                }
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double gval;
                    if (!(f >> gval)) {
                        std::cerr << "  [FAIL] " << name << ": Unexpected EOF\n";
                        return false;
                    }
                    double val = vel.u_center(i, j, k);
                    double diff = std::abs(val - gval);
                    if (diff > max_diff) {
                        max_diff = diff;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                        computed_val = val;
                        golden_val = gval;
                    }
                }
            }
        }
    }

    bool pass = (max_diff < tol);
    std::cout << "  " << std::left << std::setw(12) << name
              << ": max_diff = " << std::scientific << std::setprecision(3) << max_diff;
    if (pass) {
        std::cout << " [OK]\n";
    } else {
        if (mesh.is2D()) {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << ")";
        } else {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << "," << max_k << ")";
        }
        std::cout << " computed=" << computed_val << " golden=" << golden_val << "\n";
    }
    return pass;
}

bool compare_velocity_v(const VectorField& vel, const Mesh& mesh,
                        const std::string& golden_path, const std::string& name,
                        double tol = GOLDEN_TOL) {
    std::ifstream f(golden_path);
    if (!f) {
        std::cerr << "  [FAIL] Missing golden file: " << golden_path << "\n";
        return false;
    }

    double max_diff = 0.0;
    int max_i = 0, max_j = 0, max_k = 0;
    double computed_val = 0.0, golden_val = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double gval;
                if (!(f >> gval)) {
                    std::cerr << "  [FAIL] " << name << ": Unexpected EOF\n";
                    return false;
                }
                double val = vel.v_center(i, j);
                double diff = std::abs(val - gval);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                    computed_val = val;
                    golden_val = gval;
                }
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double gval;
                    if (!(f >> gval)) {
                        std::cerr << "  [FAIL] " << name << ": Unexpected EOF\n";
                        return false;
                    }
                    double val = vel.v_center(i, j, k);
                    double diff = std::abs(val - gval);
                    if (diff > max_diff) {
                        max_diff = diff;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                        computed_val = val;
                        golden_val = gval;
                    }
                }
            }
        }
    }

    bool pass = (max_diff < tol);
    std::cout << "  " << std::left << std::setw(12) << name
              << ": max_diff = " << std::scientific << std::setprecision(3) << max_diff;
    if (pass) {
        std::cout << " [OK]\n";
    } else {
        if (mesh.is2D()) {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << ")";
        } else {
            std::cout << " [FAIL] at (" << max_i << "," << max_j << "," << max_k << ")";
        }
        std::cout << " computed=" << computed_val << " golden=" << golden_val << "\n";
    }
    return pass;
}

bool compare_velocity_w(const VectorField& vel, const Mesh& mesh,
                        const std::string& golden_path, const std::string& name,
                        double tol = GOLDEN_TOL) {
    std::ifstream f(golden_path);
    if (!f) {
        std::cerr << "  [FAIL] Missing golden file: " << golden_path << "\n";
        return false;
    }

    double max_diff = 0.0;
    int max_i = 0, max_j = 0, max_k = 0;
    double computed_val = 0.0, golden_val = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double gval;
                if (!(f >> gval)) {
                    std::cerr << "  [FAIL] " << name << ": Unexpected EOF\n";
                    return false;
                }
                double val = vel.w_center(i, j, k);
                double diff = std::abs(val - gval);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                    max_k = k;
                    computed_val = val;
                    golden_val = gval;
                }
            }
        }
    }

    bool pass = (max_diff < tol);
    std::cout << "  " << std::left << std::setw(12) << name
              << ": max_diff = " << std::scientific << std::setprecision(3) << max_diff;
    if (pass) {
        std::cout << " [OK]\n";
    } else {
        std::cout << " [FAIL] at (" << max_i << "," << max_j << "," << max_k << ")"
                  << " computed=" << computed_val << " golden=" << golden_val << "\n";
    }
    return pass;
}

// Check if model uses transport equations (k, omega)
bool uses_transport(TurbulenceModelType type) {
    return type == TurbulenceModelType::SSTKOmega ||
           type == TurbulenceModelType::KOmega ||
           type == TurbulenceModelType::EARSM_WJ ||
           type == TurbulenceModelType::EARSM_GS ||
           type == TurbulenceModelType::EARSM_Pope;
}

// Get NN weights path if available
std::string find_nn_weights(const std::string& model_name) {
    std::vector<std::string> paths = {
        "data/models/" + model_name,
        "../data/models/" + model_name,
        "../../data/models/" + model_name
    };
    for (const auto& p : paths) {
        if (file_exists(p + "/layer0_W.txt")) return p;
    }
    return "";
}

struct TestCase {
    std::string name;
    TurbulenceModelType model;
    bool is_3d;
    std::string nn_model;  // Empty if not an NN model
};

TestCase get_test_case(const std::string& case_name) {
    if (case_name == "channel_komega") {
        return {"channel_komega", TurbulenceModelType::SSTKOmega, false, ""};
    } else if (case_name == "channel_earsm") {
        return {"channel_earsm", TurbulenceModelType::EARSM_WJ, false, ""};
    } else if (case_name == "channel_mlp") {
        return {"channel_mlp", TurbulenceModelType::NNMLP, false, "mlp_channel_caseholdout"};
    } else if (case_name == "channel_tbnn") {
        return {"channel_tbnn", TurbulenceModelType::NNTBNN, false, "tbnn_channel_caseholdout"};
    } else if (case_name == "mixing_length") {
        return {"mixing_length", TurbulenceModelType::Baseline, false, ""};
    } else if (case_name == "laminar_3d") {
        return {"laminar_3d", TurbulenceModelType::None, true, ""};
    } else {
        throw std::runtime_error("Unknown test case: " + case_name);
    }
}

int run_test(const std::string& case_name, bool regenerate) {
    std::string golden_dir = get_golden_dir();
    TestCase tc = get_test_case(case_name);

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  GOLDEN FILE TEST: " << case_name << "\n";
    std::cout << "================================================================\n";

    // Check for NN weights if needed
    std::string nn_path;
    if (!tc.nn_model.empty()) {
        nn_path = find_nn_weights(tc.nn_model);
        if (nn_path.empty()) {
            std::cout << "[SKIP] NN weights not found for " << tc.nn_model << "\n";
            return 0;  // Skip is not a failure
        }
        std::cout << "Using NN weights: " << nn_path << "\n";
    }

    // Setup mesh
    Mesh mesh;
    if (tc.is_3d) {
        // 3D laminar duct: 16x16x8
        mesh.init_uniform(16, 16, 8, 0.0, 1.0, -0.5, 0.5, 0.0, 0.5);
    } else {
        // 2D channel: 32x32
        mesh.init_uniform(32, 32, 0.0, 2.0 * 3.14159265358979, -1.0, 1.0);
    }

    // Setup config
    Config config;
    config.nu = 0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 100;
    config.tol = 1e-6;
    config.turb_model = tc.model;
    config.verbose = false;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 5;

    if (!nn_path.empty()) {
        config.nn_weights_path = nn_path;
        config.nn_scaling_path = nn_path;
    }

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    if (tc.is_3d) {
        bc.z_lo = VelocityBC::NoSlip;
        bc.z_hi = VelocityBC::NoSlip;
    }
    solver.set_velocity_bc(bc);

    // Create turbulence model
    if (tc.model != TurbulenceModelType::None) {
        auto model = create_turbulence_model(tc.model, nn_path, nn_path);
        solver.set_turbulence_model(std::move(model));
    }

    // Initialize
    solver.initialize_uniform(1.0, 0.0);

    // Set initial velocity profile (use non-const velocity() accessor)
    VectorField& vel = solver.velocity();
    if (tc.is_3d) {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                double y = mesh.y(j);
                double z = mesh.z(k);
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    // Simple profile for 3D duct
                    vel.u(i, j, k) = 0.1 * (0.25 - y*y) * (0.25 - z*z);
                }
            }
        }
    } else {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel.u(i, j) = 0.1 * (1.0 - y * y);
            }
        }
    }

    solver.sync_to_gpu();

    // Run 10 time steps
    std::cout << "Running 10 time steps...\n";
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    solver.sync_from_gpu();
    std::cout << "Simulation complete.\n\n";

    // Reference directory
    std::string ref_dir = golden_dir + "reference/";
    std::string prefix = ref_dir + case_name + "_";

    const VectorField& velocity = solver.velocity();
    const ScalarField& pressure = solver.pressure();

    if (regenerate) {
        std::cout << "Regenerating golden files...\n";

        write_velocity_u(velocity, mesh, prefix + "u.dat");
        write_velocity_v(velocity, mesh, prefix + "v.dat");
        write_scalar_field(pressure, mesh, prefix + "p.dat");

        if (tc.is_3d) {
            write_velocity_w(velocity, mesh, prefix + "w.dat");
        }

        if (uses_transport(tc.model)) {
            write_scalar_field(solver.k(), mesh, prefix + "k.dat");
            write_scalar_field(solver.omega(), mesh, prefix + "omega.dat");
        }

        std::cout << "[OK] Golden files regenerated in " << ref_dir << "\n";
        std::cout << "Review changes with: git diff tests/golden/reference/\n";
        return 0;
    }

    // Compare against golden files
    std::cout << "Comparing against golden files (tol=" << GOLDEN_TOL << "):\n";
    bool all_pass = true;

    all_pass &= compare_velocity_u(velocity, mesh, prefix + "u.dat", "u");
    all_pass &= compare_velocity_v(velocity, mesh, prefix + "v.dat", "v");
    all_pass &= compare_scalar_field(pressure, mesh, prefix + "p.dat", "p");

    if (tc.is_3d) {
        all_pass &= compare_velocity_w(velocity, mesh, prefix + "w.dat", "w");
    }

    if (uses_transport(tc.model)) {
        all_pass &= compare_scalar_field(solver.k(), mesh, prefix + "k.dat", "k");
        all_pass &= compare_scalar_field(solver.omega(), mesh, prefix + "omega.dat", "omega");
    }

    std::cout << "\n";
    if (all_pass) {
        std::cout << "[SUCCESS] All golden file comparisons passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Golden file comparison failed!\n";
        return 1;
    }
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <case_name> [--regenerate]\n\n";
    std::cerr << "Cases:\n";
    std::cerr << "  channel_komega    - 2D channel with SST k-omega\n";
    std::cerr << "  channel_earsm     - 2D channel with EARSM-WJ\n";
    std::cerr << "  channel_mlp       - 2D channel with NN-MLP\n";
    std::cerr << "  channel_tbnn      - 2D channel with NN-TBNN\n";
    std::cerr << "  mixing_length     - 2D channel with mixing length model\n";
    std::cerr << "  laminar_3d        - 3D duct with laminar flow\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --regenerate      Regenerate golden files instead of comparing\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string case_name = argv[1];
    bool regenerate = false;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--regenerate") {
            regenerate = true;
        }
    }

    try {
        return run_test(case_name, regenerate);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
