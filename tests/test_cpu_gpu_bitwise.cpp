/// CPU/GPU Cross-Backend Consistency Test
/// Compares CPU-built and GPU-built solver outputs via compact signatures.
///
/// This test REQUIRES two separate builds:
///   1. CPU build (USE_GPU_OFFLOAD=OFF): Run with --dump to generate reference
///   2. GPU build (USE_GPU_OFFLOAD=ON):  Run with --compare to compare against reference
///
/// Each scenario produces a JSON signature with scalar QoIs and probes.
/// Comparison uses per-metric tolerances to catch algorithmic divergence
/// while allowing expected FP rounding differences.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_baseline.hpp"
#include "test_utilities.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <functional>
#include <map>
#include <set>
#include <algorithm>

using nncfd::test::file_exists;
using nncfd::test::CROSS_BACKEND_TOLERANCE;

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nncfd;

//=============================================================================
// Signature: Compact representation of scenario output
//=============================================================================

struct Metric {
    double value = 0.0;
    double abs_tol = 1e-10;  // Absolute tolerance
    double rel_tol = 1e-8;   // Relative tolerance

    Metric() = default;
    Metric(double v, double at = 1e-10, double rt = 1e-8)
        : value(v), abs_tol(at), rel_tol(rt) {}
};

struct Probe {
    std::vector<double> values;  // e.g., [u, v] or [u, v, w]
    double abs_tol = 1e-10;
    double rel_tol = 1e-8;
};

struct Signature {
    int schema_version = 1;
    std::string scenario;
    std::string backend;
    std::map<std::string, Metric> metrics;
    std::map<std::string, Probe> probes;

    // Serialize to JSON
    std::string to_json() const {
        std::ostringstream os;
        os << std::setprecision(17) << std::scientific;
        os << "{\n";
        os << "  \"schema_version\": " << schema_version << ",\n";
        os << "  \"scenario\": \"" << scenario << "\",\n";
        os << "  \"backend\": \"" << backend << "\",\n";

        os << "  \"metrics\": {\n";
        bool first = true;
        for (const auto& [name, m] : metrics) {
            if (!first) os << ",\n";
            os << "    \"" << name << "\": " << m.value;
            first = false;
        }
        os << "\n  },\n";

        os << "  \"probes\": {\n";
        first = true;
        for (const auto& [name, p] : probes) {
            if (!first) os << ",\n";
            os << "    \"" << name << "\": [";
            for (size_t i = 0; i < p.values.size(); ++i) {
                if (i > 0) os << ", ";
                os << p.values[i];
            }
            os << "]";
            first = false;
        }
        os << "\n  }\n";
        os << "}\n";
        return os.str();
    }

    // Parse from JSON (simple parser for our known format)
    static Signature from_json(const std::string& json) {
        Signature sig;

        // Extract scenario name
        auto extract_string = [&](const std::string& key) -> std::string {
            std::string pattern = "\"" + key + "\": \"";
            auto pos = json.find(pattern);
            if (pos == std::string::npos) return "";
            pos += pattern.size();
            auto end = json.find("\"", pos);
            return json.substr(pos, end - pos);
        };

        sig.scenario = extract_string("scenario");
        sig.backend = extract_string("backend");

        // Extract metrics (simple: find "metrics": { ... } block)
        auto metrics_start = json.find("\"metrics\":");
        auto metrics_end = json.find("}", metrics_start);
        if (metrics_start != std::string::npos && metrics_end != std::string::npos) {
            std::string metrics_block = json.substr(metrics_start, metrics_end - metrics_start + 1);

            // Parse each metric: "name": value
            size_t pos = 0;
            while ((pos = metrics_block.find("\"", pos)) != std::string::npos) {
                size_t name_start = pos + 1;
                size_t name_end = metrics_block.find("\"", name_start);
                if (name_end == std::string::npos) break;

                std::string name = metrics_block.substr(name_start, name_end - name_start);
                if (name == "metrics") { pos = name_end + 1; continue; }

                size_t colon = metrics_block.find(":", name_end);
                if (colon == std::string::npos) break;

                // Find the value (number)
                size_t val_start = colon + 1;
                while (val_start < metrics_block.size() &&
                       (metrics_block[val_start] == ' ' || metrics_block[val_start] == '\n'))
                    ++val_start;

                size_t val_end = val_start;
                while (val_end < metrics_block.size() &&
                       (std::isdigit(metrics_block[val_end]) ||
                        metrics_block[val_end] == '.' ||
                        metrics_block[val_end] == 'e' ||
                        metrics_block[val_end] == 'E' ||
                        metrics_block[val_end] == '+' ||
                        metrics_block[val_end] == '-'))
                    ++val_end;

                if (val_end > val_start) {
                    double val = std::stod(metrics_block.substr(val_start, val_end - val_start));
                    sig.metrics[name] = Metric(val);
                }

                pos = val_end;
            }
        }

        // Extract probes similarly (simplified - just get the values)
        auto probes_start = json.find("\"probes\":");
        if (probes_start != std::string::npos) {
            size_t pos = probes_start;
            while ((pos = json.find("\"", pos)) != std::string::npos) {
                size_t name_start = pos + 1;
                size_t name_end = json.find("\"", name_start);
                if (name_end == std::string::npos) break;

                std::string name = json.substr(name_start, name_end - name_start);
                if (name == "probes" || name == "}") { pos = name_end + 1; continue; }

                size_t bracket = json.find("[", name_end);
                if (bracket == std::string::npos) break;
                size_t bracket_end = json.find("]", bracket);
                if (bracket_end == std::string::npos) break;

                std::string arr = json.substr(bracket + 1, bracket_end - bracket - 1);
                Probe p;
                std::istringstream iss(arr);
                double v;
                char comma;
                while (iss >> v) {
                    p.values.push_back(v);
                    iss >> comma;  // consume comma if present
                }
                if (!p.values.empty()) {
                    sig.probes[name] = p;
                }

                pos = bracket_end + 1;
            }
        }

        return sig;
    }
};

//=============================================================================
// Comparison result for a single metric
//=============================================================================

struct MetricResult {
    std::string name;
    double ref_value;
    double test_value;
    double abs_diff;
    double rel_diff;
    double abs_tol;
    double rel_tol;
    bool passed;

    void print() const {
        std::cout << "    " << std::left << std::setw(20) << name << ": ";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "ref=" << ref_value << " test=" << test_value;
        std::cout << " |diff|=" << abs_diff;
        if (passed) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL] (tol: abs=" << abs_tol << " rel=" << rel_tol << ")\n";
        }
    }
};

//=============================================================================
// Backend verification
//=============================================================================

std::string get_backend_name() {
#ifdef USE_GPU_OFFLOAD
    return "GPU";
#else
    return "CPU";
#endif
}

void print_backend_identity() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "EXEC_BACKEND=GPU_OFFLOAD\n";
    std::cout << "  Compiled with: USE_GPU_OFFLOAD=ON\n";
    #if defined(_OPENMP)
    std::cout << "  OMP devices: " << omp_get_num_devices() << "\n";
    std::cout << "  OMP default device: " << omp_get_default_device() << "\n";
    #endif
#else
    std::cout << "EXEC_BACKEND=CPU_ONLY\n";
    std::cout << "  Compiled with: USE_GPU_OFFLOAD=OFF\n";
#endif
}

bool verify_backend() {
#ifdef USE_GPU_OFFLOAD
    const int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cerr << "ERROR: No GPU devices found\n";
        return false;
    }
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }
    if (!on_device) {
        std::cerr << "ERROR: Target region executed on host, not GPU\n";
        return false;
    }
    std::cout << "  GPU execution verified: YES (device " << omp_get_default_device() << ")\n";
#endif
    return true;
}

//=============================================================================
// QoI computation helpers
//=============================================================================

double compute_kinetic_energy(const Mesh& mesh, const VectorField& vel) {
    double KE = 0.0;
    if (mesh.is2D()) {
        const double cell_area = mesh.dx * mesh.dy;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                KE += 0.5 * (u*u + v*v) * cell_area;
            }
        }
    } else {
        const double cell_vol = mesh.dx * mesh.dy * mesh.dz;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                    double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                    KE += 0.5 * (u*u + v*v + w*w) * cell_vol;
                }
            }
        }
    }
    return KE;
}

double compute_max_divergence(const Mesh& mesh, const VectorField& vel) {
    double max_div = 0.0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
                double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
                max_div = std::max(max_div, std::abs(dudx + dvdy));
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double dudx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / mesh.dx;
                    double dvdy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / mesh.dy;
                    double dwdz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / mesh.dz;
                    max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
                }
            }
        }
    }
    return max_div;
}

double compute_enstrophy_2d(const Mesh& mesh, const VectorField& vel) {
    double ens = 0.0;
    const double cell_area = mesh.dx * mesh.dy;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Vorticity omega_z = dv/dx - du/dy at cell center
            double dvdx = (vel.v(i+1, j) - vel.v(i, j)) / mesh.dx;
            double dudy = (vel.u(i, j+1) - vel.u(i, j)) / mesh.dy;
            double omega = dvdx - dudy;
            ens += 0.5 * omega * omega * cell_area;
        }
    }
    return ens;
}

double compute_mean_u(const Mesh& mesh, const VectorField& vel) {
    double sum = 0.0;
    int count = 0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    sum += 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    ++count;
                }
            }
        }
    }
    return sum / count;
}

double compute_pressure_rms(const Mesh& mesh, const ScalarField& p) {
    double sum_sq = 0.0;
    double mean = 0.0;
    int count = 0;

    // First pass: mean
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                mean += p(i, j);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    mean += p(i, j, k);
                    ++count;
                }
            }
        }
    }
    mean /= count;

    // Second pass: RMS
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double diff = p(i, j) - mean;
                sum_sq += diff * diff;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double diff = p(i, j, k) - mean;
                    sum_sq += diff * diff;
                }
            }
        }
    }
    return std::sqrt(sum_sq / count);
}

//=============================================================================
// Scenario definition
//=============================================================================

struct Scenario {
    std::string name;
    std::string description;
    std::function<bool()> is_available;
    std::function<Signature()> run;

    // Per-scenario metric tolerances (can override defaults)
    std::map<std::string, std::pair<double, double>> tolerances;  // metric -> (abs_tol, rel_tol)
};

//=============================================================================
// Scenario: Channel flow with body force (3D)
//=============================================================================

Signature run_channel_solver_3d() {
    const int NX = 32, NY = 32, NZ = 8;
    const int NUM_STEPS = 20;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Poiseuille IC
    double H = 1.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - H;
            double u_val = 0.01 * (H*H - y*y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = u_val;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    Signature sig;
    sig.scenario = "channel_solver_3d";
    sig.backend = get_backend_name();

    // Compute QoIs
    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));
    sig.metrics["p_rms"] = Metric(compute_pressure_rms(mesh, solver.pressure()));

    // Probes at fixed locations
    int pi = mesh.i_begin() + NX/2;
    int pj = mesh.j_begin() + NY/2;
    int pk = mesh.k_begin() + NZ/2;

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj, pk) + solver.velocity().u(pi+1, pj, pk)),
        0.5 * (solver.velocity().v(pi, pj, pk) + solver.velocity().v(pi, pj+1, pk)),
        0.5 * (solver.velocity().w(pi, pj, pk) + solver.velocity().w(pi, pj, pk+1))
    };
    sig.probes["vel_center"] = vel_probe;

    Probe p_probe;
    p_probe.values = {solver.pressure()(pi, pj, pk)};
    sig.probes["p_center"] = p_probe;

    return sig;
}

//=============================================================================
// Scenario: Taylor-Green Vortex 2D (short run)
//=============================================================================

Signature run_tgv_2d() {
    const int N = 32;
    const int NUM_STEPS = 10;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // TGV IC
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            double y = mesh.y(j);
            solver.velocity().u(i, j) = std::sin(x) * std::cos(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.yf[j];
            solver.velocity().v(i, j) = -std::cos(x) * std::sin(y);
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    Signature sig;
    sig.scenario = "tgv_2d";
    sig.backend = get_backend_name();

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["enstrophy"] = Metric(compute_enstrophy_2d(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["p_rms"] = Metric(compute_pressure_rms(mesh, solver.pressure()));

    // Probe at center
    int pi = mesh.i_begin() + N/2;
    int pj = mesh.j_begin() + N/2;

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_center"] = vel_probe;

    return sig;
}

//=============================================================================
// Scenario: Poiseuille 2D (steady-state tendency)
//=============================================================================

Signature run_poiseuille_2d() {
    const int NX = 32, NY = 32;
    const int NUM_STEPS = 15;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, 4.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.01, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Start near analytical Poiseuille
    double H = 1.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j) - H;
        double u_exact = 0.5 * 0.01 / config.nu * (H*H - y*y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_exact * 0.9;  // Start at 90%
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    Signature sig;
    sig.scenario = "poiseuille_2d";
    sig.backend = get_backend_name();

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));

    // Probe at channel center
    int pi = mesh.i_begin() + NX/2;
    int pj = mesh.j_begin() + NY/2;  // center of channel

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_center"] = vel_probe;

    return sig;
}

//=============================================================================
// Scenario: Mixing Length turbulence model
//=============================================================================

Signature run_mixing_length() {
    const int NX = 24, NY = 32;
    const int NUM_STEPS = 10;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, 4.0, 0.0, 2.0);

    Config config;
    config.nu = 0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::Baseline;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.01, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Turbulent-like IC
    double H = 1.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j) - H;
        double u_base = H*H - y*y;
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_base;
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    Signature sig;
    sig.scenario = "mixing_length";
    sig.backend = get_backend_name();

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));

    // nu_t at probe point
    int pi = mesh.i_begin() + NX/2;
    int pj = mesh.j_begin() + NY/4;  // Off-center to get turbulent region

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_quarter"] = vel_probe;

    return sig;
}

//=============================================================================
// Scenario: NN-MLP turbulence model
//=============================================================================

bool nn_mlp_available() {
    std::string path = "data/models/mlp_channel_caseholdout";
    if (file_exists(path + "/layer0_W.txt")) return true;
    path = "../data/models/mlp_channel_caseholdout";
    if (file_exists(path + "/layer0_W.txt")) return true;
    return false;
}

std::string get_nn_mlp_model_path() {
    std::string path = "data/models/mlp_channel_caseholdout";
    if (file_exists(path + "/layer0_W.txt")) return path;
    path = "../data/models/mlp_channel_caseholdout";
    if (file_exists(path + "/layer0_W.txt")) return path;
    return "";
}

Signature run_nn_mlp() {
    const int NX = 32, NY = 48;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;

    std::string model_path = get_nn_mlp_model_path();

    TurbulenceNNMLP nn_model;
    nn_model.set_nu(config.nu);
    nn_model.load(model_path, model_path);

    // Create velocity field with shear
    VectorField vel(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            vel.u(i, j) = 1.0 - y * y;
        }
    }

    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);

#ifdef USE_GPU_OFFLOAD
    TurbulenceDeviceView device_view;
    nn_model.prepare_gpu_buffers(mesh);
    nn_model.get_device_view(device_view);
    nn_model.update(mesh, vel, k, omega, nu_t, &device_view);
#else
    nn_model.update(mesh, vel, k, omega, nu_t);
#endif

    Signature sig;
    sig.scenario = "nn_mlp";
    sig.backend = get_backend_name();

    // Compute nu_t statistics
    double nu_t_sum = 0.0, nu_t_max = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            nu_t_sum += nu_t(i, j);
            nu_t_max = std::max(nu_t_max, nu_t(i, j));
            ++count;
        }
    }

    sig.metrics["nu_t_mean"] = Metric(nu_t_sum / count);
    sig.metrics["nu_t_max"] = Metric(nu_t_max);

    // Probes at different y locations
    int pi = mesh.i_begin() + NX/2;

    Probe nu_t_profile;
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + NY/4));
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + NY/2));
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + 3*NY/4));
    sig.probes["nu_t_profile"] = nu_t_profile;

    return sig;
}

//=============================================================================
// Scenario registry
//=============================================================================

std::vector<Scenario> get_scenarios() {
    std::vector<Scenario> scenarios;

    scenarios.push_back({
        "channel_solver_3d",
        "3D channel flow with body force",
        []() { return true; },
        run_channel_solver_3d,
        {}
    });

    scenarios.push_back({
        "tgv_2d",
        "2D Taylor-Green vortex",
        []() { return true; },
        run_tgv_2d,
        {}
    });

    scenarios.push_back({
        "poiseuille_2d",
        "2D Poiseuille flow development",
        []() { return true; },
        run_poiseuille_2d,
        {}
    });

    scenarios.push_back({
        "mixing_length",
        "Mixing length turbulence model",
        []() { return true; },
        run_mixing_length,
        {}
    });

    scenarios.push_back({
        "nn_mlp",
        "NN-MLP turbulence model",
        nn_mlp_available,
        run_nn_mlp,
        {}
    });

    return scenarios;
}

//=============================================================================
// Dump mode: Generate reference signatures
//=============================================================================

int run_dump_mode(const std::string& output_file, const std::set<std::string>& filter) {
#ifdef USE_GPU_OFFLOAD
    std::cerr << "ERROR: --dump requires CPU-only build\n";
    std::cerr << "       This binary was built with USE_GPU_OFFLOAD=ON\n";
    std::cerr << "       Rebuild with -DUSE_GPU_OFFLOAD=OFF\n";
    return 1;
#else
    std::cout << "=== CPU Reference Generation Mode ===\n";
    print_backend_identity();
    std::cout << "Output file: " << output_file << "\n\n";

    if (!verify_backend()) {
        std::cerr << "ERROR: Backend verification failed\n";
        return 1;
    }

    auto scenarios = get_scenarios();
    std::vector<Signature> signatures;

    int run_count = 0, skip_count = 0;

    for (const auto& scenario : scenarios) {
        // Check filter
        if (!filter.empty() && filter.find(scenario.name) == filter.end()) {
            continue;
        }

        std::cout << "--- " << scenario.name << " ---\n";
        std::cout << "  " << scenario.description << "\n";

        if (!scenario.is_available()) {
            std::cout << "  [SKIPPED] Prerequisites not met\n\n";
            ++skip_count;
            continue;
        }

        std::cout << "  Running...\n";
        Signature sig = scenario.run();
        signatures.push_back(sig);

        std::cout << "  Metrics:\n";
        for (const auto& [name, m] : sig.metrics) {
            std::cout << "    " << name << ": " << std::scientific << m.value << "\n";
        }
        std::cout << "\n";
        ++run_count;
    }

    // Write all signatures to file
    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "ERROR: Cannot open output file: " << output_file << "\n";
        return 1;
    }

    out << "[\n";
    for (size_t i = 0; i < signatures.size(); ++i) {
        if (i > 0) out << ",\n";
        out << signatures[i].to_json();
    }
    out << "]\n";

    std::cout << "[SUCCESS] Wrote " << run_count << " signatures";
    if (skip_count > 0) std::cout << " (" << skip_count << " skipped)";
    std::cout << " to " << output_file << "\n";

    return 0;
#endif
}

//=============================================================================
// Compare mode: Run GPU and compare against CPU reference
//=============================================================================

int run_compare_mode(const std::string& ref_file, const std::set<std::string>& filter) {
#ifndef USE_GPU_OFFLOAD
    std::cerr << "ERROR: --compare requires GPU build\n";
    std::cerr << "       This binary was built with USE_GPU_OFFLOAD=OFF\n";
    std::cerr << "       Rebuild with -DUSE_GPU_OFFLOAD=ON\n";
    return 1;
#else
    std::cout << "=== GPU Comparison Mode ===\n";
    print_backend_identity();
    std::cout << "Reference file: " << ref_file << "\n\n";

    if (!verify_backend()) {
        std::cerr << "ERROR: Backend verification failed\n";
        return 1;
    }

    // Load reference signatures
    std::ifstream in(ref_file);
    if (!in) {
        std::cerr << "ERROR: Cannot open reference file: " << ref_file << "\n";
        return 1;
    }

    std::string json_content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());

    // Parse JSON array of signatures (simple parser)
    std::map<std::string, Signature> ref_sigs;
    size_t pos = 0;
    while ((pos = json_content.find("{", pos)) != std::string::npos) {
        size_t end = pos + 1;
        int brace_count = 1;
        while (end < json_content.size() && brace_count > 0) {
            if (json_content[end] == '{') ++brace_count;
            else if (json_content[end] == '}') --brace_count;
            ++end;
        }

        std::string sig_json = json_content.substr(pos, end - pos);
        Signature sig = Signature::from_json(sig_json);
        if (!sig.scenario.empty()) {
            ref_sigs[sig.scenario] = sig;
        }
        pos = end;
    }

    std::cout << "Loaded " << ref_sigs.size() << " reference signatures\n\n";

    auto scenarios = get_scenarios();
    bool all_passed = true;
    int pass_count = 0, fail_count = 0, skip_count = 0;

    for (const auto& scenario : scenarios) {
        // Check filter
        if (!filter.empty() && filter.find(scenario.name) == filter.end()) {
            continue;
        }

        std::cout << "--- " << scenario.name << " ---\n";

        // Check if reference exists
        if (ref_sigs.find(scenario.name) == ref_sigs.end()) {
            std::cout << "  [SKIPPED] No reference signature\n\n";
            ++skip_count;
            continue;
        }

        if (!scenario.is_available()) {
            std::cout << "  [SKIPPED] Prerequisites not met\n\n";
            ++skip_count;
            continue;
        }

        std::cout << "  Running on GPU...\n";
        Signature gpu_sig = scenario.run();
        const Signature& cpu_sig = ref_sigs[scenario.name];

        // Compare metrics
        bool scenario_passed = true;
        std::cout << "  Comparing metrics:\n";

        for (const auto& [name, cpu_metric] : cpu_sig.metrics) {
            if (gpu_sig.metrics.find(name) == gpu_sig.metrics.end()) {
                std::cout << "    " << name << ": [FAIL] Missing in GPU results\n";
                scenario_passed = false;
                continue;
            }

            double gpu_val = gpu_sig.metrics[name].value;
            double cpu_val = cpu_metric.value;
            double abs_diff = std::abs(gpu_val - cpu_val);
            double rel_diff = abs_diff / (std::max(std::abs(cpu_val), std::abs(gpu_val)) + 1e-30);

            // Get tolerances (use scenario-specific if available, else defaults)
            double abs_tol = cpu_metric.abs_tol;
            double rel_tol = cpu_metric.rel_tol;
            if (scenario.tolerances.find(name) != scenario.tolerances.end()) {
                abs_tol = scenario.tolerances.at(name).first;
                rel_tol = scenario.tolerances.at(name).second;
            }

            bool passed = (abs_diff <= abs_tol + rel_tol * std::max(std::abs(cpu_val), std::abs(gpu_val)));

            std::cout << "    " << std::left << std::setw(15) << name << ": ";
            std::cout << std::scientific << std::setprecision(4);
            std::cout << "diff=" << abs_diff;
            if (passed) {
                std::cout << " [PASS]\n";
            } else {
                std::cout << " [FAIL] (tol=" << abs_tol << "+" << rel_tol << "*|ref|)\n";
                scenario_passed = false;
            }
        }

        // Compare probes
        std::cout << "  Comparing probes:\n";
        for (const auto& [name, cpu_probe] : cpu_sig.probes) {
            if (gpu_sig.probes.find(name) == gpu_sig.probes.end()) {
                std::cout << "    " << name << ": [FAIL] Missing in GPU results\n";
                scenario_passed = false;
                continue;
            }

            const auto& gpu_probe = gpu_sig.probes.at(name);
            if (cpu_probe.values.size() != gpu_probe.values.size()) {
                std::cout << "    " << name << ": [FAIL] Size mismatch\n";
                scenario_passed = false;
                continue;
            }

            double max_diff = 0.0;
            for (size_t i = 0; i < cpu_probe.values.size(); ++i) {
                max_diff = std::max(max_diff, std::abs(cpu_probe.values[i] - gpu_probe.values[i]));
            }

            bool passed = (max_diff <= CROSS_BACKEND_TOLERANCE);
            std::cout << "    " << std::left << std::setw(15) << name << ": ";
            std::cout << "max_diff=" << std::scientific << max_diff;
            if (passed) {
                std::cout << " [PASS]\n";
            } else {
                std::cout << " [FAIL]\n";
                scenario_passed = false;
            }
        }

        if (scenario_passed) {
            std::cout << "  [SCENARIO PASS]\n\n";
            ++pass_count;
        } else {
            std::cout << "  [SCENARIO FAIL]\n\n";
            all_passed = false;
            ++fail_count;
        }
    }

    std::cout << "=== Summary ===\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";
    std::cout << "Skipped: " << skip_count << "\n\n";

    if (all_passed && fail_count == 0) {
        std::cout << "[SUCCESS] All scenarios passed\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some scenarios failed\n";
        return 1;
    }
#endif
}

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Cross-backend consistency test using compact signatures.\n\n";
    std::cout << "Workflow:\n";
    std::cout << "  1. CPU build: " << prog << " --dump cpu_ref.json\n";
    std::cout << "  2. GPU build: " << prog << " --compare cpu_ref.json\n\n";
    std::cout << "Options:\n";
    std::cout << "  --dump <file>        Generate CPU reference signatures (CPU build only)\n";
    std::cout << "  --compare <file>     Compare GPU against CPU reference (GPU build only)\n";
    std::cout << "  --scenarios <list>   Comma-separated list of scenarios to run\n";
    std::cout << "  --list               List available scenarios\n";
    std::cout << "  --help               Show this message\n\n";
    std::cout << "Scenarios:\n";
    for (const auto& s : get_scenarios()) {
        std::cout << "  " << std::left << std::setw(20) << s.name;
        std::cout << s.description;
        if (!s.is_available()) std::cout << " (unavailable)";
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string dump_file, compare_file;
        std::set<std::string> scenario_filter;
        bool list_scenarios = false;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
                dump_file = argv[++i];
            } else if (std::strcmp(argv[i], "--compare") == 0 && i + 1 < argc) {
                compare_file = argv[++i];
            } else if (std::strcmp(argv[i], "--scenarios") == 0 && i + 1 < argc) {
                std::string list = argv[++i];
                std::istringstream iss(list);
                std::string name;
                while (std::getline(iss, name, ',')) {
                    scenario_filter.insert(name);
                }
            } else if (std::strcmp(argv[i], "--list") == 0) {
                list_scenarios = true;
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown argument: " << argv[i] << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        if (list_scenarios) {
            std::cout << "Available scenarios:\n";
            for (const auto& s : get_scenarios()) {
                std::cout << "  " << std::left << std::setw(20) << s.name;
                std::cout << (s.is_available() ? "[available]" : "[unavailable]");
                std::cout << "  " << s.description << "\n";
            }
            return 0;
        }

        std::cout << "=== Cross-Backend Consistency Test ===\n";
        std::cout << "Build: " << get_backend_name() << "\n";
        std::cout << "Tolerance: " << std::scientific << CROSS_BACKEND_TOLERANCE << "\n\n";

        if (!dump_file.empty()) {
            return run_dump_mode(dump_file, scenario_filter);
        } else if (!compare_file.empty()) {
            return run_compare_mode(compare_file, scenario_filter);
        } else {
            std::cerr << "ERROR: Specify --dump or --compare\n\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
