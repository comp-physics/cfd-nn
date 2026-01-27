/// Cross-Backend Consistency Test
/// Compares CPU-built and GPU-built solver outputs via compact signatures.
///
/// Workflow:
///   1. CPU build: ./test_cross_backend --dump cpu.json
///   2. GPU build: ./test_cross_backend --dump gpu.json
///   3. Compare:   ./test_cross_backend --compare cpu.json gpu.json
///
/// Each scenario produces a JSON signature with scalar QoIs and probes.
/// Comparison uses per-metric tolerances: |a-b| <= abs_tol + rel_tol * max(|a|,|b|)

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
#include <chrono>

using nncfd::test::file_exists;
using nncfd::test::CROSS_BACKEND_TOLERANCE;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;
using nncfd::test::compute_kinetic_energy;
using nncfd::test::compute_max_divergence;
using nncfd::test::compute_enstrophy_2d;

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nncfd;

//=============================================================================
// Build metadata (captured at compile time and runtime)
//=============================================================================

struct BuildMetadata {
    int schema_version = 2;
    std::string git_sha;
    std::string build_type;
    std::string backend;
    std::string device;
    std::string compiler;
    std::string precision;
    int omp_max_threads = 1;
    std::string timestamp;

    static BuildMetadata capture() {
        BuildMetadata meta;
        meta.schema_version = 2;

        // Git SHA (set by build system or runtime)
#ifdef GIT_SHA
        meta.git_sha = GIT_SHA;
#else
        meta.git_sha = "unknown";
#endif

        // Build type
#ifdef NDEBUG
        meta.build_type = "Release";
#else
        meta.build_type = "Debug";
#endif

        // Backend
#ifdef USE_GPU_OFFLOAD
        meta.backend = "GPU";
#else
        meta.backend = "CPU";
#endif

        // Device info
#ifdef USE_GPU_OFFLOAD
        #if defined(_OPENMP)
        int num_devices = omp_get_num_devices();
        int default_device = omp_get_default_device();
        std::ostringstream oss;
        oss << "OMP device " << default_device << "/" << num_devices;
        meta.device = oss.str();
        #else
        meta.device = "GPU (unknown)";
        #endif
#else
        meta.device = "CPU";
#endif

        // Compiler
#if defined(__clang__)
        meta.compiler = "clang";
#elif defined(__GNUC__)
        meta.compiler = "gcc";
#elif defined(__NVCOMPILER)
        meta.compiler = "nvhpc";
#else
        meta.compiler = "unknown";
#endif

        // Precision
        meta.precision = "double";  // We use double throughout

        // OpenMP threads
#if defined(_OPENMP)
        meta.omp_max_threads = omp_get_max_threads();
#else
        meta.omp_max_threads = 1;
#endif

        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&time));
        meta.timestamp = buf;

        return meta;
    }

    std::string to_json() const {
        std::ostringstream os;
        os << "  \"metadata\": {\n";
        os << "    \"schema_version\": " << schema_version << ",\n";
        os << "    \"git_sha\": \"" << git_sha << "\",\n";
        os << "    \"build_type\": \"" << build_type << "\",\n";
        os << "    \"backend\": \"" << backend << "\",\n";
        os << "    \"device\": \"" << device << "\",\n";
        os << "    \"compiler\": \"" << compiler << "\",\n";
        os << "    \"precision\": \"" << precision << "\",\n";
        os << "    \"omp_max_threads\": " << omp_max_threads << ",\n";
        os << "    \"timestamp\": \"" << timestamp << "\"\n";
        os << "  }";
        return os.str();
    }

    static BuildMetadata from_json(const std::string& json) {
        BuildMetadata meta;
        auto extract = [&](const std::string& key) -> std::string {
            std::string pattern = "\"" + key + "\": \"";
            auto pos = json.find(pattern);
            if (pos == std::string::npos) return "";
            pos += pattern.size();
            auto end = json.find("\"", pos);
            return json.substr(pos, end - pos);
        };
        auto extract_int = [&](const std::string& key) -> int {
            std::string pattern = "\"" + key + "\": ";
            auto pos = json.find(pattern);
            if (pos == std::string::npos) return 0;
            pos += pattern.size();
            return std::stoi(json.substr(pos));
        };

        meta.schema_version = extract_int("schema_version");
        meta.git_sha = extract("git_sha");
        meta.build_type = extract("build_type");
        meta.backend = extract("backend");
        meta.device = extract("device");
        meta.compiler = extract("compiler");
        meta.precision = extract("precision");
        meta.omp_max_threads = extract_int("omp_max_threads");
        meta.timestamp = extract("timestamp");
        return meta;
    }
};

//=============================================================================
// Metric with tolerance
//=============================================================================

struct Metric {
    double value = 0.0;
    double abs_tol = 1e-10;
    double rel_tol = 1e-8;

    Metric() = default;
    Metric(double v, double at = 1e-10, double rt = 1e-8)
        : value(v), abs_tol(at), rel_tol(rt) {}

    // Comparison: |a-b| <= abs_tol + rel_tol * max(|a|, |b|)
    bool matches(const Metric& other) const {
        double diff = std::abs(value - other.value);
        double scale = std::max(std::abs(value), std::abs(other.value));
        return diff <= abs_tol + rel_tol * scale;
    }
};

/// Get metric tolerance based on scenario and metric name
/// This is needed because tolerances are not serialized to JSON
Metric get_metric_with_tolerance(const std::string& scenario,
                                  const std::string& metric_name,
                                  double value) {
    // Default tolerances
    double abs_tol = 1e-10;
    double rel_tol = 1e-8;

    // Special cases for near-zero quantities where CPU/GPU roundoff differs
    if (scenario == "poiseuille_2d" && metric_name == "checksum_v") {
        // v-velocity is ~0 in Poiseuille flow, both values are roundoff noise
        abs_tol = 1e-8;
        rel_tol = 1e-8;
    }

    // Looser tolerance for divergence (different Poisson solvers)
    if (metric_name == "div_max") {
        abs_tol = 1e-8;
        rel_tol = 1e-6;
    }

    return Metric(value, abs_tol, rel_tol);
}

struct Probe {
    std::vector<double> values;
    double abs_tol = 1e-10;
    double rel_tol = 1e-8;

    bool matches(const Probe& other) const {
        if (values.size() != other.values.size()) return false;
        for (size_t i = 0; i < values.size(); ++i) {
            double diff = std::abs(values[i] - other.values[i]);
            double scale = std::max(std::abs(values[i]), std::abs(other.values[i]));
            if (diff > abs_tol + rel_tol * scale) return false;
        }
        return true;
    }
};

//=============================================================================
// Scenario signature
//=============================================================================

struct ScenarioSignature {
    std::string name;
    std::map<std::string, Metric> metrics;
    std::map<std::string, Probe> probes;

    std::string to_json() const {
        std::ostringstream os;
        os << std::setprecision(17) << std::scientific;
        os << "    {\n";
        os << "      \"name\": \"" << name << "\",\n";

        os << "      \"metrics\": {\n";
        bool first = true;
        for (const auto& [mname, m] : metrics) {
            if (!first) os << ",\n";
            os << "        \"" << mname << "\": " << m.value;
            first = false;
        }
        os << "\n      },\n";

        os << "      \"probes\": {\n";
        first = true;
        for (const auto& [pname, p] : probes) {
            if (!first) os << ",\n";
            os << "        \"" << pname << "\": [";
            for (size_t i = 0; i < p.values.size(); ++i) {
                if (i > 0) os << ", ";
                os << p.values[i];
            }
            os << "]";
            first = false;
        }
        os << "\n      }\n";
        os << "    }";
        return os.str();
    }
};

//=============================================================================
// Full signature file (metadata + scenarios)
//=============================================================================

struct SignatureFile {
    BuildMetadata metadata;
    std::vector<ScenarioSignature> scenarios;

    std::string to_json() const {
        std::ostringstream os;
        os << "{\n";
        os << metadata.to_json() << ",\n";
        os << "  \"scenarios\": [\n";
        for (size_t i = 0; i < scenarios.size(); ++i) {
            if (i > 0) os << ",\n";
            os << scenarios[i].to_json();
        }
        os << "\n  ]\n";
        os << "}\n";
        return os.str();
    }

    static SignatureFile from_json(const std::string& json) {
        SignatureFile file;
        file.metadata = BuildMetadata::from_json(json);

        // Parse scenarios array
        auto scenarios_start = json.find("\"scenarios\":");
        if (scenarios_start == std::string::npos) return file;

        size_t array_start = json.find("[", scenarios_start);
        if (array_start == std::string::npos) return file;

        // Find matching ] for the scenarios array
        size_t array_end = array_start + 1;
        int bracket_count = 1;
        while (array_end < json.size() && bracket_count > 0) {
            if (json[array_end] == '[') ++bracket_count;
            else if (json[array_end] == ']') --bracket_count;
            ++array_end;
        }

        // Find each scenario object within the array
        size_t pos = array_start;
        while ((pos = json.find("{", pos)) != std::string::npos) {
            // Check if we've gone past the scenarios array
            if (pos >= array_end) break;

            size_t end = pos + 1;
            int brace_count = 1;
            while (end < json.size() && brace_count > 0) {
                if (json[end] == '{') ++brace_count;
                else if (json[end] == '}') --brace_count;
                ++end;
            }

            std::string scenario_json = json.substr(pos, end - pos);

            ScenarioSignature sig;

            // Extract name
            auto extract_string = [&](const std::string& key) -> std::string {
                std::string pattern = "\"" + key + "\": \"";
                auto p = scenario_json.find(pattern);
                if (p == std::string::npos) return "";
                p += pattern.size();
                auto e = scenario_json.find("\"", p);
                return scenario_json.substr(p, e - p);
            };

            sig.name = extract_string("name");
            if (sig.name.empty()) { pos = end; continue; }

            // Parse metrics
            auto metrics_start = scenario_json.find("\"metrics\":");
            auto metrics_end = scenario_json.find("}", metrics_start);
            if (metrics_start != std::string::npos && metrics_end != std::string::npos) {
                std::string metrics_block = scenario_json.substr(metrics_start, metrics_end - metrics_start + 1);

                size_t mpos = 0;
                while ((mpos = metrics_block.find("\"", mpos)) != std::string::npos) {
                    size_t name_start = mpos + 1;
                    size_t name_end = metrics_block.find("\"", name_start);
                    if (name_end == std::string::npos) break;

                    std::string mname = metrics_block.substr(name_start, name_end - name_start);
                    if (mname == "metrics") { mpos = name_end + 1; continue; }

                    size_t colon = metrics_block.find(":", name_end);
                    if (colon == std::string::npos) break;

                    size_t val_start = colon + 1;
                    while (val_start < metrics_block.size() &&
                           (metrics_block[val_start] == ' ' || metrics_block[val_start] == '\n'))
                        ++val_start;

                    // Use stod with parsed-length overload to handle NaN/Inf values
                    // (character-by-character scan would miss these, masking failures)
                    size_t val_end = val_start;
                    size_t parsed = 0;
                    try {
                        double val = std::stod(metrics_block.substr(val_start), &parsed);
                        val_end = val_start + parsed;
                        // Apply per-metric tolerances (not serialized in JSON)
                        sig.metrics[mname] = get_metric_with_tolerance(sig.name, mname, val);
                    } catch (...) {
                        val_end = val_start + 1;  // Advance to avoid stalling
                    }

                    mpos = val_end;
                }
            }

            // Parse probes
            auto probes_start = scenario_json.find("\"probes\":");
            if (probes_start != std::string::npos) {
                size_t ppos = probes_start;
                while ((ppos = scenario_json.find("\"", ppos)) != std::string::npos) {
                    size_t name_start = ppos + 1;
                    size_t name_end = scenario_json.find("\"", name_start);
                    if (name_end == std::string::npos) break;

                    std::string pname = scenario_json.substr(name_start, name_end - name_start);
                    if (pname == "probes" || pname == "}") { ppos = name_end + 1; continue; }

                    size_t bracket = scenario_json.find("[", name_end);
                    if (bracket == std::string::npos) break;
                    size_t bracket_end_p = scenario_json.find("]", bracket);
                    if (bracket_end_p == std::string::npos) break;

                    std::string arr = scenario_json.substr(bracket + 1, bracket_end_p - bracket - 1);
                    Probe p;
                    std::istringstream iss(arr);
                    double v;
                    char comma;
                    while (iss >> v) {
                        p.values.push_back(v);
                        iss >> comma;
                    }
                    if (!p.values.empty()) {
                        sig.probes[pname] = p;
                    }

                    ppos = bracket_end_p + 1;
                }
            }

            file.scenarios.push_back(sig);
            pos = end;
        }

        return file;
    }
};

//=============================================================================
// Comparison result
//=============================================================================

struct ComparisonResult {
    std::string scenario;
    bool passed = true;

    struct MetricDiff {
        std::string name;
        double ref_value = 0.0;
        double test_value = 0.0;
        double abs_diff = 0.0;
        double tolerance = 0.0;
        bool passed = false;
    };
    std::vector<MetricDiff> metric_diffs;

    struct ProbeDiff {
        std::string name;
        double max_diff = 0.0;
        double tolerance = 0.0;
        bool passed = false;
    };
    std::vector<ProbeDiff> probe_diffs;
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
    #if defined(_OPENMP)
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
    #else
    std::cerr << "ERROR: USE_GPU_OFFLOAD requires OpenMP support\n";
    return false;
    #endif
#endif
    return true;
}

//=============================================================================
// QoI computation helpers (note: compute_kinetic_energy, compute_max_divergence,
// compute_enstrophy_2d are from test_utilities.hpp)
//=============================================================================

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
    if (count == 0) {
        throw std::runtime_error("compute_mean_u: empty interior domain (count == 0)");
    }
    return sum / count;
}

double compute_pressure_rms(const Mesh& mesh, const ScalarField& p) {
    double sum_sq = 0.0;
    double mean = 0.0;
    int count = 0;

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
    if (count == 0) {
        throw std::runtime_error("compute_pressure_rms: empty interior domain (count == 0)");
    }
    mean /= count;

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

double compute_mean_pressure(const Mesh& mesh, const ScalarField& p) {
    double sum = 0.0;
    int count = 0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += p(i, j);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    sum += p(i, j, k);
                    ++count;
                }
            }
        }
    }
    if (count == 0) {
        throw std::runtime_error("compute_mean_pressure: empty interior domain (count == 0)");
    }
    return sum / count;
}

/// Weighted checksum - order-sensitive metric to detect field-level differences
/// Computes sum(u[i] * weight[i]) where weight varies with position
/// This is sensitive to small perturbations that cancel in mean/RMS metrics
struct WeightedChecksum {
    double u_weighted;   // sum(u[i] * (idx+1))
    double v_weighted;   // sum(v[i] * (idx+1))
    double u_sq_weighted; // sum(u[i]^2 * (idx+1))
};

WeightedChecksum compute_weighted_checksum(const Mesh& mesh, const VectorField& vel) {
    WeightedChecksum cs{0.0, 0.0, 0.0};
    size_t idx = 1;  // Start at 1 to avoid zero weight

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                double w = static_cast<double>(idx);
                cs.u_weighted += u * w;
                cs.v_weighted += v * w;
                cs.u_sq_weighted += u * u * w;
                ++idx;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                    double w = static_cast<double>(idx);
                    cs.u_weighted += u * w;
                    cs.v_weighted += v * w;
                    cs.u_sq_weighted += u * u * w;
                    ++idx;
                }
            }
        }
    }
    return cs;
}

/// Collect 16 probes distributed across the 2D domain (4x4 grid)
/// Returns probe values as flattened array: [u0,v0, u1,v1, ...]
Probe collect_distributed_probes_2d(const Mesh& mesh, const VectorField& vel) {
    Probe p;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;

    // 4x4 grid of probes, avoiding ghost cells
    for (int pj = 0; pj < 4; ++pj) {
        for (int pi = 0; pi < 4; ++pi) {
            int i = mesh.i_begin() + (pi * Nx) / 4 + Nx / 8;
            int j = mesh.j_begin() + (pj * Ny) / 4 + Ny / 8;
            // Clamp to valid range
            i = std::min(i, mesh.i_end() - 1);
            j = std::min(j, mesh.j_end() - 1);

            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            p.values.push_back(u);
            p.values.push_back(v);
        }
    }
    return p;
}

/// Collect probes for 3D: 8 corners + center = 9 probes
Probe collect_distributed_probes_3d(const Mesh& mesh, const VectorField& vel) {
    Probe p;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;

    // 8 corners + center
    int positions[9][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},  // z=0 corners
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},  // z=1 corners
        {1, 1, 1}  // center (will be computed differently)
    };

    for (int n = 0; n < 9; ++n) {
        int i, j, k;
        if (n < 8) {
            // Corners at 1/4 and 3/4 positions
            i = mesh.i_begin() + (positions[n][0] ? 3*Nx/4 : Nx/4);
            j = mesh.j_begin() + (positions[n][1] ? 3*Ny/4 : Ny/4);
            k = mesh.k_begin() + (positions[n][2] ? 3*Nz/4 : Nz/4);
        } else {
            // Center
            i = mesh.i_begin() + Nx/2;
            j = mesh.j_begin() + Ny/2;
            k = mesh.k_begin() + Nz/2;
        }
        // Clamp
        i = std::min(i, mesh.i_end() - 1);
        j = std::min(j, mesh.j_end() - 1);
        k = std::min(k, mesh.k_end() - 1);

        double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
        double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
        double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
        p.values.push_back(u);
        p.values.push_back(v);
        p.values.push_back(w);
    }
    return p;
}

//=============================================================================
// Scenario definition
//=============================================================================

struct Scenario {
    std::string name;
    std::string description;
    std::function<bool()> is_available;
    std::function<ScenarioSignature()> run;
};

//=============================================================================
// Scenario: Channel flow with body force (3D)
//=============================================================================

ScenarioSignature run_channel_solver_3d() {
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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));

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

    ScenarioSignature sig;
    sig.name = "channel_solver_3d";

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));
    sig.metrics["p_rms"] = Metric(compute_pressure_rms(mesh, solver.pressure()));

    // Weighted checksum - sensitive to field-level differences that cancel in mean
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_v"] = Metric(cs.v_weighted);
    sig.metrics["checksum_u2"] = Metric(cs.u_sq_weighted);

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

    // Distributed probes (9 locations: 8 corners + center)
    sig.probes["vel_distributed"] = collect_distributed_probes_3d(mesh, solver.velocity());

    return sig;
}

//=============================================================================
// Scenario: Taylor-Green Vortex 2D
//=============================================================================

ScenarioSignature run_tgv_2d() {
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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

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

    ScenarioSignature sig;
    sig.name = "tgv_2d";

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["enstrophy"] = Metric(compute_enstrophy_2d(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["p_rms"] = Metric(compute_pressure_rms(mesh, solver.pressure()));

    // Weighted checksum - order-sensitive metric
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_v"] = Metric(cs.v_weighted);
    sig.metrics["checksum_u2"] = Metric(cs.u_sq_weighted);

    int pi = mesh.i_begin() + N/2;
    int pj = mesh.j_begin() + N/2;

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_center"] = vel_probe;

    // Distributed probes (4x4 grid = 16 locations)
    sig.probes["vel_distributed"] = collect_distributed_probes_2d(mesh, solver.velocity());

    return sig;
}

//=============================================================================
// Scenario: Poiseuille 2D
//=============================================================================

ScenarioSignature run_poiseuille_2d() {
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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

    double H = 1.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j) - H;
        double u_exact = 0.5 * 0.01 / config.nu * (H*H - y*y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_exact * 0.9;
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

    ScenarioSignature sig;
    sig.name = "poiseuille_2d";

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));

    // Weighted checksum
    // Note: v-velocity is ~0 for Poiseuille flow, so use larger abs_tol for checksum_v
    // to allow for roundoff noise differences between CPU and GPU
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_v"] = Metric(cs.v_weighted, 1e-8, 1e-8);  // Looser for ~0 values

    int pi = mesh.i_begin() + NX/2;
    int pj = mesh.j_begin() + NY/2;

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_center"] = vel_probe;
    sig.probes["vel_distributed"] = collect_distributed_probes_2d(mesh, solver.velocity());

    return sig;
}

//=============================================================================
// Scenario: Projection invariants (mean preservation, div reduction)
//=============================================================================

ScenarioSignature run_projection_invariants() {
    const int N = 24;
    const int NUM_STEPS = 5;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);

    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Initial condition with known mean
    double u_mean_init = 0.5;
    double v_mean_init = 0.3;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            double y = mesh.y(j);
            solver.velocity().u(i, j) = u_mean_init + 0.1 * std::sin(x) * std::cos(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.yf[j];
            solver.velocity().v(i, j) = v_mean_init - 0.1 * std::cos(x) * std::sin(y);
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

    ScenarioSignature sig;
    sig.name = "projection_invariants";

    // Mean should be preserved (periodic BCs)
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()), 1e-10, 1e-8);

    // Post-projection divergence should be small
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);

    // Pressure mean (should be ~0 for periodic)
    sig.metrics["p_mean"] = Metric(compute_mean_pressure(mesh, solver.pressure()), 1e-10, 1e-8);

    // Kinetic energy (for stability check)
    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));

    // Weighted checksum - order-sensitive metric
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_u2"] = Metric(cs.u_sq_weighted);

    // Distributed probes
    sig.probes["vel_distributed"] = collect_distributed_probes_2d(mesh, solver.velocity());

    return sig;
}

//=============================================================================
// Scenario: Mixing Length turbulence
//=============================================================================

ScenarioSignature run_mixing_length() {
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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

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

    ScenarioSignature sig;
    sig.name = "mixing_length";

    sig.metrics["ke"] = Metric(compute_kinetic_energy(mesh, solver.velocity()));
    sig.metrics["div_max"] = Metric(compute_max_divergence(mesh, solver.velocity()), 1e-8, 1e-6);
    sig.metrics["mean_u"] = Metric(compute_mean_u(mesh, solver.velocity()));

    // Weighted checksum - order-sensitive metric
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_u2"] = Metric(cs.u_sq_weighted);

    int pi = mesh.i_begin() + NX/2;
    int pj = mesh.j_begin() + NY/4;

    Probe vel_probe;
    vel_probe.values = {
        0.5 * (solver.velocity().u(pi, pj) + solver.velocity().u(pi+1, pj)),
        0.5 * (solver.velocity().v(pi, pj) + solver.velocity().v(pi, pj+1))
    };
    sig.probes["vel_quarter"] = vel_probe;
    sig.probes["vel_distributed"] = collect_distributed_probes_2d(mesh, solver.velocity());

    return sig;
}

//=============================================================================
// Scenario: NN-MLP turbulence (via RANSSolver public API)
//=============================================================================

bool nn_mlp_available() {
    // Check if model files exist
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

ScenarioSignature run_nn_mlp() {
    // Test NN-MLP turbulence model through RANSSolver public API
    // This ensures proper GPU buffer management via device_view
    const int NX = 32, NY = 48;
    const int NUM_STEPS = 5;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.Nx = NX;
    config.Ny = NY;
    config.nu = 0.001;
    config.dt = 0.001;
    config.max_steps = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::NNMLP;
    config.nn_weights_path = get_nn_mlp_model_path();
    config.nn_scaling_path = get_nn_mlp_model_path();
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.01, 0.0, 0.0);

    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

    // Initialize with parabolic profile
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_base = 1.0 - y * y;
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_base;
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run a few steps to exercise the NN-MLP turbulence model
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    ScenarioSignature sig;
    sig.name = "nn_mlp";

    // Extract nu_t statistics from solver
    const ScalarField& nu_t = solver.nu_t();
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

    // Weighted checksum - order-sensitive metric for velocity field
    auto cs = compute_weighted_checksum(mesh, solver.velocity());
    sig.metrics["checksum_u"] = Metric(cs.u_weighted);
    sig.metrics["checksum_u2"] = Metric(cs.u_sq_weighted);

    int pi = mesh.i_begin() + NX/2;

    Probe nu_t_profile;
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + NY/4));
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + NY/2));
    nu_t_profile.values.push_back(nu_t(pi, mesh.j_begin() + 3*NY/4));
    sig.probes["nu_t_profile"] = nu_t_profile;

    // Distributed velocity probes
    sig.probes["vel_distributed"] = collect_distributed_probes_2d(mesh, solver.velocity());

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
        run_channel_solver_3d
    });

    scenarios.push_back({
        "tgv_2d",
        "2D Taylor-Green vortex",
        []() { return true; },
        run_tgv_2d
    });

    scenarios.push_back({
        "poiseuille_2d",
        "2D Poiseuille flow development",
        []() { return true; },
        run_poiseuille_2d
    });

    scenarios.push_back({
        "projection_invariants",
        "Projection mean preservation and divergence",
        []() { return true; },
        run_projection_invariants
    });

    scenarios.push_back({
        "mixing_length",
        "Baseline (mixing length) turbulence model",
        []() { return true; },
        run_mixing_length
    });

    scenarios.push_back({
        "nn_mlp",
        "NN-MLP turbulence model",
        nn_mlp_available,
        run_nn_mlp
    });

    return scenarios;
}

//=============================================================================
// Dump mode: Generate signatures
//=============================================================================

int run_dump_mode(const std::string& output_file, const std::set<std::string>& filter) {
    std::cout << "=== Signature Generation Mode ===\n";
    print_backend_identity();
    std::cout << "Output file: " << output_file << "\n\n";

    if (!verify_backend()) {
        std::cerr << "ERROR: Backend verification failed\n";
        return 1;
    }

    SignatureFile sig_file;
    sig_file.metadata = BuildMetadata::capture();

    auto scenarios = get_scenarios();
    int run_count = 0, skip_count = 0;

    for (const auto& scenario : scenarios) {
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
        ScenarioSignature sig = scenario.run();
        sig_file.scenarios.push_back(sig);

        std::cout << "  Metrics:\n";
        for (const auto& [name, m] : sig.metrics) {
            std::cout << "    " << name << ": " << std::scientific << m.value << "\n";
        }
        std::cout << "\n";
        ++run_count;
    }

    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "ERROR: Cannot open output file: " << output_file << "\n";
        return 1;
    }

    out << sig_file.to_json();

    std::cout << "[SUCCESS] Wrote " << run_count << " signatures";
    if (skip_count > 0) std::cout << " (" << skip_count << " skipped)";
    std::cout << " to " << output_file << "\n";

    return 0;
}

//=============================================================================
// Compare mode: Compare two signature files
//=============================================================================

// Simple string hash for file content verification
static std::string compute_content_hash(const std::string& content) {
    // Use std::hash for a quick fingerprint (not cryptographic, but catches accidental duplication)
    std::hash<std::string> hasher;
    size_t hash = hasher(content);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

int run_compare_mode(const std::string& ref_file, const std::string& test_file) {
    std::cout << "=== Signature Comparison Mode ===\n";
    std::cout << "Reference: " << ref_file << "\n";
    std::cout << "Test:      " << test_file << "\n\n";

    // Load reference
    std::ifstream ref_in(ref_file);
    if (!ref_in) {
        std::cerr << "ERROR: Cannot open reference file: " << ref_file << "\n";
        return 1;
    }
    std::string ref_json((std::istreambuf_iterator<char>(ref_in)),
                          std::istreambuf_iterator<char>());
    SignatureFile ref_sig = SignatureFile::from_json(ref_json);

    // Load test
    std::ifstream test_in(test_file);
    if (!test_in) {
        std::cerr << "ERROR: Cannot open test file: " << test_file << "\n";
        return 1;
    }
    std::string test_json((std::istreambuf_iterator<char>(test_in)),
                           std::istreambuf_iterator<char>());
    SignatureFile test_sig = SignatureFile::from_json(test_json);

    // Compute file hashes for paranoia check
    std::string ref_hash = compute_content_hash(ref_json);
    std::string test_hash = compute_content_hash(test_json);

    // Print metadata comparison with file hashes
    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ REFERENCE (" << ref_sig.metadata.backend << ")";
    std::cout << std::string(50 - ref_sig.metadata.backend.length(), ' ') << "│\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│  File:     " << ref_file << std::string(std::max(0, 48 - (int)ref_file.length()), ' ') << "│\n";
    std::cout << "│  Hash:     " << ref_hash << "                                │\n";
    std::cout << "│  Git SHA:  " << ref_sig.metadata.git_sha << std::string(std::max(0, 48 - (int)ref_sig.metadata.git_sha.length()), ' ') << "│\n";
    std::cout << "│  Device:   " << ref_sig.metadata.device << std::string(std::max(0, 48 - (int)ref_sig.metadata.device.length()), ' ') << "│\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ TEST (" << test_sig.metadata.backend << ")";
    std::cout << std::string(54 - test_sig.metadata.backend.length(), ' ') << "│\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│  File:     " << test_file << std::string(std::max(0, 48 - (int)test_file.length()), ' ') << "│\n";
    std::cout << "│  Hash:     " << test_hash << "                                │\n";
    std::cout << "│  Git SHA:  " << test_sig.metadata.git_sha << std::string(std::max(0, 48 - (int)test_sig.metadata.git_sha.length()), ' ') << "│\n";
    std::cout << "│  Device:   " << test_sig.metadata.device << std::string(std::max(0, 48 - (int)test_sig.metadata.device.length()), ' ') << "│\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";

    // GUARDRAIL: Check file hashes - identical hashes are very suspicious
    if (ref_hash == test_hash) {
        std::cerr << "WARNING: File hashes are identical!\n";
        std::cerr << "         This could indicate comparing the same file twice.\n";
        std::cerr << "         Proceeding with caution...\n\n";
    }

    // GUARDRAIL: Fail if both files have the same backend
    // This catches the bug where we accidentally compare CPU vs CPU
    if (ref_sig.metadata.backend == test_sig.metadata.backend) {
        std::cerr << "ERROR: Both files have the same backend '" << ref_sig.metadata.backend << "'\n";
        std::cerr << "       Cross-backend test requires CPU vs GPU comparison.\n";
        std::cerr << "       Check that you're not comparing the same file twice.\n";
        return 1;
    }

    // GUARDRAIL: For GPU backend, verify we're actually on a GPU device
    if (test_sig.metadata.backend == "GPU") {
        if (test_sig.metadata.device == "CPU" || test_sig.metadata.device.empty()) {
            std::cerr << "ERROR: GPU backend but device is '" << test_sig.metadata.device << "'\n";
            std::cerr << "       GPU offload may not be working correctly.\n";
            return 1;
        }
    }

    std::cout << "[OK] Backend mismatch verified: " << ref_sig.metadata.backend
              << " vs " << test_sig.metadata.backend << "\n\n";

    // Build lookup for test scenarios
    std::map<std::string, const ScenarioSignature*> test_map;
    for (const auto& sig : test_sig.scenarios) {
        test_map[sig.name] = &sig;
    }

    bool all_passed = true;
    int pass_count = 0, fail_count = 0, skip_count = 0;
    std::vector<ComparisonResult> results;

    for (const auto& ref : ref_sig.scenarios) {
        ComparisonResult result;
        result.scenario = ref.name;

        // Clear scenario header
        std::cout << "\n════════════════════════════════════════════════════════════════\n";
        std::cout << "  SCENARIO: " << ref.name << "\n";
        std::cout << "════════════════════════════════════════════════════════════════\n";

        auto it = test_map.find(ref.name);
        if (it == test_map.end()) {
            std::cout << "  [SKIPPED] Not found in test file\n\n";
            ++skip_count;
            continue;
        }

        const ScenarioSignature& test = *it->second;

        // Compare metrics
        std::cout << "  Metrics:\n";
        for (const auto& [name, ref_metric] : ref.metrics) {
            ComparisonResult::MetricDiff diff;
            diff.name = name;
            diff.ref_value = ref_metric.value;

            auto mit = test.metrics.find(name);
            if (mit == test.metrics.end()) {
                std::cout << "    " << std::left << std::setw(15) << name
                          << ": [FAIL] Missing in test\n";
                diff.passed = false;
                result.passed = false;
            } else {
                diff.test_value = mit->second.value;
                diff.abs_diff = std::abs(diff.ref_value - diff.test_value);
                double scale = std::max(std::abs(diff.ref_value), std::abs(diff.test_value));
                diff.tolerance = ref_metric.abs_tol + ref_metric.rel_tol * scale;
                diff.passed = (diff.abs_diff <= diff.tolerance);

                std::cout << "    " << std::left << std::setw(15) << name << ": ";
                // Use higher precision to detect near-zero differences
                std::cout << std::scientific << std::setprecision(10);
                std::cout << "diff=" << diff.abs_diff << " tol=" << diff.tolerance;

                // Calculate how close we are to the tolerance
                double margin_pct = (diff.tolerance > 0) ? (diff.abs_diff / diff.tolerance) * 100.0 : 0.0;

                if (diff.passed) {
                    std::cout << " [PASS]";
                    // Warning if we're using >80% of tolerance
                    if (margin_pct > 80.0) {
                        std::cout << " [WARN: " << std::fixed << std::setprecision(0) << margin_pct << "% of tol]";
                        std::cout << std::scientific << std::setprecision(10);
                    }
                    // Always show ref/test values for full transparency
                    std::cout << "\n           ref=" << diff.ref_value << " test=" << diff.test_value << "\n";
                } else {
                    std::cout << " [FAIL]\n";
                    std::cout << "           ref=" << diff.ref_value << " test=" << diff.test_value << "\n";
                    result.passed = false;
                }
            }
            result.metric_diffs.push_back(diff);
        }

        // Compare probes
        std::cout << "  Probes:\n";
        for (const auto& [name, ref_probe] : ref.probes) {
            ComparisonResult::ProbeDiff diff;
            diff.name = name;

            auto pit = test.probes.find(name);
            if (pit == test.probes.end()) {
                std::cout << "    " << std::left << std::setw(15) << name
                          << ": [FAIL] Missing in test\n";
                diff.passed = false;
                result.passed = false;
            } else {
                const Probe& test_probe = pit->second;
                if (ref_probe.values.size() != test_probe.values.size()) {
                    std::cout << "    " << std::left << std::setw(15) << name
                              << ": [FAIL] Size mismatch\n";
                    diff.passed = false;
                    result.passed = false;
                } else {
                    diff.max_diff = 0.0;
                    for (size_t i = 0; i < ref_probe.values.size(); ++i) {
                        diff.max_diff = std::max(diff.max_diff,
                            std::abs(ref_probe.values[i] - test_probe.values[i]));
                    }
                    diff.tolerance = CROSS_BACKEND_TOLERANCE;
                    diff.passed = (diff.max_diff <= diff.tolerance);

                    std::cout << "    " << std::left << std::setw(15) << name << ": ";
                    // Use higher precision for probes too
                    std::cout << std::scientific << std::setprecision(10);
                    std::cout << "max_diff=" << diff.max_diff;
                    if (diff.passed) {
                        std::cout << " [PASS]\n";
                    } else {
                        std::cout << " [FAIL]\n";
                        result.passed = false;
                    }
                }
            }
            result.probe_diffs.push_back(diff);
        }

        if (result.passed) {
            std::cout << "  [SCENARIO PASS]\n\n";
            ++pass_count;
        } else {
            std::cout << "  [SCENARIO FAIL]\n\n";
            all_passed = false;
            ++fail_count;
        }

        results.push_back(result);
    }

    // Summary
    std::cout << "=== Summary ===\n";
    std::cout << "Passed:  " << pass_count << "\n";
    std::cout << "Failed:  " << fail_count << "\n";
    std::cout << "Skipped: " << skip_count << "\n\n";

    // Guard against false positives: if nothing was actually compared, fail
    if (pass_count == 0 && fail_count == 0) {
        std::cout << "[FAILURE] No scenarios were compared (all skipped or empty reference file)\n";
        return 1;
    }

    // Failure details
    if (!all_passed) {
        std::cout << "=== Failure Details ===\n";
        for (const auto& result : results) {
            if (!result.passed) {
                std::cout << "Scenario: " << result.scenario << "\n";
                for (const auto& md : result.metric_diffs) {
                    if (!md.passed) {
                        std::cout << "  " << md.name << ":\n";
                        std::cout << "    ref:  " << std::scientific << md.ref_value << "\n";
                        std::cout << "    test: " << std::scientific << md.test_value << "\n";
                        std::cout << "    diff: " << std::scientific << md.abs_diff << "\n";
                        std::cout << "    tol:  " << std::scientific << md.tolerance << "\n";
                    }
                }
                for (const auto& pd : result.probe_diffs) {
                    if (!pd.passed) {
                        std::cout << "  " << pd.name << ": max_diff=" << pd.max_diff
                                  << " > tol=" << pd.tolerance << "\n";
                    }
                }
                std::cout << "\n";
            }
        }

        std::cout << "Artifact paths for debugging:\n";
        std::cout << "  Reference: " << ref_file << "\n";
        std::cout << "  Test:      " << test_file << "\n\n";

        std::cout << "[FAILURE] " << fail_count << " scenario(s) failed\n";
        return 1;
    }

    std::cout << "[SUCCESS] All scenarios passed\n";
    return 0;
}

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Cross-backend consistency test using compact signatures.\n\n";
    std::cout << "Workflow:\n";
    std::cout << "  1. CPU build: " << prog << " --dump cpu.json\n";
    std::cout << "  2. GPU build: " << prog << " --dump gpu.json\n";
    std::cout << "  3. Compare:   " << prog << " --compare cpu.json gpu.json\n\n";
    std::cout << "Options:\n";
    std::cout << "  --dump <file>              Generate signatures for this backend\n";
    std::cout << "  --compare <ref> <test>     Compare two signature files\n";
    std::cout << "  --scenarios <list>         Comma-separated scenario filter\n";
    std::cout << "  --list                     List available scenarios\n";
    std::cout << "  --help                     Show this message\n\n";
    std::cout << "Scenarios:\n";
    for (const auto& s : get_scenarios()) {
        std::cout << "  " << std::left << std::setw(22) << s.name;
        std::cout << s.description;
        if (!s.is_available()) std::cout << " (unavailable)";
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string dump_file;
        std::string compare_ref, compare_test;
        std::set<std::string> scenario_filter;
        bool list_scenarios = false;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
                dump_file = argv[++i];
            } else if (std::strcmp(argv[i], "--compare") == 0 && i + 2 < argc) {
                compare_ref = argv[++i];
                compare_test = argv[++i];
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
                std::cout << "  " << std::left << std::setw(22) << s.name;
                std::cout << (s.is_available() ? "[available]  " : "[unavailable]");
                std::cout << s.description << "\n";
            }
            return 0;
        }

        std::cout << "=== Cross-Backend Consistency Test ===\n";
        std::cout << "Build: " << get_backend_name() << "\n";
        std::cout << "Tolerance: " << std::scientific << CROSS_BACKEND_TOLERANCE << "\n\n";

        if (!dump_file.empty()) {
            return run_dump_mode(dump_file, scenario_filter);
        } else if (!compare_ref.empty() && !compare_test.empty()) {
            return run_compare_mode(compare_ref, compare_test);
        } else {
            std::cerr << "ERROR: Specify --dump <file> or --compare <ref> <test>\n\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
