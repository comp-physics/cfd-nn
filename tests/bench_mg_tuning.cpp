/// @file bench_mg_tuning.cpp
/// @brief Benchmark MG tuning parameters for quality vs performance
///
/// Tests different combinations of nu1, nu2, degree and measures:
/// - Wall time per step (mean ± stddev over repeats)
/// - Post-projection divergence (||div(u)||_L2 and ||div(u)||_Linf)
/// - KE drift per step
///
/// Usage: bench_mg_tuning [grid_size] [nsteps] [nrepeats]
///   grid_size: 64, 128, or 192 (default: 128)
///   nsteps:    steps per repeat (default: 20)
///   nrepeats:  number of repeats for statistics (default: 5)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "profiling.hpp"
#include "test_utilities.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace nncfd;
using namespace std::chrono;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

struct TuningConfig {
    std::string name;
    int nu1;
    int nu2;
    int degree;
    int fixed_cycles;
    bool adaptive;
};

struct TuningResult {
    double ms_per_step;
    double div_l2;      // ||div(u)||_2 post-projection
    double div_inf;     // ||div(u)||_∞ post-projection
    double ke_drift;    // Relative KE change per step
};

struct TuningStats {
    double ms_mean, ms_std;
    double div_l2_mean, div_l2_std;
    double div_inf_mean, div_inf_std;
    double ke_drift_mean, ke_drift_std;
};

double compute_divergence_l2(const Mesh& mesh, const VectorField& vel) {
    double div_sum_sq = 0.0;
    const double inv_dx = 1.0 / mesh.dx;
    const double inv_dy = 1.0 / mesh.dy;
    const double inv_dz = 1.0 / mesh.dz;

    for (int k = 1; k <= mesh.Nz; ++k) {
        for (int j = 1; j <= mesh.Ny; ++j) {
            for (int i = 1; i <= mesh.Nx; ++i) {
                double du_dx = (vel.u(i+1, j, k) - vel.u(i, j, k)) * inv_dx;
                double dv_dy = (vel.v(i, j+1, k) - vel.v(i, j, k)) * inv_dy;
                double dw_dz = (mesh.Nz > 1) ?
                    (vel.w(i, j, k+1) - vel.w(i, j, k)) * inv_dz : 0.0;
                double div = du_dx + dv_dy + dw_dz;
                div_sum_sq += div * div;
            }
        }
    }
    return std::sqrt(div_sum_sq / (mesh.Nx * mesh.Ny * mesh.Nz));
}

double compute_divergence_inf(const Mesh& mesh, const VectorField& vel) {
    double div_max = 0.0;
    const double inv_dx = 1.0 / mesh.dx;
    const double inv_dy = 1.0 / mesh.dy;
    const double inv_dz = 1.0 / mesh.dz;

    for (int k = 1; k <= mesh.Nz; ++k) {
        for (int j = 1; j <= mesh.Ny; ++j) {
            for (int i = 1; i <= mesh.Nx; ++i) {
                double du_dx = (vel.u(i+1, j, k) - vel.u(i, j, k)) * inv_dx;
                double dv_dy = (vel.v(i, j+1, k) - vel.v(i, j, k)) * inv_dy;
                double dw_dz = (mesh.Nz > 1) ?
                    (vel.w(i, j, k+1) - vel.w(i, j, k)) * inv_dz : 0.0;
                double div = std::abs(du_dx + dv_dy + dw_dz);
                div_max = std::max(div_max, div);
            }
        }
    }
    return div_max;
}

double compute_kinetic_energy(const Mesh& mesh, const VectorField& vel) {
    double ke = 0.0;

    // Sum 0.5 * (u^2 + v^2 + w^2) over cell centers
    for (int k = 1; k <= mesh.Nz; ++k) {
        for (int j = 1; j <= mesh.Ny; ++j) {
            for (int i = 1; i <= mesh.Nx; ++i) {
                double u_c = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                double v_c = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                double w_c = (mesh.Nz > 1) ?
                    0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1)) : 0.0;
                ke += 0.5 * (u_c*u_c + v_c*v_c + w_c*w_c);
            }
        }
    }
    return ke;
}

TuningResult run_single_benchmark(const TuningConfig& cfg, int N, int nsteps) {
    // Set up mesh and solver
    const double L = 2.0 * M_PI;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = 1e-4;
    config.dt = 1e-4;
    config.adaptive_dt = false;
    config.max_steps = 20;
    config.turb_model = TurbulenceModelType::None;
    config.poisson_solver = PoissonSolverType::MG;
    config.verbose = false;
    config.output_freq = 0;
    config.write_fields = false;
    config.num_snapshots = 0;
    config.turb_guard_enabled = false;  // Disable guard for benchmark

    // MG tuning parameters
    config.poisson_fixed_cycles = cfg.fixed_cycles;
    config.poisson_nu1 = cfg.nu1;
    config.poisson_nu2 = cfg.nu2;
    config.poisson_chebyshev_degree = cfg.degree;
    config.poisson_adaptive_cycles = cfg.adaptive;
    config.poisson_check_after = 4;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));

    // Initialize Taylor-Green vortex (divergence-free)
    const double U0 = 1.0;
    for (int k = 1; k <= N; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= N; ++j) {
            double y = mesh.y(j);
            for (int i = 1; i <= N + 1; ++i) {
                double x = mesh.xf[i];
                solver.velocity().u(i, j, k) = U0 * std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = 1; k <= N; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= N + 1; ++j) {
            double y = mesh.yf[j];
            for (int i = 1; i <= N; ++i) {
                double x = mesh.x(i);
                solver.velocity().v(i, j, k) = -U0 * std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Warmup (2 steps to stabilize)
    solver.step();
    solver.step();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    // Measure initial KE
    double ke0 = compute_kinetic_energy(mesh, solver.velocity());

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Timed run
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nsteps; ++i) {
        solver.step();
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0;

    // Measure quality metrics
#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    double ke_final = compute_kinetic_energy(mesh, solver.velocity());

    TuningResult result;
    result.ms_per_step = elapsed_ms / nsteps;
    result.div_l2 = compute_divergence_l2(mesh, solver.velocity());
    result.div_inf = compute_divergence_inf(mesh, solver.velocity());
    result.ke_drift = (ke_final - ke0) / ke0 / nsteps;  // Relative per step

    return result;
}

TuningStats run_benchmark_with_stats(const TuningConfig& cfg, int N, int nsteps, int nrepeats) {
    std::vector<double> ms_samples, div_l2_samples, div_inf_samples, ke_samples;

    for (int r = 0; r < nrepeats; ++r) {
        TuningResult result = run_single_benchmark(cfg, N, nsteps);
        ms_samples.push_back(result.ms_per_step);
        div_l2_samples.push_back(result.div_l2);
        div_inf_samples.push_back(result.div_inf);
        ke_samples.push_back(result.ke_drift);
    }

    auto compute_stats = [](const std::vector<double>& v) -> std::pair<double, double> {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq_sum = 0.0;
        for (double x : v) sq_sum += (x - mean) * (x - mean);
        double std = (v.size() > 1) ? std::sqrt(sq_sum / (v.size() - 1)) : 0.0;
        return {mean, std};
    };

    TuningStats stats;
    auto [ms_m, ms_s] = compute_stats(ms_samples);
    auto [l2_m, l2_s] = compute_stats(div_l2_samples);
    auto [inf_m, inf_s] = compute_stats(div_inf_samples);
    auto [ke_m, ke_s] = compute_stats(ke_samples);

    stats.ms_mean = ms_m; stats.ms_std = ms_s;
    stats.div_l2_mean = l2_m; stats.div_l2_std = l2_s;
    stats.div_inf_mean = inf_m; stats.div_inf_std = inf_s;
    stats.ke_drift_mean = ke_m; stats.ke_drift_std = ke_s;

    return stats;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int N = (argc > 1) ? std::atoi(argv[1]) : 128;
    int nsteps = (argc > 2) ? std::atoi(argv[2]) : 20;
    int nrepeats = (argc > 3) ? std::atoi(argv[3]) : 5;

    std::cout << "======================================================================\n";
    std::cout << "  MG Tuning Benchmark: Quality vs Performance (with statistics)\n";
    std::cout << "======================================================================\n";
    std::cout << "  Grid: " << N << "³ | Steps: " << nsteps << " | Repeats: " << nrepeats << "\n";
    std::cout << "  BCs: Channel (PWP) - periodic x/z, walls y\n";
    std::cout << "======================================================================\n\n";

    // Key configurations to test (based on user suggestions)
    std::vector<TuningConfig> configs = {
        // Baseline configurations
        {"nu1=2,nu2=2,deg=4,cyc=10", 2, 2, 4, 10, false},  // Old baseline
        {"nu1=2,nu2=2,deg=4,cyc=8",  2, 2, 4, 8, false},

        // Optimal asymmetric configurations
        {"nu1=2,nu2=1,deg=4,cyc=10", 2, 1, 4, 10, false},
        {"nu1=2,nu2=1,deg=4,cyc=8",  2, 1, 4, 8, false},   // Current optimal
        {"nu1=2,nu2=1,deg=4,cyc=6",  2, 1, 4, 6, false},

        // More asymmetric: nu1=3
        {"nu1=3,nu2=1,deg=4,cyc=8",  3, 1, 4, 8, false},

        // Degree variations
        {"nu1=2,nu2=1,deg=3,cyc=8",  2, 1, 3, 8, false},
        {"nu1=2,nu2=2,deg=3,cyc=8",  2, 2, 3, 8, false},

        // Adaptive mode
        {"nu1=2,nu2=1,deg=4,adpt",   2, 1, 4, 10, true},
    };

    // Print header
    std::cout << std::left << std::setw(26) << "Configuration"
              << std::right
              << std::setw(14) << "ms/step"
              << std::setw(16) << "div_L2"
              << std::setw(16) << "div_Linf"
              << std::setw(14) << "KE_drift"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    for (const auto& cfg : configs) {
        std::cout << std::left << std::setw(26) << cfg.name << std::flush;

        TuningStats stats = run_benchmark_with_stats(cfg, N, nsteps, nrepeats);

        // Format: mean±std
        auto fmt_ms = [](double m, double s) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << m << "±" << std::setprecision(2) << s;
            return oss.str();
        };
        auto fmt_sci = [](double m, double s) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(1) << m << "±" << std::setprecision(0) << s;
            return oss.str();
        };

        std::cout << std::right
                  << std::setw(14) << fmt_ms(stats.ms_mean, stats.ms_std)
                  << std::setw(16) << fmt_sci(stats.div_l2_mean, stats.div_l2_std)
                  << std::setw(16) << fmt_sci(stats.div_inf_mean, stats.div_inf_std)
                  << std::setw(14) << fmt_sci(stats.ke_drift_mean, stats.ke_drift_std)
                  << "\n";
    }

    std::cout << std::string(86, '-') << "\n";
    std::cout << "\nKey:\n";
    std::cout << "  - div_L2/Linf: Lower is better (post-projection velocity divergence)\n";
    std::cout << "  - KE_drift: Closer to zero is better (should be ~viscous decay)\n";
    std::cout << "  - Optimize for min(ms/step) while keeping div below threshold\n";
    std::cout << "  - Quality contract: div_L2 < 1e-5 and div_Linf < 1e-3 is typically adequate\n";
    std::cout << "======================================================================\n";

    return 0;
}
