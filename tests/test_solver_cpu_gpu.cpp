/// CPU vs GPU consistency tests for staggered grid solver
/// Tests core solver kernels: divergence, convection, diffusion, projection

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

struct SolverMetrics {
    double max_abs_u = 0.0;
    double max_abs_v = 0.0;
    double u_l2 = 0.0;
    double v_l2 = 0.0;
    double p_l2 = 0.0;
};

static SolverMetrics compute_metrics(const Mesh& mesh, const VectorField& vel, const ScalarField& p) {
    SolverMetrics m;
    const int Ng = mesh.Nghost;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;

    // u at x-faces
    double sum_u2 = 0.0;
    int count_u = 0;
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            const double u = vel.u(i, j);
            m.max_abs_u = std::max(m.max_abs_u, std::abs(u));
            sum_u2 += u * u;
            ++count_u;
        }
    }

    // v at y-faces
    double sum_v2 = 0.0;
    int count_v = 0;
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            const double v = vel.v(i, j);
            m.max_abs_v = std::max(m.max_abs_v, std::abs(v));
            sum_v2 += v * v;
            ++count_v;
        }
    }

    // pressure at cell centers
    double sum_p2 = 0.0;
    int count_p = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            const double pv = p(i, j);
            sum_p2 += pv * pv;
            ++count_p;
        }
    }

    m.u_l2 = std::sqrt(sum_u2 / std::max(1, count_u));
    m.v_l2 = std::sqrt(sum_v2 / std::max(1, count_v));
    m.p_l2 = std::sqrt(sum_p2 / std::max(1, count_p));
    return m;
}

static void write_kv_file(const std::string& filename, const std::map<std::string, double>& kv) {
    std::ofstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open for write: " + filename);
    }
    f.setf(std::ios::scientific);
    f.precision(17);
    f << "# solver_cpu_gpu_reference_v1\n";
    for (const auto& [k, v] : kv) {
        f << k << "=" << v << "\n";
    }
}

[[maybe_unused]] static std::map<std::string, double> read_kv_file(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open for read: " + filename);
    }
    std::map<std::string, double> kv;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const double val = std::stod(line.substr(eq + 1));
        kv[key] = val;
    }
    return kv;
}

[[maybe_unused]] static void compare_kv(const std::map<std::string, double>& ref,
                       const std::map<std::string, double>& got,
                       double tol_abs, double tol_rel) {
    for (const auto& [k, rv] : ref) {
        auto it = got.find(k);
        if (it == got.end()) {
            throw std::runtime_error("Missing key in output: " + k);
        }
        const double gv = it->second;
        const double absd = std::abs(gv - rv);
        const double reld = absd / (std::abs(rv) + 1e-30);
        if (absd > tol_abs && reld > tol_rel) {
            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss.precision(17);
            oss << "Mismatch at " << k << ": ref=" << rv << " got=" << gv
                << " abs=" << absd << " rel=" << reld;
            throw std::runtime_error(oss.str());
        }
    }
}

static std::map<std::string, double> run_all_cases_and_collect_metrics() {
    std::map<std::string, double> kv;

    // Case A: Taylor-Green vortex
    {
        Config config;
        config.Nx = 64;
        config.Ny = 64;
        config.x_min = 0.0;
        config.x_max = 2.0 * M_PI;
        config.y_min = 0.0;
        config.y_max = 2.0 * M_PI;
        config.nu = 0.01;
        config.dt = 0.0001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        Mesh mesh;
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);

        RANSSolver solver(mesh, config);
        VelocityBC bc;
        bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        VectorField vel_init(mesh);
        const int Ng = mesh.Nghost;
        for (int j = Ng; j < Ng + mesh.Ny; ++j) {
            for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
                double x = mesh.x_min + (i - Ng) * mesh.dx;
                double y = mesh.y(j);
                vel_init.u(i, j) = -std::cos(x) * std::sin(y);
            }
        }
        for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
            for (int i = Ng; i < Ng + mesh.Nx; ++i) {
                double x = mesh.x(i);
                double y = mesh.y_min + (j - Ng) * mesh.dy;
                vel_init.v(i, j) = std::sin(x) * std::cos(y);
            }
        }
        solver.initialize(vel_init);

        for (int step = 0; step < 10; ++step) {
            solver.step();
        }

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        const auto m = compute_metrics(mesh, solver.velocity(), solver.pressure());
        kv["tg.max_abs_u"] = m.max_abs_u;
        kv["tg.max_abs_v"] = m.max_abs_v;
        kv["tg.u_l2"] = m.u_l2;
        kv["tg.v_l2"] = m.v_l2;
        kv["tg.p_l2"] = m.p_l2;
    }

    // Case B: Channel flow
    {
        Config config;
        config.Nx = 64;
        config.Ny = 32;
        config.x_min = 0.0;
        config.x_max = 4.0;
        config.y_min = -1.0;
        config.y_max = 1.0;
        config.nu = 0.01;
        config.dp_dx = -0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        Mesh mesh;
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);

        RANSSolver solver(mesh, config);
        VelocityBC bc;
        bc.x_lo = bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);
        solver.set_body_force(-config.dp_dx, 0.0);
        solver.initialize_uniform(0.1, 0.0);

        for (int step = 0; step < 10; ++step) {
            solver.step();
        }

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        const auto m = compute_metrics(mesh, solver.velocity(), solver.pressure());
        kv["ch.max_abs_u"] = m.max_abs_u;
        kv["ch.max_abs_v"] = m.max_abs_v;
        kv["ch.u_l2"] = m.u_l2;
        kv["ch.v_l2"] = m.v_l2;
        kv["ch.p_l2"] = m.p_l2;
    }

    // Case C: grid sweep (track u-face max + L2)
    {
        struct GridSize { int nx, ny; };
        std::vector<GridSize> grids = {
            {32, 32},
            {64, 48},
            {63, 97},
            {128, 64}
        };

        for (const auto& g : grids) {
            Config config;
            config.Nx = g.nx;
            config.Ny = g.ny;
            config.x_min = 0.0;
            config.x_max = 2.0 * M_PI;
            config.y_min = 0.0;
            config.y_max = 2.0 * M_PI;
            config.nu = 0.01;
            config.dt = 0.0001;
            config.adaptive_dt = false;
            config.turb_model = TurbulenceModelType::None;
            config.verbose = false;

            Mesh mesh;
            mesh.init_uniform(config.Nx, config.Ny,
                              config.x_min, config.x_max,
                              config.y_min, config.y_max);

            RANSSolver solver(mesh, config);
            VelocityBC bc;
            bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
            solver.set_velocity_bc(bc);
            solver.initialize_uniform(0.5, 0.3);

            for (int step = 0; step < 5; ++step) {
                solver.step();
            }

#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif

            const auto m = compute_metrics(mesh, solver.velocity(), solver.pressure());
            const std::string tag = "gs." + std::to_string(g.nx) + "x" + std::to_string(g.ny);
            kv[tag + ".max_abs_u"] = m.max_abs_u;
            kv[tag + ".u_l2"] = m.u_l2;
        }
    }

    // Case D: Skew-symmetric scheme test
    {
        Config config;
        config.Nx = 64;
        config.Ny = 64;
        config.x_min = 0.0;
        config.x_max = 2.0 * M_PI;
        config.y_min = 0.0;
        config.y_max = 2.0 * M_PI;
        config.nu = 0.01;
        config.dt = 0.0001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.convective_scheme = ConvectiveScheme::SkewSymmetric;  // Skew-symmetric
        config.verbose = false;

        Mesh mesh;
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);

        RANSSolver solver(mesh, config);
        VelocityBC bc;
        bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        VectorField vel_init(mesh);
        const int Ng = mesh.Nghost;
        for (int j = Ng; j < Ng + mesh.Ny; ++j) {
            for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
                double x = mesh.x_min + (i - Ng) * mesh.dx;
                double y = mesh.y(j);
                vel_init.u(i, j) = -std::cos(x) * std::sin(y);
            }
        }
        for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
            for (int i = Ng; i < Ng + mesh.Nx; ++i) {
                double x = mesh.x(i);
                double y = mesh.y_min + (j - Ng) * mesh.dy;
                vel_init.v(i, j) = std::sin(x) * std::cos(y);
            }
        }
        solver.initialize(vel_init);

        for (int step = 0; step < 10; ++step) {
            solver.step();
        }

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        const auto m = compute_metrics(mesh, solver.velocity(), solver.pressure());
        kv["skew.max_abs_u"] = m.max_abs_u;
        kv["skew.max_abs_v"] = m.max_abs_v;
        kv["skew.u_l2"] = m.u_l2;
        kv["skew.v_l2"] = m.v_l2;
        kv["skew.p_l2"] = m.p_l2;
    }

    return kv;
}


int main(int argc, char** argv) {
    // Two-build dump/compare mode:
    // - CPU-only build: --dump-prefix <prefix> writes a compact reference file
    // - GPU-offload build: --compare-prefix <prefix> recomputes on GPU and compares
    std::string dump_prefix;
    std::string compare_prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dump-prefix" && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "--compare-prefix" && i + 1 < argc) compare_prefix = argv[++i];
    }

    if (!dump_prefix.empty() && !compare_prefix.empty()) {
        std::cerr << "ERROR: choose only one of --dump-prefix or --compare-prefix\n";
        return 1;
    }

    if (!dump_prefix.empty()) {
        const auto kv = run_all_cases_and_collect_metrics();
        write_kv_file(dump_prefix + "_solver_cpu_gpu_metrics.dat", kv);
        std::cout << "[SUCCESS] Wrote CPU reference: " << dump_prefix << "_solver_cpu_gpu_metrics.dat\n";
        return 0;
    }

    if (!compare_prefix.empty()) {
#ifndef USE_GPU_OFFLOAD
        std::cerr << "ERROR: compare mode requires USE_GPU_OFFLOAD=ON build\n";
        return 1;
#else
        // Require real GPU offload (no silent host execution)
        const int num_devices = omp_get_num_devices();
        if (num_devices == 0) {
            std::cerr << "ERROR: USE_GPU_OFFLOAD enabled but no GPU devices found.\n";
            return 1;
        }
        int on_device = 0;
        #pragma omp target map(tofrom: on_device)
        {
            on_device = !omp_is_initial_device();
        }
        if (!on_device) {
            std::cerr << "ERROR: USE_GPU_OFFLOAD enabled but target region ran on host.\n";
            return 1;
        }

        const auto ref = read_kv_file(compare_prefix + "_solver_cpu_gpu_metrics.dat");
        const auto got = run_all_cases_and_collect_metrics();
        // End-to-end solver runs can differ across CPU vs GPU due to
        // reduction ordering, floating-point contraction/FMA differences, and
        // amplified sensitivity in iterative/projection steps.
        // Keep this tight enough to catch regressions, but allow small drift.
        compare_kv(ref, got, /*abs*/1e-3, /*rel*/5e-3);

        std::cout << "[SUCCESS] GPU metrics match CPU reference within tolerance\n";
        return 0;
#endif
    }

    // No legacy single-binary mode: require explicit dump/compare usage.
    std::cout << "Usage:\n"
              << "  CPU-only build:    ./test_solver_cpu_gpu --dump-prefix <name>\n"
              << "  GPU-offload build: ./test_solver_cpu_gpu --compare-prefix <name>\n"
              << "\n"
              << "Legacy single-binary CPU/GPU tests have been removed.\n"
              << "Use the two-build dump/compare workflow for CPU vs GPU validation.\n";
    return 0;
}








