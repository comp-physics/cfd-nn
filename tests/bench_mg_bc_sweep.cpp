/// @file bench_mg_bc_sweep.cpp
/// @brief Quick BC robustness sweep for MG tuning validation
///
/// Tests nu1=3,nu2=1 vs nu1=2,nu2=2 across Channel (PWP) and Duct (PWW)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "test_utilities.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace nncfd;
using namespace std::chrono;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

enum class BCType { Channel_PWP, Duct_PWW };

struct Result {
    double ms_per_step;
    double div_l2;
    double div_inf;
};

double compute_div_l2(const Mesh& mesh, const VectorField& vel) {
    double sum = 0.0;
    const double idx = 1.0/mesh.dx, idy = 1.0/mesh.dy, idz = 1.0/mesh.dz;
    for (int k = 1; k <= mesh.Nz; ++k)
        for (int j = 1; j <= mesh.Ny; ++j)
            for (int i = 1; i <= mesh.Nx; ++i) {
                double d = (vel.u(i+1,j,k)-vel.u(i,j,k))*idx +
                           (vel.v(i,j+1,k)-vel.v(i,j,k))*idy +
                           (vel.w(i,j,k+1)-vel.w(i,j,k))*idz;
                sum += d*d;
            }
    return std::sqrt(sum / (mesh.Nx * mesh.Ny * mesh.Nz));
}

double compute_div_inf(const Mesh& mesh, const VectorField& vel) {
    double mx = 0.0;
    const double idx = 1.0/mesh.dx, idy = 1.0/mesh.dy, idz = 1.0/mesh.dz;
    for (int k = 1; k <= mesh.Nz; ++k)
        for (int j = 1; j <= mesh.Ny; ++j)
            for (int i = 1; i <= mesh.Nx; ++i) {
                double d = std::abs((vel.u(i+1,j,k)-vel.u(i,j,k))*idx +
                                    (vel.v(i,j+1,k)-vel.v(i,j,k))*idy +
                                    (vel.w(i,j,k+1)-vel.w(i,j,k))*idz);
                mx = std::max(mx, d);
            }
    return mx;
}

Result run_test(int N, int nsteps, BCType bc_type, int nu1, int nu2, int cycles) {
    const double L = 2.0 * M_PI;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = 1e-4;
    config.dt = 1e-4;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.poisson_solver = PoissonSolverType::MG;
    config.verbose = false;
    config.output_freq = 0;
    config.write_fields = false;
    config.turb_guard_enabled = false;
    config.poisson_fixed_cycles = cycles;
    config.poisson_nu1 = nu1;
    config.poisson_nu2 = nu2;
    config.poisson_chebyshev_degree = 4;

    RANSSolver solver(mesh, config);

    // Set BCs based on type
    BCPattern pattern = (bc_type == BCType::Channel_PWP) ? BCPattern::Channel3D : BCPattern::Duct;
    solver.set_velocity_bc(create_velocity_bc(pattern));

    // Initialize Taylor-Green vortex
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

    // Warmup
    solver.step();
    solver.step();

    // Timed run
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nsteps; ++i) {
        solver.step();
    }
    auto end = high_resolution_clock::now();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    Result r;
    r.ms_per_step = duration_cast<microseconds>(end - start).count() / 1000.0 / nsteps;
    r.div_l2 = compute_div_l2(mesh, solver.velocity());
    r.div_inf = compute_div_inf(mesh, solver.velocity());
    return r;
}

int main() {
    const int N = 128;
    const int nsteps = 20;

    std::cout << "==========================================================================\n";
    std::cout << "  MG BC Robustness Sweep: Channel (PWP) vs Duct (PWW) at " << N << "³\n";
    std::cout << "==========================================================================\n\n";

    struct TestCase {
        const char* name;
        int nu1, nu2, cycles;
    };

    TestCase cases[] = {
        {"nu1=3,nu2=1,cyc=8 (NEW)", 3, 1, 8},
        {"nu1=2,nu2=2,cyc=10 (OLD)", 2, 2, 10},
        {"nu1=2,nu2=1,cyc=8", 2, 1, 8},
        {"nu1=2,nu2=2,cyc=8", 2, 2, 8},
    };

    std::cout << std::left << std::setw(24) << "Config"
              << std::setw(12) << "BC"
              << std::right << std::setw(10) << "ms/step"
              << std::setw(12) << "div_L2"
              << std::setw(12) << "div_Linf"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto& tc : cases) {
        for (auto bc : {BCType::Channel_PWP, BCType::Duct_PWW}) {
            const char* bc_name = (bc == BCType::Channel_PWP) ? "PWP" : "PWW";

            std::cout << std::left << std::setw(24) << tc.name
                      << std::setw(12) << bc_name << std::flush;

            Result r = run_test(N, nsteps, bc, tc.nu1, tc.nu2, tc.cycles);

            std::cout << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << r.ms_per_step
                      << std::scientific << std::setprecision(1)
                      << std::setw(12) << r.div_l2
                      << std::setw(12) << r.div_inf
                      << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "==========================================================================\n";
    std::cout << "Quality contract: div_L2 < 1e-5, div_Linf < 1e-3\n";
    std::cout << "==========================================================================\n";

    return 0;
}
