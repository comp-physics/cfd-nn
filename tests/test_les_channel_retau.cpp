/// @file test_les_channel_retau.cpp
/// @brief Physics validation: LES Smagorinsky channel produces nonzero Re_tau
///
/// Channel flow driven by dp/dx=-1 with nu=1/180 targets Re_tau=180.
/// LES run must produce: Re_tau > 20 (turbulence present, not laminar),
/// friction velocity > 0, residual finite.
/// Note: reaching steady Re_tau=180 requires O(10^5) steps; this test
/// verifies the LES pipeline is physically active, not accuracy.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_les.hpp"
#include "turbulence_model.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <memory>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

int main() {
    try {
        const int Nx = 32, Ny = 64;
        const int Nz = 1;
        const double nu = 1.0 / 180.0;
        const double dp_dx = -1.0;
        const double dt = 0.001;
        const int nsteps = 1500;

        std::cout << "LES Channel Re_tau Test\n";
        std::cout << "  Grid: " << Nx << "x" << Ny << "x" << Nz
                  << ", nu=" << nu << ", dp_dx=" << dp_dx
                  << ", dt=" << dt << ", nsteps=" << nsteps << "\n\n";

        // ----------------------------------------------------------------
        // Build mesh: uniform y for this test
        // ----------------------------------------------------------------
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, 0.0, 2.0 * M_PI, -1.0, 1.0);

        // ----------------------------------------------------------------
        // Config
        // ----------------------------------------------------------------
        Config config;
        config.Nx = Nx;
        config.Ny = Ny;
        config.Nz = Nz;
        config.nu = nu;
        config.dp_dx = dp_dx;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::Smagorinsky;
        config.poisson_tol = 1e-6;
        config.poisson_max_vcycles = 20;
        config.verbose = false;

        // ----------------------------------------------------------------
        // Solver setup
        // ----------------------------------------------------------------
        Decomposition decomp(Nz);
        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);
        solver.set_body_force(-dp_dx, 0.0);  // drives flow in +x

        auto turb_model = create_turbulence_model(TurbulenceModelType::Smagorinsky);
        CHECK(turb_model != nullptr, "Smagorinsky factory returned null");
        turb_model->set_nu(nu);
        solver.set_turbulence_model(std::move(turb_model));

        // ----------------------------------------------------------------
        // Initialize: uniform + random perturbation to trigger turbulence
        // ----------------------------------------------------------------
        solver.initialize_uniform(1.0, 0.0);

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j) += dist(rng);
            }
        }
        solver.sync_to_gpu();

        // ----------------------------------------------------------------
        // Run nsteps
        // ----------------------------------------------------------------
        double final_residual = 0.0;
        bool stable = true;
        for (int step = 0; step < nsteps; ++step) {
            double res = solver.step();
            if (!std::isfinite(res)) {
                stable = false;
                std::cerr << "  [ERROR] Blow-up at step " << step << "\n";
                break;
            }
            final_residual = res;
        }

        CHECK(stable, "LES Smagorinsky channel simulation blew up");

        // ----------------------------------------------------------------
        // Diagnostics
        // ----------------------------------------------------------------
        solver.sync_from_gpu();

        double u_tau      = solver.friction_velocity();
        double Re_tau_val = solver.Re_tau();
        double U_bulk     = solver.bulk_velocity();

        std::cout << "  u_tau          = " << std::fixed << std::setprecision(6) << u_tau << "\n";
        std::cout << "  Re_tau         = " << std::fixed << std::setprecision(2) << Re_tau_val << "\n";
        std::cout << "  U_bulk         = " << std::fixed << std::setprecision(6) << U_bulk << "\n";
        std::cout << "  final_residual = " << std::scientific << std::setprecision(3)
                  << final_residual << "\n\n";

        // ----------------------------------------------------------------
        // Assertions
        // ----------------------------------------------------------------
        CHECK(u_tau > 0.0,
              "Friction velocity must be positive");
        CHECK(Re_tau_val > 20.0,
              "Re_tau must be > 20 (flow is not completely laminar with SGS model active)");
        CHECK(Re_tau_val < 600.0,
              "Re_tau must be < 600 (not wildly unphysical)");
        CHECK(std::isfinite(final_residual),
              "Final residual must be finite");
        CHECK(U_bulk > 0.0,
              "Bulk velocity must be positive (flow driven by body force)");

        std::cout << "PASS: LES Smagorinsky channel produces physically reasonable Re_tau="
                  << std::fixed << std::setprecision(1) << Re_tau_val << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
