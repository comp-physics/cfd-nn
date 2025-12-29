/// 3D Square Duct Flow Solver
/// Solves incompressible Navier-Stokes in a square duct with:
/// - Periodic boundary conditions in x (streamwise)
/// - No-slip walls at y = y_min/max and z = z_min/max
/// - Constant body force (pressure gradient) driving the flow
///
/// Analytical solution (laminar): Series expansion with max velocity at center
/// u_max = (48/π⁴) * (dp/dx) * a² / (2μ) where a is half-width

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

/// Analytical solution for square duct laminar flow (truncated series)
/// u(y,z) = Σ Σ A_mn * sin(mπy/2a) * sin(nπz/2a)
/// Approximation using first few terms
double duct_velocity_analytical(double y, double z, double a, double dp_dx, double nu, int nterms = 5) {
    double sum = 0.0;
    double coeff = 16.0 * a * a / (M_PI * M_PI * M_PI * nu);

    for (int m = 1; m <= nterms; m += 2) {
        for (int n = 1; n <= nterms; n += 2) {
            double denom = m * n * (m * m + n * n);
            double term = std::sin(m * M_PI * (y + a) / (2.0 * a))
                        * std::sin(n * M_PI * (z + a) / (2.0 * a)) / denom;
            sum += term;
        }
    }

    return -dp_dx * coeff * sum;
}

/// Compute bulk velocity (volume-averaged)
double compute_bulk_velocity_3d(const Mesh& mesh, const VectorField& velocity) {
    double sum_u = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum_u += 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
                count++;
            }
        }
    }

    return sum_u / count;
}

/// Compute max velocity
double compute_max_velocity_3d(const Mesh& mesh, const VectorField& velocity) {
    double max_u = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
                max_u = std::max(max_u, u);
            }
        }
    }

    return max_u;
}

/// Write velocity profile at duct center (y-z plane)
void write_duct_profile(const std::string& filename, const Mesh& mesh,
                        const VectorField& velocity, double dp_dx, double nu) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    double a = (mesh.y_max - mesh.y_min) / 2.0;

    file << "# y z u_numerical u_analytical\n";

    // Sample at center x location
    int i = mesh.i_begin() + mesh.Nx / 2;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double z = mesh.z(k);
            double u_num = 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
            double u_ana = duct_velocity_analytical(y, z, a, dp_dx, nu);

            file << y << " " << z << " " << u_num << " " << u_ana << "\n";
        }
        file << "\n";  // Blank line for gnuplot splot
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --Nx N            Grid cells in x (streamwise)\n";
    std::cout << "  --Ny N            Grid cells in y\n";
    std::cout << "  --Nz N            Grid cells in z\n";
    std::cout << "  --Re R            Reynolds number\n";
    std::cout << "  --nu V            Kinematic viscosity\n";
    std::cout << "  --dp_dx D         Pressure gradient\n";
    std::cout << "  --dt T            Time step\n";
    std::cout << "  --max_iter N      Maximum iterations\n";
    std::cout << "  --tol T           Convergence tolerance\n";
    std::cout << "  --model M         Turbulence model (none, baseline, sst, etc.)\n";
    std::cout << "  --output DIR      Output directory\n";
    std::cout << "  --no_write_fields Skip VTK output\n";
    std::cout << "  --adaptive_dt     Enable adaptive time stepping\n";
}

int main(int argc, char** argv) {
    std::cout << "=== 3D Square Duct Flow Solver ===\n\n";

    // 3D mesh parameters (not in Config)
    int Nx = 16, Ny = 32, Nz = 32;
    double x_min = 0.0, x_max = 2.0 * M_PI;
    double y_min = -1.0, y_max = 1.0;
    double z_min = -1.0, z_max = 1.0;

    // Solver configuration
    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.dt = 0.001;
    config.max_iter = 10000;
    config.tol = 1e-8;
    config.output_freq = 500;
    config.verbose = true;
    config.turb_model = TurbulenceModelType::None;
    config.poisson_tol = 1e-8;
    config.poisson_max_iter = 5000;

    std::string output_dir = "output/";
    bool write_fields = true;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--Nx" && i+1 < argc) {
            Nx = std::stoi(argv[++i]);
        } else if (arg == "--Ny" && i+1 < argc) {
            Ny = std::stoi(argv[++i]);
        } else if (arg == "--Nz" && i+1 < argc) {
            Nz = std::stoi(argv[++i]);
        } else if (arg == "--nu" && i+1 < argc) {
            config.nu = std::stod(argv[++i]);
        } else if (arg == "--dp_dx" && i+1 < argc) {
            config.dp_dx = std::stod(argv[++i]);
        } else if (arg == "--dt" && i+1 < argc) {
            config.dt = std::stod(argv[++i]);
        } else if (arg == "--max_iter" && i+1 < argc) {
            config.max_iter = std::stoi(argv[++i]);
        } else if (arg == "--tol" && i+1 < argc) {
            config.tol = std::stod(argv[++i]);
        } else if (arg == "--output" && i+1 < argc) {
            output_dir = argv[++i];
            if (output_dir.back() != '/') output_dir += '/';
        } else if (arg == "--no_write_fields") {
            write_fields = false;
        } else if (arg == "--adaptive_dt") {
            config.adaptive_dt = true;
        } else if (arg == "--model" && i+1 < argc) {
            std::string model = argv[++i];
            if (model == "none") config.turb_model = TurbulenceModelType::None;
            else if (model == "baseline") config.turb_model = TurbulenceModelType::Baseline;
            else if (model == "sst") config.turb_model = TurbulenceModelType::SSTKOmega;
            else if (model == "komega") config.turb_model = TurbulenceModelType::KOmega;
            else std::cerr << "Unknown model: " << model << "\n";
        }
    }

    // Print configuration
    std::cout << "=== Configuration ===\n";
    std::cout << "Mesh: " << Nx << " x " << Ny << " x " << Nz << " (3D)\n";
    std::cout << "Domain: [" << x_min << ", " << x_max << "] x ["
              << y_min << ", " << y_max << "] x ["
              << z_min << ", " << z_max << "]\n";
    std::cout << "nu = " << config.nu << ", dp/dx = " << config.dp_dx << "\n";
    std::cout << "dt = " << config.dt << ", max_iter = " << config.max_iter << "\n";
    std::cout << "=====================\n\n";

    // Compute expected values
    double a = (y_max - y_min) / 2.0;
    double u_center_approx = duct_velocity_analytical(0.0, 0.0, a, config.dp_dx, config.nu);
    std::cout << "Expected centerline velocity (approx): " << u_center_approx << "\n\n";

    // Create 3D mesh
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, x_min, x_max, y_min, y_max, z_min, z_max);

    std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " x " << mesh.Nz << " cells (3D)\n";
    std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << ", dz = " << mesh.dz << "\n\n";

    // Create solver
    RANSSolver solver(mesh, config);

    // Set 3D boundary conditions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;  // Streamwise periodic
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;    // Walls
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;    // Walls
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Set body force (pressure gradient)
    solver.set_body_force(-config.dp_dx, 0.0, 0.0);

    // Set turbulence model if requested
    if (config.turb_model != TurbulenceModelType::None) {
        auto turb_model = create_turbulence_model(config.turb_model, "", "");
        if (turb_model) {
            turb_model->set_nu(config.nu);
            solver.set_turbulence_model(std::move(turb_model));
        }
    }

    // Initialize with small perturbation
    solver.initialize_uniform(0.1 * std::abs(u_center_approx), 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::cout << "=== Running STEADY solve ===\n";
    std::cout << "Convergence tolerance: " << config.tol << "\n\n";

    std::cout << std::setw(10) << "Iter"
              << std::setw(15) << "Residual"
              << std::setw(15) << "Max |u|"
              << std::setw(15) << "Bulk u"
              << "\n";

    ScopedTimer total_timer("Total simulation", false);

    double residual = 1.0;
    int iter = 0;
    for (iter = 1; iter <= config.max_iter && residual > config.tol; ++iter) {
        if (config.adaptive_dt) {
            (void)solver.compute_adaptive_dt();
        }
        residual = solver.step();

        if (config.verbose && (iter % config.output_freq == 0)) {
#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif
            double max_u = compute_max_velocity_3d(mesh, solver.velocity());
            double bulk_u = compute_bulk_velocity_3d(mesh, solver.velocity());

            std::cout << std::setw(10) << iter
                      << std::setw(15) << std::scientific << residual
                      << std::setw(15) << std::fixed << max_u
                      << std::setw(15) << bulk_u
                      << "\n";
        }

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at iteration " << iter << "\n";
            return 1;
        }
    }

    total_timer.stop();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
    std::cout << "Iterations: " << iter - 1 << "\n";
    std::cout << "Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";

    double max_u = compute_max_velocity_3d(mesh, solver.velocity());
    double bulk_u = compute_bulk_velocity_3d(mesh, solver.velocity());

    std::cout << "Max velocity: " << std::fixed << std::setprecision(6) << max_u << "\n";
    std::cout << "Bulk velocity: " << bulk_u << "\n";
    std::cout << "Expected centerline (approx): " << u_center_approx << "\n";

    if (config.turb_model == TurbulenceModelType::None) {
        double error = std::abs(max_u - u_center_approx) / std::abs(u_center_approx);
        std::cout << "Centerline error: " << error * 100.0 << "%\n";

        if (error < 0.05) {
            std::cout << "\n*** VALIDATION PASSED: Error < 5% ***\n";
        }
    }

    // Write output
    if (write_fields) {
        try {
            std::filesystem::create_directories(output_dir);
            solver.write_vtk(output_dir + "duct_final.vtk");
            write_duct_profile(output_dir + "duct_profile.dat", mesh,
                              solver.velocity(), config.dp_dx, config.nu);
            std::cout << "\nWrote output to " << output_dir << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write output: " << e.what() << "\n";
        }
    }

    // Print timing summary
    TimingStats::instance().print_summary();

    return 0;
}
