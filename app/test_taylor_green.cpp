/// Taylor-Green Vortex Test
/// 
/// Exact solution for 2D incompressible Navier-Stokes:
///   u(x,y,t) = -cos(x) sin(y) exp(-2*nu*t)
///   v(x,y,t) =  sin(x) cos(y) exp(-2*nu*t)
///   p(x,y,t) = -0.25 * [cos(2x) + cos(2y)] * exp(-4*nu*t)
///
/// Domain: [0, 2π] x [0, 2π] with periodic BCs in both directions
///
/// Used for verification of spatial and temporal accuracy

#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace nncfd;

/// Exact Taylor-Green vortex solution
struct TaylorGreenExact {
    double nu;
    double t;
    
    TaylorGreenExact(double nu_, double t_) : nu(nu_), t(t_) {}
    
    double u(double x, double y) const {
        return -std::cos(x) * std::sin(y) * std::exp(-2.0 * nu * t);
    }
    
    double v(double x, double y) const {
        return std::sin(x) * std::cos(y) * std::exp(-2.0 * nu * t);
    }
    
    double p(double x, double y) const {
        return -0.25 * (std::cos(2.0*x) + std::cos(2.0*y)) * std::exp(-4.0 * nu * t);
    }
};

/// Compute L2 and Linf errors against exact solution (staggered grid)
void compute_errors(const Mesh& mesh, const VectorField& velocity, 
                    const ScalarField& pressure, const TaylorGreenExact& exact,
                    double& L2_u, double& L2_v, double& Linf_u, double& Linf_v) {
    
    double sum_error_u = 0.0;
    double sum_error_v = 0.0;
    double max_error_u = 0.0;
    double max_error_v = 0.0;
    int count_u = 0;
    int count_v = 0;
    
    const int Ng = mesh.Nghost;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    
    // Error for u-velocity at x-faces
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            // u is at x-face location (x_{i-1/2}, y_j)
            double x_face = mesh.x_min + (i - Ng) * mesh.dx;
            double y_center = mesh.y(j);
            
            double u_exact = exact.u(x_face, y_center);
            double u_num = velocity.u(i, j);
            
            double err_u = u_num - u_exact;
            sum_error_u += err_u * err_u;
            max_error_u = std::max(max_error_u, std::abs(err_u));
            ++count_u;
        }
    }
    
    // Error for v-velocity at y-faces
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            // v is at y-face location (x_i, y_{j-1/2})
            double x_center = mesh.x(i);
            double y_face = mesh.y_min + (j - Ng) * mesh.dy;
            
            double v_exact = exact.v(x_center, y_face);
            double v_num = velocity.v(i, j);
            
            double err_v = v_num - v_exact;
            sum_error_v += err_v * err_v;
            max_error_v = std::max(max_error_v, std::abs(err_v));
            ++count_v;
        }
    }
    
    L2_u = std::sqrt(sum_error_u / count_u);
    L2_v = std::sqrt(sum_error_v / count_v);
    Linf_u = max_error_u;
    Linf_v = max_error_v;
}

/// Compute kinetic energy and max divergence (staggered grid)
void compute_ke_and_div(const Mesh& mesh, const VectorField& velocity,
                        double& KE, double& max_div) {
    KE = 0.0;
    max_div = 0.0;
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double cell_area = dx * dy;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // KE computed from cell-centered velocities (interpolated)
            double u_center = velocity.u_center(i, j);
            double v_center = velocity.v_center(i, j);
            KE += 0.5 * (u_center*u_center + v_center*v_center) * cell_area;
            
            // Staggered divergence: div = (u(i+1,j) - u(i,j))/dx + (v(i,j+1) - v(i,j))/dy
            double du_dx = (velocity.u(i+1, j) - velocity.u(i, j)) / dx;
            double dv_dy = (velocity.v(i, j+1) - velocity.v(i, j)) / dy;
            double div = du_dx + dv_dy;
            max_div = std::max(max_div, std::abs(div));
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Taylor-Green Vortex Test ===\n\n";
    
    // Configuration
    Config config;
    
    // Domain: [0, 2π] x [0, 2π]
    config.Nx = 64;
    config.Ny = 64;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = 0.0;
    config.y_max = 2.0 * M_PI;
    
    // Physical parameters
    config.nu = 0.01;
    
    // Time stepping  
    config.dt = 0.0001;  // Very small default for accuracy
    config.adaptive_dt = false;  // Fixed dt for accuracy testing
    config.CFL_max = 0.5;  // Conservative for explicit Euler
    
    // No body force (TG vortex is self-sustaining decay)
    config.dp_dx = 0.0;
    
    // Numerical schemes
    config.convective_scheme = ConvectiveScheme::Central;
    
    // No turbulence model
    config.turb_model = TurbulenceModelType::None;
    
    // Parse command line
    config.parse_args(argc, argv);
    config.print();
    
    // Final time (short time to minimize numerical dissipation effects)
    double T_final = 0.1;  // Run to t=0.1 (short time for accuracy)
    int num_steps = static_cast<int>(T_final / config.dt);
    
    std::cout << "\nTaylor-Green Vortex Configuration:\n";
    std::cout << "  Domain: [0, 2π] × [0, 2π]\n";
    std::cout << "  Grid: " << config.Nx << " × " << config.Ny << "\n";
    std::cout << "  nu = " << config.nu << "\n";
    std::cout << "  dt = " << config.dt << "\n";
    std::cout << "  T_final = " << T_final << "\n";
    std::cout << "  num_steps = " << num_steps << "\n\n";
    
    // Create uniform mesh
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    std::cout << "Mesh: dx = " << mesh.dx << ", dy = " << mesh.dy << "\n\n";
    
    // Create solver
    RANSSolver solver(mesh, config);
    
    // Set boundary conditions: PERIODIC in both directions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    
    // Note: The y-direction Poisson BC is hard-coded to Neumann in RANSSolver,
    // but for the fully periodic Taylor-Green vortex this might cause issues.
    // However, let's test it first - periodic velocity BC may be enough.
    
    // Initialize with Taylor-Green vortex at t=0 (staggered grid)
    TaylorGreenExact exact_t0(config.nu, 0.0);
    
    VectorField tg_init(mesh);
    const int Ng = mesh.Nghost;
    const int Nx = config.Nx;
    const int Ny = config.Ny;
    
    // Initialize u-velocity at x-faces: (x_{i-1/2}, y_j)
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            // x-face location between cells (i-1) and i
            double x_face = mesh.x_min + (i - Ng) * mesh.dx;
            double y_center = mesh.y(j);
            tg_init.u(i, j) = exact_t0.u(x_face, y_center);
        }
    }
    
    // Initialize v-velocity at y-faces: (x_i, y_{j-1/2})
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            double x_center = mesh.x(i);
            // y-face location between cells (j-1) and j
            double y_face = mesh.y_min + (j - Ng) * mesh.dy;
            tg_init.v(i, j) = exact_t0.v(x_center, y_face);
        }
    }
    
    // Use solver initialize to apply BCs and sync to GPU (if enabled)
    solver.initialize(tg_init);
    
    // Initial error
    double L2_u, L2_v, Linf_u, Linf_v;
    compute_errors(mesh, solver.velocity(), solver.pressure(), exact_t0,
                   L2_u, L2_v, Linf_u, Linf_v);
    
    std::cout << "Initial errors (should be ~0):\n";
    std::cout << "  L2(u)   = " << std::scientific << L2_u << "\n";
    std::cout << "  L2(v)   = " << L2_v << "\n";
    std::cout << "  Linf(u) = " << Linf_u << "\n";
    std::cout << "  Linf(v) = " << Linf_v << "\n\n";
    
    // Time integration
    std::cout << "Time stepping to T = " << T_final << "...\n";
    std::cout << std::setw(10) << "Step"
              << std::setw(15) << "Time"
              << std::setw(15) << "L2(u)"
              << std::setw(15) << "L2(v)"
              << "\n";
    
    double t = 0.0;
    int output_freq = std::max(1, num_steps / 10);  // 10 outputs
    
    double KE_num, max_div;
    compute_ke_and_div(mesh, solver.velocity(), KE_num, max_div);
    double KE_exact = 0.0;
    {
        // KE_exact computed from cell-centered velocities (same as numerical)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                double u = exact_t0.u(x, y);
                double v = exact_t0.v(x, y);
                KE_exact += 0.5 * (u*u + v*v) * (mesh.dx * mesh.dy);
            }
        }
    }
    std::cout << "Initial KE_num = " << KE_num << ", KE_exact = " << KE_exact
              << ", max_div = " << max_div << "\n\n";
    
    for (int step = 0; step < num_steps; ++step) {
        solver.step();
        t += config.dt;
        
        if ((step + 1) % output_freq == 0 || step == num_steps - 1) {
            TaylorGreenExact exact_t(config.nu, t);
            compute_errors(mesh, solver.velocity(), solver.pressure(), exact_t,
                          L2_u, L2_v, Linf_u, Linf_v);
            compute_ke_and_div(mesh, solver.velocity(), KE_num, max_div);
            double decay = std::exp(-4.0 * config.nu * t);  // KE decays with 4ν
            double KE_exact_t = KE_exact * decay;
            
            std::cout << std::setw(10) << step + 1
                      << std::setw(15) << std::fixed << std::setprecision(4) << t
                      << std::setw(15) << std::scientific << std::setprecision(3) << L2_u
                      << std::setw(15) << L2_v
                      << std::setw(15) << std::scientific << KE_num
                      << std::setw(15) << KE_exact_t
                      << std::setw(15) << std::scientific << max_div
                      << "\n";
        }
    }
    
    // Final errors
    TaylorGreenExact exact_final(config.nu, T_final);
    compute_errors(mesh, solver.velocity(), solver.pressure(), exact_final,
                   L2_u, L2_v, Linf_u, Linf_v);
    
    std::cout << "\n=== Final Results (T = " << T_final << ") ===\n";
    std::cout << "L2 Errors:\n";
    std::cout << "  u: " << std::scientific << std::setprecision(6) << L2_u << "\n";
    std::cout << "  v: " << L2_v << "\n";
    std::cout << "Linf Errors:\n";
    std::cout << "  u: " << Linf_u << "\n";
    std::cout << "  v: " << Linf_v << "\n";
    
    // Combined error for convergence studies
    double L2_combined = std::sqrt(L2_u*L2_u + L2_v*L2_v);
    std::cout << "\nCombined L2 error: " << L2_combined << "\n";
    
    // Write to file for convergence analysis
    std::string output_path = config.output_dir.empty() ? "." : config.output_dir;
    std::string filename = output_path + "/taylor_green_error.dat";
    std::ofstream file(filename);
    if (file) {
        file << std::scientific << std::setprecision(12);
        file << "# Taylor-Green Vortex Error at T = " << T_final << "\n";
        file << "# Nx Ny dx dy dt L2_u L2_v Linf_u Linf_v L2_combined\n";
        file << config.Nx << " " << config.Ny << " "
             << mesh.dx << " " << mesh.dy << " "
             << config.dt << " "
             << L2_u << " " << L2_v << " "
             << Linf_u << " " << Linf_v << " "
             << L2_combined << "\n";
        file.close();
        std::cout << "\nError data saved to: " << filename << "\n";
    } else {
        std::cerr << "\nWARNING: Could not write error data to: " << filename << "\n";
    }
    
    return 0;
}

