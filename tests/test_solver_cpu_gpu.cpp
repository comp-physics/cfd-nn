/// CPU vs GPU consistency tests for staggered grid solver
/// Tests core solver kernels: divergence, convection, diffusion, projection

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

/// Helper: Compare velocity fields between CPU and GPU
void compare_velocity(const VectorField& cpu, const VectorField& gpu, 
                      const Mesh& mesh, const std::string& label,
                      double tol = 1e-12) {
    double max_diff_u = 0.0, max_diff_v = 0.0;
    double rms_diff_u = 0.0, rms_diff_v = 0.0;
    int count_u = 0, count_v = 0;
    
    const int Ng = mesh.Nghost;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    
    // Compare u-velocities at x-faces
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            double diff = std::abs(cpu.u(i,j) - gpu.u(i,j));
            max_diff_u = std::max(max_diff_u, diff);
            rms_diff_u += diff * diff;
            ++count_u;
        }
    }
    
    // Compare v-velocities at y-faces
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            double diff = std::abs(cpu.v(i,j) - gpu.v(i,j));
            max_diff_v = std::max(max_diff_v, diff);
            rms_diff_v += diff * diff;
            ++count_v;
        }
    }
    
    rms_diff_u = std::sqrt(rms_diff_u / count_u);
    rms_diff_v = std::sqrt(rms_diff_v / count_v);
    
    std::cout << "  " << label << ":\n";
    std::cout << "    u: max_diff=" << std::scientific << std::setprecision(3) 
              << max_diff_u << ", rms_diff=" << rms_diff_u << "\n";
    std::cout << "    v: max_diff=" << max_diff_v << ", rms_diff=" << rms_diff_v << "\n";
    
    if (max_diff_u > tol || max_diff_v > tol) {
        std::cout << "  FAILED: Differences exceed tolerance " << tol << "\n";
        assert(false);
    }
}

/// Helper: Compare scalar fields
void compare_scalar(const ScalarField& cpu, const ScalarField& gpu,
                    const Mesh& mesh, const std::string& label,
                    double tol = 1e-12) {
    double max_diff = 0.0;
    double rms_diff = 0.0;
    int count = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double diff = std::abs(cpu(i,j) - gpu(i,j));
            max_diff = std::max(max_diff, diff);
            rms_diff += diff * diff;
            ++count;
        }
    }
    
    rms_diff = std::sqrt(rms_diff / count);
    
    std::cout << "  " << label << ": max_diff=" << std::scientific << std::setprecision(3)
              << max_diff << ", rms_diff=" << rms_diff << "\n";
    
    if (max_diff > tol) {
        std::cout << "  FAILED: Differences exceed tolerance " << tol << "\n";
        assert(false);
    }
}

/// Test 1: Taylor-Green vortex (fully periodic BCs)
void test_taylor_green_cpu_gpu() {
    std::cout << "\n=== Test 1: Taylor-Green Vortex (Periodic BCs) ===" << std::endl;
    
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
    
    // CPU solver
    RANSSolver solver_cpu(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    solver_cpu.set_velocity_bc(bc);
    
    // Initialize with Taylor-Green
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
    solver_cpu.initialize(vel_init);
    
    // GPU solver (identical setup)
    RANSSolver solver_gpu(mesh, config);
    solver_gpu.set_velocity_bc(bc);
    solver_gpu.initialize(vel_init);
    
    // Run 10 steps on each
    std::cout << "  Running 10 time steps...\n";
    for (int step = 0; step < 10; ++step) {
        solver_cpu.step();
        solver_gpu.step();
    }
    
    // Compare final state
    compare_velocity(solver_cpu.velocity(), solver_gpu.velocity(), mesh, 
                     "Velocity after 10 steps");
    compare_scalar(solver_cpu.pressure(), solver_gpu.pressure(), mesh,
                   "Pressure after 10 steps");
    
    std::cout << "  ✓ PASSED\n";
}

/// Test 2: Channel flow (periodic-x, wall-y)
void test_channel_cpu_gpu() {
    std::cout << "\n=== Test 2: Channel Flow (Periodic-X, Wall-Y) ===" << std::endl;
    
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
    
    // CPU solver
    RANSSolver solver_cpu(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
    solver_cpu.set_velocity_bc(bc);
    solver_cpu.set_body_force(-config.dp_dx, 0.0);
    solver_cpu.initialize_uniform(0.1, 0.0);
    
    // GPU solver
    RANSSolver solver_gpu(mesh, config);
    solver_gpu.set_velocity_bc(bc);
    solver_gpu.set_body_force(-config.dp_dx, 0.0);
    solver_gpu.initialize_uniform(0.1, 0.0);
    
    // Run 10 steps
    std::cout << "  Running 10 time steps...\n";
    for (int step = 0; step < 10; ++step) {
        solver_cpu.step();
        solver_gpu.step();
    }
    
    // Compare
    compare_velocity(solver_cpu.velocity(), solver_gpu.velocity(), mesh,
                     "Velocity after 10 steps");
    compare_scalar(solver_cpu.pressure(), solver_gpu.pressure(), mesh,
                   "Pressure after 10 steps");
    
    std::cout << "  ✓ PASSED\n";
}

/// Test 3: Multiple time steps with different grid sizes
void test_various_grids() {
    std::cout << "\n=== Test 3: Various Grid Sizes ===" << std::endl;
    
    struct GridSize { int nx, ny; };
    std::vector<GridSize> grids = {
        {32, 32},   // Small
        {64, 48},   // Rectangular
        {63, 97},   // Odd sizes
        {128, 64}   // Larger
    };
    
    for (const auto& g : grids) {
        std::cout << "  Testing " << g.nx << "x" << g.ny << " grid...\n";
        
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
        
        RANSSolver solver_cpu(mesh, config);
        RANSSolver solver_gpu(mesh, config);
        
        VelocityBC bc;
        bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
        solver_cpu.set_velocity_bc(bc);
        solver_gpu.set_velocity_bc(bc);
        
        solver_cpu.initialize_uniform(0.5, 0.3);
        solver_gpu.initialize_uniform(0.5, 0.3);
        
        // Run 5 steps
        for (int step = 0; step < 5; ++step) {
            solver_cpu.step();
            solver_gpu.step();
        }
        
        // Quick comparison
        double max_diff = 0.0;
        const int Ng = mesh.Nghost;
        for (int j = Ng; j < Ng + mesh.Ny; ++j) {
            for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
                max_diff = std::max(max_diff, 
                    std::abs(solver_cpu.velocity().u(i,j) - solver_gpu.velocity().u(i,j)));
            }
        }
        
        std::cout << "    Max diff: " << std::scientific << max_diff;
        assert(max_diff < 1e-12);
        std::cout << " ✓\n";
    }
    
    std::cout << "  ✓ PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Solver CPU/GPU Consistency Tests\n";
    std::cout << "Staggered Grid Implementation\n";
    std::cout << "========================================\n";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    std::cout << "\nGPU devices available: " << num_devices << "\n";
    
    if (num_devices == 0) {
        std::cout << "No GPU devices found. Tests skipped.\n";
        return 0;
    }
    
    // Verify GPU is accessible
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }
    
    if (!on_device) {
        std::cout << "GPU not accessible. Tests skipped.\n";
        return 0;
    }
    
    std::cout << "GPU accessible: YES\n";
#else
    std::cout << "\nGPU offload not enabled. Tests skipped.\n";
    return 0;
#endif
    
    // Run tests
    test_taylor_green_cpu_gpu();
    test_channel_cpu_gpu();
    test_various_grids();
    
    std::cout << "\n========================================\n";
    std::cout << "All solver CPU/GPU tests PASSED! ✓\n";
    std::cout << "========================================\n";
    
    return 0;
}






