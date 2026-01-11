/// Profile Poisson solver with warm-start and limited V-cycles
/// This app is designed for nsys profiling to analyze:
/// - Poisson solver performance with warm-start (reusing previous solution)
/// - Impact of limiting V-cycles to 3 per solve
/// - Overall solver performance over 50 timesteps

#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include <iostream>
#include <iomanip>
#include <string>

// NVTX support for profiling - try multiple header locations
#ifdef GPU_PROFILE_KERNELS
    // Try nvtx3 first (newer), then nvToolsExt (older)
    #if __has_include(<nvtx3/nvToolsExt.h>)
        #include <nvtx3/nvToolsExt.h>
        #define NVTX_PUSH(name) nvtxRangePushA(name)
        #define NVTX_POP() nvtxRangePop()
    #elif __has_include(<nvToolsExt.h>)
        #include <nvToolsExt.h>
        #define NVTX_PUSH(name) nvtxRangePushA(name)
        #define NVTX_POP() nvtxRangePop()
    #else
        // NVTX headers not found - use no-op markers
        #define NVTX_PUSH(name) do {} while(0)
        #define NVTX_POP() do {} while(0)
    #endif
#else
    #define NVTX_PUSH(name) do {} while(0)
    #define NVTX_POP() do {} while(0)
#endif

using namespace nncfd;

int main(int argc, char** argv) {
    (void)argc;  // Unused
    (void)argv;  // Unused
    std::cout << "========================================\n";
    std::cout << "Poisson Warm-Start Profiling\n";
    std::cout << "========================================\n\n";

    // Configuration for profiling
    Config config;
    config.Nx = 128;
    config.Ny = 64;
    config.x_min = 0.0;
    config.x_max = 4.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    // CRITICAL: Limit Poisson iterations to 3 V-cycles
    config.poisson_max_vcycles = 3;
    config.poisson_tol = 1e-6;  // Relative tolerance
    
    std::cout << "Grid: " << config.Nx << " x " << config.Ny << "\n";
    std::cout << "Timesteps: 50\n";
    std::cout << "Poisson max V-cycles: " << config.poisson_max_vcycles << "\n";
    std::cout << "Warm-start: YES (reuses previous solution)\n\n";

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

    std::cout << "Starting solver (50 timesteps)...\n";
    std::cout << std::setw(8) << "Step" 
              << std::setw(15) << "Max |u|"
              << std::setw(15) << "Max |v|"
              << std::setw(15) << "Residual"
              << "\n";
    std::cout << std::string(53, '-') << "\n";

    // Run 50 timesteps
    // Note: Warm-start is automatic - solver reuses previous pressure_correction_
    // as initial guess for next solve (only zeroed on iter_==0)
    for (int step = 0; step < 50; ++step) {
        // Push NVTX range for entire timestep
        std::string range_name = "timestep_" + std::to_string(step);
        NVTX_PUSH(range_name.c_str());
        
        double residual = solver.step();
        
        NVTX_POP();  // Pop timestep range
        
        if (step % 10 == 0) {
            double max_u = 0.0, max_v = 0.0;
            const auto& vel = solver.velocity();
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    max_u = std::max(max_u, std::abs(vel.u(i, j)));
                }
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    max_v = std::max(max_v, std::abs(vel.v(i, j+1)));
                }
            }
            
            std::cout << std::setw(8) << step
                      << std::setw(15) << std::fixed << std::setprecision(6) << max_u
                      << std::setw(15) << max_v
                      << std::setw(15) << std::scientific << residual
                      << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Profiling complete!\n";
    std::cout << "========================================\n";

    return 0;
}

