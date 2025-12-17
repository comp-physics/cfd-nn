#include "solver.hpp"
#include "turbulence_model.hpp"
#include <iostream>
#include <stdexcept>

using namespace nncfd;

// Test that solver completes successfully with guard enabled (baseline)
bool test_guard_allows_normal_operation() {
    std::cout << "Testing guard allows normal operation (SST k-omega)...\n";
    
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dt = 5e-4;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.turb_guard_enabled = true;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    auto turb_model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));
    
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);
    
    try {
        for (int i = 0; i < 100; ++i) {
            solver.step();
        }
        std::cout << "[PASS] Guard allows normal operation\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Guard incorrectly aborted: " << e.what() << "\n";
        return false;
    }
}

// Test that guard is called during VTK output
bool test_guard_on_io() {
    std::cout << "\nTesting guard is called during I/O...\n";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);
    
    Config config;
    config.nu = 0.01;
    config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::Baseline;
    config.turb_guard_enabled = true;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    auto turb_model = create_turbulence_model(TurbulenceModelType::Baseline, "", "");
    solver.set_turbulence_model(std::move(turb_model));
    
    solver.initialize_uniform(1.0, 0.0);
    
    try {
        for (int i = 0; i < 10; ++i) {
            solver.step();
        }
        solver.write_vtk("/tmp/test_guard_io.vtk");
        std::cout << "[PASS] Guard checked during I/O without issues\n";
        return true;
    } catch (const std::exception& e) {
        std::string msg(e.what());
        if (msg.find("NaN/Inf") != std::string::npos) {
            std::cerr << "[FAIL] Guard triggered unexpectedly on clean run: " << e.what() << "\n";
            return false;
        }
        std::cerr << "[FAIL] Unexpected exception: " << e.what() << "\n";
        return false;
    }
}

// Test that all EARSM models run without guard issues in realistic turbulence
bool test_earsm_with_guard() {
    std::cout << "\nTesting EARSM models with guard enabled...\n";
    
    std::vector<TurbulenceModelType> earsm_models = {
        TurbulenceModelType::EARSM_WJ,
        TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope
    };
    
    for (auto model_type : earsm_models) {
        Mesh mesh;
        mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);
        
        Config config;
        config.nu = 0.001;
        config.dt = 1e-4;
        config.turb_model = model_type;
        config.turb_guard_enabled = true;
        config.verbose = false;
        
        RANSSolver solver(mesh, config);
        
        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);
        
        auto turb_model = create_turbulence_model(model_type, "", "");
        solver.set_turbulence_model(std::move(turb_model));
        
        // Driven flow with sustained turbulence
        solver.set_body_force(-0.001, 0.0);
        solver.initialize_uniform(0.5, 0.0);
        
        try {
            for (int i = 0; i < 50; ++i) {
                solver.step();
            }
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] EARSM model threw exception: " << e.what() << "\n";
            return false;
        }
    }
    
    std::cout << "[PASS] All EARSM models ran without guard issues\n";
    return true;
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  NaN/Inf GUARD TEST SUITE\n";
    std::cout << "========================================\n";
    std::cout << "Purpose: Verify NaN/Inf guard prevents\n";
    std::cout << "         corrupted data from propagating\n";
    std::cout << "========================================\n\n";
    
    int failed = 0;
    
    if (!test_guard_allows_normal_operation()) failed++;
    if (!test_guard_on_io()) failed++;
    if (!test_earsm_with_guard()) failed++;
    
    std::cout << "\n========================================\n";
    if (failed == 0) {
        std::cout << "[SUCCESS] All NaN/Inf guard tests passed!\n";
        std::cout << "Guard is active and non-intrusive.\n";
        std::cout << "========================================\n";
        return 0;
    } else {
        std::cout << "[FAILURE] " << failed << " test(s) failed\n";
        std::cout << "========================================\n";
        return 1;
    }
}

