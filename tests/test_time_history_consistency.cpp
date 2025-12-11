/// Time-history consistency test: CPU vs GPU over multiple time steps
/// Verifies no drift accumulates over time

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "turbulence_baseline.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

struct TimeSnapshot {
    double kinetic_energy;
    double mass_flux;
    double max_u;
    double max_v;
    double avg_nu_t;
};

TimeSnapshot compute_diagnostics(const Mesh& mesh, const VectorField& vel, const ScalarField& nu_t) {
    TimeSnapshot snap;
    snap.kinetic_energy = 0.0;
    snap.mass_flux = 0.0;
    snap.max_u = 0.0;
    snap.max_v = 0.0;
    double sum_nu_t = 0.0;
    int count = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = vel.u(i, j);
            double v = vel.v(i, j);
            
            snap.kinetic_energy += 0.5 * (u*u + v*v);
            snap.mass_flux += u;
            snap.max_u = std::max(snap.max_u, std::abs(u));
            snap.max_v = std::max(snap.max_v, std::abs(v));
            sum_nu_t += nu_t(i, j);
            ++count;
        }
    }
    
    snap.kinetic_energy /= count;
    snap.mass_flux /= count;
    snap.avg_nu_t = sum_nu_t / count;
    
    return snap;
}

void compare_snapshots(const TimeSnapshot& cpu, const TimeSnapshot& gpu, int step, double& max_ke_diff, double& max_flux_diff) {
    double ke_diff = std::abs(cpu.kinetic_energy - gpu.kinetic_energy);
    double flux_diff = std::abs(cpu.mass_flux - gpu.mass_flux);
    double u_diff = std::abs(cpu.max_u - gpu.max_u);
    double nut_diff = std::abs(cpu.avg_nu_t - gpu.avg_nu_t);
    
    max_ke_diff = std::max(max_ke_diff, ke_diff);
    max_flux_diff = std::max(max_flux_diff, flux_diff);
    
    std::cout << "  Step " << std::setw(4) << step << ": "
              << "KE_diff=" << std::scientific << std::setprecision(3) << ke_diff << ", "
              << "flux_diff=" << flux_diff << ", "
              << "u_diff=" << u_diff << ", "
              << "nut_diff=" << nut_diff << "\n";
}

void test_time_history() {
    std::cout << "\n=== Time-History Consistency Test ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
    return;
#endif
    
    // Small grid for speed
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, 0.0, 1.0, 1);
    
    Config config;
    config.nu = 0.001;
    config.dp_dx = -0.0001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 50;
    config.tol = 1e-8;
    config.turb_model = TurbulenceModelType::Baseline;
    config.verbose = false;
    
    // Create CPU solver
    RANSSolver solver_cpu(mesh, config);
    auto turb_cpu = std::make_unique<MixingLengthModel>();
    turb_cpu->set_nu(config.nu);
    turb_cpu->set_delta(0.5);
    solver_cpu.set_turbulence_model(std::move(turb_cpu));
    solver_cpu.set_body_force(-config.dp_dx, 0.0);
    solver_cpu.initialize_uniform(0.1, 0.0);
    
    // Create GPU solver (same IC)
    RANSSolver solver_gpu(mesh, config);
    auto turb_gpu = std::make_unique<MixingLengthModel>();
    turb_gpu->set_nu(config.nu);
    turb_gpu->set_delta(0.5);
    solver_gpu.set_turbulence_model(std::move(turb_gpu));
    solver_gpu.set_body_force(-config.dp_dx, 0.0);
    solver_gpu.initialize_uniform(0.1, 0.0);
    
    // Time-stepping
    const int num_steps = 50;
    const int snapshot_interval = 10;
    
    std::cout << "\nRunning " << num_steps << " time steps...\n";
    std::cout << std::fixed;
    
    double max_ke_diff = 0.0;
    double max_flux_diff = 0.0;
    
    for (int step = 1; step <= num_steps; ++step) {
        // Advance both
        solver_cpu.step();
        solver_gpu.step();
        
        // Compare at intervals
        if (step % snapshot_interval == 0) {
            // Get turbulent viscosity fields
            const ScalarField& nu_t_cpu = solver_cpu.nu_t();
            const ScalarField& nu_t_gpu = solver_gpu.nu_t();
            
            auto snap_cpu = compute_diagnostics(mesh, solver_cpu.velocity(), nu_t_cpu);
            auto snap_gpu = compute_diagnostics(mesh, solver_gpu.velocity(), nu_t_gpu);
            
            compare_snapshots(snap_cpu, snap_gpu, step, max_ke_diff, max_flux_diff);
        }
    }
    
    // Final comparison
    std::cout << "\nFinal field comparison...\n";
    const VectorField& vel_cpu = solver_cpu.velocity();
    const VectorField& vel_gpu = solver_gpu.velocity();
    
    double max_u_diff = 0.0, max_v_diff = 0.0;
    double rms_u = 0.0, rms_v = 0.0;
    int n = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double du = std::abs(vel_cpu.u(i, j) - vel_gpu.u(i, j));
            double dv = std::abs(vel_cpu.v(i, j) - vel_gpu.v(i, j));
            
            max_u_diff = std::max(max_u_diff, du);
            max_v_diff = std::max(max_v_diff, dv);
            rms_u += du*du;
            rms_v += dv*dv;
            ++n;
        }
    }
    
    rms_u = std::sqrt(rms_u / n);
    rms_v = std::sqrt(rms_v / n);
    
    std::cout << std::scientific;
    std::cout << "  Max u_diff: " << max_u_diff << "\n";
    std::cout << "  Max v_diff: " << max_v_diff << "\n";
    std::cout << "  RMS u_diff: " << rms_u << "\n";
    std::cout << "  RMS v_diff: " << rms_v << "\n";
    std::cout << "  Max KE_diff over time: " << max_ke_diff << "\n";
    std::cout << "  Max flux_diff over time: " << max_flux_diff << "\n";
    
    // Tolerances
    const double tol_field = 1e-7;
    const double tol_scalar = 1e-8;
    
    bool passed = true;
    if (max_u_diff > tol_field || max_v_diff > tol_field) {
        std::cout << "\n✗ FAILED: Field differences exceed tolerance (" << tol_field << ")\n";
        passed = false;
    }
    
    if (max_ke_diff > tol_scalar || max_flux_diff > tol_scalar) {
        std::cout << "\n✗ FAILED: Scalar differences exceed tolerance (" << tol_scalar << ")\n";
        passed = false;
    }
    
    if (passed) {
        std::cout << "\n✓ PASSED: CPU and GPU remain consistent over " << num_steps << " time steps\n";
    } else {
        assert(false);
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Time-History Consistency Test\n";
    std::cout << "========================================\n";
    
#ifdef USE_GPU_OFFLOAD
    std::cout << "\nGPU Configuration:\n";
    int num_devices = omp_get_num_devices();
    std::cout << "  GPU devices: " << num_devices << "\n";
#else
    std::cout << "\nGPU offload: NOT ENABLED\n";
#endif
    
    test_time_history();
    
    std::cout << "\n========================================\n";
    std::cout << "Test complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}










