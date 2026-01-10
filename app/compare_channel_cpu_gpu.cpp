/// (Deprecated) Single-binary CPU vs GPU lockstep comparison
/// Historically attempted to run "CPU-forced" vs "GPU" turbulence paths in one binary.
/// The project now enforces a simpler model:
///   - CPU-only build (USE_GPU_OFFLOAD=OFF) runs on CPU
///   - GPU-offload build (USE_GPU_OFFLOAD=ON) runs on GPU
/// Cross-platform consistency should be validated by comparing outputs between the two builds,
/// e.g. via `.github/scripts/compare_cpu_gpu_builds.sh`.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace nncfd;

struct FieldDiff {
    int i, j;
    double cpu_val, gpu_val, diff;
};

void compare_scalar_field(const std::string& name, const Mesh& mesh,
                          const ScalarField& cpu, const ScalarField& gpu,
                          double tol_linf, double tol_l2) {
    std::cout << "\n=== Comparing " << name << " ===" << std::endl;
    
    double max_diff = 0.0;
    double sum_sq = 0.0;
    int max_i = -1, max_j = -1;
    int n = 0;
    
    std::vector<FieldDiff> diffs;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double c = cpu(i, j);
            double g = gpu(i, j);
            double d = std::abs(c - g);
            
            sum_sq += d * d;
            n++;
            
            if (d > max_diff) {
                max_diff = d;
                max_i = i;
                max_j = j;
            }
            
            if (d > 1e-12) {  // Store significant differences
                diffs.push_back({i, j, c, g, d});
            }
        }
    }
    
    double l2 = std::sqrt(sum_sq / n);
    
    std::cout << "  L∞ diff: " << std::scientific << std::setprecision(6) << max_diff;
    if (max_diff > tol_linf) {
        std::cout << " [FAIL] EXCEEDS TOLERANCE (" << tol_linf << ")";
    } else {
        std::cout << " [OK]";
    }
    std::cout << std::endl;
    
    std::cout << "  L2 diff: " << l2;
    if (l2 > tol_l2) {
        std::cout << " [FAIL] EXCEEDS TOLERANCE (" << tol_l2 << ")";
    } else {
        std::cout << " [OK]";
    }
    std::cout << std::endl;
    
    if (max_diff > 0) {
        std::cout << "  Max diff at (" << max_i << ", " << max_j << "): "
                  << "CPU=" << std::fixed << std::setprecision(12) << cpu(max_i, max_j)
                  << ", GPU=" << gpu(max_i, max_j) << std::endl;
    }
    
    // Show top 10 largest differences
    if (!diffs.empty()) {
        std::sort(diffs.begin(), diffs.end(),
                 [](const FieldDiff& a, const FieldDiff& b) { return a.diff > b.diff; });
        
        int show_count = std::min(10, (int)diffs.size());
        if (show_count > 0) {
            std::cout << "\n  Top " << show_count << " largest differences:\n";
            std::cout << std::fixed;
            for (int k = 0; k < show_count; ++k) {
                const auto& fd = diffs[k];
                std::cout << "    " << std::setw(2) << (k+1) << ". "
                          << "(" << std::setw(3) << fd.i << "," << std::setw(3) << fd.j << ") "
                          << "CPU=" << std::setprecision(8) << std::setw(15) << fd.cpu_val << " "
                          << "GPU=" << std::setw(15) << fd.gpu_val << " "
                          << "diff=" << std::scientific << std::setprecision(3) << fd.diff
                          << std::endl;
            }
        }
    }
}

void compare_vector_field(const std::string& name, const Mesh& mesh,
                          const VectorField& cpu, const VectorField& gpu,
                          double tol_linf, double tol_l2) {
    std::cout << "\n=== Comparing " << name << " ===" << std::endl;
    
    double max_u_diff = 0.0, max_v_diff = 0.0;
    double sum_sq_u = 0.0, sum_sq_v = 0.0;
    int max_u_i = -1, max_u_j = -1;
    int max_v_i = -1, max_v_j = -1;
    int n = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double du = std::abs(cpu.u(i, j) - gpu.u(i, j));
            double dv = std::abs(cpu.v(i, j) - gpu.v(i, j));
            
            sum_sq_u += du * du;
            sum_sq_v += dv * dv;
            n++;
            
            if (du > max_u_diff) {
                max_u_diff = du;
                max_u_i = i;
                max_u_j = j;
            }
            if (dv > max_v_diff) {
                max_v_diff = dv;
                max_v_i = i;
                max_v_j = j;
            }
        }
    }
    
    double l2_u = std::sqrt(sum_sq_u / n);
    double l2_v = std::sqrt(sum_sq_v / n);
    
    std::cout << "  u component:\n";
    std::cout << "    L∞: " << std::scientific << std::setprecision(6) << max_u_diff;
    if (max_u_diff > tol_linf) std::cout << " [FAIL]";
    else std::cout << " [OK]";
    std::cout << "  L2: " << l2_u;
    if (l2_u > tol_l2) std::cout << " [FAIL]";
    else std::cout << " [OK]";
    std::cout << std::endl;
    
    std::cout << "  v component:\n";
    std::cout << "    L∞: " << max_v_diff;
    if (max_v_diff > tol_linf) std::cout << " [FAIL]";
    else std::cout << " [OK]";
    std::cout << "  L2: " << l2_v;
    if (l2_v > tol_l2) std::cout << " [FAIL]";
    else std::cout << " [OK]";
    std::cout << std::endl;
    
    if (max_u_diff > 0) {
        std::cout << "  Max u diff at (" << max_u_i << ", " << max_u_j << "): "
                  << "CPU=" << std::fixed << std::setprecision(8) << cpu.u(max_u_i, max_u_j)
                  << ", GPU=" << gpu.u(max_u_i, max_u_j) << std::endl;
    }
    if (max_v_diff > 0) {
        std::cout << "  Max v diff at (" << max_v_i << ", " << max_v_j << "): "
                  << "CPU=" << cpu.v(max_v_i, max_v_j)
                  << ", GPU=" << gpu.v(max_v_i, max_v_j) << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "Lockstep CPU vs GPU Comparison\n";
    std::cout << "========================================\n\n";
    
    // Parse configuration
    Config config;
    config.Nx = 64;
    config.Ny = 128;
    config.x_min = 0.0;
    config.x_max = 2.0;
    config.y_min = 0.0;
    config.y_max = 1.0;
    config.nu = 0.001;
    config.dp_dx = -0.0001;
    config.dt = 0.001;
    config.max_steps = 200;
    config.tol = 1e-8;
    config.turb_model = TurbulenceModelType::Baseline;
    config.verbose = false;
    
    config.parse_args(argc, argv);
    
    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << config.Nx << "x" << config.Ny << "\n";
    std::cout << "  nu: " << config.nu << "\n";
    std::cout << "  dp/dx: " << config.dp_dx << "\n";
    std::cout << "  max_steps: " << config.max_steps << "\n";
    std::cout << "  Model: ";
    switch (config.turb_model) {
        case TurbulenceModelType::None: std::cout << "none\n"; break;
        case TurbulenceModelType::Baseline: std::cout << "baseline\n"; break;
        default: std::cout << "other\n"; break;
    }
    std::cout << "\n";
    
    // Create mesh
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                     config.x_min, config.x_max,
                     config.y_min, config.y_max);
    
    double H = (config.y_max - config.y_min) / 2.0;
    
    std::cout << "NOTE: Single-binary CPU-forced turbulence is no longer supported.\n"
              << "      This tool now runs the same configuration twice and compares fields.\n"
              << "      For true CPU-vs-GPU validation, compare two separate builds.\n\n";
    
    // ========== Run A ==========
    std::cout << "=== Running case A ===" << std::endl;
    
    RANSSolver solver_cpu(mesh, config);
    {
        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver_cpu.set_velocity_bc(bc);
        
        solver_cpu.set_body_force(-config.dp_dx, 0.0);
        
        if (config.turb_model != TurbulenceModelType::None) {
            auto turb_model = create_turbulence_model(config.turb_model,
                                                      config.nn_weights_path,
                                                      config.nn_scaling_path);
            if (turb_model) {
                turb_model->set_nu(config.nu);
                if (auto* ml = dynamic_cast<MixingLengthModel*>(turb_model.get())) {
                    ml->set_delta(H);
                }
                solver_cpu.set_turbulence_model(std::move(turb_model));
            }
        }
        
        solver_cpu.initialize_uniform(0.1, 0.0);
        
        auto [residual, iterations] = solver_cpu.solve_steady();
        
        std::cout << "  Final residual: " << std::scientific << residual << "\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";
        std::cout << "  Bulk velocity: " << std::fixed << solver_cpu.bulk_velocity() << "\n\n";
    }
    
    // ========== Run B ==========
    std::cout << "=== Running case B ===" << std::endl;
    
    RANSSolver solver_gpu(mesh, config);
    {
        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver_gpu.set_velocity_bc(bc);
        
        solver_gpu.set_body_force(-config.dp_dx, 0.0);
        
        if (config.turb_model != TurbulenceModelType::None) {
            auto turb_model = create_turbulence_model(config.turb_model,
                                                      config.nn_weights_path,
                                                      config.nn_scaling_path);
            if (turb_model) {
                turb_model->set_nu(config.nu);
                if (auto* ml = dynamic_cast<MixingLengthModel*>(turb_model.get())) {
                    ml->set_delta(H);
                }
                solver_gpu.set_turbulence_model(std::move(turb_model));
            }
        }
        
        solver_gpu.initialize_uniform(0.1, 0.0);
        
        auto [residual, iterations] = solver_gpu.solve_steady();
        
        std::cout << "  Final residual: " << std::scientific << residual << "\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";
        std::cout << "  Bulk velocity: " << std::fixed << solver_gpu.bulk_velocity() << "\n\n";
    }
    
    // Compare results
    std::cout << "========================================\n";
    std::cout << "Field Comparison\n";
    std::cout << "========================================\n";
    
    const double tol_linf = 1e-7;
    const double tol_l2 = 1e-8;
    
    compare_vector_field("Velocity", mesh,
                        solver_cpu.velocity(), solver_gpu.velocity(),
                        tol_linf, tol_l2);
    
    compare_scalar_field("Pressure", mesh,
                        solver_cpu.pressure(), solver_gpu.pressure(),
                        tol_linf, tol_l2);
    
    if (config.turb_model != TurbulenceModelType::None) {
        compare_scalar_field("Eddy viscosity (nu_t)", mesh,
                            solver_cpu.nu_t(), solver_gpu.nu_t(),
                            1e-10, 1e-12);
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Comparison complete\n";
    std::cout << "========================================\n";
    
    return 0;
}

