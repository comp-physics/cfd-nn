// Comprehensive GPU diagnostic: verify GPU execution and CPU/GPU comparison
#include "mesh.hpp"
#include "fields.hpp"
#include "turbulence_baseline.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

int main() {
    std::cout << "=== Comprehensive GPU Diagnostic Test ===" << std::endl;
    std::cout << std::fixed << std::setprecision(12);

#ifdef USE_GPU_OFFLOAD
    std::cout << "\nGPU Configuration:" << std::endl;
    int num_devices = omp_get_num_devices();
    std::cout << "  Number of GPU devices: " << num_devices << std::endl;
    std::cout << "  (Note: This is a diagnostic test, not a proof of bit-exact equality)" << std::endl;
    
    if (num_devices > 0) {
        // Test GPU device accessibility
        int on_device = 0;
        #pragma omp target map(tofrom: on_device)
        {
            on_device = !omp_is_initial_device();
        }
        std::cout << "  GPU accessible: " << (on_device ? "YES" : "NO") << std::endl;
        
        if (!on_device) {
            std::cout << "ERROR: GPU not accessible despite devices present" << std::endl;
            return 1;
        }
    } else {
        std::cout << "ERROR: No GPU devices found" << std::endl;
        return 1;
    }
#else
    std::cout << "ERROR: GPU offload not enabled at compile time" << std::endl;
    return 1;
#endif

    // Create a mesh (64x64 triggers GPU path)
    std::cout << "\nCreating 64x64 mesh..." << std::endl;
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, 0.0, 1.0, 1);

    // Create fields
    VectorField velocity(mesh);
    ScalarField nu_t_gpu(mesh);
    ScalarField nu_t_cpu(mesh);
    ScalarField k(mesh);
    ScalarField omega(mesh);
    
    // SANITY CHECK: Verify fields have different memory addresses
    std::cout << "Memory addresses:" << std::endl;
    std::cout << "  nu_t_gpu: " << static_cast<const void*>(nu_t_gpu.data().data()) << std::endl;
    std::cout << "  nu_t_cpu: " << static_cast<const void*>(nu_t_cpu.data().data()) << std::endl;
    if (nu_t_gpu.data().data() == nu_t_cpu.data().data()) {
        std::cout << "ERROR: GPU and CPU fields are aliased!" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Fields have distinct storage" << std::endl;

    // Initialize velocity with parabolic profile
    std::cout << "Initializing velocity field..." << std::endl;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y = mesh.yc[j];
            double u_parab = 4.0 * y * (1.0 - y);
            velocity.u(i, j) = u_parab;
            velocity.v(i, j) = 0.0;
        }
    }

    // Create turbulence model
    MixingLengthModel model_gpu;
    model_gpu.set_nu(1.0 / 10000.0);
    model_gpu.set_delta(0.5);

    std::cout << "\n=== Running GPU Path ===" << std::endl;
    model_gpu.update(mesh, velocity, k, omega, nu_t_gpu, nullptr);

    // Statistics for GPU
    double min_gpu = 1e20, max_gpu = -1e20, sum_gpu = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double val = nu_t_gpu(i, j);
            min_gpu = std::min(min_gpu, val);
            max_gpu = std::max(max_gpu, val);
            sum_gpu += val;
        }
    }
    double avg_gpu = sum_gpu / (mesh.Nx * mesh.Ny);

    std::cout << "GPU Results:" << std::endl;
    std::cout << "  min(nu_t) = " << min_gpu << std::endl;
    std::cout << "  max(nu_t) = " << max_gpu << std::endl;
    std::cout << "  avg(nu_t) = " << avg_gpu << std::endl;

    // Now run CPU reference (force it by using small grid or direct CPU code)
    std::cout << "\n=== Computing CPU Reference ===" << std::endl;
    
    // Manually compute CPU version to avoid GPU path
    ScalarField dudx(mesh), dudy(mesh), dvdx(mesh), dvdy(mesh);
    compute_all_velocity_gradients(mesh, velocity, dudx, dudy, dvdx, dvdy);
    
    const double nu = 1.0 / 10000.0;
    const double kappa = 0.41;
    const double A_plus = 26.0;
    const double delta = 0.5;
    
    // Estimate u_tau
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_wall = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            dudy_wall += std::abs(dudy(i, j));
            ++count;
        }
        dudy_wall /= count;
        double tau_w = nu * dudy_wall;
        u_tau = std::sqrt(tau_w);
    }
    u_tau = std::max(u_tau, 1e-10);
    
    // CPU computation
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / nu;
            double damping = 1.0 - std::exp(-y_plus / A_plus);
            double l_mix = kappa * y_wall * damping;
            l_mix = std::min(l_mix, 0.5 * delta);
            
            double Sxx = dudx(i, j);
            double Syy = dvdy(i, j);
            double Sxy = 0.5 * (dudy(i, j) + dvdx(i, j));
            double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            nu_t_cpu(i, j) = l_mix * l_mix * S_mag;
        }
    }

    // Statistics for CPU
    double min_cpu = 1e20, max_cpu = -1e20, sum_cpu = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double val = nu_t_cpu(i, j);
            min_cpu = std::min(min_cpu, val);
            max_cpu = std::max(max_cpu, val);
            sum_cpu += val;
        }
    }
    double avg_cpu = sum_cpu / (mesh.Nx * mesh.Ny);

    std::cout << "CPU Results:" << std::endl;
    std::cout << "  min(nu_t) = " << min_cpu << std::endl;
    std::cout << "  max(nu_t) = " << max_cpu << std::endl;
    std::cout << "  avg(nu_t) = " << avg_cpu << std::endl;

    // Compare GPU vs CPU
    std::cout << "\n=== CPU vs GPU Comparison ===" << std::endl;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int max_i = -1, max_j = -1;
    double rms_diff = 0.0;
    double sum_abs_diff = 0.0;  // Track sum for extra verification
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double gpu_val = nu_t_gpu(i, j);
            double cpu_val = nu_t_cpu(i, j);
            double abs_diff = std::abs(gpu_val - cpu_val);
            double rel_diff = abs_diff / (std::abs(cpu_val) + 1e-20);
            
            sum_abs_diff += abs_diff;
            rms_diff += abs_diff * abs_diff;
            
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                max_rel_diff = rel_diff;
                max_i = i;
                max_j = j;
            }
        }
    }
    
    rms_diff = std::sqrt(rms_diff / (mesh.Nx * mesh.Ny));
    
    std::cout << "  Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "  RMS difference: " << rms_diff << std::endl;
    std::cout << "  Sum of absolute differences: " << sum_abs_diff << std::endl;
    if (max_abs_diff > 0) {
        std::cout << "  Location of max diff (i,j): (" << max_i << ", " << max_j << ")" << std::endl;
        std::cout << "    GPU value: " << nu_t_gpu(max_i, max_j) << std::endl;
        std::cout << "    CPU value: " << nu_t_cpu(max_i, max_j) << std::endl;
    }

    // Verdict
    std::cout << "\n=== Test Verdict ===" << std::endl;
    const double tol_abs = 1e-10;
    const double tol_rel = 1e-10;
    
    if (max_abs_diff < tol_abs || max_rel_diff < tol_rel) {
        std::cout << "✓ CPU and GPU results agree within tight numerical tolerance" << std::endl;
        std::cout << "  (abs_tol=" << tol_abs << ", rel_tol=" << tol_rel << ")" << std::endl;
        if (max_abs_diff == 0.0 && sum_abs_diff == 0.0) {
            std::cout << "  Note: Results happen to be bit-for-bit identical for this test case" << std::endl;
            std::cout << "        (not guaranteed in general; depends on compiler, ops, etc.)" << std::endl;
        }
        return 0;
    } else if (max_rel_diff < 1e-6) {
        std::cout << "✓ CPU and GPU results match to reasonable precision" << std::endl;
        std::cout << "  (Differences likely due to floating-point rounding)" << std::endl;
        return 0;
    } else {
        std::cout << "⚠ WARNING: Significant CPU vs GPU differences detected" << std::endl;
        std::cout << "  This may indicate a bug in the GPU implementation" << std::endl;
        return 1;
    }
}

