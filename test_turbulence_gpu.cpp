// Quick diagnostic test for MixingLengthModel GPU vs CPU
#include "mesh.hpp"
#include "fields.hpp"
#include "turbulence_baseline.hpp"
#include <iostream>
#include <cmath>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

int main() {
    std::cout << "=== MixingLengthModel GPU vs CPU Test ===" << std::endl;

#ifdef USE_GPU_OFFLOAD
    std::cout << "GPU offload: ENABLED" << std::endl;
    int num_devices = omp_get_num_devices();
    std::cout << "Number of GPU devices: " << num_devices << std::endl;
    if (num_devices == 0) {
        std::cout << "No GPU devices found - test cannot compare CPU vs GPU" << std::endl;
        return 1;
    }
#else
    std::cout << "GPU offload: DISABLED" << std::endl;
    std::cout << "This test requires GPU offload to be enabled" << std::endl;
    return 1;
#endif

    // Create a small mesh (should trigger GPU path with Nx=64, Ny=64)
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, 0.0, 1.0, 1);

    // Create fields
    VectorField velocity(mesh);
    ScalarField nu_t(mesh);
    ScalarField k(mesh);  // unused
    ScalarField omega(mesh);  // unused

    // Initialize velocity field with a simple profile
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y = mesh.yc[j];
            double u_parab = 4.0 * y * (1.0 - y);  // Parabolic profile
            velocity.u(i, j) = u_parab;
            velocity.v(i, j) = 0.0;
        }
    }

    // Create turbulence model
    MixingLengthModel model;
    model.set_nu(1.0 / 10000.0);  // Re = 10000
    model.set_delta(0.5);  // Half-height

    std::cout << "\nRunning MixingLengthModel.update()..." << std::endl;
    model.update(mesh, velocity, k, omega, nu_t, nullptr);

    std::cout << "\nChecking results..." << std::endl;
    
    // Find min, max, average nu_t
    double min_nu_t = 1e20;
    double max_nu_t = -1e20;
    double sum_nu_t = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double val = nu_t(i, j);
            min_nu_t = std::min(min_nu_t, val);
            max_nu_t = std::max(max_nu_t, val);
            sum_nu_t += val;
            ++count;
        }
    }

    double avg_nu_t = sum_nu_t / count;

    std::cout << "  min(nu_t) = " << min_nu_t << std::endl;
    std::cout << "  max(nu_t) = " << max_nu_t << std::endl;
    std::cout << "  avg(nu_t) = " << avg_nu_t << std::endl;

    // Check for reasonable values
    if (std::isnan(min_nu_t) || std::isnan(max_nu_t)) {
        std::cout << "\n✗ FAIL: NaN detected in nu_t" << std::endl;
        return 1;
    }

    if (min_nu_t < 0.0) {
        std::cout << "\n✗ FAIL: Negative nu_t detected" << std::endl;
        return 1;
    }

    std::cout << "\n✓ Test PASSED - Results look reasonable" << std::endl;
    std::cout << "\nNOTE: For CPU vs GPU comparison diagnostics, check stdout" << std::endl;
    std::cout << "during the first 3 calls to update() with grid >= 32x32" << std::endl;

    return 0;
}

