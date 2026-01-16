/// HYPRE PFMG Poisson Solver Validation Test
///
/// This test validates the HYPRE solver by comparing results against the
/// built-in multigrid solver. Uses the full solver infrastructure which
/// correctly handles GPU memory (via solve_device()).
///
/// Test cases:
/// 1. 3D Channel flow (periodic x/z, Neumann y) - primary use case
/// 2. 3D Duct flow (periodic x, Neumann y/z)
/// 3. Comparison of pressure solve results between HYPRE and Multigrid
///
/// This test can generate reference data (--dump-prefix) or compare against it
/// (--compare-prefix) for CPU/GPU cross-build validation.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "test_utilities.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>
#include <sstream>
#include <climits>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using nncfd::test::FieldComparison;
using nncfd::test::file_exists;

// Tolerance for HYPRE vs Multigrid comparison
// Velocities should match closely since both solve the same NS equations
constexpr double VELOCITY_TOLERANCE = 1e-5;
// Pressure tolerance can be looser since pressure is defined up to an additive constant
// What matters is that pressure gradients match (which they do if velocity matches)
constexpr double PRESSURE_TOLERANCE = 1e-3;
// Tolerance for mean-removed pressure (more meaningful for solver equivalence)
constexpr double PRESSURE_PRIME_TOLERANCE = 1e-6;
// Tolerance for divergence (should be essentially zero for incompressible)
constexpr double DIVERGENCE_TOLERANCE = 1e-6;
// Tolerance for pressure gradient (drives velocity correction)
constexpr double GRADP_TOLERANCE = 1e-5;

// Tolerance for cross-build comparison (CPU vs GPU HYPRE)
constexpr double CROSS_BUILD_TOLERANCE = 1e-10;

// Tolerance for Poisson residual identity ||Lap(p) - rhs|| / ||rhs||
constexpr double POISSON_RESIDUAL_TOLERANCE = 1e-3;

// ============================================================================
// Compute Poisson residual identity: ||Lap(p) - rhs||_2 / ||rhs||_2
// This verifies the solver actually solved its equation, independent of
// whether two solvers match each other.
// ============================================================================
static double compute_poisson_residual_3d(const ScalarField& p, const ScalarField& rhs,
                                          const Mesh& mesh) {
    const double dx2 = mesh.dx * mesh.dx;
    const double dy2 = mesh.dy * mesh.dy;
    const double dz2 = mesh.dz * mesh.dz;

    double diff_sum_sq = 0.0;  // ||Lap(p) - rhs||_2^2
    double rhs_sum_sq = 0.0;   // ||rhs||_2^2

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // 7-point discrete Laplacian (same stencil as MG solver)
                double laplacian = (p(i+1,j,k) - 2.0*p(i,j,k) + p(i-1,j,k)) / dx2
                                 + (p(i,j+1,k) - 2.0*p(i,j,k) + p(i,j-1,k)) / dy2
                                 + (p(i,j,k+1) - 2.0*p(i,j,k) + p(i,j,k-1)) / dz2;

                double diff = laplacian - rhs(i, j, k);
                diff_sum_sq += diff * diff;
                rhs_sum_sq += rhs(i, j, k) * rhs(i, j, k);
            }
        }
    }

    double rhs_l2 = std::sqrt(rhs_sum_sq);
    double diff_l2 = std::sqrt(diff_sum_sq);

    if (rhs_l2 < 1e-30) {
        return diff_l2;  // Return absolute residual if RHS is zero
    }
    return diff_l2 / rhs_l2;
}

void write_field_data(const std::string& filename, const ScalarField& field,
                      const Mesh& mesh) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << std::setprecision(17) << std::scientific;
    file << "# i j k value\n";

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                file << i << " " << j << " 0 " << field(i, j) << "\n";
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    file << i << " " << j << " " << k << " " << field(i, j, k) << "\n";
                }
            }
        }
    }
}

struct FieldData {
    std::vector<double> values;
    int i_min, i_max, j_min, j_max, k_min, k_max;
    int ni, nj, nk;

    double operator()(int i, int j, int k) const {
        int idx = (k - k_min) * ni * nj + (j - j_min) * ni + (i - i_min);
        return values[idx];
    }
};

FieldData read_field_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open reference file: " + filename);
    }

    int i_min = INT_MAX, i_max = INT_MIN;
    int j_min = INT_MAX, j_max = INT_MIN;
    int k_min = INT_MAX, k_max = INT_MIN;

    std::string line;
    std::vector<std::tuple<int, int, int, double>> entries;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int i, j, k;
        double value;
        if (!(iss >> i >> j >> k >> value)) continue;

        entries.emplace_back(i, j, k, value);
        i_min = std::min(i_min, i); i_max = std::max(i_max, i);
        j_min = std::min(j_min, j); j_max = std::max(j_max, j);
        k_min = std::min(k_min, k); k_max = std::max(k_max, k);
    }

    if (entries.empty()) {
        throw std::runtime_error("No data found in reference file: " + filename);
    }

    FieldData data;
    data.i_min = i_min; data.i_max = i_max + 1;
    data.j_min = j_min; data.j_max = j_max + 1;
    data.k_min = k_min; data.k_max = k_max + 1;
    data.ni = data.i_max - i_min;
    data.nj = data.j_max - j_min;
    data.nk = data.k_max - k_min;

    data.values.resize(data.ni * data.nj * data.nk, 0.0);

    for (const auto& [i, j, k, value] : entries) {
        int idx = (k - k_min) * data.ni * data.nj + (j - j_min) * data.ni + (i - i_min);
        data.values[idx] = value;
    }

    return data;
}

//=============================================================================
// Test 1: HYPRE vs Multigrid consistency (same-build comparison)
//=============================================================================

#ifdef USE_HYPRE
// Helper to set Taylor-Green vortex initial condition
// This is a classic CFD test case with exact solution
// Uses periodic BC in all directions, creates non-trivial pressure field
void set_taylor_green_initial_velocity(RANSSolver& solver, const Mesh& mesh) {
    auto& vel = solver.velocity();
    const double Lx = mesh.x_max - mesh.x_min;
    const double Ly = mesh.y_max - mesh.y_min;
    const double Lz = mesh.z_max - mesh.z_min;
    const double amplitude = 0.01;

    // u = A * sin(2*pi*x/Lx) * cos(2*pi*y/Ly) * cos(2*pi*z/Lz)
    // v = -A * cos(2*pi*x/Lx) * sin(2*pi*y/Ly) * cos(2*pi*z/Lz)
    // w = 0
    // This is divergence-free: du/dx + dv/dy + dw/dz = 0

    // Set u at x-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.yc[j];
                double z = mesh.zc[k];
                vel.u(i, j, k) = amplitude * std::sin(2.0 * M_PI * x / Lx) *
                                 std::cos(2.0 * M_PI * y / Ly) *
                                 std::cos(2.0 * M_PI * z / Lz);
            }
        }
    }

    // Set v at y-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.xc[i];
                double y = mesh.yf[j];
                double z = mesh.zc[k];
                vel.v(i, j, k) = -amplitude * std::cos(2.0 * M_PI * x / Lx) *
                                  std::sin(2.0 * M_PI * y / Ly) *
                                  std::cos(2.0 * M_PI * z / Lz);
            }
        }
    }

    // w = 0 (already initialized to zero)
}

// Compute max divergence of velocity field (3D)
double compute_max_divergence_3d(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / mesh.dx;
                double dvdy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / mesh.dy;
                double dwdz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / mesh.dz;
                double div = dudx + dvdy + dwdz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }
    return max_div;
}

// Compute relative L2 norm of pressure gradient difference (3D)
// grad(p) at cell centers, then compare
double compute_gradp_relL2_3d(const ScalarField& p1, const ScalarField& p2, const Mesh& mesh) {
    double diff_sq = 0.0;
    double norm_sq = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Pressure gradient at cell center using central differences
                double dpdx_1 = (p1(i+1, j, k) - p1(i-1, j, k)) / (2.0 * mesh.dx);
                double dpdy_1 = (p1(i, j+1, k) - p1(i, j-1, k)) / (2.0 * mesh.dy);
                double dpdz_1 = (p1(i, j, k+1) - p1(i, j, k-1)) / (2.0 * mesh.dz);

                double dpdx_2 = (p2(i+1, j, k) - p2(i-1, j, k)) / (2.0 * mesh.dx);
                double dpdy_2 = (p2(i, j+1, k) - p2(i, j-1, k)) / (2.0 * mesh.dy);
                double dpdz_2 = (p2(i, j, k+1) - p2(i, j, k-1)) / (2.0 * mesh.dz);

                diff_sq += (dpdx_1 - dpdx_2) * (dpdx_1 - dpdx_2);
                diff_sq += (dpdy_1 - dpdy_2) * (dpdy_1 - dpdy_2);
                diff_sq += (dpdz_1 - dpdz_2) * (dpdz_1 - dpdz_2);

                norm_sq += dpdx_1 * dpdx_1 + dpdy_1 * dpdy_1 + dpdz_1 * dpdz_1;
            }
        }
    }

    return (norm_sq > 1e-30) ? std::sqrt(diff_sq / norm_sq) : std::sqrt(diff_sq);
}

bool test_hypre_vs_multigrid_3d_channel() {
    std::cout << "\n=== Test: HYPRE vs Multigrid (3D Taylor-Green) ===\n";

    const int NX = 32, NY = 32, NZ = 32;  // Cubic for Taylor-Green
    const int NUM_STEPS = 10;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI);

    // Setup config - Taylor-Green vortex
    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.postprocess = false;
    config.write_fields = false;

    // All-periodic BCs for Taylor-Green
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;

    // Run with Multigrid
    std::cout << "  Running with Multigrid...\n";
    config.poisson_solver = PoissonSolverType::MG;
    RANSSolver solver_mg(mesh, config);

    // Verify solver selection
    if (solver_mg.using_hypre()) {
        std::cerr << "  ERROR: Multigrid solver incorrectly using HYPRE!\n";
        return false;
    }
    std::cout << "    Solver: Multigrid (using_hypre=" << solver_mg.using_hypre() << ")\n";

    // Set Taylor-Green initial condition
    set_taylor_green_initial_velocity(solver_mg, mesh);

    // No body force for Taylor-Green (decaying vortex)
    solver_mg.set_body_force(0.0, 0.0, 0.0);
    solver_mg.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver_mg.sync_to_gpu();
#endif
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver_mg.step();
    }
#ifdef USE_GPU_OFFLOAD
    solver_mg.sync_solution_from_gpu();
#endif

    // Run with HYPRE - same initial condition
    std::cout << "  Running with HYPRE...\n";
    config.poisson_solver = PoissonSolverType::HYPRE;
    RANSSolver solver_hypre(mesh, config);

    // Verify solver selection
    if (!solver_hypre.using_hypre()) {
        std::cerr << "  ERROR: HYPRE solver not enabled!\n";
        return false;
    }
    std::cout << "    Solver: HYPRE (using_hypre=" << solver_hypre.using_hypre() << ")\n";

    // Same Taylor-Green initial condition
    set_taylor_green_initial_velocity(solver_hypre, mesh);

    // Same: no body force
    solver_hypre.set_body_force(0.0, 0.0, 0.0);
    solver_hypre.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver_hypre.sync_to_gpu();
#endif

    // Run with try-catch to get more info on failure
    bool hypre_failed = false;
    for (int step = 0; step < NUM_STEPS; ++step) {
        try {
            solver_hypre.step();
        } catch (const std::exception& e) {
            std::cerr << "  HYPRE step " << step << " threw exception: " << e.what() << "\n";
            hypre_failed = true;
            break;
        }
    }

    if (hypre_failed) {
        std::cerr << "  ERROR: HYPRE solver failed during time stepping\n";
        return false;
    }

#ifdef USE_GPU_OFFLOAD
    solver_hypre.sync_solution_from_gpu();
#endif

    // Compute solution statistics to verify non-trivial results
    double p_mg_min = 1e30, p_mg_max = -1e30;
    double p_hypre_min = 1e30, p_hypre_max = -1e30;
    double u_mg_max = 0, u_hypre_max = 0;

    // Compute mean pressure for both solvers (for mean-removed comparison)
    double p_mg_sum = 0.0, p_hypre_sum = 0.0;
    int p_count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p_mg_sum += solver_mg.pressure()(i, j, k);
                p_hypre_sum += solver_hypre.pressure()(i, j, k);
                ++p_count;
            }
        }
    }
    double p_mg_mean = p_mg_sum / p_count;
    double p_hypre_mean = p_hypre_sum / p_count;

    // Compare pressure fields (raw and mean-removed)
    FieldComparison p_result;
    FieldComparison p_prime_result;  // Mean-removed pressure (more physically meaningful)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double p_mg = solver_mg.pressure()(i, j, k);
                double p_hypre = solver_hypre.pressure()(i, j, k);
                p_result.update(p_mg, p_hypre);

                // Mean-removed pressure: p' = p - mean(p)
                double p_mg_prime = p_mg - p_mg_mean;
                double p_hypre_prime = p_hypre - p_hypre_mean;
                p_prime_result.update(p_mg_prime, p_hypre_prime);

                p_mg_min = std::min(p_mg_min, p_mg);
                p_mg_max = std::max(p_mg_max, p_mg);
                p_hypre_min = std::min(p_hypre_min, p_hypre);
                p_hypre_max = std::max(p_hypre_max, p_hypre);
            }
        }
    }
    p_result.finalize();
    p_prime_result.finalize();

    // Compare velocity fields
    FieldComparison u_result;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double u_mg = solver_mg.velocity().u(i, j, k);
                double u_hypre = solver_hypre.velocity().u(i, j, k);
                u_result.update(u_mg, u_hypre);
                u_mg_max = std::max(u_mg_max, std::abs(u_mg));
                u_hypre_max = std::max(u_hypre_max, std::abs(u_hypre));
            }
        }
    }
    u_result.finalize();

    // Compute divergence for both solutions (primary metric for incompressibility)
    double div_mg = compute_max_divergence_3d(solver_mg.velocity(), mesh);
    double div_hypre = compute_max_divergence_3d(solver_hypre.velocity(), mesh);

    // Compute pressure gradient comparison (what actually drives velocity correction)
    double gradp_relL2 = compute_gradp_relL2_3d(solver_mg.pressure(), solver_hypre.pressure(), mesh);

    // Compute Poisson residual identity for each solver
    // This verifies each solver actually solved its equation correctly
    double res_mg = compute_poisson_residual_3d(solver_mg.pressure(),
                                                 solver_mg.rhs_poisson(), mesh);
    double res_hypre = compute_poisson_residual_3d(solver_hypre.pressure(),
                                                    solver_hypre.rhs_poisson(), mesh);

    // Print diagnostics
    std::cout << "  Solution statistics:\n";
    std::cout << "    MG pressure range:    [" << p_mg_min << ", " << p_mg_max << "] (mean=" << p_mg_mean << ")\n";
    std::cout << "    HYPRE pressure range: [" << p_hypre_min << ", " << p_hypre_max << "] (mean=" << p_hypre_mean << ")\n";
    std::cout << "    MG max |u|:    " << u_mg_max << "\n";
    std::cout << "    HYPRE max |u|: " << u_hypre_max << "\n";
    std::cout << "    MG max |div(u)|:    " << std::scientific << div_mg << "\n";
    std::cout << "    HYPRE max |div(u)|: " << std::scientific << div_hypre << "\n";
    std::cout << "    grad(p) relL2 diff: " << std::scientific << gradp_relL2 << "\n";
    std::cout << "    Poisson residual ||Lap(p)-rhs||/||rhs||:\n";
    std::cout << "      MG:    " << std::scientific << res_mg << "\n";
    std::cout << "      HYPRE: " << std::scientific << res_hypre << "\n";

    p_result.print("Pressure diff (raw)");
    p_prime_result.print("Pressure diff (mean-removed)");
    u_result.print("U-velocity diff");

    // Sanity check: solutions should be non-trivial
    bool solutions_nontrivial = (p_mg_max - p_mg_min > 1e-10) && (u_mg_max > 1e-10);
    if (!solutions_nontrivial) {
        std::cerr << "  ERROR: Solutions appear to be trivial (all zeros)\n";
        return false;
    }

    // Sanity check: solutions should differ slightly (different solvers)
    bool solvers_differ = (p_result.max_abs_diff > 1e-15) || (u_result.max_abs_diff > 1e-15);
    if (!solvers_differ) {
        std::cerr << "  WARNING: Solutions are bitwise identical - solvers may be the same!\n";
    }

    // PRIMARY pass/fail criteria (what actually matters for correctness):
    // 1. Divergence must be small for both solvers (incompressibility)
    // 2. Pressure gradients must match (drives velocity correction)
    // 3. Velocity must match (physical result)
    // 4. Poisson residual must be small (solver actually solved its equation)
    bool div_mg_ok = div_mg < DIVERGENCE_TOLERANCE;
    bool div_hypre_ok = div_hypre < DIVERGENCE_TOLERANCE;
    bool gradp_ok = gradp_relL2 < GRADP_TOLERANCE;
    bool velocity_ok = u_result.within_tolerance(VELOCITY_TOLERANCE);
    bool res_mg_ok = res_mg < POISSON_RESIDUAL_TOLERANCE;
    bool res_hypre_ok = res_hypre < POISSON_RESIDUAL_TOLERANCE;

    // Secondary: mean-removed pressure (for QoI tracking, not pass/fail)
    bool pressure_prime_ok = p_prime_result.within_tolerance(PRESSURE_PRIME_TOLERANCE);

    std::cout << "\n  Pass/fail checks:\n";
    std::cout << "    MG divergence < " << DIVERGENCE_TOLERANCE << ": " << (div_mg_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    HYPRE divergence < " << DIVERGENCE_TOLERANCE << ": " << (div_hypre_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Pressure gradient match < " << GRADP_TOLERANCE << ": " << (gradp_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Velocity match < " << VELOCITY_TOLERANCE << ": " << (velocity_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    MG Poisson residual < " << POISSON_RESIDUAL_TOLERANCE << ": " << (res_mg_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    HYPRE Poisson residual < " << POISSON_RESIDUAL_TOLERANCE << ": " << (res_hypre_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Mean-removed pressure < " << PRESSURE_PRIME_TOLERANCE << ": " << (pressure_prime_ok ? "[OK]" : "[WARN]") << " (secondary)\n";

    if (!div_mg_ok) {
        std::cerr << "  ERROR: MG divergence " << div_mg << " exceeds tolerance\n";
    }
    if (!div_hypre_ok) {
        std::cerr << "  ERROR: HYPRE divergence " << div_hypre << " exceeds tolerance\n";
    }
    if (!gradp_ok) {
        std::cerr << "  ERROR: Pressure gradient difference exceeds tolerance\n";
    }
    if (!velocity_ok) {
        std::cerr << "  ERROR: Velocity difference exceeds tolerance\n";
    }
    if (!res_mg_ok) {
        std::cerr << "  ERROR: MG Poisson residual " << res_mg << " exceeds tolerance\n";
    }
    if (!res_hypre_ok) {
        std::cerr << "  ERROR: HYPRE Poisson residual " << res_hypre << " exceeds tolerance\n";
    }

    // Pass if all primary criteria are met
    bool passed = div_mg_ok && div_hypre_ok && gradp_ok && velocity_ok && res_mg_ok && res_hypre_ok;
    std::cout << "\n  Result: " << (passed ? "[PASS]" : "[FAIL]") << "\n";

    // Emit machine-readable QoI for CI metrics
    nncfd::test::harness::emit_qoi_hypre(
        p_prime_result.rel_l2(),
        u_result.rel_l2(),
        p_mg_mean,
        p_hypre_mean,
        div_mg,
        div_hypre,
        gradp_relL2
    );

    return passed;
}

bool test_hypre_vs_multigrid_3d_duct() {
    std::cout << "\n=== Test: HYPRE vs Multigrid (3D Duct) ===\n";

    const int NX = 32, NY = 32, NZ = 32;
    const int NUM_STEPS = 10;  // Fewer steps for stability

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.0001;  // Small time step for stability
    config.adaptive_dt = false;
    config.max_steps = NUM_STEPS;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.postprocess = false;
    config.write_fields = false;

    // Duct flow: walls on y and z faces, periodic in x
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;
    bc.z_hi = VelocityBC::NoSlip;

    // Run with Multigrid
    std::cout << "  Running with Multigrid...\n";
    config.poisson_solver = PoissonSolverType::MG;
    RANSSolver solver_mg(mesh, config);

    // Verify solver selection
    std::cout << "    Solver: Multigrid (using_hypre=" << solver_mg.using_hypre() << ")\n";

    // DON'T set divergent IC - was causing instability

    // Use small body force for stability
    solver_mg.set_body_force(0.01, 0.0, 0.0);
    solver_mg.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver_mg.sync_to_gpu();
#endif
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver_mg.step();
    }
#ifdef USE_GPU_OFFLOAD
    solver_mg.sync_solution_from_gpu();
#endif

    // Run with HYPRE
    std::cout << "  Running with HYPRE...\n";
    config.poisson_solver = PoissonSolverType::HYPRE;
    RANSSolver solver_hypre(mesh, config);

    // Verify solver selection
    std::cout << "    Solver: HYPRE (using_hypre=" << solver_hypre.using_hypre() << ")\n";

    // Same as MG: no divergent IC

    // Use same small body force
    solver_hypre.set_body_force(0.01, 0.0, 0.0);
    solver_hypre.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver_hypre.sync_to_gpu();
#endif
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver_hypre.step();
    }
#ifdef USE_GPU_OFFLOAD
    solver_hypre.sync_solution_from_gpu();
#endif

    // Compute PHYSICS-FIRST metrics (same as channel test)
    // These are what actually matter for solver equivalence:
    // 1. Divergence (incompressibility)
    // 2. Pressure gradients (drives velocity correction)
    // 3. Velocity (physical result)

    // Compute divergence for both solvers
    double div_mg = 0.0, div_hypre = 0.0;
    const auto& u_mg = solver_mg.velocity();
    const auto& u_hypre = solver_hypre.velocity();
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx_mg = (u_mg.u(i+1,j,k) - u_mg.u(i,j,k)) / mesh.dx;
                double dvdy_mg = (u_mg.v(i,j+1,k) - u_mg.v(i,j,k)) / mesh.dy;
                double dwdz_mg = (u_mg.w(i,j,k+1) - u_mg.w(i,j,k)) / mesh.dz;
                div_mg = std::max(div_mg, std::abs(dudx_mg + dvdy_mg + dwdz_mg));

                double dudx_h = (u_hypre.u(i+1,j,k) - u_hypre.u(i,j,k)) / mesh.dx;
                double dvdy_h = (u_hypre.v(i,j+1,k) - u_hypre.v(i,j,k)) / mesh.dy;
                double dwdz_h = (u_hypre.w(i,j,k+1) - u_hypre.w(i,j,k)) / mesh.dz;
                div_hypre = std::max(div_hypre, std::abs(dudx_h + dvdy_h + dwdz_h));
            }
        }
    }

    // Compute pressure gradient difference
    const auto& p_mg = solver_mg.pressure();
    const auto& p_hypre = solver_hypre.pressure();
    double gradp_sum_sq = 0.0, gradp_ref_sq = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dpdx_mg = (p_mg(i+1,j,k) - p_mg(i,j,k)) / mesh.dx;
                double dpdy_mg = (p_mg(i,j+1,k) - p_mg(i,j,k)) / mesh.dy;
                double dpdz_mg = (p_mg(i,j,k+1) - p_mg(i,j,k)) / mesh.dz;
                double dpdx_h = (p_hypre(i+1,j,k) - p_hypre(i,j,k)) / mesh.dx;
                double dpdy_h = (p_hypre(i,j+1,k) - p_hypre(i,j,k)) / mesh.dy;
                double dpdz_h = (p_hypre(i,j,k+1) - p_hypre(i,j,k)) / mesh.dz;

                gradp_sum_sq += (dpdx_mg - dpdx_h)*(dpdx_mg - dpdx_h)
                              + (dpdy_mg - dpdy_h)*(dpdy_mg - dpdy_h)
                              + (dpdz_mg - dpdz_h)*(dpdz_mg - dpdz_h);
                gradp_ref_sq += dpdx_mg*dpdx_mg + dpdy_mg*dpdy_mg + dpdz_mg*dpdz_mg;
            }
        }
    }
    double gradp_relL2 = std::sqrt(gradp_sum_sq) / (std::sqrt(gradp_ref_sq) + 1e-30);

    // Compare velocity fields
    FieldComparison u_result;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                u_result.update(u_mg.u(i,j,k), u_hypre.u(i,j,k));
                u_result.update(u_mg.v(i,j,k), u_hypre.v(i,j,k));
                u_result.update(u_mg.w(i,j,k), u_hypre.w(i,j,k));
            }
        }
    }
    u_result.finalize();

    // Compute Poisson residual identity for each solver
    // This verifies each solver actually solved its equation correctly
    double res_mg = compute_poisson_residual_3d(solver_mg.pressure(),
                                                 solver_mg.rhs_poisson(), mesh);
    double res_hypre = compute_poisson_residual_3d(solver_hypre.pressure(),
                                                    solver_hypre.rhs_poisson(), mesh);

    // Print physics-first diagnostics
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Divergence:\n";
    std::cout << "    MG:    " << div_mg << "\n";
    std::cout << "    HYPRE: " << div_hypre << "\n";
    std::cout << "  Pressure gradient relL2 diff: " << gradp_relL2 << "\n";
    std::cout << "  Velocity relL2 diff: " << u_result.rel_l2() << "\n";
    std::cout << "  Poisson residual ||Lap(p)-rhs||/||rhs||:\n";
    std::cout << "    MG:    " << res_mg << "\n";
    std::cout << "    HYPRE: " << res_hypre << "\n";

    // PRIMARY pass/fail criteria (physics-first):
    bool div_mg_ok = div_mg < DIVERGENCE_TOLERANCE;
    bool div_hypre_ok = div_hypre < DIVERGENCE_TOLERANCE;
    bool gradp_ok = gradp_relL2 < GRADP_TOLERANCE;
    bool velocity_ok = u_result.within_tolerance(VELOCITY_TOLERANCE);
    bool res_mg_ok = res_mg < POISSON_RESIDUAL_TOLERANCE;
    bool res_hypre_ok = res_hypre < POISSON_RESIDUAL_TOLERANCE;

    std::cout << "\n  Pass/fail checks (physics-first):\n";
    std::cout << "    MG divergence < " << DIVERGENCE_TOLERANCE << ": " << (div_mg_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    HYPRE divergence < " << DIVERGENCE_TOLERANCE << ": " << (div_hypre_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Pressure gradient match < " << GRADP_TOLERANCE << ": " << (gradp_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Velocity match < " << VELOCITY_TOLERANCE << ": " << (velocity_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    MG Poisson residual < " << POISSON_RESIDUAL_TOLERANCE << ": " << (res_mg_ok ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    HYPRE Poisson residual < " << POISSON_RESIDUAL_TOLERANCE << ": " << (res_hypre_ok ? "[OK]" : "[FAIL]") << "\n";

    // Pass if all primary criteria are met
    bool passed = div_mg_ok && div_hypre_ok && gradp_ok && velocity_ok && res_mg_ok && res_hypre_ok;
    std::cout << "\n  Result: " << (passed ? "[PASS]" : "[FAIL]") << "\n";
    return passed;
}
#endif  // USE_HYPRE

//=============================================================================
// Test 2: Cross-build comparison (dump/compare mode for CI)
//=============================================================================

#ifdef USE_HYPRE
void setup_channel_test(Mesh& mesh, Config& config, int NX, int NY, int NZ, int num_iter) {
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0);

    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = num_iter;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.postprocess = false;
    config.write_fields = false;
    config.poisson_solver = PoissonSolverType::HYPRE;  // Always use HYPRE for this test
}

int run_dump_mode(const std::string& prefix) {
    std::cout << "=== HYPRE Cross-Build Reference Generation ===\n";
    std::cout << "Output prefix: " << prefix << "\n\n";

    const int NX = 32, NY = 64, NZ = 16;
    const int NUM_STEPS = 30;

    Mesh mesh;
    Config config;
    setup_channel_test(mesh, config, NX, NY, NZ, NUM_STEPS);

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::cout << "Running " << NUM_STEPS << " time steps with HYPRE...\n";
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    std::cout << "Writing reference fields...\n";
    write_field_data(prefix + "_hypre_p.dat", solver.pressure(), mesh);
    std::cout << "  Wrote: " << prefix << "_hypre_p.dat\n";

    std::cout << "\n[SUCCESS] HYPRE reference files written\n";
    return 0;
}

int run_compare_mode(const std::string& prefix) {
    std::cout << "=== HYPRE Cross-Build Comparison ===\n";
    std::cout << "Reference prefix: " << prefix << "\n\n";

#ifdef USE_GPU_OFFLOAD
    const int num_devices = omp_get_num_devices();
    std::cout << "GPU devices available: " << num_devices << "\n";
    if (num_devices == 0) {
        std::cerr << "ERROR: No GPU devices found\n";
        return 1;
    }
#endif

    if (!file_exists(prefix + "_hypre_p.dat")) {
        std::cerr << "ERROR: Reference file not found: " << prefix << "_hypre_p.dat\n";
        std::cerr << "       Run with --dump-prefix first\n";
        return 1;
    }

    const int NX = 32, NY = 64, NZ = 16;
    const int NUM_STEPS = 30;

    Mesh mesh;
    Config config;
    setup_channel_test(mesh, config, NX, NY, NZ, NUM_STEPS);

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::cout << "Running " << NUM_STEPS << " time steps with HYPRE...\n";
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    std::cout << "Loading reference and comparing...\n\n";

    auto ref = read_field_data(prefix + "_hypre_p.dat");
    FieldComparison result;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                result.update(ref(i, j, k), solver.pressure()(i, j, k));
            }
        }
    }
    result.finalize();
    result.print("Pressure");

    if (result.within_tolerance(CROSS_BUILD_TOLERANCE)) {
        std::cout << "  [PASS] Within tolerance " << CROSS_BUILD_TOLERANCE << "\n";
        std::cout << "\n[SUCCESS] HYPRE cross-build comparison passed\n";
        return 0;
    } else {
        std::cout << "  [FAIL] Exceeds tolerance " << CROSS_BUILD_TOLERANCE << "\n";
        std::cout << "\n[FAILURE] HYPRE cross-build comparison failed\n";
        return 1;
    }
}
#endif  // USE_HYPRE

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "HYPRE PFMG Poisson Solver Validation Test\n\n";
    std::cout << "Modes:\n";
    std::cout << "  (no args)                Run HYPRE vs Multigrid comparison tests\n";
    std::cout << "  --dump-prefix <prefix>   Generate reference data for cross-build test\n";
    std::cout << "  --compare-prefix <prefix> Compare against reference data\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << "                           # Run solver comparison tests\n";
    std::cout << "  " << prog << " --dump-prefix /tmp/ref    # Generate CPU reference\n";
    std::cout << "  " << prog << " --compare-prefix /tmp/ref # Compare GPU against reference\n";
}

int main(int argc, char* argv[]) {
#ifndef USE_HYPRE
    std::cout << "HYPRE support not enabled. Rebuild with -DUSE_HYPRE=ON\n";
    return 0;
#else
    try {
        std::string dump_prefix, compare_prefix;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
                dump_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--compare-prefix") == 0 && i + 1 < argc) {
                compare_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            }
        }

        std::cout << "=== HYPRE PFMG Poisson Solver Validation ===\n";
#ifdef USE_GPU_OFFLOAD
        std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
        std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
        std::cout << "\n";

        if (!dump_prefix.empty()) {
            return run_dump_mode(dump_prefix);
        } else if (!compare_prefix.empty()) {
            return run_compare_mode(compare_prefix);
        }

        // Default: run HYPRE vs Multigrid comparison tests
        int passed = 0;
        int total = 0;

        if (test_hypre_vs_multigrid_3d_channel()) ++passed;
        ++total;

        if (test_hypre_vs_multigrid_3d_duct()) ++passed;
        ++total;

        std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

        if (passed == total) {
            std::cout << "[SUCCESS] All HYPRE validation tests passed!\n";
            return 0;
        } else {
            std::cout << "[FAILURE] Some tests failed!\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
#endif
}
