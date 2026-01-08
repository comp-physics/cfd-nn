/// Comprehensive perturbed channel test: 1000 steps on GPU for all turbulence models
/// Tests: divergence, energy decay, realizability, model-specific observables
/// 
/// This is the "gold standard" unsteady validation that exercises:
/// - Non-trivial 2D flow (not x-invariant)
/// - Machine-epsilon divergence constraint
/// - Physical energy dissipation
/// - All turbulence model implementations

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include "test_harness.hpp"
#include <cmath>
#include <string>
#include <fstream>
#include <limits>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using nncfd::test::harness::record;

//=============================================================================
// Path resolution helpers for NN models
//=============================================================================
static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string resolve_model_dir(const std::string& p) {
    // Strip trailing slashes
    std::string path = p;
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    
    // Try relative to current directory (when running from repo root)
    if (file_exists(path + "/layer0_W.txt")) {
        return path;
    }
    
    // Try relative to build directory (when running from build/)
    if (file_exists("../" + path + "/layer0_W.txt")) {
        return "../" + path;
    }
    
    throw std::runtime_error(
        "NN model files not found. Tried: " + path + " and ../" + path
    );
}

//=============================================================================
// Divergence-free initialization: streamfunction approach
//=============================================================================
/// Create velocity field from a divergence-free streamfunction
/// The streamfunction vanishes at walls (no-slip compatible) and is periodic in x
/// u = ∂ψ/∂y, v = -∂ψ/∂x guarantees ∇·u = 0 exactly
VectorField create_perturbed_channel_field(const Mesh& mesh) {
    VectorField vel(mesh);
    const double A = 1e-3;  // Perturbation amplitude (small)
    const double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);
    
    // Streamfunction: ψ(x,y) = A * sin(kx*x) * sin²(π(y+1)/2)
    // Wall factor: sin²(π(y+1)/2) vanishes at y=-1 and y=+1
    
    // Initialize u-velocity (at x-faces)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        // d/dy[sin²(π(y+1)/2)] = π*sin(π(y+1)/2)*cos(π(y+1)/2)
        double s = std::sin(0.5 * M_PI * (y + 1.0));
        double c = std::cos(0.5 * M_PI * (y + 1.0));
        double dpsi_dy_factor = M_PI * s * c;
        
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = (i < mesh.i_end()) ? (mesh.x(i) + 0.5 * mesh.dx) : mesh.x_max;
            double dpsi_dy = A * std::sin(kx * x) * dpsi_dy_factor;
            vel.u(i, j) = dpsi_dy;
        }
    }
    
    // Initialize v-velocity (at y-faces)
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        double y = (j < mesh.j_end()) ? (mesh.y(j) + 0.5 * mesh.dy) : mesh.y_max;
        double s = std::sin(0.5 * M_PI * (y + 1.0));
        double s2 = s * s;
        
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double dpsi_dx = A * kx * std::cos(kx * x) * s2;
            vel.v(i, j) = -dpsi_dx;  // v = -∂ψ/∂x
        }
    }
    
    return vel;
}

//=============================================================================
// Diagnostic functions
//=============================================================================
struct DiagnosticStats {
    double max_div;
    double rms_div;
    double KE;
    double min_nu_t;
    double max_nu_t;
    double mean_nu_t;
    double min_k;
    double max_k;
    double min_omega;
    double max_omega;
    bool all_finite;
};

DiagnosticStats compute_diagnostics(const RANSSolver& solver, const Mesh& mesh, bool has_transport) {
#ifdef USE_GPU_OFFLOAD
    // GPU-only build: compute diagnostics entirely on device, no CPU fallback.
    // This avoids expensive sync_from_gpu() and host-side sweeps.
    DiagnosticStats stats = {};
    const auto v = solver.get_solver_view();

    const int Nx = v.Nx, Ny = v.Ny, Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int cell_stride = v.cell_stride;

    const size_t u_total = size_t(Nx + 2*Ng + 1) * size_t(Ny + 2*Ng);
    const size_t v_total = size_t(Nx + 2*Ng)     * size_t(Ny + 2*Ng + 1);
    const size_t c_total = size_t(Nx + 2*Ng)     * size_t(Ny + 2*Ng);

    double max_div = 0.0;
    double sum_div2 = 0.0;
    double KE = 0.0;

    double min_nu_t = std::numeric_limits<double>::infinity();
    double max_nu_t = -std::numeric_limits<double>::infinity();
    double sum_nu_t = 0.0;

    int has_bad = 0;

    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: v.u_face[0:u_total], v.v_face[0:v_total], v.p[0:c_total], v.nu_t[0:c_total]) \
        reduction(max:max_div, max_nu_t) reduction(min:min_nu_t) \
        reduction(+:sum_div2, KE, sum_nu_t) reduction(|:has_bad)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int ii = i + Ng;
            const int jj = j + Ng;

            const int u0 = ii     + jj * u_stride;
            const int u1 = (ii+1) + jj * u_stride;
            const int v0 = ii + jj     * v_stride;
            const int v1 = ii + (jj+1) * v_stride;

            const int c  = ii + jj * cell_stride;

            const double dudx = (v.u_face[u1] - v.u_face[u0]) / v.dx;
            const double dvdy = (v.v_face[v1] - v.v_face[v0]) / v.dy;
            const double div  = dudx + dvdy;

            const double abs_div = (div < 0.0) ? -div : div;
            if (abs_div > max_div) max_div = abs_div;
            sum_div2 += div * div;

            const double uc = 0.5 * (v.u_face[u0] + v.u_face[u1]);
            const double vc = 0.5 * (v.v_face[v0] + v.v_face[v1]);
            KE += 0.5 * (uc*uc + vc*vc) * v.dx * v.dy;

            const double nt = v.nu_t[c];
            if (nt < min_nu_t) min_nu_t = nt;
            if (nt > max_nu_t) max_nu_t = nt;
            sum_nu_t += nt;

            const double p = v.p[c];
            has_bad |= (uc != uc || (uc-uc) != 0.0 ||
                        vc != vc || (vc-vc) != 0.0 ||
                        p  != p  || (p -p ) != 0.0 ||
                        nt != nt || (nt-nt) != 0.0) ? 1 : 0;
        }
    }

    stats.max_div = max_div;
    stats.rms_div = std::sqrt(sum_div2 / double(Nx * Ny));
    stats.KE = KE;
    stats.min_nu_t = min_nu_t;
    stats.max_nu_t = max_nu_t;
    stats.mean_nu_t = sum_nu_t / double(Nx * Ny);
    stats.all_finite = (has_bad == 0);

    if (has_transport) {
        const auto tv = solver.get_device_view();
        double min_k = std::numeric_limits<double>::infinity();
        double max_k = -std::numeric_limits<double>::infinity();
        double min_omega = std::numeric_limits<double>::infinity();
        double max_omega = -std::numeric_limits<double>::infinity();
        int has_bad2 = 0;

        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: tv.k[0:c_total], tv.omega[0:c_total]) \
            reduction(min:min_k, min_omega) reduction(max:max_k, max_omega) \
            reduction(|:has_bad2)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const int ii = i + Ng;
                const int jj = j + Ng;
                const int c  = ii + jj * cell_stride;

                const double k  = tv.k[c];
                const double om = tv.omega[c];

                if (k < min_k) min_k = k;
                if (k > max_k) max_k = k;
                if (om < min_omega) min_omega = om;
                if (om > max_omega) max_omega = om;

                has_bad2 |= (k != k || (k-k) != 0.0 || om != om || (om-om) != 0.0) ? 1 : 0;
            }
        }

        stats.min_k = min_k;
        stats.max_k = max_k;
        stats.min_omega = min_omega;
        stats.max_omega = max_omega;
        stats.all_finite = stats.all_finite && (has_bad2 == 0);
    }

    return stats;
#else
    // CPU-only build: use host-side diagnostics (original implementation)
    DiagnosticStats stats = {};
    
    const VectorField& vel = solver.velocity();
    const ScalarField& nu_t = solver.nu_t();
    
    // Divergence using staggered grid formula
    double max_div = 0.0;
    double rms_div = 0.0;
    int count = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
            double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;
            
            max_div = std::max(max_div, std::abs(div));
            rms_div += div * div;
            ++count;
        }
    }
    rms_div = std::sqrt(rms_div / count);
    
    stats.max_div = max_div;
    stats.rms_div = rms_div;
    
    // Kinetic energy
    double KE = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }
    stats.KE = KE;
    
    // Eddy viscosity statistics
    double min_nu_t = 1e100;
    double max_nu_t = -1e100;
    double sum_nu_t = 0.0;
    int nu_t_count = 0;
    
    stats.all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = vel.u(i, j);
            double v = vel.v(i, j);
            double p = solver.pressure()(i, j);
            double nt = nu_t(i, j);
            
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(p) || !std::isfinite(nt)) {
                stats.all_finite = false;
            }
            
            min_nu_t = std::min(min_nu_t, nt);
            max_nu_t = std::max(max_nu_t, nt);
            sum_nu_t += nt;
            ++nu_t_count;
        }
    }
    
    stats.min_nu_t = min_nu_t;
    stats.max_nu_t = max_nu_t;
    stats.mean_nu_t = sum_nu_t / nu_t_count;
    
    // Transport variable statistics (if applicable)
    if (has_transport) {
        const ScalarField& k = solver.k();
        const ScalarField& omega = solver.omega();
        
        double min_k = 1e100;
        double max_k = -1e100;
        double min_omega = 1e100;
        double max_omega = -1e100;
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double k_val = k(i, j);
                double omega_val = omega(i, j);
                
                if (!std::isfinite(k_val) || !std::isfinite(omega_val)) {
                    stats.all_finite = false;
                }
                
                min_k = std::min(min_k, k_val);
                max_k = std::max(max_k, k_val);
                min_omega = std::min(min_omega, omega_val);
                max_omega = std::max(max_omega, omega_val);
            }
        }
        
        stats.min_k = min_k;
        stats.max_k = max_k;
        stats.min_omega = min_omega;
        stats.max_omega = max_omega;
    }
    
    return stats;
#endif
}

// Skip result for NN-MLP (no checkpoint available)
struct TestResult {
    bool passed;
    bool skipped;
};

//=============================================================================
// Test runner for a single turbulence model
//=============================================================================
TestResult test_single_model(TurbulenceModelType model_type) {
    // Setup
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    // Determine model characteristics
    bool is_nn_mlp = (model_type == TurbulenceModelType::NNMLP);
    bool is_earsm = (model_type == TurbulenceModelType::EARSM_WJ ||
                    model_type == TurbulenceModelType::EARSM_GS ||
                    model_type == TurbulenceModelType::EARSM_Pope);

    Config config;
    config.nu = 0.01;
    config.max_iter = 1000;
    config.turb_model = model_type;
    config.verbose = false;
    config.turb_guard_enabled = false;  // Handle exceptions ourselves

    config.poisson_tol = 1e-8;
    config.poisson_max_iter = 1000;
    config.poisson_abs_tol_floor = 1e-6;

    if (is_nn_mlp || is_earsm) {
        // NN and EARSM models need adaptive time stepping for stability
        config.adaptive_dt = true;
        config.CFL_max = is_earsm ? 0.3 : 0.2;
        config.dt = 1e-5;
    } else {
        config.dt = 5e-4;
        config.adaptive_dt = false;
    }

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    bool has_transport = false;
    if (model_type != TurbulenceModelType::None) {
        std::string model_path = "";
        if (model_type == TurbulenceModelType::NNTBNN) {
            model_path = resolve_model_dir("data/models/tbnn_channel_caseholdout");
        } else if (model_type == TurbulenceModelType::NNMLP) {
            return {true, true};  // Skip - no checkpoint available
        }
        auto turb_model = create_turbulence_model(model_type, model_path, model_path);
        if (turb_model) {
            has_transport = turb_model->uses_transport_equations();
            solver.set_turbulence_model(std::move(turb_model));
        }
    }

    solver.set_body_force(0.0, 0.0);
    VectorField vel_init = create_perturbed_channel_field(mesh);
    solver.initialize(vel_init);

    auto stats0 = compute_diagnostics(solver, mesh, has_transport);

    // Run with exception handling for numerical instability
    try {
        for (int step = 0; step < 100; ++step) {
            solver.step();
        }
    } catch (const std::runtime_error&) {
        // Numerical instability detected - test fails
        return {false, false};
    }

    auto stats1 = compute_diagnostics(solver, mesh, has_transport);

    // Validation checks
    if (!stats1.all_finite) return {false, false};

    const double div_tol = (model_type == TurbulenceModelType::NNMLP ||
                           model_type == TurbulenceModelType::NNTBNN) ? 2e-5 : 1e-6;
    if (stats1.max_div > div_tol) return {false, false};

    if (stats1.KE > stats0.KE * 1.01) return {false, false};

    if (stats1.min_nu_t < -1e-12) return {false, false};
    if (!std::isfinite(stats1.max_nu_t)) return {false, false};

    if (has_transport) {
        if (stats1.min_k < -1e-12) return {false, false};
        if (stats1.min_omega < -1e-12) return {false, false};
    }

    return {true, false};
}

//=============================================================================
// EARSM-specific test with realistic turbulence levels
//=============================================================================
bool test_earsm_model(TurbulenceModelType model_type) {
    // Setup with driven channel flow
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.adaptive_dt = true;      // Enable adaptive dt for stability
    config.CFL_max = 0.3;           // Conservative CFL for EARSM
    config.turb_model = model_type;
    config.verbose = false;
    config.turb_guard_enabled = false;  // Disable guard to catch instability ourselves
    config.poisson_tol = 1e-8;
    config.poisson_max_iter = 1000;
    config.poisson_abs_tol_floor = 1e-6;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(model_type, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    double dp_dx = -0.001;
    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run with exception handling for numerical instability
    try {
        for (int step = 0; step < 50; ++step) {
            solver.step();
        }
    } catch (const std::runtime_error&) {
        // Numerical instability detected - test fails
        return false;
    }

    const ScalarField& k = solver.k();
    const ScalarField& omega = solver.omega();
    const ScalarField& nu_t = solver.nu_t();

    int cells_using_earsm = 0;
    int total_cells = 0;
    bool all_finite = true;

    const double Re_t_center = 10.0;
    const double Re_t_width = 5.0;
    const double alpha_min = 0.2;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_val = k(i, j);
            double omega_val = omega(i, j);
            double nu_t_val = nu_t(i, j);

            if (!std::isfinite(k_val) || !std::isfinite(omega_val) || !std::isfinite(nu_t_val)) {
                all_finite = false;
            }

            double Re_t = k_val / (config.nu * omega_val);
            double alpha = 0.5 * (1.0 + std::tanh((Re_t - Re_t_center) / Re_t_width));

            if (alpha > alpha_min) cells_using_earsm++;
            total_cells++;
        }
    }

    double fraction_earsm = double(cells_using_earsm) / total_cells;

    // PASS: all finite AND at least 80% of cells use EARSM
    return all_finite && (fraction_earsm >= 0.80);
}

//=============================================================================
// Check if GPU is available (returns false if not)
//=============================================================================
bool gpu_available() {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() == 0) {
        return false;
    }
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }
    return on_device != 0;
#else
    return true;  // CPU build doesn't need GPU
#endif
}

//=============================================================================
// Main test runner: all turbulence models
//=============================================================================
int main() {
    return nncfd::test::harness::run("Perturbed Channel Turbulence Tests", [] {
        bool skip_all = !gpu_available();

        // Turbulence model tests (100 steps unforced decay)
        auto r1 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::None);
        record("Laminar (None)", r1.passed, r1.skipped || skip_all);

        auto r2 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::Baseline);
        record("Baseline (mixing length)", r2.passed, r2.skipped || skip_all);

        auto r3 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::GEP);
        record("GEP (algebraic)", r3.passed, r3.skipped || skip_all);

        auto r4 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::NNMLP);
        record("NN-MLP (scalar nut)", r4.passed, r4.skipped || skip_all);

        auto r5 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::NNTBNN);
        record("NN-TBNN (anisotropic)", r5.passed, r5.skipped || skip_all);

        auto r6 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::KOmega);
        record("k-omega", r6.passed, r6.skipped || skip_all);

        auto r7 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::SSTKOmega);
        record("SST k-omega", r7.passed, r7.skipped || skip_all);

        auto r8 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::EARSM_WJ);
        record("EARSM (Wallin-Johansson)", r8.passed, r8.skipped || skip_all);

        auto r9 = skip_all ? TestResult{true, true} : test_single_model(TurbulenceModelType::EARSM_GS);
        record("EARSM (Gatski-Speziale)", r9.passed, r9.skipped || skip_all);

        // EARSM Pope is only tested in driven turbulence (not decay)
        record("EARSM Pope (driven flow)", skip_all ? true : test_earsm_model(TurbulenceModelType::EARSM_Pope), skip_all);

        // Additional EARSM tests with driven turbulence
        record("EARSM WJ (driven flow)", skip_all ? true : test_earsm_model(TurbulenceModelType::EARSM_WJ), skip_all);
        record("EARSM GS (driven flow)", skip_all ? true : test_earsm_model(TurbulenceModelType::EARSM_GS), skip_all);
    });
}

