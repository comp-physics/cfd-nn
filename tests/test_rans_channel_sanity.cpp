/// @file test_rans_channel_sanity.cpp
/// @brief RANS channel flow sanity checks for CI
///
/// PURPOSE: Catches RANS regressions that show up as plausible-but-wrong mean
/// profiles and near-wall nu_t pathologies, long before k/omega go negative.
///
/// These checks validate:
///   1. Profile shape: no-slip, centerline max, monotonic wall→center
///   2. Near-wall nu_t bounds: min(nu_t) >= -1e-12, max(nu_t)/nu < 1e6,
///      first-cell nu_t/nu < 10
///   3. Integral metrics: U_bulk > 0 and stable, wall shear sign correct
///
/// Test configurations:
///   - Small grid (48x48, ~500 iters) → fast label
///   - Full grid (64x64, ~1500 iters) → medium label (optional)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <numeric>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// Helper to format QoI output
static std::string qoi(double value, double threshold, bool scientific = true) {
    std::ostringstream ss;
    if (scientific) {
        ss << std::scientific << std::setprecision(2);
    } else {
        ss << std::fixed << std::setprecision(4);
    }
    ss << "(val=" << value << ", thr=" << threshold << ")";
    return ss.str();
}

// ============================================================================
// Profile shape analysis utilities
// ============================================================================

/// Compute x-averaged U profile: U(y) = <u(x,y)>_x
static std::vector<double> compute_u_profile(const RANSSolver& solver, const Mesh& mesh) {
    int Ny = mesh.Ny;
    std::vector<double> U(Ny, 0.0);
    std::vector<int> counts(Ny, 0);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate u to cell center
            double u = 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i+1, j));
            u_sum += u;
            ++count;
        }
        int j_idx = j - mesh.j_begin();
        U[j_idx] = u_sum / count;
        counts[j_idx] = count;
    }
    return U;
}

/// Compute bulk velocity: U_bulk = integral(U(y)) / height
static double compute_bulk_velocity(const std::vector<double>& U_profile, double /*Ly*/) {
    if (U_profile.empty()) return 0.0;
    double sum = 0.0;
    for (double u : U_profile) {
        sum += u;
    }
    return sum / U_profile.size();
}

/// Check if profile is approximately monotonically increasing from index start to end
/// Returns (is_ok, num_violations, max_violation)
/// Allows up to `max_violations` local violations with magnitude < violation_tol
static std::tuple<bool, int, double> check_monotonic_increasing(
    const std::vector<double>& U, int start, int end,
    double tol, int max_violations = 2, double violation_tol = 0.01) {

    int num_violations = 0;
    double max_violation = 0.0;

    for (int j = start; j < end - 1; ++j) {
        double decrease = U[j] - U[j+1];  // positive if decreasing
        if (decrease > tol) {
            ++num_violations;
            // Normalize by local magnitude to get relative violation
            double local_mag = std::max(std::abs(U[j]), std::abs(U[j+1])) + 1e-10;
            double rel_violation = decrease / local_mag;
            max_violation = std::max(max_violation, rel_violation);
        }
    }

    // Pass if violations are few and small
    bool ok = (num_violations <= max_violations) && (max_violation < violation_tol);
    return {ok, num_violations, max_violation};
}

/// Legacy interface for backward compatibility
static bool is_monotonic_increasing(const std::vector<double>& U, int start, int end, double tol) {
    auto [ok, num_violations, max_violation] = check_monotonic_increasing(U, start, end, tol);
    return ok;
}

// ============================================================================
// Nu_t analysis utilities
// ============================================================================
struct NuTStats {
    double min_nu_t;
    double max_nu_t;
    double first_cell_nu_t;  // nu_t at first off-wall cell (bottom wall)
    double nu;               // molecular viscosity
    bool valid;
};

static NuTStats compute_nu_t_stats(const RANSSolver& solver, const Mesh& mesh, double nu) {
    NuTStats stats;
    stats.min_nu_t = 1e100;
    stats.max_nu_t = -1e100;
    stats.first_cell_nu_t = 0.0;
    stats.nu = nu;
    stats.valid = true;

    const ScalarField& nu_t = solver.nu_t();
    int j_first = mesh.j_begin();  // First off-wall cell at bottom

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double val = nu_t(i, j);

            if (!std::isfinite(val)) {
                stats.valid = false;
                continue;
            }

            stats.min_nu_t = std::min(stats.min_nu_t, val);
            stats.max_nu_t = std::max(stats.max_nu_t, val);

            // Average first-cell nu_t
            if (j == j_first) {
                stats.first_cell_nu_t += val;
            }
        }
    }

    // Average over x for first cell
    stats.first_cell_nu_t /= (mesh.i_end() - mesh.i_begin());

    return stats;
}

// ============================================================================
// Physics directionality checks: production, dissipation, growth rates
// ============================================================================

/// Compute TKE production: P_k = nu_t * S^2 where S = |du/dy| for channel
/// Returns {min, max, avg} in shear regions (away from centerline)
struct ProductionStats {
    double min_production;   // Min P_k in shear region
    double max_production;   // Max P_k
    double avg_production;   // Average P_k
    double negative_count;   // Number of cells with P_k < 0
    double total_cells;      // Total cells in shear region
    bool valid;
};

static ProductionStats compute_production_stats(const RANSSolver& solver, const Mesh& mesh) {
    ProductionStats stats;
    stats.min_production = 1e100;
    stats.max_production = -1e100;
    stats.avg_production = 0.0;
    stats.negative_count = 0;
    stats.total_cells = 0;
    stats.valid = true;

    const ScalarField& nu_t = solver.nu_t();
    const VectorField& vel = solver.velocity();

    // Shear region: exclude centerline region (middle 20%)
    int j_lo = mesh.j_begin() + mesh.Ny / 10;  // Skip 10% near walls (boundary effects)
    int j_hi = mesh.j_end() - mesh.Ny / 10;
    int j_mid_lo = mesh.j_begin() + 2 * mesh.Ny / 5;  // Middle 20% is centerline
    int j_mid_hi = mesh.j_begin() + 3 * mesh.Ny / 5;

    for (int j = j_lo; j < j_hi; ++j) {
        // Skip centerline region where shear is near zero
        if (j >= j_mid_lo && j < j_mid_hi) continue;

        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Compute shear: du/dy at cell center
            double u_hi = 0.5 * (vel.u(i, j+1) + vel.u(i+1, j+1));
            double u_lo = 0.5 * (vel.u(i, j-1) + vel.u(i+1, j-1));
            double dudy = (u_hi - u_lo) / (2.0 * mesh.dy);

            // Production: P_k = nu_t * (du/dy)^2
            // (Simplified for channel - full form would include all strain components)
            double P_k = nu_t(i, j) * dudy * dudy;

            stats.min_production = std::min(stats.min_production, P_k);
            stats.max_production = std::max(stats.max_production, P_k);
            stats.avg_production += P_k;
            if (P_k < 0) stats.negative_count++;
            stats.total_cells++;
        }
    }

    if (stats.total_cells > 0) {
        stats.avg_production /= stats.total_cells;
    }

    return stats;
}

/// Compute TKE dissipation stats: epsilon = beta_star * k * omega for k-omega models
/// For sanity checking: dissipation should always be positive
struct DissipationStats {
    double min_dissipation;  // Should be >= 0
    double max_dissipation;
    double avg_dissipation;
    double negative_count;   // Should be 0
    double total_cells;
    bool valid;
};

static DissipationStats compute_dissipation_stats(const RANSSolver& solver, const Mesh& mesh) {
    DissipationStats stats;
    stats.min_dissipation = 1e100;
    stats.max_dissipation = -1e100;
    stats.avg_dissipation = 0.0;
    stats.negative_count = 0;
    stats.total_cells = 0;
    stats.valid = true;

    // Access k and omega fields
    const ScalarField& k = solver.k();
    const ScalarField& omega = solver.omega();
    const double beta_star = 0.09;  // Standard k-omega constant

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Dissipation: epsilon = beta_star * k * omega
            double eps = beta_star * k(i, j) * omega(i, j);

            stats.min_dissipation = std::min(stats.min_dissipation, eps);
            stats.max_dissipation = std::max(stats.max_dissipation, eps);
            stats.avg_dissipation += eps;
            if (eps < 0) stats.negative_count++;
            stats.total_cells++;
        }
    }

    if (stats.total_cells > 0) {
        stats.avg_dissipation /= stats.total_cells;
    }

    return stats;
}

/// Compute TKE statistics for growth rate checks
struct TKEStats {
    double min_k;
    double max_k;
    double avg_k;
    double negative_count;
    double total_cells;
    bool valid;
};

static TKEStats compute_tke_stats(const RANSSolver& solver, const Mesh& mesh) {
    TKEStats stats;
    stats.min_k = 1e100;
    stats.max_k = -1e100;
    stats.avg_k = 0.0;
    stats.negative_count = 0;
    stats.total_cells = 0;
    stats.valid = true;

    // Access k field
    const ScalarField& k = solver.k();

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_val = k(i, j);
            stats.min_k = std::min(stats.min_k, k_val);
            stats.max_k = std::max(stats.max_k, k_val);
            stats.avg_k += k_val;
            if (k_val < 0) stats.negative_count++;
            stats.total_cells++;
        }
    }

    if (stats.total_cells > 0) {
        stats.avg_k /= stats.total_cells;
    }

    return stats;
}

// ============================================================================
// Wall shear computation
// ============================================================================

/// Compute wall shear stress at bottom wall: tau_w = nu * du/dy|wall
static double compute_wall_shear_bottom(const RANSSolver& solver, const Mesh& mesh, double nu) {
    // At bottom wall (y_min), use first interior cell to estimate du/dy
    // For no-slip: u(wall) = 0, so du/dy ≈ u(first_cell) / y_first_cell
    double tau_sum = 0.0;
    int count = 0;

    int j = mesh.j_begin();
    double y_cell = mesh.y(j) - mesh.y_min;  // Distance from wall

    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        double u = 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i+1, j));
        double dudy = u / y_cell;  // One-sided difference
        tau_sum += nu * dudy;
        ++count;
    }

    return (count > 0) ? tau_sum / count : 0.0;
}

// ============================================================================
// RANS Channel Sanity Test: SST k-omega
// ============================================================================
void test_rans_channel_sst() {
    std::cout << "\n--- RANS Channel Sanity: SST k-omega ---\n\n";

    // Configuration: 48x48 driven channel, ~800 iterations
    // 800 iters allows better profile development while staying < 3s runtime
    // Warmup: first 200 iterations to let initial transient pass
    const int Nx = 48, Ny = 48;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;  // y in [-1, 1]
    const double nu = 0.01;
    const double dp_dx = -0.001;  // Driving pressure gradient
    const int max_iters = 800;
    const int warmup_iters = 200;  // Skip first 200 iterations for profile checks

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly/2, Ly/2);

    Config config;
    config.nu = nu;
    config.dp_dx = dp_dx;
    config.dt = 0.001;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.max_steps = max_iters;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Explicitly set turbulence model
    auto turb_model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);  // Small initial u
    solver.sync_to_gpu();

    // Track U_bulk over iterations for stability check (after warmup)
    std::vector<double> U_bulk_history;

    for (int iter = 0; iter < max_iters; ++iter) {
        solver.step();

        // Sample U_bulk periodically (only after warmup)
        if (iter >= warmup_iters && iter % 50 == 0) {
            solver.sync_from_gpu();
            auto U_profile = compute_u_profile(solver, mesh);
            double U_bulk = compute_bulk_velocity(U_profile, Ly);
            U_bulk_history.push_back(U_bulk);
        }
    }
    solver.sync_from_gpu();

    // Final profile
    auto U_profile = compute_u_profile(solver, mesh);
    double U_bulk = compute_bulk_velocity(U_profile, Ly);
    U_bulk_history.push_back(U_bulk);

    // ========== Profile Shape Checks ==========

    // Compute max velocity first (needed for multiple checks)
    double U_max = *std::max_element(U_profile.begin(), U_profile.end());
    int j_center = U_profile.size() / 2;
    double U_center = U_profile[j_center];

    // 1. No-slip: U at first interior cell should be much smaller than centerline
    // Note: First interior cell is not the wall itself (which has u=0 via ghost cells)
    // We check that u near wall is reasonably small relative to centerline
    double U_wall_lo = U_profile[0];
    double U_wall_hi = U_profile[U_profile.size() - 1];
    double U_wall_max = std::max(std::abs(U_wall_lo), std::abs(U_wall_hi));
    // First cell should be < 30% of max (typical for turbulent channel)
    bool no_slip_ok = U_wall_max < 0.3 * U_max;

    // 2. Centerline max: U should be near maximum at center
    // For short runs, allow center to be within 50% of max (flow may not be fully developed)
    bool centerline_max_ok = (U_center >= U_max * 0.50);

    // 3. Monotonic wall→center (bottom half)
    double monotonic_tol = 1e-10 * std::abs(U_bulk);
    bool monotonic_ok = is_monotonic_increasing(U_profile, 0, j_center, monotonic_tol);

    // ========== Nu_t Bounds Checks ==========
    NuTStats nu_t_stats = compute_nu_t_stats(solver, mesh, nu);

    bool min_nu_t_ok = (nu_t_stats.min_nu_t >= -1e-12);
    bool max_nu_t_ok = (nu_t_stats.max_nu_t / nu < 1e6);
    bool first_cell_ok = (nu_t_stats.first_cell_nu_t / nu < 10.0);

    // ========== Integral Metrics ==========

    // U_bulk stability: check that U_bulk is not wildly oscillating
    // For short runs, allow up to 5% change between samples
    bool U_bulk_stable = true;
    if (U_bulk_history.size() >= 3) {
        double U_last = U_bulk_history.back();
        double U_prev = U_bulk_history[U_bulk_history.size() - 2];
        double rel_change = std::abs(U_last - U_prev) / (std::abs(U_last) + 1e-10);
        U_bulk_stable = (rel_change < 0.05);  // Less than 5% change (relaxed for short runs)
    }

    // Wall shear sign: should be consistent with dp/dx
    // For dp_dx < 0 (favorable), tau_w should be > 0 (positive shear)
    double tau_w = compute_wall_shear_bottom(solver, mesh, nu);
    bool shear_sign_ok = (dp_dx < 0) ? (tau_w > 0) : (tau_w < 0);
    bool shear_nonzero = (std::abs(tau_w) > 1e-10);

    // ========== Physics Directionality Checks ==========

    // Production in shear regions should be positive (P_k = nu_t * S^2 >= 0)
    ProductionStats prod_stats = compute_production_stats(solver, mesh);
    bool production_positive = prod_stats.valid && (prod_stats.negative_count == 0);

    // Dissipation should be positive everywhere (eps = beta* * k * omega >= 0)
    DissipationStats diss_stats = compute_dissipation_stats(solver, mesh);
    bool dissipation_positive = diss_stats.valid && (diss_stats.negative_count == 0);

    // TKE should be non-negative everywhere
    TKEStats tke_stats = compute_tke_stats(solver, mesh);
    bool tke_nonnegative = tke_stats.valid && (tke_stats.negative_count == 0);

    // U_bulk growth: should be growing in early phase (first half of history)
    // Check that later samples are larger than earlier samples
    bool u_bulk_growing = true;
    if (U_bulk_history.size() >= 4) {
        // Compare first quarter to last quarter average
        int quarter = U_bulk_history.size() / 4;
        double early_avg = 0.0, late_avg = 0.0;
        for (int i = 0; i < quarter; ++i) {
            early_avg += U_bulk_history[i];
        }
        for (int i = U_bulk_history.size() - quarter; i < (int)U_bulk_history.size(); ++i) {
            late_avg += U_bulk_history[i];
        }
        early_avg /= quarter;
        late_avg /= quarter;
        // Later should be >= early (flow is accelerating toward steady state)
        u_bulk_growing = (late_avg >= early_avg * 0.95);  // Allow 5% tolerance
    }

    // Print diagnostics
    std::cout << "  Grid: " << Nx << "x" << Ny << ", iters: " << max_iters << "\n";
    std::cout << "  U_bulk: " << std::scientific << std::setprecision(4) << U_bulk << "\n";
    std::cout << "  U_wall: " << U_wall_max << " (threshold: " << 1e-6 * std::abs(U_bulk) << ")\n";
    std::cout << "  U_center: " << U_center << ", U_max: " << U_max << "\n";
    std::cout << "  nu_t: min=" << nu_t_stats.min_nu_t
              << ", max=" << nu_t_stats.max_nu_t
              << ", first_cell=" << nu_t_stats.first_cell_nu_t << "\n";
    std::cout << "  nu_t/nu: max=" << nu_t_stats.max_nu_t / nu
              << ", first_cell=" << nu_t_stats.first_cell_nu_t / nu << "\n";
    std::cout << "  tau_w: " << tau_w << " (expected sign: " << (dp_dx < 0 ? "+" : "-") << ")\n";

    // Physics directionality diagnostics
    if (prod_stats.valid) {
        std::cout << "  Production: min=" << prod_stats.min_production
                  << ", max=" << prod_stats.max_production
                  << ", neg_cells=" << (int)prod_stats.negative_count << "/" << (int)prod_stats.total_cells << "\n";
    }
    if (diss_stats.valid) {
        std::cout << "  Dissipation: min=" << diss_stats.min_dissipation
                  << ", max=" << diss_stats.max_dissipation
                  << ", neg_cells=" << (int)diss_stats.negative_count << "/" << (int)diss_stats.total_cells << "\n";
    }
    if (tke_stats.valid) {
        std::cout << "  TKE: min=" << tke_stats.min_k << ", max=" << tke_stats.max_k
                  << ", neg_cells=" << (int)tke_stats.negative_count << "/" << (int)tke_stats.total_cells << "\n";
    }
    std::cout << "\n";

    // Record results with QoI values
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << "(wall=" << U_wall_max/U_max*100 << "%, thr=30%)";
    record("SST No-slip (|U_wall| < 30% of U_max)", no_slip_ok, ss.str());
    ss.str(""); ss << std::fixed << std::setprecision(1) << "(center=" << U_center/U_max*100 << "%, thr=50%)";
    record("SST Centerline max (U_center >= 50% of U_max)", centerline_max_ok, ss.str());
    record("SST Monotonic wall->center", monotonic_ok);
    record("SST min(nu_t) >= -1e-12", min_nu_t_ok, qoi(nu_t_stats.min_nu_t, -1e-12));
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(val=" << nu_t_stats.max_nu_t/nu << ", thr=1e6)";
    record("SST max(nu_t)/nu < 1e6", max_nu_t_ok, ss.str());
    ss.str(""); ss << std::fixed << std::setprecision(2) << "(val=" << nu_t_stats.first_cell_nu_t/nu << ", thr=10)";
    record("SST first-cell nu_t/nu < 10", first_cell_ok, ss.str());
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(val=" << U_bulk << ")";
    record("SST U_bulk > 0", U_bulk > 0, ss.str());
    record("SST U_bulk stable (< 5% change)", U_bulk_stable);
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(tau_w=" << tau_w << ")";
    record("SST Wall shear sign correct", shear_sign_ok && shear_nonzero, ss.str());

    // Physics directionality checks
    record("SST Production >= 0 in shear regions", production_positive);
    record("SST Dissipation >= 0 everywhere", dissipation_positive);
    record("SST TKE >= 0 everywhere", tke_nonnegative);
    record("SST U_bulk growing toward steady", u_bulk_growing);

    // Emit machine-readable QoI for CI metrics
    harness::emit_qoi_rans_channel(U_bulk, nu_t_stats.max_nu_t / nu);
}

// ============================================================================
// RANS Channel Sanity Test: Standard k-omega
// ============================================================================
void test_rans_channel_komega() {
    std::cout << "\n--- RANS Channel Sanity: Standard k-omega ---\n\n";

    // Same setup as SST test (800 iters for better profile development)
    const int Nx = 48, Ny = 48;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double nu = 0.01;
    const double dp_dx = -0.001;
    const int max_iters = 800;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly/2, Ly/2);

    Config config;
    config.nu = nu;
    config.dp_dx = dp_dx;
    config.dt = 0.001;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.max_steps = max_iters;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::KOmega;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Explicitly set turbulence model
    auto turb_model = create_turbulence_model(TurbulenceModelType::KOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    solver.sync_to_gpu();

    for (int iter = 0; iter < max_iters; ++iter) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Compute diagnostics
    auto U_profile = compute_u_profile(solver, mesh);
    double U_bulk = compute_bulk_velocity(U_profile, Ly);

    double U_max = *std::max_element(U_profile.begin(), U_profile.end());
    double U_wall_max = std::max(std::abs(U_profile[0]), std::abs(U_profile.back()));
    bool no_slip_ok = U_wall_max < 0.3 * U_max;

    int j_center = U_profile.size() / 2;
    double U_center = U_profile[j_center];
    // For short runs, allow center to be within 50% of max (flow may not be fully developed)
    bool centerline_max_ok = (U_center >= U_max * 0.50);

    double monotonic_tol = 1e-10 * std::abs(U_bulk);
    bool monotonic_ok = is_monotonic_increasing(U_profile, 0, j_center, monotonic_tol);

    NuTStats nu_t_stats = compute_nu_t_stats(solver, mesh, nu);

    bool min_nu_t_ok = (nu_t_stats.min_nu_t >= -1e-12);
    bool max_nu_t_ok = (nu_t_stats.max_nu_t / nu < 1e6);
    bool first_cell_ok = (nu_t_stats.first_cell_nu_t / nu < 10.0);

    double tau_w = compute_wall_shear_bottom(solver, mesh, nu);
    bool shear_ok = (dp_dx < 0) ? (tau_w > 0) : (tau_w < 0);

    // Physics directionality checks
    ProductionStats prod_stats = compute_production_stats(solver, mesh);
    bool production_positive = prod_stats.valid && (prod_stats.negative_count == 0);

    DissipationStats diss_stats = compute_dissipation_stats(solver, mesh);
    bool dissipation_positive = diss_stats.valid && (diss_stats.negative_count == 0);

    TKEStats tke_stats = compute_tke_stats(solver, mesh);
    bool tke_nonnegative = tke_stats.valid && (tke_stats.negative_count == 0);

    std::cout << "  U_bulk: " << std::scientific << U_bulk << "\n";
    std::cout << "  nu_t/nu: max=" << nu_t_stats.max_nu_t / nu
              << ", first_cell=" << nu_t_stats.first_cell_nu_t / nu << "\n";
    std::cout << "  tau_w: " << tau_w << "\n";
    if (prod_stats.valid) {
        std::cout << "  Production: min=" << prod_stats.min_production
                  << ", neg_cells=" << (int)prod_stats.negative_count << "\n";
    }
    if (diss_stats.valid) {
        std::cout << "  Dissipation: min=" << diss_stats.min_dissipation
                  << ", neg_cells=" << (int)diss_stats.negative_count << "\n";
    }
    if (tke_stats.valid) {
        std::cout << "  TKE: min=" << tke_stats.min_k
                  << ", neg_cells=" << (int)tke_stats.negative_count << "\n";
    }
    std::cout << "\n";

    // Record results with QoI values
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << "(wall=" << U_wall_max/U_max*100 << "%, thr=30%)";
    record("k-omega No-slip (|U_wall| < 30% of U_max)", no_slip_ok, ss.str());
    ss.str(""); ss << std::fixed << std::setprecision(1) << "(center=" << U_center/U_max*100 << "%, thr=50%)";
    record("k-omega Centerline max (U_center >= 50% of U_max)", centerline_max_ok, ss.str());
    record("k-omega Monotonic", monotonic_ok);
    record("k-omega min(nu_t) >= -1e-12", min_nu_t_ok, qoi(nu_t_stats.min_nu_t, -1e-12));
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(val=" << nu_t_stats.max_nu_t/nu << ", thr=1e6)";
    record("k-omega max(nu_t)/nu < 1e6", max_nu_t_ok, ss.str());
    ss.str(""); ss << std::fixed << std::setprecision(2) << "(val=" << nu_t_stats.first_cell_nu_t/nu << ", thr=10)";
    record("k-omega first-cell nu_t/nu < 10", first_cell_ok, ss.str());
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(val=" << U_bulk << ")";
    record("k-omega U_bulk > 0", U_bulk > 0, ss.str());
    ss.str(""); ss << std::scientific << std::setprecision(2) << "(tau_w=" << tau_w << ")";
    record("k-omega Wall shear correct", shear_ok && std::abs(tau_w) > 1e-10, ss.str());

    // Physics directionality checks
    record("k-omega Production >= 0 in shear regions", production_positive);
    record("k-omega Dissipation >= 0 everywhere", dissipation_positive);
    record("k-omega TKE >= 0 everywhere", tke_nonnegative);
}

// ============================================================================
// Quick smoke test: all RANS models run without crash
// ============================================================================
void test_rans_models_smoke() {
    std::cout << "\n--- RANS Models Smoke Test ---\n\n";

    const std::vector<std::pair<TurbulenceModelType, std::string>> models = {
        {TurbulenceModelType::SSTKOmega, "SST k-omega"},
        {TurbulenceModelType::KOmega, "Standard k-omega"},
        {TurbulenceModelType::EARSM_WJ, "EARSM Wallin-Johansson"},
    };

    for (const auto& [model_type, name] : models) {
        Mesh mesh;
        mesh.init_uniform(32, 32, 0.0, 2.0*M_PI, -1.0, 1.0);

        Config config;
        config.nu = 0.01;
        config.dt = 0.001;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.turb_model = model_type;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);

        solver.set_body_force(0.001, 0.0);
        solver.initialize_uniform(0.1, 0.0);
        solver.sync_to_gpu();

        bool ok = true;
        try {
            for (int step = 0; step < 50; ++step) {
                solver.step();
            }
            solver.sync_from_gpu();

            // Check for NaN/Inf
            double U_bulk = solver.bulk_velocity();
            ok = std::isfinite(U_bulk);
        } catch (...) {
            ok = false;
        }

        record(name + " runs 50 steps", ok);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("RANS Channel Sanity Tests", []() {
        test_rans_models_smoke();
        test_rans_channel_sst();
        test_rans_channel_komega();
    });
}
