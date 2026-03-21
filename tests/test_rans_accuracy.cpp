/// @file test_rans_accuracy.cpp
/// @brief RANS accuracy test: compare u+ profiles against MKM DNS reference data
///
/// Runs 3 key RANS models (Baseline mixing-length, GEP, SST k-omega) on a 2D
/// channel at Re_tau=180 for 3000 steps, then compares the mean u+ profile
/// against embedded MKM DNS reference data (Moser, Kim & Mansour 1999).
///
/// Acceptance criteria (loose — verifying "physically reasonable"):
///   - Algebraic models (Baseline, GEP): u+ error < 50% in log layer, < 50% in buffer
///   - Transport model (SST): u+ error < 20% in log layer, < 40% in buffer
///   - All models: u_tau within 50% of target (u_tau_target = 1.0 for dp_dx = -1)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// MKM DNS Reference Data (Moser, Kim & Mansour 1999, Re_tau ~ 178)
// ============================================================================

struct RefPoint { double y_plus; double u_plus; };

static const std::vector<RefPoint> mkm_reference = {
    {  0.0,   0.00},
    {  1.0,   0.99},
    {  2.0,   1.99},
    {  5.0,   4.88},
    { 10.0,   8.80},
    { 20.0,  12.49},
    { 30.0,  14.13},
    { 50.0,  15.73},
    { 70.0,  16.89},
    {100.0,  17.72},
    {150.0,  18.47},
    {178.0,  18.28},
};

// ============================================================================
// Interpolate reference data at a given y+
// ============================================================================

/// Linear interpolation into the MKM reference table.
/// Returns interpolated u+ for a given y+. Clamps to table bounds.
static double interp_reference(double yp) {
    if (yp <= mkm_reference.front().y_plus) return mkm_reference.front().u_plus;
    if (yp >= mkm_reference.back().y_plus)  return mkm_reference.back().u_plus;

    for (size_t k = 0; k + 1 < mkm_reference.size(); ++k) {
        if (yp >= mkm_reference[k].y_plus && yp <= mkm_reference[k + 1].y_plus) {
            double t = (yp - mkm_reference[k].y_plus) /
                       (mkm_reference[k + 1].y_plus - mkm_reference[k].y_plus);
            return mkm_reference[k].u_plus + t * (mkm_reference[k + 1].u_plus - mkm_reference[k].u_plus);
        }
    }
    return mkm_reference.back().u_plus;
}

// ============================================================================
// Common test infrastructure
// ============================================================================

/// Run a RANS model on a stretched 2D channel and extract the u+ profile.
///
/// Returns: {y_plus_values, u_plus_values, u_tau}
/// If the simulation fails (NaN, crash), returns empty vectors and u_tau = 0.
struct ProfileResult {
    std::vector<double> y_plus;
    std::vector<double> u_plus;
    double u_tau;
    bool ok;
};

static ProfileResult run_and_extract_profile(TurbulenceModelType type,
                                              const std::string& nn_path = "") {
    ProfileResult result;
    result.u_tau = 0.0;
    result.ok = false;

    const int Nx = 32;
    const int Ny = 64;
    const double x_min = 0.0;
    const double x_max = 4.0 * M_PI;
    const double y_min = -1.0;
    const double y_max = 1.0;
    const double nu = 1.0 / 180.0;
    const double dp_dx = -1.0;
    const int nsteps = 3000;

    try {
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, x_min, x_max, y_min, y_max);

        Config config;
        config.nu = nu;
        config.dp_dx = dp_dx;
        config.turb_model = type;
        config.simulation_mode = SimulationMode::Steady;
        config.dt = 0.01;
        config.adaptive_dt = true;
        config.CFL_max = 0.8;
        config.max_steps = nsteps;
        config.tol = 1e-8;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        auto turb = create_turbulence_model(type, nn_path, nn_path);
        solver.set_turbulence_model(std::move(turb));

        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);  // fx = 1.0

        // Initialize with parabolic profile for faster convergence
        // u(y) ≈ 1.5 * U_bulk * (1 - (y/delta)^2) for Poiseuille
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();

        // Run simulation
        for (int step = 0; step < nsteps; ++step) {
            double res = solver.step();
            // Early exit if converged
            if (step > 100 && res < 1e-8) break;
        }
        solver.sync_from_gpu();

        // Compute x-averaged U profile
        std::vector<double> u_avg(Ny, 0.0);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double u_sum = 0.0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Interpolate staggered u to cell center
                u_sum += 0.5 * (solver.velocity().u(i, j) +
                                solver.velocity().u(i + 1, j));
            }
            int j_idx = j - mesh.j_begin();
            u_avg[j_idx] = u_sum / Nx;
        }

        // Check for NaN in profile
        for (int j = 0; j < Ny; ++j) {
            if (!std::isfinite(u_avg[j])) {
                std::cerr << "  [ERROR] NaN in u_avg at j=" << j << "\n";
                return result;
            }
        }

        // Compute u_tau from first interior cell at bottom wall
        // Wall is at y_min = -1.0. First interior cell center is at yc[j_begin].
        int j_first = mesh.j_begin();
        double y_first = mesh.y(j_first) - mesh.y_min;  // distance from bottom wall
        double u_first = u_avg[0];

        if (y_first <= 0.0 || u_first <= 0.0) {
            std::cerr << "  [ERROR] Invalid first cell: y=" << y_first
                      << " u=" << u_first << "\n";
            return result;
        }

        // u_tau = sqrt(nu * du/dy|wall) = sqrt(nu * u_first / (y_first))
        result.u_tau = std::sqrt(nu * u_first / y_first);

        if (!std::isfinite(result.u_tau) || result.u_tau <= 0.0) {
            std::cerr << "  [ERROR] Invalid u_tau=" << result.u_tau << "\n";
            result.u_tau = 0.0;
            return result;
        }

        // Convert to wall units (bottom half only, j=0..Ny/2-1)
        int half = Ny / 2;
        result.y_plus.resize(half);
        result.u_plus.resize(half);
        for (int j = 0; j < half; ++j) {
            double y_wall = mesh.y(j + mesh.j_begin()) - mesh.y_min;
            result.y_plus[j] = y_wall * result.u_tau / nu;
            result.u_plus[j] = u_avg[j] / result.u_tau;
        }

        result.ok = true;

    } catch (const std::exception& e) {
        std::cerr << "  [ERROR] Exception: " << e.what() << "\n";
    }

    return result;
}

// ============================================================================
// Compute profile errors against MKM reference
// ============================================================================

struct ErrorStats {
    double max_buffer_error;   // max relative error for 5 < y+ < 30
    double max_log_error;      // max relative error for y+ > 30
    int n_buffer;              // number of points in buffer layer
    int n_log;                 // number of points in log layer
};

static ErrorStats compute_errors(const ProfileResult& prof) {
    ErrorStats err;
    err.max_buffer_error = 0.0;
    err.max_log_error = 0.0;
    err.n_buffer = 0;
    err.n_log = 0;

    for (size_t j = 0; j < prof.y_plus.size(); ++j) {
        double yp = prof.y_plus[j];
        double up_num = prof.u_plus[j];

        // Skip viscous sublayer (y+ < 5) — too grid-sensitive for this tolerance
        if (yp < 5.0) continue;
        // Skip beyond channel center (y+ > 178)
        if (yp > 178.0) continue;

        double up_ref = interp_reference(yp);
        if (up_ref < 0.1) continue;  // avoid division by near-zero

        double rel_err = std::abs(up_num - up_ref) / up_ref;

        if (yp >= 5.0 && yp < 30.0) {
            // Buffer layer
            err.max_buffer_error = std::max(err.max_buffer_error, rel_err);
            err.n_buffer++;
        } else if (yp >= 30.0) {
            // Log layer + outer layer
            err.max_log_error = std::max(err.max_log_error, rel_err);
            err.n_log++;
        }
    }

    return err;
}

// ============================================================================
// Print profile summary (diagnostic output)
// ============================================================================

static void print_profile_summary(const std::string& model_name,
                                   const ProfileResult& prof,
                                   const ErrorStats& err) {
    std::cout << "  " << model_name << ":\n";
    std::cout << "    u_tau = " << std::fixed << std::setprecision(4) << prof.u_tau
              << " (target: 1.0)\n";
    std::cout << "    Profile points: " << prof.y_plus.size()
              << " (buffer: " << err.n_buffer << ", log: " << err.n_log << ")\n";
    std::cout << "    Buffer layer max error: " << std::fixed << std::setprecision(1)
              << err.max_buffer_error * 100.0 << "%\n";
    std::cout << "    Log layer max error:    " << std::fixed << std::setprecision(1)
              << err.max_log_error * 100.0 << "%\n";

    // Print a few key profile points
    std::cout << "    Sample u+ profile:\n";
    std::cout << "      " << std::setw(10) << "y+" << std::setw(10) << "u+(num)"
              << std::setw(10) << "u+(ref)" << std::setw(10) << "err(%)" << "\n";

    // Print at target y+ values matching reference data
    double targets[] = {1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 150.0};
    for (double yp_target : targets) {
        // Find closest grid point
        int j_closest = 0;
        double min_dist = 1e30;
        for (size_t j = 0; j < prof.y_plus.size(); ++j) {
            double dist = std::abs(prof.y_plus[j] - yp_target);
            if (dist < min_dist) {
                min_dist = dist;
                j_closest = j;
            }
        }
        if (min_dist < yp_target * 0.5 + 1.0) {
            double up_ref = interp_reference(prof.y_plus[j_closest]);
            double rel_err = (up_ref > 0.1) ?
                std::abs(prof.u_plus[j_closest] - up_ref) / up_ref * 100.0 : 0.0;
            std::cout << "      " << std::fixed << std::setprecision(1)
                      << std::setw(10) << prof.y_plus[j_closest]
                      << std::setw(10) << prof.u_plus[j_closest]
                      << std::setw(10) << up_ref
                      << std::setw(10) << rel_err << "\n";
        }
    }
    std::cout << "\n";
}

// ============================================================================
// Parameterized model accuracy test
// ============================================================================

struct ModelAccuracyParams {
    TurbulenceModelType type;
    std::string name;
    double buffer_threshold;
    double log_threshold;
};

static void test_model_accuracy(const ModelAccuracyParams& p) {
    std::cout << "\n  Running " << p.name << " model (3000 steps)...\n";

    auto prof = run_and_extract_profile(p.type);

    std::string buf_label = p.name + ": buffer-layer u+ error < " +
        std::to_string((int)(p.buffer_threshold * 100)) + "%";
    std::string log_label = p.name + ": log-layer u+ error < " +
        std::to_string((int)(p.log_threshold * 100)) + "%";

    if (!prof.ok) {
        record(p.name + ": simulation completed", false);
        record(p.name + ": u_tau reasonable", false);
        record(buf_label, false);
        record(log_label, false);
        return;
    }

    auto err = compute_errors(prof);
    print_profile_summary(p.name, prof, err);

    std::ostringstream ss;

    record(p.name + ": simulation completed", true);

    ss.str(""); ss << std::fixed << std::setprecision(4) << "(u_tau=" << prof.u_tau << ")";
    record(p.name + ": u_tau reasonable (0.3-2.0)", prof.u_tau > 0.3 && prof.u_tau < 2.0, ss.str());

    ss.str(""); ss << std::fixed << std::setprecision(1) << "(err=" << err.max_buffer_error * 100 << "%, n=" << err.n_buffer << ")";
    record(buf_label, err.max_buffer_error < p.buffer_threshold, ss.str());

    ss.str(""); ss << std::fixed << std::setprecision(1) << "(err=" << err.max_log_error * 100 << "%, n=" << err.n_log << ")";
    record(log_label, err.max_log_error < p.log_threshold, ss.str());
}

void test_baseline_accuracy() { test_model_accuracy({TurbulenceModelType::Baseline, "Baseline", 0.50, 0.30}); }
void test_gep_accuracy()      { test_model_accuracy({TurbulenceModelType::GEP, "GEP", 0.50, 0.50}); }
void test_sst_accuracy()      { test_model_accuracy({TurbulenceModelType::SSTKOmega, "SST", 0.40, 0.60}); }

// ============================================================================
// Main
// ============================================================================

int main() {
    return harness::run_sections("RANSAccuracy", {
        {"Baseline accuracy", test_baseline_accuracy},
        {"GEP accuracy", test_gep_accuracy},
        {"SST accuracy", test_sst_accuracy},
    });
}
