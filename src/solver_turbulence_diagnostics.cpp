#include "solver.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace nncfd {

// ============================================================================
// Friction velocity from forcing (exact for fully-developed channel)
// ============================================================================

double RANSSolver::u_tau_from_forcing() const {
    // For channel with half-height delta and body force f_x = -dp/dx:
    // tau_w = delta * |dp/dx| (momentum balance)
    // u_tau = sqrt(tau_w / rho) = sqrt(delta * |dp/dx|)
    // During ramp: use TARGET forcing for consistent reference
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    double fx_ref = fx_;
    if (force_ramp_enabled_) {
        fx_ref = fx_target_;  // Use target during ramp for consistent reference
    }
    return std::sqrt(delta * std::abs(fx_ref));
}

// ============================================================================
// Re_tau targeting helpers
// ============================================================================

double RANSSolver::Re_tau_from_forcing() const {
    // Re_tau = u_tau * delta / nu
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    double u_tau = u_tau_from_forcing();
    return u_tau * delta / config_.nu;
}

// Static helper: compute nu for target Re_tau given dp/dx and delta
double RANSSolver::nu_for_Re_tau(double Re_tau_target, double dp_dx, double delta) {
    // u_tau = sqrt(delta * |dp/dx|)
    // Re_tau = u_tau * delta / nu
    // => nu = u_tau * delta / Re_tau = sqrt(delta * |dp/dx|) * delta / Re_tau
    double u_tau = std::sqrt(delta * std::abs(dp_dx));
    return u_tau * delta / Re_tau_target;
}

// Static helper: compute dp/dx for target Re_tau given nu and delta
double RANSSolver::dp_dx_for_Re_tau(double Re_tau_target, double nu, double delta) {
    // Re_tau = u_tau * delta / nu
    // u_tau = Re_tau * nu / delta
    // u_tau^2 = delta * |dp/dx|
    // => |dp/dx| = u_tau^2 / delta = (Re_tau * nu / delta)^2 / delta
    double u_tau = Re_tau_target * nu / delta;
    return -(u_tau * u_tau) / delta;  // Negative for driving force in +x
}

// ============================================================================
// 2nd-order wall shear stress using quadratic fit
// ============================================================================

double RANSSolver::wall_shear_stress_2nd_order(bool bottom) const {
    // Use quadratic fit through wall (u=0) and first two cell centers
    // dU/dy|_w = (u1 * y2^2 - u2 * y1^2) / (y1 * y2 * (y2 - y1))
    //
    // This is 2nd-order accurate on non-uniform grids

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    // Determine wall location and cell indices
    double y_wall, y1, y2;
    int j1, j2;

    if (bottom) {
        y_wall = mesh_->y_min;
        j1 = Ng;          // First interior cell
        j2 = Ng + 1;      // Second interior cell
        y1 = mesh_->yc[j1] - y_wall;
        y2 = mesh_->yc[j2] - y_wall;
    } else {
        y_wall = mesh_->y_max;
        j1 = Ng + Ny - 1; // Last interior cell
        j2 = Ng + Ny - 2; // Second-to-last interior cell
        y1 = y_wall - mesh_->yc[j1];
        y2 = y_wall - mesh_->yc[j2];
    }

    double dudy_sum = 0.0;
    int count = 0;

    // DEBUG: track NaN appearances
    static int debug_count = 0;
    static bool nan_detected = false;

    // Average over x (and z for 3D)
    if (Nz > 1) {
        // 3D case
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double u1 = velocity_.u(i, j1, k);
                double u2 = velocity_.u(i, j2, k);

                // DEBUG: output first few calls AND when NaN first appears
                if (bottom && i == Ng && k == Ng) {
                    bool is_nan = std::isnan(u1) || std::isnan(u2);
                    if (debug_count < 5 || (is_nan && !nan_detected)) {
                        std::cerr << "[DEBUG wall_shear #" << debug_count << "] Nx=" << Nx
                                  << " Ny=" << Ny << " Nz=" << Nz << " Ng=" << Ng
                                  << " j1=" << j1 << " j2=" << j2
                                  << " y1=" << y1 << " y2=" << y2
                                  << " u1=" << u1 << " u2=" << u2
                                  << " y_wall=" << y_wall << " yc[j1]=" << mesh_->yc[j1]
                                  << " yc[j2]=" << mesh_->yc[j2];
                        if (is_nan) std::cerr << " *** NAN DETECTED ***";
                        std::cerr << "\n";
                        if (is_nan) nan_detected = true;
                    }
                    ++debug_count;
                }

                // Quadratic fit: dU/dy|_w = (u1*y2^2 - u2*y1^2) / (y1*y2*(y2-y1))
                double dudy = (u1 * y2 * y2 - u2 * y1 * y1) / (y1 * y2 * (y2 - y1));
                dudy_sum += dudy;
                ++count;
            }
        }
    } else {
        // 2D case
        for (int i = Ng; i < Ng + Nx; ++i) {
            double u1 = velocity_.u(i, j1);
            double u2 = velocity_.u(i, j2);

            double dudy = (u1 * y2 * y2 - u2 * y1 * y1) / (y1 * y2 * (y2 - y1));
            dudy_sum += dudy;
            ++count;
        }
    }

    double dudy_avg = dudy_sum / count;

    // tau_w = mu * dU/dy = rho * nu * dU/dy (with rho=1)
    // Sign: for bottom wall, positive dudy means positive tau_w
    // For top wall, negative dudy (in our coord system) means positive tau_w
    return config_.nu * std::abs(dudy_avg);
}

double RANSSolver::friction_velocity_2nd_order(bool bottom) const {
    double tau_w = wall_shear_stress_2nd_order(bottom);
    return std::sqrt(tau_w);  // u_tau = sqrt(tau_w / rho) with rho=1
}

// ============================================================================
// Resolution diagnostics
// ============================================================================

RANSSolver::ResolutionDiagnostics RANSSolver::compute_resolution_diagnostics() const {
    ResolutionDiagnostics diag;

    // u_tau from forcing (exact reference)
    diag.u_tau_force = u_tau_from_forcing();

    // u_tau from 2nd-order wall shear at both walls
    diag.u_tau_bot = friction_velocity_2nd_order(true);
    diag.u_tau_top = friction_velocity_2nd_order(false);

    // Use average u_tau for resolution estimates
    double u_tau = 0.5 * (diag.u_tau_bot + diag.u_tau_top);
    if (u_tau < 1e-12) u_tau = diag.u_tau_force;  // Fallback

    double nu = config_.nu;

    // y1+ at bottom wall
    const int Ng = mesh_->Nghost;
    double y1_bot = mesh_->yc[Ng] - mesh_->y_min;
    diag.y1_plus_bot = y1_bot * diag.u_tau_bot / nu;

    // y1+ at top wall
    double y1_top = mesh_->y_max - mesh_->yc[Ng + mesh_->Ny - 1];
    diag.y1_plus_top = y1_top * diag.u_tau_top / nu;

    // dx+ and dz+
    diag.dx_plus = mesh_->dx * u_tau / nu;
    diag.dz_plus = (mesh_->Nz > 1) ? mesh_->dz * u_tau / nu : 0.0;

    return diag;
}

// ============================================================================
// Statistics accumulation (for time-averaging)
// ============================================================================

void RANSSolver::reset_statistics() {
    const int Ny = mesh_->Ny;

    stats_samples_ = 0;
    stats_U_mean_.assign(Ny, 0.0);
    stats_uu_.assign(Ny, 0.0);
    stats_vv_.assign(Ny, 0.0);
    stats_ww_.assign(Ny, 0.0);
    stats_uv_.assign(Ny, 0.0);
    stats_dUdy_.assign(Ny, 0.0);
}

void RANSSolver::accumulate_statistics() {
    // Sync velocity data from GPU to CPU before computing statistics
    // Without this, CPU reads stale initial-condition data (Poiseuille)
    if (gpu_ready_) {
        sync_solution_from_gpu();
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    // Initialize on first call
    if (stats_samples_ == 0) {
        stats_U_mean_.assign(Ny, 0.0);
        stats_uu_.assign(Ny, 0.0);
        stats_vv_.assign(Ny, 0.0);
        stats_ww_.assign(Ny, 0.0);
        stats_uv_.assign(Ny, 0.0);
        stats_dUdy_.assign(Ny, 0.0);
    }

    // For each y-location, compute plane-averaged quantities
    for (int jj = 0; jj < Ny; ++jj) {
        int j = jj + Ng;

        double U_sum = 0.0, V_sum = 0.0, W_sum = 0.0;
        double uu_sum = 0.0, vv_sum = 0.0, ww_sum = 0.0, uv_sum = 0.0;
        int count = 0;

        if (Nz > 1) {
            // 3D: average over x and z
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    double u = velocity_.u(i, j, k);
                    // v at y-faces: average to cell center
                    double v = 0.5 * (velocity_.v(i, j, k) + velocity_.v(i, j+1, k));
                    double w = velocity_.w(i, j, k);

                    U_sum += u;
                    V_sum += v;
                    W_sum += w;
                    ++count;
                }
            }
        } else {
            // 2D: average over x only
            for (int i = Ng; i < Ng + Nx; ++i) {
                double u = velocity_.u(i, j);
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));

                U_sum += u;
                V_sum += v;
                ++count;
            }
        }

        double U_mean = U_sum / count;
        double V_mean = V_sum / count;
        double W_mean = (Nz > 1) ? W_sum / count : 0.0;

        // Second pass: compute fluctuations
        uu_sum = vv_sum = ww_sum = uv_sum = 0.0;

        if (Nz > 1) {
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    double u_prime = velocity_.u(i, j, k) - U_mean;
                    double v = 0.5 * (velocity_.v(i, j, k) + velocity_.v(i, j+1, k));
                    double v_prime = v - V_mean;
                    double w_prime = velocity_.w(i, j, k) - W_mean;

                    uu_sum += u_prime * u_prime;
                    vv_sum += v_prime * v_prime;
                    ww_sum += w_prime * w_prime;
                    uv_sum += u_prime * v_prime;
                }
            }
        } else {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double u_prime = velocity_.u(i, j) - U_mean;
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                double v_prime = v - V_mean;

                uu_sum += u_prime * u_prime;
                vv_sum += v_prime * v_prime;
                uv_sum += u_prime * v_prime;
            }
        }

        // Accumulate running averages
        double n = static_cast<double>(stats_samples_);
        stats_U_mean_[jj] = (stats_U_mean_[jj] * n + U_mean) / (n + 1.0);
        stats_uu_[jj] = (stats_uu_[jj] * n + uu_sum / count) / (n + 1.0);
        stats_vv_[jj] = (stats_vv_[jj] * n + vv_sum / count) / (n + 1.0);
        stats_ww_[jj] = (stats_ww_[jj] * n + ww_sum / count) / (n + 1.0);
        stats_uv_[jj] = (stats_uv_[jj] * n + uv_sum / count) / (n + 1.0);
    }

    // Compute dU/dy using central differences
    const double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    for (int jj = 0; jj < Ny; ++jj) {
        double dUdy;
        if (jj == 0) {
            // One-sided at bottom
            double dy = mesh_->yc[Ng + 1] - mesh_->yc[Ng];
            dUdy = (stats_U_mean_[1] - stats_U_mean_[0]) / dy;
        } else if (jj == Ny - 1) {
            // One-sided at top
            double dy = mesh_->yc[Ng + Ny - 1] - mesh_->yc[Ng + Ny - 2];
            dUdy = (stats_U_mean_[Ny - 1] - stats_U_mean_[Ny - 2]) / dy;
        } else {
            // Central difference
            double dy = mesh_->yc[Ng + jj + 1] - mesh_->yc[Ng + jj - 1];
            dUdy = (stats_U_mean_[jj + 1] - stats_U_mean_[jj - 1]) / dy;
        }

        double n = static_cast<double>(stats_samples_);
        stats_dUdy_[jj] = (stats_dUdy_[jj] * n + dUdy) / (n + 1.0);
    }

    ++stats_samples_;
}

// ============================================================================
// Momentum balance diagnostics
// ============================================================================

RANSSolver::MomentumBalanceDiagnostics RANSSolver::compute_momentum_balance() const {
    MomentumBalanceDiagnostics diag;

    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const double nu = config_.nu;
    const double delta = (mesh_->y_max - mesh_->y_min) / 2.0;

    // Use forcing-based u_tau as reference (less noise)
    double u_tau = u_tau_from_forcing();
    double u_tau_sq = u_tau * u_tau;

    diag.y.resize(Ny);
    diag.y_plus.resize(Ny);
    diag.tau_visc.resize(Ny);
    diag.tau_reynolds.resize(Ny);
    diag.tau_total.resize(Ny);
    diag.tau_theory.resize(Ny);
    diag.residual.resize(Ny);

    for (int jj = 0; jj < Ny; ++jj) {
        // Distance from wall (using bottom wall as reference)
        double y_phys = mesh_->yc[jj + Ng] - mesh_->y_min;

        diag.y[jj] = y_phys;
        diag.y_plus[jj] = y_phys * u_tau / nu;

        // Viscous stress: nu * dU/dy
        double dUdy = (stats_samples_ > 0) ? stats_dUdy_[jj] : 0.0;
        diag.tau_visc[jj] = nu * dUdy;

        // Reynolds stress: -<u'v'> (note: stats_uv_ is <u'v'>, need negative)
        diag.tau_reynolds[jj] = (stats_samples_ > 0) ? -stats_uv_[jj] : 0.0;

        // Total shear stress
        diag.tau_total[jj] = diag.tau_visc[jj] + diag.tau_reynolds[jj];

        // Theoretical linear profile: tau = u_tau^2 * (1 - y/delta)
        // This is valid for y in [0, delta] (lower half)
        // For y in [delta, 2*delta], use symmetry: tau = u_tau^2 * (1 - (2*delta - y)/delta)
        double y_norm = y_phys / delta;  // Normalized distance from wall
        if (y_norm <= 1.0) {
            diag.tau_theory[jj] = u_tau_sq * (1.0 - y_norm);
        } else {
            // Upper half: reflect
            diag.tau_theory[jj] = u_tau_sq * (1.0 - (2.0 - y_norm));
        }

        // Residual
        diag.residual[jj] = diag.tau_total[jj] - diag.tau_theory[jj];
    }

    return diag;
}

// ============================================================================
// Reynolds stress profiles
// ============================================================================

RANSSolver::ReynoldsStressProfiles RANSSolver::compute_reynolds_stress_profiles() const {
    ReynoldsStressProfiles prof;

    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const double nu = config_.nu;

    // Use forcing-based u_tau for normalization
    double u_tau = u_tau_from_forcing();
    double u_tau_sq = u_tau * u_tau;

    prof.y_plus.resize(Ny);
    prof.uu_plus.resize(Ny);
    prof.vv_plus.resize(Ny);
    prof.ww_plus.resize(Ny);
    prof.uv_plus.resize(Ny);

    for (int jj = 0; jj < Ny; ++jj) {
        double y_phys = mesh_->yc[jj + Ng] - mesh_->y_min;
        prof.y_plus[jj] = y_phys * u_tau / nu;

        if (stats_samples_ > 0 && u_tau_sq > 1e-12) {
            prof.uu_plus[jj] = stats_uu_[jj] / u_tau_sq;
            prof.vv_plus[jj] = stats_vv_[jj] / u_tau_sq;
            prof.ww_plus[jj] = stats_ww_[jj] / u_tau_sq;
            prof.uv_plus[jj] = -stats_uv_[jj] / u_tau_sq;  // Note: -<u'v'> is positive
        } else {
            prof.uu_plus[jj] = 0.0;
            prof.vv_plus[jj] = 0.0;
            prof.ww_plus[jj] = 0.0;
            prof.uv_plus[jj] = 0.0;
        }
    }

    return prof;
}

bool RANSSolver::ReynoldsStressProfiles::passes_stress_ordering() const {
    // Check that <u'u'> > <w'w'> > <v'v'> for most y locations
    // (This is a characteristic of channel turbulence)
    int violations = 0;
    int valid_points = 0;

    for (size_t i = 0; i < y_plus.size(); ++i) {
        // Only check in the buffer/log layer (10 < y+ < 100)
        if (y_plus[i] > 10.0 && y_plus[i] < 100.0) {
            ++valid_points;
            // Allow small tolerance for numerical noise
            if (uu_plus[i] < ww_plus[i] - 0.1 || ww_plus[i] < vv_plus[i] - 0.1) {
                ++violations;
            }
        }
    }

    // Pass if < 20% violations
    return (valid_points == 0) || (violations < 0.2 * valid_points);
}

bool RANSSolver::ReynoldsStressProfiles::passes_uv_shape() const {
    // Check that -<u'v'>+ is:
    // 1. Near zero at walls
    // 2. Positive in the interior
    // 3. Has reasonable magnitude (order 1)

    if (y_plus.empty()) return false;

    // Find values near walls (y+ < 5) and in interior (y+ ~ 30-50)
    double uv_wall = 0.0;
    double uv_interior = 0.0;
    int wall_count = 0, interior_count = 0;

    for (size_t i = 0; i < y_plus.size(); ++i) {
        if (y_plus[i] < 5.0) {
            uv_wall += std::abs(uv_plus[i]);
            ++wall_count;
        } else if (y_plus[i] > 30.0 && y_plus[i] < 50.0) {
            uv_interior += uv_plus[i];
            ++interior_count;
        }
    }

    if (wall_count > 0) uv_wall /= wall_count;
    if (interior_count > 0) uv_interior /= interior_count;

    // Wall value should be small, interior should be positive and O(1)
    bool wall_ok = (wall_count == 0) || (uv_wall < 0.2);
    bool interior_ok = (interior_count == 0) || (uv_interior > 0.3 && uv_interior < 1.5);

    return wall_ok && interior_ok;
}

// ============================================================================
// Spanwise spectrum
// ============================================================================

RANSSolver::SpanwiseSpectrum RANSSolver::compute_spanwise_spectrum(double y_plus_target) const {
    SpanwiseSpectrum spec;

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double nu = config_.nu;
    const double Lz = mesh_->z_max - mesh_->z_min;

    if (Nz <= 1) {
        // 2D simulation - no spanwise spectrum
        return spec;
    }

    double u_tau = u_tau_from_forcing();

    // Find j-index closest to y_plus_target
    int j_target = Ng;
    double min_diff = std::abs(mesh_->yc[Ng] * u_tau / nu - y_plus_target);
    for (int jj = 0; jj < Ny; ++jj) {
        double y_plus = (mesh_->yc[jj + Ng] - mesh_->y_min) * u_tau / nu;
        double diff = std::abs(y_plus - y_plus_target);
        if (diff < min_diff) {
            min_diff = diff;
            j_target = jj + Ng;
        }
    }

    // Initialize spectrum
    int Nk = Nz / 2 + 1;  // Number of unique wavenumbers
    spec.k_z.resize(Nk);
    spec.E_uu.assign(Nk, 0.0);

    // Wavenumbers
    for (int ik = 0; ik < Nk; ++ik) {
        spec.k_z[ik] = 2.0 * M_PI * ik / Lz;
    }

    // For each x-location, compute 1D FFT in z and accumulate power spectrum
    // Using simple DFT (for robustness - could use FFTW for performance)
    std::vector<double> u_line(Nz);
    std::vector<double> power(Nk, 0.0);

    for (int i = Ng; i < Ng + Nx; ++i) {
        // Extract u along z at (i, j_target)
        for (int k = 0; k < Nz; ++k) {
            u_line[k] = velocity_.u(i, j_target, k + Ng);
        }

        // Remove mean
        double mean = std::accumulate(u_line.begin(), u_line.end(), 0.0) / Nz;
        for (double& u : u_line) u -= mean;

        // DFT to get power spectrum
        for (int ik = 0; ik < Nk; ++ik) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < Nz; ++k) {
                double phase = 2.0 * M_PI * ik * k / Nz;
                re += u_line[k] * std::cos(phase);
                im -= u_line[k] * std::sin(phase);
            }
            // Power = |FFT|^2 / N^2, normalized to preserve Parseval
            power[ik] = (re * re + im * im) / (Nz * Nz);
        }

        // Accumulate
        for (int ik = 0; ik < Nk; ++ik) {
            spec.E_uu[ik] += power[ik];
        }
    }

    // Average over x
    for (double& e : spec.E_uu) {
        e /= Nx;
    }

    return spec;
}

bool RANSSolver::SpanwiseSpectrum::has_recirculation_spike(double /*x_recycle*/,
                                                            double /*U_bulk*/,
                                                            double tol) const {
    if (k_z.empty() || E_uu.empty()) return false;

    // Recirculation timescale: tau = x_recycle / U_bulk
    // Corresponding frequency spike would appear if there's inlet memory
    // This is more of a time-series check, but we can look for narrow peaks

    // Find the peak energy and check if any single wavenumber has >> average
    double E_mean = std::accumulate(E_uu.begin(), E_uu.end(), 0.0) / E_uu.size();

    for (size_t i = 1; i < E_uu.size() - 1; ++i) {  // Skip k=0 and Nyquist
        if (E_uu[i] > tol * E_mean) {
            // Check if it's a narrow peak (not a broad feature)
            double local_avg = (E_uu[i-1] + E_uu[i+1]) / 2.0;
            if (E_uu[i] > 3.0 * local_avg) {
                return true;  // Narrow spike detected
            }
        }
    }

    return false;
}

bool RANSSolver::SpanwiseSpectrum::has_aliasing_pileup(double tol) const {
    if (E_uu.size() < 4) return false;

    // Check if energy piles up at high wavenumbers (near Nyquist)
    // Compare last few modes to the bulk of the spectrum

    size_t n = E_uu.size();

    // Average of mid-range modes
    double E_mid = 0.0;
    int mid_count = 0;
    for (size_t i = n / 4; i < 3 * n / 4; ++i) {
        E_mid += E_uu[i];
        ++mid_count;
    }
    E_mid /= mid_count;

    // Average of last few modes (near Nyquist)
    double E_high = 0.0;
    int high_count = 0;
    for (size_t i = n - 3; i < n; ++i) {
        E_high += E_uu[i];
        ++high_count;
    }
    E_high /= high_count;

    // Aliasing pileup if high-k energy exceeds mid-k energy significantly
    return E_high > tol * E_mid;
}

// ============================================================================
// Full turbulence realism validation
// ============================================================================

RANSSolver::TurbulenceRealismReport RANSSolver::validate_turbulence_realism() const {
    TurbulenceRealismReport report;

    // Resolution diagnostics
    report.resolution = compute_resolution_diagnostics();
    report.resolution_ok = report.resolution.passes_resolution_gates();
    report.utau_consistency_ok = report.resolution.passes_utau_consistency(0.02);

    // Turbulence presence indicators
    report.presence = compute_turbulence_presence();

    // Momentum balance
    report.momentum_balance = compute_momentum_balance();
    double u_tau = report.resolution.u_tau_force;
    report.momentum_closure_ok = (stats_samples_ > 0) &&
                                  (report.momentum_balance.max_residual_normalized(u_tau) <= 0.02);

    // Reynolds stress profiles
    report.stress_profiles = compute_reynolds_stress_profiles();
    report.stress_shape_ok = report.stress_profiles.passes_stress_ordering() &&
                             report.stress_profiles.passes_uv_shape();

    // Spectral check (at y+ ~ 15)
    if (mesh_->Nz > 1) {
        auto spec = compute_spanwise_spectrum(15.0);
        double U_bulk = bulk_velocity();
        double x_recycle = (use_recycling_) ? mesh_->xc[recycle_i_] : mesh_->x_max;
        report.spectrum_ok = !spec.has_recirculation_spike(x_recycle, U_bulk) &&
                             !spec.has_aliasing_pileup();
    } else {
        report.spectrum_ok = true;  // Skip for 2D
    }

    // Turbulence presence check
    report.turbulence_present_ok = report.presence.is_turbulent_or_transitional();

    return report;
}

void RANSSolver::TurbulenceRealismReport::print() const {
    std::cout << "\n=== Turbulence Realism Validation Report ===\n";
    std::cout << "Mode: " << (mode == ValidationMode::Quick ? "QUICK (machinery validation)" : "FULL (DNS realism)") << "\n";

    std::cout << "\n--- Trust Region ---\n";
    std::cout << "  x ∈ [" << std::fixed << std::setprecision(3) << presence.trust_x_min
              << ", " << presence.trust_x_max << "]\n";
    std::cout << "  (indices " << presence.trust_x_start << " to " << presence.trust_x_end << ")\n";

    std::cout << "\n--- Turbulence Presence ---\n";
    std::cout << "  State:           " << presence.state_string() << "\n";
    std::cout << "  u_tau/u_tau_ref: " << std::fixed << std::setprecision(3) << presence.u_tau_ratio << "\n";
    std::cout << "  u_rms_mid:       " << std::scientific << std::setprecision(2) << presence.u_rms_mid << "\n";
    std::cout << "  TKE_mid:         " << presence.tke_mid << std::fixed << "\n";
    std::cout << "  max(-<u'v'>+):   " << std::fixed << std::setprecision(3) << presence.max_uv_plus << "\n";
    std::cout << "  Turbulent:       " << (turbulence_present_ok ? "YES" : "NO") << "\n";

    std::cout << "\n--- Resolution Diagnostics ---\n";
    std::cout << "  y1+ (bottom): " << std::fixed << std::setprecision(3)
              << resolution.y1_plus_bot << " (target: 0.3-0.8, max: 1.0)\n";
    std::cout << "  y1+ (top):    " << resolution.y1_plus_top << "\n";
    std::cout << "  dx+:          " << resolution.dx_plus << " (max: 15)\n";
    std::cout << "  dz+:          " << resolution.dz_plus << " (max: 8)\n";
    if (mode == ValidationMode::Quick) {
        std::cout << "  Resolution:   SKIPPED (Quick mode)\n";
    } else {
        std::cout << "  Resolution:   " << (resolution_ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\n--- u_tau Consistency ---\n";
    std::cout << "  u_tau (force): " << std::setprecision(5) << resolution.u_tau_force << "\n";
    std::cout << "  u_tau (bot):   " << resolution.u_tau_bot
              << " (err: " << std::setprecision(2)
              << 100.0 * std::abs(resolution.u_tau_bot - resolution.u_tau_force) / (resolution.u_tau_force + 1e-12) << "%)\n";
    std::cout << "  u_tau (top):   " << resolution.u_tau_top
              << " (err: " << 100.0 * std::abs(resolution.u_tau_top - resolution.u_tau_force) / (resolution.u_tau_force + 1e-12) << "%)\n";
    std::cout << "  Tolerance:     " << (mode == ValidationMode::Quick ? "20%" : "2%") << "\n";
    std::cout << "  Consistency:   " << (utau_consistency_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Momentum Balance ---\n";
    double u_tau = resolution.u_tau_force;
    double closure_tol = (mode == ValidationMode::Quick) ? QUICK_CLOSURE_TOL : FULL_CLOSURE_TOL;
    std::cout << "  Max |R|/u_tau^2: " << std::setprecision(3)
              << momentum_balance.max_residual_normalized(u_tau) * 100.0 << "% (max: " << closure_tol * 100.0 << "%)\n";
    std::cout << "  L2 R/u_tau^2:    " << momentum_balance.l2_residual_normalized(u_tau) * 100.0 << "%\n";
    std::cout << "  Closure:         " << (momentum_closure_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Reynolds Stress Shape ---\n";
    if (mode == ValidationMode::Quick) {
        std::cout << "  Shape checks:    SKIPPED (Quick mode)\n";
    } else {
        std::cout << "  Stress ordering: " << (stress_profiles.passes_stress_ordering() ? "PASS" : "FAIL") << "\n";
        std::cout << "  -<u'v'>+ shape:  " << (stress_profiles.passes_uv_shape() ? "PASS" : "FAIL") << "\n";
        std::cout << "  Overall shape:   " << (stress_shape_ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\n--- Spectral Check ---\n";
    std::cout << "  Spectrum:        " << (spectrum_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n=== OVERALL: " << (passes_all() ? "PASS" : "FAIL") << " ===\n\n";
}

// ============================================================================
// Turbulence presence indicators (robust detection)
// ============================================================================

RANSSolver::TurbulencePresenceIndicators RANSSolver::compute_turbulence_presence(
    int trust_x_start, int trust_x_end) const {

    TurbulencePresenceIndicators ind;

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    const double nu = config_.nu;

    // Use forcing-based u_tau as reference
    // During ramp: use TARGET forcing, not current effective fx_ (avoids false positives)
    // The target forcing represents the steady-state we're ramping toward
    double fx_ref = fx_;
    if (force_ramp_enabled_) {
        fx_ref = fx_target_;  // Use target for consistent reference during ramp
    }
    ind.u_tau_force = std::sqrt(delta * std::abs(fx_ref));

    // Current u_tau from wall shear
    double tau_w = wall_shear_stress_2nd_order(true);
    ind.u_tau_current = (tau_w > 0) ? std::sqrt(tau_w) : 0.0;
    ind.u_tau_ratio = (ind.u_tau_force > 0) ? ind.u_tau_current / ind.u_tau_force : 0.0;

    // Auto-compute trust region if not specified
    // Exclude: fringe zone (2δ at inlet), outlet zone (2δ at outlet)
    double dx = mesh_->dx;
    int inlet_exclude = static_cast<int>(std::ceil(2.0 * delta / dx));
    int outlet_exclude = static_cast<int>(std::ceil(2.0 * delta / dx));

    ind.trust_x_start = (trust_x_start >= 0) ? trust_x_start : inlet_exclude;
    ind.trust_x_end = (trust_x_end >= 0) ? trust_x_end : (Nx - outlet_exclude);

    // Clamp to valid range
    ind.trust_x_start = std::max(0, std::min(ind.trust_x_start, Nx - 1));
    ind.trust_x_end = std::max(ind.trust_x_start + 1, std::min(ind.trust_x_end, Nx));

    // Store physical trust region coordinates for diagnostics
    ind.trust_x_min = mesh_->xc[ind.trust_x_start + Ng];
    ind.trust_x_max = mesh_->xc[ind.trust_x_end - 1 + Ng];

    // Mid-channel RMS (at y/delta ~ 0.5, i.e., y = 0 for channel from -delta to delta)
    // Find j-index closest to channel center
    int j_mid = Ng;
    double min_dist = std::abs(mesh_->yc[Ng]);
    for (int jj = 0; jj < Ny; ++jj) {
        double dist = std::abs(mesh_->yc[jj + Ng]);
        if (dist < min_dist) {
            min_dist = dist;
            j_mid = jj + Ng;
        }
    }

    // Compute mid-channel RMS in trust region
    double u_sum = 0.0, v_sum = 0.0, w_sum = 0.0;
    double u_sq_sum = 0.0, v_sq_sum = 0.0, w_sq_sum = 0.0;
    int count = 0;

    if (Nz > 1) {
        // 3D case
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int i = Ng + ind.trust_x_start; i < Ng + ind.trust_x_end; ++i) {
                double u = velocity_.u(i, j_mid, k);
                double v = 0.5 * (velocity_.v(i, j_mid, k) + velocity_.v(i, j_mid + 1, k));
                double w = velocity_.w(i, j_mid, k);

                u_sum += u;
                v_sum += v;
                w_sum += w;
                u_sq_sum += u * u;
                v_sq_sum += v * v;
                w_sq_sum += w * w;
                ++count;
            }
        }
    } else {
        // 2D case
        for (int i = Ng + ind.trust_x_start; i < Ng + ind.trust_x_end; ++i) {
            double u = velocity_.u(i, j_mid);
            double v = 0.5 * (velocity_.v(i, j_mid) + velocity_.v(i, j_mid + 1));

            u_sum += u;
            v_sum += v;
            u_sq_sum += u * u;
            v_sq_sum += v * v;
            ++count;
        }
    }

    if (count > 0) {
        double u_mean = u_sum / count;
        double v_mean = v_sum / count;
        double w_mean = w_sum / count;
        double u_var = std::max(0.0, u_sq_sum / count - u_mean * u_mean);
        double v_var = std::max(0.0, v_sq_sum / count - v_mean * v_mean);
        double w_var = (Nz > 1) ? std::max(0.0, w_sq_sum / count - w_mean * w_mean) : 0.0;
        ind.u_rms_mid = std::sqrt(u_var);
        ind.v_rms_mid = std::sqrt(v_var);
        ind.w_rms_mid = std::sqrt(w_var);
        // Turbulent kinetic energy at mid-channel: k = 0.5 * (u'² + v'² + w'²)
        ind.tke_mid = 0.5 * (u_var + v_var + w_var);
    }

    // Reynolds stress peak from accumulated statistics (in trust region)
    if (stats_samples_ > 0) {
        double u_tau = u_tau_from_forcing();
        double u_tau_sq = u_tau * u_tau;
        ind.max_uv_plus = 0.0;

        for (int jj = 0; jj < Ny; ++jj) {
            double y_plus = (mesh_->yc[jj + Ng] - mesh_->y_min) * u_tau / nu;
            // Only check in buffer/log region (5 < y+ < 150)
            if (y_plus > 5.0 && y_plus < 150.0) {
                double uv_plus = -stats_uv_[jj] / u_tau_sq;  // -<u'v'>+
                if (uv_plus > ind.max_uv_plus) {
                    ind.max_uv_plus = uv_plus;
                    ind.max_uv_plus_y_index = jj;
                }
            }
        }
    }

    // Wall shear drift rate from history
    if (wall_shear_history_.size() >= 2) {
        const auto& recent = wall_shear_history_.back();
        const auto& prev = wall_shear_history_[wall_shear_history_.size() - 2];
        double dt_hist = recent.time - prev.time;
        if (dt_hist > 1e-12 && recent.u_tau_avg > 1e-12) {
            ind.u_tau_drift_rate = std::abs(recent.u_tau_avg - prev.u_tau_avg) /
                                   (recent.u_tau_avg * dt_hist);
            ind.is_settling = (ind.u_tau_drift_rate > 0.01);  // 1% per unit time
        }
    }

    return ind;
}

// ============================================================================
// Wall shear history tracking
// ============================================================================

void RANSSolver::record_wall_shear_sample(double time) {
    WallShearSample sample;
    sample.time = time;
    sample.u_tau_bot = friction_velocity_2nd_order(true);
    sample.u_tau_top = friction_velocity_2nd_order(false);
    sample.u_tau_avg = 0.5 * (sample.u_tau_bot + sample.u_tau_top);
    wall_shear_history_.push_back(sample);
}

bool RANSSolver::is_wall_shear_settled(int window_samples, double drift_threshold) const {
    // Only meaningful after force ramp is complete
    if (force_ramp_enabled_ && is_force_ramp_active()) {
        return false;  // Still ramping, can't evaluate settling
    }

    if (static_cast<int>(wall_shear_history_.size()) < window_samples) {
        return false;  // Not enough samples
    }

    // Check drift over the window
    size_t start_idx = wall_shear_history_.size() - window_samples;
    const auto& start = wall_shear_history_[start_idx];
    const auto& end = wall_shear_history_.back();

    double dt = end.time - start.time;
    if (dt < 1e-12 || end.u_tau_avg < 1e-12) {
        return false;
    }

    double drift_rate = std::abs(end.u_tau_avg - start.u_tau_avg) / (end.u_tau_avg * dt);
    return (drift_rate < drift_threshold);
}

// ============================================================================
// Time-windowed turbulence classification
// ============================================================================

void RANSSolver::record_turbulence_sample() {
    // Get current turbulence indicators
    auto ind = compute_turbulence_presence();

    // Create sample
    TurbulenceSample sample;
    sample.time = current_time_;
    sample.u_tau_ratio = ind.u_tau_ratio;
    sample.u_rms_mid = ind.u_rms_mid;
    sample.tke_mid = ind.tke_mid;
    sample.max_uv_plus = ind.max_uv_plus;
    sample.state = ind.classify();

    // Add to rolling buffer
    turb_samples_.push_back(sample);

    // Trim buffer to window size
    const int window_size = turb_classifier_.window_size;
    while (static_cast<int>(turb_samples_.size()) > window_size) {
        turb_samples_.erase(turb_samples_.begin());
    }

    // Update window statistics
    if (!turb_samples_.empty()) {
        double sum_ratio = 0.0, sum_rms = 0.0, sum_tke = 0.0, sum_uv = 0.0;
        for (const auto& s : turb_samples_) {
            sum_ratio += s.u_tau_ratio;
            sum_rms += s.u_rms_mid;
            sum_tke += s.tke_mid;
            sum_uv += s.max_uv_plus;
        }
        double n = static_cast<double>(turb_samples_.size());
        turb_classifier_.u_tau_ratio_mean = sum_ratio / n;
        turb_classifier_.u_rms_mid_mean = sum_rms / n;
        turb_classifier_.tke_mid_mean = sum_tke / n;
        turb_classifier_.max_uv_plus_mean = sum_uv / n;
    }

    // Classify based on window mean
    TurbulenceState instant_state = turb_classifier_.classify_instant();

    // Update hysteresis counters
    if (instant_state == TurbulenceState::TURBULENT) {
        turb_classifier_.consecutive_turbulent++;
        turb_classifier_.consecutive_transitional = 0;
        turb_classifier_.consecutive_laminar = 0;
    } else if (instant_state == TurbulenceState::TRANSITIONAL) {
        turb_classifier_.consecutive_turbulent = 0;
        turb_classifier_.consecutive_transitional++;
        turb_classifier_.consecutive_laminar = 0;
    } else {
        turb_classifier_.consecutive_turbulent = 0;
        turb_classifier_.consecutive_transitional = 0;
        turb_classifier_.consecutive_laminar++;
    }

    // Apply hysteresis to update confirmed state
    const int hyst = turb_classifier_.hysteresis_count;
    if (turb_classifier_.consecutive_turbulent >= hyst) {
        turb_classifier_.confirmed_state = TurbulenceState::TURBULENT;
    } else if (turb_classifier_.consecutive_laminar >= hyst) {
        turb_classifier_.confirmed_state = TurbulenceState::LAMINAR;
    } else if (turb_classifier_.consecutive_transitional >= hyst) {
        turb_classifier_.confirmed_state = TurbulenceState::TRANSITIONAL;
    }
    // Otherwise, keep previous confirmed state (hysteresis)
}

// ============================================================================
// Ramped forcing
// ============================================================================

void RANSSolver::enable_force_ramp(double ramp_time) {
    force_ramp_enabled_ = true;
    force_ramp_tau_ = ramp_time;
    fx_target_ = fx_;
    fy_target_ = fy_;
    fz_target_ = fz_;
}

double RANSSolver::get_effective_fx() const {
    if (!force_ramp_enabled_) return fx_;
    double ramp = 1.0 - std::exp(-current_time_ / force_ramp_tau_);
    return fx_target_ * ramp;
}

double RANSSolver::get_effective_fy() const {
    if (!force_ramp_enabled_) return fy_;
    double ramp = 1.0 - std::exp(-current_time_ / force_ramp_tau_);
    return fy_target_ * ramp;
}

double RANSSolver::get_effective_fz() const {
    if (!force_ramp_enabled_) return fz_;
    double ramp = 1.0 - std::exp(-current_time_ / force_ramp_tau_);
    return fz_target_ * ramp;
}

// ============================================================================
// Initial velocity projection (divergence cleanup after perturbation)
// ============================================================================

void RANSSolver::project_initial_velocity() {
    // Compute divergence of current velocity
    compute_divergence(VelocityWhich::Current, div_velocity_);

    // Solve pressure Poisson: ∇²φ = ∇·u
    // Note: This is NOT scaled by 1/dt since we're just removing divergence
    rhs_poisson_ = div_velocity_;

    // Use default Poisson config
    PoissonConfig pcfg;
    pcfg.max_vcycles = config_.poisson_max_vcycles;
    pcfg.tol_rhs = config_.poisson_tol_rhs;
    pcfg.fixed_cycles = config_.poisson_fixed_cycles;

    // Solve Poisson equation
    // For initial projection, always use MG solver (works on both CPU/GPU, one-time cost)
    // This avoids complexity with FFT solvers that only have solve_device()
    mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);

    // Correct velocity: u = u - ∇φ
    // Note: Using dt=1.0 effectively since we didn't scale the RHS
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dz = mesh_->dz;

    if (Nz > 1) {
        // 3D case
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                // Correct u at x-faces
                for (int i = Ng; i < Ng + Nx + 1; ++i) {
                    double dp_dx = (pressure_correction_(i, j, k) -
                                    pressure_correction_(i - 1, j, k)) / dx;
                    velocity_.u(i, j, k) -= dp_dx;
                }
                // Correct v at y-faces
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int jf = j;
                    if (jf > Ng && jf < Ng + Ny) {
                        double dy_f = mesh_->yc[jf] - mesh_->yc[jf - 1];
                        double dp_dy = (pressure_correction_(i, jf, k) -
                                        pressure_correction_(i, jf - 1, k)) / dy_f;
                        velocity_.v(i, jf, k) -= dp_dy;
                    }
                }
            }
            // Correct w at z-faces
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    if (k > Ng && k < Ng + Nz) {
                        double dp_dz = (pressure_correction_(i, j, k) -
                                        pressure_correction_(i, j, k - 1)) / dz;
                        velocity_.w(i, j, k) -= dp_dz;
                    }
                }
            }
        }
    } else {
        // 2D case
        for (int j = Ng; j < Ng + Ny; ++j) {
            // Correct u
            for (int i = Ng; i < Ng + Nx + 1; ++i) {
                double dp_dx = (pressure_correction_(i, j) -
                                pressure_correction_(i - 1, j)) / dx;
                velocity_.u(i, j) -= dp_dx;
            }
            // Correct v
            for (int i = Ng; i < Ng + Nx; ++i) {
                if (j > Ng && j < Ng + Ny) {
                    double dy_f = mesh_->yc[j] - mesh_->yc[j - 1];
                    double dp_dy = (pressure_correction_(i, j) -
                                    pressure_correction_(i, j - 1)) / dy_f;
                    velocity_.v(i, j) -= dp_dy;
                }
            }
        }
    }

    // Re-apply boundary conditions
    apply_velocity_bc();
}

// ============================================================================
// Stage F validation with mode
// ============================================================================

RANSSolver::TurbulenceRealismReport RANSSolver::validate_turbulence_realism(ValidationMode mode) const {
    TurbulenceRealismReport report;
    report.mode = mode;

    // Resolution diagnostics
    report.resolution = compute_resolution_diagnostics();

    // Turbulence presence indicators
    report.presence = compute_turbulence_presence();

    // Thresholds depend on mode
    double closure_tol, utau_tol;
    if (mode == ValidationMode::Quick) {
        closure_tol = TurbulenceRealismReport::QUICK_CLOSURE_TOL;
        utau_tol = TurbulenceRealismReport::QUICK_UTAU_CONSISTENCY;
    } else {
        closure_tol = TurbulenceRealismReport::FULL_CLOSURE_TOL;
        utau_tol = TurbulenceRealismReport::FULL_UTAU_CONSISTENCY;
    }

    // Resolution check (only enforced in Full mode)
    if (mode == ValidationMode::Quick) {
        report.resolution_ok = true;  // Skip resolution gates in Quick mode
    } else {
        report.resolution_ok = report.resolution.passes_resolution_gates();
    }

    // u_tau consistency
    report.utau_consistency_ok = report.resolution.passes_utau_consistency(utau_tol);

    // Momentum balance
    report.momentum_balance = compute_momentum_balance();
    double u_tau = report.resolution.u_tau_force;
    report.momentum_closure_ok = (stats_samples_ > 0) &&
                                  (report.momentum_balance.max_residual_normalized(u_tau) <= closure_tol);

    // Reynolds stress profiles
    report.stress_profiles = compute_reynolds_stress_profiles();

    if (mode == ValidationMode::Quick) {
        // Quick mode: just check turbulence is present, not shape details
        report.stress_shape_ok = true;
    } else {
        report.stress_shape_ok = report.stress_profiles.passes_stress_ordering() &&
                                 report.stress_profiles.passes_uv_shape();
    }

    // Spectral check (at y+ ~ 15)
    if (mesh_->Nz > 1) {
        auto spec = compute_spanwise_spectrum(15.0);
        double U_bulk = bulk_velocity();
        double x_recycle = (use_recycling_) ? mesh_->xc[recycle_i_] : mesh_->x_max;
        report.spectrum_ok = !spec.has_recirculation_spike(x_recycle, U_bulk) &&
                             !spec.has_aliasing_pileup();
    } else {
        report.spectrum_ok = true;  // Skip for 2D
    }

    // Turbulence presence check
    report.turbulence_present_ok = report.presence.is_turbulent_or_transitional();

    return report;
}

} // namespace nncfd
