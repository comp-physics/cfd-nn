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
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    return std::sqrt(delta * std::abs(fx_));
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

    // Average over x (and z for 3D)
    if (Nz > 1) {
        // 3D case
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double u1 = velocity_.u(i, j1, k);
                double u2 = velocity_.u(i, j2, k);

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

bool RANSSolver::SpanwiseSpectrum::has_recirculation_spike(double x_recycle,
                                                            double U_bulk,
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

    return report;
}

void RANSSolver::TurbulenceRealismReport::print() const {
    std::cout << "\n=== Turbulence Realism Validation Report ===\n";

    std::cout << "\n--- Resolution Diagnostics ---\n";
    std::cout << "  y1+ (bottom): " << std::fixed << std::setprecision(3)
              << resolution.y1_plus_bot << " (target: 0.3-0.8, max: 1.0)\n";
    std::cout << "  y1+ (top):    " << resolution.y1_plus_top << "\n";
    std::cout << "  dx+:          " << resolution.dx_plus << " (max: 15)\n";
    std::cout << "  dz+:          " << resolution.dz_plus << " (max: 8)\n";
    std::cout << "  Resolution:   " << (resolution_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- u_tau Consistency ---\n";
    std::cout << "  u_tau (force): " << std::setprecision(5) << resolution.u_tau_force << "\n";
    std::cout << "  u_tau (bot):   " << resolution.u_tau_bot
              << " (err: " << std::setprecision(2)
              << 100.0 * std::abs(resolution.u_tau_bot - resolution.u_tau_force) / resolution.u_tau_force << "%)\n";
    std::cout << "  u_tau (top):   " << resolution.u_tau_top
              << " (err: " << 100.0 * std::abs(resolution.u_tau_top - resolution.u_tau_force) / resolution.u_tau_force << "%)\n";
    std::cout << "  Consistency:   " << (utau_consistency_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Momentum Balance ---\n";
    double u_tau = resolution.u_tau_force;
    std::cout << "  Max |R|/u_tau^2: " << std::setprecision(3)
              << momentum_balance.max_residual_normalized(u_tau) * 100.0 << "% (max: 2%)\n";
    std::cout << "  L2 R/u_tau^2:    " << momentum_balance.l2_residual_normalized(u_tau) * 100.0 << "% (max: 1%)\n";
    std::cout << "  Closure:         " << (momentum_closure_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Reynolds Stress Shape ---\n";
    std::cout << "  Stress ordering: " << (stress_profiles.passes_stress_ordering() ? "PASS" : "FAIL") << "\n";
    std::cout << "  -<u'v'>+ shape:  " << (stress_profiles.passes_uv_shape() ? "PASS" : "FAIL") << "\n";
    std::cout << "  Overall shape:   " << (stress_shape_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Spectral Check ---\n";
    std::cout << "  Spectrum:        " << (spectrum_ok ? "PASS" : "FAIL") << "\n";

    std::cout << "\n=== OVERALL: " << (passes_all() ? "PASS" : "FAIL") << " ===\n\n";
}

// ============================================================================
// Plane statistics (CPU-only to avoid nvc++ compiler crash)
// Note: For GPU builds, caller should sync velocity from GPU before calling
// ============================================================================

RANSSolver::PlaneStats RANSSolver::compute_plane_stats(int i_global) const {
    PlaneStats stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int ig = i_global + Ng;

    const int Nz_loop = mesh_->is2D() ? 1 : Nz;
    const int n_points = Ny * Nz_loop;
    if (n_points == 0) return stats;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int u_plane = velocity_.u_plane_stride();
    const int v_plane = velocity_.v_plane_stride();
    const int w_stride = velocity_.w_stride();
    const int w_plane = velocity_.w_plane_stride();

    double* u_ptr = velocity_u_ptr_;
    double* v_ptr = velocity_v_ptr_;
    double* w_ptr = velocity_w_ptr_;

    double sum_u = 0.0, sum_v = 0.0, sum_w = 0.0;

    // First pass: compute means
    if (mesh_->is2D()) {
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double u = u_ptr[jg * u_stride + ig];
            double v = 0.5 * (v_ptr[jg * v_stride + ig] + v_ptr[(jg + 1) * v_stride + ig]);
            sum_u += u;
            sum_v += v;
        }
    } else {
        for (int idx = 0; idx < n_points; ++idx) {
            int j = idx % Ny + Ng;
            int k = idx / Ny + Ng;
            double u = u_ptr[k * u_plane + j * u_stride + ig];
            double v = 0.5 * (v_ptr[k * v_plane + j * v_stride + ig] + v_ptr[k * v_plane + (j + 1) * v_stride + ig]);
            double w = 0.5 * (w_ptr[k * w_plane + j * w_stride + ig] + w_ptr[(k + 1) * w_plane + j * w_stride + ig]);
            sum_u += u;
            sum_v += v;
            sum_w += w;
        }
    }

    stats.u_mean = sum_u / n_points;
    stats.v_mean = sum_v / n_points;
    stats.w_mean = sum_w / n_points;

    // Second pass: fluctuations and Reynolds stress
    double sum_uu = 0.0, sum_vv = 0.0, sum_ww = 0.0, sum_uv = 0.0;
    const double u_mean = stats.u_mean;
    const double v_mean = stats.v_mean;
    const double w_mean = stats.w_mean;

    if (mesh_->is2D()) {
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double u = u_ptr[jg * u_stride + ig];
            double v = 0.5 * (v_ptr[jg * v_stride + ig] + v_ptr[(jg + 1) * v_stride + ig]);
            double u_prime = u - u_mean;
            double v_prime = v - v_mean;
            sum_uu += u_prime * u_prime;
            sum_vv += v_prime * v_prime;
            sum_uv += u_prime * v_prime;
        }
    } else {
        for (int idx = 0; idx < n_points; ++idx) {
            int j = idx % Ny + Ng;
            int k = idx / Ny + Ng;
            double u = u_ptr[k * u_plane + j * u_stride + ig];
            double v = 0.5 * (v_ptr[k * v_plane + j * v_stride + ig] + v_ptr[k * v_plane + (j + 1) * v_stride + ig]);
            double w = 0.5 * (w_ptr[k * w_plane + j * w_stride + ig] + w_ptr[(k + 1) * w_plane + j * w_stride + ig]);
            double u_prime = u - u_mean;
            double v_prime = v - v_mean;
            double w_prime = w - w_mean;
            sum_uu += u_prime * u_prime;
            sum_vv += v_prime * v_prime;
            sum_ww += w_prime * w_prime;
            sum_uv += u_prime * v_prime;
        }
    }

    stats.u_rms = std::sqrt(sum_uu / n_points);
    stats.v_rms = std::sqrt(sum_vv / n_points);
    stats.w_rms = std::sqrt(sum_ww / n_points);
    stats.uv_reynolds = -sum_uv / n_points;  // -<u'v'>

    return stats;
}

} // namespace nncfd
