#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#ifdef USE_FFT_POISSON
#include "poisson_solver_fft.hpp"
#include "poisson_solver_fft1d.hpp"
#include "poisson_solver_fft2d.hpp"
#endif
#include "turbulence_model.hpp"
#include "config.hpp"
#include <memory>
#include <functional>

namespace nncfd {

/// Selector for which velocity field to use in unified GPU/CPU routines
enum class VelocityWhich {
    Current,  ///< Current velocity (velocity_)
    Star,     ///< Predictor velocity (velocity_star_)
    Old       ///< Previous time step (velocity_old_)
};

/// Helper struct for velocity pointer pairs from SolverDeviceView
struct FaceVelPtrs {
    const double* u;
    const double* v;
    const double* w;
    int u_stride;
    int v_stride;
    int w_stride;
    int u_plane_stride;
    int v_plane_stride;
    int w_plane_stride;
};

/// Select velocity pointers from SolverDeviceView based on which field
inline FaceVelPtrs select_face_velocity(const SolverDeviceView& v, VelocityWhich which) {
    switch (which) {
    case VelocityWhich::Current: return {v.u_face, v.v_face, v.w_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    case VelocityWhich::Star:    return {v.u_star_face, v.v_star_face, v.w_star_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    case VelocityWhich::Old:     return {v.u_old_face, v.v_old_face, v.w_old_face,
                                         v.u_stride, v.v_stride, v.w_stride,
                                         v.u_plane_stride, v.v_plane_stride, v.w_plane_stride};
    }
    return {nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0};
}

/// Boundary condition specification for velocity
struct VelocityBC {
    enum Type { NoSlip, Periodic, Inflow, Outflow };
    Type x_lo = Periodic;
    Type x_hi = Periodic;
    Type y_lo = NoSlip;
    Type y_hi = NoSlip;
    Type z_lo = Periodic;  // Default periodic for spanwise direction
    Type z_hi = Periodic;

    // Inflow values (if applicable) - 2D backward compatible
    std::function<double(double y)> u_inflow = [](double) { return 0.0; };
    std::function<double(double y)> v_inflow = [](double) { return 0.0; };

    // 3D inflow values (if applicable)
    std::function<double(double y, double z)> u_inflow_3d = [](double, double) { return 0.0; };
    std::function<double(double y, double z)> v_inflow_3d = [](double, double) { return 0.0; };
    std::function<double(double y, double z)> w_inflow_3d = [](double, double) { return 0.0; };
};

/// Incompressible RANS solver using projection method
class RANSSolver {
public:
    explicit RANSSolver(const Mesh& mesh, const Config& config);
    ~RANSSolver();  // Clean up GPU resources
    
    /// Set turbulence model (takes ownership)
    void set_turbulence_model(std::unique_ptr<TurbulenceModel> model);
    
    /// Set velocity boundary conditions
    void set_velocity_bc(const VelocityBC& bc);
    
    /// Set body force (pressure gradient equivalent)
    void set_body_force(double fx, double fy, double fz = 0.0);

    /// Get the selected Poisson solver type (after auto-selection)
    PoissonSolverType poisson_solver_type() const { return selected_solver_; }

    /// Get the reason for Poisson solver selection (for debugging/observability)
    /// Returns a human-readable string explaining why the solver was chosen
    const std::string& selection_reason() const { return selection_reason_; }

    /// Poisson solve termination status
    enum class PoissonSolveStatus {
        ConvergedToTol,  ///< Met tolerance criterion before max_vcycles
        HitMaxCycles,    ///< Ran max_vcycles without meeting tolerance
        FixedCycles      ///< Ran exactly fixed_cycles (no convergence check)
    };

    /// Poisson solve statistics from last step (for testing/diagnostics)
    struct PoissonStats {
        int cycles = 0;           ///< V-cycles (or iterations) performed
        PoissonSolveStatus status = PoissonSolveStatus::ConvergedToTol;

        // Poisson residual norms
        double rhs_norm_l2 = 0.0; ///< ||b||_L2 at start of solve
        double rhs_norm_inf = 0.0;///< ||b||_∞ at start of solve
        double res_norm_l2 = 0.0; ///< ||r||_L2 after solve
        double res_norm_inf = 0.0;///< ||r||_∞ after solve
        double res_over_rhs = 0.0;///< ||r||/||b|| (relative residual) - key metric

        // Post-projection divergence (raw)
        double div_after_proj_linf = 0.0; ///< ||div(u)||_∞ after projection
        double div_after_proj_l2 = 0.0;   ///< ||div(u)||_L2 after projection

        // Dimensionless divergence (scaled by dx_min / u_ref for portable thresholds)
        double div_scaled_linf = 0.0;     ///< ||div(u)||_∞ * dx_min / u_ref
        double div_scaled_l2 = 0.0;       ///< ||div(u)||_L2 * dx_min / u_ref
        double dx_min = 0.0;              ///< Min cell size used for scaling
        double u_ref = 1.0;               ///< Reference velocity (u_tau or bulk)

        /// Helper to get status string for logging
        const char* status_string() const {
            switch (status) {
                case PoissonSolveStatus::ConvergedToTol: return "TOL";
                case PoissonSolveStatus::HitMaxCycles:   return "MAX_CYCLES";
                case PoissonSolveStatus::FixedCycles:    return "FIXED";
            }
            return "UNKNOWN";
        }

        /// Check if solve achieved tolerance (not fixed-cycle or max-cycles)
        bool converged_to_tol() const { return status == PoissonSolveStatus::ConvergedToTol; }
    };

    /// Get statistics from last Poisson solve (updated after each step())
    const PoissonStats& poisson_stats() const { return poisson_stats_; }

    /// Print solver configuration (call after set_velocity_bc() for final info)
    /// Prints: selected solver, selection reason, solver parameters, build info
    void print_solver_info() const;

#ifdef USE_HYPRE
    /// Enable/disable HYPRE PFMG Poisson solver (legacy API, prefer --poisson=hypre)
    void set_use_hypre(bool use) { if (use) selected_solver_ = PoissonSolverType::HYPRE; }
    bool using_hypre() const { return selected_solver_ == PoissonSolverType::HYPRE; }
#endif

    /// Initialize velocity field
    void initialize(const VectorField& initial_velocity);
    void initialize_uniform(double u0, double v0);
    
    /// Advance one pseudo-time step
    /// Returns velocity residual (for convergence check)
    double step();
    
    /// Run to steady state
    /// Returns final residual and number of iterations
    std::pair<double, int> solve_steady();
    
    /// Run to steady state with optional VTK snapshots
    /// @param output_prefix Base name for VTK files (e.g., "output/flow")
    /// @param num_snapshots Number of snapshots to write during simulation (0 = final only)
    /// @param snapshot_freq Override automatic snapshot frequency (optional, -1 = auto)
    /// @return Final residual and number of iterations
    std::pair<double, int> solve_steady_with_snapshots(
        const std::string& output_prefix = "",
        int num_snapshots = 0,
        int snapshot_freq = -1
    );
    
    /// Advance unsteady simulation for multiple time steps
    /// @param dt Time step size
    /// @param nsteps Number of time steps to take
    void advance_unsteady(double dt, int nsteps);
    
    /// Compute adaptive time step based on CFL and diffusion stability
    double compute_adaptive_dt() const;
    
    /// Get current time step
    double current_dt() const { return current_dt_; }
    
    /// Access fields
    const VectorField& velocity() const { return velocity_; }
    const ScalarField& pressure() const { return pressure_; }
    const ScalarField& nu_t() const { return nu_t_; }
    const ScalarField& k() const { return k_; }
    const ScalarField& omega() const { return omega_; }
    const ScalarField& nu_eff() const { return nu_eff_; }
    const ScalarField& div_velocity() const { return div_velocity_; }  ///< For diagnostics
    const ScalarField& rhs_poisson() const { return rhs_poisson_; }    ///< For diagnostics
    const ScalarField& pressure_correction() const { return pressure_correction_; }  ///< For Poisson residual checks

    VectorField& velocity() { return velocity_; }
    ScalarField& pressure() { return pressure_; }
    
    /// Get current iteration
    int iteration() const { return iter_; }
    
    /// Compute bulk velocity (area-averaged)
    double bulk_velocity() const;
    
    /// Compute wall shear stress (at y_min wall)
    double wall_shear_stress() const;
    
    /// Compute friction velocity u_tau
    double friction_velocity() const;
    
    /// Compute friction Reynolds number Re_tau
    double Re_tau() const;

    /// Compute convective KE production rate: <u, conv(u)>
    /// Returns the rate of KE change due to advection (should be ~0 for skew-symmetric form)
    /// Call this after compute_convective_term() to get meaningful results
    double compute_convective_ke_production() const;
    
    /// Print velocity profile at x = x_loc
    void print_velocity_profile(double x_loc = 0.0) const;
    
    /// Write fields to files
    void write_fields(const std::string& prefix) const;
    
    /// Write VTK output for ParaView visualization
    void write_vtk(const std::string& filename) const;
    
    /// GPU buffer management (public for testing and initialization)
    void sync_to_gpu();                 // Update GPU after CPU-side modifications (e.g., after initialization)
    void sync_from_gpu();               // Update CPU copy for I/O (data stays on GPU) - downloads all fields
    void sync_solution_from_gpu();      // Sync only solution fields (u,v,p,nu_t) - use for diagnostics/I/O
    void sync_transport_from_gpu();     // Sync transport fields (k,omega) only if needed - guards laminar runs

    /// Verify all GPU-mapped fields are present on device (for testing/debugging)
    /// Returns true if all critical fields pass omp_target_is_present() check.
    /// Use this after sync_to_gpu() to catch missing field mappings early.
    bool verify_gpu_field_presence() const;

    /// Device-side QOI computation (avoids broken D2H sync in NVHPC)
    /// These compute quantities directly on GPU and return scalars via reduction.
    /// Use these for GPU tests instead of syncing full fields.
    double compute_kinetic_energy_device() const;   // Total KE = 0.5 * integral(u^2 + v^2 + w^2) dV
    double compute_max_velocity_device() const;     // max(|u|, |v|, |w|) over domain
    double compute_divergence_linf_device() const;  // L-inf norm of divergence
    double compute_divergence_l2_device() const;    // L2 norm of divergence
    double compute_max_conv_device() const;         // max(|conv_u|, |conv_v|, |conv_w|) - verify convection active

    /// Energy balance diagnostics (for recycling/inflow validation)
    /// @{
    double compute_kinetic_energy() const;          // KE = 0.5 * integral(u^2+v^2+w^2) dV (CPU version)
    double compute_bulk_velocity() const;           // U_b = integral(u) / V
    double compute_power_input() const;             // P_in = f_x * integral(u) dV
    double compute_viscous_dissipation() const;     // epsilon = 2*nu * integral(S_ij S_ij) dV
    /// @}

    /// Plane-averaged turbulence statistics at x-index i
    struct PlaneStats {
        double u_mean;      // Mean streamwise velocity
        double v_mean;      // Mean wall-normal velocity
        double w_mean;      // Mean spanwise velocity
        double u_rms;       // RMS of u fluctuations
        double v_rms;       // RMS of v fluctuations
        double w_rms;       // RMS of w fluctuations
        double uv_reynolds; // Reynolds shear stress -<u'v'>
    };
    PlaneStats compute_plane_stats(int i_global) const;  // i_global is x-index (0 to Nx-1)

    //=========================================================================
    // Turbulence Presence Indicators (robust turbulence detection)
    //=========================================================================

    /// Turbulence state classification
    enum class TurbulenceState {
        LAMINAR,       ///< Flow is laminar (no turbulent fluctuations)
        TRANSITIONAL,  ///< Flow is transitioning (some fluctuations, unstable)
        TURBULENT      ///< Flow is fully turbulent (sustained fluctuations)
    };

    /// Validation mode for Stage F
    enum class ValidationMode {
        Quick,  ///< Machinery validation: looser thresholds, coarse grids OK
        Full    ///< DNS realism: strict resolution gates, production quality
    };

    /// Turbulence presence indicators (robust detection beyond just -<u'v'>+)
    struct TurbulencePresenceIndicators {
        // Mid-channel RMS (time-averaged after spin-up)
        double u_rms_mid = 0.0;         ///< u'_rms at y/delta=0.5
        double v_rms_mid = 0.0;         ///< v'_rms at y/delta=0.5
        double w_rms_mid = 0.0;         ///< w'_rms at y/delta=0.5 (3D only)

        // Turbulent kinetic energy at mid-channel
        double tke_mid = 0.0;           ///< k = 0.5*(u'² + v'² + w'²) at y/delta=0.5

        // Wall shear tracking (use forcing-based reference for consistency)
        double u_tau_current = 0.0;     ///< Current u_tau from wall shear
        double u_tau_force = 0.0;       ///< Reference u_tau from forcing: sqrt(delta * |dp/dx_effective|)
        double u_tau_ratio = 0.0;       ///< u_tau_current / u_tau_force

        // Reynolds stress peak
        double max_uv_plus = 0.0;       ///< max(-<u'v'>+) in trust region
        int max_uv_plus_y_index = 0;    ///< y-index where max occurs

        // Trust region bounds (exclude fringe + 2δ at inlet, 2δ at outlet)
        int trust_x_start = 0;          ///< Start x-index of trust region
        int trust_x_end = 0;            ///< End x-index of trust region
        double trust_x_min = 0.0;       ///< Physical x-coordinate of trust region start
        double trust_x_max = 0.0;       ///< Physical x-coordinate of trust region end

        // Temporal stability (check if turbulence is sustained)
        double u_tau_drift_rate = 0.0;  ///< d(u_tau)/dt normalized by u_tau
        bool is_settling = false;       ///< True if u_tau is still drifting

        /// Classify turbulence state based on instantaneous indicators
        TurbulenceState classify() const {
            // Primary criterion: wall shear ratio
            if (u_tau_ratio > 1.2) {
                return TurbulenceState::TURBULENT;
            } else if (u_tau_ratio > 1.05) {
                return TurbulenceState::TRANSITIONAL;
            }

            // Secondary: Reynolds stress presence
            if (max_uv_plus >= 0.5) {
                return TurbulenceState::TURBULENT;
            } else if (max_uv_plus >= 0.1) {
                return TurbulenceState::TRANSITIONAL;
            }

            // Tertiary: mid-channel TKE (more robust than RMS alone)
            if (tke_mid > 0.01 * u_tau_force * u_tau_force) {
                return TurbulenceState::TRANSITIONAL;
            }

            return TurbulenceState::LAMINAR;
        }

        /// Get human-readable state string
        const char* state_string() const {
            switch (classify()) {
                case TurbulenceState::LAMINAR:      return "LAMINAR";
                case TurbulenceState::TRANSITIONAL: return "TRANSITIONAL";
                case TurbulenceState::TURBULENT:    return "TURBULENT";
            }
            return "UNKNOWN";
        }

        /// Check if turbulence is present (transitional or turbulent)
        bool is_turbulent_or_transitional() const {
            return classify() != TurbulenceState::LAMINAR;
        }
    };

    /// Sample for time-windowed turbulence tracking
    struct TurbulenceSample {
        double time = 0.0;
        double u_tau_ratio = 0.0;
        double u_rms_mid = 0.0;
        double tke_mid = 0.0;
        double max_uv_plus = 0.0;
        TurbulenceState state = TurbulenceState::LAMINAR;
    };

    /// Time-windowed turbulence classifier with hysteresis
    /// Prevents flickering classifications from momentary spikes/dips
    struct TurbulenceClassifier {
        // Window configuration
        static constexpr int DEFAULT_WINDOW_SIZE = 20;     ///< Samples in rolling window
        static constexpr int DEFAULT_HYSTERESIS = 5;       ///< Consecutive windows to change state

        // Window statistics (computed from rolling buffer)
        double u_tau_ratio_mean = 0.0;
        double u_rms_mid_mean = 0.0;
        double tke_mid_mean = 0.0;
        double max_uv_plus_mean = 0.0;

        // Current confirmed state (with hysteresis)
        TurbulenceState confirmed_state = TurbulenceState::LAMINAR;
        int consecutive_turbulent = 0;      ///< Consecutive windows classified turbulent
        int consecutive_transitional = 0;   ///< Consecutive windows classified transitional
        int consecutive_laminar = 0;        ///< Consecutive windows classified laminar

        // Configuration
        int window_size = DEFAULT_WINDOW_SIZE;
        int hysteresis_count = DEFAULT_HYSTERESIS;

        /// Classify based on windowed means (used internally)
        TurbulenceState classify_instant() const {
            if (u_tau_ratio_mean > 1.2 || max_uv_plus_mean >= 0.5) {
                return TurbulenceState::TURBULENT;
            } else if (u_tau_ratio_mean > 1.05 || max_uv_plus_mean >= 0.1 || tke_mid_mean > 0.01) {
                return TurbulenceState::TRANSITIONAL;
            }
            return TurbulenceState::LAMINAR;
        }

        /// Get confirmed state (with hysteresis applied)
        TurbulenceState state() const { return confirmed_state; }

        /// Get state string
        const char* state_string() const {
            switch (confirmed_state) {
                case TurbulenceState::LAMINAR:      return "LAMINAR";
                case TurbulenceState::TRANSITIONAL: return "TRANSITIONAL";
                case TurbulenceState::TURBULENT:    return "TURBULENT";
            }
            return "UNKNOWN";
        }
    };

    /// Wall shear history sample (for temporal tracking)
    struct WallShearSample {
        double time = 0.0;              ///< Simulation time
        double u_tau_bot = 0.0;         ///< u_tau at bottom wall
        double u_tau_top = 0.0;         ///< u_tau at top wall
        double u_tau_avg = 0.0;         ///< Average of both walls
    };

    //=========================================================================
    // Stage F: Turbulence Realism Validation (DNS/LES quality checks)
    //=========================================================================

    /// Resolution diagnostics for DNS/LES validation
    struct ResolutionDiagnostics {
        double u_tau_force = 0.0;   ///< u_tau from forcing: sqrt(delta * |dp/dx|)
        double u_tau_bot = 0.0;     ///< u_tau from wall shear at bottom wall
        double u_tau_top = 0.0;     ///< u_tau from wall shear at top wall
        double y1_plus_bot = 0.0;   ///< First cell y+ at bottom wall
        double y1_plus_top = 0.0;   ///< First cell y+ at top wall
        double dx_plus = 0.0;       ///< Streamwise spacing in wall units
        double dz_plus = 0.0;       ///< Spanwise spacing in wall units

        /// Pass resolution gates (y1+ <= 1, dx+ <= 15, dz+ <= 8)
        bool passes_resolution_gates() const {
            return y1_plus_bot <= 1.0 && y1_plus_top <= 1.0 &&
                   dx_plus <= 15.0 && (dz_plus <= 8.0 || dz_plus == 0.0);
        }

        /// Pass u_tau consistency check (wall vs forcing)
        bool passes_utau_consistency(double tol = 0.02) const {
            double err_bot = std::abs(u_tau_bot - u_tau_force) / (u_tau_force + 1e-12);
            double err_top = std::abs(u_tau_top - u_tau_force) / (u_tau_force + 1e-12);
            return err_bot <= tol && err_top <= tol;
        }
    };

    /// Momentum balance diagnostics: tau_tot(y) vs tau_lin(y)
    struct MomentumBalanceDiagnostics {
        std::vector<double> y;           ///< Distance from wall
        std::vector<double> y_plus;      ///< Wall units y+
        std::vector<double> tau_visc;    ///< Viscous stress: nu * dU/dy
        std::vector<double> tau_reynolds;///< Reynolds stress: -<u'v'>
        std::vector<double> tau_total;   ///< Total: tau_visc + tau_reynolds
        std::vector<double> tau_theory;  ///< Linear theory: u_tau^2 * (1 - y/delta)
        std::vector<double> residual;    ///< tau_total - tau_theory

        /// Max |residual| / u_tau^2
        double max_residual_normalized(double u_tau) const {
            double u_tau_sq = u_tau * u_tau;
            double max_res = 0.0;
            for (double r : residual) {
                max_res = std::max(max_res, std::abs(r) / (u_tau_sq + 1e-12));
            }
            return max_res;
        }

        /// L2 residual / u_tau^2
        double l2_residual_normalized(double u_tau) const {
            double u_tau_sq = u_tau * u_tau;
            double sum_sq = 0.0;
            for (double r : residual) {
                sum_sq += (r * r) / (u_tau_sq * u_tau_sq + 1e-24);
            }
            return std::sqrt(sum_sq / (residual.size() + 1));
        }
    };

    /// Reynolds stress profiles in wall units
    struct ReynoldsStressProfiles {
        std::vector<double> y_plus;   ///< Wall coordinate y+
        std::vector<double> uu_plus;  ///< <u'u'>+
        std::vector<double> vv_plus;  ///< <v'v'>+
        std::vector<double> ww_plus;  ///< <w'w'>+
        std::vector<double> uv_plus;  ///< -<u'v'>+

        /// Check stress ordering: <u'u'> > <w'w'> > <v'v'> in buffer/log layer
        bool passes_stress_ordering() const;

        /// Check -<u'v'>+ shape: near-zero at wall, positive interior
        bool passes_uv_shape() const;
    };

    /// Spanwise energy spectrum for recycling artifact detection
    struct SpanwiseSpectrum {
        std::vector<double> k_z;      ///< Wavenumbers
        std::vector<double> E_uu;     ///< Energy spectrum of u

        /// Check for narrow spike at recirculation frequency
        bool has_recirculation_spike(double x_recycle, double U_bulk, double tol = 5.0) const;

        /// Check for aliasing pileup at high wavenumbers
        bool has_aliasing_pileup(double tol = 1.5) const;
    };

    /// Full turbulence realism validation report
    struct TurbulenceRealismReport {
        ResolutionDiagnostics resolution;
        MomentumBalanceDiagnostics momentum_balance;
        ReynoldsStressProfiles stress_profiles;
        TurbulencePresenceIndicators presence;  ///< Turbulence presence indicators

        ValidationMode mode = ValidationMode::Full;  ///< Validation mode used

        bool resolution_ok = false;
        bool utau_consistency_ok = false;
        bool momentum_closure_ok = false;
        bool stress_shape_ok = false;
        bool spectrum_ok = false;
        bool turbulence_present_ok = false;  ///< True if turbulence is detected (Quick mode)

        /// Pass criteria depend on validation mode:
        /// - Full: all checks must pass (DNS quality)
        /// - Quick: turbulence present + no explosion (machinery validation)
        bool passes_all() const {
            if (mode == ValidationMode::Quick) {
                // Quick mode: turbulence present, closure reasonable, not exploding
                return turbulence_present_ok &&
                       (momentum_balance.max_residual_normalized(resolution.u_tau_force) < 1.0);
            }
            // Full mode: all checks must pass
            return resolution_ok && utau_consistency_ok &&
                   momentum_closure_ok && stress_shape_ok && spectrum_ok;
        }

        void print() const;  ///< Print detailed report to stdout

        /// Thresholds for Quick mode (machinery validation on coarse grids)
        static constexpr double QUICK_CLOSURE_TOL = 0.50;      ///< 50% momentum closure
        static constexpr double QUICK_UTAU_CONSISTENCY = 0.20; ///< 20% u_tau consistency

        /// Thresholds for Full mode (DNS realism)
        static constexpr double FULL_CLOSURE_TOL = 0.02;       ///< 2% momentum closure
        static constexpr double FULL_UTAU_CONSISTENCY = 0.02;  ///< 2% u_tau consistency
    };

    /// Friction velocity from body forcing (exact for fully-developed channel)
    double u_tau_from_forcing() const;

    /// Re_tau from current forcing and viscosity
    double Re_tau_from_forcing() const;

    /// Compute nu for target Re_tau: nu = sqrt(delta * |dp/dx|) * delta / Re_tau
    static double nu_for_Re_tau(double Re_tau_target, double dp_dx, double delta = 1.0);

    /// Compute dp/dx for target Re_tau: dp/dx = -(Re_tau * nu / delta)^2 / delta
    static double dp_dx_for_Re_tau(double Re_tau_target, double nu, double delta = 1.0);

    /// Wall shear stress using 2nd-order accurate quadratic fit
    double wall_shear_stress_2nd_order(bool bottom = true) const;

    /// Friction velocity from 2nd-order wall shear
    double friction_velocity_2nd_order(bool bottom = true) const;

    /// Compute resolution diagnostics (y+, dx+, dz+, u_tau consistency)
    ResolutionDiagnostics compute_resolution_diagnostics() const;

    /// Reset time-averaged statistics
    void reset_statistics();

    /// Accumulate statistics sample (call after each timestep during averaging window)
    void accumulate_statistics();

    /// Number of statistics samples accumulated
    int statistics_samples() const { return stats_samples_; }

    /// Compute momentum balance (requires accumulated statistics)
    MomentumBalanceDiagnostics compute_momentum_balance() const;

    /// Compute Reynolds stress profiles (requires accumulated statistics)
    ReynoldsStressProfiles compute_reynolds_stress_profiles() const;

    /// Compute spanwise energy spectrum at target y+ location
    SpanwiseSpectrum compute_spanwise_spectrum(double y_plus_target) const;

    /// Run full Stage F turbulence realism validation
    TurbulenceRealismReport validate_turbulence_realism() const;

    /// Run Stage F validation with specified mode (Quick or Full)
    TurbulenceRealismReport validate_turbulence_realism(ValidationMode mode) const;

    /// Compute turbulence presence indicators (robust detection)
    /// @param trust_x_start Start x-index for trust region (auto-computed if -1)
    /// @param trust_x_end End x-index for trust region (auto-computed if -1)
    TurbulencePresenceIndicators compute_turbulence_presence(int trust_x_start = -1,
                                                              int trust_x_end = -1) const;

    /// Record wall shear sample (call periodically during spinup for drift detection)
    void record_wall_shear_sample(double time);

    /// Get wall shear history
    const std::vector<WallShearSample>& get_wall_shear_history() const { return wall_shear_history_; }

    /// Clear wall shear history (call when starting statistics collection)
    void clear_wall_shear_history() { wall_shear_history_.clear(); }

    /// Check if wall shear has settled (drift rate below threshold)
    /// @param window_samples Number of recent samples to check
    /// @param drift_threshold Max allowed |d(u_tau)/dt| / u_tau per unit time
    /// @note Only meaningful after force ramp is complete
    bool is_wall_shear_settled(int window_samples = 10, double drift_threshold = 0.01) const;

    //=========================================================================
    // Time-windowed turbulence classifier
    //=========================================================================

    /// Record turbulence sample for windowed classification
    /// Call this periodically (e.g., every 10-50 steps) during spinup/stats
    void record_turbulence_sample();

    /// Get the time-windowed turbulence classifier state
    const TurbulenceClassifier& turbulence_classifier() const { return turb_classifier_; }

    /// Clear turbulence sample history (call when starting fresh)
    void clear_turbulence_samples() {
        turb_samples_.clear();
        turb_classifier_ = TurbulenceClassifier();
    }

    /// Configure turbulence classifier window
    void configure_turbulence_classifier(int window_size, int hysteresis_count) {
        turb_classifier_.window_size = window_size;
        turb_classifier_.hysteresis_count = hysteresis_count;
    }

    //=========================================================================
    // Ramped forcing for startup stabilization
    //=========================================================================

    /// Enable ramped forcing: dp/dx(t) = (1 - e^{-t/T}) * dp/dx_target
    /// @param ramp_time Time constant T for exponential ramp (bulk time units)
    void enable_force_ramp(double ramp_time);

    /// Disable force ramping (use constant forcing)
    void disable_force_ramp() { force_ramp_enabled_ = false; }

    /// Check if force ramping is active
    bool is_force_ramp_active() const { return force_ramp_enabled_ && current_time_ < 5.0 * force_ramp_tau_; }

    /// Get current effective body force (accounts for ramping)
    double get_effective_fx() const;
    double get_effective_fy() const;
    double get_effective_fz() const;

    /// Get current simulation time
    double current_time() const { return current_time_; }

    /// Project velocity field to remove divergence (one-time cleanup after perturbation)
    /// This performs a single pressure projection without advancing time
    void project_initial_velocity();

    //=========================================================================

    /// Stage-by-stage diagnostics for recycling inflow pipeline
    /// Tracks L2 norms and invariants to catch regressions
    struct RecycleDiagnostics {
        // Stage L2 norms (u-component, area-weighted)
        double L2_copy = 0.0;      // After copy+shift from recycle plane
        double L2_ar1 = 0.0;       // After AR1 filter (or =L2_copy if disabled)
        double L2_mean = 0.0;      // After mean correction
        double L2_final = 0.0;     // After transverse mean removal (final inlet)

        // Stage-to-stage relative deltas (for regression detection)
        double rel_d_copy_ar1 = 0.0;    // |u_ar1 - u_copy| / |u_copy|
        double rel_d_ar1_mean = 0.0;    // |u_mean - u_ar1| / |u_ar1|
        double rel_d_mean_final = 0.0;  // |u_final - u_mean| / |u_mean|
        double rel_d_total = 0.0;       // |u_final - u_copy| / |u_copy|

        // Invariants
        double u_mean_before_corr = 0.0;  // Mean u before mean correction
        double u_mean_after_corr = 0.0;   // Mean u after mean correction
        double u_rms_before_corr = 0.0;   // u' RMS before mean correction
        double u_rms_after_corr = 0.0;    // u' RMS after mean correction (should match)
        double v_mean_final = 0.0;        // Mean v after transverse removal (should be ~0)
        double w_mean_final = 0.0;        // Mean w after transverse removal (should be ~0)

        // Clamp/scale telemetry
        double scale_factor = 1.0;      // Applied scale for mean correction
        bool clamp_hit = false;         // Whether scale was clamped

        // Metadata
        int step = 0;
        int shift_k = 0;
    };

    /// Get the most recent recycling diagnostics (call after process_recycle_inflow)
    const RecycleDiagnostics& get_recycle_diagnostics() const { return recycle_diag_; }

    /// Log recycling diagnostics to stdout (called every recycle_diag_interval steps)
    void log_recycle_diagnostics() const;

    /// Running statistics for recycling controller health monitoring
    struct RecycleRunningStats {
        int n_samples = 0;           // Number of samples collected
        int n_clamp_hits = 0;        // Number of times clamp was hit
        double scale_sum = 0.0;      // Sum of scale factors
        double scale_sum_sq = 0.0;   // Sum of scale factors squared

        void reset() {
            n_samples = 0;
            n_clamp_hits = 0;
            scale_sum = 0.0;
            scale_sum_sq = 0.0;
        }

        double clamp_hit_rate() const {
            return (n_samples > 0) ? static_cast<double>(n_clamp_hits) / n_samples : 0.0;
        }

        double scale_mean() const {
            return (n_samples > 0) ? scale_sum / n_samples : 1.0;
        }

        double scale_std() const {
            if (n_samples < 2) return 0.0;
            double mean = scale_mean();
            double var = scale_sum_sq / n_samples - mean * mean;
            return (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    };

    /// Get running statistics for recycling controller
    const RecycleRunningStats& get_recycle_running_stats() const { return recycle_stats_; }

    /// Reset running statistics (call after spin-up to measure steady-state behavior)
    void reset_recycle_running_stats() { recycle_stats_.reset(); }

    /// Check for NaN/Inf in solution fields and abort if detected
    /// @param step Current step number (used for guard interval checking)
    /// @throws std::runtime_error if NaN/Inf detected and guard is enabled
    /// @note Checks velocity, pressure, nu_t, and transport fields (k, omega)
    void check_for_nan_inf(int step) const;

    /// Recycling inflow public interface (for testing and advanced use)
    /// @{
    void extract_recycle_plane();          ///< Copy velocity at recycle plane to buffers
    void process_recycle_inflow();         ///< Apply shift, filter, mass-flux correction
    void apply_recycling_inlet_bc();       ///< Apply processed inflow as inlet BC
    void apply_fringe_blending();          ///< Blend inlet velocity in fringe zone
    void correct_inlet_divergence();       ///< Make first slab div-free by correcting inlet u
    bool is_recycling_enabled() const { return use_recycling_; }
    /// @}

private:
    const Mesh* mesh_;
    Config config_;
    
    // Solution fields
    VectorField velocity_;
    /// Scratch buffer for predictor velocity. NEVER assumed valid unless explicitly
    /// filled immediately before use. Used by correct_velocity() as input source.
    VectorField velocity_star_;
    ScalarField pressure_;
    ScalarField pressure_correction_;
    ScalarField nu_t_;           // Eddy viscosity
    ScalarField k_;              // TKE (for transport models)
    ScalarField omega_;          // Specific dissipation (for transport models)
    TensorField tau_ij_;         // Reynolds stresses (for TBNN)
    
    // Auxiliary fields
    ScalarField rhs_poisson_;    // RHS for pressure Poisson equation
    ScalarField div_velocity_;   // Velocity divergence
    
    // Persistent work fields (avoid reallocation every step)
    ScalarField nu_eff_;         // Effective viscosity nu + nu_t
    VectorField conv_;           // Convective term
    VectorField diff_;           // Diffusive term
    VectorField velocity_old_;   // Previous velocity for residual (GPU-resident when offload enabled)
    VectorField velocity_rk_;    // Work buffer for RK stages (stores u^n during multi-stage update)
    
    // Gradient scratch buffers for turbulence models (GPU-resident)
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    ScalarField wall_distance_;  // Precomputed wall distance field
    
    // Solvers
    PoissonSolver poisson_solver_;
    MultigridPoissonSolver mg_poisson_solver_;
#ifdef USE_HYPRE
    std::unique_ptr<HyprePoissonSolver> hypre_poisson_solver_;
#endif
#ifdef USE_FFT_POISSON
    std::unique_ptr<FFTPoissonSolver> fft_poisson_solver_;      // 2D FFT (periodic x AND z) - 3D only
    std::unique_ptr<FFT1DPoissonSolver> fft1d_poisson_solver_;  // 1D FFT (periodic x OR z) - 3D only
    std::unique_ptr<FFT2DPoissonSolver> fft2d_poisson_solver_;  // 2D mesh FFT (periodic x, walls y)
#endif
    PoissonSolverType selected_solver_ = PoissonSolverType::MG;  // Actually selected solver (after auto)
    std::string selection_reason_;  // Human-readable reason for solver selection (observability)
    PoissonStats poisson_stats_;    // Statistics from last Poisson solve (for testing/diagnostics)
    std::unique_ptr<TurbulenceModel> turb_model_;
    bool use_multigrid_ = true;  // Use multigrid by default for speed
    
    // Boundary conditions
    VelocityBC velocity_bc_;
    PoissonBC poisson_bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC poisson_bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC poisson_bc_z_lo_ = PoissonBC::Periodic;
    PoissonBC poisson_bc_z_hi_ = PoissonBC::Periodic;
    double fx_ = 0.0, fy_ = 0.0, fz_ = 0.0;  // Body force (3D)

    // Recycling inflow state and buffers
    bool use_recycling_ = false;           ///< Recycling inflow enabled
    int recycle_i_ = -1;                   ///< Grid index of recycle plane
    int recycle_shift_k_ = 0;              ///< Current spanwise shift (z-index units)
    int recycle_shift_step_ = 0;           ///< Steps since last shift update
    double recycle_target_Q_ = -1.0;       ///< Target mass flux for flow rate control
    double recycle_filter_alpha_ = 0.0;    ///< AR1 filter coefficient (0 = no filter)
    int fringe_i_end_ = -1;                ///< Grid index where fringe zone ends

    // Recycling inflow plane buffers (Ny × Nz for 3D)
    // u at inlet: Ny × Nz (cell-face values at i=Ng)
    // v at inlet: (Ny+1) × Nz (y-face values at i=Ng)
    // w at inlet: Ny × (Nz+1) (z-face values at i=Ng)
    std::vector<double> recycle_u_buf_;    ///< Recycled u at inlet (Ny × Nz)
    std::vector<double> recycle_v_buf_;    ///< Recycled v at inlet ((Ny+1) × Nz)
    std::vector<double> recycle_w_buf_;    ///< Recycled w at inlet (Ny × (Nz+1))
    std::vector<double> inlet_u_buf_;      ///< Processed inlet u (after shift/filter)
    std::vector<double> inlet_v_buf_;      ///< Processed inlet v
    std::vector<double> inlet_w_buf_;      ///< Processed inlet w
    std::vector<double> inlet_u_filt_;     ///< Temporally filtered u (for AR1)
    std::vector<double> inlet_v_filt_;     ///< Temporally filtered v
    std::vector<double> inlet_w_filt_;     ///< Temporally filtered w

    // Diagnostics staging buffers (for L2 breakdown - only allocated if diag enabled)
    std::vector<double> diag_u_copy_;      ///< u after copy+shift stage
    std::vector<double> diag_u_ar1_;       ///< u after AR1 stage
    std::vector<double> diag_u_mean_;      ///< u after mean correction stage
    RecycleDiagnostics recycle_diag_;      ///< Most recent diagnostics snapshot
    RecycleRunningStats recycle_stats_;    ///< Running statistics for controller health

    // Stage F: Time-averaged turbulence statistics
    int stats_samples_ = 0;                ///< Number of samples accumulated
    std::vector<double> stats_U_mean_;     ///< Mean streamwise velocity U(y)
    std::vector<double> stats_uu_;         ///< <u'u'>(y)
    std::vector<double> stats_vv_;         ///< <v'v'>(y)
    std::vector<double> stats_ww_;         ///< <w'w'>(y)
    std::vector<double> stats_uv_;         ///< <u'v'>(y)
    std::vector<double> stats_dUdy_;       ///< dU/dy(y) for momentum balance

    // State
    int iter_ = 0;
    int step_count_ = 0;  // Track current step for guard
    double current_dt_;
    double current_time_ = 0.0;  // Simulation time (accumulated from dt)

    // Force ramping state (for startup stabilization)
    bool force_ramp_enabled_ = false;
    double force_ramp_tau_ = 1.0;     // Ramp time constant (bulk time units)
    double fx_target_ = 0.0;          // Target body force (used when ramping)
    double fy_target_ = 0.0;
    double fz_target_ = 0.0;

    // Wall shear history for turbulence settling detection
    std::vector<WallShearSample> wall_shear_history_;

    // Turbulence sample buffer for windowed classification
    std::vector<TurbulenceSample> turb_samples_;
    TurbulenceClassifier turb_classifier_;
    
    // Original internal methods
    void apply_velocity_bc();
    void compute_convective_term(const VectorField& vel, VectorField& conv);
    void compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, VectorField& diff);
    void compute_divergence(VelocityWhich which, ScalarField& div);

    /// Correct velocity using pressure gradient: vel_out = vel_in - dt * grad(p_correction)
    /// @param vel_in  Input velocity field (unprojected)
    /// @param vel_out Output velocity field (projected, can alias vel_in for in-place)
    void correct_velocity(const VectorField& vel_in, VectorField& vel_out);

    /// Legacy wrapper: correct_velocity(velocity_star_, velocity_)
    void correct_velocity();

    double compute_residual();
    
    // IMEX methods
    void solve_implicit_diffusion(const VectorField& u_explicit, const ScalarField& nu_eff, 
                                  VectorField& u_new, double dt);
    
    // Time integration methods
    void euler_substep(VectorField& vel_in, VectorField& vel_out, double dt);
    void project_velocity(VectorField& vel, double dt);
    void ssprk2_step(double dt);
    void ssprk3_step(double dt);

    /// Fill periodic ghost layers on device for a velocity field (GPU-resident, no swaps)
    /// This is called after predictor and after correction to ensure halos are consistent.
    void enforce_periodic_halos_device(double* u_ptr, double* v_ptr, double* w_ptr = nullptr);

    // Recycling inflow initialization (public methods declared above)
    void initialize_recycling_inflow();    ///< Setup recycling (compute indices, allocate buffers)

    // Gradient computations
    void compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy);
    
    // GPU buffers (always present for ABI stability)
    // Simplified GPU strategy: Data mapped once at initialization, stays resident on GPU
    // All kernels use is_device_ptr to access already-mapped data
    // No temporary map(to:/from:) clauses in kernels - eliminates mapping conflicts
    bool gpu_ready_ = false;
    
    // Persistent pointers to GPU-resident arrays (mapped for solver lifetime)
    double* velocity_u_ptr_ = nullptr;
    double* velocity_v_ptr_ = nullptr;
    double* velocity_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_star_u_ptr_ = nullptr;
    double* velocity_star_v_ptr_ = nullptr;
    double* velocity_star_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_old_u_ptr_ = nullptr;  // Device-resident old velocity for residual
    double* velocity_old_v_ptr_ = nullptr;  // Device-resident old velocity for residual
    double* velocity_old_w_ptr_ = nullptr;  // 3D w-velocity
    double* velocity_rk_u_ptr_ = nullptr;   // RK work buffer for u
    double* velocity_rk_v_ptr_ = nullptr;   // RK work buffer for v
    double* velocity_rk_w_ptr_ = nullptr;   // RK work buffer for w (3D)
    double* pressure_ptr_ = nullptr;
    double* pressure_corr_ptr_ = nullptr;
    double* nu_t_ptr_ = nullptr;
    double* nu_eff_ptr_ = nullptr;
    double* conv_u_ptr_ = nullptr;
    double* conv_v_ptr_ = nullptr;
    double* conv_w_ptr_ = nullptr;  // 3D convective term
    double* diff_u_ptr_ = nullptr;
    double* diff_v_ptr_ = nullptr;
    double* diff_w_ptr_ = nullptr;  // 3D diffusive term
    double* rhs_poisson_ptr_ = nullptr;
    double* div_velocity_ptr_ = nullptr;
    double* k_ptr_ = nullptr;
    double* omega_ptr_ = nullptr;
    
    // Reynolds stress tensor pointers (for EARSM/TBNN models)
    double* tau_xx_ptr_ = nullptr;
    double* tau_xy_ptr_ = nullptr;
    double* tau_xz_ptr_ = nullptr;  // 3D
    double* tau_yy_ptr_ = nullptr;
    double* tau_yz_ptr_ = nullptr;  // 3D
    double* tau_zz_ptr_ = nullptr;  // 3D

    // Gradient scratch buffers (cell-centered, for turbulence models)
    double* dudx_ptr_ = nullptr;
    double* dudy_ptr_ = nullptr;
    double* dudz_ptr_ = nullptr;  // 3D
    double* dvdx_ptr_ = nullptr;
    double* dvdy_ptr_ = nullptr;
    double* dvdz_ptr_ = nullptr;  // 3D
    double* dwdx_ptr_ = nullptr;  // 3D
    double* dwdy_ptr_ = nullptr;  // 3D
    double* dwdz_ptr_ = nullptr;  // 3D
    double* wall_distance_ptr_ = nullptr;

    // Recycling inflow GPU pointers (mapped for solver lifetime when enabled)
    double* recycle_u_ptr_ = nullptr;      ///< Device: recycled u at recycle plane
    double* recycle_v_ptr_ = nullptr;      ///< Device: recycled v at recycle plane
    double* recycle_w_ptr_ = nullptr;      ///< Device: recycled w at recycle plane
    double* inlet_u_ptr_ = nullptr;        ///< Device: processed inlet u
    double* inlet_v_ptr_ = nullptr;        ///< Device: processed inlet v
    double* inlet_w_ptr_ = nullptr;        ///< Device: processed inlet w
    size_t recycle_u_size_ = 0;            ///< Size of u recycle buffer
    size_t recycle_v_size_ = 0;            ///< Size of v recycle buffer
    size_t recycle_w_size_ = 0;            ///< Size of w recycle buffer

    // Area-weighting for mass flux correction (stretched mesh support)
    std::vector<double> inlet_area_weights_;  ///< Host: dy[j] * dz[k] for each inlet cell
    double* inlet_area_ptr_ = nullptr;        ///< Device: area weights for weighted reduction
    double total_inlet_area_ = 0.0;           ///< Total inlet area: sum(dy[j] * dz[k])
    bool inlet_needs_area_weight_ = false;    ///< True if mesh is stretched (dyv or dzv non-empty)

    // Y-metric arrays for non-uniform y-spacing (D·G = L consistency for projection step)
    // These are const pointers to mesh_->dyv/dyc data, mapped to GPU for stretched grids
    const double* dyv_ptr_ = nullptr;  ///< Cell height at row j: yf[j+1] - yf[j]
    const double* dyc_ptr_ = nullptr;  ///< Center-to-center spacing at face j: yc[j] - yc[j-1]
    size_t y_metrics_size_ = 0;        ///< Size of dyv/dyc arrays (total_Ny())

    size_t field_total_size_ = 0;  // (Nx+2)*(Ny+2) for fields with ghost cells

    // Scratch buffer for sync_from_gpu workaround (NVHPC member pointer requirement)
    mutable std::vector<double> sync_scratch_;
    mutable double* sync_scratch_ptr_ = nullptr;

    void extract_field_pointers();  // Set raw pointers to field data (shared by CPU/GPU paths)
    void initialize_gpu_buffers();  // Map data to GPU (called once in constructor)
    void cleanup_gpu_buffers();     // Unmap and copy results back (called in destructor)

    /// Map VectorField reference to its corresponding raw pointer triplet
    /// This enables compute functions to use argument-based device pointers
    /// instead of always using member pointers (critical for RK stages)
    void get_velocity_ptrs(const VectorField& vel,
                           double*& u_ptr, double*& v_ptr, double*& w_ptr) const;

#ifndef NDEBUG
    /// Verify field pointers haven't changed since GPU mapping (catches std::vector reallocation)
    void verify_mapping_integrity() const;
#endif
    
public:
    /// Get device view for turbulence models (GPU-resident pointers)
    /// Returns a view with pointers to solver-owned GPU data.
    /// Only valid if gpu_ready_ == true.
    TurbulenceDeviceView get_device_view() const;
    
    /// Get solver view for core NS/projection kernels
    /// Returns GPU-resident pointers if gpu_ready_, else host pointers
    SolverDeviceView get_solver_view() const;
};

} // namespace nncfd


