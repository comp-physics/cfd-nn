/// @file solver_time_simple.cpp
/// @brief SIMPLE steady-state solver for incompressible RANS
///
/// Diagonal-approximation SIMPLE (comparable to OpenFOAM simpleFoam).
/// Eliminates warm-up, tau_div ramp, and diffusion-stability dt constraints.
/// NN turbulence models active from iteration 1 with cold-start k/omega.

#include "solver.hpp"
#include "solver_time_kernels.hpp"
#include "poisson_solver.hpp"
#include "timing.hpp"

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// SIMPLE step: one complete SIMPLE iteration
// Returns: residual = max|u_new - u_old|
// ============================================================================

double RANSSolver::simple_step() {
    TIMED_SCOPE("solver_step");

    const int Ng = mesh_->Nghost;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const bool is_2d = mesh_->is2D();

    // ================================================================
    // 0. Ensure BCs are applied (ghost cells valid for Jacobi stencil)
    // ================================================================
    apply_velocity_bc();

    // ================================================================
    // 0b. Copy velocity to velocity_old_ for residual + under-relaxation
    // ================================================================
    {
        [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
        [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
        const int u_stride = Nx + 2 * Ng + 1;
        const int v_stride = Nx + 2 * Ng;

        if (is_2d) {
            time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_old_u_ptr_,
                                     velocity_v_ptr_, velocity_old_v_ptr_,
                                     Nx, Ny, Ng, u_stride, v_stride,
                                     u_total_size, v_total_size);
        } else {
            const int u_plane = u_stride * (Ny + 2 * Ng);
            const int v_plane = v_stride * (Ny + 2 * Ng + 1);
            const int w_stride = Nx + 2 * Ng;
            const int w_plane = w_stride * (Ny + 2 * Ng);
            time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_old_u_ptr_,
                                      velocity_v_ptr_, velocity_old_v_ptr_,
                                      velocity_w_ptr_, velocity_old_w_ptr_,
                                      Nx, Ny, Nz, Ng,
                                      u_stride, v_stride, w_stride,
                                      u_plane, v_plane, w_plane,
                                      u_total_size, v_total_size,
                                      velocity_.w_total_size());
        }
    }

    // ================================================================
    // 1. Zero tau_div arrays (unless frozen)
    // ================================================================
    if (tau_div_u_ptr_ && !tau_div_frozen_) {
        [[maybe_unused]] const size_t u_sz = velocity_.u_total_size();
        [[maybe_unused]] const size_t v_sz = velocity_.v_total_size();
        [[maybe_unused]] const size_t w_sz = velocity_.w_total_size();
        double* tdu = tau_div_u_ptr_;
        double* tdv = tau_div_v_ptr_;
        double* tdw = tau_div_w_ptr_;
        #pragma omp target teams distribute parallel for map(present: tdu[0:u_sz])
        for (size_t i = 0; i < u_sz; ++i) tdu[i] = 0.0;
        #pragma omp target teams distribute parallel for map(present: tdv[0:v_sz])
        for (size_t i = 0; i < v_sz; ++i) tdv[i] = 0.0;
        #pragma omp target teams distribute parallel for map(present: tdw[0:w_sz])
        for (size_t i = 0; i < w_sz; ++i) tdw[i] = 0.0;
    }

    // ================================================================
    // 2. Update turbulence model (same preamble as step())
    //    Key difference: NO tau_div ramp, NO adaptive_dt recheck
    // ================================================================

    // 2a. Advance turbulence transport (k, omega)
    {
        TurbulenceModel* transport_to_advance = nullptr;
        if (turb_model_ && turb_model_->uses_transport_equations()) {
            transport_to_advance = turb_model_.get();
        } else if (bg_transport_ && bg_transport_->uses_transport_equations()) {
            transport_to_advance = bg_transport_.get();
        }

        if (transport_to_advance) {
            TIMED_SCOPE("turbulence_transport");

            const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
            TurbulenceDeviceView device_view = get_device_view();
            if (device_view.is_valid()) {
                device_view_ptr = &device_view;
            }
#endif
            // Transport dt: must respect SST stability (omega can be O(1000) at walls).
            // Compute CFL-like dt from current velocity + diffusion stability.
            double u_max_t = 0.0;
            double nu_max_t = config_.nu;
            for (size_t idx = 0; idx < field_total_size_; ++idx) {
                if (nu_eff_ptr_[idx] > nu_max_t) nu_max_t = nu_eff_ptr_[idx];
            }
            double dx_min_t = is_2d ? std::min(mesh_->dx, mesh_->dy)
                                    : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
            // Quick u_max from a few sample points (avoid full scan)
            {
                const int us = Nx + 2 * Ng + 1;
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); j += std::max(1, Ny/8))
                    for (int i = mesh_->i_begin(); i <= mesh_->i_end(); i += std::max(1, Nx/8)) {
                        double val = velocity_u_ptr_[j * us + i];
                        if (val < 0) val = -val;
                        if (val > u_max_t) u_max_t = val;
                    }
            }
            if (u_max_t < 1e-6) u_max_t = 1e-6;
            double dt_c = dx_min_t / u_max_t;
            double dt_d = 0.25 * dx_min_t * dx_min_t / nu_max_t;
            double transport_dt = std::min(dt_c, dt_d);
            if (transport_dt > 0.1) transport_dt = 0.1;  // cap for safety
            if (config_.verbose && step_count_ < 5) {
                std::cerr << "[SIMPLE] transport_dt=" << transport_dt
                          << " nu_max=" << nu_max_t << " u_max=" << u_max_t << "\n";
            }
            transport_to_advance->advance_turbulence(
                *mesh_, velocity_, transport_dt, k_, omega_, nu_t_,
                device_view_ptr);

#ifdef USE_GPU_OFFLOAD
            if (!transport_to_advance->is_gpu_ready()) {
                #pragma omp target update to(k_ptr_[0:field_total_size_])
                #pragma omp target update to(omega_ptr_[0:field_total_size_])
            }
#endif
        }
    }

    // 2b. Update turbulence model (compute nu_t, tau_ij)
    if (turb_model_) {
        TIMED_SCOPE("turbulence_update");

        const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
        TurbulenceDeviceView device_view = get_device_view();
        if (device_view.is_valid()) {
            device_view_ptr = &device_view;
        }
        if (gpu_ready_ && !turb_model_->is_gpu_ready()) {
            sync_solution_from_gpu();
        }
        if (gpu_ready_ && turb_model_->is_gpu_ready() &&
            (!device_view_ptr || !device_view_ptr->is_valid())) {
            throw std::runtime_error("GPU simulation requires valid TurbulenceDeviceView");
        }
#endif

        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_,
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr,
                           device_view_ptr);

        // Decomposition: restore SST nu_t for tensor models
        if (bg_transport_ && turb_model_->provides_reynolds_stresses()) {
            bg_transport_->update(*mesh_, velocity_, k_, omega_, nu_t_, nullptr, device_view_ptr);
        }

        // Clamp nu_t
        if (config_.nu_t_max > 0.0 && config_.nu_t_max < 1e10) {
            const double nu_t_limit = config_.nu_t_max;
            [[maybe_unused]] const size_t nt_sz = field_total_size_;
            double* nt_ptr = nu_t_ptr_;
            #pragma omp target teams distribute parallel for map(present: nt_ptr[0:nt_sz])
            for (size_t i = 0; i < nt_sz; ++i) {
                if (nt_ptr[i] > nu_t_limit) nt_ptr[i] = nu_t_limit;
                if (nt_ptr[i] < 0.0) nt_ptr[i] = 0.0;
            }
        }

#ifdef USE_GPU_OFFLOAD
        bool model_used_gpu = (device_view_ptr && device_view_ptr->is_valid() &&
                               turb_model_->is_gpu_ready());
        if (!model_used_gpu) {
            #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
        }
#endif
    }

    // 2c. Under-relax nu_t
    if (config_.nu_t_relaxation < 1.0 && config_.nu_t_relaxation > 0.0) {
        const double alpha = config_.nu_t_relaxation;
        const double nu_mol = config_.nu;
        [[maybe_unused]] const size_t nt_sz = field_total_size_;
        double* nt = nu_t_ptr_;
        double* ne = nu_eff_ptr_;
        #pragma omp target teams distribute parallel for map(present: nt[0:nt_sz], ne[0:nt_sz])
        for (size_t i = 0; i < nt_sz; ++i) {
            double nu_t_old = ne[i] - nu_mol;
            if (nu_t_old < 0.0) nu_t_old = 0.0;
            nt[i] = alpha * nt[i] + (1.0 - alpha) * nu_t_old;
        }
    }

    // 2d. Compute nu_eff = nu + nu_t
    {
        TIMED_SCOPE("nu_eff_computation");
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const size_t total_size = field_total_size_;
        const double nu = config_.nu;
        double* nu_eff_ptr = nu_eff_ptr_;
        const double* nu_t_ptr = nu_t_ptr_;

#ifdef USE_GPU_OFFLOAD
        if (gpu_ready_) {
            // Fill all cells with nu
            #pragma omp target teams distribute parallel for \
                map(present: nu_eff_ptr[0:total_size])
            for (size_t idx = 0; idx < total_size; ++idx) {
                nu_eff_ptr[idx] = nu;
            }
            // Add nu_t to interior
            if (turb_model_) {
                if (is_2d) {
                    const int n_cells = Nx * Ny;
                    #pragma omp target teams distribute parallel for \
                        map(present: nu_eff_ptr[0:total_size], nu_t_ptr[0:total_size])
                    for (int idx = 0; idx < n_cells; ++idx) {
                        int i = idx % Nx + Ng;
                        int j = idx / Nx + Ng;
                        nu_eff_ptr[j * stride + i] = nu + nu_t_ptr[j * stride + i];
                    }
                } else {
                    const int n_cells = Nx * Ny * Nz;
                    #pragma omp target teams distribute parallel for \
                        map(present: nu_eff_ptr[0:total_size], nu_t_ptr[0:total_size])
                    for (int idx = 0; idx < n_cells; ++idx) {
                        int i = idx % Nx + Ng;
                        int j = (idx / Nx) % Ny + Ng;
                        int k = idx / (Nx * Ny) + Ng;
                        nu_eff_ptr[k * plane_stride + j * stride + i] =
                            nu + nu_t_ptr[k * plane_stride + j * stride + i];
                    }
                }
            }
        } else
#endif
        {
            nu_eff_.fill(config_.nu);
            if (turb_model_) {
                if (is_2d) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j)
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i)
                            nu_eff_(i, j) = config_.nu + nu_t_(i, j);
                } else {
                    for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k)
                        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j)
                            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i)
                                nu_eff_(i, j, k) = config_.nu + nu_t_(i, j, k);
                }
            }
        }
    }

    // ================================================================
    // 3. Compute tau_div (anisotropic stress divergence)
    //    NO RAMP — full strength from iteration 1
    // ================================================================
    if (turb_model_ && turb_model_->provides_reynolds_stresses() && !tau_div_frozen_) {
        compute_tau_divergence();
    }

    // ================================================================
    // 4-5. Momentum solve
    //      Jacobi (simple_jacobi_sweeps > 0): implicit convection + diffusion
    //      Diagonal approximation (simple_jacobi_sweeps <= 0): explicit predictor
    // ================================================================
    {
        const int u_stride = Nx + 2 * Ng + 1;
        const int v_stride = Nx + 2 * Ng;
        const int cell_stride = Nx + 2 * Ng;

        // Pseudo-transient damping: vol / dt_pseudo added to diagonal
        double u_max = 0.0;
        double nu_eff_max = config_.nu;
        if (is_2d) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j)
                for (int i = mesh_->i_begin(); i <= mesh_->i_end(); ++i) {
                    double val = velocity_u_ptr_[j * u_stride + i];
                    if (val < 0) val = -val;
                    if (val > u_max) u_max = val;
                }
        }
        for (size_t idx = 0; idx < field_total_size_; ++idx) {
            if (nu_eff_ptr_[idx] > nu_eff_max) nu_eff_max = nu_eff_ptr_[idx];
        }
        if (u_max < 1e-6) u_max = 1e-6;
        double dx_min = is_2d ? std::min(mesh_->dx, mesh_->dy)
                              : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
        // Pseudo_dt controls under-relaxation strength.
        double pseudo_dt;
        if (config_.simple_jacobi_sweeps > 0) {
            // RB-GS path: fixed pseudo_dt from first iteration
            if (simple_pseudo_dt_fixed_ > 0) {
                pseudo_dt = simple_pseudo_dt_fixed_;
            } else {
                pseudo_dt = std::min(dx_min / std::max(u_max, 1.0),
                                     0.25 * dx_min * dx_min / nu_eff_max);
                simple_pseudo_dt_fixed_ = pseudo_dt;
            }
        } else {
            // Diagonal-approx path: adaptive (recomputed each step)
            pseudo_dt = std::min(dx_min / u_max,
                                 0.25 * dx_min * dx_min / nu_eff_max);
        }
        if (pseudo_dt < 1e-15) pseudo_dt = 1e-15;
        double pseudo_dt_inv = 1.0 / pseudo_dt;
        // Store for Rhie-Chow correction (used in correct_velocity_simple)
        current_dt_ = pseudo_dt;

        if (config_.verbose && step_count_ < 5) {
            std::cerr << "[SIMPLE] pseudo_dt=" << pseudo_dt
                      << " nu_max=" << nu_eff_max << " u_max=" << u_max << "\n";
        }

        const int n_sweeps = config_.simple_jacobi_sweeps;

        if (n_sweeps > 0) {
        // Red-Black Gauss-Seidel: in-place update on velocity_star_
        // First copy velocity_ → velocity_star_ as initial guess
        {
            [[maybe_unused]] const size_t u_total = velocity_.u_total_size();
            [[maybe_unused]] const size_t v_total = velocity_.v_total_size();
            time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_star_u_ptr_,
                                     velocity_v_ptr_, velocity_star_v_ptr_,
                                     Nx, Ny, Ng, u_stride, v_stride,
                                     u_total, v_total);
        }

        for (int sweep = 0; sweep < n_sweeps; ++sweep) {
            if (is_2d) {
                // Red pass (color=0): update faces where (i+j) % 2 == 0
                time_kernels::simple_rbgs_momentum_2d(
                    velocity_star_u_ptr_, velocity_star_v_ptr_,  // in-place
                    velocity_old_u_ptr_, velocity_old_v_ptr_,
                    nu_eff_ptr_, pressure_ptr_,
                    tau_div_u_ptr_, tau_div_v_ptr_,
                    fx_, fy_, mesh_->dx, mesh_->dy,
                    pseudo_dt_inv, /*color=*/0,
                    Nx, Ny, Ng, u_stride, v_stride, cell_stride);

                // Black pass (color=1): update faces where (i+j) % 2 == 1
                time_kernels::simple_rbgs_momentum_2d(
                    velocity_star_u_ptr_, velocity_star_v_ptr_,  // in-place
                    velocity_old_u_ptr_, velocity_old_v_ptr_,
                    nu_eff_ptr_, pressure_ptr_,
                    tau_div_u_ptr_, tau_div_v_ptr_,
                    fx_, fy_, mesh_->dx, mesh_->dy,
                    pseudo_dt_inv, /*color=*/1,
                    Nx, Ny, Ng, u_stride, v_stride, cell_stride);
            } else {
                break;  // TODO: 3D RB-GS
            }

            // Apply BCs between sweeps (in-place on velocity_star_)
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
            std::swap(velocity_, velocity_star_);
            apply_velocity_bc();
            std::swap(velocity_, velocity_star_);
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
        }

        // Result is in velocity_star_ (in-place). No copy needed.

        // Compute a_P for pressure correction (still needed for u = u* - (1/a_P)*grad(p'))
        if (is_2d) {
            time_kernels::simple_compute_aP_2d(
                a_p_u_ptr_, a_p_v_ptr_, nu_eff_ptr_,
                velocity_u_ptr_, velocity_v_ptr_,
                Nx, Ny, Ng, u_stride, v_stride, cell_stride,
                mesh_->dx, mesh_->dy, pseudo_dt_inv);
        }
        } else {
        // Diagonal-approximation predictor: u* = u + (H - dp/dx) / a_P
        // Works for laminar flows. For turbulent flows (SST), use time_integrator=euler
        // with solve_steady() instead — the SIMPLE diagonal approx is equivalent to
        // explicit Euler when a_P includes the pseudo-transient term.
        {
            TIMED_SCOPE("convective_term");
            compute_convective_term(velocity_, conv_);
        }
        {
            TIMED_SCOPE("diffusive_term");
            compute_diffusive_term(velocity_, nu_eff_, diff_);
        }

        // Compute a_P with convective diagonal
        if (is_2d) {
            time_kernels::simple_compute_aP_2d(
                a_p_u_ptr_, a_p_v_ptr_, nu_eff_ptr_,
                velocity_u_ptr_, velocity_v_ptr_,
                Nx, Ny, Ng, u_stride, v_stride, cell_stride,
                mesh_->dx, mesh_->dy, pseudo_dt_inv);
        }

        // Predictor: u* = u + (H - dp/dx) / a_P, under-relaxed
        if (is_2d) {
            time_kernels::simple_predictor_2d(
                velocity_star_u_ptr_, velocity_star_v_ptr_,
                velocity_u_ptr_, velocity_v_ptr_,
                velocity_old_u_ptr_, velocity_old_v_ptr_,
                conv_.u_data().data(), conv_.v_data().data(),
                diff_.u_data().data(), diff_.v_data().data(),
                tau_div_u_ptr_, tau_div_v_ptr_,
                a_p_u_ptr_, a_p_v_ptr_,
                pressure_ptr_,
                fx_, fy_, mesh_->dx, mesh_->dy,
                config_.simple_alpha_u,
                Nx, Ny, Ng, u_stride, v_stride, cell_stride);
        }
        }
    }

    // ================================================================
    // 6. Apply BCs to predictor velocity (velocity_star_)
    // ================================================================
    {
        // Temporarily swap velocity_ ↔ velocity_star_ so apply_velocity_bc works on star
        std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
        std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
        if (!is_2d) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
        std::swap(velocity_, velocity_star_);

        apply_velocity_bc();

        // IBM pre-forcing on predictor
        if (ibm_) {
            if (gpu_ready_ && ibm_->is_gpu_ready()) {
                ibm_->apply_forcing_device(velocity_u_ptr_, velocity_v_ptr_,
                                           is_2d ? nullptr : velocity_w_ptr_, 1.0);
            }
        }

        // Swap back — velocity_ has original, velocity_star_ has predicted+BC'd
        std::swap(velocity_, velocity_star_);
        std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
        std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
        if (!is_2d) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
    }

    // ================================================================
    // 7. Compute divergence of u*, build Poisson RHS, solve Poisson
    //    For Jacobi path: use vol/mean(a_P) as pseudo_dt for proper coupling.
    //    For diagonal path: use CFL-limited pseudo_dt.
    // ================================================================
    {
        // For Jacobi: Rhie-Chow principle — the Poisson pseudo_dt uses the
        // UNDAMPED a_P (diffusion + convection only, without vol/pseudo_dt).
        // This allows the pressure to build up correctly even though the
        // Jacobi momentum solve uses damped a_P for stability.
        double pseudo_dt_proj;
        if (config_.simple_jacobi_sweeps > 0) {
            // Compute undamped a_P = (diffusion + convection diagonal) at u-faces
            const int u_stride_p = Nx + 2 * Ng + 1;
            const int cell_stride_p = Nx + 2 * Ng;
            double inv_dx2 = 1.0 / (mesh_->dx * mesh_->dx);
            double inv_dy2 = 1.0 / (mesh_->dy * mesh_->dy);
            double vol = is_2d ? mesh_->dx * mesh_->dy
                               : mesh_->dx * mesh_->dy * mesh_->dz;
            double sum_aP_undamped = 0.0;
            int n_interior = (Nx + 1) * Ny;
            for (int j_m = 0; j_m < Ny; ++j_m) {
                for (int i_m = 0; i_m <= Nx; ++i_m) {
                    int jg = j_m + Ng;
                    int ig = i_m + Ng;
                    // Diffusion diagonal (same nu_avg as Jacobi)
                    int cl = jg * cell_stride_p + (ig - 1);
                    int cr = jg * cell_stride_p + ig;
                    double nu_avg = 0.5 * (nu_eff_ptr_[cl] + nu_eff_ptr_[cr]);
                    double aP_diff = nu_avg * (2.0 * inv_dx2 + 2.0 * inv_dy2) * vol;
                    // Convection diagonal (upwind)
                    double u_e = velocity_old_u_ptr_[jg * u_stride_p + (ig+1)];
                    double u_w = velocity_old_u_ptr_[jg * u_stride_p + (ig-1)];
                    double Fw = u_w * mesh_->dy;
                    double Fe = u_e * mesh_->dy;
                    double aP_conv = (Fw < 0 ? -Fw : 0.0) + (Fe > 0 ? Fe : 0.0);
                    // v fluxes negligible for first approximation
                    sum_aP_undamped += aP_diff + aP_conv;
                }
            }
            double mean_aP_undamped = (n_interior > 0) ? sum_aP_undamped / n_interior : 1.0;
            if (mean_aP_undamped < 1e-20) mean_aP_undamped = 1e-20;
            pseudo_dt_proj = vol / mean_aP_undamped;
        } else {
            // Diagonal approx: use CFL-limited pseudo_dt
            double u_max_p = 0.0;
            double nu_eff_max_p = config_.nu;
            const int u_stride_p = Nx + 2 * Ng + 1;
            if (is_2d) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j)
                    for (int i = mesh_->i_begin(); i <= mesh_->i_end(); ++i) {
                        double val = velocity_star_u_ptr_[j * u_stride_p + i];
                        if (val < 0) val = -val;
                        if (val > u_max_p) u_max_p = val;
                    }
            }
            for (size_t idx = 0; idx < field_total_size_; ++idx) {
                if (nu_eff_ptr_[idx] > nu_eff_max_p) nu_eff_max_p = nu_eff_ptr_[idx];
            }
            if (u_max_p < 1e-6) u_max_p = 1e-6;
            double dx_min_p = is_2d ? std::min(mesh_->dx, mesh_->dy)
                                    : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
            pseudo_dt_proj = std::min(dx_min_p / u_max_p,
                                      0.25 * dx_min_p * dx_min_p / nu_eff_max_p);
        }
        if (pseudo_dt_proj < 1e-15) pseudo_dt_proj = 1e-15;
        current_dt_ = pseudo_dt_proj;

        // Compute divergence of velocity_star_
        compute_divergence(VelocityWhich::Star, div_velocity_);

        // Build Poisson RHS: rhs = (div - mean_div) * mean(a_P) / vol
        // This approximates ∇·(1/a_P · ∇p') ≈ (1/mean(1/a_P)) * ∇²p' = mean_aP_undamped * ∇²p'
        // But the MG solves ∇²p' = rhs, so rhs = div * mean_aP_undamped
        // For simplicity, use 1/pseudo_dt_proj which = mean_aP_undamped/vol
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        [[maybe_unused]] const size_t cell_sz = field_total_size_;
        double* rhs_sv = rhs_poisson_ptr_;
        double* div_sv = div_velocity_ptr_;
        const double dt_inv = 1.0 / pseudo_dt_proj;
        const int count = is_2d ? Nx * Ny : Nx * Ny * Nz;

        double mean_div = 0.0;
        if (is_2d) {
            double sum_d = 0.0;
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:sum_d) \
                map(present: div_sv[0:cell_sz])
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    sum_d += div_sv[(j + Ng) * stride + (i + Ng)];
                }
            }
            mean_div = (count > 0) ? sum_d / count : 0.0;

            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: div_sv[0:cell_sz], rhs_sv[0:cell_sz])
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (j + Ng) * stride + (i + Ng);
                    rhs_sv[idx] = (div_sv[idx] - mean_div) * dt_inv
                            * (config_.simple_jacobi_sweeps > 0 ? config_.simple_alpha_u : 1.0);
                }
            }
        } else {
            double sum_d = 0.0;
            #pragma omp target teams distribute parallel for collapse(3) reduction(+:sum_d) \
                map(present: div_sv[0:cell_sz])
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        sum_d += div_sv[(k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng)];
                    }
                }
            }
            mean_div = (count > 0) ? sum_d / count : 0.0;

            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: div_sv[0:cell_sz], rhs_sv[0:cell_sz])
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                        rhs_sv[idx] = (div_sv[idx] - mean_div) * dt_inv
                            * (config_.simple_jacobi_sweeps > 0 ? config_.simple_alpha_u : 1.0);
                    }
                }
            }
        }

        // IBM RHS masking
        if (ibm_ && ibm_->is_gpu_ready()) {
            ibm_->mask_rhs_device(rhs_poisson_ptr_);
        }

        // Solve Poisson (first pass: constant-coefficient approximation)
        PoissonConfig pcfg;
        pcfg.max_vcycles = config_.poisson_max_vcycles;
        pcfg.tol_rhs = config_.poisson_tol_rhs;
        if (config_.poisson_fixed_cycles > 0) {
            pcfg.fixed_cycles = config_.poisson_fixed_cycles;
        }

#ifdef USE_GPU_OFFLOAD
        if (gpu_ready_) {
            switch (selected_solver_) {
                case PoissonSolverType::FFT:
                case PoissonSolverType::FFT2D:
                case PoissonSolverType::FFT1D:
                    if (fft_poisson_solver_)
                        fft_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                    break;
                default:
                    mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                    break;
            }
        } else
#endif
        {
            mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
        }

        // Defect-correction: iterate to solve variable-coefficient Poisson
        // ∇·(D · ∇p') = div(u*) - mean_div, where D = 1/a_P
        // The MG solves ∇²q = rhs (coefficient = 1), so we scale:
        //   MG: ∇²q = r / D_mean, then δp = q / D_mean
        // where D_mean = mean(1/a_P) and r is the varcoeff residual.
    }

    // ================================================================
    // 8. SIMPLE velocity correction: u = u* - (1/a_P) * grad(p')
    //    Uses a_P (which includes pseudo-transient damping) for stability.
    //    Then accumulate pressure: p += alpha_p * p'
    // ================================================================
    correct_velocity_simple();

    // ================================================================
    // 8. Post-correction: IBM + BCs + halo exchange
    // ================================================================
    if (ibm_ && ibm_->is_ghost_cell_ibm()) {
        TIMED_SCOPE("ibm_ghost_cell");
        ibm_->apply_ghost_cell(velocity_u_ptr_, velocity_v_ptr_,
                               is_2d ? nullptr : velocity_w_ptr_);
    }

    apply_velocity_bc();

#ifdef USE_MPI
    if (halo_exchange_) {
        // Exchange halos for velocity
        const int u_stride = Nx + 2 * Ng + 1;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_stride = Nx + 2 * Ng;
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        halo_exchange_->exchange(velocity_u_ptr_, u_stride, u_plane);
        halo_exchange_->exchange(velocity_v_ptr_, v_stride, v_plane);
        if (!is_2d) {
            const int w_stride = Nx + 2 * Ng;
            const int w_plane = w_stride * (Ny + 2 * Ng);
            halo_exchange_->exchange(velocity_w_ptr_, w_stride, w_plane);
        }
    }
#endif

    // ================================================================
    // 9. Residual
    //    RB-GS path: MOMENTUM EQUATION RESIDUAL (prevents premature convergence)
    //    Diagonal-approx path: velocity change (momentum residual doesn't converge
    //    for diagonal approximation since it doesn't solve the full momentum eq)
    // ================================================================
    {
        TIMED_SCOPE("residual_computation");
        const int u_stride = Nx + 2 * Ng + 1;
        const int v_stride = Nx + 2 * Ng;
        const int cell_stride = Nx + 2 * Ng;

        double max_res = 0.0;

        if (config_.simple_jacobi_sweeps > 0 && is_2d) {
        // RB-GS path: momentum equation residual
        const double inv_dx2 = 1.0 / (mesh_->dx * mesh_->dx);
        const double inv_dy2 = 1.0 / (mesh_->dy * mesh_->dy);
        const double vol = is_2d ? mesh_->dx * mesh_->dy
                                 : mesh_->dx * mesh_->dy * mesh_->dz;
        const double pdt_inv = (simple_pseudo_dt_fixed_ > 0)
                               ? 1.0 / simple_pseudo_dt_fixed_ : 0.0;

        {
            // u-momentum residual
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i <= Nx; ++i) {
                    int jg = j + Ng;
                    int ig = i + Ng;
                    int u_idx = jg * u_stride + ig;
                    int cl = jg * cell_stride + (ig - 1);
                    int cr = jg * cell_stride + ig;

                    // Diffusion coefficients
                    double nu_L = nu_eff_ptr_[cl];
                    double nu_R = nu_eff_ptr_[cr];
                    double nu_S = 0.25 * (nu_eff_ptr_[cl] + nu_eff_ptr_[cr]
                        + nu_eff_ptr_[(jg-1)*cell_stride+(ig-1)] + nu_eff_ptr_[(jg-1)*cell_stride+ig]);
                    double nu_N = 0.25 * (nu_eff_ptr_[cl] + nu_eff_ptr_[cr]
                        + nu_eff_ptr_[(jg+1)*cell_stride+(ig-1)] + nu_eff_ptr_[(jg+1)*cell_stride+ig]);

                    double a_W_d = nu_L * inv_dx2 * vol;
                    double a_E_d = nu_R * inv_dx2 * vol;
                    double a_S_d = nu_S * inv_dy2 * vol;
                    double a_N_d = nu_N * inv_dy2 * vol;

                    // Convection (upwind from frozen)
                    double F_w = velocity_old_u_ptr_[jg*u_stride+(ig-1)] * mesh_->dy;
                    double F_e = velocity_old_u_ptr_[jg*u_stride+(ig+1)] * mesh_->dy;
                    double F_s = 0.5*(velocity_old_v_ptr_[jg*v_stride+(ig-1)]
                                    + velocity_old_v_ptr_[jg*v_stride+ig]) * mesh_->dx;
                    double F_n = 0.5*(velocity_old_v_ptr_[(jg+1)*v_stride+(ig-1)]
                                    + velocity_old_v_ptr_[(jg+1)*v_stride+ig]) * mesh_->dx;

                    double a_W = a_W_d + (F_w > 0 ? F_w : 0);
                    double a_E = a_E_d + (F_e < 0 ? -F_e : 0);
                    double a_S = a_S_d + (F_s > 0 ? F_s : 0);
                    double a_N = a_N_d + (F_n < 0 ? -F_n : 0);

                    double a_P_val = (a_W_d + a_E_d + a_S_d + a_N_d)
                        + ((F_w<0?-F_w:0) + (F_e>0?F_e:0) + (F_s<0?-F_s:0) + (F_n>0?F_n:0))
                        + vol * pdt_inv;

                    double sum_nb = a_W * velocity_u_ptr_[jg*u_stride+(ig-1)]
                                  + a_E * velocity_u_ptr_[jg*u_stride+(ig+1)]
                                  + a_S * velocity_u_ptr_[(jg-1)*u_stride+ig]
                                  + a_N * velocity_u_ptr_[(jg+1)*u_stride+ig];

                    double source = (tau_div_u_ptr_[u_idx] + fx_) * vol;
                    double dp_dx = (pressure_ptr_[cr] - pressure_ptr_[cl]) / mesh_->dx * vol;
                    double transient = vol * pdt_inv * velocity_old_u_ptr_[u_idx];

                    // Residual: how far is a_P*u from (sum_nb + source - dp + transient)?
                    double R = a_P_val * velocity_u_ptr_[u_idx]
                             - (sum_nb + source - dp_dx + transient);
                    if (R < 0) R = -R;
                    // Normalize by a_P to get velocity units
                    if (a_P_val > 1e-20) R /= a_P_val;
                    if (R > max_res) max_res = R;
                }
            }

            // v-momentum residual (simplified — just use velocity change as proxy)
            for (int j = mesh_->j_begin(); j <= mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    int idx = j * v_stride + i;
                    double dv = velocity_v_ptr_[idx] - velocity_old_v_ptr_[idx];
                    if (dv < 0) dv = -dv;
                    if (dv > max_res) max_res = dv;
                }
            }
        }
        } else {
            // Diagonal-approx or 3D fallback: velocity-change residual
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i <= mesh_->i_end(); ++i) {
                    int idx = j * u_stride + i;
                    double du = velocity_u_ptr_[idx] - velocity_old_u_ptr_[idx];
                    if (du < 0) du = -du;
                    if (du > max_res) max_res = du;
                }
            }
            for (int j = mesh_->j_begin(); j <= mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    int idx = j * v_stride + i;
                    double dv = velocity_v_ptr_[idx] - velocity_old_v_ptr_[idx];
                    if (dv < 0) dv = -dv;
                    if (dv > max_res) max_res = dv;
                }
            }
        }

        if (max_res != max_res) max_res = 1e30;  // NaN guard

        // SER (Switched Evolution/Relaxation): adapt pseudo_dt based on residual
        // If residual decreases: increase pseudo_dt (less damping, faster convergence)
        // If residual increases: decrease pseudo_dt (more damping, stability)
        if (config_.simple_jacobi_sweeps > 0 && simple_pseudo_dt_fixed_ > 0) {
            static double prev_residual = -1.0;
            if (prev_residual > 0 && max_res > 0 && max_res < 1e20) {
                double ratio = prev_residual / max_res;
                // Clamp ratio to prevent wild swings
                if (ratio > 2.0) ratio = 2.0;
                if (ratio < 0.5) ratio = 0.5;
                simple_pseudo_dt_fixed_ *= ratio;
                // Cap pseudo_dt growth to prevent over-relaxation
                double dx_min = is_2d ? std::min(mesh_->dx, mesh_->dy)
                                      : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
                double max_pseudo_dt = 10.0 * dx_min;  // 10× CFL=1
                if (simple_pseudo_dt_fixed_ > max_pseudo_dt)
                    simple_pseudo_dt_fixed_ = max_pseudo_dt;
            }
            prev_residual = max_res;
        }

        return max_res;
    }
}

// ============================================================================
// SIMPLE velocity correction: u = u* - (1/a_P) * grad(p')
// and pressure update: p += alpha_p * p'
// ============================================================================

void RANSSolver::correct_velocity_simple() {
    TIMED_SCOPE("velocity_correction");

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const bool is_2d = mesh_->is2D();

    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    const int cell_stride = Nx + 2 * Ng;

    if (is_2d) {
        time_kernels::simple_correct_velocity_2d(
            velocity_u_ptr_, velocity_v_ptr_, pressure_ptr_,
            velocity_star_u_ptr_, velocity_star_v_ptr_,
            pressure_corr_ptr_,
            a_p_u_ptr_, a_p_v_ptr_,
            config_.simple_alpha_p, mesh_->dx, mesh_->dy,
            Nx, Ny, Ng, u_stride, v_stride, cell_stride);
    } else {
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride = Nx + 2 * Ng;
        const int w_plane = w_stride * (Ny + 2 * Ng);
        const int cell_plane = cell_stride * (Ny + 2 * Ng);

        time_kernels::simple_correct_velocity_3d(
            velocity_u_ptr_, velocity_v_ptr_, velocity_w_ptr_, pressure_ptr_,
            velocity_star_u_ptr_, velocity_star_v_ptr_, velocity_star_w_ptr_,
            pressure_corr_ptr_,
            a_p_u_ptr_, a_p_v_ptr_, a_p_w_ptr_,
            config_.simple_alpha_p,
            mesh_->dx, mesh_->dy, mesh_->dz,
            Nx, Ny, Nz, Ng,
            u_stride, u_plane, v_stride, v_plane,
            w_stride, w_plane, cell_stride, cell_plane);
    }
}

} // namespace nncfd
