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

#ifdef USE_HYPRE
#include "momentum_solver_hypre.hpp"
#endif

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
            if (config_.verbose) {
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
        // ============================================================
        // GPU SIMPLE momentum solve.
        // When USE_HYPRE: BiCGSTAB + PFMG preconditioner (robust, fast)
        // Fallback: Jacobi-preconditioned BiCGSTAB (our implementation)
        // n_sweeps = max solver iterations.
        // Followed by variable-coefficient Poisson pressure correction.
        // ============================================================
        {
            const double alpha_u_s = config_.simple_alpha_u;
            const double ddx = mesh_->dx;
            const double ddy = mesh_->dy;
            const int v_stride_s = Nx + 2 * Ng;
            [[maybe_unused]] const size_t u_sz = velocity_.u_total_size();
            const int n_u_interior = (Nx + 1) * Ny;

            // Compute a_P for pressure correction.
            // For HYPRE SIMPLE: NO pseudo-transient. HYPRE's PFMG provides
            // the multigrid damping that bare Jacobi/RB-GS lacked.
            // Pure Patankar: a_P = a_P_phys / alpha_u (no vol/dt term).
            const double simple_pdt_inv = 0.0;  // Pure SIMPLE
            if (is_2d) {
                time_kernels::simple_compute_aP_2d(
                    a_p_u_ptr_, a_p_v_ptr_, nu_eff_ptr_,
                    velocity_old_u_ptr_, velocity_old_v_ptr_,
                    Nx, Ny, Ng, u_stride, v_stride_s, cell_stride,
                    ddx, ddy, simple_pdt_inv);
            }
            // TODO: simple_compute_aP_3d for 3D

            // --- U-momentum solve ---
            {
                const int n_u_x = Nx + 1;
                const int Nz_eff = is_2d ? 1 : Nz;
                const int n_u = n_u_x * Ny * Nz_eff;
                std::vector<double> aW(n_u), aE(n_u), aS(n_u), aN(n_u);
                std::vector<double> aB, aF;
                std::vector<double> aP(n_u), rhs_mom(n_u);
                if (!is_2d) { aB.resize(n_u); aF.resize(n_u); }

                if (is_2d) {
                    time_kernels::simple_assemble_momentum_u_2d(
                        aW.data(), aE.data(), aS.data(), aN.data(),
                        aP.data(), rhs_mom.data(),
                        velocity_old_u_ptr_, velocity_old_v_ptr_,
                        nu_eff_ptr_, pressure_ptr_, tau_div_u_ptr_,
                        fx_, alpha_u_s, simple_pdt_inv, ddx, ddy,
                        Nx, Ny, Ng, u_stride, v_stride_s, cell_stride);
                } else {
                    const int u_plane = u_stride * (Ny + 2*Ng);
                    const int v_plane = v_stride_s * (Ny + 2*Ng + 1);
                    const int w_stride = Nx + 2*Ng;
                    const int w_plane = w_stride * (Ny + 2*Ng);
                    const int cell_plane_s = cell_stride * (Ny + 2*Ng);
                    time_kernels::simple_assemble_momentum_u_3d(
                        aW.data(), aE.data(), aS.data(), aN.data(),
                        aB.data(), aF.data(), aP.data(), rhs_mom.data(),
                        velocity_old_u_ptr_, velocity_old_v_ptr_, velocity_old_w_ptr_,
                        nu_eff_ptr_, pressure_ptr_, tau_div_u_ptr_,
                        fx_, alpha_u_s, simple_pdt_inv, ddx, ddy, mesh_->dz,
                        Nx, Ny, Nz, Ng,
                        u_stride, u_plane, v_stride_s, v_plane,
                        w_stride, w_plane, cell_stride, cell_plane_s);
                }

#ifdef USE_HYPRE
                static std::unique_ptr<HypreMomentumSolver> hypre_u;
                if (!hypre_u) hypre_u = std::make_unique<HypreMomentumSolver>(*mesh_, true);

                std::vector<double> x_flat(n_u);
                if (is_2d) {
                    for (int j = 0; j < Ny; ++j)
                        for (int i = 0; i <= Nx; ++i)
                            x_flat[j*n_u_x+i] = velocity_old_u_ptr_[(j+Ng)*u_stride+(i+Ng)];
                } else {
                    const int u_plane = u_stride * (Ny + 2*Ng);
                    for (int k = 0; k < Nz; ++k)
                        for (int j = 0; j < Ny; ++j)
                            for (int i = 0; i <= Nx; ++i)
                                x_flat[k*(Ny*n_u_x)+j*n_u_x+i] =
                                    velocity_old_u_ptr_[(k+Ng)*u_plane+(j+Ng)*u_stride+(i+Ng)];
                }

                hypre_u->set_coefficients(aW.data(), aE.data(), aS.data(), aN.data(),
                    is_2d ? nullptr : aB.data(), is_2d ? nullptr : aF.data(),
                    aP.data(), n_u);
                // Limit HYPRE to 1-2 iterations (approximate solve, like OpenFOAM's 2 GS sweeps).
                // An exact solve gives u* ≈ u_old (Patankar pulls toward old solution),
                // leaving div(u*)=0 and no pressure correction driver.
                // An approximate solve leaves residual divergence that drives pressure.
                int max_mom_iters = std::min(n_sweeps, 2);
                int u_iters = hypre_u->solve(rhs_mom.data(), x_flat.data(), 1e-2, max_mom_iters);

                if (config_.verbose && step_count_ < 5)
                    std::cerr << "[SIMPLE] HYPRE u-mom: " << u_iters
                              << " iters, res=" << hypre_u->final_residual() << "\n";

                if (is_2d) {
                    for (int j = 0; j < Ny; ++j)
                        for (int i = 0; i <= Nx; ++i)
                            velocity_star_u_ptr_[(j+Ng)*u_stride+(i+Ng)] = x_flat[j*n_u_x+i];
                } else {
                    const int u_plane = u_stride * (Ny + 2*Ng);
                    for (int k = 0; k < Nz; ++k)
                        for (int j = 0; j < Ny; ++j)
                            for (int i = 0; i <= Nx; ++i)
                                velocity_star_u_ptr_[(k+Ng)*u_plane+(j+Ng)*u_stride+(i+Ng)] =
                                    x_flat[k*(Ny*n_u_x)+j*n_u_x+i];
                }
#else
                // Fallback: Jacobi diagonal approximation
                if (is_2d) {
                    for (int j = 0; j < Ny; ++j)
                        for (int i = 0; i <= Nx; ++i)
                            velocity_star_u_ptr_[(j+Ng)*u_stride+(i+Ng)] =
                                rhs_mom[j*(Nx+1)+i] / aP[j*(Nx+1)+i];
                }
#endif
            }

            // --- V-momentum solve (2D and 3D) ---
            {
                const int Nz_v = is_2d ? 1 : Nz;
                const int n_v = Nx * (Ny+1) * Nz_v;
                std::vector<double> aW_v(n_v), aE_v(n_v), aS_v(n_v), aN_v(n_v);
                std::vector<double> aB_v, aF_v;
                std::vector<double> aP_v(n_v), rhs_v(n_v);
                if (!is_2d) { aB_v.resize(n_v); aF_v.resize(n_v); }

                if (is_2d) {
                    // 2D v-momentum assembly (inline, same structure as u)
                    const double inv_dx2 = 1.0/(ddx*ddx), inv_dy2 = 1.0/(ddy*ddy);
                    const double vol = ddx * ddy;
                    for (int j = 0; j <= Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int jg = j + Ng, ig = i + Ng;
                            int v_idx = jg * v_stride_s + ig;
                            int cb = (jg-1)*cell_stride + ig;
                            int ct = jg*cell_stride + ig;
                            int flat = j*Nx + i;

                            double nu_B = nu_eff_ptr_[cb], nu_T = nu_eff_ptr_[ct];
                            double nu_W = 0.25*(nu_eff_ptr_[cb]+nu_eff_ptr_[ct]
                                +nu_eff_ptr_[(jg-1)*cell_stride+(ig-1)]+nu_eff_ptr_[jg*cell_stride+(ig-1)]);
                            double nu_E = 0.25*(nu_eff_ptr_[cb]+nu_eff_ptr_[ct]
                                +nu_eff_ptr_[(jg-1)*cell_stride+(ig+1)]+nu_eff_ptr_[jg*cell_stride+(ig+1)]);

                            double aW = nu_W*inv_dx2*vol, aE = nu_E*inv_dx2*vol;
                            double aS = nu_B*inv_dy2*vol, aN = nu_T*inv_dy2*vol;

                            double F_w = 0.5*(velocity_old_u_ptr_[(jg-1)*u_stride+ig]
                                             +velocity_old_u_ptr_[jg*u_stride+ig]) * ddy;
                            double F_e = 0.5*(velocity_old_u_ptr_[(jg-1)*u_stride+(ig+1)]
                                             +velocity_old_u_ptr_[jg*u_stride+(ig+1)]) * ddy;
                            double F_s = velocity_old_v_ptr_[(jg-1)*v_stride_s+ig] * ddx;
                            double F_n = velocity_old_v_ptr_[(jg+1)*v_stride_s+ig] * ddx;

                            aW += (F_w > 0 ? F_w : 0); aE += (F_e < 0 ? -F_e : 0);
                            aS += (F_s > 0 ? F_s : 0); aN += (F_n < 0 ? -F_n : 0);

                            double aP_phys = (nu_W+nu_E)*inv_dx2*vol + (nu_B+nu_T)*inv_dy2*vol
                                + ((F_w<0?-F_w:0)+(F_e>0?F_e:0)+(F_s<0?-F_s:0)+(F_n>0?F_n:0));
                            if (aP_phys < 1e-20) aP_phys = 1e-20;
                            double aP_eff = aP_phys / alpha_u_s;

                            aW_v[flat] = aW; aE_v[flat] = aE;
                            aS_v[flat] = aS; aN_v[flat] = aN;
                            aP_v[flat] = aP_eff;

                            double source = (tau_div_v_ptr_[v_idx] + fy_) * vol;
                            double dp_dy = (pressure_ptr_[ct] - pressure_ptr_[cb]) / ddy * vol;
                            double relax_src = (aP_eff - aP_phys) * velocity_old_v_ptr_[v_idx];
                            rhs_v[flat] = aW*velocity_old_v_ptr_[jg*v_stride_s+(ig-1)]
                                + aE*velocity_old_v_ptr_[jg*v_stride_s+(ig+1)]
                                + aS*velocity_old_v_ptr_[(jg-1)*v_stride_s+ig]
                                + aN*velocity_old_v_ptr_[(jg+1)*v_stride_s+ig]
                                + source - dp_dy + relax_src;
                        }
                    }
                } else {
                    const int u_plane = u_stride * (Ny + 2*Ng);
                    const int v_plane = v_stride_s * (Ny + 2*Ng + 1);
                    const int w_stride = Nx + 2*Ng;
                    const int w_plane = w_stride * (Ny + 2*Ng);
                    const int cell_plane_s = cell_stride * (Ny + 2*Ng);
                    time_kernels::simple_assemble_momentum_v_3d(
                        aW_v.data(), aE_v.data(), aS_v.data(), aN_v.data(),
                        aB_v.data(), aF_v.data(), aP_v.data(), rhs_v.data(),
                        velocity_old_u_ptr_, velocity_old_v_ptr_, velocity_old_w_ptr_,
                        nu_eff_ptr_, pressure_ptr_, tau_div_v_ptr_,
                        fy_, alpha_u_s, simple_pdt_inv, ddx, ddy, mesh_->dz,
                        Nx, Ny, Nz, Ng,
                        u_stride, u_plane, v_stride_s, v_plane,
                        w_stride, w_plane, cell_stride, cell_plane_s);
                }

#ifdef USE_HYPRE
                static std::unique_ptr<HypreMomentumSolver> hypre_v;
                if (!hypre_v) hypre_v = std::make_unique<HypreMomentumSolver>(*mesh_, false);

                std::vector<double> x_flat(n_v);
                if (is_2d) {
                    for (int j = 0; j <= Ny; ++j)
                        for (int i = 0; i < Nx; ++i)
                            x_flat[j*Nx+i] = velocity_old_v_ptr_[(j+Ng)*v_stride_s+(i+Ng)];
                } else {
                    const int v_plane = v_stride_s * (Ny + 2*Ng + 1);
                    for (int k = 0; k < Nz_v; ++k)
                        for (int j = 0; j <= Ny; ++j)
                            for (int i = 0; i < Nx; ++i)
                                x_flat[k*((Ny+1)*Nx)+j*Nx+i] =
                                    velocity_old_v_ptr_[(k+Ng)*v_plane+(j+Ng)*v_stride_s+(i+Ng)];
                }

                hypre_v->set_coefficients(aW_v.data(), aE_v.data(), aS_v.data(), aN_v.data(),
                    is_2d ? nullptr : aB_v.data(), is_2d ? nullptr : aF_v.data(),
                    aP_v.data(), n_v);
                int max_mom_iters_v = std::min(n_sweeps, 2);
                int v_iters = hypre_v->solve(rhs_v.data(), x_flat.data(), 1e-2, max_mom_iters_v);

                if (config_.verbose)
                    std::cerr << "[SIMPLE] HYPRE v-mom: " << v_iters
                              << " iters, res=" << hypre_v->final_residual() << "\n";

                if (is_2d) {
                    for (int j = 0; j <= Ny; ++j)
                        for (int i = 0; i < Nx; ++i)
                            velocity_star_v_ptr_[(j+Ng)*v_stride_s+(i+Ng)] = x_flat[j*Nx+i];
                } else {
                    const int v_plane = v_stride_s * (Ny + 2*Ng + 1);
                    for (int k = 0; k < Nz_v; ++k)
                        for (int j = 0; j <= Ny; ++j)
                            for (int i = 0; i < Nx; ++i)
                                velocity_star_v_ptr_[(k+Ng)*v_plane+(j+Ng)*v_stride_s+(i+Ng)] =
                                    x_flat[k*((Ny+1)*Nx)+j*Nx+i];
                }
#endif
            }

            // --- W-momentum solve (3D only) ---
            if (!is_2d) {
                const int n_w = Nx * Ny * (Nz+1);
                std::vector<double> aW_w(n_w), aE_w(n_w), aS_w(n_w), aN_w(n_w);
                std::vector<double> aB_w(n_w), aF_w(n_w);
                std::vector<double> aP_w(n_w), rhs_w(n_w);

                const int u_plane = u_stride * (Ny + 2*Ng);
                const int v_plane = v_stride_s * (Ny + 2*Ng + 1);
                const int w_stride = Nx + 2*Ng;
                const int w_plane = w_stride * (Ny + 2*Ng);
                const int cell_plane_s = cell_stride * (Ny + 2*Ng);

                time_kernels::simple_assemble_momentum_w_3d(
                    aW_w.data(), aE_w.data(), aS_w.data(), aN_w.data(),
                    aB_w.data(), aF_w.data(), aP_w.data(), rhs_w.data(),
                    velocity_old_u_ptr_, velocity_old_v_ptr_, velocity_old_w_ptr_,
                    nu_eff_ptr_, pressure_ptr_, tau_div_w_ptr_,
                    fz_, alpha_u_s, simple_pdt_inv, ddx, ddy, mesh_->dz,
                    Nx, Ny, Nz, Ng,
                    u_stride, u_plane, v_stride_s, v_plane,
                    w_stride, w_plane, cell_stride, cell_plane_s);

                // W-momentum HYPRE solve would go here
                // For now, use Jacobi fallback for w
                for (int idx = 0; idx < n_w; ++idx)
                    if (aP_w[idx] > 1e-20) {
                        // Unpack to velocity_star_w (TODO: proper pack/unpack)
                    }
                // Simplified: copy w from old for now
                for (size_t i_w = 0; i_w < velocity_.w_total_size(); ++i_w)
                    velocity_star_w_ptr_[i_w] = velocity_old_w_ptr_[i_w];
            }

            // Apply BCs to velocity_star (all components)
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
            if (!is_2d) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
            std::swap(velocity_, velocity_star_);
            apply_velocity_bc();
            std::swap(velocity_, velocity_star_);
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
            if (!is_2d) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
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
        // a_P includes pseudo-transient (vol/pseudo_dt) for stability.
        // CRITICAL: the SAME a_P must be used for BOTH the predictor AND
        // the pressure correction. This ensures the momentum-pressure
        // coupling is consistent, preventing divergence.
        const double aP_pdt_inv = pseudo_dt_inv;
        if (is_2d) {
            time_kernels::simple_compute_aP_2d(
                a_p_u_ptr_, a_p_v_ptr_, nu_eff_ptr_,
                velocity_u_ptr_, velocity_v_ptr_,
                Nx, Ny, Ng, u_stride, v_stride, cell_stride,
                mesh_->dx, mesh_->dy, aP_pdt_inv);
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

    // Both paths (ADI and diagonal-approx) now use sections 6-8 for pressure correction.
    // The ADI path provides a better momentum solve (implicit y-diffusion);
    // the diagonal-approx path uses explicit predictor. Both need the standard
    // Poisson + velocity correction for pressure-velocity coupling.

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
            // For SIMPLE: pseudo_dt = vol / mean(a_P) where a_P is the
            // diagonal from the momentum equation (already stored in a_p_u_ptr_).
            // This ensures the pressure RHS scaling matches the velocity correction.
            double vol = is_2d ? mesh_->dx * mesh_->dy
                               : mesh_->dx * mesh_->dy * mesh_->dz;
            double sum_aP = 0.0;
            const int u_stride_p = Nx + 2 * Ng + 1;
            const int Nz_loop = is_2d ? 1 : Nz;
            const int u_plane_p = u_stride_p * (Ny + 2 * Ng);
            int n_total = 0;
            for (int k = 0; k < Nz_loop; ++k)
                for (int j = 0; j < Ny; ++j)
                    for (int i = 0; i <= Nx; ++i) {
                        int idx = (is_2d ? 0 : (k+Ng)*u_plane_p) + (j+Ng)*u_stride_p + (i+Ng);
                        sum_aP += a_p_u_ptr_[idx];
                        n_total++;
                    }
            double mean_aP = (n_total > 0) ? sum_aP / n_total : 1.0;
            if (mean_aP < 1e-20) mean_aP = 1e-20;
            pseudo_dt_proj = vol / mean_aP;
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

        // SIMPLE pressure equation: ∇·(D · ∇p') = ∇·u*
        // where D_u = 1/a_P_u (at u-faces), D_v = 1/a_P_v (at v-faces).
        // This is the CORRECT variable-coefficient pressure correction.
        //
        // For SIMPLE mode: set variable coefficients on the MG solver so it
        // solves the proper pressure equation. The RHS is just div(u*).
        // For explicit RK3 mode: use constant-coefficient (1/dt) as before.
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        [[maybe_unused]] const size_t cell_sz = field_total_size_;
        double* rhs_sv = rhs_poisson_ptr_;
        double* div_sv = div_velocity_ptr_;
        const int count = is_2d ? Nx * Ny : Nx * Ny * Nz;

        // Set variable coefficients for SIMPLE pressure equation
        if (config_.time_integrator == TimeIntegrator::SIMPLE && is_2d) {
            const int u_stride = Nx + 2 * Ng + 1;
            const int v_stride = Nx + 2 * Ng;
            // The MG expects D = 1/a_P (diffusion coefficient), not a_P itself.
            // Compute 1/a_P into scratch buffers for the pressure solve.
            double* D_u = bicg_r_ptr_;  // reuse scratch buffer
            double* D_v = bicg_s_ptr_;  // reuse scratch buffer
            for (size_t ii = 0; ii < velocity_.u_total_size(); ++ii)
                D_u[ii] = (a_p_u_ptr_[ii] > 1e-20) ? 1.0 / a_p_u_ptr_[ii] : 0.0;
            for (size_t ii = 0; ii < velocity_.v_total_size(); ++ii)
                D_v[ii] = (a_p_v_ptr_[ii] > 1e-20) ? 1.0 / a_p_v_ptr_[ii] : 0.0;
            mg_poisson_solver_.set_variable_coefficients(
                D_u, D_v,
                u_stride, v_stride,
                mesh_->dx, mesh_->dy);
            // RHS = div(u*) - mean_div (solvability for periodic/Neumann)
            double mean_div = 0.0;
            {
                double sum_d = 0.0;
                #pragma omp target teams distribute parallel for collapse(2) reduction(+:sum_d) \
                    map(present: div_sv[0:cell_sz])
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        sum_d += div_sv[(j + Ng) * stride + (i + Ng)];
                    }
                }
                mean_div = (count > 0) ? sum_d / count : 0.0;
                if (config_.verbose) {
                    double max_div = 0.0;
                    #pragma omp target teams distribute parallel for collapse(2) \
                        map(present: div_sv[0:cell_sz]) reduction(max:max_div)
                    for (int j = 0; j < Ny; ++j)
                        for (int i = 0; i < Nx; ++i) {
                            double d = div_sv[(j+Ng)*stride+(i+Ng)];
                            if (d < 0) d = -d;
                            if (d > max_div) max_div = d;
                        }
                    std::cerr << "[SIMPLE] div(u*): sum=" << sum_d
                              << " max=" << max_div
                              << " mean=" << mean_div << "\n";
                }
            }
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: div_sv[0:cell_sz], rhs_sv[0:cell_sz])
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (j + Ng) * stride + (i + Ng);
                    rhs_sv[idx] = div_sv[idx] - mean_div;
                }
            }
        } else {
            // RK3 path: constant-coefficient Poisson (1/dt scaling)
            const double dt_inv = 1.0 / pseudo_dt_proj;
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
                        rhs_sv[idx] = (div_sv[idx] - mean_div) * dt_inv;
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
                            rhs_sv[idx] = (div_sv[idx] - mean_div) * dt_inv;
                        }
                    }
                }
            }
            // Clear variable coefficients (use constant-coefficient MG)
            mg_poisson_solver_.clear_variable_coefficients();
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
            // SIMPLE with varcoeff: MUST use MG (FFT is constant-coefficient only)
            bool use_mg = mg_poisson_solver_.has_variable_coefficients();
            if (!use_mg) {
                switch (selected_solver_) {
                    case PoissonSolverType::FFT:
                    case PoissonSolverType::FFT2D:
                    case PoissonSolverType::FFT1D:
                        if (fft_poisson_solver_)
                            fft_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        break;
                    default:
                        use_mg = true;
                        break;
                }
            }
            if (use_mg) {
                mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
            }
        } else
#endif
        {
            mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
        }

        // Debug: check p' magnitude (GPU reduction)
        if (config_.verbose) {
            double max_pp = 0.0;
            double* pp = pressure_corr_ptr_;
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: pp[0:cell_sz]) reduction(max:max_pp)
            for (int j = 0; j < Ny; ++j)
                for (int i = 0; i < Nx; ++i) {
                    double v = pp[(j+Ng)*stride+(i+Ng)];
                    if (v < 0) v = -v;
                    if (v > max_pp) max_pp = v;
                }
            std::cerr << "[SIMPLE] max|p'|=" << max_pp << "\n";
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

    // Wrap periodic pressure ghost cells after pressure update.
    // Without this, p(ghost) stays stale while p(interior) grows,
    // creating a spurious dp/dx at boundary faces.
    if (is_2d) {
        const int cell_stride = Nx + 2 * Ng;
        if (velocity_bc_.x_lo == VelocityBC::Periodic) {
            for (int j = 0; j < Ny; ++j) {
                int jg = j + Ng;
                pressure_ptr_[jg*cell_stride + (Ng-1)] =
                    pressure_ptr_[jg*cell_stride + (Ng+Nx-1)];
                pressure_ptr_[jg*cell_stride + (Ng+Nx)] =
                    pressure_ptr_[jg*cell_stride + Ng];
            }
        }
        if (velocity_bc_.y_lo == VelocityBC::NoSlip) {
            for (int i = 0; i < Nx; ++i) {
                int ig = i + Ng;
                pressure_ptr_[(Ng-1)*cell_stride + ig] =
                    pressure_ptr_[Ng*cell_stride + ig];
                pressure_ptr_[(Ng+Ny)*cell_stride + ig] =
                    pressure_ptr_[(Ng+Ny-1)*cell_stride + ig];
            }
        }
    }

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
    // 9. Residual: velocity change max|u^{n+1} - u^n|
    // ================================================================
    {
        TIMED_SCOPE("residual_computation");
        const int u_stride = Nx + 2 * Ng + 1;
        const int v_stride = Nx + 2 * Ng;

        double max_res = 0.0;

        {
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
