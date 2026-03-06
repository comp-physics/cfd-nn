/// @file solver_time_kernels_implicit.cpp
/// @brief Thomas algorithm kernels for implicit y-direction diffusion
///
/// Solves (I - dt * nu_eff * d²/dy²) u_new = u_old per column.
/// Free functions to avoid NVHPC this-pointer transfer.
/// Uses use_device_ptr pattern for zero host-device data transfer.

#include "solver_time_kernels.hpp"
#include <stdexcept>

namespace nncfd {
namespace time_kernels {

static constexpr int MAX_NY = 4096;

void thomas_y_diffusion_2d(double* u_ptr, double* v_ptr, double* nu_ptr,
                           int Nx, int Ny, int Ng,
                           int u_stride, int v_stride, int cell_stride,
                           double dt, double dy) {
    if (Ny > MAX_NY) {
        throw std::runtime_error("thomas_y_diffusion_2d: Ny=" + std::to_string(Ny) +
                                 " exceeds MAX_NY=" + std::to_string(MAX_NY));
    }
    const double dt_inv_dy2 = dt / (dy * dy);
    const int Nv_int = Ny - 1;

    #pragma omp target data use_device_ptr(u_ptr, v_ptr, nu_ptr)
    {
        // Thomas solve for u — Ny unknowns per column
        // No-slip: u_ghost = -u_interior → diagonal = 1+3*alpha at walls
        #pragma omp target teams distribute parallel for
        for (int i = Ng; i <= Ng + Nx; ++i) {
            double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
            // Forward elimination
            for (int jj = 0; jj < Ny; ++jj) {
                int j = jj + Ng;
                double nu_avg = 0.5 * (nu_ptr[j * cell_stride + (i - 1)] +
                                       nu_ptr[j * cell_stride + i]);
                double alpha = dt_inv_dy2 * nu_avg;

                double a_j = (jj > 0) ? -alpha : 0.0;
                double c_j = (jj < Ny - 1) ? -alpha : 0.0;
                double b_j = 1.0 + 2.0 * alpha;
                if (jj == 0) b_j += alpha;
                if (jj == Ny - 1) b_j += alpha;
                double d_j = u_ptr[j * u_stride + i];

                if (jj == 0) {
                    c_prime[0] = c_j / b_j;
                    d_prime[0] = d_j / b_j;
                } else {
                    double denom = b_j - a_j * c_prime[jj - 1];
                    c_prime[jj] = c_j / denom;
                    d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                }
            }
            // Backward substitution
            u_ptr[(Ng + Ny - 1) * u_stride + i] = d_prime[Ny - 1];
            for (int jj = Ny - 2; jj >= 0; --jj) {
                int j = jj + Ng;
                u_ptr[j * u_stride + i] =
                    d_prime[jj] - c_prime[jj] * u_ptr[(j + 1) * u_stride + i];
            }
        }

        // Thomas solve for v — only interior faces (Ny-1 unknowns)
        // v at wall faces pinned to 0 (no-penetration)
        if (Nv_int > 0) {
            #pragma omp target teams distribute parallel for
            for (int i = Ng; i < Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Nv_int; ++jj) {
                    int j = jj + Ng + 1;
                    double nu_avg = 0.5 * (nu_ptr[(j - 1) * cell_stride + i] +
                                           nu_ptr[j * cell_stride + i]);
                    double alpha = dt_inv_dy2 * nu_avg;

                    double a_j = (jj > 0) ? -alpha : 0.0;
                    double c_j = (jj < Nv_int - 1) ? -alpha : 0.0;
                    double b_j = 1.0 + 2.0 * alpha;
                    double d_j = v_ptr[j * v_stride + i];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                v_ptr[(Ng + Nv_int) * v_stride + i] = d_prime[Nv_int - 1];
                for (int jj = Nv_int - 2; jj >= 0; --jj) {
                    int j = jj + Ng + 1;
                    v_ptr[j * v_stride + i] =
                        d_prime[jj] - c_prime[jj] * v_ptr[(j + 1) * v_stride + i];
                }
            }
        }
    }
}

void thomas_y_diffusion_3d(double* u_ptr, double* v_ptr, double* w_ptr,
                           double* nu_ptr,
                           int Nx, int Ny, int Nz, int Ng,
                           int u_stride, int v_stride, int w_stride,
                           int u_plane, int v_plane, int w_plane,
                           int cell_stride, int cell_plane,
                           double dt, double dy) {
    if (Ny > MAX_NY) {
        throw std::runtime_error("thomas_y_diffusion_3d: Ny=" + std::to_string(Ny) +
                                 " exceeds MAX_NY=" + std::to_string(MAX_NY));
    }
    const double dt_inv_dy2 = dt / (dy * dy);
    const int Nv_int = Ny - 1;

    #pragma omp target data use_device_ptr(u_ptr, v_ptr, w_ptr, nu_ptr)
    {
        // Thomas solve for u — Ny unknowns per (i,k) column
        #pragma omp target teams distribute parallel for collapse(2)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Ny; ++jj) {
                    int j = jj + Ng;
                    int idx = k * u_plane + j * u_stride + i;
                    double nu_avg = 0.5 * (nu_ptr[k * cell_plane + j * cell_stride + (i - 1)] +
                                           nu_ptr[k * cell_plane + j * cell_stride + i]);
                    double alpha = dt_inv_dy2 * nu_avg;

                    double a_j = (jj > 0) ? -alpha : 0.0;
                    double c_j = (jj < Ny - 1) ? -alpha : 0.0;
                    double b_j = 1.0 + 2.0 * alpha;
                    if (jj == 0) b_j += alpha;
                    if (jj == Ny - 1) b_j += alpha;
                    double d_j = u_ptr[idx];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                u_ptr[k * u_plane + (Ng + Ny - 1) * u_stride + i] = d_prime[Ny - 1];
                for (int jj = Ny - 2; jj >= 0; --jj) {
                    int j = jj + Ng;
                    u_ptr[k * u_plane + j * u_stride + i] =
                        d_prime[jj] - c_prime[jj] * u_ptr[k * u_plane + (j + 1) * u_stride + i];
                }
            }
        }

        // Thomas solve for v — only interior faces (Ny-1 unknowns)
        if (Nv_int > 0) {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                    for (int jj = 0; jj < Nv_int; ++jj) {
                        int j = jj + Ng + 1;
                        int idx = k * v_plane + j * v_stride + i;
                        double nu_avg = 0.5 * (nu_ptr[k * cell_plane + (j - 1) * cell_stride + i] +
                                               nu_ptr[k * cell_plane + j * cell_stride + i]);
                        double alpha = dt_inv_dy2 * nu_avg;

                        double a_j = (jj > 0) ? -alpha : 0.0;
                        double c_j = (jj < Nv_int - 1) ? -alpha : 0.0;
                        double b_j = 1.0 + 2.0 * alpha;
                        double d_j = v_ptr[idx];

                        if (jj == 0) {
                            c_prime[0] = c_j / b_j;
                            d_prime[0] = d_j / b_j;
                        } else {
                            double denom = b_j - a_j * c_prime[jj - 1];
                            c_prime[jj] = c_j / denom;
                            d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                        }
                    }
                    v_ptr[k * v_plane + (Ng + Nv_int) * v_stride + i] = d_prime[Nv_int - 1];
                    for (int jj = Nv_int - 2; jj >= 0; --jj) {
                        int j = jj + Ng + 1;
                        v_ptr[k * v_plane + j * v_stride + i] =
                            d_prime[jj] - c_prime[jj] * v_ptr[k * v_plane + (j + 1) * v_stride + i];
                    }
                }
            }
        }

        // Thomas solve for w — Ny unknowns per (i,k) column
        #pragma omp target teams distribute parallel for collapse(2)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Ny; ++jj) {
                    int j = jj + Ng;
                    int idx = k * w_plane + j * w_stride + i;
                    double nu_avg = 0.5 * (nu_ptr[(k - 1) * cell_plane + j * cell_stride + i] +
                                           nu_ptr[k * cell_plane + j * cell_stride + i]);
                    double alpha = dt_inv_dy2 * nu_avg;

                    double a_j = (jj > 0) ? -alpha : 0.0;
                    double c_j = (jj < Ny - 1) ? -alpha : 0.0;
                    double b_j = 1.0 + 2.0 * alpha;
                    if (jj == 0) b_j += alpha;
                    if (jj == Ny - 1) b_j += alpha;
                    double d_j = w_ptr[idx];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                w_ptr[k * w_plane + (Ng + Ny - 1) * w_stride + i] = d_prime[Ny - 1];
                for (int jj = Ny - 2; jj >= 0; --jj) {
                    int j = jj + Ng;
                    w_ptr[k * w_plane + j * w_stride + i] =
                        d_prime[jj] - c_prime[jj] * w_ptr[k * w_plane + (j + 1) * w_stride + i];
                }
            }
        }
    }
}

// ============================================================================
// Stretched-grid variants: per-row dyv[j]/dyc[j] instead of uniform dy.
//
// For u (cell-centered in y) at row j:
//   alpha_lo = dt*nu / (dyv[j] * dyc[j])    — south coupling
//   alpha_hi = dt*nu / (dyv[j] * dyc[j+1])  — north coupling
//   Wall (no-slip): ghost = -interior → extra alpha_{lo|hi} added to diagonal
//
// For v (face-centered in y) at face j (j = Ng+1..Ng+Ny-1):
//   alpha_lo = dt*nu / (dyc[j] * dyv[j-1])
//   alpha_hi = dt*nu / (dyc[j] * dyv[j])
//   Wall (no-penetration): Dirichlet v=0, no ghost correction needed
// ============================================================================

void thomas_y_diffusion_2d_stretched(double* u_ptr, double* v_ptr, double* nu_ptr,
                                     int Nx, int Ny, int Ng,
                                     int u_stride, int v_stride, int cell_stride,
                                     double dt,
                                     const double* dyv, const double* dyc) {
    if (Ny > MAX_NY)
        throw std::runtime_error("thomas_y_diffusion_2d_stretched: Ny=" + std::to_string(Ny) +
                                 " exceeds MAX_NY=" + std::to_string(MAX_NY));
    const int Nv_int = Ny - 1;

    #pragma omp target data use_device_ptr(u_ptr, v_ptr, nu_ptr, dyv, dyc)
    {
        #pragma omp target teams distribute parallel for
        for (int i = Ng; i <= Ng + Nx; ++i) {
            double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
            for (int jj = 0; jj < Ny; ++jj) {
                int j = jj + Ng;
                double nu_avg = 0.5 * (nu_ptr[j * cell_stride + (i - 1)] +
                                       nu_ptr[j * cell_stride + i]);
                double alpha_lo = dt * nu_avg / (dyv[j] * dyc[j]);
                double alpha_hi = dt * nu_avg / (dyv[j] * dyc[j + 1]);

                double a_j = (jj > 0) ? -alpha_lo : 0.0;
                double c_j = (jj < Ny - 1) ? -alpha_hi : 0.0;
                double b_j = 1.0 + alpha_lo + alpha_hi;
                if (jj == 0)      b_j += alpha_lo;  // no-slip ghost = -interior
                if (jj == Ny - 1) b_j += alpha_hi;
                double d_j = u_ptr[j * u_stride + i];

                if (jj == 0) {
                    c_prime[0] = c_j / b_j;
                    d_prime[0] = d_j / b_j;
                } else {
                    double denom = b_j - a_j * c_prime[jj - 1];
                    c_prime[jj] = c_j / denom;
                    d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                }
            }
            u_ptr[(Ng + Ny - 1) * u_stride + i] = d_prime[Ny - 1];
            for (int jj = Ny - 2; jj >= 0; --jj) {
                int j = jj + Ng;
                u_ptr[j * u_stride + i] =
                    d_prime[jj] - c_prime[jj] * u_ptr[(j + 1) * u_stride + i];
            }
        }

        if (Nv_int > 0) {
            #pragma omp target teams distribute parallel for
            for (int i = Ng; i < Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Nv_int; ++jj) {
                    int j = jj + Ng + 1;
                    double nu_avg = 0.5 * (nu_ptr[(j - 1) * cell_stride + i] +
                                           nu_ptr[j * cell_stride + i]);
                    double alpha_lo = dt * nu_avg / (dyc[j] * dyv[j - 1]);
                    double alpha_hi = dt * nu_avg / (dyc[j] * dyv[j]);

                    double a_j = (jj > 0) ? -alpha_lo : 0.0;
                    double c_j = (jj < Nv_int - 1) ? -alpha_hi : 0.0;
                    double b_j = 1.0 + alpha_lo + alpha_hi;
                    double d_j = v_ptr[j * v_stride + i];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                v_ptr[(Ng + Nv_int) * v_stride + i] = d_prime[Nv_int - 1];
                for (int jj = Nv_int - 2; jj >= 0; --jj) {
                    int j = jj + Ng + 1;
                    v_ptr[j * v_stride + i] =
                        d_prime[jj] - c_prime[jj] * v_ptr[(j + 1) * v_stride + i];
                }
            }
        }
    }
}

void thomas_y_diffusion_3d_stretched(double* u_ptr, double* v_ptr, double* w_ptr,
                                     double* nu_ptr,
                                     int Nx, int Ny, int Nz, int Ng,
                                     int u_stride, int v_stride, int w_stride,
                                     int u_plane, int v_plane, int w_plane,
                                     int cell_stride, int cell_plane,
                                     double dt,
                                     const double* dyv, const double* dyc) {
    if (Ny > MAX_NY)
        throw std::runtime_error("thomas_y_diffusion_3d_stretched: Ny=" + std::to_string(Ny) +
                                 " exceeds MAX_NY=" + std::to_string(MAX_NY));
    const int Nv_int = Ny - 1;

    #pragma omp target data use_device_ptr(u_ptr, v_ptr, w_ptr, nu_ptr, dyv, dyc)
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Ny; ++jj) {
                    int j = jj + Ng;
                    double nu_avg = 0.5 * (nu_ptr[k * cell_plane + j * cell_stride + (i - 1)] +
                                           nu_ptr[k * cell_plane + j * cell_stride + i]);
                    double alpha_lo = dt * nu_avg / (dyv[j] * dyc[j]);
                    double alpha_hi = dt * nu_avg / (dyv[j] * dyc[j + 1]);

                    double a_j = (jj > 0) ? -alpha_lo : 0.0;
                    double c_j = (jj < Ny - 1) ? -alpha_hi : 0.0;
                    double b_j = 1.0 + alpha_lo + alpha_hi;
                    if (jj == 0)      b_j += alpha_lo;
                    if (jj == Ny - 1) b_j += alpha_hi;
                    double d_j = u_ptr[k * u_plane + j * u_stride + i];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                u_ptr[k * u_plane + (Ng + Ny - 1) * u_stride + i] = d_prime[Ny - 1];
                for (int jj = Ny - 2; jj >= 0; --jj) {
                    int j = jj + Ng;
                    u_ptr[k * u_plane + j * u_stride + i] =
                        d_prime[jj] - c_prime[jj] * u_ptr[k * u_plane + (j + 1) * u_stride + i];
                }
            }
        }

        if (Nv_int > 0) {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                    for (int jj = 0; jj < Nv_int; ++jj) {
                        int j = jj + Ng + 1;
                        double nu_avg = 0.5 * (nu_ptr[k * cell_plane + (j - 1) * cell_stride + i] +
                                               nu_ptr[k * cell_plane + j * cell_stride + i]);
                        double alpha_lo = dt * nu_avg / (dyc[j] * dyv[j - 1]);
                        double alpha_hi = dt * nu_avg / (dyc[j] * dyv[j]);

                        double a_j = (jj > 0) ? -alpha_lo : 0.0;
                        double c_j = (jj < Nv_int - 1) ? -alpha_hi : 0.0;
                        double b_j = 1.0 + alpha_lo + alpha_hi;
                        double d_j = v_ptr[k * v_plane + j * v_stride + i];

                        if (jj == 0) {
                            c_prime[0] = c_j / b_j;
                            d_prime[0] = d_j / b_j;
                        } else {
                            double denom = b_j - a_j * c_prime[jj - 1];
                            c_prime[jj] = c_j / denom;
                            d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                        }
                    }
                    v_ptr[k * v_plane + (Ng + Nv_int) * v_stride + i] = d_prime[Nv_int - 1];
                    for (int jj = Nv_int - 2; jj >= 0; --jj) {
                        int j = jj + Ng + 1;
                        v_ptr[k * v_plane + j * v_stride + i] =
                            d_prime[jj] - c_prime[jj] * v_ptr[k * v_plane + (j + 1) * v_stride + i];
                    }
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double c_prime[MAX_NY] = {0.0}, d_prime[MAX_NY] = {0.0};
                for (int jj = 0; jj < Ny; ++jj) {
                    int j = jj + Ng;
                    double nu_avg = 0.5 * (nu_ptr[(k - 1) * cell_plane + j * cell_stride + i] +
                                           nu_ptr[k * cell_plane + j * cell_stride + i]);
                    double alpha_lo = dt * nu_avg / (dyv[j] * dyc[j]);
                    double alpha_hi = dt * nu_avg / (dyv[j] * dyc[j + 1]);

                    double a_j = (jj > 0) ? -alpha_lo : 0.0;
                    double c_j = (jj < Ny - 1) ? -alpha_hi : 0.0;
                    double b_j = 1.0 + alpha_lo + alpha_hi;
                    if (jj == 0)      b_j += alpha_lo;
                    if (jj == Ny - 1) b_j += alpha_hi;
                    double d_j = w_ptr[k * w_plane + j * w_stride + i];

                    if (jj == 0) {
                        c_prime[0] = c_j / b_j;
                        d_prime[0] = d_j / b_j;
                    } else {
                        double denom = b_j - a_j * c_prime[jj - 1];
                        c_prime[jj] = c_j / denom;
                        d_prime[jj] = (d_j - a_j * d_prime[jj - 1]) / denom;
                    }
                }
                w_ptr[k * w_plane + (Ng + Ny - 1) * w_stride + i] = d_prime[Ny - 1];
                for (int jj = Ny - 2; jj >= 0; --jj) {
                    int j = jj + Ng;
                    w_ptr[k * w_plane + j * w_stride + i] =
                        d_prime[jj] - c_prime[jj] * w_ptr[k * w_plane + (j + 1) * w_stride + i];
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
