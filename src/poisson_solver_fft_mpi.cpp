/// @file poisson_solver_fft_mpi.cpp
/// @brief Distributed FFT Poisson solver with MPI pencil transpose
///
/// Single-rank: delegates to FFTPoissonSolver for optimal performance.
/// Multi-rank: implements x-FFT (local) → z-transpose (MPI) → z-FFT (local) →
/// y-tridiagonal (local) → inverse z-FFT → z-transpose back → inverse x-FFT.
///
/// CPU path uses manual DFT for correctness; GPU path delegates to the serial
/// FFT solver which uses cuFFT + cuSPARSE.

#include "poisson_solver_fft_mpi.hpp"
#ifdef USE_FFT_POISSON
#include "poisson_solver_fft.hpp"
#endif
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace nncfd {

FFTMPIPoissonSolver::FFTMPIPoissonSolver(const Mesh& mesh, const Decomposition& decomp)
    : mesh_(&mesh), decomp_(&decomp)
{
    Nx_ = mesh.Nx;
    Ny_ = mesh.Ny;
    Ng_ = mesh.Nghost;
    Nz_global_ = decomp.nz_global();
    Nz_local_ = decomp.nz_local();
    distributed_ = decomp.is_parallel();

    if (!distributed_) {
#ifdef USE_FFT_POISSON
        // Single-rank: use the serial FFT solver directly (GPU only)
        serial_solver_ = std::make_unique<FFTPoissonSolver>(mesh);
#endif
    } else {
        initialize_distributed();
    }
}

FFTMPIPoissonSolver::~FFTMPIPoissonSolver() = default;

void FFTMPIPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                  PoissonBC y_lo, PoissonBC y_hi,
                                  PoissonBC z_lo, PoissonBC z_hi) {
    bc_x_lo_ = x_lo; bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo; bc_y_hi_ = y_hi;
    bc_z_lo_ = z_lo; bc_z_hi_ = z_hi;

#ifdef USE_FFT_POISSON
    if (serial_solver_) {
        serial_solver_->set_bc(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);
    }
#endif
}

void FFTMPIPoissonSolver::set_space_order(int order) {
    space_order_ = order;
#ifdef USE_FFT_POISSON
    if (serial_solver_) {
        serial_solver_->set_space_order(order);
    }
#endif
    if (distributed_) {
        compute_eigenvalues();
    }
}

bool FFTMPIPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                       PoissonBC y_lo, PoissonBC y_hi,
                                       PoissonBC z_lo, PoissonBC z_hi,
                                       bool uniform_x, bool uniform_z) {
    // Same requirements as serial FFT: periodic x/z, non-periodic y, uniform x/z
    return (x_lo == PoissonBC::Periodic && x_hi == PoissonBC::Periodic &&
            z_lo == PoissonBC::Periodic && z_hi == PoissonBC::Periodic &&
            y_lo != PoissonBC::Periodic && y_hi != PoissonBC::Periodic &&
            uniform_x && uniform_z);
}

bool FFTMPIPoissonSolver::using_gpu() const {
#ifdef USE_FFT_POISSON
    if (serial_solver_) return serial_solver_->using_gpu();
#endif
    return false;  // distributed CPU path
}

int FFTMPIPoissonSolver::solve(const ScalarField& rhs, ScalarField& p,
                                const PoissonConfig& cfg) {
    if (!distributed_) {
        // Single-rank: use CPU path of serial solver
        // Note: serial FFT solver only has solve_device; for CPU we'd
        // need to implement a host path. For now, this is a placeholder.
        throw std::runtime_error("FFTMPIPoissonSolver::solve() CPU path "
                                "requires distributed mode or GPU");
    }
    return solve_distributed_cpu(rhs, p);
}

int FFTMPIPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr,
                                       const PoissonConfig& cfg) {
#ifdef USE_FFT_POISSON
    if (!distributed_ && serial_solver_) {
        int result = serial_solver_->solve_device(rhs_ptr, p_ptr, cfg);
        residual_ = serial_solver_->residual();
        return result;
    }
#endif

    // Multi-rank GPU path: would use CUDA-aware MPI + cuFFT
    // For now, fall back to error since this needs GPU + MPI testing
    throw std::runtime_error("FFTMPIPoissonSolver::solve_device() distributed "
                            "GPU path not yet implemented (requires CUDA-aware MPI)");
}

// ============================================================================
// Distributed solve implementation (CPU path)
// ============================================================================

void FFTMPIPoissonSolver::initialize_distributed() {
    compute_eigenvalues();
    compute_tridiagonal_coeffs();
    compute_alltoallv_params();

    // Allocate work arrays for the distributed solve
    // After x-FFT + z-transpose, each rank has:
    //   - Full z (Nz_global) for a subset of kx modes
    //   - Full y (Ny)
    int n_kx_local = Nx_ / decomp_->nprocs();  // approximate
    int work_size = Nx_ * Ny_ * Nz_global_;  // conservative upper bound
    work_real_.resize(work_size, 0.0);
    work_imag_.resize(work_size, 0.0);

    // MPI buffers
    int max_send = Nx_ * Ny_ * Nz_local_ * 2;  // real + imag
    int max_recv = Nx_ * Ny_ * Nz_global_ * 2;
    send_buf_.resize(max_send, 0.0);
    recv_buf_.resize(max_recv, 0.0);
}

void FFTMPIPoissonSolver::compute_eigenvalues() {
    const double dx = mesh_->dx;
    const double dz = mesh_->dz;
    const double pi = M_PI;

    lambda_x_.resize(Nx_);
    lambda_z_.resize(Nz_global_);

    if (space_order_ == 4) {
        for (int kx = 0; kx < Nx_; ++kx) {
            double theta = 2.0 * pi * kx / Nx_;
            double c1 = std::cos(theta);
            double c2 = std::cos(2.0 * theta);
            double c3 = std::cos(3.0 * theta);
            lambda_x_[kx] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dx * dx);
        }
        for (int kz = 0; kz < Nz_global_; ++kz) {
            double theta = 2.0 * pi * kz / Nz_global_;
            double c1 = std::cos(theta);
            double c2 = std::cos(2.0 * theta);
            double c3 = std::cos(3.0 * theta);
            lambda_z_[kz] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dz * dz);
        }
    } else {
        for (int kx = 0; kx < Nx_; ++kx) {
            lambda_x_[kx] = (2.0 - 2.0 * std::cos(2.0 * pi * kx / Nx_)) / (dx * dx);
        }
        for (int kz = 0; kz < Nz_global_; ++kz) {
            lambda_z_[kz] = (2.0 - 2.0 * std::cos(2.0 * pi * kz / Nz_global_)) / (dz * dz);
        }
    }
}

void FFTMPIPoissonSolver::compute_tridiagonal_coeffs() {
    const int Ny = Ny_;
    const int Ng = Ng_;
    const double* y = mesh_->yc.data();

    tri_lower_.resize(Ny);
    tri_upper_.resize(Ny);
    tri_diag_.resize(Ny);

    for (int j = 0; j < Ny; ++j) {
        const int jg = j + Ng;
        double dy_south = y[jg] - y[jg - 1];
        double dy_north = y[jg + 1] - y[jg];
        double dy_center = 0.5 * (dy_south + dy_north);

        double aS = 1.0 / (dy_south * dy_center);
        double aN = 1.0 / (dy_north * dy_center);

        if (j == 0 && bc_y_lo_ == PoissonBC::Neumann) aS = 0.0;
        if (j == Ny - 1 && bc_y_hi_ == PoissonBC::Neumann) aN = 0.0;

        tri_lower_[j] = aS;
        tri_upper_[j] = aN;
        tri_diag_[j] = -(aS + aN);
    }
}

void FFTMPIPoissonSolver::compute_alltoallv_params() {
    int nprocs = decomp_->nprocs();
    send_counts_.resize(nprocs);
    send_displs_.resize(nprocs);
    recv_counts_.resize(nprocs);
    recv_displs_.resize(nprocs);

    // Each rank sends its Nz_local z-planes for all (kx, y) to each other rank
    // After transpose, each rank gets full z for a subset of kx modes
    // Partition kx modes evenly across ranks
    int kx_base = Nx_ / nprocs;
    int kx_rem = Nx_ % nprocs;

    int send_offset = 0;
    int recv_offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kx_count = kx_base + (r < kx_rem ? 1 : 0);

        // Send: kx_count modes × Ny × Nz_local (real+imag)
        send_counts_[r] = kx_count * Ny_ * Nz_local_ * 2;
        send_displs_[r] = send_offset;
        send_offset += send_counts_[r];

        // Recv: kx_count modes × Ny × nz_for_rank_r (real+imag)
        // We receive from rank r their Nz_local z-planes
        int nz_from_r = decomp_->nz_for_rank(r);
        recv_counts_[r] = kx_count * Ny_ * nz_from_r * 2;
        recv_displs_[r] = recv_offset;
        recv_offset += recv_counts_[r];
    }
}

int FFTMPIPoissonSolver::solve_distributed_cpu(const ScalarField& rhs, ScalarField& p) {
    // This implements the full distributed FFT Poisson solve on CPU.
    // Steps:
    //   1. Pack RHS from ghost-cell layout, subtract mean
    //   2. Forward DFT in x (local, each rank has full x)
    //   3. MPI_Alltoallv: redistribute by kx modes (z-slabs → kx-pencils)
    //   4. Forward DFT in z (local, each rank now has full z for its kx modes)
    //   5. Tridiagonal solve in y for each (kx,kz) mode
    //   6. Inverse DFT in z
    //   7. MPI_Alltoallv: redistribute back (kx-pencils → z-slabs)
    //   8. Inverse DFT in x
    //   9. Unpack solution to ghost-cell layout

    const int Nx = Nx_;
    const int Ny = Ny_;
    const int Nz_local = Nz_local_;
    const int Nz_global = Nz_global_;
    const int Ng = Ng_;
    const double pi = M_PI;

    // Step 1: Pack RHS, compute and subtract volume-weighted mean
    // On stretched grids, solvability requires sum(f*dyv[j])=0 (volume-weighted)
    std::vector<double> rhs_packed(Nx * Ny * Nz_local, 0.0);
    double local_weighted_sum = 0.0;
    double local_volume = 0.0;

    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            double dyv_j = mesh_->yf[j + Ng + 1] - mesh_->yf[j + Ng];
            for (int i = 0; i < Nx; ++i) {
                double val = rhs(i + Ng, j + Ng, k + Ng);
                rhs_packed[k * Nx * Ny + j * Nx + i] = val;
                local_weighted_sum += val * dyv_j;
                local_volume += dyv_j;
            }
        }
    }

    // Global volume-weighted mean subtraction for solvability
    double global_weighted_sum = local_weighted_sum;
    double global_volume = local_volume;
    decomp_->allreduce_sum(&global_weighted_sum, 1);
    decomp_->allreduce_sum(&global_volume, 1);
    double mean = global_weighted_sum / global_volume;

    for (auto& v : rhs_packed) v -= mean;

    // Step 2: Forward DFT in x for each (j, k_local)
    // rhs_hat[k][j][kx] = sum_i rhs[k][j][i] * exp(-2*pi*i*kx*i/Nx)
    std::vector<double> hat_real(Nx * Ny * Nz_local, 0.0);
    std::vector<double> hat_imag(Nx * Ny * Nz_local, 0.0);

    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int kx = 0; kx < Nx; ++kx) {
                double re = 0.0, im = 0.0;
                for (int i = 0; i < Nx; ++i) {
                    double angle = -2.0 * pi * kx * i / Nx;
                    double val = rhs_packed[k * Nx * Ny + j * Nx + i];
                    re += val * std::cos(angle);
                    im += val * std::sin(angle);
                }
                int idx = k * Nx * Ny + j * Nx + kx;
                hat_real[idx] = re;
                hat_imag[idx] = im;
            }
        }
    }

#ifdef USE_MPI
    // Step 3: MPI_Alltoallv — redistribute from z-slabs to kx-pencils
    // Pack: for each dest rank, send their kx range × Ny × Nz_local
    int nprocs = decomp_->nprocs();
    int kx_base = Nx / nprocs;
    int kx_rem = Nx % nprocs;

    // Pack send buffer
    int offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kx_start = r * kx_base + std::min(r, kx_rem);
        int kx_count = kx_base + (r < kx_rem ? 1 : 0);
        for (int k = 0; k < Nz_local; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int kx = kx_start; kx < kx_start + kx_count; ++kx) {
                    int src_idx = k * Nx * Ny + j * Nx + kx;
                    send_buf_[offset++] = hat_real[src_idx];
                    send_buf_[offset++] = hat_imag[src_idx];
                }
            }
        }
    }

    MPI_Alltoallv(send_buf_.data(), send_counts_.data(), send_displs_.data(), MPI_DOUBLE,
                  recv_buf_.data(), recv_counts_.data(), recv_displs_.data(), MPI_DOUBLE,
                  decomp_->comm());

    // Unpack: each rank now has full z for its kx range
    int my_kx_start = decomp_->rank() * kx_base + std::min(decomp_->rank(), kx_rem);
    int my_kx_count = kx_base + (decomp_->rank() < kx_rem ? 1 : 0);

    // pencil_real/imag[kz][j][kx_local]
    std::vector<double> pencil_real(Nz_global * Ny * my_kx_count, 0.0);
    std::vector<double> pencil_imag(Nz_global * Ny * my_kx_count, 0.0);

    offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        int nz_from_r = decomp_->nz_for_rank(r);
        int kz_start = decomp_->k_global_start_for_rank(r);
        for (int kl = 0; kl < nz_from_r; ++kl) {
            int kz = kz_start + kl;
            for (int j = 0; j < Ny; ++j) {
                for (int kx_l = 0; kx_l < my_kx_count; ++kx_l) {
                    int dst_idx = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                    pencil_real[dst_idx] = recv_buf_[offset++];
                    pencil_imag[dst_idx] = recv_buf_[offset++];
                }
            }
        }
    }

    // Step 4: Forward DFT in z for each (kx_local, j)
    std::vector<double> full_real(Nz_global * Ny * my_kx_count, 0.0);
    std::vector<double> full_imag(Nz_global * Ny * my_kx_count, 0.0);

    for (int kx_l = 0; kx_l < my_kx_count; ++kx_l) {
        for (int j = 0; j < Ny; ++j) {
            for (int kz = 0; kz < Nz_global; ++kz) {
                double re = 0.0, im = 0.0;
                for (int iz = 0; iz < Nz_global; ++iz) {
                    double angle = -2.0 * pi * kz * iz / Nz_global;
                    int src = iz * Ny * my_kx_count + j * my_kx_count + kx_l;
                    double sr = pencil_real[src];
                    double si = pencil_imag[src];
                    re += sr * std::cos(angle) - si * std::sin(angle);
                    im += sr * std::sin(angle) + si * std::cos(angle);
                }
                int dst = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                full_real[dst] = re;
                full_imag[dst] = im;
            }
        }
    }

    // Step 5: Tridiagonal solve in y for each (kx, kz) mode
    std::vector<double> sol_real(Nz_global * Ny * my_kx_count, 0.0);
    std::vector<double> sol_imag(Nz_global * Ny * my_kx_count, 0.0);

    // Thomas algorithm workspace
    std::vector<double> c_prime(Ny), d_prime_r(Ny), d_prime_i(Ny);

    for (int kx_l = 0; kx_l < my_kx_count; ++kx_l) {
        int kx = my_kx_start + kx_l;
        for (int kz = 0; kz < Nz_global; ++kz) {
            // Skip zero mode (singular)
            if (kx == 0 && kz == 0) {
                for (int j = 0; j < Ny; ++j) {
                    int idx = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                    sol_real[idx] = 0.0;
                    sol_imag[idx] = 0.0;
                }
                continue;
            }

            double shift = -(lambda_x_[kx] + lambda_z_[kz]);

            // Forward sweep
            {
                double diag = tri_diag_[0] + shift;
                c_prime[0] = tri_upper_[0] / diag;
                int idx0 = kz * Ny * my_kx_count + 0 * my_kx_count + kx_l;
                d_prime_r[0] = full_real[idx0] / diag;
                d_prime_i[0] = full_imag[idx0] / diag;
            }
            for (int j = 1; j < Ny; ++j) {
                double diag = tri_diag_[j] + shift - tri_lower_[j] * c_prime[j-1];
                if (j < Ny - 1) {
                    c_prime[j] = tri_upper_[j] / diag;
                }
                int idx = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                d_prime_r[j] = (full_real[idx] - tri_lower_[j] * d_prime_r[j-1]) / diag;
                d_prime_i[j] = (full_imag[idx] - tri_lower_[j] * d_prime_i[j-1]) / diag;
            }

            // Back substitution
            {
                int idx = kz * Ny * my_kx_count + (Ny-1) * my_kx_count + kx_l;
                sol_real[idx] = d_prime_r[Ny-1];
                sol_imag[idx] = d_prime_i[Ny-1];
            }
            for (int j = Ny - 2; j >= 0; --j) {
                int idx = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                int idx1 = kz * Ny * my_kx_count + (j+1) * my_kx_count + kx_l;
                sol_real[idx] = d_prime_r[j] - c_prime[j] * sol_real[idx1];
                sol_imag[idx] = d_prime_i[j] - c_prime[j] * sol_imag[idx1];
            }
        }
    }

    // Step 6: Inverse DFT in z
    std::vector<double> isol_real(Nz_global * Ny * my_kx_count, 0.0);
    std::vector<double> isol_imag(Nz_global * Ny * my_kx_count, 0.0);

    double inv_Nz = 1.0 / Nz_global;
    for (int kx_l = 0; kx_l < my_kx_count; ++kx_l) {
        for (int j = 0; j < Ny; ++j) {
            for (int iz = 0; iz < Nz_global; ++iz) {
                double re = 0.0, im = 0.0;
                for (int kz = 0; kz < Nz_global; ++kz) {
                    double angle = 2.0 * pi * kz * iz / Nz_global;
                    int src = kz * Ny * my_kx_count + j * my_kx_count + kx_l;
                    double sr = sol_real[src];
                    double si = sol_imag[src];
                    re += sr * std::cos(angle) - si * std::sin(angle);
                    im += sr * std::sin(angle) + si * std::cos(angle);
                }
                int dst = iz * Ny * my_kx_count + j * my_kx_count + kx_l;
                isol_real[dst] = re * inv_Nz;
                isol_imag[dst] = im * inv_Nz;
            }
        }
    }

    // Step 7: MPI_Alltoallv — redistribute back (kx-pencils → z-slabs)
    // Pack send buffer: for each dest rank r, send their z-planes for my kx range
    offset = 0;
    // Recompute recv/send for reverse direction (swap roles)
    std::vector<int> rev_send_counts(nprocs);
    std::vector<int> rev_send_displs(nprocs);
    std::vector<int> rev_recv_counts(nprocs);
    std::vector<int> rev_recv_displs(nprocs);

    int s_off = 0, r_off = 0;
    for (int r = 0; r < nprocs; ++r) {
        int nz_for_r = decomp_->nz_for_rank(r);
        // Send: my_kx_count modes × Ny × nz_for_r planes × 2 (real+imag)
        rev_send_counts[r] = my_kx_count * Ny * nz_for_r * 2;
        rev_send_displs[r] = s_off;
        s_off += rev_send_counts[r];

        // Recv: kx_count_for_r modes × Ny × Nz_local × 2
        int kx_for_r = kx_base + (r < kx_rem ? 1 : 0);
        rev_recv_counts[r] = kx_for_r * Ny * Nz_local * 2;
        rev_recv_displs[r] = r_off;
        r_off += rev_recv_counts[r];
    }

    // Pack
    std::vector<double> rev_send(s_off, 0.0);
    offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kz_start = decomp_->k_global_start_for_rank(r);
        int nz_for_r = decomp_->nz_for_rank(r);
        for (int kl = 0; kl < nz_for_r; ++kl) {
            int iz = kz_start + kl;
            for (int j = 0; j < Ny; ++j) {
                for (int kx_l = 0; kx_l < my_kx_count; ++kx_l) {
                    int src = iz * Ny * my_kx_count + j * my_kx_count + kx_l;
                    rev_send[offset++] = isol_real[src];
                    rev_send[offset++] = isol_imag[src];
                }
            }
        }
    }

    std::vector<double> rev_recv(r_off, 0.0);
    MPI_Alltoallv(rev_send.data(), rev_send_counts.data(), rev_send_displs.data(), MPI_DOUBLE,
                  rev_recv.data(), rev_recv_counts.data(), rev_recv_displs.data(), MPI_DOUBLE,
                  decomp_->comm());

    // Unpack: rebuild hat_real/hat_imag in z-slab layout
    std::fill(hat_real.begin(), hat_real.end(), 0.0);
    std::fill(hat_imag.begin(), hat_imag.end(), 0.0);

    offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kx_start_r = r * kx_base + std::min(r, kx_rem);
        int kx_count_r = kx_base + (r < kx_rem ? 1 : 0);
        for (int k = 0; k < Nz_local; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int kx_l = 0; kx_l < kx_count_r; ++kx_l) {
                    int kx = kx_start_r + kx_l;
                    int idx = k * Nx * Ny + j * Nx + kx;
                    hat_real[idx] = rev_recv[offset++];
                    hat_imag[idx] = rev_recv[offset++];
                }
            }
        }
    }

    // Step 8: Inverse DFT in x
    double inv_Nx = 1.0 / Nx;
    std::vector<double> p_packed(Nx * Ny * Nz_local, 0.0);

    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                double re = 0.0;
                for (int kx = 0; kx < Nx; ++kx) {
                    double angle = 2.0 * pi * kx * i / Nx;
                    int idx = k * Nx * Ny + j * Nx + kx;
                    re += hat_real[idx] * std::cos(angle) - hat_imag[idx] * std::sin(angle);
                }
                p_packed[k * Nx * Ny + j * Nx + i] = re * inv_Nx;
            }
        }
    }

    // Step 9: Unpack to ghost-cell layout
    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                p(i + Ng, j + Ng, k + Ng) = p_packed[k * Nx * Ny + j * Nx + i];
            }
        }
    }

    // Apply Neumann BCs in y (ghost cells)
    for (int k = 0; k < Nz_local; ++k) {
        for (int i = 0; i < Nx; ++i) {
            if (bc_y_lo_ == PoissonBC::Neumann) {
                p(i + Ng, Ng - 1, k + Ng) = p(i + Ng, Ng, k + Ng);
            }
            if (bc_y_hi_ == PoissonBC::Neumann) {
                p(i + Ng, Ny + Ng, k + Ng) = p(i + Ng, Ny + Ng - 1, k + Ng);
            }
        }
    }

#else
    // Without MPI, this shouldn't be called (distributed_ should be false)
    (void)rhs; (void)p;
    throw std::runtime_error("FFTMPIPoissonSolver: distributed solve requires USE_MPI");
#endif

    residual_ = 0.0;  // Direct solver
    return 1;
}

} // namespace nncfd
