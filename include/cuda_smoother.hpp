#pragma once

namespace nncfd {
namespace cuda_kernels {

/// Launch optimized CUDA Chebyshev smoother with shared memory tiling
/// @param u       Solution array (device pointer)
/// @param f       RHS array (device pointer)
/// @param tmp     Temporary array (device pointer)
/// @param Nx,Ny,Nz Interior dimensions
/// @param Ng      Ghost cell width
/// @param inv_dx2,inv_dy2,inv_dz2 Inverse squared spacings
/// @param degree  Number of Chebyshev iterations
/// @param lambda_min,lambda_max Chebyshev eigenvalue bounds
/// @param bc_periodic_x,bc_periodic_y,bc_periodic_z BC flags
/// @param stream  CUDA stream
void launch_chebyshev_3d_smem(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream = nullptr);

/// Launch optimized CUDA Chebyshev smoother for non-uniform y grids
void launch_chebyshev_3d_smem_nonuniform(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* aS, const double* aN, const double* aP,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream = nullptr);

/// Check if CUDA smoother is available at runtime
bool cuda_smoother_available();

} // namespace cuda_kernels
} // namespace nncfd
