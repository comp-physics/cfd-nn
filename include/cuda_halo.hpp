#pragma once

/// @file cuda_halo.hpp
/// @brief Fused boundary condition and z-face pack/unpack kernels for MPI halo exchange
///
/// Provides:
/// - launch_apply_bc_3d_fused: Single kernel replacing 6 separate BC kernel launches
/// - launch_pack_z_face / launch_unpack_z_face: GPU-side halo packing for MPI z-slab decomposition
///
/// BC types: 0=periodic, 1=neumann, 2=dirichlet
/// All kernels operate on 3D arrays with ghost cells (Ng layers per side).

namespace nncfd {
namespace cuda_kernels {

/// Fused boundary condition application for all 6 faces
/// Replaces 6 separate kernel launches with 1
/// @param u       3D array with ghost cells (device pointer)
/// @param Nx,Ny,Nz Interior dimensions
/// @param Ng      Ghost cell width
/// @param bc_x_lo/hi BC type for x direction (0=periodic, 1=neumann, 2=dirichlet)
/// @param bc_y_lo/hi BC type for y direction
/// @param bc_z_lo/hi BC type for z direction
/// @param stream  CUDA stream (nullptr for default)
void launch_apply_bc_3d_fused(
    double* u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    void* stream = nullptr);

/// Pack z-face data into contiguous buffer (for MPI halo exchange)
/// @param field     Source 3D array (device pointer)
/// @param buffer    Destination buffer, size = (Nx+2Ng)*(Ny+2Ng) (device pointer)
/// @param Nx,Ny,Nz  Interior dimensions
/// @param Ng        Ghost cell width
/// @param pack_lo   true = pack first interior z-plane, false = pack last interior z-plane
/// @param stream    CUDA stream
void launch_pack_z_face(
    const double* field,
    double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool pack_lo,
    void* stream = nullptr);

/// Unpack z-face data from contiguous buffer into ghost layer
/// @param field      Destination 3D array (device pointer)
/// @param buffer     Source buffer, size = (Nx+2Ng)*(Ny+2Ng) (device pointer)
/// @param Nx,Ny,Nz   Interior dimensions
/// @param Ng         Ghost cell width
/// @param unpack_lo  true = unpack into low-z ghost, false = unpack into high-z ghost
/// @param stream     CUDA stream
void launch_unpack_z_face(
    double* field,
    const double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool unpack_lo,
    void* stream = nullptr);

} // namespace cuda_kernels
} // namespace nncfd
