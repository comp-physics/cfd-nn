#pragma once

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu_kernels {

// ============================================================================
// Unified Kernel: Compute cell-centered gradients from staggered MAC grid
// ============================================================================
// This is the SINGLE SOURCE OF TRUTH for gradient computation. It provides:
//   - GPU path: Uses OpenMP target offloading (when USE_GPU_OFFLOAD defined)
//   - CPU path: Same logic without offloading (when USE_GPU_OFFLOAD not defined)
//
// The abstraction wrapper (compute_gradients_from_mac in features.cpp) calls this
// function after extracting raw pointers from the abstraction types.
//
// This kernel matches VectorField's staggered layout:
//   - u stored at x-faces: size (Ny+2Ng) × (Nx+2Ng+1)
//   - v stored at y-faces: size (Ny+2Ng+1) × (Nx+2Ng)
//   - Outputs are cell-centered: size (Ny+2Ng) × (Nx+2Ng)
//
// Uses central differences:
//   dudx(i,j) = (u(i+1,j) - u(i-1,j)) / (2*dx)
//   dudy(i,j) = (u(i,j+1) - u(i,j-1)) / (2*dy)
//   etc.
// ============================================================================
void compute_gradients_from_mac_gpu(
    const double* u_face,        // u at x-faces
    const double* v_face,        // v at y-faces
    const double* w_face,        // w at z-faces (nullptr for 2D)
    double* dudx_cell,           // Output: cell-centered gradients
    double* dudy_cell,
    double* dvdx_cell,
    double* dvdy_cell,
    double* dudz_cell,           // Output: 3D gradients (nullptr for 2D)
    double* dvdz_cell,
    double* dwdx_cell,
    double* dwdy_cell,
    double* dwdz_cell,
    int Nx, int Ny, int Nz,      // Interior dimensions
    int Ng,                      // Ghost cells
    double dx, double dy, double dz, // Grid spacing
    int u_stride,                // u row stride = Nx+2Ng+1
    int v_stride,                // v row stride = Nx+2Ng
    int cell_stride,             // cell row stride = Nx+2Ng
    int u_plane_stride,          // u plane stride (for 3D)
    int v_plane_stride,          // v plane stride (for 3D)
    int w_stride,                // w row stride (for 3D)
    int w_plane_stride,          // w plane stride (for 3D)
    int cell_plane_stride,       // cell plane stride (for 3D)
    int u_total_size,            // Total u array size
    int v_total_size,            // Total v array size
    int w_total_size,            // Total w array size
    int cell_total_size,         // Total cell array size
    const double* dyc = nullptr, // Center-to-center y-spacing for stretched grids
    int dyc_size = 0             // Size of dyc array (for map clause)
);

// ============================================================================
// GPU Kernel: Compute 5 Pope invariants for MLP (same inputs as TBNN)
// ============================================================================
// Input: gradients (dudx, dudy, dvdx, dvdy), k, omega
//        Plus staggered velocity fields for z-gradients in 3D
// Output: features (5 per cell): [tr(S_hat^2), tr(Omega_hat^2),
//         tr(S_hat^3), tr(Omega_hat^2*S_hat), tr(Omega_hat^2*S_hat^2)]
// S_hat = (k/epsilon) * S_ij, Omega_hat = (k/epsilon) * Omega_ij
// ============================================================================
void compute_pope_invariants_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* u_face, const double* v_face,
    const double* w_face,                  // w at z-faces (nullptr for 2D)
    double* features,                      // Output: n_cells * 5
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int total_cells, int u_total, int v_total, int w_total,
    double dx, double dy, double dz
);

// ============================================================================
// GPU Kernel: Compute TBNN features and tensor basis for all cells
// ============================================================================
// Input: gradients (dudx, dudy, dvdx, dvdy) with ghosts, k, omega, wall_distance
// Output: features (5 per cell), basis (10*6 = 60 per cell) - interior only
// ============================================================================
void compute_tbnn_features_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* wall_distance,
    double* features,                      // Output: Nx*Ny*Nz * 5
    double* basis,                         // Output: Nx*Ny*Nz * 60 (10 basis tensors, 6 components each)
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int total_cells,
    double nu, double delta
);

// ============================================================================
// GPU Kernel: Postprocess MLP outputs (scalar nu_t) to ghosted field
// ============================================================================
// Input: NN outputs (scalar nu_t predictions)
// Output: nu_t field with ghost cells, applying clipping and realizability
// ============================================================================
void postprocess_mlp_outputs_gpu(
    const double* nn_outputs,              // Input: n_cells * 1 (interior only)
    double* nu_t_field,                    // Output: full field with ghosts
    int Nx, int Ny, int Nz, int Ng,
    int stride,                            // Row stride = Nx+2Ng
    int plane_stride,                      // Plane stride for 3D
    int total_field_size,                  // Total field array size
    double nu_t_max                        // Maximum eddy viscosity
);

// ============================================================================
// GPU Kernel: Postprocess TBNN outputs to compute nu_t and tau_ij
// ============================================================================
// Input: NN outputs (G coefficients), basis tensors, k, gradients
// Output: nu_t (with ghosts), optionally tau_ij (with ghosts)
// ============================================================================
void postprocess_nn_outputs_gpu(
    const double* nn_outputs,              // Input: Nx*Ny*Nz * output_dim (interior only)
    const double* basis,                   // Input: Nx*Ny*Nz * 60 (interior only)
    const double* k,                       // Input: total_cells (with ghosts)
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    double* nu_t,                          // Output: total_cells (with ghosts)
    double* tau_xx, double* tau_xy, double* tau_yy,  // Optional output (can be nullptr)
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int total_cells,
    int output_dim,
    double nu_ref                          // Reference viscosity for clipping
);


// ============================================================================
// GPU Kernel: Boussinesq k-omega closure (ν_t = k/ω with limiters)
// ============================================================================
// Computes eddy viscosity from k and omega fields with realizability constraints.
// This is the simplest closure for k-ω models.
// ============================================================================
void compute_boussinesq_closure_gpu(
    const double* k,                    // TKE field (cell-centered, with ghost cells)
    const double* omega,                // Specific dissipation rate (cell-centered, with ghost cells)
    double* nu_t,                       // Output: eddy viscosity (cell-centered, with ghost cells)
    int Nx, int Ny, int Nz,             // Interior dimensions (Nz=1 for 2D)
    int Ng,                             // Ghost cells
    int stride,                         // Row stride = Nx+2Ng
    int cell_plane_stride,              // Plane stride = (Nx+2Ng)*(Ny+2Ng)
    int total_size,                     // Total array size
    double nu,                          // Laminar viscosity
    double k_min, double omega_min,     // Minimum values for clipping
    double nu_t_max                     // Maximum eddy viscosity (relative to nu)
);

// ============================================================================
// GPU Kernel: SST k-omega closure (ν_t = a₁k / max(a₁ω, SF₂))
// ============================================================================
// Computes SST eddy viscosity with strain-rate limiter and F2 blending function.
// More sophisticated than Boussinesq for separated flows.
// ============================================================================
void compute_sst_closure_gpu(
    const double* k,                    // TKE field (cell-centered, with ghost cells)
    const double* omega,                // Specific dissipation rate (cell-centered, with ghost cells)
    const double* dudx,                 // Velocity gradients (cell-centered, with ghost cells)
    const double* dudy,
    const double* dvdx,
    const double* dvdy,
    const double* wall_distance,        // Wall distance (cell-centered, with ghost cells)
    double* nu_t,                       // Output: eddy viscosity (cell-centered, with ghost cells)
    int Nx, int Ny, int Nz,             // Interior dimensions (Nz=1 for 2D)
    int Ng,                             // Ghost cells
    int stride,                         // Row stride = Nx+2Ng
    int cell_plane_stride,              // Plane stride = (Nx+2Ng)*(Ny+2Ng)
    int total_size,                     // Total 3D array size for k/omega/nu_t/gradients
    int wall_dist_size,                 // Size of wall_distance array
    double nu,                          // Laminar viscosity
    double a1,                          // SST constant (0.31)
    double beta_star,                   // SST constant (0.09)
    double k_min, double omega_min,     // Minimum values for clipping
    double nu_t_max                     // Maximum eddy viscosity (relative to nu)
);

// ============================================================================
// GPU Kernel: k-omega transport step (single timestep advance)
// ============================================================================
// Advances k and omega fields by one timestep using explicit Euler.
// Computes production, advection, diffusion, and source/sink terms.
// ============================================================================
void komega_transport_step_gpu(
    // Current fields (with ghost cells)
    const double* u, const double* v,   // Velocity
    double* k, double* omega,           // k and omega (modified in-place)
    const double* nu_t_prev,            // Previous eddy viscosity (cell-centered, with ghost cells)
    // Mesh parameters
    int Nx, int Ny, int Nz, int Ng,     // Interior dims and ghost cells (Nz=1 for 2D)
    int stride,                         // Row stride = Nx+2Ng
    int cell_plane_stride,              // Plane stride = (Nx+2Ng)*(Ny+2Ng)
    int u_stride, int v_stride,         // Strides for staggered velocity
    int u_plane_stride, int v_plane_stride, // Plane strides for staggered velocity
    int total_size,                     // Total size for k/omega/nu_t (with ghosts)
    int vel_u_size, int vel_v_size,     // Total sizes for u/v
    double dx, double dy, double dt,    // Grid spacing and timestep
    // Model constants
    double nu, double sigma_k, double sigma_omega,
    double beta, double beta_star, double alpha,
    double k_min, double k_max,
    double omega_min, double omega_max
);

} // namespace gpu_kernels
} // namespace nncfd


