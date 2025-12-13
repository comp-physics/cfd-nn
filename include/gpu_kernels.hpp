#pragma once

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu_kernels {

// ============================================================================
// GPU Kernel: Compute velocity gradients for all cells
// ============================================================================
// Input: u, v velocity arrays (flat, size = total_cells)
// Output: dudx, dudy, dvdx, dvdy gradient arrays (flat, size = n_interior)
// 
// Mesh layout: total cells = (Nx+2) * (Ny+2) with ghost cells
// Interior cells: i in [1, Nx], j in [1, Ny]
// ============================================================================
void compute_velocity_gradients_gpu(
    const double* u, const double* v,      // Input velocity (total_cells)
    double* dudx, double* dudy,            // Output gradients (n_interior)
    double* dvdx, double* dvdy,
    int Nx, int Ny,                        // Interior dimensions
    double dx, double dy                   // Grid spacing
);

// ============================================================================
// GPU Kernel: Compute cell-centered gradients from staggered MAC grid
// ============================================================================
// This kernel matches VectorField's staggered layout:
//   - u stored at x-faces: size (Ny+2Ng) × (Nx+2Ng+1)
//   - v stored at y-faces: size (Ny+2Ng+1) × (Nx+2Ng)
//   - Outputs are cell-centered: size (Ny+2Ng) × (Nx+2Ng)
//
// Uses central differences matching CPU compute_velocity_gradient():
//   dudx(i,j) = (u(i+1,j) - u(i-1,j)) / (2*dx)
//   dudy(i,j) = (u(i,j+1) - u(i,j-1)) / (2*dy)
//   etc.
// ============================================================================
void compute_gradients_from_mac_gpu(
    const double* u_face,        // u at x-faces: (Ny+2Ng) × (Nx+2Ng+1)
    const double* v_face,        // v at y-faces: (Ny+2Ng+1) × (Nx+2Ng)
    double* dudx_cell,           // Output: cell-centered (Ny+2Ng) × (Nx+2Ng)
    double* dudy_cell,
    double* dvdx_cell,
    double* dvdy_cell,
    int Nx, int Ny,              // Interior dimensions
    int Ng,                      // Ghost cells
    double dx, double dy,        // Grid spacing
    int u_stride,                // u row stride = Nx+2Ng+1
    int v_stride,                // v row stride = Nx+2Ng
    int cell_stride,             // cell row stride = Nx+2Ng
    int u_total_size,            // Total u array size
    int v_total_size,            // Total v array size
    int cell_total_size          // Total cell array size
);

// ============================================================================
// GPU Kernel: Compute TBNN features and tensor basis for all cells
// ============================================================================
// Input: gradients (dudx, dudy, dvdx, dvdy), k, omega, wall_distance
// Output: features (5 per cell), basis (4*3 = 12 per cell)
// ============================================================================
void compute_tbnn_features_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* wall_distance,
    double* features,                      // Output: n_cells * 5
    double* basis,                         // Output: n_cells * 12 (4 basis tensors, 3 components each)
    int n_cells,
    double nu, double delta
);

// ============================================================================
// GPU Kernel: Postprocess NN outputs to compute nu_t
// ============================================================================
// Input: NN outputs (G coefficients), basis tensors, k, gradients
// Output: nu_t, optionally tau_ij
// ============================================================================
void postprocess_nn_outputs_gpu(
    const double* nn_outputs,              // Input: n_cells * output_dim
    const double* basis,                   // Input: n_cells * 12
    const double* k,                       // Input: n_cells
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    double* nu_t,                          // Output: n_cells
    double* tau_xx, double* tau_xy, double* tau_yy,  // Optional output (can be nullptr)
    int n_cells, int output_dim,
    double nu_ref                          // Reference viscosity for clipping
);

// ============================================================================
// GPU Kernel: Full TBNN pipeline (features + NN + postprocess)
// ============================================================================
// This combines all operations to minimize data transfers
// ============================================================================
void tbnn_full_pipeline_gpu(
    // Velocity field (with ghost cells)
    const double* u, const double* v,
    // Turbulence quantities (interior only)
    const double* k, const double* omega,
    const double* wall_distance,
    // NN weights (already on GPU via MLP::upload_to_gpu)
    const double* nn_weights, const double* nn_biases,
    const int* weight_offsets, const int* bias_offsets,
    const int* layer_dims, const int* activation_types,
    int n_layers,
    // Scaling parameters (can be nullptr)
    const double* input_means, const double* input_stds,
    int scale_size,
    // Workspace (pre-allocated on GPU)
    double* workspace,
    // Output
    double* nu_t,
    double* tau_xx, double* tau_xy, double* tau_yy,  // Can be nullptr
    // Mesh parameters
    int Nx, int Ny, double dx, double dy,
    double nu, double delta
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
    int Nx, int Ny,                     // Interior dimensions
    int Ng,                             // Ghost cells
    int stride,                         // Row stride = Nx+2Ng
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
    const double* wall_distance,        // Wall distance (cell-centered, NO ghost cells)
    double* nu_t,                       // Output: eddy viscosity (cell-centered, with ghost cells)
    int Nx, int Ny,                     // Interior dimensions
    int Ng,                             // Ghost cells
    int stride,                         // Row stride = Nx+2Ng
    int total_size,                     // Total array size for k/omega/nu_t/gradients
    int wall_dist_size,                 // Size of wall_distance array (interior only)
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
    int Nx, int Ny, int Ng,             // Interior dims and ghost cells
    int stride,                         // Row stride = Nx+2Ng
    int u_stride, int v_stride,         // Strides for staggered velocity
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


