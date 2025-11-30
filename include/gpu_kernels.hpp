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

} // namespace gpu_kernels
} // namespace nncfd

