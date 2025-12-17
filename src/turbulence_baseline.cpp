/// @file turbulence_baseline.cpp
/// @brief Implementation of algebraic turbulence models (mixing length, GEP)
///
/// This file implements simple algebraic eddy viscosity models:
/// - Mixing length model with van Driest damping (wall-resolved RANS)
/// - Algebraic k-omega model (simplified without transport equations)
/// - Factory function for creating turbulence models from configuration
///
/// These models provide fast, zero-equation closures suitable for attached
/// boundary layers but limited for separated flows.

#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "turbulence_transport.hpp"
#include "turbulence_earsm.hpp"
#include "gpu_kernels.hpp"
#include "features.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nncfd {

// ============================================================================
// Unified CPU/GPU kernel for mixing length turbulence model
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

inline void mixing_length_cell_kernel(
    int cell_idx, double u_tau, double nu,
    double kappa, double A_plus, double delta,
    double y_wall,  // Pass as value, not via pointer
    const double* dudx_ptr, const double* dudy_ptr,
    const double* dvdx_ptr, const double* dvdy_ptr,
    double* nu_t_ptr)
{
    // Get strain rate components
    double Sxx = dudx_ptr[cell_idx];
    double Syy = dvdy_ptr[cell_idx];
    double Sxy = 0.5 * (dudy_ptr[cell_idx] + dvdx_ptr[cell_idx]);

    // Strain rate magnitude
    double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));

    // y+ from wall distance
    double y_plus = y_wall * u_tau / nu;

    // van Driest damping
    double damping = 1.0 - exp(-y_plus / A_plus);

    // Mixing length (capped at delta/2)
    double l_mix = kappa * y_wall * damping;
    if (l_mix > 0.5 * delta) {
        l_mix = 0.5 * delta;
    }

    // Eddy viscosity
    nu_t_ptr[cell_idx] = l_mix * l_mix * S_mag;
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

MixingLengthModel::MixingLengthModel() = default;

MixingLengthModel::~MixingLengthModel() {
    cleanup_gpu_buffers();
}

void MixingLengthModel::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    const int total_cells = mesh.total_cells();
    
    // Check if already allocated for this mesh size
    if (buffers_on_gpu_ && cached_total_cells_ == total_cells) {
        gpu_ready_ = true;
        return;
    }
    
    // Free old buffers if they exist
    free_gpu_arrays();
    
    // Allocate CPU storage for gradients if needed
    if (dudx_.data().empty()) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
    }
    
    // Allocate GPU buffers
    allocate_gpu_arrays(mesh);
    
    cached_total_cells_ = total_cells;
    buffers_on_gpu_ = true;
    gpu_ready_ = true;
#else
    (void)mesh;
    gpu_ready_ = false;
#endif
}

#ifdef USE_GPU_OFFLOAD
void MixingLengthModel::allocate_gpu_arrays(const Mesh& mesh) {
    const int total_cells = mesh.total_cells();
    const int stride = mesh.total_Nx();
    
    // Allocate flat arrays
    nu_t_gpu_flat_.resize(total_cells, 0.0);
    y_wall_flat_.resize(total_cells, 0.0);
    
    // Precompute wall distances
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            int cell_idx = j * stride + i;
            y_wall_flat_[cell_idx] = mesh.wall_distance(i, j);
        }
    }
    
    // Get pointers
    double* dudx_ptr = dudx_.data().data();
    double* dudy_ptr = dudy_.data().data();
    double* dvdx_ptr = dvdx_.data().data();
    double* dvdy_ptr = dvdy_.data().data();
    double* nu_t_ptr = nu_t_gpu_flat_.data();
    double* y_wall_ptr = y_wall_flat_.data();
    
    // Map to GPU persistently
    #pragma omp target enter data map(alloc: dudx_ptr[0:total_cells])
    #pragma omp target enter data map(alloc: dudy_ptr[0:total_cells])
    #pragma omp target enter data map(alloc: dvdx_ptr[0:total_cells])
    #pragma omp target enter data map(alloc: dvdy_ptr[0:total_cells])
    #pragma omp target enter data map(to: y_wall_ptr[0:total_cells])
    #pragma omp target enter data map(alloc: nu_t_ptr[0:total_cells])
}

void MixingLengthModel::free_gpu_arrays() {
    if (!buffers_on_gpu_) return;
    
    const int size = cached_total_cells_;
    if (size == 0) return;
    
    double* dudx_ptr = dudx_.data().data();
    double* dudy_ptr = dudy_.data().data();
    double* dvdx_ptr = dvdx_.data().data();
    double* dvdy_ptr = dvdy_.data().data();
    double* nu_t_ptr = nu_t_gpu_flat_.data();
    double* y_wall_ptr = y_wall_flat_.data();
    
    #pragma omp target exit data map(delete: dudx_ptr[0:size])
    #pragma omp target exit data map(delete: dudy_ptr[0:size])
    #pragma omp target exit data map(delete: dvdx_ptr[0:size])
    #pragma omp target exit data map(delete: dvdy_ptr[0:size])
    #pragma omp target exit data map(delete: y_wall_ptr[0:size])
    #pragma omp target exit data map(delete: nu_t_ptr[0:size])
    
    buffers_on_gpu_ = false;
}
#endif

void MixingLengthModel::cleanup_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    free_gpu_arrays();
#endif
    gpu_ready_ = false;
}

void MixingLengthModel::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    
    (void)k;
    (void)omega;
    (void)tau_ij;
    
    // PHASE 1 GPU OPTIMIZATION:
    // If device_view is provided and valid, use GPU path with MAC gradient kernel
    // Otherwise fall back to CPU path
    
#ifdef USE_GPU_OFFLOAD
    if (device_view && device_view->is_valid()) {
        // ==================================================================
        // GPU PATH: Use solver-owned device buffers and MAC gradient kernel
        // ==================================================================
        
        // First, estimate u_tau from wall gradient (CPU side, once per step)
        double u_tau = 0.0;
        {
            int j = mesh.j_begin();
            // Compute wall shear from velocity at first interior point
            // For staggered grid: u is at x-faces, so u(i,j) is at y=0 boundary
            double dudy_wall = 0.0;
            int count = 0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u_wall = velocity.u(i, j);
                double u_next = velocity.u(i, j+1);
                dudy_wall += std::abs((u_next - u_wall) / mesh.dy);
                ++count;
            }
            dudy_wall /= count;
            double tau_w = nu_ * dudy_wall;
            u_tau = std::sqrt(tau_w);
        }
        u_tau = std::max(u_tau, 1e-10);
        
        // Compute mixing length eddy viscosity on GPU
        const int Nx = device_view->Nx;
        const int Ny = device_view->Ny;
        const int Ng = device_view->Ng;
        
        // Calculate array sizes for map(present:...) clauses
        const int u_total_size = device_view->u_stride * (Ny + 2*Ng);
        const int v_total_size = device_view->v_stride * (Ny + 2*Ng + 1);
        const int cell_total_size = device_view->cell_stride * (Ny + 2*Ng);
        
        // Compute gradients on GPU using MAC-aware kernel
        gpu_kernels::compute_gradients_from_mac_gpu(
            device_view->u_face,
            device_view->v_face,
            device_view->dudx,
            device_view->dudy,
            device_view->dvdx,
            device_view->dvdy,
            Nx, Ny, Ng,
            device_view->dx,
            device_view->dy,
            device_view->u_stride,
            device_view->v_stride,
            device_view->cell_stride,
            u_total_size,
            v_total_size,
            cell_total_size
        );
        const int stride = device_view->cell_stride;
        const int n_cells = Nx * Ny;
        
        // Copy member variables to local scope (NVHPC workaround)
        const double nu_local = nu_;
        const double kappa_local = kappa_;
        const double A_plus_local = A_plus_;
        const double delta_local = delta_;
        
        // Get device pointers (already on GPU via device_view)
        double* dudx_ptr = device_view->dudx;
        double* dudy_ptr = device_view->dudy;
        double* dvdx_ptr = device_view->dvdx;
        double* dvdy_ptr = device_view->dvdy;
        double* nu_t_ptr = device_view->nu_t;
        double* wall_dist_ptr = device_view->wall_distance;
        
        // GPU kernel: compute mixing length eddy viscosity
        // Use map(present:...) since these are solver-mapped host pointers
        #pragma omp target teams distribute parallel for \
            map(present: dudx_ptr[0:cell_total_size], dudy_ptr[0:cell_total_size], \
                         dvdx_ptr[0:cell_total_size], dvdy_ptr[0:cell_total_size], \
                         nu_t_ptr[0:cell_total_size], wall_dist_ptr[0:cell_total_size])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;  // interior i (add ghost offset)
            const int j = idx / Nx + Ng;  // interior j (add ghost offset)
            const int cell_idx = j * stride + i;
            
            // Get wall distance
            const double y_wall = wall_dist_ptr[cell_idx];
            
            // Compute strain rate magnitude
            const double Sxx = dudx_ptr[cell_idx];
            const double Syy = dvdy_ptr[cell_idx];
            const double Sxy = 0.5 * (dudy_ptr[cell_idx] + dvdx_ptr[cell_idx]);
            const double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            // y+ and van Driest damping
            const double y_plus = y_wall * u_tau / nu_local;
            const double damping = 1.0 - exp(-y_plus / A_plus_local);
            
            // Mixing length (capped at delta/2)
            double l_mix = kappa_local * y_wall * damping;
            if (l_mix > 0.5 * delta_local) {
                l_mix = 0.5 * delta_local;
            }
            
            // Eddy viscosity
            nu_t_ptr[cell_idx] = l_mix * l_mix * S_mag;
        }
        
        // Done! nu_t is now on GPU, will be synced by solver when needed
        return;
    }
#else
    (void)device_view;  // Unused in CPU-only build
#endif
    
    // ==================================================================
    // CPU PATH: MAC-aware gradient computation (matches GPU)
    // ==================================================================
    
    // Initialize gradient fields if needed (lazy initialization)
    if (dudx_.data().empty()) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
    }
    
    // Compute velocity gradients on CPU using MAC-aware function
    // This matches the GPU kernel's indexing for consistent results
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    // Mixing length model with van Driest damping:
    // l_m = kappa * y * (1 - exp(-y+/A+))
    // nu_t = l_m^2 * |S|
    
    // First, estimate u_tau from wall gradient
    // Use same formula as GPU path for bit-for-bit consistency
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_wall = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u_wall = velocity.u(i, j);
            double u_next = velocity.u(i, j+1);
            dudy_wall += std::abs((u_next - u_wall) / mesh.dy);
            ++count;
        }
        dudy_wall /= count;
        double tau_w = nu_ * dudy_wall;
        u_tau = std::sqrt(tau_w);
    }
    
    u_tau = std::max(u_tau, 1e-10);  // Avoid division by zero

    // CPU path
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / nu_;
            
            // Use same formulation as GPU (single expression)
            double damping = 1.0 - std::exp(-y_plus / A_plus_);
            
            double l_mix = kappa_ * y_wall * damping;
            // Use same min operation as GPU
            if (l_mix > 0.5 * delta_) {
                l_mix = 0.5 * delta_;
            }
            
            double Sxx = dudx_(i, j);
            double Syy = dvdy_(i, j);
            double Sxy = 0.5 * (dudy_(i, j) + dvdx_(i, j));
            double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            nu_t(i, j) = l_mix * l_mix * S_mag;
        }
    }
}

AlgebraicKOmegaModel::AlgebraicKOmegaModel() = default;

void AlgebraicKOmegaModel::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    
    (void)k;
    (void)omega;
    (void)tau_ij;
    (void)device_view;  // Not yet implemented for this model
    
    // Algebraic model that estimates k and omega from mean flow
    // Based on equilibrium assumptions
    
    if (dudx_.data().empty()) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
    }
    
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    // Estimate friction velocity
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_wall = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            dudy_wall += std::abs(dudy_(i, j));
            ++count;
        }
        dudy_wall /= count;
        double tau_w = nu_ * dudy_wall;
        u_tau = std::sqrt(tau_w);
    }
    
    u_tau = std::max(u_tau, 1e-10);
    
    // In log-law region: k ~ u_tau^2 / sqrt(C_mu)
    // omega ~ u_tau / (kappa * y)
    
    const double kappa = 0.41;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / nu_;
            
            // Estimate k (with damping near wall)
            double f_mu = 1.0 - std::exp(-y_plus / 26.0);
            double k_est = (u_tau * u_tau / std::sqrt(C_mu_)) * f_mu * f_mu;
            
            // Estimate omega
            double omega_est = u_tau / (kappa * std::max(y_wall, 1e-10) * f_mu + 1e-10);
            
            // Blend with viscous sublayer estimate
            if (y_plus < 5.0) {
                omega_est = 6.0 * nu_ / (0.075 * y_wall * y_wall + 1e-20);
            }
            
            // nu_t = k / omega
            nu_t(i, j) = std::max(0.0, k_est / std::max(omega_est, 1e-10));
            
            // Limit nu_t
            nu_t(i, j) = std::min(nu_t(i, j), 1000.0 * nu_);
        }
    }
}

// Factory function implementation
std::unique_ptr<TurbulenceModel> create_turbulence_model(
    TurbulenceModelType type,
    const std::string& weights_path,
    const std::string& scaling_path) {
    
    switch (type) {
        case TurbulenceModelType::None:
            return nullptr;
            
        case TurbulenceModelType::Baseline:
            return std::make_unique<MixingLengthModel>();
            
        case TurbulenceModelType::GEP:
            return std::make_unique<TurbulenceGEP>();
            
        case TurbulenceModelType::NNMLP: {
            auto model = std::make_unique<TurbulenceNNMLP>();
            if (!weights_path.empty()) {
                model->load(weights_path, scaling_path);
                // Upload NN weights to GPU (if available)
                model->upload_to_gpu();
            }
            return model;
        }
            
        case TurbulenceModelType::NNTBNN: {
            auto model = std::make_unique<TurbulenceNNTBNN>();
            if (!weights_path.empty()) {
                model->load(weights_path, scaling_path);
                // Upload NN weights to GPU (if available)
                model->upload_to_gpu();
            }
            return model;
        }
        
        // Transport equation models
        case TurbulenceModelType::SSTKOmega: {
            return std::make_unique<SSTKOmegaTransport>();
        }
            
        case TurbulenceModelType::KOmega: {
            return std::make_unique<KOmegaTransport>();
        }
        
        // EARSM models (SST transport + EARSM closure)
        case TurbulenceModelType::EARSM_WJ: {
            return std::make_unique<SSTWithEARSM>(EARSMType::WallinJohansson2000);
        }
            
        case TurbulenceModelType::EARSM_GS: {
            return std::make_unique<SSTWithEARSM>(EARSMType::GatskiSpeziale1993);
        }
            
        case TurbulenceModelType::EARSM_Pope: {
            return std::make_unique<SSTWithEARSM>(EARSMType::Pope1975);
        }
            
        default:
            return nullptr;
    }
}

} // namespace nncfd

