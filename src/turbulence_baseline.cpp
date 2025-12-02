#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "turbulence_transport.hpp"
#include "turbulence_earsm.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

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

void MixingLengthModel::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij) {
    
    (void)k;
    (void)omega;
    (void)tau_ij;
    
    // Compute velocity gradients
    if (dudx_.data().empty()) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
    }
    
    compute_all_velocity_gradients(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    // Mixing length model with van Driest damping:
    // l_m = kappa * y * (1 - exp(-y+/A+))
    // nu_t = l_m^2 * |S|
    
    // First, estimate u_tau from wall gradient
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
    
    u_tau = std::max(u_tau, 1e-10);  // Avoid division by zero

    [[maybe_unused]] const int Nx = mesh.Nx;
    [[maybe_unused]] const int Ny = mesh.Ny;
    const int stride = mesh.total_Nx();
    const int total_cells = mesh.total_cells();

    // Mesh parameters for wall distance computation
    const double y_min = mesh.y_min;
    const double y_max = mesh.y_max;
    const double dy = mesh.dy;
    const int Ng = mesh.Nghost;

#ifdef USE_GPU_OFFLOAD
    // GPU path: same kernel, different parallelization + data source
    if (omp_get_num_devices() > 0 && Nx >= 32 && Ny >= 32) {
        const int n_cells = Nx * Ny;

        // Get references to underlying vectors (not pointers, to avoid NVHPC bug)
        std::vector<double>& dudx_vec = dudx_.data();
        std::vector<double>& dudy_vec = dudy_.data();
        std::vector<double>& dvdx_vec = dvdx_.data();
        std::vector<double>& dvdy_vec = dvdy_.data();
        std::vector<double>& nu_t_vec = nu_t.data();

        // GPU kernel: compute mixing length eddy viscosity (inlined to match CPU exactly)
        #pragma omp target teams distribute parallel for \
            map(to: dudx_vec[0:total_cells], dudy_vec[0:total_cells], \
                    dvdx_vec[0:total_cells], dvdy_vec[0:total_cells]) \
            map(from: nu_t_vec[0:total_cells]) \
            firstprivate(u_tau, nu_, kappa_, A_plus_, delta_, stride, Nx, Ng, y_min, y_max, dy)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;  // interior i (skip ghost)
            int j = idx / Nx + 1;  // interior j (skip ghost)
            int cell_idx = j * stride + i;

            // Compute wall distance on-the-fly (channel flow: min distance to y_min or y_max)
            double y = y_min + (j - Ng + 0.5) * dy;
            double dist_lo = (y - y_min > 0) ? (y - y_min) : -(y - y_min);
            double dist_hi = (y_max - y > 0) ? (y_max - y) : -(y_max - y);
            double y_wall = (dist_lo < dist_hi) ? dist_lo : dist_hi;

            // Inline kernel computation (same as CPU path, for guaranteed matching results)
            double Sxx = dudx_vec[cell_idx];
            double Syy = dvdy_vec[cell_idx];
            double Sxy = 0.5 * (dudy_vec[cell_idx] + dvdx_vec[cell_idx]);
            double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            double y_plus = y_wall * u_tau / nu_;
            double damping = 1.0 - exp(-y_plus / A_plus_);
            double l_mix = kappa_ * y_wall * damping;
            if (l_mix > 0.5 * delta_) {
                l_mix = 0.5 * delta_;
            }
            nu_t_vec[cell_idx] = l_mix * l_mix * S_mag;
        }

        return;
    }
#endif

    // CPU path
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / nu_;
            
            double damping = std::exp(-y_plus / A_plus_);
            damping = 1.0 - damping;
            
            double l_mix = kappa_ * y_wall * damping;
            l_mix = std::min(l_mix, 0.5 * delta_);
            
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
    TensorField* tau_ij) {
    
    (void)k;
    (void)omega;
    (void)tau_ij;
    
    // Algebraic model that estimates k and omega from mean flow
    // Based on equilibrium assumptions
    
    if (dudx_.data().empty()) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
    }
    
    compute_all_velocity_gradients(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
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

