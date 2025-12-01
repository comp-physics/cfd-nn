#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

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
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: use OpenMP target offload if available
    if (omp_get_num_devices() > 0) {
        const int n_cells = mesh.Nx * mesh.Ny;
        
        // Precompute wall distances (Mesh doesn't have direct data access)
        std::vector<double> wall_dist_flat(n_cells);
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                wall_dist_flat[idx++] = mesh.wall_distance(i, j);
            }
        }
        
        // Get raw pointers for GPU
        const double* dudx_ptr = dudx_.data().data();
        const double* dudy_ptr = dudy_.data().data();
        const double* dvdx_ptr = dvdx_.data().data();
        const double* dvdy_ptr = dvdy_.data().data();
        const double* wall_dist_ptr = wall_dist_flat.data();
        double* nu_t_ptr = nu_t.data().data();
        
        // GPU kernel: compute mixing length eddy viscosity
        #pragma omp target teams distribute parallel for \
            map(to: dudx_ptr[0:n_cells], dudy_ptr[0:n_cells], \
                    dvdx_ptr[0:n_cells], dvdy_ptr[0:n_cells], \
                    wall_dist_ptr[0:n_cells]) \
            map(from: nu_t_ptr[0:n_cells])
        for (int idx = 0; idx < n_cells; ++idx) {
            // Get strain rate components
            double Sxx = dudx_ptr[idx];
            double Syy = dvdy_ptr[idx];
            double Sxy = 0.5 * (dudy_ptr[idx] + dvdx_ptr[idx]);
            
            // Strain rate magnitude
            double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            // Wall distance and y+
            double y_wall = wall_dist_ptr[idx];
            double y_plus = y_wall * u_tau / nu_;
            
            // van Driest damping
            double damping = 1.0 - exp(-y_plus / A_plus_);
            
            // Mixing length (capped at delta/2)
            double l_mix = kappa_ * y_wall * damping;
            if (l_mix > 0.5 * delta_) {
                l_mix = 0.5 * delta_;
            }
            
            // Eddy viscosity
            nu_t_ptr[idx] = l_mix * l_mix * S_mag;
        }
        
        return;
    }
#endif
    
    // CPU path
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / nu_;
            
            // van Driest damping
            double damping = 1.0 - std::exp(-y_plus / A_plus_);
            
            // Mixing length (capped at delta/2)
            double l_mix = kappa_ * y_wall * damping;
            l_mix = std::min(l_mix, 0.5 * delta_);
            
            // Strain rate magnitude
            double Sxx = dudx_(i, j);
            double Syy = dvdy_(i, j);
            double Sxy = 0.5 * (dudy_(i, j) + dvdx_(i, j));
            double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            // Eddy viscosity
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
            
        default:
            return nullptr;
    }
}

} // namespace nncfd

