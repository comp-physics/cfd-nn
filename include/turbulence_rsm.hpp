#pragma once

/// @file turbulence_rsm.hpp
/// @brief Reynolds Stress Model with SSG pressure-strain and omega equation
///
/// Full RSM solving 7 transport equations: 6 Reynolds stress components
/// (R_xx, R_yy, R_zz, R_xy, R_xz, R_yz) + omega length-scale equation.
/// Uses SSG (Speziale-Sarkar-Gatski 1991) pressure-strain closure and
/// omega equation similar to Wilcox's stress-omega model.

#include "turbulence_model.hpp"
#include "fields.hpp"
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// SSG pressure-strain + omega transport constants
struct RSMConstants {
    // SSG pressure-strain coefficients (Speziale, Sarkar, Gatski 1991)
    double C1       = 3.4;
    double C1_star  = 1.8;
    double C2       = 4.2;
    double C3       = 0.8;
    double C3_star  = 1.3;
    double C4       = 1.25;
    double C5       = 0.4;

    // Omega equation constants (Wilcox stress-omega)
    double alpha_omega = 0.52;
    double beta_omega  = 0.0708;
    double beta_star   = 0.09;
    double sigma_omega = 0.5;

    // Reynolds stress diffusion
    double sigma_R = 0.22;

    // Numerical limits
    double k_min     = 1e-10;
    double k_max     = 100.0;
    double omega_min = 1e-10;
    double omega_max = 1e8;
    double R_min     = -100.0;  ///< Floor for off-diagonal stresses
    double R_max     = 100.0;   ///< Ceiling for stress components
};

/// Reynolds Stress Model: SSG pressure-strain + omega equation
class RSMModel : public TurbulenceModel {
public:
    RSMModel(const RSMConstants& constants = RSMConstants());
    ~RSMModel();

    // Delete copy/move to prevent double-free of GPU buffers
    RSMModel(const RSMModel&) = delete;
    RSMModel& operator=(const RSMModel&) = delete;
    RSMModel(RSMModel&&) = delete;
    RSMModel& operator=(RSMModel&&) = delete;

    // TurbulenceModel interface
    void initialize(const Mesh& mesh, const VectorField& velocity) override;
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return buffers_on_gpu_; }

    bool uses_transport_equations() const override { return true; }
    bool provides_reynolds_stresses() const override { return true; }

    void advance_turbulence(
        const Mesh& mesh,
        const VectorField& velocity,
        double dt,
        ScalarField& k,
        ScalarField& omega,
        const ScalarField& nu_t_prev,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;

    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;

    std::string name() const override { return "RSM-SSG"; }

    /// Access constants for tuning
    RSMConstants& constants() { return constants_; }
    const RSMConstants& constants() const { return constants_; }

private:
    RSMConstants constants_;

    // Internal Reynolds stress storage (6 symmetric components)
    ScalarField R_xx_, R_yy_, R_zz_, R_xy_, R_xz_, R_yz_;

    bool initialized_ = false;
    [[maybe_unused]] bool buffers_on_gpu_ = false;
    [[maybe_unused]] int cached_total_cells_ = 0;

    // GPU flat buffers for R_ij (6 arrays)
    std::vector<double> Rxx_flat_, Ryy_flat_, Rzz_flat_;
    std::vector<double> Rxy_flat_, Rxz_flat_, Ryz_flat_;

    void ensure_initialized(const Mesh& mesh, const ScalarField& k);
    void allocate_gpu_buffers(const Mesh& mesh);
    void free_gpu_buffers();
};

} // namespace nncfd
