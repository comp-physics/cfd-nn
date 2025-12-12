#pragma once

#include "turbulence_model.hpp"
#include "features.hpp"

namespace nncfd {

/// Simple algebraic eddy viscosity model (mixing length)
/// nu_t = (kappa * y)^2 * |S| with van Driest damping
class MixingLengthModel : public TurbulenceModel {
public:
    MixingLengthModel();
    ~MixingLengthModel();
    
    // Delete copy/move to prevent double-free of GPU buffers
    MixingLengthModel(const MixingLengthModel&) = delete;
    MixingLengthModel& operator=(const MixingLengthModel&) = delete;
    MixingLengthModel(MixingLengthModel&&) = delete;
    MixingLengthModel& operator=(MixingLengthModel&&) = delete;
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;
    
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return gpu_ready_; }
    
    std::string name() const override { return "MixingLength"; }
    
    /// Set model constants
    void set_kappa(double kappa) { kappa_ = kappa; }
    void set_A_plus(double A_plus) { A_plus_ = A_plus; }
    void set_delta(double delta) { delta_ = delta; }
    
private:
    double kappa_ = 0.41;   ///< von Karman constant
    double A_plus_ = 26.0;  ///< van Driest damping constant
    double delta_ = 1.0;    ///< Channel half-height
    
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    
    // GPU management (members ALWAYS present for ABI stability)
    bool gpu_ready_ = false;
    [[maybe_unused]] bool buffers_on_gpu_ = false;
    [[maybe_unused]] int cached_total_cells_ = 0;
    
    // Flat arrays for GPU
    std::vector<double> nu_t_gpu_flat_;
    std::vector<double> y_wall_flat_;
    
#ifdef USE_GPU_OFFLOAD
    void allocate_gpu_arrays(const Mesh& mesh);
    void free_gpu_arrays();
#else
    void allocate_gpu_arrays(const Mesh& mesh) { (void)mesh; }
    void free_gpu_arrays() {}
#endif
};

/// Simple k-omega model (without transport equations for now)
/// Uses algebraic relations to estimate k and omega, then computes nu_t
class AlgebraicKOmegaModel : public TurbulenceModel {
public:
    AlgebraicKOmegaModel();
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;
    
    std::string name() const override { return "AlgebraicKOmega"; }
    
    void set_delta(double delta) { delta_ = delta; }
    void set_C_mu(double C_mu) { C_mu_ = C_mu; }
    
private:
    double delta_ = 1.0;
    double C_mu_ = 0.09;
    
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
};

} // namespace nncfd


