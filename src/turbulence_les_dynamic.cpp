/// @file turbulence_les_dynamic.cpp
/// @brief Dynamic Smagorinsky class methods
///
/// GPU kernels are split across separate files (each with minimal includes)
/// because nvc++ crashes on OpenMP target regions in large translation units:
///   - turbulence_les_dynamic_interp.cpp   — staggered-to-cell-center interpolation
///   - turbulence_les_dynamic_germano.cpp  — Germano identity (box filter + LM/MM accumulation)
///   - turbulence_les_dynamic_apply.cpp    — Cs² computation and nu_t application
///   - turbulence_les_dynamic_gpu.cpp      — GPU buffer allocation/deallocation

#include "turbulence_les.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <iostream>

namespace nncfd {

// GPU kernel free functions (defined in separate files with minimal includes)
void dsmag_pass0_interpolate(const TurbulenceDeviceView* dv,
                              double* ucc, double* vcc, double* wcc, int cc_sz);
void dsmag_pass1_germano(const TurbulenceDeviceView* dv,
                          double* ucc, double* vcc, double* wcc,
                          double* lm_plane, double* mm_plane,
                          int cc_sz, int ny_sz);
void dsmag_pass2_apply(const TurbulenceDeviceView* dv,
                        double* lm_plane, double* mm_plane,
                        double* cs2_plane, int ny_sz);
void dsmag_init_gpu_buffers(double*& ucc, double*& vcc, double*& wcc,
                             double*& lm, double*& mm, double*& cs2,
                             int cell_total, int Ny);
void dsmag_cleanup_gpu_buffers(double*& ucc, double*& vcc, double*& wcc,
                                double*& lm, double*& mm, double*& cs2,
                                int cell_total, int Ny, bool gpu_ready);

DynamicSmagorinskyModel::~DynamicSmagorinskyModel() {
    cleanup_dynamic_gpu();
}

void DynamicSmagorinskyModel::init_dynamic_gpu(const TurbulenceDeviceView* dv) {
    if (dyn_gpu_ready_) return;
    Ny_ = dv->Ny;
    cell_total_ = dv->cell_total;
    dsmag_init_gpu_buffers(u_cc_, v_cc_, w_cc_, LM_plane_, MM_plane_, Cs2_plane_,
                            cell_total_, Ny_);
    dyn_gpu_ready_ = true;
}

void DynamicSmagorinskyModel::cleanup_dynamic_gpu() {
    dsmag_cleanup_gpu_buffers(u_cc_, v_cc_, w_cc_, LM_plane_, MM_plane_, Cs2_plane_,
                               cell_total_, Ny_, dyn_gpu_ready_);
    dyn_gpu_ready_ = false;
}

double DynamicSmagorinskyModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    static bool warned = false;
    if (!warned) {
        std::cerr << "[LES] WARNING: DynamicSmagorinsky CPU fallback active — "
                  << "using static Cs=0.17 instead of Germano procedure.\n";
        warned = true;
    }
    double S11 = g[0], S22 = g[4], S33 = g[8];
    double S12 = 0.5*(g[1]+g[3]), S13 = 0.5*(g[2]+g[6]), S23 = 0.5*(g[5]+g[7]);
    double S_mag = std::sqrt(2.0*(S11*S11+S22*S22+S33*S33+2.0*(S12*S12+S13*S13+S23*S23)));
    return (0.17*delta)*(0.17*delta)*S_mag;
}

void DynamicSmagorinskyModel::update(
    const Mesh& mesh, const VectorField& velocity,
    const ScalarField& k, const ScalarField& omega,
    ScalarField& nu_t, TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    LESModel::update(mesh, velocity, k, omega, nu_t, tau_ij, device_view);
}

void DynamicSmagorinskyModel::update_gpu(const TurbulenceDeviceView* dv) {
    if (!dyn_gpu_ready_) init_dynamic_gpu(dv);
    dsmag_pass0_interpolate(dv, u_cc_, v_cc_, w_cc_, cell_total_);
    dsmag_pass1_germano(dv, u_cc_, v_cc_, w_cc_, LM_plane_, MM_plane_, cell_total_, Ny_);

    // MPI: allreduce plane sums so Cs²(j) uses the global LM/MM
#ifdef USE_MPI
    if (decomp_ && decomp_->is_parallel()) {
        // Sync LM/MM from GPU to host for allreduce
        #pragma omp target update from(LM_plane_[0:Ny_], MM_plane_[0:Ny_])
        decomp_->allreduce_sum(LM_plane_, Ny_);
        decomp_->allreduce_sum(MM_plane_, Ny_);
        #pragma omp target update to(LM_plane_[0:Ny_], MM_plane_[0:Ny_])
    }
#endif

    dsmag_pass2_apply(dv, LM_plane_, MM_plane_, Cs2_plane_, Ny_);
}

} // namespace nncfd
