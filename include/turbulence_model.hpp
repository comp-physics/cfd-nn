#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "config.hpp"
#include "gpu_utils.hpp"
#include <memory>
#include <string>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// Device view for turbulence models: pointers to solver-owned GPU-resident data
/// This struct is passed to turbulence model update() calls when GPU is enabled
/// to avoid repeated map(present:) clauses and pointer aliasing issues.
///
/// MODEL 1 CONTRACT: All pointers are HOST pointers that have been persistently
/// mapped to GPU via `target enter data`. Kernels use `map(present: ...)` to
/// access device copies. No device-to-host transfers occur during compute.
struct TurbulenceDeviceView {
    // Velocity field (staggered MAC grid, solver-owned, persistent on GPU)
    double* u_face = nullptr;           // u at x-faces: (Ny+2Ng) × (Nx+2Ng+1)
    double* v_face = nullptr;           // v at y-faces: (Ny+2Ng+1) × (Nx+2Ng)
    int u_stride = 0;                   // u row stride
    int v_stride = 0;                   // v row stride
    
    // Turbulence fields (cell-centered, solver-owned, persistent on GPU)
    double* k = nullptr;                // TKE (if transport model)
    double* omega = nullptr;            // Specific dissipation (if transport model)
    double* nu_t = nullptr;             // Eddy viscosity output
    int cell_stride = 0;                // Cell-centered row stride
    
    // Reynolds stress tensor components (cell-centered, for EARSM/TBNN)
    double* tau_xx = nullptr;
    double* tau_xy = nullptr;
    double* tau_yy = nullptr;
    
    // Scratch buffers for gradients (solver-owned, persistent on GPU)
    double* dudx = nullptr;             // Cell-centered gradients
    double* dudy = nullptr;
    double* dvdx = nullptr;
    double* dvdy = nullptr;
    
    // Wall distance (cell-centered, precomputed, persistent on GPU)
    double* wall_distance = nullptr;
    
    // Mesh parameters (scalars, passed by value to GPU kernels)
    int Nx = 0;                         // Interior cells in x
    int Ny = 0;                         // Interior cells in y
    int Ng = 0;                         // Ghost cells
    double dx = 0.0;                    // Grid spacing
    double dy = 0.0;
    double delta = 0.0;                 // Reference length scale
    
    // Check if view is valid (all essential pointers non-null)
    bool is_valid() const {
        return (u_face != nullptr && v_face != nullptr && nu_t != nullptr &&
                dudx != nullptr && dudy != nullptr && dvdx != nullptr && dvdy != nullptr &&
                Nx > 0 && Ny > 0);
    }
};

/// Device view for core solver: pointers to GPU-resident solver arrays
/// Parallel to TurbulenceDeviceView but for projection/NS step
///
/// MODEL 1 CONTRACT: All pointers are HOST pointers that have been persistently
/// mapped to GPU via `target enter data`. Kernels use `map(present: ...)` to
/// access device copies. No device-to-host transfers occur during compute.
struct SolverDeviceView {
    // Velocity fields (staggered) - 2D/3D
    double* u_face = nullptr;
    double* v_face = nullptr;
    double* w_face = nullptr;  // 3D
    double* u_star_face = nullptr;
    double* v_star_face = nullptr;
    double* w_star_face = nullptr;  // 3D
    double* u_old_face = nullptr;
    double* v_old_face = nullptr;
    double* w_old_face = nullptr;  // 3D
    int u_stride = 0;
    int v_stride = 0;
    int w_stride = 0;  // 3D
    int u_plane_stride = 0;  // 3D
    int v_plane_stride = 0;  // 3D
    int w_plane_stride = 0;  // 3D

    // Scalar fields (cell-centered)
    double* p = nullptr;
    double* p_corr = nullptr;
    double* nu_t = nullptr;
    double* nu_eff = nullptr;
    double* rhs = nullptr;
    double* div = nullptr;
    int cell_stride = 0;
    int cell_plane_stride = 0;  // 3D

    // Work arrays
    double* conv_u = nullptr;
    double* conv_v = nullptr;
    double* conv_w = nullptr;  // 3D
    double* diff_u = nullptr;
    double* diff_v = nullptr;
    double* diff_w = nullptr;  // 3D

    // Mesh parameters
    int Nx = 0, Ny = 0, Nz = 1, Ng = 0;
    double dx = 0.0, dy = 0.0, dz = 1.0, dt = 0.0;

    bool is_valid() const {
        return (u_face && v_face && p && nu_eff && Nx > 0 && Ny > 0);
    }

    bool is3D() const { return Nz > 1; }
};

/// Abstract base class for turbulence closures
class TurbulenceModel {
public:
    virtual ~TurbulenceModel() = default;
    
    /// Update turbulent quantities given current mean flow
    /// @param mesh       Computational mesh
    /// @param velocity   Mean velocity field
    /// @param k          Turbulent kinetic energy (optional, may be unused)
    /// @param omega      Specific dissipation rate (optional, may be unused)
    /// @param nu_t       [out] Eddy viscosity field
    /// @param tau_ij     [out] Reynolds stress tensor (optional, for TBNN)
    /// @param device_view [optional] Device view for GPU-resident data (nullptr = use CPU path)
    virtual void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) = 0;
    
    /// Get model name
    virtual std::string name() const = 0;
    
    /// Check if model provides explicit Reynolds stresses (vs. eddy viscosity only)
    virtual bool provides_reynolds_stresses() const { return false; }
    
    /// Does this model solve transport equations (e.g., k-ω, k-ε)?
    /// If true, advance_turbulence() should be called each time step.
    virtual bool uses_transport_equations() const { return false; }
    
    /// Advance turbulence transport variables (k, ω, etc.) by one time step.
    /// Default: no-op (for algebraic / NN models).
    /// @param mesh       Computational mesh
    /// @param velocity   Mean velocity field
    /// @param dt         Time step size
    /// @param k          [in/out] Turbulent kinetic energy
    /// @param omega      [in/out] Specific dissipation rate
    /// @param nu_t_prev  Eddy viscosity from previous step (for diffusion coefficients)
    /// @param device_view [optional] Device view for GPU-resident data (nullptr = use CPU path)
    virtual void advance_turbulence(
        const Mesh& mesh,
        const VectorField& velocity,
        double dt,
        ScalarField& k,
        ScalarField& omega,
        const ScalarField& nu_t_prev,
        const TurbulenceDeviceView* device_view = nullptr
    ) {
        (void)mesh;
        (void)velocity;
        (void)dt;
        (void)k;
        (void)omega;
        (void)nu_t_prev;
        (void)device_view;
    }
    
    /// Initialize any model-specific fields (e.g., k, omega for transport models)
    virtual void initialize(const Mesh& /*mesh*/, const VectorField& /*velocity*/) {}
    
    /// GPU buffer management interface (optional - base implementation does nothing)
    /// Derived classes should override these if they support GPU offloading
    virtual void initialize_gpu_buffers(const Mesh& mesh) { (void)mesh; }
    virtual void cleanup_gpu_buffers() {}
    virtual bool is_gpu_ready() const { return false; }
    
    /// Set laminar viscosity (needed for some models)
    void set_nu(double nu) { nu_ = nu; }
    double nu() const { return nu_; }
    
    /// Set reference length scale (channel half-height, etc.)
    void set_delta(double delta) { delta_ = delta; }
    double delta() const { return delta_; }
    
protected:
    double nu_ = 0.001;    ///< Laminar viscosity
    double delta_ = 1.0;   ///< Reference length scale
    
    /// Helper to check GPU availability (safe to call from any derived class)
    static bool gpu_available() {
        return gpu::is_gpu_available();
    }
};

/// Factory function to create turbulence model from config
std::unique_ptr<TurbulenceModel> create_turbulence_model(
    TurbulenceModelType type,
    const std::string& weights_path = "",
    const std::string& scaling_path = ""
);

} // namespace nncfd

