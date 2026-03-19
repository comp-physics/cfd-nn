#pragma once

#include "mesh.hpp"
#include "fields.hpp"

namespace nncfd {

/// Boundary condition type for Poisson equation
enum class PoissonBC {
    Dirichlet,
    Neumann,
    Periodic
};

/// Configuration for Poisson solver
struct PoissonConfig {
    double tol = 1e-6;       ///< Legacy absolute tolerance (deprecated, use tol_abs)
    int max_vcycles = 10000; ///< Maximum V-cycles per Poisson solve.
    double omega = 1.5;      ///< Relaxation parameter (legacy, unused by MG/FFT solvers)
    bool verbose = false;    ///< Print convergence info

    // Robust convergence criteria for multigrid (recommended for projection)
    // Converged when: ||r||_∞ ≤ tol_abs  OR  ||r||/||b|| ≤ tol_rhs  OR  ||r||/||r0|| ≤ tol_rel
    double tol_abs = 0.0;    ///< Absolute tolerance on ||r||_∞ (0 = disabled)
    double tol_rhs = 1e-3;   ///< RHS-relative tolerance: ||r||/||b|| (recommended for projection)
    double tol_rel = 1e-3;   ///< Initial-residual relative: ||r||/||r0|| (backup criterion)
    int check_interval = 3;  ///< Check convergence every N V-cycles (reduces GPU→CPU sync frequency)
    bool use_l2_norm = true; ///< Use L2 norm for convergence (smoother than L∞, less sensitive to hot cells)
    double linf_safety_factor = 10.0; ///< L∞ safety cap: even with L2 convergence, require ||r||_∞/||b||_∞ ≤ tol_rhs * factor

    // Fixed-cycle mode: run exactly N V-cycles without convergence checks (fastest for projection)
    // When > 0, skips all residual computation and D→H transfers during solve.
    // Optimal: 8 cycles with nu1=2, nu2=1 gives 16% faster + 58% better divergence vs baseline.
    int fixed_cycles = 0;    ///< Fixed V-cycle count (0 = use convergence-based termination)

    // Adaptive fixed-cycle mode: run bulk cycles graphed, then check, add more if needed
    // Pattern: run check_after cycles, check residual, add 2 more cycles if needed, cap at fixed_cycles
    // Enable by setting both fixed_cycles > 0 and adaptive_cycles = true
    bool adaptive_cycles = false;  ///< Enable adaptive checking within fixed-cycle mode
    int check_after = 4;           ///< Check residual after this many cycles (default: 4)

    // MG smoother tuning parameters
    // Optimal at 128³ channel: nu1=3, nu2=1 (more pre-smooth for wall BCs)
    // Benchmark: nu1=3,nu2=1,cyc=8 is 13% faster AND 10× lower div_L2 than baseline
    int nu1 = 0;             ///< Pre-smoothing sweeps (0 = auto: 3 for wall BCs)
    int nu2 = 0;             ///< Post-smoothing sweeps (0 = auto: 1)
    int chebyshev_degree = 4; ///< Chebyshev polynomial degree (3-4 typical, lower = faster)

    // CUDA Graph acceleration (GPU only)
    // Captures entire V-cycle as single graph - massive speedup from reduced kernel launches
    // Environment variable MG_USE_VCYCLE_GRAPH=0 can override to disable
    bool use_vcycle_graph = true;  ///< Enable V-cycle CUDA Graph (default: ON)
};

} // namespace nncfd
