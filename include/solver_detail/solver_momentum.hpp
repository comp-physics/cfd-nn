/// @file solver_detail/solver_momentum.hpp
/// @brief Momentum equation discretization for RANSSolver
///
/// This header declares the momentum-related methods that would be
/// extracted from solver.cpp in a full refactor. The methods handle:
/// - Convective term computation (Adams-Bashforth)
/// - Diffusive term computation (with effective viscosity)
/// - Body forces
///
/// Currently these are inline implementations within solver.cpp.
/// A full refactor would move them to solver_momentum.cpp.

#pragma once

namespace nncfd {

class RANSSolver;  // Forward declaration

namespace solver_detail {

/// Compute convective term using Adams-Bashforth extrapolation
/// conv = 1.5 * H(u^n) - 0.5 * H(u^{n-1})
/// where H(u) = (u·∇)u
///
/// Implemented in solver.cpp:RANSSolver::compute_convective_term()
void compute_convective_term_impl(
    const class Mesh& mesh,
    const class VectorField& vel,
    class VectorField& conv,
    bool is_3d
);

/// Compute diffusive term: ∇·(ν_eff ∇u)
/// Uses central differencing with variable viscosity
///
/// Implemented in solver.cpp:RANSSolver::compute_diffusive_term()
void compute_diffusive_term_impl(
    const class Mesh& mesh,
    const class VectorField& vel,
    const class ScalarField& nu_eff,
    class VectorField& diff,
    double nu_molecular,
    bool is_3d
);

/// Apply body forces to RHS
void apply_body_force(
    class VectorField& rhs,
    double fx, double fy, double fz,
    const class Mesh& mesh
);

} // namespace solver_detail
} // namespace nncfd
