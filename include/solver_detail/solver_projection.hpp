/// @file solver_detail/solver_projection.hpp
/// @brief Pressure projection methods for RANSSolver
///
/// This header declares the projection-related methods that would be
/// extracted from solver.cpp in a full refactor. The methods handle:
/// - Divergence computation
/// - Poisson solve (pressure correction)
/// - Velocity correction
///
/// Currently these are inline implementations within solver.cpp.
/// A full refactor would move them to solver_projection.cpp.

#pragma once

namespace nncfd {

class RANSSolver;  // Forward declaration

namespace solver_detail {

/// Compute divergence of velocity field
/// div = ∂u/∂x + ∂v/∂y + ∂w/∂z
///
/// Implemented in solver.cpp:RANSSolver::compute_divergence()
void compute_divergence_impl(
    const class Mesh& mesh,
    const class VectorField& vel,
    class ScalarField& div,
    bool is_3d
);

/// Compute pressure gradient
/// ∂p/∂x, ∂p/∂y, ∂p/∂z
///
/// Implemented in solver.cpp:RANSSolver::compute_pressure_gradient()
void compute_pressure_gradient_impl(
    const class Mesh& mesh,
    const class ScalarField& p,
    class ScalarField& dp_dx,
    class ScalarField& dp_dy,
    class ScalarField& dp_dz,
    bool is_3d
);

/// Correct velocity to satisfy continuity
/// u* = u** - dt * ∇p
///
/// Implemented in solver.cpp:RANSSolver::correct_velocity()
void correct_velocity_impl(
    const class Mesh& mesh,
    class VectorField& vel,
    const class ScalarField& dp_dx,
    const class ScalarField& dp_dy,
    const class ScalarField& dp_dz,
    double dt,
    bool is_3d
);

} // namespace solver_detail
} // namespace nncfd
