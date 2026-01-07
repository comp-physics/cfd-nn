/// @file test_fixtures.hpp
/// @brief Common test fixtures: manufactured solutions for Poisson solver validation

#pragma once

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nncfd {
namespace test {

//=============================================================================
// Boundary Condition Types for Manufactured Solutions
//=============================================================================

/// Boundary condition type for manufactured solutions
enum class BCType {
    Periodic,   ///< Periodic BC: k = 2*pi/L, uses sin
    Neumann,    ///< Neumann BC (zero gradient): k = pi/L, uses cos
    Dirichlet   ///< Dirichlet BC (zero value): k = pi/L, uses sin
};

//=============================================================================
// 3D Manufactured Solution Template
//=============================================================================

/// Template for 3D manufactured solutions with arbitrary boundary conditions
/// Wave numbers are computed based on BC types:
///   - Periodic: k = 2*pi/L (full wave fits in domain)
///   - Neumann:  k = pi/L (cos function, zero derivative at boundaries)
///   - Dirichlet: k = pi/L (sin function, zero value at boundaries)
template<BCType BCx, BCType BCy, BCType BCz>
struct ManufacturedSolution3D {
    double Lx, Ly, Lz;
    double kx, ky, kz;
    double lap_coeff;

    ManufacturedSolution3D(double lx, double ly, double lz)
        : Lx(lx), Ly(ly), Lz(lz) {
        kx = (BCx == BCType::Periodic) ? (2.0 * M_PI / Lx) : (M_PI / Lx);
        ky = (BCy == BCType::Periodic) ? (2.0 * M_PI / Ly) : (M_PI / Ly);
        kz = (BCz == BCType::Periodic) ? (2.0 * M_PI / Lz) : (M_PI / Lz);
        lap_coeff = -(kx*kx + ky*ky + kz*kz);
    }

    /// Exact solution p(x,y,z)
    double p(double x, double y, double z) const {
        double fx = (BCx == BCType::Neumann) ? std::cos(kx * x) : std::sin(kx * x);
        double fy = (BCy == BCType::Neumann) ? std::cos(ky * y) : std::sin(ky * y);
        double fz = (BCz == BCType::Neumann) ? std::cos(kz * z) : std::sin(kz * z);
        return fx * fy * fz;
    }

    /// Right-hand side: rhs = Laplacian(p) = lap_coeff * p
    double rhs(double x, double y, double z) const {
        return lap_coeff * p(x, y, z);
    }

    /// Alias for exact solution
    double exact(double x, double y, double z) const {
        return p(x, y, z);
    }
};

//=============================================================================
// 2D Manufactured Solution Template
//=============================================================================

/// Template for 2D manufactured solutions
template<BCType BCx, BCType BCy>
struct ManufacturedSolution2D {
    double Lx, Ly;
    double kx, ky;
    double lap_coeff;

    ManufacturedSolution2D(double lx, double ly)
        : Lx(lx), Ly(ly) {
        kx = (BCx == BCType::Periodic) ? (2.0 * M_PI / Lx) : (M_PI / Lx);
        ky = (BCy == BCType::Periodic) ? (2.0 * M_PI / Ly) : (M_PI / Ly);
        lap_coeff = -(kx*kx + ky*ky);
    }

    double p(double x, double y) const {
        double fx = (BCx == BCType::Neumann) ? std::cos(kx * x) : std::sin(kx * x);
        double fy = (BCy == BCType::Neumann) ? std::cos(ky * y) : std::sin(ky * y);
        return fx * fy;
    }

    double rhs(double x, double y) const {
        return lap_coeff * p(x, y);
    }
};

//=============================================================================
// Common Solution Type Aliases
//=============================================================================

// 3D Solutions
using ChannelSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Neumann, BCType::Periodic>;
using DuctSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Neumann, BCType::Neumann>;
using PeriodicSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Periodic, BCType::Periodic>;
using DirichletSolution3D = ManufacturedSolution3D<BCType::Dirichlet, BCType::Dirichlet, BCType::Dirichlet>;
using MixedBCSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Dirichlet, BCType::Neumann>;

// 2D Solutions
using ChannelSolution2D = ManufacturedSolution2D<BCType::Periodic, BCType::Neumann>;
using DirichletSolution2D = ManufacturedSolution2D<BCType::Dirichlet, BCType::Dirichlet>;
using PeriodicSolution2D = ManufacturedSolution2D<BCType::Periodic, BCType::Periodic>;

// Legacy aliases
using ChannelSolution = ChannelSolution3D;
using DuctSolution = DuctSolution3D;
using PeriodicSolution = PeriodicSolution3D;
using Channel2DSolution = ChannelSolution2D;

} // namespace test
} // namespace nncfd
