/// @file test_fixtures.hpp
/// @brief Common test fixtures: manufactured solutions, mesh/config factories
///
/// This header consolidates duplicated manufactured solution structs from:
///   - test_poisson_manufactured.cpp (ChannelSolution, DuctSolution, etc.)
///   - test_poisson_fft_manufactured.cpp (ChannelManufactured, DuctManufactured)
///   - test_poisson_dirichlet_mixed.cpp (DirichletSolution3D, MixedBCSolution3D)
///   - test_fft1d_validation.cpp (ManufacturedSolution)

#pragma once

#include "mesh.hpp"
#include "config.hpp"
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
        // Compute wave numbers based on BC type
        kx = (BCx == BCType::Periodic) ? (2.0 * M_PI / Lx) : (M_PI / Lx);
        ky = (BCy == BCType::Periodic) ? (2.0 * M_PI / Ly) : (M_PI / Ly);
        kz = (BCz == BCType::Periodic) ? (2.0 * M_PI / Lz) : (M_PI / Lz);
        lap_coeff = -(kx*kx + ky*ky + kz*kz);
    }

    /// Exact solution p(x,y,z)
    /// Uses sin for Periodic/Dirichlet, cos for Neumann
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

    /// Alias for exact solution (some tests use this name)
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
/// Channel flow: periodic X/Z, Neumann Y (walls)
using ChannelSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Neumann, BCType::Periodic>;

/// Duct flow: periodic X, Neumann Y/Z (FFT1D compatible)
using DuctSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Neumann, BCType::Neumann>;

/// Fully periodic (Taylor-Green like)
using PeriodicSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Periodic, BCType::Periodic>;

/// Pure Dirichlet (homogeneous at all boundaries)
using DirichletSolution3D = ManufacturedSolution3D<BCType::Dirichlet, BCType::Dirichlet, BCType::Dirichlet>;

/// Mixed: periodic X, Dirichlet Y, Neumann Z
using MixedBCSolution3D = ManufacturedSolution3D<BCType::Periodic, BCType::Dirichlet, BCType::Neumann>;

// 2D Solutions
/// 2D Channel: periodic X, Neumann Y
using ChannelSolution2D = ManufacturedSolution2D<BCType::Periodic, BCType::Neumann>;

/// 2D Dirichlet: homogeneous at all boundaries
using DirichletSolution2D = ManufacturedSolution2D<BCType::Dirichlet, BCType::Dirichlet>;

/// 2D Periodic: periodic in both directions
using PeriodicSolution2D = ManufacturedSolution2D<BCType::Periodic, BCType::Periodic>;

// Legacy aliases (for backward compatibility with existing tests)
using ChannelSolution = ChannelSolution3D;
using DuctSolution = DuctSolution3D;
using PeriodicSolution = PeriodicSolution3D;
using Channel2DSolution = ChannelSolution2D;

//=============================================================================
// Mesh Factory Functions
//=============================================================================

/// Create a 2D uniform mesh
inline Mesh create_uniform_mesh_2d(int nx, int ny, double Lx, double Ly,
                                   double x0 = 0.0, double y0 = 0.0) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, x0, x0 + Lx, y0, y0 + Ly, 1);
    return mesh;
}

/// Create a 3D uniform mesh
inline Mesh create_uniform_mesh_3d(int nx, int ny, int nz,
                                   double Lx, double Ly, double Lz,
                                   double x0 = 0.0, double y0 = 0.0, double z0 = 0.0) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, nz, x0, x0 + Lx, y0, y0 + Ly, z0, z0 + Lz);
    return mesh;
}

/// Create a standard channel mesh (periodic X, walls at Y=0,Ly)
inline Mesh create_channel_mesh(int nx = 16, int ny = 32, double Lx = 4.0, double H = 1.0) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, 0.0, Lx, -H, H, 1);
    return mesh;
}

/// Create a 3D channel mesh
inline Mesh create_channel_mesh_3d(int nx = 16, int ny = 32, int nz = 8,
                                   double Lx = 4.0, double H = 1.0, double Lz = 2.0) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, nz, 0.0, Lx, -H, H, 0.0, Lz);
    return mesh;
}

/// Create a Taylor-Green mesh (cubic, periodic)
inline Mesh create_taylor_green_mesh(int n = 32) {
    return create_uniform_mesh_3d(n, n, n, 2.0*M_PI, 2.0*M_PI, 2.0*M_PI);
}

/// Create a 2D Taylor-Green mesh
inline Mesh create_taylor_green_mesh_2d(int n = 32) {
    return create_uniform_mesh_2d(n, n, 2.0*M_PI, 2.0*M_PI);
}

//=============================================================================
// Config Factory Functions
//=============================================================================

/// Create a basic unsteady flow config
inline Config create_unsteady_config(double nu = 0.01, double dt = 0.01) {
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    return config;
}

/// Create a channel flow config with pressure gradient
inline Config create_channel_config(double nu = 0.01, double dp_dx = -1.0) {
    Config config = create_unsteady_config(nu);
    config.dp_dx = dp_dx;
    return config;
}

/// Create a validation config with conservative settings
inline Config create_validation_config(double nu = 0.01, int max_iter = 100) {
    Config config = create_unsteady_config(nu, 0.01);
    config.max_iter = max_iter;
    config.tol = 1e-10;
    return config;
}

/// Create a Poisson solver config
inline PoissonConfig create_poisson_config(double tol = 1e-6, int max_iter = 50) {
    PoissonConfig cfg;
    cfg.tol = tol;
    cfg.max_iter = max_iter;
    return cfg;
}

} // namespace test
} // namespace nncfd
