/// @file solver_detail/solver_io.hpp
/// @brief I/O and diagnostics for RANSSolver
///
/// This header declares the I/O-related methods that would be
/// extracted from solver.cpp in a full refactor. The methods handle:
/// - VTK output
/// - Field writing (raw binary)
/// - Statistics printing
/// - Debugging output
///
/// Currently these are inline implementations within solver.cpp.
/// A full refactor would move them to solver_io.cpp.

#pragma once

#include <string>
#include <ostream>

namespace nncfd {

class RANSSolver;  // Forward declaration
class Mesh;
class VectorField;
class ScalarField;

namespace solver_io {

/// Write velocity profile at given x location
/// Outputs u(y), v(y) to stdout in formatted columns
void print_velocity_profile(
    const Mesh& mesh,
    const VectorField& vel,
    double x_loc,
    std::ostream& out
);

/// Write all fields to binary files
/// Creates {prefix}_u.bin, {prefix}_v.bin, {prefix}_p.bin, etc.
void write_fields_binary(
    const std::string& prefix,
    const Mesh& mesh,
    const VectorField& vel,
    const ScalarField& p,
    const ScalarField* nu_t = nullptr
);

/// Write VTK file for visualization
/// Format: VTK Legacy ASCII (.vtk)
void write_vtk_legacy(
    const std::string& filename,
    const Mesh& mesh,
    const VectorField& vel,
    const ScalarField& p,
    const ScalarField* nu_t = nullptr,
    const ScalarField* k = nullptr,
    const ScalarField* omega = nullptr
);

/// Write VTK XML file for parallel visualization
/// Format: VTK XML ImageData (.vti)
void write_vti(
    const std::string& filename,
    const Mesh& mesh,
    const VectorField& vel,
    const ScalarField& p
);

/// Print solver configuration summary
void print_solver_info(
    const Mesh& mesh,
    const class Config& config,
    const std::string& turbulence_model_name,
    std::ostream& out
);

/// Check all fields for NaN/Inf and throw if found
/// @param step Current timestep (for error message)
/// @throws std::runtime_error if NaN/Inf detected
void check_for_nan_inf(
    const Mesh& mesh,
    const VectorField& vel,
    const ScalarField& p,
    int step
);

/// Compute statistics for diagnostics
struct FlowStatistics {
    double u_max, v_max, w_max;
    double u_mean, v_mean;
    double p_max, p_min;
    double kinetic_energy;
    double enstrophy;
    double max_divergence;
};

FlowStatistics compute_statistics(
    const Mesh& mesh,
    const VectorField& vel,
    const ScalarField& p
);

} // namespace solver_io
} // namespace nncfd
