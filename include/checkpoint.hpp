#pragma once

/// @file checkpoint.hpp
/// @brief HDF5 checkpoint/restart for simulation state
///
/// Saves and loads complete simulation state (velocity, pressure, step count,
/// time) to HDF5 files for restart capability. Requires HDF5 library.

#include "mesh.hpp"
#include "fields.hpp"
#include <string>

namespace nncfd {

/// Write simulation state to HDF5 checkpoint file
/// @param filename  Output HDF5 file path
/// @param mesh      Computational mesh
/// @param vel       Velocity field
/// @param pressure  Pressure field
/// @param step      Current time step number
/// @param time      Current simulation time
/// @param dt        Current time step size
void write_checkpoint(const std::string& filename,
                      const Mesh& mesh,
                      const VectorField& vel,
                      const ScalarField& pressure,
                      int step, double time, double dt);

/// Read simulation state from HDF5 checkpoint file
/// @param filename  Input HDF5 file path
/// @param mesh      Computational mesh (must match saved mesh dimensions)
/// @param vel       [out] Velocity field
/// @param pressure  [out] Pressure field
/// @param step      [out] Time step number
/// @param time      [out] Simulation time
/// @param dt        [out] Time step size
/// @return true if checkpoint was loaded successfully
bool read_checkpoint(const std::string& filename,
                     const Mesh& mesh,
                     VectorField& vel,
                     ScalarField& pressure,
                     int& step, double& time, double& dt);

} // namespace nncfd
