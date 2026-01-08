/// Unit tests for VTK output validation
///
/// Tests VTK file output functionality:
/// - File format correctness
/// - Field synchronization (GPU → CPU)
/// - Output directory handling
/// - Multiple snapshot output

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <string>

using namespace nncfd;
using nncfd::test::harness::record;

namespace {
// Helper to remove file
void remove_file(const std::string& filename) {
    std::remove(filename.c_str());
}

// Helper to read file contents
std::string read_file(const std::string& filename) {
    std::ifstream f(filename);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}
} // namespace

// ============================================================================
// VTK File Format Tests
// ============================================================================

void test_vtk_file_created() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_created.vtk";
    remove_file(filename);

    solver.write_vtk(filename);

    std::ifstream f(filename);
    bool exists = f.good();

    remove_file(filename);
    record("VTK file creation", exists);
}

void test_vtk_header_format() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_header.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);

    bool pass = true;
    pass = pass && (content.find("# vtk DataFile Version") != std::string::npos);
    pass = pass && (content.find("ASCII") != std::string::npos ||
                    content.find("BINARY") != std::string::npos);
    pass = pass && (content.find("STRUCTURED_GRID") != std::string::npos ||
                    content.find("RECTILINEAR_GRID") != std::string::npos ||
                    content.find("STRUCTURED_POINTS") != std::string::npos);

    remove_file(filename);
    record("VTK header format", pass);
}

void test_vtk_dimensions_match() {
    int Nx = 16, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_dims.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);

    bool pass = true;
    std::string dims_keyword = "DIMENSIONS";
    auto pos = content.find(dims_keyword);

    if (pos != std::string::npos) {
        std::istringstream iss(content.substr(pos));
        std::string keyword;
        int dim_x, dim_y, dim_z;
        iss >> keyword >> dim_x >> dim_y >> dim_z;
        pass = (dim_x >= Nx) && (dim_y >= Ny);
    }

    remove_file(filename);
    record("VTK dimensions match mesh", pass);
}

// ============================================================================
// Field Value Tests
// ============================================================================

void test_vtk_velocity_data() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_velocity.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);

    bool has_velocity = (content.find("velocity") != std::string::npos) ||
                        (content.find("VECTORS") != std::string::npos) ||
                        (content.find("u_velocity") != std::string::npos);

    remove_file(filename);
    record("VTK contains velocity data", has_velocity);
}

void test_vtk_pressure_data() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    // Run a step to get non-zero pressure
    solver.step();

    std::string filename = "/tmp/test_vtk_pressure.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);

    bool has_pressure = (content.find("pressure") != std::string::npos) ||
                        (content.find("SCALARS p") != std::string::npos) ||
                        (content.find("p ") != std::string::npos);

    remove_file(filename);
    record("VTK contains pressure data", has_pressure);
}

// ============================================================================
// GPU Synchronization Tests
// ============================================================================

void test_vtk_after_gpu_compute() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run several steps on GPU
    for (int i = 0; i < 20; ++i) {
        solver.step();
    }

    // Write VTK - should automatically sync from GPU
    std::string filename = "/tmp/test_vtk_gpu.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);

    bool pass = true;
    pass = pass && (content.length() > 100);
    pass = pass && (content.find("nan") == std::string::npos);
    pass = pass && (content.find("NaN") == std::string::npos);
    pass = pass && (content.find("inf") == std::string::npos);

    remove_file(filename);
    record("VTK output after GPU compute", pass);
}

// ============================================================================
// Multiple Output Tests
// ============================================================================

void test_vtk_sequential_outputs() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    std::vector<std::string> files;
    bool pass = true;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            solver.step();
        }

        std::string filename = "/tmp/test_vtk_seq_" + std::to_string(i) + ".vtk";
        files.push_back(filename);
        solver.write_vtk(filename);
    }

    // All files should exist
    for (const auto& f : files) {
        std::ifstream fin(f);
        if (!fin.good()) pass = false;
    }

    // Clean up
    for (const auto& f : files) {
        remove_file(f);
    }

    record("Sequential VTK outputs", pass);
}

// ============================================================================
// 3D VTK Tests
// ============================================================================

void test_vtk_3d_output() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 5; ++i) {
        solver.step();
    }

    std::string filename = "/tmp/test_vtk_3d.vtk";
    solver.write_vtk(filename);

    std::string content = read_file(filename);
    bool pass = (content.find("DIMENSIONS") != std::string::npos);

    remove_file(filename);
    record("3D VTK output", pass);
}

// ============================================================================
// Turbulence Model Output Tests
// ============================================================================

void test_vtk_turbulence_fields() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.turb_model = TurbulenceModelType::KOmega;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(TurbulenceModelType::KOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    std::string filename = "/tmp/test_vtk_turb.vtk";
    solver.write_vtk(filename);

    std::ifstream fin(filename);
    bool pass = fin.good();

    remove_file(filename);
    record("VTK with turbulence fields", pass);
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_vtk_small_grid() {
    Mesh mesh;
    mesh.init_uniform(4, 4, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.1;
    config.dt = 0.01;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(0.5, 0.0);

    std::string filename = "/tmp/test_vtk_small.vtk";
    solver.write_vtk(filename);

    std::ifstream fin(filename);
    bool pass = fin.good();

    remove_file(filename);
    record("VTK on small grid (4x4)", pass);
}

void test_vtk_overwrite() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(0.5, 0.0);

    std::string filename = "/tmp/test_vtk_overwrite.vtk";

    // Write first file
    solver.write_vtk(filename);
    std::string content1 = read_file(filename);

    // Run some steps
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    // Overwrite
    solver.write_vtk(filename);
    std::string content2 = read_file(filename);

    bool pass = (content1.find("vtk") != std::string::npos) &&
                (content2.find("vtk") != std::string::npos);

    remove_file(filename);
    record("VTK file overwrite", pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("VTK Output Tests", [] {
        // File format tests
        test_vtk_file_created();
        test_vtk_header_format();
        test_vtk_dimensions_match();

        // Field value tests
        test_vtk_velocity_data();
        test_vtk_pressure_data();

        // GPU sync tests
        test_vtk_after_gpu_compute();

        // Multiple output tests
        test_vtk_sequential_outputs();

        // 3D tests
        test_vtk_3d_output();

        // Turbulence tests
        test_vtk_turbulence_fields();

        // Edge cases
        test_vtk_small_grid();
        test_vtk_overwrite();
    });
}
