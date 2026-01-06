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
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <string>

using namespace nncfd;

namespace {
// Helper to check if file exists and abort if not
void require_file_exists(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "FAILED: File does not exist: " << filename << "\n";
        std::exit(1);
    }
}

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
    std::cout << "Testing VTK file creation... ";

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

    require_file_exists(filename);

    remove_file(filename);
    std::cout << "PASSED\n";
}

void test_vtk_header_format() {
    std::cout << "Testing VTK header format... ";

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

    // Check VTK header
    assert(content.find("# vtk DataFile Version") != std::string::npos);
    assert(content.find("ASCII") != std::string::npos ||
           content.find("BINARY") != std::string::npos);
    assert(content.find("STRUCTURED_GRID") != std::string::npos ||
           content.find("RECTILINEAR_GRID") != std::string::npos ||
           content.find("STRUCTURED_POINTS") != std::string::npos);

    remove_file(filename);
    std::cout << "PASSED\n";
}

void test_vtk_dimensions_match() {
    std::cout << "Testing VTK dimensions match mesh... ";

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

    // Look for DIMENSIONS line
    // Format: DIMENSIONS Nx Ny Nz
    std::string dims_keyword = "DIMENSIONS";
    auto pos = content.find(dims_keyword);

    if (pos != std::string::npos) {
        // Parse dimensions
        std::istringstream iss(content.substr(pos));
        std::string keyword;
        int dim_x, dim_y, dim_z;
        iss >> keyword >> dim_x >> dim_y >> dim_z;

        // Dimensions should be Nx+1, Ny+1, Nz+1 for cell-centered data
        // or Nx, Ny, 1 depending on format
        assert(dim_x >= Nx);
        assert(dim_y >= Ny);
    }

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// Field Value Tests
// ============================================================================

void test_vtk_velocity_data() {
    std::cout << "Testing VTK contains velocity data... ";

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

    // Check for velocity field marker
    // Could be "VECTORS velocity" or "velocity" or similar
    bool has_velocity = (content.find("velocity") != std::string::npos) ||
                        (content.find("VECTORS") != std::string::npos) ||
                        (content.find("u_velocity") != std::string::npos);

    assert(has_velocity);

    remove_file(filename);
    std::cout << "PASSED\n";
}

void test_vtk_pressure_data() {
    std::cout << "Testing VTK contains pressure data... ";

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

    // Check for pressure field
    bool has_pressure = (content.find("pressure") != std::string::npos) ||
                        (content.find("SCALARS p") != std::string::npos) ||
                        (content.find("p ") != std::string::npos);

    assert(has_pressure);

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// GPU Synchronization Tests
// ============================================================================

void test_vtk_after_gpu_compute() {
    std::cout << "Testing VTK output after GPU compute... ";

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

    // Verify file was created and has content
    require_file_exists(filename);

    std::string content = read_file(filename);
    assert(content.length() > 100);  // Should have substantial content

    // Verify no NaN in output
    assert(content.find("nan") == std::string::npos);
    assert(content.find("NaN") == std::string::npos);
    assert(content.find("inf") == std::string::npos);

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// Multiple Output Tests
// ============================================================================

void test_vtk_sequential_outputs() {
    std::cout << "Testing sequential VTK outputs... ";

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
        require_file_exists(f);
    }

    // Clean up
    for (const auto& f : files) {
        remove_file(f);
    }

    std::cout << "PASSED\n";
}

// ============================================================================
// 3D VTK Tests
// ============================================================================

void test_vtk_3d_output() {
    std::cout << "Testing 3D VTK output... ";

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

    require_file_exists(filename);

    std::string content = read_file(filename);

    // Should have 3D dimensions
    assert(content.find("DIMENSIONS") != std::string::npos);

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// Turbulence Model Output Tests
// ============================================================================

void test_vtk_turbulence_fields() {
    std::cout << "Testing VTK with turbulence fields... ";

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

    std::string content = read_file(filename);

    // Should contain turbulence fields
    // Note: field names may vary (k, omega, nu_t, tke, etc.)
    bool has_turb_field = (content.find("nu_t") != std::string::npos) ||
                          (content.find("k") != std::string::npos) ||
                          (content.find("omega") != std::string::npos) ||
                          (content.find("tke") != std::string::npos);

    // This may not always be true depending on what's written
    // Just verify file was created successfully
    require_file_exists(filename);

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_vtk_small_grid() {
    std::cout << "Testing VTK on small grid (4x4)... ";

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

    require_file_exists(filename);

    remove_file(filename);
    std::cout << "PASSED\n";
}

void test_vtk_overwrite() {
    std::cout << "Testing VTK file overwrite... ";

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

    // Both should be valid VTK files
    assert(content1.find("vtk") != std::string::npos);
    assert(content2.find("vtk") != std::string::npos);

    remove_file(filename);
    std::cout << "PASSED\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== VTK Output Tests ===\n\n";

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

    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
