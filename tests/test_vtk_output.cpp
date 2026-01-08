/// Unit tests for VTK output validation
///
/// Tests VTK file output functionality:
/// - File format correctness
/// - Field synchronization (GPU → CPU)
/// - Output directory handling
/// - Multiple snapshot output

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::make_test_solver_domain;
using nncfd::test::make_test_solver_3d_domain;
using nncfd::test::BCPattern;

namespace {
void remove_file(const std::string& filename) { std::remove(filename.c_str()); }

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
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_created.vtk";
    remove_file(filename);
    ts->write_vtk(filename);

    std::ifstream f(filename);
    bool exists = f.good();
    remove_file(filename);
    record("VTK file creation", exists);
}

void test_vtk_header_format() {
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_header.vtk";
    ts->write_vtk(filename);
    std::string content = read_file(filename);

    bool pass = (content.find("# vtk DataFile Version") != std::string::npos) &&
                (content.find("ASCII") != std::string::npos || content.find("BINARY") != std::string::npos) &&
                (content.find("STRUCTURED_GRID") != std::string::npos ||
                 content.find("RECTILINEAR_GRID") != std::string::npos ||
                 content.find("STRUCTURED_POINTS") != std::string::npos);

    remove_file(filename);
    record("VTK header format", pass);
}

void test_vtk_dimensions_match() {
    int Nx = 16, Ny = 32;
    auto ts = make_test_solver_domain(Nx, Ny, 0.0, 1.0, 0.0, 2.0);
    ts->initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_dims.vtk";
    ts->write_vtk(filename);
    std::string content = read_file(filename);

    bool pass = true;
    auto pos = content.find("DIMENSIONS");
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
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(1.0, 0.0);

    std::string filename = "/tmp/test_vtk_velocity.vtk";
    ts->write_vtk(filename);
    std::string content = read_file(filename);

    bool has_velocity = (content.find("velocity") != std::string::npos) ||
                        (content.find("VECTORS") != std::string::npos) ||
                        (content.find("u_velocity") != std::string::npos);

    remove_file(filename);
    record("VTK contains velocity data", has_velocity);
}

void test_vtk_pressure_data() {
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(1.0, 0.0);
    ts->step();

    std::string filename = "/tmp/test_vtk_pressure.vtk";
    ts->write_vtk(filename);
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
    auto ts = make_test_solver_domain(16, 32, 0.0, 1.0, -0.5, 0.5);
    ts->set_body_force(-0.001, 0.0);
    ts->initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 20; ++i) ts->step();

    std::string filename = "/tmp/test_vtk_gpu.vtk";
    ts->write_vtk(filename);
    std::string content = read_file(filename);

    bool pass = (content.length() > 100) &&
                (content.find("nan") == std::string::npos) &&
                (content.find("NaN") == std::string::npos) &&
                (content.find("inf") == std::string::npos);

    remove_file(filename);
    record("VTK output after GPU compute", pass);
}

// ============================================================================
// Multiple Output Tests
// ============================================================================

void test_vtk_sequential_outputs() {
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->set_body_force(-0.001, 0.0);
    ts->initialize_uniform(0.5, 0.0);

    std::vector<std::string> files;
    bool pass = true;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) ts->step();
        std::string filename = "/tmp/test_vtk_seq_" + std::to_string(i) + ".vtk";
        files.push_back(filename);
        ts->write_vtk(filename);
    }

    for (const auto& f : files) {
        std::ifstream fin(f);
        if (!fin.good()) pass = false;
    }
    for (const auto& f : files) remove_file(f);

    record("Sequential VTK outputs", pass);
}

// ============================================================================
// 3D VTK Tests
// ============================================================================

void test_vtk_3d_output() {
    auto ts = make_test_solver_3d_domain(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 5; ++i) ts->step();

    std::string filename = "/tmp/test_vtk_3d.vtk";
    ts->write_vtk(filename);
    std::string content = read_file(filename);

    bool pass = (content.find("DIMENSIONS") != std::string::npos);
    remove_file(filename);
    record("3D VTK output", pass);
}

// ============================================================================
// Turbulence Model Output Tests
// ============================================================================

void test_vtk_turbulence_fields() {
    // Need custom setup for turbulence model
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.turb_model = TurbulenceModelType::KOmega;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(nncfd::test::create_velocity_bc(BCPattern::Channel2D));

    auto turb_model = create_turbulence_model(TurbulenceModelType::KOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 10; ++i) solver.step();

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
    auto ts = make_test_solver_domain(4, 4, 0.0, 1.0, 0.0, 1.0, BCPattern::Channel2D, 0.1, 0.01);
    ts->initialize_uniform(0.5, 0.0);

    std::string filename = "/tmp/test_vtk_small.vtk";
    ts->write_vtk(filename);

    std::ifstream fin(filename);
    bool pass = fin.good();
    remove_file(filename);
    record("VTK on small grid (4x4)", pass);
}

void test_vtk_overwrite() {
    auto ts = make_test_solver_domain(8, 8, 0.0, 1.0, 0.0, 1.0);
    ts->initialize_uniform(0.5, 0.0);

    std::string filename = "/tmp/test_vtk_overwrite.vtk";
    ts->write_vtk(filename);
    std::string content1 = read_file(filename);

    for (int i = 0; i < 10; ++i) ts->step();

    ts->write_vtk(filename);
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
        test_vtk_file_created();
        test_vtk_header_format();
        test_vtk_dimensions_match();
        test_vtk_velocity_data();
        test_vtk_pressure_data();
        test_vtk_after_gpu_compute();
        test_vtk_sequential_outputs();
        test_vtk_3d_output();
        test_vtk_turbulence_fields();
        test_vtk_small_grid();
        test_vtk_overwrite();
    });
}
