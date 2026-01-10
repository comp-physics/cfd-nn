/// 2D vs 3D Solver Comparison Tests
/// Validates that 3D solver produces correct results by comparing to 2D reference

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include <cassert>
#include <cmath>
#include <vector>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// Test parameters
constexpr int NX = 32, NY = 32;
constexpr double LX = 2.0, LY = 2.0, LZ = 1.0;
constexpr double NU = 0.01, DP_DX = -0.001;
constexpr int MAX_ITER = 500;

static double vec_l2_diff(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size() && "vec_l2_diff: vectors must have same size");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; sum += d * d; }
    return std::sqrt(sum / a.size());
}

static double vec_linf_diff(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size() && "vec_linf_diff: vectors must have same size");
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static std::vector<double> extract_2d_u(const RANSSolver& s, const Mesh& m) {
    std::vector<double> v;
    for (int j = m.j_begin(); j < m.j_end(); ++j)
        for (int i = m.i_begin(); i <= m.i_end(); ++i)
            v.push_back(s.velocity().u(i, j));
    return v;
}

static std::vector<double> extract_2d_v(const RANSSolver& s, const Mesh& m) {
    std::vector<double> v;
    for (int j = m.j_begin(); j <= m.j_end(); ++j)
        for (int i = m.i_begin(); i < m.i_end(); ++i)
            v.push_back(s.velocity().v(i, j));
    return v;
}

static std::vector<double> extract_3d_u_slice(const RANSSolver& s, const Mesh& m, int k) {
    std::vector<double> v;
    for (int j = m.j_begin(); j < m.j_end(); ++j)
        for (int i = m.i_begin(); i <= m.i_end(); ++i)
            v.push_back(s.velocity().u(i, j, k));
    return v;
}

static std::vector<double> extract_3d_v_slice(const RANSSolver& s, const Mesh& m, int k) {
    std::vector<double> v;
    for (int j = m.j_begin(); j <= m.j_end(); ++j)
        for (int i = m.i_begin(); i < m.i_end(); ++i)
            v.push_back(s.velocity().v(i, j, k));
    return v;
}

static double max_div_3d(const RANSSolver& s, const Mesh& m) {
    double max_div = 0.0;
    FOR_INTERIOR_3D(m, i, j, k) {
        double div = (s.velocity().u(i+1, j, k) - s.velocity().u(i, j, k)) / m.dx
                   + (s.velocity().v(i, j+1, k) - s.velocity().v(i, j, k)) / m.dy
                   + (s.velocity().w(i, j, k+1) - s.velocity().w(i, j, k)) / m.dz;
        max_div = std::max(max_div, std::abs(div));
    }
    return max_div;
}

static double max_div_3d_slice(const RANSSolver& s, const Mesh& m, int k) {
    double max_div = 0.0;
    FOR_INTERIOR_2D(m, i, j) {
        double div = (s.velocity().u(i+1, j, k) - s.velocity().u(i, j, k)) / m.dx
                   + (s.velocity().v(i, j+1, k) - s.velocity().v(i, j, k)) / m.dy;
        max_div = std::max(max_div, std::abs(div));
    }
    return max_div;
}

static void init_poiseuille_2d(RANSSolver& s, const Mesh& m, double dp_dx, double nu) {
    double H = LY / 2.0, y_mid = LY / 2.0;
    FOR_INTERIOR_2D(m, i, j) {
        double y = m.y(j) - y_mid;
        s.velocity().u(i, j) = 0.9 * (-dp_dx / (2.0 * nu)) * (H * H - y * y);
    }
    for (int j = m.j_begin(); j < m.j_end(); ++j) {
        double y = m.y(j) - y_mid;
        s.velocity().u(m.i_end(), j) = 0.9 * (-dp_dx / (2.0 * nu)) * (H * H - y * y);
    }
    for (int j = m.j_begin(); j <= m.j_end(); ++j)
        for (int i = m.i_begin(); i < m.i_end(); ++i)
            s.velocity().v(i, j) = 0.0;
}

static void init_poiseuille_3d(RANSSolver& s, const Mesh& m, double dp_dx, double nu) {
    double H = LY / 2.0, y_mid = LY / 2.0;
    FOR_INTERIOR_3D(m, i, j, k) {
        double y = m.y(j) - y_mid;
        s.velocity().u(i, j, k) = 0.9 * (-dp_dx / (2.0 * nu)) * (H * H - y * y);
    }
    // Staggered u at i_end
    for (int k = m.k_begin(); k < m.k_end(); ++k)
        for (int j = m.j_begin(); j < m.j_end(); ++j) {
            double y = m.y(j) - y_mid;
            s.velocity().u(m.i_end(), j, k) = 0.9 * (-dp_dx / (2.0 * nu)) * (H * H - y * y);
        }
    // v = 0
    for (int k = m.k_begin(); k < m.k_end(); ++k)
        for (int j = m.j_begin(); j <= m.j_end(); ++j)
            for (int i = m.i_begin(); i < m.i_end(); ++i)
                s.velocity().v(i, j, k) = 0.0;
    // w = 0
    for (int k = m.k_begin(); k <= m.k_end(); ++k)
        for (int j = m.j_begin(); j < m.j_end(); ++j)
            for (int i = m.i_begin(); i < m.i_end(); ++i)
                s.velocity().w(i, j, k) = 0.0;
}

static Config make_test_config() {
    Config c;
    c.nu = NU; c.dp_dx = DP_DX;
    c.adaptive_dt = true; c.max_iter = MAX_ITER;
    c.tol = 1e-6; c.turb_model = TurbulenceModelType::None;
    c.verbose = false;
    return c;
}

//=============================================================================
// TEST 1: Degenerate case - 3D with Nz=1 should match 2D
//=============================================================================
void test_degenerate_nz1() {
    // 2D
    Mesh mesh_2d;
    mesh_2d.init_uniform(NX, NY, 0.0, LX, 0.0, LY);
    Config cfg = make_test_config();
    RANSSolver solver_2d(mesh_2d, cfg);
    solver_2d.set_body_force(-cfg.dp_dx, 0.0);
    init_poiseuille_2d(solver_2d, mesh_2d, cfg.dp_dx, cfg.nu);
    solver_2d.sync_to_gpu();
    solver_2d.solve_steady();

    // 3D with Nz=1
    Mesh mesh_3d;
    mesh_3d.init_uniform(NX, NY, 1, 0.0, LX, 0.0, LY, 0.0, LZ);
    RANSSolver solver_3d(mesh_3d, cfg);
    solver_3d.set_body_force(-cfg.dp_dx, 0.0, 0.0);
    // Use 2D init to match solver_2d exactly; explicitly zero w for completeness
    init_poiseuille_2d(solver_3d, mesh_3d, cfg.dp_dx, cfg.nu);
    for (int k = mesh_3d.k_begin(); k <= mesh_3d.k_end(); ++k)
        for (int j = mesh_3d.j_begin(); j < mesh_3d.j_end(); ++j)
            for (int i = mesh_3d.i_begin(); i < mesh_3d.i_end(); ++i)
                solver_3d.velocity().w(i, j, k) = 0.0;
    solver_3d.sync_to_gpu();
    solver_3d.solve_steady();

    auto u_2d = extract_2d_u(solver_2d, mesh_2d);
    auto v_2d = extract_2d_v(solver_2d, mesh_2d);
    auto u_3d = extract_3d_u_slice(solver_3d, mesh_3d, 0);
    auto v_3d = extract_3d_v_slice(solver_3d, mesh_3d, 0);

    double div_3d = mesh_3d.is2D() ? max_div_3d_slice(solver_3d, mesh_3d, 0)
                                   : max_div_3d(solver_3d, mesh_3d);

    bool passed = vec_l2_diff(u_2d, u_3d) <= 1e-8 && vec_l2_diff(v_2d, v_3d) <= 1e-8 && div_3d <= 1e-8;
    record("Degenerate case (Nz=1 vs 2D)", passed);
}

//=============================================================================
// TEST 2: Z-invariant flow - 3D with Nz=4 should match 2D
//=============================================================================
void test_z_invariant_poiseuille() {
    constexpr int NZ = 4;

    // 2D
    Mesh mesh_2d;
    mesh_2d.init_uniform(NX, NY, 0.0, LX, 0.0, LY);
    Config cfg = make_test_config();
    RANSSolver solver_2d(mesh_2d, cfg);
    solver_2d.set_body_force(-cfg.dp_dx, 0.0);
    init_poiseuille_2d(solver_2d, mesh_2d, cfg.dp_dx, cfg.nu);
    solver_2d.sync_to_gpu();
    solver_2d.solve_steady();

    // 3D with Nz=4
    Mesh mesh_3d;
    mesh_3d.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);
    RANSSolver solver_3d(mesh_3d, cfg);
    solver_3d.set_body_force(-cfg.dp_dx, 0.0, 0.0);
    solver_3d.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    init_poiseuille_3d(solver_3d, mesh_3d, cfg.dp_dx, cfg.nu);
    solver_3d.sync_to_gpu();
    solver_3d.solve_steady();

    auto u_2d = extract_2d_u(solver_2d, mesh_2d);
    auto v_2d = extract_2d_v(solver_2d, mesh_2d);

    double max_u_err = 0.0, max_v_err = 0.0, max_z_var = 0.0;
    auto u_plane0 = extract_3d_u_slice(solver_3d, mesh_3d, mesh_3d.k_begin());

    for (int k = mesh_3d.k_begin(); k < mesh_3d.k_end(); ++k) {
        auto u_3d = extract_3d_u_slice(solver_3d, mesh_3d, k);
        auto v_3d = extract_3d_v_slice(solver_3d, mesh_3d, k);
        max_u_err = std::max(max_u_err, vec_l2_diff(u_2d, u_3d));
        max_v_err = std::max(max_v_err, vec_l2_diff(v_2d, v_3d));
        if (k > mesh_3d.k_begin())
            max_z_var = std::max(max_z_var, vec_linf_diff(u_plane0, u_3d));
    }

    bool passed = max_u_err <= 1e-3 && max_v_err <= 1e-3 && max_div_3d(solver_3d, mesh_3d) <= 1e-4 && max_z_var <= 5e-4;
    record("Z-invariant Poiseuille (Nz=4 vs 2D)", passed);
}

//=============================================================================
// TEST 3: Verify w stays zero for z-invariant flow
//=============================================================================
void test_w_stays_zero() {
    constexpr int NZ = 4;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);
    Config cfg = make_test_config();
    RANSSolver solver(mesh, cfg);
    solver.set_body_force(-cfg.dp_dx, 0.0, 0.0);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    init_poiseuille_3d(solver, mesh, cfg.dp_dx, cfg.nu);
    solver.sync_to_gpu();
    solver.solve_steady();

    double max_w = 0.0, max_u = 0.0;
    FOR_INTERIOR_3D(mesh, i, j, k) {
        max_u = std::max(max_u, std::abs(solver.velocity().u(i, j, k)));
    }
    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                max_u = std::max(max_u, std::abs(solver.velocity().u(i, j, k)));

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                max_w = std::max(max_w, std::abs(solver.velocity().w(i, j, k)));

    record("Verify w stays zero for z-invariant flow", max_w / std::max(max_u, 1e-10) < 1e-3);
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    return nncfd::test::harness::run("2D vs 3D Solver Comparison Tests", [] {
        test_degenerate_nz1();
        test_z_invariant_poiseuille();
        test_w_stays_zero();
    });
}
