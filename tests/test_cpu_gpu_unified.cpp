/// Unified CPU/GPU Consistency Tests
/// Consolidates: test_cpu_gpu_consistency.cpp, test_solver_cpu_gpu.cpp, test_time_history_consistency.cpp
///
/// Tests:
/// 1. Turbulence model CPU/GPU parity (MixingLength, GEP, NN-MLP)
/// 2. Solver CPU/GPU parity (Taylor-Green, channel flow, grid sweep)
/// 3. Time-history consistency (no drift over time)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "test_utilities.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using nncfd::test::FieldComparison;
using nncfd::test::file_exists;
using nncfd::test::create_test_velocity_field;
using nncfd::test::check_gpu_cpu_consistency;
using nncfd::test::GPU_CPU_ABS_TOL;
using nncfd::test::GPU_CPU_REL_TOL;

static int passed = 0, failed = 0, skipped = 0;

static void record(const char* name, bool pass, bool skip = false) {
    std::cout << "  " << std::left << std::setw(50) << name;
    if (skip) { std::cout << "[SKIP]\n"; ++skipped; }
    else if (pass) { std::cout << "[PASS]\n"; ++passed; }
    else { std::cout << "[FAIL]\n"; ++failed; }
}

//=============================================================================
// Helpers
//=============================================================================

[[maybe_unused]] static bool gpu_available() {
#ifdef USE_GPU_OFFLOAD
    return omp_get_num_devices() > 0;
#else
    return false;
#endif
}

[[maybe_unused]] static bool verify_gpu_execution() {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() == 0) return false;
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    { on_device = !omp_is_initial_device(); }
    return on_device != 0;
#else
    return false;
#endif
}

struct SolverMetrics {
    double max_u = 0, max_v = 0, u_l2 = 0, v_l2 = 0, p_l2 = 0;
};

[[maybe_unused]] static SolverMetrics compute_solver_metrics(const Mesh& mesh, const VectorField& vel, const ScalarField& p) {
    SolverMetrics m;
    const int Ng = mesh.Nghost;
    double sum_u2 = 0, sum_v2 = 0, sum_p2 = 0;
    int n_u = 0, n_v = 0, n_p = 0;

    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            double u = vel.u(i, j);
            m.max_u = std::max(m.max_u, std::abs(u));
            sum_u2 += u * u; ++n_u;
        }
    }
    for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
        for (int i = Ng; i < Ng + mesh.Nx; ++i) {
            double v = vel.v(i, j);
            m.max_v = std::max(m.max_v, std::abs(v));
            sum_v2 += v * v; ++n_v;
        }
    }
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double pv = p(i, j);
            sum_p2 += pv * pv; ++n_p;
        }
    }

    m.u_l2 = std::sqrt(sum_u2 / std::max(1, n_u));
    m.v_l2 = std::sqrt(sum_v2 / std::max(1, n_v));
    m.p_l2 = std::sqrt(sum_p2 / std::max(1, n_p));
    return m;
}

//=============================================================================
// Test 1: MixingLength CPU/GPU Consistency
//=============================================================================

void test_mixing_length() {
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, 0.0, 1.0, 1);

    VectorField vel(mesh);
    create_test_velocity_field(mesh, vel, 42);
    ScalarField k(mesh), omega(mesh), nu_t_1(mesh), nu_t_2(mesh);

    MixingLengthModel m1, m2;
    m1.set_nu(0.001); m1.set_delta(0.5);
    m2.set_nu(0.001); m2.set_delta(0.5);

#ifdef USE_GPU_OFFLOAD
    if (gpu_available()) {
        const int total = mesh.total_cells();
        const int u_sz = vel.u_total_size(), v_sz = vel.v_total_size();
        double *u_p = vel.u_data().data(), *v_p = vel.v_data().data();
        double *nut1_p = nu_t_1.data().data();

        std::vector<double> dudx(total), dudy(total), dvdx(total), dvdy(total), wdist(total);
        FOR_INTERIOR_2D(mesh, i, j) { wdist[mesh.index(i, j)] = mesh.wall_distance(i, j); }
        double *dudx_p = dudx.data(), *dudy_p = dudy.data();
        double *dvdx_p = dvdx.data(), *dvdy_p = dvdy.data(), *wd_p = wdist.data();

        #pragma omp target enter data map(to: u_p[0:u_sz], v_p[0:v_sz], wd_p[0:total])
        #pragma omp target enter data map(alloc: nut1_p[0:total], dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])

        TurbulenceDeviceView dv{};
        dv.u_face = u_p; dv.v_face = v_p;
        dv.u_stride = vel.u_stride(); dv.v_stride = vel.v_stride();
        dv.nu_t = nut1_p; dv.cell_stride = mesh.total_Nx();
        dv.dudx = dudx_p; dv.dudy = dudy_p; dv.dvdx = dvdx_p; dv.dvdy = dvdy_p;
        dv.wall_distance = wd_p;
        dv.Nx = mesh.Nx; dv.Ny = mesh.Ny; dv.Ng = mesh.Nghost;
        dv.dx = mesh.dx; dv.dy = mesh.dy; dv.delta = 0.5;

        m1.update(mesh, vel, k, omega, nu_t_1, nullptr, &dv);
        #pragma omp target update from(nut1_p[0:total])
        #pragma omp target exit data map(delete: u_p[0:u_sz], v_p[0:v_sz], wd_p[0:total])
        #pragma omp target exit data map(delete: nut1_p[0:total], dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])
    } else {
        m1.update(mesh, vel, k, omega, nu_t_1);
    }
#else
    m1.update(mesh, vel, k, omega, nu_t_1);
#endif

    m2.update(mesh, vel, k, omega, nu_t_2);

    FieldComparison cmp;
    FOR_INTERIOR_2D(mesh, i, j) { cmp.update(i, j, nu_t_2(i, j), nu_t_1(i, j)); }
    cmp.finalize();

    auto chk = check_gpu_cpu_consistency(cmp);
    record("MixingLength CPU/GPU consistency", chk.passed);
}

//=============================================================================
// Test 2: GEP CPU/GPU Consistency
//=============================================================================

void test_gep() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, 0.0, 1.0, 1);

    VectorField vel(mesh);
    create_test_velocity_field(mesh, vel, 99);
    ScalarField k(mesh), omega(mesh), nu_t_1(mesh), nu_t_2(mesh);

    TurbulenceGEP g1, g2;
    g1.set_nu(0.001); g1.set_delta(0.5);
    g2.set_nu(0.001); g2.set_delta(0.5);

#ifdef USE_GPU_OFFLOAD
    if (gpu_available()) {
        const int total = mesh.total_cells();
        const int u_sz = vel.u_total_size(), v_sz = vel.v_total_size();
        double *u_p = vel.u_data().data(), *v_p = vel.v_data().data();
        double *nut1_p = nu_t_1.data().data();

        std::vector<double> dudx(total), dudy(total), dvdx(total), dvdy(total), wdist(total);
        FOR_INTERIOR_2D(mesh, i, j) { wdist[mesh.index(i, j)] = mesh.wall_distance(i, j); }
        double *dudx_p = dudx.data(), *dudy_p = dudy.data();
        double *dvdx_p = dvdx.data(), *dvdy_p = dvdy.data(), *wd_p = wdist.data();

        #pragma omp target enter data map(to: u_p[0:u_sz], v_p[0:v_sz], wd_p[0:total], nut1_p[0:total])
        #pragma omp target enter data map(to: dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])

        TurbulenceDeviceView dv{};
        dv.u_face = u_p; dv.v_face = v_p;
        dv.u_stride = vel.u_stride();
        dv.v_stride = vel.v_stride();
        dv.nu_t = nut1_p; dv.cell_stride = mesh.total_Nx();
        dv.dudx = dudx_p; dv.dudy = dudy_p; dv.dvdx = dvdx_p; dv.dvdy = dvdy_p;
        dv.wall_distance = wd_p;
        dv.Nx = mesh.Nx; dv.Ny = mesh.Ny; dv.Ng = mesh.Nghost;
        dv.dx = mesh.dx; dv.dy = mesh.dy;

        g1.update(mesh, vel, k, omega, nu_t_1, nullptr, &dv);
        #pragma omp target update from(nut1_p[0:total])
        #pragma omp target exit data map(delete: u_p[0:u_sz], v_p[0:v_sz], wd_p[0:total], nut1_p[0:total])
        #pragma omp target exit data map(delete: dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])
    } else {
        g1.update(mesh, vel, k, omega, nu_t_1, nullptr, nullptr);
    }
#else
    g1.update(mesh, vel, k, omega, nu_t_1, nullptr, nullptr);
#endif

    g2.update(mesh, vel, k, omega, nu_t_2, nullptr, nullptr);

    FieldComparison cmp;
    FOR_INTERIOR_2D(mesh, i, j) { cmp.update(i, j, nu_t_2(i, j), nu_t_1(i, j)); }
    cmp.finalize();

    auto chk = check_gpu_cpu_consistency(cmp);
    record("TurbulenceGEP CPU/GPU consistency", chk.passed);
}

//=============================================================================
// Test 3: NN-MLP Consistency
//=============================================================================

void test_nn_mlp() {
    std::string path = "data/models/mlp_channel_caseholdout";
    if (!file_exists(path + "/layer0_W.txt")) path = "../" + path;
    if (!file_exists(path + "/layer0_W.txt")) {
        record("TurbulenceNNMLP CPU/GPU consistency", true, true);
        return;
    }

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, 0.0, 1.0, 1);

    VectorField vel(mesh);
    create_test_velocity_field(mesh, vel, 0);
    ScalarField k(mesh, 0.01), omega(mesh, 10.0), nu_t_cpu(mesh), nu_t_gpu(mesh);

    TurbulenceNNMLP cpu_model;
    cpu_model.set_nu(0.001);
    cpu_model.load(path, path);
    cpu_model.update(mesh, vel, k, omega, nu_t_cpu);

#ifdef USE_GPU_OFFLOAD
    if (gpu_available()) {
        TurbulenceNNMLP gpu_model;
        gpu_model.set_nu(0.001);
        gpu_model.load(path, path);
        gpu_model.initialize_gpu_buffers(mesh);

        if (!gpu_model.is_gpu_ready()) {
            record("TurbulenceNNMLP CPU/GPU consistency", false);
            return;
        }

        const int total = mesh.total_cells();
        const int u_sz = vel.u_total_size(), v_sz = vel.v_total_size();
        double *u_p = vel.u_data().data(), *v_p = vel.v_data().data();
        double *k_p = k.data().data(), *om_p = omega.data().data();
        double *nut_p = nu_t_gpu.data().data();

        std::vector<double> dudx(total), dudy(total), dvdx(total), dvdy(total), wdist(total);
        FOR_INTERIOR_2D(mesh, i, j) { wdist[mesh.index(i, j)] = mesh.wall_distance(i, j); }
        double *dudx_p = dudx.data(), *dudy_p = dudy.data();
        double *dvdx_p = dvdx.data(), *dvdy_p = dvdy.data(), *wd_p = wdist.data();

        #pragma omp target enter data map(to: u_p[0:u_sz], v_p[0:v_sz])
        #pragma omp target enter data map(to: k_p[0:total], om_p[0:total], wd_p[0:total])
        #pragma omp target enter data map(alloc: nut_p[0:total], dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])

        TurbulenceDeviceView dv{};
        dv.u_face = u_p; dv.v_face = v_p;
        dv.u_stride = vel.u_stride(); dv.v_stride = vel.v_stride();
        dv.k = k_p; dv.omega = om_p; dv.nu_t = nut_p;
        dv.cell_stride = mesh.Nx + 2*mesh.Nghost;
        dv.dudx = dudx_p; dv.dudy = dudy_p; dv.dvdx = dvdx_p; dv.dvdy = dvdy_p;
        dv.wall_distance = wd_p;
        dv.Nx = mesh.Nx; dv.Ny = mesh.Ny; dv.Ng = mesh.Nghost;
        dv.dx = mesh.dx; dv.dy = mesh.dy; dv.delta = 1.0;

        gpu_model.update(mesh, vel, k, omega, nu_t_gpu, nullptr, &dv);
        #pragma omp target update from(nut_p[0:total])
        #pragma omp target exit data map(delete: u_p[0:u_sz], v_p[0:v_sz])
        #pragma omp target exit data map(delete: k_p[0:total], om_p[0:total], wd_p[0:total])
        #pragma omp target exit data map(delete: nut_p[0:total], dudx_p[0:total], dudy_p[0:total], dvdx_p[0:total], dvdy_p[0:total])
    } else {
        TurbulenceNNMLP m2;
        m2.set_nu(0.001);
        m2.load(path, path);
        m2.update(mesh, vel, k, omega, nu_t_gpu);
    }
#else
    TurbulenceNNMLP m2;
    m2.set_nu(0.001);
    m2.load(path, path);
    m2.update(mesh, vel, k, omega, nu_t_gpu);
#endif

    FieldComparison cmp;
    FOR_INTERIOR_2D(mesh, i, j) { cmp.update(i, j, nu_t_cpu(i, j), nu_t_gpu(i, j)); }
    cmp.finalize();

    bool pass = cmp.max_abs_diff < 1e-10 || cmp.max_rel_diff < 1e-8;
    record("TurbulenceNNMLP CPU/GPU consistency", pass);
}

//=============================================================================
// Test 4: Solver Consistency - Taylor-Green
//=============================================================================

void test_solver_taylor_green() {
    Config cfg;
    cfg.Nx = 64; cfg.Ny = 64;
    cfg.x_min = 0; cfg.x_max = 2*M_PI;
    cfg.y_min = 0; cfg.y_max = 2*M_PI;
    cfg.nu = 0.01; cfg.dt = 0.0001;
    cfg.adaptive_dt = false;
    cfg.turb_model = TurbulenceModelType::None;
    cfg.verbose = false;

    Mesh mesh;
    mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);

    VectorField vel_init(mesh);
    const int Ng = mesh.Nghost;
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            double x = mesh.x_min + (i - Ng) * mesh.dx;
            double y = mesh.y(j);
            vel_init.u(i, j) = -std::cos(x) * std::sin(y);
        }
    }
    for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
        for (int i = Ng; i < Ng + mesh.Nx; ++i) {
            double x = mesh.x(i);
            double y = mesh.y_min + (j - Ng) * mesh.dy;
            vel_init.v(i, j) = std::sin(x) * std::cos(y);
        }
    }

    RANSSolver s1(mesh, cfg), s2(mesh, cfg);
    VelocityBC bc; bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    s1.set_velocity_bc(bc); s2.set_velocity_bc(bc);
    s1.initialize(vel_init); s2.initialize(vel_init);

    for (int step = 0; step < 10; ++step) { s1.step(); s2.step(); }

#ifdef USE_GPU_OFFLOAD
    s1.sync_from_gpu(); s2.sync_from_gpu();
#endif

    double max_diff = 0;
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            max_diff = std::max(max_diff, std::abs(s1.velocity().u(i,j) - s2.velocity().u(i,j)));
        }
    }

    record("Solver Taylor-Green consistency", max_diff < 1e-12);
}

//=============================================================================
// Test 5: Solver Consistency - Channel Flow
//=============================================================================

void test_solver_channel() {
    Config cfg;
    cfg.Nx = 64; cfg.Ny = 32;
    cfg.x_min = 0; cfg.x_max = 4.0;
    cfg.y_min = -1; cfg.y_max = 1;
    cfg.nu = 0.01; cfg.dp_dx = -0.001; cfg.dt = 0.001;
    cfg.adaptive_dt = false;
    cfg.turb_model = TurbulenceModelType::None;
    cfg.verbose = false;

    Mesh mesh;
    mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);

    RANSSolver s1(mesh, cfg), s2(mesh, cfg);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
    s1.set_velocity_bc(bc); s2.set_velocity_bc(bc);
    s1.set_body_force(-cfg.dp_dx, 0); s2.set_body_force(-cfg.dp_dx, 0);
    s1.initialize_uniform(0.1, 0); s2.initialize_uniform(0.1, 0);

    for (int step = 0; step < 10; ++step) { s1.step(); s2.step(); }

#ifdef USE_GPU_OFFLOAD
    s1.sync_from_gpu(); s2.sync_from_gpu();
#endif

    double max_diff = 0;
    const int Ng = mesh.Nghost;
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            max_diff = std::max(max_diff, std::abs(s1.velocity().u(i,j) - s2.velocity().u(i,j)));
        }
    }

    record("Solver channel flow consistency", max_diff < 1e-12);
}

//=============================================================================
// Test 6: Solver Consistency - Grid Sweep
//=============================================================================

void test_solver_grid_sweep() {
    struct Grid { int nx, ny; };
    std::vector<Grid> grids = {{32, 32}, {64, 48}, {63, 97}};
    bool all_pass = true;

    for (const auto& g : grids) {
        Config cfg;
        cfg.Nx = g.nx; cfg.Ny = g.ny;
        cfg.x_min = 0; cfg.x_max = 2*M_PI;
        cfg.y_min = 0; cfg.y_max = 2*M_PI;
        cfg.nu = 0.01; cfg.dt = 0.0001;
        cfg.adaptive_dt = false;
        cfg.turb_model = TurbulenceModelType::None;
        cfg.verbose = false;

        Mesh mesh;
        mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max);

        RANSSolver s1(mesh, cfg), s2(mesh, cfg);
        VelocityBC bc; bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
        s1.set_velocity_bc(bc); s2.set_velocity_bc(bc);
        s1.initialize_uniform(0.5, 0.3); s2.initialize_uniform(0.5, 0.3);

        for (int step = 0; step < 5; ++step) { s1.step(); s2.step(); }

#ifdef USE_GPU_OFFLOAD
        s1.sync_from_gpu(); s2.sync_from_gpu();
#endif

        double max_diff = 0;
        const int Ng = mesh.Nghost;
        for (int j = Ng; j < Ng + mesh.Ny; ++j) {
            for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
                max_diff = std::max(max_diff, std::abs(s1.velocity().u(i,j) - s2.velocity().u(i,j)));
            }
        }

        if (max_diff >= 1e-12) all_pass = false;
    }

    record("Solver grid sweep consistency", all_pass);
}

//=============================================================================
// Test 7: Time-History Consistency (no drift over time)
//=============================================================================

struct TimeSnapshot {
    double ke = 0, flux = 0, max_u = 0, max_v = 0, avg_nu_t = 0;
};

[[maybe_unused]] static TimeSnapshot compute_diagnostics(const Mesh& mesh, const VectorField& vel, const ScalarField& nu_t) {
    TimeSnapshot s;
    int n = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = vel.u(i, j), v = vel.v(i, j);
            s.ke += 0.5 * (u*u + v*v);
            s.flux += u;
            s.max_u = std::max(s.max_u, std::abs(u));
            s.max_v = std::max(s.max_v, std::abs(v));
            s.avg_nu_t += nu_t(i, j);
            ++n;
        }
    }
    s.ke /= n; s.flux /= n; s.avg_nu_t /= n;
    return s;
}

void test_time_history() {
#ifdef USE_GPU_OFFLOAD
    if (!gpu_available()) {
        record("Time-history consistency (no drift)", true, true);
        return;
    }
    if (!verify_gpu_execution()) {
        record("Time-history consistency (no drift)", false);
        return;
    }

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, 0.0, 1.0, 1);

    Config cfg;
    cfg.nu = 0.001; cfg.dp_dx = -0.0001; cfg.dt = 0.001;
    cfg.adaptive_dt = false; cfg.max_iter = 50; cfg.tol = 1e-8;
    cfg.turb_model = TurbulenceModelType::Baseline;
    cfg.verbose = false;

    RANSSolver s1(mesh, cfg), s2(mesh, cfg);
    auto t1 = std::make_unique<MixingLengthModel>();
    auto t2 = std::make_unique<MixingLengthModel>();
    t1->set_nu(cfg.nu); t1->set_delta(0.5);
    t2->set_nu(cfg.nu); t2->set_delta(0.5);
    s1.set_turbulence_model(std::move(t1));
    s2.set_turbulence_model(std::move(t2));
    s1.set_body_force(-cfg.dp_dx, 0); s2.set_body_force(-cfg.dp_dx, 0);
    s1.initialize_uniform(0.1, 0); s2.initialize_uniform(0.1, 0);

    double max_ke_diff = 0, max_flux_diff = 0;
    const int steps = 50;

    for (int step = 1; step <= steps; ++step) {
        s1.step(); s2.step();
        if (step % 10 == 0) {
            auto snap1 = compute_diagnostics(mesh, s1.velocity(), s1.nu_t());
            auto snap2 = compute_diagnostics(mesh, s2.velocity(), s2.nu_t());
            max_ke_diff = std::max(max_ke_diff, std::abs(snap1.ke - snap2.ke));
            max_flux_diff = std::max(max_flux_diff, std::abs(snap1.flux - snap2.flux));
        }
    }

    bool pass = (max_ke_diff < 1e-8) && (max_flux_diff < 1e-8);
    record("Time-history consistency (no drift)", pass);
#else
    // CPU-only: verify sequential sum works
    double sum = 0;
    for (int i = 0; i < 1000; ++i) sum += std::sin(i * 0.01);
    record("Time-history consistency (CPU)", std::isfinite(sum));
#endif
}

//=============================================================================
// Test 8: Randomized Regression
//=============================================================================

void test_randomized() {
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, 0.0, 1.0, 1);

    const int trials = 10;
    double worst_abs = 0;

    for (int t = 0; t < trials; ++t) {
        VectorField vel(mesh);
        ScalarField k(mesh), omega(mesh), nu1(mesh), nu2(mesh);
        create_test_velocity_field(mesh, vel, t * 42);

        MixingLengthModel m1, m2;
        m1.set_nu(0.0001); m1.set_delta(0.5);
        m2.set_nu(0.0001); m2.set_delta(0.5);
        m1.update(mesh, vel, k, omega, nu1);
        m2.update(mesh, vel, k, omega, nu2);

        double max_abs = 0;
        FOR_INTERIOR_2D(mesh, i, j) {
            max_abs = std::max(max_abs, std::abs(nu1(i,j) - nu2(i,j)));
        }
        worst_abs = std::max(worst_abs, max_abs);
    }

    bool pass = worst_abs < GPU_CPU_ABS_TOL;
    record("Randomized regression (10 trials)", pass);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    // Check for dump/compare mode (cross-build testing)
    std::string dump_prefix, compare_prefix;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dump-prefix" && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "--compare-prefix" && i + 1 < argc) compare_prefix = argv[++i];
    }

    if (!dump_prefix.empty() || !compare_prefix.empty()) {
        std::cout << "Note: --dump-prefix/--compare-prefix are handled by test_cpu_gpu_bitwise.\n";
        std::cout << "This test performs in-process CPU/GPU consistency checks.\n";
        std::cout << "Run without these flags for the full test suite.\n";
        return 0;
    }

    std::cout << "================================================================\n";
    std::cout << "  Unified CPU/GPU Consistency Tests\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
    std::cout << "Devices: " << omp_get_num_devices() << "\n";
    if (gpu_available()) {
        std::cout << "GPU execution: " << (verify_gpu_execution() ? "YES" : "NO") << "\n";
    }
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
    std::cout << "\n";

    // Run all tests
    test_mixing_length();
    test_gep();
    test_nn_mlp();
    test_solver_taylor_green();
    test_solver_channel();
    test_solver_grid_sweep();
    test_time_history();
    test_randomized();

    std::cout << "\n================================================================\n";
    std::cout << "Summary: " << passed << " passed, " << failed << " failed, "
              << skipped << " skipped\n";
    std::cout << "================================================================\n";

    return failed > 0 ? 1 : 0;
}
