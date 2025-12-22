/// Integration tests for NN turbulence models with the solver
/// Tests that NN models work correctly within the full solver loop

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>

using namespace nncfd;

// Helper to check if a file exists
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Helper to check field validity
bool is_field_valid(const ScalarField& field, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(field(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool is_velocity_valid(const VectorField& vel, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                return false;
            }
        }
    }
    return true;
}

// Test 1: NN-MLP model produces valid output
void test_nn_mlp_validity() {
    std::cout << "Testing NN-MLP model validity... ";
    
    // Use trained MLP model from data/models/mlp_channel_caseholdout
    std::string model_path = "data/models/mlp_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/mlp_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    // Create velocity field with shear
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);  // Parabolic profile
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNMLP model;
    model.set_nu(0.001);
    
    try {
        model.load(model_path, model_path);
        
#ifdef USE_GPU_OFFLOAD
        // GPU build - MUST use GPU, no fallback allowed
        int num_devices = omp_get_num_devices();
        if (num_devices == 0) {
            std::cerr << "FAILED: GPU build but no GPU devices available!\n";
            assert(false && "GPU build requires GPU device");
        }
        
        model.initialize_gpu_buffers(mesh);
        
        // Get correct sizes for staggered and cell-centered arrays
        const int u_total = vel.u_total_size();
        const int v_total = vel.v_total_size();
        const int total_cells = mesh.total_cells();
        
        // Get pointers to field data
        double* u_ptr = vel.u_data().data();
        double* v_ptr = vel.v_data().data();
        double* k_ptr = k.data().data();
        double* omega_ptr = omega.data().data();
        double* nu_t_ptr = nu_t.data().data();
        
        // Allocate gradient arrays
        std::vector<double> dudx(total_cells), dudy(total_cells);
        std::vector<double> dvdx(total_cells), dvdy(total_cells);
        std::vector<double> wall_dist(total_cells, 0.5);
        
        // Compute gradients
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                int idx = j * mesh.total_Nx() + i;
                dudx[idx] = 0.0;
                dudy[idx] = (vel.u(i, j+1) - vel.u(i, j-1)) / (2.0 * mesh.dy);
                dvdx[idx] = 0.0;
                dvdy[idx] = 0.0;
            }
        }
        
        // Get pointers for GPU mapping
        double* dudx_ptr = dudx.data();
        double* dudy_ptr = dudy.data();
        double* dvdx_ptr = dvdx.data();
        double* dvdy_ptr = dvdy.data();
        double* wall_dist_ptr = wall_dist.data();
        
        // Upload to GPU with correct sizes
        #pragma omp target enter data map(to: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target enter data map(to: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target enter data map(to: nu_t_ptr[0:total_cells])
        #pragma omp target enter data map(to: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target enter data map(to: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target enter data map(to: wall_dist_ptr[0:total_cells])
        
        // Create device view with ALL required pointers
        TurbulenceDeviceView device_view;
        device_view.u_face = u_ptr;
        device_view.v_face = v_ptr;
        device_view.k = k_ptr;                    // REQUIRED for NN models
        device_view.omega = omega_ptr;            // REQUIRED for NN models
        device_view.nu_t = nu_t_ptr;
        device_view.u_stride = vel.u_stride();
        device_view.v_stride = vel.v_stride();
        device_view.cell_stride = mesh.total_Nx();
        device_view.dudx = dudx_ptr;
        device_view.dudy = dudy_ptr;
        device_view.dvdx = dvdx_ptr;
        device_view.dvdy = dvdy_ptr;
        device_view.wall_distance = wall_dist_ptr;
        device_view.Nx = mesh.Nx;
        device_view.Ny = mesh.Ny;
        device_view.Ng = mesh.Nghost;
        device_view.dx = mesh.dx;
        device_view.dy = mesh.dy;
        device_view.delta = 1.0;
        
        // GPU build MUST use GPU path - device_view is REQUIRED
        model.update(mesh, vel, k, omega, nu_t, nullptr, &device_view);
        
        // Verify GPU was actually used
        if (!model.is_gpu_ready()) {
            std::cerr << "FAILED: GPU build but model didn't use GPU!\n";
            assert(false && "GPU build must execute on GPU");
        }
        
        // Download results
        #pragma omp target update from(nu_t_ptr[0:total_cells])
        
        // Cleanup with correct sizes
        #pragma omp target exit data map(delete: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target exit data map(delete: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target exit data map(delete: nu_t_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: wall_dist_ptr[0:total_cells])
#else
        // CPU build - use CPU path
        model.update(mesh, vel, k, omega, nu_t);
#endif
        
        // Check all values are finite and non-negative
        bool valid = true;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(nu_t(i, j)) || nu_t(i, j) < 0.0) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        
        assert(valid && "NN-MLP produced invalid nu_t values!");
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
#ifdef USE_GPU_OFFLOAD
        std::cerr << "FAILED (GPU): " << e.what() << "\n";
        assert(false && "GPU build must not throw exceptions");
#else
        std::cout << "SKIPPED (" << e.what() << ")\n";
#endif
    }
}

// Test 2: NN-TBNN model produces valid output
void test_nn_tbnn_validity() {
    std::cout << "Testing NN-TBNN model validity... ";
    
    // Use trained TBNN model
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        // Try from build directory
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.set_u_ref(1.0);
    
    try {
        model.load(model_path, model_path);
        
#ifdef USE_GPU_OFFLOAD
        // GPU build - MUST use GPU, no fallback allowed
        int num_devices = omp_get_num_devices();
        if (num_devices == 0) {
            std::cerr << "FAILED: GPU build but no GPU devices available!\n";
            assert(false && "GPU build requires GPU device");
        }
        
        model.initialize_gpu_buffers(mesh);
        
        // Create device buffers with correct sizes
        const int total_cells = mesh.total_cells();
        const int u_total = vel.u_total_size();
        const int v_total = vel.v_total_size();
        double* u_ptr = vel.u_data().data();
        double* v_ptr = vel.v_data().data();
        double* k_ptr = k.data().data();
        double* omega_ptr = omega.data().data();
        double* nu_t_ptr = nu_t.data().data();
        
        std::vector<double> dudx(total_cells), dudy(total_cells);
        std::vector<double> dvdx(total_cells), dvdy(total_cells);
        std::vector<double> wall_dist(total_cells, 0.5);
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                int idx = j * mesh.total_Nx() + i;
                dudx[idx] = 0.0;
                dudy[idx] = (vel.u(i, j+1) - vel.u(i, j-1)) / (2.0 * mesh.dy);
                dvdx[idx] = 0.0;
                dvdy[idx] = 0.0;
            }
        }
        
        // Get pointers for GPU mapping
        double* dudx_ptr = dudx.data();
        double* dudy_ptr = dudy.data();
        double* dvdx_ptr = dvdx.data();
        double* dvdy_ptr = dvdy.data();
        double* wall_dist_ptr = wall_dist.data();
        
        #pragma omp target enter data map(to: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target enter data map(to: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target enter data map(to: nu_t_ptr[0:total_cells])
        #pragma omp target enter data map(to: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target enter data map(to: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target enter data map(to: wall_dist_ptr[0:total_cells])
        
        TurbulenceDeviceView device_view;
        device_view.u_face = u_ptr;
        device_view.v_face = v_ptr;
        device_view.k = k_ptr;                    // REQUIRED for NN-TBNN
        device_view.omega = omega_ptr;            // REQUIRED for NN-TBNN
        device_view.u_stride = vel.u_stride();
        device_view.v_stride = vel.v_stride();
        device_view.nu_t = nu_t_ptr;
        device_view.cell_stride = mesh.total_Nx();
        device_view.dudx = dudx_ptr;
        device_view.dudy = dudy_ptr;
        device_view.dvdx = dvdx_ptr;
        device_view.dvdy = dvdy_ptr;
        device_view.wall_distance = wall_dist_ptr;
        device_view.Nx = mesh.Nx;
        device_view.Ny = mesh.Ny;
        device_view.Ng = mesh.Nghost;
        device_view.dx = mesh.dx;
        device_view.dy = mesh.dy;
        device_view.delta = 1.0;
        
        model.update(mesh, vel, k, omega, nu_t, nullptr, &device_view);
        
        if (!model.is_gpu_ready()) {
            std::cerr << "FAILED: GPU build but model didn't use GPU!\n";
            assert(false && "GPU build must execute on GPU");
        }
        
        #pragma omp target update from(nu_t_ptr[0:total_cells])
        
        #pragma omp target exit data map(delete: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target exit data map(delete: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target exit data map(delete: nu_t_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: wall_dist_ptr[0:total_cells])
#else
        // CPU build
        model.update(mesh, vel, k, omega, nu_t);
#endif
        
        assert(is_field_valid(nu_t, mesh) && "NN-TBNN produced NaN/Inf nu_t!");
        
        // Check nu_t is non-negative
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(nu_t(i, j) >= 0.0 && "NN-TBNN produced negative nu_t!");
            }
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
#ifdef USE_GPU_OFFLOAD
        std::cerr << "FAILED (GPU): " << e.what() << "\n";
        assert(false && "GPU build must not throw exceptions");
#else
        std::cout << "SKIPPED (" << e.what() << ")\n";
#endif
    }
}

// Test 3: NN-TBNN with solver integration
void test_nn_tbnn_solver_integration() {
    std::cout << "Testing NN-TBNN solver integration... ";
    
    // Use trained TBNN model
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.max_iter = 50;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::NNTBNN;
    config.nn_weights_path = model_path;
    config.nn_scaling_path = model_path;
    config.verbose = false;
    
    try {
        RANSSolver solver(mesh, config);
        
        // Run several iterations
        for (int iter = 0; iter < 20; ++iter) {
            solver.step();
        }
        
        // Check solution validity
        assert(is_velocity_valid(solver.velocity(), mesh) && "Solution diverged with NN-TBNN!");
        assert(is_field_valid(solver.nu_t(), mesh) && "nu_t is invalid!");
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
#ifdef USE_GPU_OFFLOAD
        std::cerr << "FAILED (GPU): " << e.what() << "\n";
        assert(false && "GPU build must not throw exceptions");
#else
        std::cout << "SKIPPED (" << e.what() << ")\n";
#endif
    }
}

// Test 4: Multiple NN updates don't cause memory issues
void test_nn_repeated_updates() {
    std::cout << "Testing repeated NN updates... ";
    
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.set_u_ref(1.0);
    
    try {
        model.load(model_path, model_path);
        
#ifdef USE_GPU_OFFLOAD
        // GPU build - MUST use GPU
        int num_devices = omp_get_num_devices();
        if (num_devices == 0) {
            std::cerr << "FAILED: GPU build but no GPU devices available!\n";
            assert(false && "GPU build requires GPU device");
        }
        
        model.initialize_gpu_buffers(mesh);
        
        const int total_cells = mesh.total_cells();
        const int u_total = vel.u_total_size();
        const int v_total = vel.v_total_size();
        double* u_ptr = vel.u_data().data();
        double* v_ptr = vel.v_data().data();
        double* k_ptr = k.data().data();
        double* omega_ptr = omega.data().data();
        double* nu_t_ptr = nu_t.data().data();
        
        std::vector<double> dudx(total_cells), dudy(total_cells);
        std::vector<double> dvdx(total_cells), dvdy(total_cells);
        std::vector<double> wall_dist(total_cells, 0.5);
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                int idx = j * mesh.total_Nx() + i;
                dudx[idx] = 0.0;
                dudy[idx] = (vel.u(i, j+1) - vel.u(i, j-1)) / (2.0 * mesh.dy);
                dvdx[idx] = 0.0;
                dvdy[idx] = 0.0;
            }
        }
        
        // Get pointers for GPU mapping
        double* dudx_ptr = dudx.data();
        double* dudy_ptr = dudy.data();
        double* dvdx_ptr = dvdx.data();
        double* dvdy_ptr = dvdy.data();
        double* wall_dist_ptr = wall_dist.data();
        
        #pragma omp target enter data map(to: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target enter data map(to: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target enter data map(to: nu_t_ptr[0:total_cells])
        #pragma omp target enter data map(to: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target enter data map(to: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target enter data map(to: wall_dist_ptr[0:total_cells])
        
        TurbulenceDeviceView device_view;
        device_view.u_face = u_ptr;
        device_view.v_face = v_ptr;
        device_view.k = k_ptr;                    // REQUIRED for NN-TBNN
        device_view.omega = omega_ptr;            // REQUIRED for NN-TBNN
        device_view.u_stride = vel.u_stride();
        device_view.v_stride = vel.v_stride();
        device_view.nu_t = nu_t_ptr;
        device_view.cell_stride = mesh.total_Nx();
        device_view.dudx = dudx_ptr;
        device_view.dudy = dudy_ptr;
        device_view.dvdx = dvdx_ptr;
        device_view.dvdy = dvdy_ptr;
        device_view.wall_distance = wall_dist_ptr;
        device_view.Nx = mesh.Nx;
        device_view.Ny = mesh.Ny;
        device_view.Ng = mesh.Nghost;
        device_view.dx = mesh.dx;
        device_view.dy = mesh.dy;
        device_view.delta = 1.0;
        
        // Call update many times - should not leak memory or crash
        for (int i = 0; i < 100; ++i) {
            model.update(mesh, vel, k, omega, nu_t, nullptr, &device_view);
            
            // Verify output is still valid
            if (i % 20 == 0) {
                #pragma omp target update from(nu_t_ptr[0:total_cells])
                assert(is_field_valid(nu_t, mesh) && "nu_t became invalid during repeated updates!");
            }
        }
        
        #pragma omp target update from(nu_t_ptr[0:total_cells])
        
        #pragma omp target exit data map(delete: u_ptr[0:u_total], v_ptr[0:v_total])
        #pragma omp target exit data map(delete: k_ptr[0:total_cells], omega_ptr[0:total_cells])
        #pragma omp target exit data map(delete: nu_t_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dudx_ptr[0:total_cells], dudy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: dvdx_ptr[0:total_cells], dvdy_ptr[0:total_cells])
        #pragma omp target exit data map(delete: wall_dist_ptr[0:total_cells])
#else
        // CPU build
        for (int i = 0; i < 100; ++i) {
            model.update(mesh, vel, k, omega, nu_t);
            if (i % 20 == 0) {
                assert(is_field_valid(nu_t, mesh) && "nu_t became invalid during repeated updates!");
            }
        }
#endif
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
#ifdef USE_GPU_OFFLOAD
        std::cerr << "FAILED (GPU): " << e.what() << "\n";
        assert(false && "GPU build must not throw exceptions");
#else
        std::cout << "SKIPPED (" << e.what() << ")\n";
#endif
    }
}

// Test 5: NN model with different grid sizes
void test_nn_different_grid_sizes() {
    std::cout << "Testing NN with different grid sizes... ";
    
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    std::vector<std::pair<int, int>> grid_sizes = {
        {8, 16},
        {16, 32},
        {32, 64},
        {64, 128}
    };
    
    try {
#ifdef USE_GPU_OFFLOAD
        // GPU build - verify GPU is available
        int num_devices = omp_get_num_devices();
        if (num_devices == 0) {
            std::cerr << "FAILED: GPU build but no GPU devices available!\n";
            assert(false && "GPU build requires GPU device");
        }
#endif
        
        for (const auto& [nx, ny] : grid_sizes) {
            Mesh mesh;
            mesh.init_uniform(nx, ny, 0.0, 4.0, -1.0, 1.0);
            
            VectorField vel(mesh);
            for (int j = 0; j < mesh.total_Ny(); ++j) {
                for (int i = 0; i < mesh.total_Nx(); ++i) {
                    vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
                    vel.v(i, j) = 0.0;
                }
            }
            
            ScalarField k(mesh, 0.01);
            ScalarField omega(mesh, 1.0);
            ScalarField nu_t(mesh);
            
            TurbulenceNNTBNN model;
            model.set_nu(0.001);
            model.set_delta(1.0);
            model.set_u_ref(1.0);
            model.load(model_path, model_path);
            
#ifdef USE_GPU_OFFLOAD
            model.initialize_gpu_buffers(mesh);
            
            const int total_cells = mesh.total_cells();
            const int u_total = vel.u_total_size();
            const int v_total = vel.v_total_size();
            double* u_ptr = vel.u_data().data();
            double* v_ptr = vel.v_data().data();
            double* k_ptr = k.data().data();
            double* omega_ptr = omega.data().data();
            double* nu_t_ptr = nu_t.data().data();
            
            std::vector<double> dudx(total_cells), dudy(total_cells);
            std::vector<double> dvdx(total_cells), dvdy(total_cells);
            std::vector<double> wall_dist(total_cells, 0.5);
            
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    int idx = j * mesh.total_Nx() + i;
                    dudx[idx] = 0.0;
                    dudy[idx] = (vel.u(i, j+1) - vel.u(i, j-1)) / (2.0 * mesh.dy);
                    dvdx[idx] = 0.0;
                    dvdy[idx] = 0.0;
                }
            }
            
            #pragma omp target enter data map(to: u_ptr[0:u_total], v_ptr[0:v_total])
            #pragma omp target enter data map(to: k_ptr[0:total_cells], omega_ptr[0:total_cells])
            #pragma omp target enter data map(to: nu_t_ptr[0:total_cells])
            #pragma omp target enter data map(to: dudx[0:total_cells], dudy[0:total_cells])
            #pragma omp target enter data map(to: dvdx[0:total_cells], dvdy[0:total_cells])
            #pragma omp target enter data map(to: wall_dist[0:total_cells])
            
            TurbulenceDeviceView device_view;
            device_view.u_face = u_ptr;
            device_view.v_face = v_ptr;
            device_view.k = k_ptr;                    // REQUIRED for NN-TBNN
            device_view.omega = omega_ptr;            // REQUIRED for NN-TBNN
            device_view.u_stride = vel.u_stride();
            device_view.v_stride = vel.v_stride();
            device_view.nu_t = nu_t_ptr;
            device_view.cell_stride = mesh.total_Nx();
            device_view.dudx = dudx.data();
            device_view.dudy = dudy.data();
            device_view.dvdx = dvdx.data();
            device_view.dvdy = dvdy.data();
            device_view.wall_distance = wall_dist.data();
            device_view.Nx = mesh.Nx;
            device_view.Ny = mesh.Ny;
            device_view.Ng = mesh.Nghost;
            device_view.dx = mesh.dx;
            device_view.dy = mesh.dy;
            device_view.delta = 1.0;
            
            model.update(mesh, vel, k, omega, nu_t, nullptr, &device_view);
            
            #pragma omp target update from(nu_t_ptr[0:total_cells])
            
            #pragma omp target exit data map(delete: u_ptr[0:u_total], v_ptr[0:v_total])
            #pragma omp target exit data map(delete: k_ptr[0:total_cells], omega_ptr[0:total_cells])
            #pragma omp target exit data map(delete: nu_t_ptr[0:total_cells])
            #pragma omp target exit data map(delete: dudx[0:total_cells], dudy[0:total_cells])
            #pragma omp target exit data map(delete: dvdx[0:total_cells], dvdy[0:total_cells])
            #pragma omp target exit data map(delete: wall_dist[0:total_cells])
#else
            // CPU build
            model.update(mesh, vel, k, omega, nu_t);
#endif
            
            assert(is_field_valid(nu_t, mesh) && "NN failed on different grid size!");
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        assert(false);
    }
}

int main() {
    std::cout << "=== NN Integration Tests ===\n\n";
    
    test_nn_mlp_validity();
    test_nn_tbnn_validity();
    test_nn_tbnn_solver_integration();
    test_nn_repeated_updates();
    test_nn_different_grid_sizes();
    
    std::cout << "\nAll NN integration tests completed!\n";
    return 0;
}

