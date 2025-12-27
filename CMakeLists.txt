cmake_minimum_required(VERSION 3.16)
project(nn_cfd VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  add_custom_target(clangd_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}/compile_commands.json
    BYPRODUCTS ${CMAKE_SOURCE_DIR}/compile_commands.json
    COMMENT "Copy compile_commands.json to source dir for clangd"
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GPU offload option (must be set before finding OpenMP)
option(USE_GPU_OFFLOAD "Enable OpenMP GPU target offloading" OFF)

# GPU profiling options
option(GPU_PROFILE_TRANSFERS "Profile GPU data transfers (verbose output)" OFF)
option(GPU_PROFILE_KERNELS "Enable NVTX markers for kernel profiling" OFF)

# Suppress unknown pragma warnings in CPU builds (allows OpenMP pragmas to be ignored silently)
option(SUPPRESS_UNKNOWN_OMP_PRAGMAS "Silence -Wunknown-pragmas for OpenMP pragmas in CPU build" ON)

# Find OpenMP ONLY if GPU offload is enabled
# CPU builds (USE_GPU_OFFLOAD=OFF) are OpenMP-free to enforce compile-time separation
if(USE_GPU_OFFLOAD)
    find_package(OpenMP REQUIRED)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "GPU offload requires OpenMP support")
    endif()
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# GPU offloading configuration
if(USE_GPU_OFFLOAD)
    add_definitions(-DUSE_GPU_OFFLOAD)
    message(STATUS "GPU offloading ENABLED (GPU required at runtime)")
    
    # Detect compiler and set appropriate flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
        # NVIDIA HPC SDK - best for NVIDIA GPUs
        # No -gpu=ccXX specified - will auto-detect at runtime or use PTX JIT
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu")
        message(STATUS "Using NVIDIA HPC SDK GPU offloading")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Clang/LLVM with CUDA backend (auto-detect GPU arch)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda")
        message(STATUS "Using Clang OpenMP GPU offloading")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # GCC with nvptx offloading
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -foffload=nvptx-none -foffload=-lm -fno-lto")
        message(STATUS "Using GCC OpenMP GPU offloading")
    else()
        message(WARNING "Unknown compiler for GPU offloading: ${CMAKE_CXX_COMPILER_ID}")
    endif()
else()
    message(STATUS "CPU build (OpenMP disabled, single-threaded)")
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources
set(LIB_SOURCES
    src/mesh.cpp
    src/fields.cpp
    src/poisson_solver.cpp
    src/poisson_solver_multigrid.cpp
    src/solver.cpp
    src/timing.cpp
    src/config.cpp
    src/features.cpp
    src/nn_core.cpp
    src/turbulence_baseline.cpp
    src/turbulence_gep.cpp
    src/turbulence_nn_mlp.cpp
    src/turbulence_nn_tbnn.cpp
    src/turbulence_transport.cpp
    src/turbulence_earsm.cpp
    src/gpu_kernels.cpp
)

# Core library
add_library(nn_cfd_core ${LIB_SOURCES})

# Add GPU-only sources and link OpenMP when GPU offload is enabled
if(USE_GPU_OFFLOAD)
    target_sources(nn_cfd_core PRIVATE src/gpu_init.cpp)
    target_link_libraries(nn_cfd_core OpenMP::OpenMP_CXX)
    # Propagate USE_GPU_OFFLOAD to all targets that link this library
    # This ensures tests compile with the same GPU settings as the library
    target_compile_definitions(nn_cfd_core PUBLIC USE_GPU_OFFLOAD)
endif()

# Suppress unknown pragma warnings in CPU builds
if(NOT USE_GPU_OFFLOAD AND SUPPRESS_UNKNOWN_OMP_PRAGMAS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(nn_cfd_core PRIVATE -Wno-unknown-pragmas)
        message(STATUS "Suppressing unknown pragma warnings for CPU build")
    endif()
endif()

# GPU profiling compile definitions
if(GPU_PROFILE_TRANSFERS)
    target_compile_definitions(nn_cfd_core PRIVATE GPU_PROFILE_TRANSFERS)
    message(STATUS "GPU transfer profiling ENABLED")
endif()

if(GPU_PROFILE_KERNELS)
    target_compile_definitions(nn_cfd_core PUBLIC GPU_PROFILE_KERNELS)
    message(STATUS "GPU kernel profiling with NVTX ENABLED")
    
    # Add NVTX include paths for NVHPC compiler
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
        # NVHPC stores NVTX headers in include-stdpar
        get_filename_component(NVHPC_ROOT ${CMAKE_CXX_COMPILER} DIRECTORY)
        get_filename_component(NVHPC_ROOT ${NVHPC_ROOT} DIRECTORY)
        set(NVTX_INCLUDE_DIR "${NVHPC_ROOT}/compilers/include-stdpar")
        if(EXISTS "${NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
            target_include_directories(nn_cfd_core PUBLIC ${NVTX_INCLUDE_DIR})
            message(STATUS "  NVTX headers found: ${NVTX_INCLUDE_DIR}")
        endif()
    endif()
    
    # Try to find NVTX library (optional)
    find_library(NVTX_LIB NAMES nvtx3 nvToolsExt PATHS ENV LD_LIBRARY_PATH)
    if(NVTX_LIB)
        target_link_libraries(nn_cfd_core ${NVTX_LIB})
        message(STATUS "  NVTX library found: ${NVTX_LIB}")
    else()
        message(STATUS "  NVTX library not found - using header-only implementation")
    endif()
endif()

# Main application executables
add_executable(channel app/main_channel.cpp)
target_link_libraries(channel nn_cfd_core)

add_executable(periodic_hills app/main_periodic_hills.cpp)
target_link_libraries(periodic_hills nn_cfd_core)

# CPU vs GPU comparison tool (for validation)
add_executable(compare_channel_cpu_gpu app/compare_channel_cpu_gpu.cpp)
target_link_libraries(compare_channel_cpu_gpu nn_cfd_core)

# Profiling driver (limited steps for nsys analysis)
add_executable(profile_solver app/profile_solver.cpp)
target_link_libraries(profile_solver nn_cfd_core)

# Poisson warm-start profiling (3 V-cycles, 50 steps)
add_executable(profile_poisson_warmstart app/profile_poisson_warmstart.cpp)
target_link_libraries(profile_poisson_warmstart nn_cfd_core)

# Large-scale grid profiling (2000x2000, 10 steps, 30 V-cycles)
add_executable(profile_large_grid app/profile_large_grid.cpp)
target_link_libraries(profile_large_grid nn_cfd_core)

# Tests (optional)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_mesh tests/test_mesh.cpp)
    target_link_libraries(test_mesh nn_cfd_core)
    add_test(NAME MeshTest COMMAND test_mesh)
    
    add_executable(test_poisson tests/test_poisson.cpp)
    target_link_libraries(test_poisson nn_cfd_core)
    add_test(NAME PoissonTest COMMAND test_poisson)
    
    add_executable(test_solver tests/test_solver.cpp)
    target_link_libraries(test_solver nn_cfd_core)
    add_test(NAME SolverTest COMMAND test_solver)

    add_executable(test_3d_solver tests/test_3d_solver.cpp)
    target_link_libraries(test_3d_solver nn_cfd_core)
    add_test(NAME Solver3DTest COMMAND test_3d_solver)

    add_executable(test_2d_3d_comparison tests/test_2d_3d_comparison.cpp)
    target_link_libraries(test_2d_3d_comparison nn_cfd_core)
    add_test(NAME Comparison2D3DTest COMMAND test_2d_3d_comparison)

    add_executable(test_features tests/test_features.cpp)
    target_link_libraries(test_features nn_cfd_core)
    add_test(NAME FeaturesTest COMMAND test_features)
    
    add_executable(test_nn_core tests/test_nn_core.cpp)
    target_link_libraries(test_nn_core nn_cfd_core)
    add_test(NAME NNCoreTest COMMAND test_nn_core)
    
    add_executable(test_turbulence tests/test_turbulence.cpp)
    target_link_libraries(test_turbulence nn_cfd_core)
    add_test(NAME TurbulenceTest COMMAND test_turbulence)
    # Turbulence models now use unified persistent mapping - GPU enabled
    
    add_executable(test_stability tests/test_stability.cpp)
    target_link_libraries(test_stability nn_cfd_core)
    add_test(NAME StabilityTest COMMAND test_stability)
    
    add_executable(test_nn_integration tests/test_nn_integration.cpp)
    target_link_libraries(test_nn_integration nn_cfd_core)
    add_test(NAME NNIntegrationTest COMMAND test_nn_integration)
    
    add_executable(test_backend_execution tests/test_backend_execution.cpp)
    target_link_libraries(test_backend_execution nn_cfd_core)
    add_test(NAME BackendExecutionTest COMMAND test_backend_execution)
    
    add_executable(test_cpu_gpu_consistency tests/test_cpu_gpu_consistency.cpp)
    target_link_libraries(test_cpu_gpu_consistency nn_cfd_core)
    add_test(NAME ConsistencyTest COMMAND test_cpu_gpu_consistency)
    
    add_executable(test_solver_cpu_gpu tests/test_solver_cpu_gpu.cpp)
    target_link_libraries(test_solver_cpu_gpu nn_cfd_core)
    add_test(NAME SolverCPUGPUTest COMMAND test_solver_cpu_gpu)
    
    add_executable(test_divergence_all_bcs tests/test_divergence_all_bcs.cpp)
    target_link_libraries(test_divergence_all_bcs nn_cfd_core)
    add_test(NAME DivergenceAllBCsTest COMMAND test_divergence_all_bcs)
    
    add_executable(test_time_history_consistency tests/test_time_history_consistency.cpp)
    target_link_libraries(test_time_history_consistency nn_cfd_core)
    add_test(NAME TimeHistoryConsistencyTest COMMAND test_time_history_consistency)
    
    add_executable(test_physics_validation tests/test_physics_validation.cpp)
    target_link_libraries(test_physics_validation nn_cfd_core)
    add_test(NAME PhysicsValidationTest COMMAND test_physics_validation)
    
    # Taylor-Green vortex validation - verifies viscous decay and time integration
    add_executable(test_tg_validation tests/test_taylor_green.cpp)
    target_link_libraries(test_tg_validation nn_cfd_core)
    add_test(NAME TaylorGreenValidationTest COMMAND test_tg_validation)
    
    # Perturbed channel validation - comprehensive turbulence model testing (1000 steps on GPU)
    add_executable(test_perturbed_channel tests/test_perturbed_channel.cpp)
    target_link_libraries(test_perturbed_channel nn_cfd_core)
    add_test(NAME PerturbedChannelTest COMMAND test_perturbed_channel)
    
    # NaN/Inf guard test - verifies abort-on-NaN behavior
    add_executable(test_turbulence_guard tests/test_turbulence_guard.cpp)
    target_link_libraries(test_turbulence_guard nn_cfd_core)
    add_test(NAME NanInfGuardTest COMMAND test_turbulence_guard)
    
    # Turbulence feature tests - analytic validation of features, invariants, and model response
    add_executable(test_turbulence_features tests/test_turbulence_features.cpp)
    target_link_libraries(test_turbulence_features nn_cfd_core)
    add_test(NAME TurbulenceFeaturesTest COMMAND test_turbulence_features)

    # 3D Poisson CPU vs GPU comparison - verifies GPU implementation matches CPU exactly
    add_executable(test_poisson_cpu_gpu_3d tests/test_poisson_cpu_gpu_3d.cpp)
    target_link_libraries(test_poisson_cpu_gpu_3d nn_cfd_core)
    add_test(NAME PoissonCPUGPU3DTest COMMAND test_poisson_cpu_gpu_3d)

    # CPU vs GPU timestep comparison - finds where they diverge
    add_executable(test_cpu_gpu_timestep tests/test_cpu_gpu_timestep.cpp)
    target_link_libraries(test_cpu_gpu_timestep nn_cfd_core)
    add_test(NAME CPUGPUTimestepTest COMMAND test_cpu_gpu_timestep)
endif()

# Installation
install(TARGETS channel periodic_hills DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
