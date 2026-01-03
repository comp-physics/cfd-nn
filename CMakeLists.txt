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

# Hypre solver option (requires CUDA toolkit)
option(USE_HYPRE "Enable HYPRE PFMG Poisson solver" OFF)

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

# Hypre configuration - download and build from source with CUDA support
if(USE_HYPRE)
    message(STATUS "HYPRE solver ENABLED - downloading and building from source")

    # Enable CUDA language for hypre build
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "  CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")

    # Detect GPU architecture if not specified
    # Default to Ampere (A100=sm80). Use -DGPU_CC=90 for Hopper (H100)
    if(NOT DEFINED GPU_CC)
        set(GPU_CC "80" CACHE STRING "GPU compute capability (80=A100, 90=H100)")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${GPU_CC}" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    include(FetchContent)
    FetchContent_Declare(
        hypre
        GIT_REPOSITORY https://github.com/hypre-space/hypre.git
        GIT_TAG v2.31.0
        SOURCE_SUBDIR src
    )

    # Configure hypre build options before FetchContent_MakeAvailable
    # FORCE these options to ensure they take effect in the subproject
    # Note: hypre uses HYPRE_WITH_CUDA (not HYPRE_ENABLE_CUDA)
    set(HYPRE_WITH_CUDA ON CACHE BOOL "Enable CUDA support in HYPRE" FORCE)
    set(HYPRE_ENABLE_UNIFIED_MEMORY ON CACHE BOOL "Use unified memory for GPU-CPU sharing" FORCE)
    set(HYPRE_ENABLE_GPU_PROFILING OFF CACHE BOOL "Disable GPU profiling" FORCE)
    set(HYPRE_CUDA_SM "${GPU_CC}" CACHE STRING "CUDA SM architecture" FORCE)

    # Disable features we don't need
    set(HYPRE_WITH_MPI OFF CACHE BOOL "Disable MPI (single-node only)" FORCE)
    set(HYPRE_WITH_OPENMP OFF CACHE BOOL "Use CUDA instead of OpenMP for GPU" FORCE)
    set(HYPRE_ENABLE_SHARED OFF CACHE BOOL "Build static library" FORCE)
    set(HYPRE_ENABLE_MIXEDINT OFF CACHE BOOL "Use consistent integer types" FORCE)
    set(HYPRE_PRINT_ERRORS ON CACHE BOOL "Print error messages" FORCE)
    set(HYPRE_ENABLE_BIGINT OFF CACHE BOOL "Use 32-bit integers" FORCE)

    FetchContent_MakeAvailable(hypre)

    message(STATUS "  HYPRE source dir: ${hypre_SOURCE_DIR}")
    message(STATUS "  HYPRE binary dir: ${hypre_BINARY_DIR}")
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
        # GPU compute capability: cc80=A100, cc90=H100 (default: cc80 for A100)
        if(NOT DEFINED GPU_CC)
            set(GPU_CC "80" CACHE STRING "GPU compute capability (80=A100, 90=H100)")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu -gpu=cc${GPU_CC}")
        # Add -cuda for OpenMP+CUDA interoperability
        # Required for: hypre CUDA backend, FFT solver CUDA kernels
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -cuda")
        message(STATUS "  Added -cuda for OpenMP+CUDA interop")
        message(STATUS "Using NVIDIA HPC SDK GPU offloading (cc${GPU_CC})")
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

# Add HYPRE Poisson solver when enabled
if(USE_HYPRE)
    target_sources(nn_cfd_core PRIVATE src/poisson_solver_hypre.cpp)
    target_include_directories(nn_cfd_core PUBLIC ${hypre_SOURCE_DIR}/src)
    target_include_directories(nn_cfd_core PUBLIC ${hypre_BINARY_DIR})
    target_link_libraries(nn_cfd_core HYPRE)
    # Link CUDA runtime for hypre's CUDA backend
    target_link_libraries(nn_cfd_core CUDA::cudart)
    target_compile_definitions(nn_cfd_core PUBLIC USE_HYPRE)
    message(STATUS "  HYPRE Poisson solver added to nn_cfd_core")
endif()

# Add FFT-hybrid Poisson solver (for periodic x/z cases)
# Requires GPU offload and cuFFT
if(USE_GPU_OFFLOAD)
    # cuFFT is part of CUDA toolkit
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_sources(nn_cfd_core PRIVATE
            src/poisson_solver_fft.cpp
            src/poisson_solver_fft1d.cpp)
        target_link_libraries(nn_cfd_core CUDA::cufft CUDA::cusparse CUDA::cudart)
        target_compile_definitions(nn_cfd_core PUBLIC USE_FFT_POISSON)
        message(STATUS "  FFT Poisson solvers added (2D FFT + 1D FFT + cuSPARSE)")
    endif()
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

# Comprehensive kernel-level profiling (all turbulence models + 3D)
add_executable(profile_kernels app/profile_kernels.cpp)
target_link_libraries(profile_kernels nn_cfd_core)

# 3D applications
add_executable(duct app/main_duct.cpp)
target_link_libraries(duct nn_cfd_core)

add_executable(taylor_green_3d app/main_taylor_green_3d.cpp)
target_link_libraries(taylor_green_3d nn_cfd_core)

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

    add_executable(test_poisson_solvers tests/test_poisson_solvers.cpp)
    target_link_libraries(test_poisson_solvers nn_cfd_core)
    add_test(NAME PoissonSolversTest COMMAND test_poisson_solvers)

    add_executable(test_solver tests/test_solver.cpp)
    target_link_libraries(test_solver nn_cfd_core)
    add_test(NAME SolverTest COMMAND test_solver)

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

    # Fast 3D validation tests - quick smoke tests (~5s)
    add_executable(test_3d_quick_validation tests/test_3d_quick_validation.cpp)
    target_link_libraries(test_3d_quick_validation nn_cfd_core)
    add_test(NAME Quick3DValidationTest COMMAND test_3d_quick_validation)

    # Fast 3D Poiseuille tests - analytical validation (~10s)
    add_executable(test_3d_poiseuille_fast tests/test_3d_poiseuille_fast.cpp)
    target_link_libraries(test_3d_poiseuille_fast nn_cfd_core)
    add_test(NAME Fast3DPoiseuilleTest COMMAND test_3d_poiseuille_fast)

    # 3D boundary condition tests (~5s)
    add_executable(test_3d_bc_application tests/test_3d_bc_application.cpp)
    target_link_libraries(test_3d_bc_application nn_cfd_core)
    add_test(NAME BC3DApplicationTest COMMAND test_3d_bc_application)

    # CPU/GPU bitwise comparison - enforces code sharing paradigm (~15s)
    add_executable(test_cpu_gpu_bitwise tests/test_cpu_gpu_bitwise.cpp)
    target_link_libraries(test_cpu_gpu_bitwise nn_cfd_core)
    add_test(NAME CPUGPUBitwiseTest COMMAND test_cpu_gpu_bitwise)

    # 3D gradient tests - verifies gradient computation (~5s)
    add_executable(test_3d_gradients tests/test_3d_gradients.cpp)
    target_link_libraries(test_3d_gradients nn_cfd_core)
    add_test(NAME Gradients3DTest COMMAND test_3d_gradients)

    # 3D w-velocity tests - tests the 3D-specific component (~5s)
    add_executable(test_3d_w_velocity tests/test_3d_w_velocity.cpp)
    target_link_libraries(test_3d_w_velocity nn_cfd_core)
    add_test(NAME WVelocity3DTest COMMAND test_3d_w_velocity)

    # 3D Taylor-Green vortex - verifies 3D viscous decay and time integration
    add_executable(test_taylor_green_3d tests/test_taylor_green_3d.cpp)
    target_link_libraries(test_taylor_green_3d nn_cfd_core)
    add_test(NAME TaylorGreen3DTest COMMAND test_taylor_green_3d)

    # All turbulence models smoke test - verifies all 10 models run without crashing
    add_executable(test_all_turbulence_models_smoke tests/test_all_turbulence_models_smoke.cpp)
    target_link_libraries(test_all_turbulence_models_smoke nn_cfd_core)
    add_test(NAME AllTurbulenceModelsSmokeTest COMMAND test_all_turbulence_models_smoke)

    # Transport equation realizability - verifies k>0, omega>0 over long runs
    add_executable(test_transport_realizability tests/test_transport_realizability.cpp)
    target_link_libraries(test_transport_realizability nn_cfd_core)
    add_test(NAME TransportRealizabilityTest COMMAND test_transport_realizability)

    # EARSM trace-free constraint - verifies b_xx + b_yy = 0
    add_executable(test_earsm_trace_free tests/test_earsm_trace_free.cpp)
    target_link_libraries(test_earsm_trace_free nn_cfd_core)
    add_test(NAME EARSMTraceFreeTest COMMAND test_earsm_trace_free)

    # GPU utilization test - validates compute runs on GPU for GPU builds
    add_executable(test_gpu_utilization tests/test_gpu_utilization.cpp)
    target_link_libraries(test_gpu_utilization nn_cfd_core)
    add_test(NAME GPUUtilizationTest COMMAND test_gpu_utilization)

    # HYPRE all BC configurations test
    if(USE_HYPRE)
        add_executable(test_hypre_all_bcs tests/test_hypre_all_bcs.cpp)
        target_link_libraries(test_hypre_all_bcs nn_cfd_core)
        add_test(NAME HypreAllBCsTest COMMAND test_hypre_all_bcs)

        # HYPRE validation test - compares HYPRE vs Multigrid
        add_executable(test_hypre_validation tests/test_hypre_validation.cpp)
        target_link_libraries(test_hypre_validation nn_cfd_core)
        add_test(NAME HypreValidationTest COMMAND test_hypre_validation)
    endif()

    # Poisson solver selection state machine test - prevents selection logic drift
    add_executable(test_poisson_selection tests/test_poisson_selection.cpp)
    target_link_libraries(test_poisson_selection nn_cfd_core)
    add_test(NAME PoissonSelectionTest COMMAND test_poisson_selection)

    # FFT1D dedicated validation test - forces FFT1D selection + correctness check
    add_executable(test_fft1d_validation tests/test_fft1d_validation.cpp)
    target_link_libraries(test_fft1d_validation nn_cfd_core)
    add_test(NAME FFT1DValidationTest COMMAND test_fft1d_validation)

    # Endurance stability test - catches NaN-after-N-steps class bugs
    add_executable(test_endurance_stability tests/test_endurance_stability.cpp)
    target_link_libraries(test_endurance_stability nn_cfd_core)
    add_test(NAME EnduranceStabilityTest COMMAND test_endurance_stability)

    # Manufactured solution Poisson correctness test - catches "solver runs but wrong"
    add_executable(test_poisson_manufactured tests/test_poisson_manufactured.cpp)
    target_link_libraries(test_poisson_manufactured nn_cfd_core)
    add_test(NAME PoissonManufacturedTest COMMAND test_poisson_manufactured)

    # Dirichlet/mixed-BC Poisson test - validates BC handling
    add_executable(test_poisson_dirichlet_mixed tests/test_poisson_dirichlet_mixed.cpp)
    target_link_libraries(test_poisson_dirichlet_mixed nn_cfd_core)
    add_test(NAME PoissonDirichletMixedTest COMMAND test_poisson_dirichlet_mixed)

    # Repeatability envelope test - catches race conditions and nondeterminism
    add_executable(test_repeatability tests/test_repeatability.cpp)
    target_link_libraries(test_repeatability nn_cfd_core)
    add_test(NAME RepeatabilityTest COMMAND test_repeatability)

    # Performance regression sentinel - catches catastrophic slowdowns
    add_executable(test_perf_sentinel tests/test_perf_sentinel.cpp)
    target_link_libraries(test_perf_sentinel nn_cfd_core)
    add_test(NAME PerfSentinelTest COMMAND test_perf_sentinel)

    # Stretched/anisotropic grid test - validates MG/HYPRE on high aspect ratio cells
    add_executable(test_poisson_stretched_grid tests/test_poisson_stretched_grid.cpp)
    target_link_libraries(test_poisson_stretched_grid nn_cfd_core)
    add_test(NAME PoissonStretchedGridTest COMMAND test_poisson_stretched_grid)

    # Nullspace/gauge handling test - validates singular Poisson (pure Neumann/periodic)
    add_executable(test_poisson_nullspace tests/test_poisson_nullspace.cpp)
    target_link_libraries(test_poisson_nullspace nn_cfd_core)
    add_test(NAME PoissonNullspaceTest COMMAND test_poisson_nullspace)

    # Cross-solver consistency test - validates all solvers produce equivalent results
    add_executable(test_poisson_cross_solver tests/test_poisson_cross_solver.cpp)
    target_link_libraries(test_poisson_cross_solver nn_cfd_core)
    add_test(NAME PoissonCrossSolverTest COMMAND test_poisson_cross_solver)

    # Projection method invariants test - validates time-stepper coupling
    add_executable(test_projection_invariants tests/test_projection_invariants.cpp)
    target_link_libraries(test_projection_invariants nn_cfd_core)
    add_test(NAME ProjectionInvariantsTest COMMAND test_projection_invariants)
endif()

# Installation
install(TARGETS channel duct taylor_green_3d DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
