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
    message(STATUS "HYPRE solver ENABLED")

    # Enable CUDA language for hypre build
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "  CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")

    # Detect GPU architecture if not specified
    # Priority: 1. User-specified -DGPU_CC=XX, 2. Auto-detect via script, 3. Default to 80
    if(NOT DEFINED GPU_CC)
        # Try auto-detection using our detection script
        execute_process(
            COMMAND ${CMAKE_SOURCE_DIR}/scripts/detect_gpu_cc.sh
            OUTPUT_VARIABLE DETECTED_GPU_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE DETECT_RESULT
        )
        if(DETECT_RESULT EQUAL 0 AND DETECTED_GPU_CC MATCHES "^[0-9]+$")
            set(GPU_CC "${DETECTED_GPU_CC}" CACHE STRING "GPU compute capability (auto-detected)")
            message(STATUS "  GPU_CC auto-detected: ${GPU_CC}")
        else()
            set(GPU_CC "80" CACHE STRING "GPU compute capability (80=A100, 90=H100)")
            message(STATUS "  GPU_CC fallback: ${GPU_CC} (auto-detection failed)")
        endif()
    else()
        message(STATUS "  GPU_CC user-specified: ${GPU_CC}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${GPU_CC}" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Check for cached HYPRE build from CI
    # CI caches _deps/hypre-src and _deps/hypre-build but removes hypre-subbuild
    # (which contains CMake internals with absolute paths). Without hypre-subbuild,
    # FetchContent thinks HYPRE needs to be rebuilt even when libHYPRE.a exists.
    # This logic detects cached builds and creates an IMPORTED target to skip rebuild.
    set(HYPRE_CACHED_SOURCE "${CMAKE_BINARY_DIR}/_deps/hypre-src")
    set(HYPRE_CACHED_BUILD "${CMAKE_BINARY_DIR}/_deps/hypre-build")
    set(HYPRE_CACHED_LIB "${HYPRE_CACHED_BUILD}/libHYPRE.a")

    if(EXISTS "${HYPRE_CACHED_LIB}" AND EXISTS "${HYPRE_CACHED_SOURCE}/src/HYPRE.h")
        message(STATUS "  Found cached HYPRE build - skipping rebuild")
        message(STATUS "    Library: ${HYPRE_CACHED_LIB}")
        message(STATUS "    Source: ${HYPRE_CACHED_SOURCE}")

        # Create IMPORTED target to use the cached library
        # We can't use find_package(HYPRE CONFIG) because the generated HYPRETargets.cmake
        # contains absolute paths from the original build, which break when cache is
        # restored to a different runner path in CI.
        add_library(HYPRE STATIC IMPORTED GLOBAL)
        set_target_properties(HYPRE PROPERTIES
            IMPORTED_LOCATION "${HYPRE_CACHED_LIB}"
        )

        # HYPRE headers are scattered across many subdirectories
        # This list matches what HYPRE's own CMake exports in HYPRETargets.cmake
        set(HYPRE_INCLUDE_DIRS
            "${HYPRE_CACHED_SOURCE}/src"
            "${HYPRE_CACHED_SOURCE}/src/blas"
            "${HYPRE_CACHED_SOURCE}/src/lapack"
            "${HYPRE_CACHED_SOURCE}/src/utilities"
            "${HYPRE_CACHED_SOURCE}/src/multivector"
            "${HYPRE_CACHED_SOURCE}/src/krylov"
            "${HYPRE_CACHED_SOURCE}/src/seq_mv"
            "${HYPRE_CACHED_SOURCE}/src/seq_block_mv"
            "${HYPRE_CACHED_SOURCE}/src/parcsr_mv"
            "${HYPRE_CACHED_SOURCE}/src/parcsr_block_mv"
            "${HYPRE_CACHED_SOURCE}/src/distributed_matrix"
            "${HYPRE_CACHED_SOURCE}/src/IJ_mv"
            "${HYPRE_CACHED_SOURCE}/src/matrix_matrix"
            "${HYPRE_CACHED_SOURCE}/src/distributed_ls"
            "${HYPRE_CACHED_SOURCE}/src/parcsr_ls"
            "${HYPRE_CACHED_SOURCE}/src/struct_mv"
            "${HYPRE_CACHED_SOURCE}/src/struct_ls"
            "${HYPRE_CACHED_SOURCE}/src/sstruct_mv"
            "${HYPRE_CACHED_SOURCE}/src/sstruct_ls"
            "${HYPRE_CACHED_BUILD}"
        )
        set_target_properties(HYPRE PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
        )

        # HYPRE with CUDA requires linking CUDA libraries (matching HYPRETargets.cmake)
        target_link_libraries(HYPRE INTERFACE CUDA::cusparse_static CUDA::curand_static)

        # Set variables expected by downstream code
        set(hypre_SOURCE_DIR "${HYPRE_CACHED_SOURCE}" CACHE PATH "HYPRE source directory" FORCE)
        set(hypre_BINARY_DIR "${HYPRE_CACHED_BUILD}" CACHE PATH "HYPRE binary directory" FORCE)

        message(STATUS "  HYPRE source dir: ${hypre_SOURCE_DIR}")
        message(STATUS "  HYPRE binary dir: ${hypre_BINARY_DIR}")
    else()
        message(STATUS "  No cached HYPRE build found - building from source")

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
        # Priority: 1. User-specified -DGPU_CC=XX, 2. Auto-detect, 3. Default to 80
        if(NOT DEFINED GPU_CC)
            execute_process(
                COMMAND ${CMAKE_SOURCE_DIR}/scripts/detect_gpu_cc.sh
                OUTPUT_VARIABLE DETECTED_GPU_CC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE DETECT_RESULT
            )
            if(DETECT_RESULT EQUAL 0 AND DETECTED_GPU_CC MATCHES "^[0-9]+$")
                set(GPU_CC "${DETECTED_GPU_CC}" CACHE STRING "GPU compute capability (auto-detected)")
                message(STATUS "  GPU_CC auto-detected: ${GPU_CC}")
            else()
                set(GPU_CC "80" CACHE STRING "GPU compute capability (default=A100)")
                message(STATUS "  GPU_CC fallback: ${GPU_CC}")
            endif()
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
    target_sources(nn_cfd_core PRIVATE
        src/gpu_init.cpp
        src/mg_cuda_kernels.cpp
    )
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
            src/poisson_solver_fft1d.cpp
            src/poisson_solver_fft2d.cpp)
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

# Comprehensive profiling across BCs, Poisson solvers, turbulence models
add_executable(profile_comprehensive app/profile_comprehensive.cpp)
target_link_libraries(profile_comprehensive nn_cfd_core)

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

    # Function to add a test with standard configuration and labels
    # Usage: add_nncfd_test(test_name [TEST_NAME_SUFFIX name] [LABELS label1 label2 ...])
    #
    # Labels for filtering with ctest -L <label> or ctest -LE <label>:
    #   fast  - Runs in <30s, suitable for quick validation
    #   slow  - Runs in >60s, excluded from fast CI
    #   gpu   - Requires GPU hardware (USE_GPU_OFFLOAD builds)
    #   hypre - Requires HYPRE solver (USE_HYPRE builds)
    #   fft   - Requires FFT solver (USE_FFT_POISSON builds)
    function(add_nncfd_test test_name)
        set(options "")
        set(oneValueArgs TEST_NAME_SUFFIX)
        set(multiValueArgs LABELS)
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} nn_cfd_core)

        # Generate test name: convert test_foo_bar to FooBarTest
        if(ARG_TEST_NAME_SUFFIX)
            set(test_suffix "${ARG_TEST_NAME_SUFFIX}")
        else()
            string(REPLACE "test_" "" base_name ${test_name})
            string(REPLACE "_" ";" name_parts ${base_name})
            set(test_suffix "")
            foreach(part ${name_parts})
                string(SUBSTRING ${part} 0 1 first_letter)
                string(TOUPPER ${first_letter} first_letter)
                string(SUBSTRING ${part} 1 -1 rest)
                string(APPEND test_suffix "${first_letter}${rest}")
            endforeach()
            string(APPEND test_suffix "Test")
        endif()

        add_test(NAME ${test_suffix} COMMAND ${test_name})

        # Apply labels for ctest filtering
        if(ARG_LABELS)
            set_tests_properties(${test_suffix} PROPERTIES LABELS "${ARG_LABELS}")
        endif()
    endfunction()

    # =========================================================================
    # Fast tests (<30s) - run with: ctest -L fast
    # =========================================================================
    add_nncfd_test(test_mesh LABELS fast)
    add_nncfd_test(test_features LABELS fast)
    add_nncfd_test(test_nn_core TEST_NAME_SUFFIX NNCoreTest LABELS fast)
    add_nncfd_test(test_config LABELS fast)
    add_nncfd_test(test_mpi_guard LABELS fast)
    add_nncfd_test(test_vtk_output TEST_NAME_SUFFIX VTKOutputTest LABELS fast)
    add_nncfd_test(test_backend_unified LABELS fast)
    add_nncfd_test(test_residual_consistency LABELS fast)
    add_nncfd_test(test_mesh_edge_cases LABELS fast)
    add_nncfd_test(test_error_recovery LABELS fast)
    add_nncfd_test(test_adaptive_dt LABELS fast)
    add_nncfd_test(test_numerics LABELS fast)
    add_nncfd_test(test_operator_convergence LABELS fast)
    add_nncfd_test(test_gpu_buffer TEST_NAME_SUFFIX GPUBufferTest LABELS fast)

    # =========================================================================
    # Medium tests (30-120s)
    # =========================================================================
    add_nncfd_test(test_poisson_unified)
    add_nncfd_test(test_nn_integration TEST_NAME_SUFFIX NNIntegrationTest)
    add_nncfd_test(test_turbulence_unified)
    add_nncfd_test(test_projection_invariants)
    add_nncfd_test(test_perf_sentinel)

    # =========================================================================
    # Slow tests (>120s) - exclude with: ctest -LE slow
    # =========================================================================
    add_nncfd_test(test_2d_3d_comparison TEST_NAME_SUFFIX Comparison2D3DTest LABELS slow)
    add_nncfd_test(test_3d_unified TEST_NAME_SUFFIX ThreeDUnifiedTest LABELS slow)
    add_nncfd_test(test_perturbed_channel LABELS slow)
    add_nncfd_test(test_endurance_stability LABELS slow)
    add_nncfd_test(test_repeatability LABELS slow)
    add_nncfd_test(test_physics_validation_advanced LABELS slow)
    add_nncfd_test(test_kernel_parity_detailed LABELS slow)

    # =========================================================================
    # GPU-specific tests - run with: ctest -L gpu
    # =========================================================================
    # test_cpu_gpu_bitwise must be built unconditionally (used by both CPU and GPU CI builds)
    # but only registered as a test when GPU offload is enabled
    add_executable(test_cpu_gpu_bitwise tests/test_cpu_gpu_bitwise.cpp)
    target_link_libraries(test_cpu_gpu_bitwise nn_cfd_core)

    if(USE_GPU_OFFLOAD)
        add_nncfd_test(test_cpu_gpu_unified TEST_NAME_SUFFIX CPUGPUUnifiedTest LABELS gpu)
        add_test(NAME CPUGPUBitwiseTest COMMAND test_cpu_gpu_bitwise)
        set_tests_properties(CPUGPUBitwiseTest PROPERTIES LABELS "gpu")
        add_nncfd_test(test_gpu_utilization TEST_NAME_SUFFIX GPUUtilizationTest LABELS gpu)
    endif()

    # =========================================================================
    # FFT tests (require USE_FFT_POISSON) - run with: ctest -L fft
    # =========================================================================
    if(USE_GPU_OFFLOAD AND CUDAToolkit_FOUND)
        add_nncfd_test(test_fft_unified TEST_NAME_SUFFIX FFTUnifiedTest LABELS "fft;gpu")
    endif()

    # =========================================================================
    # HYPRE tests (require USE_HYPRE) - run with: ctest -L hypre
    # =========================================================================
    if(USE_HYPRE)
        add_nncfd_test(test_hypre_all_bcs LABELS hypre)
        add_nncfd_test(test_hypre_validation LABELS hypre)
        add_nncfd_test(test_hypre_backend LABELS hypre)
    endif()

    # =========================================================================
    # Benchmarks (not run as tests, for profiling only)
    # =========================================================================
    add_executable(bench_fft_vs_mg tests/bench_fft_vs_mg.cpp)
    target_link_libraries(bench_fft_vs_mg PRIVATE nn_cfd_core)
    add_executable(bench_mg_tuning tests/bench_mg_tuning.cpp)
    target_link_libraries(bench_mg_tuning PRIVATE nn_cfd_core)
    add_executable(bench_mg_bc_sweep tests/bench_mg_bc_sweep.cpp)
    target_link_libraries(bench_mg_bc_sweep PRIVATE nn_cfd_core)
    if(USE_GPU_OFFLOAD)
        add_executable(bench_mg_cuda_graphs tests/bench_mg_cuda_graphs.cpp)
        target_link_libraries(bench_mg_cuda_graphs PRIVATE nn_cfd_core)
    endif()

    # =========================================================================
    # Convenience targets for running tests
    # =========================================================================
    # Run all tests: make check
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all tests"
    )

    # Run fast tests only: make check-fast
    add_custom_target(check-fast
        COMMAND ${CMAKE_CTEST_COMMAND} -L fast --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running fast tests only"
    )

    # Run tests excluding slow: make check-quick
    add_custom_target(check-quick
        COMMAND ${CMAKE_CTEST_COMMAND} -LE slow --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests (excluding slow)"
    )
endif()

# Installation
install(TARGETS channel duct taylor_green_3d DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
