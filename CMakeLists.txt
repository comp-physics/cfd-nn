cmake_minimum_required(VERSION 3.16)
project(nn_cfd VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GPU offload option (must be set before finding OpenMP)
option(USE_GPU_OFFLOAD "Enable OpenMP GPU target offloading" OFF)

# Find OpenMP (optional on macOS where AppleClang lacks OpenMP)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found - building without OpenMP parallelization")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# GPU offloading configuration
if(USE_GPU_OFFLOAD)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "GPU offloading requires OpenMP support")
    endif()
    add_definitions(-DUSE_GPU_OFFLOAD)
    message(STATUS "GPU offloading ENABLED")
    
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
elseif(OpenMP_CXX_FOUND)
    message(STATUS "GPU offloading DISABLED (CPU-only OpenMP)")
else()
    message(STATUS "OpenMP not available - single-threaded build")
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optional BLAS support
option(USE_BLAS "Use BLAS for linear algebra operations" OFF)
if(USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        add_definitions(-DUSE_BLAS)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources
set(LIB_SOURCES
    src/mesh.cpp
    src/fields.cpp
    src/poisson_solver.cpp
    src/solver.cpp
    src/timing.cpp
    src/config.cpp
    src/features.cpp
    src/nn_core.cpp
    src/turbulence_baseline.cpp
    src/turbulence_gep.cpp
    src/turbulence_nn_mlp.cpp
    src/turbulence_nn_tbnn.cpp
    src/gpu_kernels.cpp
)

# Core library
add_library(nn_cfd_core ${LIB_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_cfd_core OpenMP::OpenMP_CXX)
endif()

if(USE_BLAS AND BLAS_FOUND)
    target_link_libraries(nn_cfd_core ${BLAS_LIBRARIES})
endif()

# Executables
add_executable(channel app/main_channel.cpp)
target_link_libraries(channel nn_cfd_core)

add_executable(periodic_hills app/main_periodic_hills.cpp)
target_link_libraries(periodic_hills nn_cfd_core)

add_executable(test_nn_simple app/test_nn_simple.cpp)
target_link_libraries(test_nn_simple nn_cfd_core)

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
    
    add_executable(test_features tests/test_features.cpp)
    target_link_libraries(test_features nn_cfd_core)
    add_test(NAME FeaturesTest COMMAND test_features)
    
    add_executable(test_nn_core tests/test_nn_core.cpp)
    target_link_libraries(test_nn_core nn_cfd_core)
    add_test(NAME NNCoreTest COMMAND test_nn_core)
    
    add_executable(test_turbulence tests/test_turbulence.cpp)
    target_link_libraries(test_turbulence nn_cfd_core)
    add_test(NAME TurbulenceTest COMMAND test_turbulence)
endif()

# Installation
install(TARGETS channel periodic_hills DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
