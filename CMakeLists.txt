cmake_minimum_required(VERSION 3.16)
project(nn_cfd VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

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
)

# Core library
add_library(nn_cfd_core ${LIB_SOURCES})

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
endif()

# Installation
install(TARGETS channel periodic_hills DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)


