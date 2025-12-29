#!/bin/bash
#
# Build script for cfd-nn
#
# Usage:
#   ./build.sh          # CPU build (default)
#   ./build.sh cpu      # CPU build
#   ./build.sh gpu      # GPU build with OpenMP offloading
#   ./build.sh clean    # Remove all build artifacts
#
# Options:
#   --rebuild           # Force clean rebuild
#   --debug             # Debug build (default: Release)
#   --jobs N            # Parallel jobs (default: auto)
#
# Examples:
#   ./build.sh gpu --rebuild
#   ./build.sh cpu --debug --jobs 4

set -euo pipefail

# Script location and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Defaults
BUILD_MODE="cpu"
BUILD_TYPE="Release"
REBUILD=false
JOBS=$(nproc 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        cpu|CPU)
            BUILD_MODE="cpu"
            shift
            ;;
        gpu|GPU)
            BUILD_MODE="gpu"
            shift
            ;;
        clean)
            echo "Cleaning build directory..."
            rm -rf "$BUILD_DIR"/CMake* "$BUILD_DIR"/Makefile "$BUILD_DIR"/*.cmake
            rm -rf "$BUILD_DIR"/*.a "$BUILD_DIR"/compile_commands.json
            rm -rf "$BUILD_DIR"/channel "$BUILD_DIR"/duct "$BUILD_DIR"/taylor_green_3d
            rm -rf "$BUILD_DIR"/compare_channel_cpu_gpu
            rm -rf "$BUILD_DIR"/profile_* "$BUILD_DIR"/test_*
            echo "Clean complete."
            exit 0
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [cpu|gpu|clean] [options]"
            echo ""
            echo "Build modes:"
            echo "  cpu       CPU build (default, single-threaded)"
            echo "  gpu       GPU build with OpenMP target offloading"
            echo "  clean     Remove all build artifacts"
            echo ""
            echo "Options:"
            echo "  --rebuild   Force clean rebuild"
            echo "  --debug     Debug build (default: Release)"
            echo "  --jobs N    Parallel jobs (default: $JOBS)"
            echo ""
            echo "Examples:"
            echo "  $0 gpu --rebuild"
            echo "  $0 cpu --debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Load NVHPC module if available and nvc++ not found
if ! command -v nvc++ &>/dev/null; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        source /etc/profile.d/lmod.sh
    fi
    if command -v module &>/dev/null; then
        echo "Loading nvhpc module..."
        module load nvhpc 2>/dev/null || true
    fi
fi

# Verify nvc++ is available
if ! command -v nvc++ &>/dev/null; then
    echo "ERROR: nvc++ not found."
    echo ""
    echo "Please load the NVIDIA HPC SDK:"
    echo "  module load nvhpc"
    exit 1
fi

CXX_COMPILER=$(command -v nvc++)

# Clean if rebuild requested
if [[ "$REBUILD" == true ]]; then
    echo "Cleaning for rebuild..."
    rm -rf "$BUILD_DIR"/CMake* "$BUILD_DIR"/Makefile "$BUILD_DIR"/*.cmake
    rm -rf "$BUILD_DIR"/*.a "$BUILD_DIR"/compile_commands.json
fi

# Set up cmake options based on build mode
if [[ "$BUILD_MODE" == "gpu" ]]; then
    echo "=============================================="
    echo "  GPU Build (OpenMP Target Offloading)"
    echo "=============================================="

    CMAKE_OPTS=(
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
        -DUSE_GPU_OFFLOAD=ON
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    )

    echo "Compiler: $CXX_COMPILER"
    echo "GPU offload: ON"
else
    echo "=============================================="
    echo "  CPU Build"
    echo "=============================================="

    CMAKE_OPTS=(
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
        -DUSE_GPU_OFFLOAD=OFF
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    )

    echo "Compiler: $CXX_COMPILER"
    echo "GPU offload: OFF"
fi

echo "Build type: $BUILD_TYPE"
echo "Parallel jobs: $JOBS"
echo ""

# Run cmake if needed
if [[ ! -f "$BUILD_DIR/Makefile" ]]; then
    echo "Running cmake..."
    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" "${CMAKE_OPTS[@]}"
    echo ""
fi

# Build
echo "Building with $JOBS parallel jobs..."
cd "$BUILD_DIR"
make -j"$JOBS"

echo ""
echo "=============================================="
echo "  Build complete!"
echo "=============================================="
echo ""
echo "Executables:"
for exe in channel duct taylor_green_3d; do
    if [[ -x "$BUILD_DIR/$exe" ]]; then
        echo "  ./$exe"
    fi
done
echo ""
echo "Run tests with: make test"
echo ""
