#!/bin/bash
#
# Build script for cfd-nn
#
# Usage:
#   ./make.sh              # CPU build (default)
#   ./make.sh cpu          # CPU build
#   ./make.sh gpu          # GPU build with OpenMP offloading
#   ./make.sh clean        # Remove all build artifacts
#
# Options:
#   --rebuild              # Force clean rebuild
#   --debug                # Debug build (default: Release)
#   --jobs N               # Parallel jobs (default: auto)
#   --hdf5                 # Enable HDF5 checkpoint/restart
#   --mpi                  # Enable MPI domain decomposition
#   --hypre                # Enable HYPRE Poisson solver
#   --gpu-cc N             # GPU compute capability (default: auto-detect)
#   --build-dir DIR        # Custom build directory (default: build/)
#   --all-features         # Enable HDF5 + MPI (GPU builds also get HYPRE)
#
# Examples:
#   ./make.sh gpu --rebuild
#   ./make.sh gpu --hdf5 --mpi
#   ./make.sh cpu --debug --hdf5
#   ./make.sh gpu --all-features --gpu-cc 90
#   ./make.sh gpu --build-dir build_production

set -euo pipefail

# Script location and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Defaults
BUILD_MODE="cpu"
BUILD_TYPE="Release"
BUILD_DIR=""
REBUILD=false
JOBS=$(nproc 2>/dev/null || echo 4)
USE_HDF5=false
USE_MPI=false
USE_HYPRE=false
GPU_CC=""
ALL_FEATURES=false

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
            echo "Cleaning build directories..."
            rm -rf "$PROJECT_ROOT"/build/CMake* "$PROJECT_ROOT"/build/Makefile "$PROJECT_ROOT"/build/*.cmake
            rm -rf "$PROJECT_ROOT"/build/*.a "$PROJECT_ROOT"/build/compile_commands.json
            rm -rf "$PROJECT_ROOT"/build/channel "$PROJECT_ROOT"/build/duct "$PROJECT_ROOT"/build/taylor_green_3d
            rm -rf "$PROJECT_ROOT"/build/cylinder "$PROJECT_ROOT"/build/airfoil
            rm -rf "$PROJECT_ROOT"/build/compare_channel_cpu_gpu
            rm -rf "$PROJECT_ROOT"/build/profile_* "$PROJECT_ROOT"/build/test_* "$PROJECT_ROOT"/build/bench_*
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
        --hdf5)
            USE_HDF5=true
            shift
            ;;
        --mpi)
            USE_MPI=true
            shift
            ;;
        --hypre)
            USE_HYPRE=true
            shift
            ;;
        --gpu-cc)
            GPU_CC="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --all-features)
            ALL_FEATURES=true
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
            echo "  --rebuild       Force clean rebuild"
            echo "  --debug         Debug build (default: Release)"
            echo "  --jobs N        Parallel jobs (default: $JOBS)"
            echo "  --hdf5          Enable HDF5 checkpoint/restart"
            echo "  --mpi           Enable MPI domain decomposition"
            echo "  --hypre         Enable HYPRE Poisson solver (GPU only)"
            echo "  --gpu-cc N      GPU compute capability (e.g., 80=A100, 90=H200)"
            echo "  --build-dir DIR Custom build directory"
            echo "  --all-features  Enable all optional features"
            echo ""
            echo "Configurations:"
            echo "  $0 cpu                          # Quick CPU dev build"
            echo "  $0 cpu --debug                  # CPU debug build"
            echo "  $0 cpu --hdf5                   # CPU + checkpoint support"
            echo "  $0 gpu                          # GPU production build"
            echo "  $0 gpu --all-features           # GPU + HDF5 + MPI + HYPRE"
            echo "  $0 gpu --all-features --gpu-cc 90  # Full H200 build"
            echo "  $0 gpu --mpi --build-dir build_mpi # Separate MPI build dir"
            echo ""
            echo "Test commands (run from build directory):"
            echo "  ctest --output-on-failure        # All tests"
            echo "  ctest -L fast                    # Fast tests (<30s)"
            echo "  ctest -L medium                  # Medium tests (30-120s)"
            echo "  ctest -LE slow                   # Skip slow tests"
            echo "  ctest -L gpu                     # GPU-only tests"
            echo "  ctest -L fft                     # FFT solver tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Apply --all-features
if [[ "$ALL_FEATURES" == true ]]; then
    USE_HDF5=true
    USE_MPI=true
    if [[ "$BUILD_MODE" == "gpu" ]]; then
        USE_HYPRE=true
    fi
fi

# Set build directory (default based on mode)
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$PROJECT_ROOT/build"
fi
mkdir -p "$BUILD_DIR"

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

# Build cmake options array
CMAKE_OPTS=(
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ "$BUILD_MODE" == "gpu" ]]; then
    CMAKE_OPTS+=(-DUSE_GPU_OFFLOAD=ON)
    if [[ -n "$GPU_CC" ]]; then
        CMAKE_OPTS+=(-DGPU_CC="$GPU_CC")
    fi
else
    CMAKE_OPTS+=(-DUSE_GPU_OFFLOAD=OFF)
fi

if [[ "$USE_HDF5" == true ]]; then
    CMAKE_OPTS+=(-DUSE_HDF5=ON)
fi

if [[ "$USE_MPI" == true ]]; then
    CMAKE_OPTS+=(-DUSE_MPI=ON)
fi

if [[ "$USE_HYPRE" == true ]]; then
    CMAKE_OPTS+=(-DUSE_HYPRE=ON)
fi

# Print configuration
echo "=============================================="
if [[ "$BUILD_MODE" == "gpu" ]]; then
    echo "  GPU Build (OpenMP Target Offloading)"
else
    echo "  CPU Build"
fi
echo "=============================================="
echo "Compiler:    $CXX_COMPILER"
echo "Build type:  $BUILD_TYPE"
echo "Build dir:   $BUILD_DIR"
echo "GPU offload: $(if [[ $BUILD_MODE == gpu ]]; then echo ON; else echo OFF; fi)"
[[ -n "$GPU_CC" ]] && echo "GPU CC:      $GPU_CC"
echo "HDF5:        $(if $USE_HDF5; then echo ON; else echo OFF; fi)"
echo "MPI:         $(if $USE_MPI; then echo ON; else echo OFF; fi)"
echo "HYPRE:       $(if $USE_HYPRE; then echo ON; else echo OFF; fi)"
echo "Jobs:        $JOBS"
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
for exe in channel duct taylor_green_3d cylinder airfoil; do
    if [[ -x "$BUILD_DIR/$exe" ]]; then
        echo "  ./$exe --config <file.cfg>"
    fi
done
echo ""
echo "Run tests:   cd $BUILD_DIR && ctest --output-on-failure"
echo "Fast tests:  cd $BUILD_DIR && ctest -L fast --output-on-failure"
echo ""
