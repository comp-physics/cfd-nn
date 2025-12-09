#!/bin/bash
# Comprehensive GPU CI Test - Matches CI workflow exactly
# This script reproduces the COMPLETE CI test sequence locally

set -e

echo "========================================"
echo "Comprehensive GPU CI Test (Local)"
echo "========================================"
echo "This matches the full CI workflow exactly"
echo "Date: $(date)"
echo ""

# Check if we're on a GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are you on a GPU node?"
    echo "This script should be run on a GPU node via SLURM."
    exit 1
fi

# Display GPU info
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Go to project directory
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
echo "Project directory: $PROJECT_DIR"
echo ""

# Load modules (same as CI)
echo "=== Loading modules ==="
module reset
module load nvhpc/25.5
echo ""

# Clean and rebuild (like CI does)
echo "========================================"
echo "Building with GPU support (NVHPC 25.5)"
echo "========================================"

BUILD_DIR="${PROJECT_DIR}/build_ci_comprehensive"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure (exactly like CI)
cmake .. \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_OFFLOAD=ON

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
make -j8

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "Build completed successfully"
echo ""

# Now run the COMPLETE test suite (exactly like CI)
echo "========================================"
echo "Running COMPREHENSIVE GPU Test Suite"
echo "========================================"
echo ""

# Create output directory
mkdir -p output/gpu_validation

# Make test scripts executable
chmod +x ../.github/scripts/*.sh

FAILED=0

# ========================================
# 1. Unit Tests
# ========================================
echo "========================================"
echo "1. Unit Tests"
echo "========================================"
ctest --output-on-failure
if [ $? -ne 0 ]; then
    echo "ERROR: Unit tests failed!"
    FAILED=1
fi
echo ""

# ========================================
# 2. Algebraic Models
# ========================================
echo "========================================"
echo "2. Algebraic Models"
echo "========================================"

../.github/scripts/test_turbulence_model_gpu.sh baseline "Baseline" 256 512 50000 output/gpu_validation/baseline
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline model test failed!"
    FAILED=1
fi

../.github/scripts/test_turbulence_model_gpu.sh gep "GEP" 256 512 50000 output/gpu_validation/gep
if [ $? -ne 0 ]; then
    echo "ERROR: GEP model test failed!"
    FAILED=1
fi

echo ""

# ========================================
# 3. Neural Network Models
# ========================================
echo "========================================"
echo "3. Neural Network Models"
echo "========================================"
echo "   3a. NN-MLP (skipping - requires model files)"
echo "   3b. NN-TBNN (skipping - requires model files)"
echo ""

# ========================================
# 4. Transport Equation Models
# ========================================
echo "========================================"
echo "4. Transport Equation Models"
echo "========================================"

../.github/scripts/test_turbulence_model_gpu.sh sst "SST k-omega" 256 512 1000 output/gpu_validation/sst
if [ $? -ne 0 ]; then
    echo "ERROR: SST model test failed!"
    FAILED=1
fi

../.github/scripts/test_turbulence_model_gpu.sh komega "k-omega (Wilcox)" 256 512 1000 output/gpu_validation/komega
if [ $? -ne 0 ]; then
    echo "ERROR: k-omega model test failed!"
    FAILED=1
fi

echo ""

# ========================================
# 5. EARSM Models
# ========================================
echo "========================================"
echo "5. EARSM Models"
echo "========================================"

../.github/scripts/test_turbulence_model_gpu.sh earsm_wj "Wallin-Johansson EARSM" 256 512 1000 output/gpu_validation/earsm_wj
if [ $? -ne 0 ]; then
    echo "ERROR: EARSM-WJ model test failed!"
    FAILED=1
fi

../.github/scripts/test_turbulence_model_gpu.sh earsm_gs "Gatski-Speziale EARSM" 256 512 1000 output/gpu_validation/earsm_gs
if [ $? -ne 0 ]; then
    echo "ERROR: EARSM-GS model test failed!"
    FAILED=1
fi

../.github/scripts/test_turbulence_model_gpu.sh earsm_pope "Pope Quadratic EARSM" 256 512 1000 output/gpu_validation/earsm_pope
if [ $? -ne 0 ]; then
    echo "ERROR: EARSM-Pope model test failed!"
    FAILED=1
fi

echo ""

# ========================================
# 6. Periodic Hills - Complex Geometry
# ========================================
echo "========================================"
echo "6. Periodic Hills - Complex Geometry"
echo "========================================"

echo "   6a. Baseline"
./periodic_hills --Nx 128 --Ny 96 --nu 0.001 --max_iter 500 --model baseline --num_snapshots 0
if [ $? -ne 0 ]; then
    echo "ERROR: Periodic hills baseline test failed!"
    FAILED=1
fi

echo "   6b. SST k-omega"
./periodic_hills --Nx 128 --Ny 96 --nu 0.001 --max_iter 1000 --model sst --num_snapshots 0
if [ $? -ne 0 ]; then
    echo "ERROR: Periodic hills SST test failed!"
    FAILED=1
fi

echo "   6c. EARSM (WJ)"
./periodic_hills --Nx 128 --Ny 96 --nu 0.001 --max_iter 1000 --model earsm_wj --num_snapshots 0
if [ $? -ne 0 ]; then
    echo "ERROR: Periodic hills EARSM test failed!"
    FAILED=1
fi

echo ""

# ========================================
# Summary
# ========================================
echo "========================================"
echo "COMPREHENSIVE GPU CI TEST COMPLETE"
echo "========================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo ""
    echo "This confirms the code matches CI requirements."
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "Review the output above for details."
    exit 1
fi

