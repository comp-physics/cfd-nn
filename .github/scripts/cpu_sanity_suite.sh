#!/bin/bash
# CPU Sanity Suite for H200 Runner
#
# This script runs a subset of CPU tests on the H200 node to catch
# environment-specific issues that hosted CI might miss:
#   - NVHPC compiler quirks
#   - Slurm job context differences
#   - Module environment issues
#   - Filesystem quirks
#
# Run this BEFORE the GPU build to establish CPU correctness baseline.

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  CPU Sanity Suite (H200 Runner)"
echo "==================================================================="
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

# Ensure NVHPC compilers are used (same as GPU build)
export CC=nvc
export CXX=nvc++
echo "Compiler: $CXX"
$CXX --version | head -1
echo ""

# Clean and build CPU version
echo "==================================================================="
echo "  Building CPU version (USE_GPU_OFFLOAD=OFF)"
echo "==================================================================="
echo ""

rm -rf build_ci_cpu_sanity
mkdir -p build_ci_cpu_sanity
cd build_ci_cpu_sanity

cmake "$WORKDIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=OFF \
    -DBUILD_TESTS=ON \
    2>&1 | tee cmake_config.log

# Check for common cmake issues
if grep -q "Could not find" cmake_config.log; then
    echo "[ERROR] CMake configuration has missing dependencies"
    exit 1
fi

make -j8 2>&1 | tee build.log

# Check for build warnings that might indicate issues
WARNING_COUNT=$(grep -c "warning:" build.log 2>/dev/null || echo 0)
if [ "$WARNING_COUNT" -gt 10 ]; then
    echo "[WARN] Build produced $WARNING_COUNT warnings"
fi

echo ""
echo "==================================================================="
echo "  Running CPU Sanity Tests"
echo "==================================================================="
echo ""

PASSED=0
FAILED=0
TESTS_RUN=""

run_test() {
    local name=$1
    local binary=$2
    local timeout=${3:-60}

    if [ ! -f "$binary" ]; then
        echo "  [SKIP] $name (not built)"
        return 0
    fi

    echo -n "  $name... "
    TESTS_RUN="$TESTS_RUN $name"

    if timeout "$timeout" "$binary" > /tmp/test_output_$$.txt 2>&1; then
        echo "[PASS]"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL]"
        echo "    Output (last 20 lines):"
        tail -20 /tmp/test_output_$$.txt | sed 's/^/      /'
        FAILED=$((FAILED + 1))
    fi
    rm -f /tmp/test_output_$$.txt
}

# Fast unit tests
echo "--- Fast Unit Tests ---"
run_test "Mesh" "./test_mesh" 30
run_test "Features" "./test_features" 30
run_test "NN Core" "./test_nn_core" 30

# 3D validation tests
echo ""
echo "--- 3D Validation Tests ---"
run_test "3D Quick Validation" "./test_3d_quick_validation" 120
run_test "3D Gradients" "./test_3d_gradients" 60

# Poisson solver tests
echo ""
echo "--- Poisson Solver Tests ---"
run_test "Poisson Unified" "./test_poisson_unified" 180
run_test "Residual Consistency" "./test_residual_consistency" 120

# MPI guard test
echo ""
echo "--- Infrastructure Tests ---"
run_test "MPI Guard" "./test_mpi_guard" 30

# Kernel parity (standalone mode)
echo ""
echo "--- Kernel Tests ---"
run_test "Kernel Parity" "./test_kernel_parity" 60

echo ""
echo "==================================================================="
echo "  CPU Sanity Suite Summary"
echo "==================================================================="
echo ""
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "[FAIL] CPU sanity suite failed"
    echo "       Environment-specific issues detected on H200 node"
    exit 1
else
    echo "[PASS] CPU sanity suite completed"
    echo "       CPU correctness verified in H200 environment"
    echo ""
    echo "       Proceed with GPU build for full validation."
fi
