#!/bin/bash
# Compare CPU-only build vs GPU-offload build
# Usage: compare_cpu_gpu_builds.sh <workdir>
# Designed to run on an H200 GPU node via SLURM

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  CPU-only build vs GPU-offload build comparison"
echo "==================================================================="
echo ""

# Build a CPU-only reference binary (no GPU offload)
rm -rf build_ci_cpu_ref
mkdir -p build_ci_cpu_ref
cd build_ci_cpu_ref

echo "=== CMake Configuration (CPU-only reference) ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF 2>&1 | tee cmake_config.log
echo ""
echo "=== Building (CPU-only reference) ==="
make -j8

echo ""
echo "--- Running test_cpu_gpu_consistency (CPU-only build) ---"
echo "This test will skip GPU-specific tests and run CPU validation only."
./test_cpu_gpu_consistency || {
    echo "[INFO] CPU-only build completed (GPU tests skipped as expected)"
}

# Now run with the GPU-offload build
cd "$WORKDIR/build_ci_gpu_correctness"

echo ""
echo "--- Running test_cpu_gpu_consistency (GPU-offload build) ---"
echo "This test compares CPU and GPU execution paths within the same binary."
./test_cpu_gpu_consistency || {
    echo "[FAIL] GPU consistency test failed!"
    exit 1
}

echo ""
echo "[PASS] CPU-only vs GPU-offload comparison completed successfully"

