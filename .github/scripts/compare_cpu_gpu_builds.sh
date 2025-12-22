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
echo "--- Step 1: Generate CPU reference outputs ---"
mkdir -p cpu_gpu_comparison
./test_cpu_gpu_consistency --dump-prefix cpu_gpu_comparison/cpu_ref || {
    echo "[FAIL] CPU reference generation failed!"
    exit 1
}

echo ""
echo "--- Step 2: Run GPU and compare against CPU reference ---"
cd "$WORKDIR/build_ci_gpu_correctness"

./test_cpu_gpu_consistency --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/cpu_ref" || {
    echo "[FAIL] GPU vs CPU comparison failed!"
    exit 1
}

echo ""
echo "[PASS] CPU-only vs GPU-offload comparison completed successfully"
echo "      GPU results match CPU reference within tolerance"

