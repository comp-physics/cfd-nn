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
rm -rf cpu_gpu_comparison
mkdir -p cpu_gpu_comparison

./test_cpu_gpu_bitwise --dump-prefix cpu_gpu_comparison/bitwise || {
    echo "[FAIL] Bitwise CPU reference generation failed!"
    exit 1
}
./test_poisson_cpu_gpu_3d --dump-prefix cpu_gpu_comparison/poisson3d || {
    echo "[FAIL] Poisson 3D CPU reference generation failed!"
    exit 1
}
./test_cpu_gpu_consistency --dump-prefix cpu_gpu_comparison/consistency || {
    echo "[FAIL] Consistency CPU reference generation failed!"
    exit 1
}
./test_solver_cpu_gpu --dump-prefix cpu_gpu_comparison/solver || {
    echo "[FAIL] Solver CPU reference generation failed!"
    exit 1
}
./test_time_history_consistency --dump-prefix cpu_gpu_comparison/timehistory || {
    echo "[FAIL] Time-history CPU reference generation failed!"
    exit 1
}

echo ""
echo "--- Step 2: Run GPU and compare against CPU reference ---"
if [ ! -d "$WORKDIR/build_ci_gpu_correctness" ]; then
    echo "[INFO] build_ci_gpu_correctness not found; creating GPU-offload build..."
    rm -rf "$WORKDIR/build_ci_gpu_correctness"
    mkdir -p "$WORKDIR/build_ci_gpu_correctness"
    cd "$WORKDIR/build_ci_gpu_correctness"

    echo "=== CMake Configuration (GPU-offload) ==="
    CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON 2>&1 | tee cmake_config.log
    echo ""
    echo "=== Building (GPU-offload) ==="
    make -j8
else
    cd "$WORKDIR/build_ci_gpu_correctness"
    echo "=== Building (GPU-offload incremental) ==="
    make -j8
fi

./test_cpu_gpu_bitwise --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/bitwise" || {
    echo "[FAIL] Bitwise GPU vs CPU comparison failed!"
    exit 1
}
./test_poisson_cpu_gpu_3d --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/poisson3d" || {
    echo "[FAIL] Poisson 3D GPU vs CPU comparison failed!"
    exit 1
}
./test_cpu_gpu_consistency --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/consistency" || {
    echo "[FAIL] Consistency GPU vs CPU comparison failed!"
    exit 1
}
./test_solver_cpu_gpu --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/solver" || {
    echo "[FAIL] Solver GPU vs CPU comparison failed!"
    exit 1
}
./test_time_history_consistency --compare-prefix "$WORKDIR/build_ci_cpu_ref/cpu_gpu_comparison/timehistory" || {
    echo "[FAIL] Time-history GPU vs CPU comparison failed!"
    exit 1
}

echo ""
echo "[PASS] CPU-only vs GPU-offload comparison completed successfully"
echo "      GPU results match CPU reference within tolerance"

