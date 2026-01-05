#!/bin/bash
# GPU Performance Suite - GPU absolute performance validation
# Usage: gpu_perf_suite.sh <workdir>
# Designed to run on an H200 GPU node via SLURM
#
# Strategy:
#   - Gate on 3D FFT/FFT1D cases where GPU should win decisively
#   - Use warmup steps to exclude initialization overhead from timing
#   - Check absolute GPU performance (ms/step) rather than speedup ratios
#   - Makes MG-only cases non-gating (logged for trending)

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  GPU Performance Suite - Absolute Performance Validation"
echo "==================================================================="
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

chmod +x .github/scripts/*.sh

# Build GPU-offload binary only (we're validating GPU absolute perf, not speedup)
# Preserves _deps (HYPRE cache) while rebuilding project code
echo "==================================================================="
echo "  Building GPU-offload binary (Release)"
echo "==================================================================="
echo ""
mkdir -p build_gpu
cd build_gpu
# Clean project artifacts but preserve _deps (HYPRE cache)
if [ -d _deps ]; then
    echo "Preserving HYPRE cache in _deps/"
    find . -mindepth 1 -maxdepth 1 ! -name '_deps' -exec rm -rf {} +
fi
# H200 requires cc90 (Hopper architecture)
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 2>&1 | tee cmake_config.log
echo "=== Building ==="
make -j8 channel duct
mkdir -p output/gpu_perf
cd ..

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=""

# Run GPU performance gate
# Usage: run_gpu_perf_gate <name> <executable> <max_ms_per_step> [args...]
run_gpu_perf_gate() {
    local name="$1"
    local exe="$2"
    local max_ms="$3"
    shift 3

    echo ""
    echo "==================================================================="
    echo "GATE: ${name} (max ${max_ms} ms/step)"
    echo "==================================================================="
    echo "CMD: ./${exe} $*"
    echo ""

    export OMP_TARGET_OFFLOAD=MANDATORY
    local log="gpu_${name}.log"
    (cd build_gpu && "./${exe}" "$@") 2>&1 | tee "$log"

    # Extract solver_step average time
    local avg_ms=$(grep -E '^\s*solver_step\s+' "$log" | awk '{print $4}')

    if [ -z "$avg_ms" ]; then
        echo "[FAIL] ${name}: Could not extract timing"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS} ${name}"
        return 1
    fi

    # Check threshold (using bc for floating point comparison)
    local passed=$(echo "$avg_ms <= $max_ms" | bc -l)

    if [ "$passed" -eq 1 ]; then
        echo "[PASS] ${name}: ${avg_ms} ms/step <= ${max_ms} ms/step threshold"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo "[FAIL] ${name}: ${avg_ms} ms/step > ${max_ms} ms/step threshold"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS} ${name}"
        return 1
    fi
}

# Run non-gating GPU performance check (logged but doesn't fail CI)
# Usage: run_gpu_perf_info <name> <executable> [args...]
run_gpu_perf_info() {
    local name="$1"
    local exe="$2"
    shift 2

    echo ""
    echo "==================================================================="
    echo "INFO: ${name} (non-gating)"
    echo "==================================================================="
    echo "CMD: ./${exe} $*"
    echo ""

    export OMP_TARGET_OFFLOAD=MANDATORY
    local log="gpu_${name}.log"
    (cd build_gpu && "./${exe}" "$@") 2>&1 | tee "$log"

    # Extract solver_step average time
    local avg_ms=$(grep -E '^\s*solver_step\s+' "$log" | awk '{print $4}')

    if [ -z "$avg_ms" ]; then
        echo "[INFO] ${name}: Could not extract timing"
    else
        echo "[INFO] ${name}: ${avg_ms} ms/step"
    fi
}

echo ""
echo "==================================================================="
echo "  Running Performance Gates"
echo "==================================================================="

# Gate 1: 3D Duct with FFT1D (64x64x64, 25 steps with 5 warmup = 20 timed)
# GPU FFT1D should be very fast - expect < 3 ms/step
# This validates cuFFT + cuSPARSE tridiagonal solve is working
run_gpu_perf_gate "duct_fft1d_64" \
    duct 3.0 \
    --Nx 64 --Ny 64 --Nz 64 \
    --nu 0.001 --dp_dx -1.0 \
    --max_iter 25 --warmup_steps 5 \
    --poisson fft1d \
    --simulation_mode unsteady \
    --no_postprocess --no_write_fields --verbose

# Gate 2: Larger 3D duct (128x64x64) to verify scaling
# Expect < 8 ms/step with FFT1D
run_gpu_perf_gate "duct_fft1d_128" \
    duct 8.0 \
    --Nx 128 --Ny 64 --Nz 64 \
    --nu 0.001 --dp_dx -1.0 \
    --max_iter 25 --warmup_steps 5 \
    --poisson fft1d \
    --simulation_mode unsteady \
    --no_postprocess --no_write_fields --verbose

# Gate 3: 2D channel with MG at larger size (512x512)
# At 512x512, GPU MG should beat CPU - expect < 100 ms/step
run_gpu_perf_gate "channel_mg_512" \
    channel 100.0 \
    --Nx 512 --Ny 512 \
    --nu 0.001 --dp_dx -0.0001 \
    --max_iter 25 --warmup_steps 5 \
    --poisson mg \
    --simulation_mode unsteady \
    --no_postprocess --no_write_fields --verbose

echo ""
echo "==================================================================="
echo "  Running Non-Gating Performance Checks (for trending)"
echo "==================================================================="

# Info: Small 2D channel MG (expected to be slow on GPU due to overhead)
# This is logged for trend tracking but won't fail CI
run_gpu_perf_info "channel_mg_128_info" \
    channel \
    --Nx 128 --Ny 128 \
    --nu 0.001 --dp_dx -0.0001 \
    --max_iter 25 --warmup_steps 5 \
    --poisson mg \
    --simulation_mode unsteady \
    --no_postprocess --no_write_fields --verbose

echo ""
echo "==================================================================="
echo "  Performance Suite Summary"
echo "==================================================================="
echo ""
echo "Gates passed: ${TESTS_PASSED}"
echo "Gates failed: ${TESTS_FAILED}"

if [ "$TESTS_FAILED" -gt 0 ]; then
    echo ""
    echo "FAILED GATES:${FAILED_TESTS}"
    echo ""
    echo "[FAIL] GPU Performance Suite - ${TESTS_FAILED} gate(s) failed"
    exit 1
else
    echo ""
    echo "[PASS] GPU Performance Suite - all gates passed"
fi
