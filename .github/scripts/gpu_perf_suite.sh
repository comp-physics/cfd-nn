#!/bin/bash
# GPU Performance Suite - Build and run timed performance benchmarks
# Usage: gpu_perf_suite.sh <workdir>
# Designed to run on an H200 GPU node via SLURM

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  GPU Performance Suite"
echo "==================================================================="
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Hard-require OpenMP target offload (fail if it falls back to CPU)
export OMP_TARGET_OFFLOAD=MANDATORY

chmod +x .github/scripts/*.sh

# Clean rebuild (perf suite must rebuild first)
rm -rf build_ci_gpu_perf
mkdir -p build_ci_gpu_perf
cd build_ci_gpu_perf

echo "==================================================================="
echo "  Building GPU-offload binary (Release)"
echo "==================================================================="
echo ""

echo "=== CMake Configuration ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON 2>&1 | tee cmake_config.log
echo ""
echo "=== Building ==="
make -j8
mkdir -p output/gpu_perf

echo ""
echo "==================================================================="
echo "  GPU Performance Cases (timed)"
echo "==================================================================="
echo ""
echo "NOTE: Two runs per case (warmup + timed). Output suppressed for clean timing."
echo ""

run_case () {
    local name="$1"
    shift 1
    echo "-------------------------------------------"
    echo "CASE: ${name}"
    echo "CMD:  $*"
    echo ""
    # warmup
    "$@" >/dev/null 2>&1
    # timed
    echo "Timing run..."
    /usr/bin/time -p "$@" >/dev/null 2>&1
    echo "PERF_CASE_DONE name=\"${name}\""
    echo ""
}

# Case 1: Channel baseline (medium grid, long run)
run_case "channel_baseline_256x512_2000" \
    ./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 2000 \
             --model baseline --dp_dx -0.0001 \
             --output output/gpu_perf/channel_baseline --num_snapshots 0 --quiet

# Case 2: Channel SST transport (medium grid, moderate iters)
run_case "channel_sst_256x512_500" \
    ./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 500 \
             --model sst --dp_dx -0.0001 \
             --output output/gpu_perf/channel_sst --num_snapshots 0 --quiet

# Case 3: Periodic hills baseline (complex geometry)
run_case "periodic_hills_baseline_128x96_400" \
    ./periodic_hills --Nx 128 --Ny 96 --nu 0.001 --max_iter 400 \
                    --model baseline --num_snapshots 0

echo ""
echo "==================================================================="
echo "  Performance Summary"
echo "==================================================================="
echo ""
echo "Extract PERF_CASE_DONE markers and preceding 'time -p' output from log."
echo "Timing data is printed above (real/user/sys for each case)."
echo ""
echo "✅ GPU Performance Suite completed successfully"
