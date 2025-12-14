#!/bin/bash
# GPU Performance Suite - CPU vs GPU performance comparison
# Usage: gpu_perf_suite.sh <workdir>
# Designed to run on an H200 GPU node via SLURM

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  GPU Performance Suite - CPU vs GPU Comparison"
echo "==================================================================="
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

chmod +x .github/scripts/*.sh

# Build CPU-only binary (single-threaded)
echo "==================================================================="
echo "  Building CPU-only binary (Release, single-threaded)"
echo "==================================================================="
echo ""
rm -rf build_cpu
mkdir -p build_cpu
cd build_cpu
echo "=== CMake Configuration ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF 2>&1 | tee cmake_config.log
echo ""
echo "=== Building ==="
make -j8
mkdir -p output/cpu_perf
cd ..

# Build GPU-offload binary
echo ""
echo "==================================================================="
echo "  Building GPU-offload binary (Release)"
echo "==================================================================="
echo ""
rm -rf build_gpu
mkdir -p build_gpu
cd build_gpu
echo "=== CMake Configuration ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON 2>&1 | tee cmake_config.log
echo ""
echo "=== Building ==="
make -j8
mkdir -p output/gpu_perf
cd ..

# Run comparison for one case
run_comparison_case() {
    local name="$1"
    shift 1
    
    echo ""
    echo "==================================================================="
    echo "CASE: ${name}"
    echo "==================================================================="
    echo "CMD: $*"
    echo ""
    
    # CPU run (single-threaded)
    echo "--- CPU (single-threaded) ---"
    export OMP_NUM_THREADS=1
    export OMP_PROC_BIND=true
    local cpu_log="cpu_${name}.log"
    (cd build_cpu && "$@") 2>&1 | tee "$cpu_log"
    
    echo ""
    echo "--- GPU ---"
    export OMP_TARGET_OFFLOAD=MANDATORY
    unset OMP_NUM_THREADS
    local gpu_log="gpu_${name}.log"
    (cd build_gpu && "$@") 2>&1 | tee "$gpu_log"
    
    echo ""
    echo "--- Quick Summary ---"
    # Use Python script to extract timings and compute speedup
    local summary=$(python3 .github/scripts/compute_speedup.py "${name}" "$cpu_log" "$gpu_log")
    echo "$summary" | head -n 3
    
    # Last line is pipe-delimited data for table
    echo "$summary" | tail -n 1 >> perf_results.txt
}

# Initialize results file
rm -f perf_results.txt
echo "Case|CPU_Total(s)|CPU_PerStep(ms)|GPU_Total(s)|GPU_PerStep(ms)|Speedup_Total|Speedup_PerStep" > perf_results.txt

echo ""
echo "==================================================================="
echo "  Running Performance Benchmarks"
echo "==================================================================="

# Case 1: Channel baseline (medium grid, long run)
run_comparison_case "channel_baseline_256x512_2000" \
    ./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 2000 \
             --model baseline --dp_dx -0.0001 \
             --output output/perf/channel_baseline --num_snapshots 0

# Case 2: Channel SST transport (medium grid, moderate iters)
run_comparison_case "channel_sst_256x512_500" \
    ./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 500 \
             --model sst --dp_dx -0.0001 \
             --output output/perf/channel_sst --num_snapshots 0

# Case 3: Periodic hills baseline (complex geometry)
run_comparison_case "periodic_hills_baseline_128x96_400" \
    ./periodic_hills --Nx 128 --Ny 96 --nu 0.001 --max_iter 400 \
                    --model baseline --num_snapshots 0

# Print final summary table
echo ""
echo "==================================================================="
echo "  Performance Summary: CPU vs GPU"
echo "==================================================================="
echo ""
printf "%-35s %12s %15s %12s %15s %10s %10s\n" \
       "Case" "CPU Total" "CPU Per-Step" "GPU Total" "GPU Per-Step" "Speedup" "Speedup"
printf "%-35s %12s %15s %12s %15s %10s %10s\n" \
       "" "(s)" "(ms)" "(s)" "(ms)" "(Total)" "(Step)"
printf "%s\n" "$(printf '%.0s-' {1..115})"

tail -n +2 perf_results.txt | while IFS='|' read -r case cpu_t cpu_avg gpu_t gpu_avg sp_t sp_avg; do
    printf "%-35s %12s %15s %12s %15s %9sx %9sx\n" \
           "$case" "$cpu_t" "$cpu_avg" "$gpu_t" "$gpu_avg" "$sp_t" "$sp_avg"
done

echo ""
echo "[PASS] GPU Performance Suite completed successfully"

