#!/bin/bash
#SBATCH -J gpu_large_profile
#SBATCH -N1 --ntasks-per-node=12
#SBATCH --mem=64G
#SBATCH -t2:00:00
#SBATCH -qembers
#SBATCH -Agts-sbryngelson3-paid
#SBATCH --gres=gpu:1

# ============================================================================
# Large-Scale GPU Profiling
# Tests with production-size grids to measure true GPU performance
# ============================================================================

module load nvhpc

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

echo "========================================"
echo "Large-Scale GPU Performance Profiling"
echo "========================================"
echo ""
echo "GPU Memory Available: 32 GB"
echo "Testing multiple grid sizes for scaling analysis"
echo ""

# Test parameters - will run 3 sizes
MODEL="baseline"
MAX_ITER=50  # Fewer iterations for large grids

# Grid sizes to test (increasing complexity)
declare -a GRIDS=(
    "2048 1024"   # ~2M cells, baseline
    "4096 2048"   # ~8M cells, medium
    "8192 4096"   # ~33M cells, large (good GPU utilization)
)

RESULTS_FILE="large_scale_profiling_results.txt"
> $RESULTS_FILE  # Clear file

echo "========================================"  >> $RESULTS_FILE
echo "Large-Scale GPU Profiling Results"      >> $RESULTS_FILE
echo "Date: $(date)"                           >> $RESULTS_FILE
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $RESULTS_FILE
echo "========================================"  >> $RESULTS_FILE
echo ""                                         >> $RESULTS_FILE

# ============================================
# Function to run CPU test
# ============================================
run_cpu_test() {
    local nx=$1
    local ny=$2
    local cells=$((nx * ny))
    
    echo ""
    echo "========================================" | tee -a $RESULTS_FILE
    echo "CPU Test: ${nx} × ${ny} (${cells} cells)" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    
    if [ ! -d "build_cpu_profile" ]; then
        echo "Building CPU version..."
        mkdir -p build_cpu_profile
        cd build_cpu_profile
        CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF
        make channel -j12
        cd ..
    fi
    
    export OMP_NUM_THREADS=1  # Single thread baseline
    
    cd build_cpu_profile
    START=$(date +%s.%N)
    ./channel --Nx $nx --Ny $ny --nu 0.001 --max_iter $MAX_ITER \
             --model $MODEL --dp_dx -0.0001 \
             --output ../cpu_profile_${nx}x${ny} --num_snapshots 0 \
             > ../cpu_${nx}x${ny}.log 2>&1
    END=$(date +%s.%N)
    cd ..
    
    CPU_TIME=$(echo "$END - $START" | bc)
    
    # Extract timing breakdown
    grep "solver_step" cpu_${nx}x${ny}.log | tail -1 | tee -a $RESULTS_FILE
    echo "Total wall time: ${CPU_TIME} s" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    echo "$CPU_TIME"
}

# ============================================
# Function to run GPU test
# ============================================
run_gpu_test() {
    local nx=$1
    local ny=$2
    local cells=$((nx * ny))
    
    echo ""
    echo "========================================" | tee -a $RESULTS_FILE
    echo "GPU Test: ${nx} × ${ny} (${cells} cells)" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    
    cd build_refactor
    START=$(date +%s.%N)
    ./channel --Nx $nx --Ny $ny --nu 0.001 --max_iter $MAX_ITER \
             --model $MODEL --dp_dx -0.0001 \
             --output ../gpu_profile_${nx}x${ny} --num_snapshots 0 \
             > ../gpu_${nx}x${ny}.log 2>&1
    END=$(date +%s.%N)
    cd ..
    
    GPU_TIME=$(echo "$END - $START" | bc)
    
    # Extract timing breakdown and memory usage
    grep "Mapping" gpu_${nx}x${ny}.log | tee -a $RESULTS_FILE
    grep "GPU_PROFILE" gpu_${nx}x${ny}.log | tee -a $RESULTS_FILE
    grep "solver_step" gpu_${nx}x${ny}.log | tail -1 | tee -a $RESULTS_FILE
    echo "Total wall time: ${GPU_TIME} s" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    echo "$GPU_TIME"
}

# ============================================
# Run profiling for each grid size
# ============================================

echo "Grid Size,Cells,CPU Time (s),GPU Time (s),Speedup" > scaling_data.csv

for grid in "${GRIDS[@]}"; do
    read NX NY <<< "$grid"
    CELLS=$((NX * NY))
    
    echo ""
    echo "###############################################"
    echo "# Testing ${NX} × ${NY} = ${CELLS} cells"
    echo "###############################################"
    echo ""
    
    # Run CPU baseline
    CPU_TIME=$(run_cpu_test $NX $NY)
    
    # Run GPU version
    GPU_TIME=$(run_gpu_test $NX $NY)
    
    # Calculate speedup
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    
    echo ""
    echo "========================================"
    echo "Results for ${NX} × ${NY}:"
    echo "  CPU time: ${CPU_TIME} s"
    echo "  GPU time: ${GPU_TIME} s"
    echo "  Speedup:  ${SPEEDUP}×"
    echo "========================================"
    echo ""
    
    # Save to CSV
    echo "${NX}x${NY},${CELLS},${CPU_TIME},${GPU_TIME},${SPEEDUP}" >> scaling_data.csv
    
    # Save to results file
    echo "----------------------------------------" >> $RESULTS_FILE
    echo "${NX} × ${NY}: CPU=${CPU_TIME}s, GPU=${GPU_TIME}s, Speedup=${SPEEDUP}×" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
done

# ============================================
# Run detailed Nsight profiling on largest grid
# ============================================

echo ""
echo "========================================"
echo "Part 3: Detailed Nsight Profiling"
echo "Running on largest grid: 8192 × 4096"
echo "========================================"
echo ""

cd build_refactor

# Run with full Nsight profiling
nsys profile --trace=cuda,openmp,nvtx --output=gpu_profile_large \
     ./channel --Nx 8192 --Ny 4096 --nu 0.001 --max_iter $MAX_ITER \
              --model $MODEL --dp_dx -0.0001 \
              --output ../gpu_large --num_snapshots 0

# Generate reports
echo ""
echo "Generating profiling reports..."
nsys stats --report nvtx_sum gpu_profile_large.nsys-rep
echo ""
nsys stats --report cuda_gpu_kern_sum gpu_profile_large.nsys-rep | head -30
echo ""
nsys stats --report cuda_gpu_mem_time_sum gpu_profile_large.nsys-rep
echo ""
nsys stats --report cuda_gpu_mem_size_sum gpu_profile_large.nsys-rep

cd ..

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo "PROFILING COMPLETE"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_FILE (detailed breakdown)"
echo "  - scaling_data.csv (speedup data)"
echo "  - build_refactor/gpu_profile_large.nsys-rep (Nsight report)"
echo ""
echo "Scaling Summary:"
cat scaling_data.csv
echo ""
echo "To view Nsight GUI:"
echo "  nsys-ui build_refactor/gpu_profile_large.nsys-rep"
echo ""

