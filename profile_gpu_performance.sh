#!/bin/bash
#SBATCH -J gpu_profile_perf
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH -t1:00:00
#SBATCH -qembers
#SBATCH -Agts-sbryngelson3-paid
#SBATCH --gres=gpu:1

# ============================================================================
# GPU Performance Profiling with Nsight Systems
# Measures kernel execution time and CPU/GPU speedup
# ============================================================================

module load nvhpc

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

echo "========================================"
echo "GPU Performance Profiling"
echo "========================================"
echo ""

# Test parameters
NX=256
NY=512
MAX_ITER=100
MODEL="baseline"

# ============================================
# Part 1: CPU Baseline (single core)
# ============================================
echo "Part 1: CPU Baseline (single threaded)"
echo "----------------------------------------"

if [ ! -d "build_cpu_profile" ]; then
    echo "Building CPU version..."
    mkdir -p build_cpu_profile
    cd build_cpu_profile
    CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF
    make channel -j8
    cd ..
fi

export OMP_NUM_THREADS=1  # Single thread for fair comparison

echo "Running CPU baseline (1 thread)..."
cd build_cpu_profile
/usr/bin/time -v ./channel --Nx $NX --Ny $NY --nu 0.001 --max_iter $MAX_ITER \
         --model $MODEL --dp_dx -0.0001 \
         --output ../cpu_profile --num_snapshots 0 \
         2>&1 | tee ../cpu_timing.log
cd ..

CPU_TIME=$(grep "Elapsed (wall clock)" cpu_timing.log | awk '{print $NF}' | \
           awk -F: '{if (NF==3) print $1*60+$2; else print $1}')
echo "CPU Time: $CPU_TIME seconds"
echo ""

# ============================================
# Part 2: GPU with Nsight profiling
# ============================================
echo "Part 2: GPU with Nsight Systems profiling"
echo "-------------------------------------------"

if [ ! -d "build_refactor" ]; then
    echo "ERROR: build_refactor not found!"
    exit 1
fi

cd build_refactor

echo "Running GPU version with profiling..."
nsys profile \
    --stats=true \
    --force-overwrite=true \
    --output=gpu_profile \
    --trace=cuda,nvtx,openmp \
    ./channel --Nx $NX --Ny $NY --nu 0.001 --max_iter $MAX_ITER \
             --model $MODEL --dp_dx -0.0001 \
             --output ../gpu_profile --num_snapshots 0 \
    2>&1 | tee ../gpu_timing.log

cd ..

# Extract GPU time (from actual output if possible)
if grep -q "Total time:" gpu_timing.log; then
    GPU_TIME=$(grep "Total time:" gpu_timing.log | awk '{print $3}')
else
    GPU_TIME=$(grep "Elapsed (wall clock)" gpu_timing.log | awk '{print $NF}' | \
               awk -F: '{if (NF==3) print $1*60+$2; else print $1}')
fi
echo "GPU Time: $GPU_TIME seconds"
echo ""

# ============================================
# Part 3: Analysis
# ============================================
echo "========================================"
echo "Performance Analysis"
echo "========================================"
echo ""

# Check if we have valid numbers
if [ -z "$CPU_TIME" ] || [ -z "$GPU_TIME" ]; then
    echo "ERROR: Could not extract timing information"
    echo "CPU_TIME=$CPU_TIME, GPU_TIME=$GPU_TIME"
    exit 1
fi

SPEEDUP=$(echo "$CPU_TIME $GPU_TIME" | awk '{printf "%.1f", $1/$2}')

echo "Grid: ${NX}x${NY}, Iterations: $MAX_ITER"
echo "Model: $MODEL"
echo ""
echo "Results:"
echo "  CPU (1 thread): ${CPU_TIME}s"
echo "  GPU:            ${GPU_TIME}s"
echo "  Speedup:        ${SPEEDUP}x"
echo ""

# Performance assessment
if (( $(echo "$SPEEDUP < 5" | bc -l) )); then
    echo "❌ POOR: GPU speedup is less than 5x!"
    echo "   Something is seriously wrong."
elif (( $(echo "$SPEEDUP < 10" | bc -l) )); then
    echo "⚠️  WARNING: GPU speedup is less than 10x!"
    echo "   Expected: >10x for this grid size"
    echo "   Possible causes:"
    echo "     - Too much data movement"
    echo "     - Small kernel launches"
    echo "     - Host-device synchronization"
elif (( $(echo "$SPEEDUP < 20" | bc -l) )); then
    echo "✓ GOOD: GPU performance is acceptable (${SPEEDUP}x speedup)"
else
    echo "✓✓ EXCELLENT: GPU performance is very good (${SPEEDUP}x speedup)"
fi

echo ""
echo "========================================"
echo "Nsight Systems Output"
echo "========================================"
echo ""
echo "Report saved to: build_refactor/gpu_profile.nsys-rep"
echo ""
echo "To view interactively (requires X11):"
echo "  nsys-ui build_refactor/gpu_profile.nsys-rep"
echo ""
echo "To view stats:"
echo "  cd build_refactor"
echo "  nsys stats gpu_profile.nsys-rep"
echo ""
echo "Key metrics to check in Nsight:"
echo "  1. CUDA API calls - should be minimal"
echo "  2. Memory transfers - should see only init + final"
echo "  3. Kernel execution - should be majority of time"
echo "  4. CPU wait time - should be minimal"
echo ""

# Try to extract basic stats
if [ -f "build_refactor/gpu_profile.sqlite" ]; then
    echo "Extracting kernel statistics..."
    nsys stats --report cuda_gpu_kern_sum build_refactor/gpu_profile.nsys-rep 2>/dev/null || echo "  (nsys stats failed)"
fi





