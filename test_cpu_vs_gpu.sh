#!/bin/bash
#SBATCH -J test_cpu_gpu_compare
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -q embers
#SBATCH -A gts-sbryngelson3
#SBATCH -o test_cpu_gpu_%j.out
#SBATCH -e test_cpu_gpu_%j.err

set -e

echo "========================================="
echo "CPU vs GPU Comparison Test - $(date)"
echo "========================================="

module reset
module load nvhpc

# Build both CPU and GPU versions
echo "=== Building CPU version ==="
rm -rf build_cpu_test
mkdir -p build_cpu_test
cd build_cpu_test
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF
make channel -j8
cd ..

echo ""
echo "=== Building GPU version ==="
rm -rf build_gpu_test  
mkdir -p build_gpu_test
cd build_gpu_test
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON
make channel -j8
cd ..

# Test parameters
NX=256
NY=512
NU=0.001
DP_DX=-0.0001
ITERS=100

echo ""
echo "========================================="
echo "Test Configuration:"
echo "Grid: ${NX}x${NY}"
echo "nu: ${NU}"
echo "dp/dx: ${DP_DX}"
echo "Iterations: ${ITERS}"
echo "========================================="

# Run CPU version
echo ""
echo "=== Running CPU version ==="
mkdir -p output_cpu_test
cd build_cpu_test
./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model baseline --dp_dx $DP_DX \
          --output ../output_cpu_test/cpu --num_snapshots 0
cd ..

echo ""
echo "=== CPU Results ==="
grep -A 10 "=== Results ===" build_cpu_test/output_cpu_test/cpu*.dat 2>/dev/null | head -20 || \
  echo "Checking output files..." && ls -lh output_cpu_test/

# Run GPU version
echo ""
echo "=== Running GPU version ==="
mkdir -p output_gpu_test
cd build_gpu_test
./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model baseline --dp_dx $DP_DX \
          --output ../output_gpu_test/gpu --num_snapshots 0
cd ..

echo ""
echo "=== GPU Results ==="
grep -A 10 "=== Results ===" build_gpu_test/output_gpu_test/gpu*.dat 2>/dev/null | head -20 || \
  echo "Checking output files..." && ls -lh output_gpu_test/

# Compare velocity fields
echo ""
echo "========================================="
echo "Comparing CPU vs GPU Results"
echo "========================================="

if [ -f "output_cpu_test/cpuchannel_velocity.dat" ] && [ -f "output_gpu_test/gpuchannel_velocity.dat" ]; then
    echo "Analyzing velocity fields..."
    
    # Get max velocities
    CPU_MAX=$(awk 'NR>1 {u=$3; v=$4; mag=sqrt(u*u+v*v); if(mag>max) max=mag} END{print max}' output_cpu_test/cpuchannel_velocity.dat)
    GPU_MAX=$(awk 'NR>1 {u=$3; v=$4; mag=sqrt(u*u+v*v); if(mag>max) max=mag} END{print max}' output_gpu_test/gpuchannel_velocity.dat)
    
    echo "CPU max velocity: $CPU_MAX"
    echo "GPU max velocity: $GPU_MAX"
    
    # Compute difference
    DIFF=$(echo "$CPU_MAX $GPU_MAX" | awk '{print ($1-$2)/$1*100}')
    echo "Relative difference: ${DIFF}%"
    
    # Check if they're close
    IS_CLOSE=$(echo "$DIFF" | awk '{if ($1<0) $1=-$1; if($1<1.0) print "YES"; else print "NO"}')
    
    if [ "$IS_CLOSE" = "YES" ]; then
        echo "✓ CPU and GPU results are similar (< 1% difference)"
    else
        echo "✗ WARNING: CPU and GPU results differ by ${DIFF}%!"
    fi
    
    # Sample a few velocity values
    echo ""
    echo "Sample velocity values (first 10 points):"
    echo "CPU:"
    head -11 output_cpu_test/cpuchannel_velocity.dat | tail -10
    echo ""
    echo "GPU:"
    head -11 output_gpu_test/gpuchannel_velocity.dat | tail -10
    
else
    echo "✗ Could not find velocity output files for comparison"
fi

echo ""
echo "=== Test completed ==="

