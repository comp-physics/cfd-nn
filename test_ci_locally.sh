#!/bin/bash
#SBATCH -J test_ci_local
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -q embers
#SBATCH -A gts-sbryngelson3
#SBATCH -o test_ci_local_%j.out
#SBATCH -e test_ci_local_%j.err

set -e

echo "========================================="
echo "Local CI Test Reproduction - $(date)"
echo "========================================="

# Load modules (same as CI)
module reset
module load nvhpc

# Show GPU info
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Build directory (use current code, not pulled)
BUILD_DIR="${SLURM_SUBMIT_DIR}/build_ci_test_local"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "=== Building with NVHPC and GPU offload ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON 2>&1 | tee cmake_config.log

echo ""
echo "=== CMake Configuration Summary ==="
grep -E "GPU offload|CXX compiler|OpenMP|NVIDIA" cmake_config.log || echo "No GPU-related config found"

echo ""
echo "=== Compiling ==="
make -j8

echo ""
echo "=== Running Unit Tests ==="
ctest --output-on-failure

echo ""
echo "=== Running Validation Test (Baseline, 50 iterations) ==="
mkdir -p output/test_validation
./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 50 \
          --model baseline --dp_dx -0.0001 \
          --output output/test_validation/baseline --num_snapshots 0

echo ""
echo "=== Checking Results ==="
echo "Output files:"
ls -lh output/test_validation/

# Extract velocity info
if [ -f "output/test_validation/baselinechannel_final.vtk" ]; then
    echo ""
    echo "Analyzing VTK output..."
    VELOCITIES=$(awk '/^VECTORS velocity/{flag=1; next} /^SCALARS/{flag=0} flag && NF==3' output/test_validation/baselinechannel_final.vtk)
    MAX_VEL=$(echo "$VELOCITIES" | awk '{
        u = $1; v = $2;
        mag = sqrt(u*u + v*v);
        if (mag > max_mag) max_mag = mag;
    } END {print max_mag}')
    echo "Max velocity magnitude: $MAX_VEL"
    echo "Expected (analytical): ~0.05"
    echo "Expected bulk velocity: ~0.033"
fi

echo ""
echo "=== Now testing with MORE iterations (500) ==="
./channel --Nx 256 --Ny 512 --nu 0.001 --max_iter 500 \
          --model baseline --dp_dx -0.0001 \
          --output output/test_validation/baseline_500 --num_snapshots 0

if [ -f "output/test_validation/baseline_500channel_final.vtk" ]; then
    echo ""
    echo "Analyzing 500-iteration results..."
    VELOCITIES=$(awk '/^VECTORS velocity/{flag=1; next} /^SCALARS/{flag=0} flag && NF==3' output/test_validation/baseline_500channel_final.vtk)
    MAX_VEL=$(echo "$VELOCITIES" | awk '{
        u = $1; v = $2;
        mag = sqrt(u*u + v*v);
        if (mag > max_mag) max_mag = mag;
    } END {print max_mag}')
    echo "Max velocity magnitude with 500 iters: $MAX_VEL"
fi

echo ""
echo "=== Test completed ==="

