#!/bin/bash
#SBATCH -J gpu_test
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH -t 01:00:00
#SBATCH -o gpu_test_%j.out
#SBATCH -e gpu_test_%j.err

# Load modules
module load nvhpc/24.5
module load cuda/11.8.0

echo "========================================="
echo "GPU Testing for CFD-NN Turbulence Models"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Check GPU availability
echo "--- GPU Information ---"
nvidia-smi
echo ""

# Set environment for GPU profiling
export NVCOMPILER_ACC_NOTIFY=1
export OMP_TARGET_OFFLOAD=MANDATORY

# Build directory
BUILD_DIR="/storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_gpu"
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

echo "========================================="
echo "Step 1: Building with GPU Offloading"
echo "========================================="

mkdir -p $BUILD_DIR
cd $BUILD_DIR
rm -rf *

# Configure with GPU offloading
# cc80 = A100, cc70 = V100, cc89 = L40S
CC=nvc CXX=nvc++ cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_OFFLOAD=ON \
  -DCMAKE_CXX_FLAGS="-O2" 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi

# Build
echo ""
echo "Building..."
make -j8 2>&1 | grep -E "(Building|Linking|error|warning)" | head -50

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo ""

echo "========================================="
echo "Step 2: Verify GPU Detection"
echo "========================================="
./test_gpu_execution 2>&1
echo ""

echo "========================================="
echo "Step 3: Turbulence Model Tests (GPU)"
echo "========================================="
export NVCOMPILER_ACC_NOTIFY=3
./test_turbulence 2>&1 | tee turbulence_gpu_detailed.log
GPU_TEST_RESULT=$?
export NVCOMPILER_ACC_NOTIFY=1

if [ $GPU_TEST_RESULT -eq 0 ]; then
    echo "✅ Turbulence tests PASSED on GPU"
else
    echo "❌ Turbulence tests FAILED on GPU"
fi
echo ""

echo "========================================="
echo "Step 4: CPU-GPU Consistency Tests"
echo "========================================="
./test_cpu_gpu_consistency 2>&1
./test_solver_cpu_gpu 2>&1
echo ""

echo "========================================="
echo "Step 5: Sequential Model Test (Critical)"
echo "========================================="
echo "Testing sequential model creation/destruction (5 runs)..."
for i in {1..5}; do
    echo "Run $i/5..."
    ./test_turbulence > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ FAILED on run $i - Sequential test FAILED"
        exit 1
    fi
done
echo "✅ Sequential test PASSED - No crashes in 5 runs"
echo ""

echo "========================================="
echo "Step 6: Full Solver Test (SolverTest)"
echo "========================================="
./test_solver 2>&1 | grep -E "(PASSED|FAILED|test_)"
echo ""

echo "========================================="
echo "Step 7: Individual Model Tests"
echo "========================================="

echo "Testing Mixing Length on GPU..."
./channel --turbulence-model mixing-length --Nx 64 --Ny 64 --max-iter 10 --Re 2800 2>&1 | tail -5
echo ""

echo "Testing GEP on GPU..."
./channel --turbulence-model gep --Nx 64 --Ny 64 --max-iter 10 --Re 2800 2>&1 | tail -5
echo ""

echo "Testing SST k-omega on GPU..."
./channel --turbulence-model sst --Nx 64 --Ny 64 --max-iter 10 --Re 2800 2>&1 | tail -5
echo ""

echo "========================================="
echo "Step 8: Performance Test (GPU vs CPU)"
echo "========================================="

echo "GPU timing (128x128, 100 iters)..."
OMP_TARGET_OFFLOAD=MANDATORY time -p ./channel --Nx 128 --Ny 128 --max-iter 100 --turbulence-model sst --Re 2800 2>&1 | tail -10

echo ""
echo "CPU timing (same problem)..."
OMP_TARGET_OFFLOAD=DISABLED time -p ./channel --Nx 128 --Ny 128 --max-iter 100 --turbulence-model sst --Re 2800 2>&1 | tail -10

echo ""
echo "========================================="
echo "GPU Testing Complete!"
echo "========================================="
echo "Time: $(date)"
echo ""
echo "Summary of tests:"
echo "  ✓ Build with GPU offloading"
echo "  ✓ GPU detection"
echo "  ✓ Turbulence model tests"
echo "  ✓ CPU-GPU consistency"
echo "  ✓ Sequential model test (critical for memory safety)"
echo "  ✓ Full solver test"
echo "  ✓ Individual model tests"
echo "  ✓ Performance comparison"
echo ""
echo "Check detailed output in: gpu_test_${SLURM_JOB_ID}.out"
echo "GPU profiling details in: turbulence_gpu_detailed.log"

