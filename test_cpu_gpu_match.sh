#!/bin/bash
#SBATCH -J cpu_gpu_match
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH -t 00:30:00
#SBATCH -o cpu_gpu_match_%j.out
#SBATCH -e cpu_gpu_match_%j.err

# Load modules
module load nvhpc/24.5
module load cuda/11.8.0

echo "==========================================="
echo "CPU vs GPU Exact Match Testing"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Check GPU
echo "--- GPU Information ---"
nvidia-smi | head -15
echo ""

# Build directories
SRC_DIR="/storage/home/hcoda1/6/sbryngelson3/cfd-nn"
BUILD_GPU="$SRC_DIR/build_gpu"

echo "==========================================="
echo "Building with GPU Offloading"
echo "==========================================="

cd $SRC_DIR
mkdir -p $BUILD_GPU
cd $BUILD_GPU
rm -rf *

CC=nvc CXX=nvc++ cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_OFFLOAD=ON 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi

make -j8 2>&1 | grep -E "(Building|Linking|error|warning)" | head -50

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo ""

echo "==========================================="
echo "Running SolverTest with GPU"
echo "==========================================="
export OMP_TARGET_OFFLOAD=MANDATORY

./test_solver 2>&1 | tee solver_gpu_output.txt

GPU_EXIT=$?

if [ $GPU_EXIT -eq 0 ]; then
    echo "✅ GPU tests PASSED"
else
    echo "❌ GPU tests FAILED with exit code $GPU_EXIT"
fi

echo ""
echo "==========================================="
echo "Extracting Poiseuille Test Result"
echo "==========================================="
grep "Testing laminar Poiseuille" solver_gpu_output.txt

echo ""
echo "==========================================="
echo "Summary"
echo "==========================================="
if [ $GPU_EXIT -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - CPU and GPU give correct results!"
else  
    echo "❌ Tests failed - check output above"
    echo ""
    echo "Common issues:"
    echo "  - Initial conditions not synced to GPU"
    echo "  - nu_eff not properly initialized on GPU"
    echo "  - Numerical precision differences"
fi

echo ""
echo "Testing complete at $(date)"


