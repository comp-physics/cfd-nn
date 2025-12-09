#!/bin/bash
#SBATCH -JGPU_Refactor_Test
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t0:30:00
#SBATCH -qembers
#SBATCH -Agts-sbryngelson3-paid
#SBATCH -ogpu_refactor_test_%j.out
#SBATCH --gres=gpu:1

# GPU Refactor Validation Test
# Runs on GPU node and compares with CPU results

echo "========================================="
echo "GPU Refactor Test - GPU Node"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
module load nvhpc

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Go to project directory
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

# Make test script executable
chmod +x test_gpu_refactor.sh

# Run the test
echo "========================================="
echo "Running GPU test case..."
echo "========================================="
echo ""

# Run from build directory
if [ -d "build_refactor" ]; then
    cd build_refactor
    bash ../test_gpu_refactor.sh
else
    echo "ERROR: build_refactor directory not found!"
    exit 1
fi

echo ""
echo "========================================="
echo "GPU Test Complete"
echo "End time: $(date)"
echo "========================================="

