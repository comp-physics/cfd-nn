#!/bin/bash
#SBATCH -Jtwo_build_test
#SBATCH -A gts-sbryngelson3
#SBATCH -q embers
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100|H200
#SBATCH -o two_build_test_%j.out

module load nvhpc

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

echo "=========================================="
echo "Step 1: Building CPU-only version"
echo "=========================================="

# Create CPU-only build directory
mkdir -p build_cpu
cd build_cpu

# Configure with GPU offload OFF
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF -DCMAKE_CXX_COMPILER=nvc++
make -j8 test_solver_cpu_gpu

if [ $? -ne 0 ]; then
    echo "CPU build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running CPU-only build to generate reference"
echo "=========================================="

./test_solver_cpu_gpu --dump-prefix cpu_ref

if [ $? -ne 0 ]; then
    echo "CPU reference generation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Running GPU build to compare"
echo "=========================================="

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build
./test_solver_cpu_gpu --compare-prefix ../build_cpu/cpu_ref

if [ $? -ne 0 ]; then
    echo "GPU comparison failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Two-build CPU vs GPU test PASSED!"
echo "=========================================="

