#!/bin/bash
#SBATCH -J gpu_benchmark
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH -q embers
#SBATCH -A gts-sbryngelson3
#SBATCH -t 00:30:00
#SBATCH -o gpu_benchmark_%j.out
#SBATCH -e gpu_benchmark_%j.err

module reset
module load nvhpc

echo "=== GPU-Accelerated CFD-NN Benchmark ==="
echo "Date: $(date)"
echo ""

nvidia-smi -L
echo ""

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn

# Build if needed
if [ ! -f "build/channel" ]; then
    echo "Building with GPU offload support..."
    rm -rf build
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON
    make -j8
    cd ..
    echo ""
fi

echo "=== Benchmark 1: 64x128 grid, 100 iterations ==="
./build/channel --Nx 64 --Ny 128 --nu 0.01 --max_iter 100 --model nn_tbnn --nn_preset example_tbnn

echo ""
echo "=== Benchmark 2: 128x256 grid, 100 iterations ==="
./build/channel --Nx 128 --Ny 256 --nu 0.01 --max_iter 100 --model nn_tbnn --nn_preset example_tbnn

echo ""
echo "=== Benchmark 3: 256x512 grid, 50 iterations ==="
./build/channel --Nx 256 --Ny 512 --nu 0.01 --max_iter 50 --model nn_tbnn --nn_preset example_tbnn

echo ""
echo "=== GPU Status After Benchmarks ==="
nvidia-smi
echo ""
echo "=== Benchmark Complete ==="


