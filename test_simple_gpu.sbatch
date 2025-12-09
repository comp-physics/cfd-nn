#!/bin/bash
#SBATCH -Jgpu_simple_test
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t0:10:00
#SBATCH -qembers
#SBATCH -Agts-sbryngelson3-paid
#SBATCH -ogpu_simple_%j.out
#SBATCH --gres=gpu:1

module load nvhpc
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_refactor

echo "Running simple test with verbose error checking..."
./test_solver 2>&1

echo ""
echo "Testing basic channel flow (smaller grid)..."
./channel --Nx 32 --Ny 64 --nu 0.01 --max_iter 10 --verbose 2>&1
