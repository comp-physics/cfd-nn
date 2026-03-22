#!/bin/bash
#SBATCH -J cans-bench-4xv100
#SBATCH -A gts-sbryngelson3
#SBATCH -N 2 --ntasks-per-node=2
#SBATCH -p gpu-v100
#SBATCH --gpus-per-node=2
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_v100_4gpu_%j.out
#SBATCH -e results/paper/tables/bench_v100_4gpu_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
HPCX=$NVHPC_ROOT/comm_libs/12.4/hpcx/hpcx-2.19/ompi
export PATH=$NVHPC_ROOT/compilers/bin:$HPCX/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "============================================"
echo "  4x V100 MPI Benchmark (2 nodes x 2 GPUs)"
echo "  GPU: $GPU"
echo "  Nodes: $SLURM_JOB_NODELIST"
echo "============================================"

# Use pre-built binary (built on H100 with CC=70,90)
SOLVER=build_mpi_bench/channel

echo ""
echo "--- 512x256x144 = 18874368 cells ---"
for NR in 1 2 4; do
    echo "  ${NR} GPU(s):"
    OUTPUT=$(mpirun --bind-to none --mca pml ucx -n $NR $SOLVER --config scripts/paper/bench_cans_512x256x144.cfg 2>&1)
    MS=$(echo "$OUTPUT" | grep "solver_step" | head -1 | awk '{print $4}')
    if [ -n "$MS" ]; then
        python3 -c "
cells=18874368; ms=float('$MS'); n=$NR
print(f'    {ms:.2f} ms/step, {cells/(ms/1000)/1e6:.1f} Mcells/s ({cells/(ms/1000)/1e6/n:.1f}/GPU)')
"
    else
        echo "    Failed"
        echo "$OUTPUT" | tail -3
    fi
done

echo ""
echo "CaNS reference (4x V100-32GB DGX-2): 481 ms/step = 39 Mcells/s"
