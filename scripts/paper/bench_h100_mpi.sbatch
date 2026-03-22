#!/bin/bash
#SBATCH -J cans-bench-h100-mpi
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH -p gpu-h100
#SBATCH -G 4
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_h100_mpi_%j.out
#SBATCH -e results/paper/tables/bench_h100_mpi_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
HPCX=$NVHPC_ROOT/comm_libs/12.4/hpcx/hpcx-2.19/ompi
export PATH=$NVHPC_ROOT/compilers/bin:$HPCX/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64:${LD_LIBRARY_PATH:-}
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
NGPU=$(nvidia-smi -L | wc -l)
echo "============================================"
echo "  MPI GPU Benchmark — CaNS Comparison"
echo "  GPU: $GPU x $NGPU"
echo "============================================"

# Build with MPI + GPU for H100
BUILD_DIR=build_h100_mpi
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building solver with MPI+GPU (CC=90)..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DUSE_MPI=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90 \
             -DMPI_CXX_COMPILER=$HPCX/bin/mpicxx \
             -DMPI_C_COMPILER=$HPCX/bin/mpicc 2>&1 | tail -5
    make -j$(nproc) channel 2>&1 | tail -5
    cd ..
fi

SOLVER=$BUILD_DIR/channel

for NRANKS in 1 2 4; do
    echo ""
    echo "--- $NRANKS GPU(s), Grid 512x256x144 = 18874368 cells ---"
    OUTPUT=$(mpirun --oversubscribe -n $NRANKS $SOLVER --config scripts/paper/bench_cans_512x256x144.cfg 2>&1)
    STEP_LINE=$(echo "$OUTPUT" | grep "solver_step" | head -1)
    if [ -n "$STEP_LINE" ]; then
        AVG_MS=$(echo "$STEP_LINE" | awk '{print $4}')
        python3 -c "
cells = 18874368; ms = float('$AVG_MS'); nranks = $NRANKS
print(f'  {ms:.3f} ms/step, {cells/(ms/1000)/1e6:.1f} Mcells/s total, {cells/(ms/1000)/1e6/nranks:.1f} Mcells/s/GPU')
"
    else
        echo "  Failed:"
        echo "$OUTPUT" | tail -5
    fi
done

echo ""
echo "CaNS reference (4x V100-32GB): 481 ms/step = 39 Mcells/s"
