#!/bin/bash
#SBATCH -J cans-bench-mpi
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=2
#SBATCH -p gpu-v100
#SBATCH -G 2
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_v100_mpi_%j.out
#SBATCH -e results/paper/tables/bench_v100_mpi_%j.err

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
echo ""

# Build with MPI + GPU for V100
BUILD_DIR=build_v100_mpi
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building solver with MPI+GPU (CC=70)..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DUSE_MPI=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=70 \
             -DMPI_CXX_COMPILER=$HPCX/bin/mpicxx \
             -DMPI_C_COMPILER=$HPCX/bin/mpicc 2>&1 | tail -5
    make -j$(nproc) channel 2>&1 | tail -5
    cd ..
fi

SOLVER=$BUILD_DIR/channel

# Run at CaNS benchmark grid: 512x256x144
# z-slab decomposition: Nz=144 / 4 ranks = 36 z-slabs per rank
for NRANKS in 1 2; do
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
    echo ""
done

echo "============================================"
echo "  CaNS Reference (Costa et al. 2021)"
echo "  V100-SXM2-32GB (DGX-2), 512x256x144"
echo "============================================"
echo "  4x V100:  481 ms/step =  39 Mcells/s (9.8/GPU)"
echo "  8x V100:  251 ms/step =  75 Mcells/s (9.4/GPU)"
echo "  16x V100: 140 ms/step = 135 Mcells/s (8.4/GPU)"
echo ""
echo "Both: 2nd-order central FD, RK3, explicit diffusion, FFT Poisson"
