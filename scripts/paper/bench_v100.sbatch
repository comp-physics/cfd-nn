#!/bin/bash
#SBATCH -J cans-bench-v100
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-v100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_v100_%j.out
#SBATCH -e results/paper/tables/bench_v100_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64:${LD_LIBRARY_PATH:-}
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "============================================"
echo "  Solver Benchmark — CaNS Comparison"
echo "  GPU: $GPU"
echo "============================================"
echo ""

# Build for V100 (CC=70) in separate directory
BUILD_DIR=build_v100
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building solver for V100 (CC=70)..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release -DGPU_CC=70 2>&1 | tail -3
    make -j$(nproc) channel 2>&1 | tail -3
    cd ..
fi

SOLVER=$BUILD_DIR/channel

echo "Solver: $SOLVER"
echo ""

# Run benchmarks at multiple grid sizes
for cfg in scripts/paper/bench_cans_256x128x72.cfg scripts/paper/bench_cans_256x256x256.cfg scripts/paper/bench_cans_512x256x144.cfg; do
    GRID=$(basename $cfg .cfg | sed 's/bench_cans_//')
    IFS='x' read -r NX NY NZ <<< "$GRID"
    CELLS=$((NX * NY * NZ))
    
    echo "--- Grid ${GRID} = ${CELLS} cells ---"
    
    # Warmup run (10 steps)
    timeout 120 $SOLVER --config $cfg --max_steps 10 > /dev/null 2>&1 || true
    
    # Benchmark run (200 steps)
    OUTPUT=$(timeout 300 $SOLVER --config $cfg 2>&1)
    
    # Extract timing
    STEP_LINE=$(echo "$OUTPUT" | grep "solver_step")
    if [ -n "$STEP_LINE" ]; then
        TOTAL_S=$(echo "$STEP_LINE" | awk '{print $2}')
        CALLS=$(echo "$STEP_LINE" | awk '{print $3}')
        AVG_MS=$(echo "$STEP_LINE" | awk '{print $4}')
        python3 -c "
cells = $CELLS
ms = float('$AVG_MS')
mcells = cells / (ms/1000) / 1e6
print(f'  {ms:.3f} ms/step, {mcells:.1f} Mcells/s')
"
    else
        echo "  No timing data (may have crashed)"
        echo "$OUTPUT" | tail -5
    fi
    echo ""
done

echo "============================================"
echo "  CaNS Reference (Costa et al. 2021)"
echo "  V100 (DGX-2), 512x256x144, Re_tau=590"
echo "============================================"
echo "  4x V100:  481 ms/step =  39 Mcells/s"
echo "  8x V100:  251 ms/step =  75 Mcells/s"
echo "  16x V100: 140 ms/step = 135 Mcells/s"
echo ""
echo "Both solvers: 2nd-order central FD, RK3, explicit diffusion, FFT Poisson"
