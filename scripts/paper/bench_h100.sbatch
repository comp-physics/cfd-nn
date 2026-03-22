#!/bin/bash
#SBATCH -J cans-bench-h100
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_h100_%j.out
#SBATCH -e results/paper/tables/bench_h100_%j.err

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

SOLVER=./channel
if [ ! -f "$SOLVER" ]; then
    SOLVER=build_bench/channel
fi

for cfg in scripts/paper/bench_cans_256x128x72.cfg scripts/paper/bench_cans_256x256x256.cfg scripts/paper/bench_cans_512x256x144.cfg; do
    GRID=$(basename $cfg .cfg | sed 's/bench_cans_//')
    IFS='x' read -r NX NY NZ <<< "$GRID"
    CELLS=$((NX * NY * NZ))
    echo ""
    echo "--- Grid ${GRID} = ${CELLS} cells ---"
    OUTPUT=$(timeout 300 $SOLVER --config $cfg 2>&1)
    STEP_LINE=$(echo "$OUTPUT" | grep "solver_step")
    if [ -n "$STEP_LINE" ]; then
        AVG_MS=$(echo "$STEP_LINE" | awk '{print $4}')
        python3 -c "
cells = $CELLS; ms = float('$AVG_MS')
print(f'  {ms:.3f} ms/step, {cells/(ms/1000)/1e6:.1f} Mcells/s')
"
    else
        echo "  Failed:"
        echo "$OUTPUT" | tail -3
    fi
done
