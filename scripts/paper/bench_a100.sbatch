#!/bin/bash
#SBATCH -J bench-a100-cales
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-a100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_a100_%j.out
#SBATCH -e results/paper/tables/bench_a100_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "============================================"
echo "  A100 Benchmark — CaLES Comparison"
echo "  GPU: $GPU"
echo "============================================"
echo ""
echo "CaLES (Xiao et al. 2024): 1x A100-64GB"
echo "  DNS explicit: 1.50 ns/step/point (512x384x1440, 283M cells)"
echo "  WRLES:        1.80 ns/step/point (same grid)"
echo ""

# Build for A100 (CC=80)
BUILD_DIR=build_a100_bench
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building for A100 (CC=80)..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=80 2>&1 | tail -3
    make -j$(nproc) channel 2>&1 | tail -3
    cd ..
fi

SOLVER=$BUILD_DIR/channel

for GRID in "128 128 128" "256 256 256" "512 256 144" "384 256 256" "512 256 256" "512 384 256" "512 384 384"; do
    read -r NX NY NZ <<< "$GRID"
    CELLS=$((NX * NY * NZ))

    cat > /tmp/bench.cfg << CFGEOF
Nx = $NX
Ny = $NY
Nz = $NZ
x_min = 0.0
x_max = 6.0
y_min = -1.0
y_max = 1.0
z_min = 0.0
z_max = 3.0
nu = 0.000177305
dp_dx = -1.0
convective_scheme = central
time_integrator = rk3
adaptive_dt = false
dt = 0.00002
max_steps = 100
gpu_only_mode = true
perf_mode = true
benchmark = true
CFGEOF

    echo "--- ${NX}x${NY}x${NZ} = ${CELLS} cells ---"
    OUTPUT=$(timeout 300 $SOLVER --config /tmp/bench.cfg 2>&1)
    MS=$(echo "$OUTPUT" | grep "solver_step" | head -1 | awk '{print $4}')
    if [ -n "$MS" ]; then
        python3 -c "
cells=$CELLS; ms=float('$MS')
ns_pt = ms * 1e6 / cells
mcells = cells / (ms/1000) / 1e6
print(f'  {ms:.2f} ms/step, {mcells:.0f} Mcells/s, {ns_pt:.2f} ns/point')
"
    else
        echo "  Failed (likely OOM or timeout)"
    fi
done
