#!/bin/bash
#SBATCH -J bench-h100-large
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/tables/bench_h100_large_%j.out
#SBATCH -e results/paper/tables/bench_h100_large_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "============================================"
echo "  Large Grid Benchmark for CaLES Comparison"
echo "  GPU: $GPU"
echo "============================================"

# Build for H100 if needed
BUILD_DIR=build_h100
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building solver for H100 (CC=90)..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90 2>&1 | tail -3
    make -j$(nproc) channel 2>&1 | tail -3
    cd ..
fi

SOLVER=$BUILD_DIR/channel

echo ""
echo "CaLES reference (Xiao et al. 2024):"
echo "  1x A100-64GB, 512x384x1440 (283M cells): 0.72 ns/step/point (DNS explicit)"
echo ""

# Test grids from small to large
for GRID in "128 128 128" "256 256 256" "512 256 144" "384 256 256" "512 256 256" "512 384 256"; do
    read -r NX NY NZ <<< "$GRID"
    CELLS=$((NX * NY * NZ))

    cat > /tmp/bench_large.cfg << CFGEOF
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
    OUTPUT=$(timeout 180 $SOLVER --config /tmp/bench_large.cfg 2>&1)
    MS=$(echo "$OUTPUT" | grep "solver_step" | head -1 | awk '{print $4}')
    if [ -n "$MS" ]; then
        python3 -c "
cells=$CELLS; ms=float('$MS')
ns_pt = ms * 1e6 / cells
mcells = cells / (ms/1000) / 1e6
print(f'  {ms:.2f} ms/step, {mcells:.0f} Mcells/s, {ns_pt:.2f} ns/point')
"
    else
        echo "  Failed (likely OOM)"
    fi
done
