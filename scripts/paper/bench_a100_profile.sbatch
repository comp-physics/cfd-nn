#!/bin/bash
#SBATCH -J bench-a100-profile
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-a100
#SBATCH -G 1
#SBATCH --qos=inferno
#SBATCH -t 00:15:00
#SBATCH -o results/paper/tables/bench_a100_profile_%j.out
#SBATCH -e results/paper/tables/bench_a100_profile_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64
export OMP_TARGET_OFFLOAD=MANDATORY

# Enable NVHPC acc profiling
export NVCOMPILER_ACC_TIME=1

GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "GPU: $GPU"
echo "NVCOMPILER_ACC_TIME=1 (kernel-level profiling)"
echo ""

SOLVER=build_a100_bench/channel
if [ ! -f "$SOLVER" ]; then
    SOLVER=build_h100/channel
fi

cat > /tmp/bench_profile.cfg << 'CFGEOF'
Nx = 256
Ny = 256
Nz = 256
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
max_steps = 50
gpu_only_mode = true
perf_mode = true
benchmark = true
CFGEOF

echo "--- 256^3 = 16777216 cells, 50 steps ---"
$SOLVER --config /tmp/bench_profile.cfg 2>&1
