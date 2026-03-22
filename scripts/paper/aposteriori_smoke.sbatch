#!/bin/bash
#SBATCH -J aposteriori-smoke
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:30:00
#SBATCH -o results/paper/aposteriori/smoke_%j.out
#SBATCH -e results/paper/aposteriori/smoke_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn

NVHPC_ROOT=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"

SOLVER=build_h200/channel

cat > /tmp/channel_smoke.cfg << 'CFGEOF'
Nx = 128
Ny = 96
Nz = 128
x_min = 0.0
x_max = 6.283185
y_min = -1.0
y_max = 1.0
z_min = 0.0
z_max = 3.141593
nu = 0.005556
dp_dx = -1.0
convective_scheme = skew
time_integrator = rk3
adaptive_dt = true
CFL_max = 0.5
stretch_y = true
stretch_beta = 2.0
simulation_mode = steady
tol = 1e-8
max_steps = 100
verbose = false
output_freq = 50
warmup_steps = 10
CFGEOF

for MODEL_SPEC in \
    "none||baseline" \
    "sst||sst" \
    "nn_mlp|--nn_preset mlp_paper|mlp" \
    "nn_tbnn|--nn_preset tbnn_paper|tbnn" \
    "nn_tbrf|--nn_preset tbrf_1t_paper|tbrf1t"; do

    IFS='|' read -r TURB EXTRA LABEL <<< "$MODEL_SPEC"

    echo ""
    echo "=== $LABEL ==="
    $SOLVER --config /tmp/channel_smoke.cfg --turb_model $TURB $EXTRA 2>&1 | \
        grep -E "solver_step|turbulence|nn_|Converged|residual|Re_tau|Wall shear|Timing"
    echo "  exit: $?"
done
