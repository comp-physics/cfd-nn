#!/bin/bash
#SBATCH -J aposteriori-channel
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 4:00:00
#SBATCH -o results/paper/aposteriori/channel/slurm_%j.out
#SBATCH -e results/paper/aposteriori/channel/slurm_%j.err

set -euo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn
module load nvhpc/24.5
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "============================================"
echo "  A Posteriori Channel Flow (H200)"
echo "  GPU: $GPU"
echo "  $(date)"
echo "============================================"

# Build for H200 (CC=90)
BUILD_DIR=build_h200
if [ ! -f $BUILD_DIR/channel ]; then
    echo "Building..."
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90 2>&1 | tail -3
    make -j$(nproc) channel 2>&1 | tail -3
    cd ..
fi
SOLVER=$BUILD_DIR/channel

RESULTS=results/paper/aposteriori/channel
mkdir -p $RESULTS

# Base config (no turb model — each model adds its own)
BASE_CFG=/tmp/channel_base.cfg
cat > $BASE_CFG << 'CFGEOF'
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
max_steps = 15000
verbose = false
output_freq = 1000
write_fields = true
num_snapshots = 5
postprocess = true
warmup_steps = 50
CFGEOF

# Run each model with its own config file
for LABEL in baseline sst mlp tbnn tbrf1t; do
    OUTDIR=$RESULTS/$LABEL
    mkdir -p $OUTDIR
    CFG=/tmp/channel_${LABEL}.cfg
    cp $BASE_CFG $CFG

    case $LABEL in
        baseline) echo "turb_model = none" >> $CFG ;;
        sst)      echo "turb_model = sst" >> $CFG ;;
        mlp)      echo "turb_model = nn_mlp" >> $CFG
                  echo "nn_preset = mlp_paper" >> $CFG ;;
        tbnn)     echo "turb_model = nn_tbnn" >> $CFG
                  echo "nn_preset = tbnn_paper" >> $CFG ;;
        tbrf1t)   echo "turb_model = nn_tbrf" >> $CFG
                  echo "nn_preset = tbrf_1t_paper" >> $CFG ;;
    esac

    echo ""
    echo "============================================"
    echo "  $LABEL — $(date)"
    echo "============================================"

    $SOLVER --config $CFG --output $OUTDIR/ 2>&1 | tee $OUTDIR/run.log

    echo "  Exit: ${PIPESTATUS[0]}"
    echo "  Done: $(date)"
done

echo ""
echo "============================================"
echo "  All complete — $(date)"
echo "============================================"
