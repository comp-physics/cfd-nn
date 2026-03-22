#!/bin/bash
#SBATCH -J aposteriori-channel
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-h200,gpu-h100,gpu-l40s
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 2:00:00
#SBATCH -o results/paper/aposteriori/channel/slurm_%j.out
#SBATCH -e results/paper/aposteriori/channel/slurm_%j.err

# A posteriori evaluation: run channel flow Re_tau=180 with multiple
# turbulence models and compare against MKM DNS data.

set -euo pipefail

ROOT_DIR="/storage/scratch1/6/sbryngelson3/cfd-nn"
BUILD_DIR="$ROOT_DIR/build_paper"
CONFIG="$ROOT_DIR/examples/paper_experiments/channel_A.cfg"
RESULTS_BASE="$ROOT_DIR/results/paper/aposteriori/channel"

module reset
module load nvhpc

export OMP_TARGET_OFFLOAD=MANDATORY

cd "$ROOT_DIR"

echo "=== A POSTERIORI CHANNEL EVALUATION ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start: $(date -Iseconds)"
echo ""

# Build if needed
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "Building solver in $BUILD_DIR ..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$ROOT_DIR" \
        -DCMAKE_CXX_COMPILER=nvc++ \
        -DUSE_GPU_OFFLOAD=ON \
        -DGPU_CC=90 \
        -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)" channel
    cd "$ROOT_DIR"
    echo "Build complete."
    echo ""
fi

# ============================================================================
# Model definitions: (name, --model flag, extra CLI args)
# ============================================================================
MODELS=(
    "none"
    "sst"
    "earsm_pope"
    "nn_mlp"
    "nn_tbnn"
)

declare -A MODEL_FLAGS
MODEL_FLAGS[none]="--model none"
MODEL_FLAGS[sst]="--model sst"
MODEL_FLAGS[earsm_pope]="--model earsm_pope"
MODEL_FLAGS[nn_mlp]="--model nn_mlp --nn_preset mlp_paper"
MODEL_FLAGS[nn_tbnn]="--model nn_tbnn --nn_preset tbnn_paper"

# ============================================================================
# Run each model sequentially
# ============================================================================
total=${#MODELS[@]}
count=0

for model in "${MODELS[@]}"; do
    count=$((count + 1))
    outdir="$RESULTS_BASE/$model"
    mkdir -p "$outdir"

    flags="${MODEL_FLAGS[$model]}"

    echo "============================================================"
    echo "[$count/$total] Running model: $model"
    echo "  Output: $outdir"
    echo "  Flags: $flags"
    echo "  Time: $(date -Iseconds)"
    echo "============================================================"

    "$BUILD_DIR/channel" \
        --config "$CONFIG" \
        $flags \
        --scheme skew \
        --integrator rk3 \
        --output "$outdir/" \
        --warmup_steps 50 \
        2>&1 | tee "$outdir/run.log"

    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "WARNING: model $model exited with code $exit_code"
    fi

    echo ""
    echo "Model $model completed at $(date -Iseconds)"
    echo ""
done

echo "============================================================"
echo "All models completed at $(date -Iseconds)"
echo "Results in: $RESULTS_BASE"
echo "============================================================"
