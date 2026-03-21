#!/bin/bash
# Submit all periodic hills experiments: 5 models x 4 grids = 20 jobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build_paper"
RESULTS_BASE="$ROOT_DIR/results/paper/hills"
CONFIG_DIR="$ROOT_DIR/examples/paper_experiments"

MODELS=(baseline sst earsm_pope nn_mlp nn_tbnn)
GRIDS=(D A B C)

# Time limits per grid level
declare -A TIME_LIMITS
TIME_LIMITS[D]="6:00:00"
TIME_LIMITS[A]="3:00:00"
TIME_LIMITS[B]="1:00:00"
TIME_LIMITS[C]="0:30:00"

# NN preset flags for hills
declare -A NN_FLAGS
NN_FLAGS[baseline]=""
NN_FLAGS[sst]=""
NN_FLAGS[earsm_pope]=""
NN_FLAGS[nn_mlp]="--nn_preset mlp_phll_caseholdout"
NN_FLAGS[nn_tbnn]="--nn_preset tbnn_phll_caseholdout"

# Build if needed
if [ ! -f "$BUILD_DIR/hills" ]; then
    echo "Building solver in $BUILD_DIR ..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$ROOT_DIR" \
        -DCMAKE_CXX_COMPILER=nvc++ \
        -DUSE_GPU_OFFLOAD=ON \
        -DGPU_CC=90 \
        -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)" channel hills
    cd "$ROOT_DIR"
    echo "Build complete."
fi

submitted=0

for model in "${MODELS[@]}"; do
    for grid in "${GRIDS[@]}"; do
        outdir="$RESULTS_BASE/${model}_${grid}"
        mkdir -p "$outdir"

        time_limit="${TIME_LIMITS[$grid]}"
        nn_flags="${NN_FLAGS[$model]}"

        jobname="nn-paper-hills-${model}-${grid}"
        config="$CONFIG_DIR/hills_${grid}.cfg"

        sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH -J ${jobname}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t ${time_limit}
#SBATCH -o ${outdir}/slurm_%j.out
#SBATCH -e ${outdir}/slurm_%j.err

module reset
module load nvhpc

cd ${ROOT_DIR}
export OMP_TARGET_OFFLOAD=MANDATORY

echo "=== JOB INFO ==="
echo "Case: hills"
echo "Model: ${model}"
echo "Grid: ${grid}"
echo "Config: ${config}"
echo "Output: ${outdir}"
echo "Host: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start: \$(date -Iseconds)"
echo ""

./build_paper/hills \
    --config ${config} \
    --model ${model} ${nn_flags} \
    --output ${outdir}/ \
    --warmup_steps 50

echo ""
echo "=== TIMING SUMMARY ==="
echo "End: \$(date -Iseconds)"
SBATCH_EOF

        submitted=$((submitted + 1))
        echo "Submitted: ${jobname} (${time_limit})"
    done
done

echo ""
echo "Total hills jobs submitted: ${submitted}"
