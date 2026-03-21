#!/bin/bash
#
# Smoke test: verify all 5 models × 2 cases are numerically stable
# Run on the coarsest grid (C) for 200 steps each.
# Should complete in ~5 minutes on a GPU node.
#
# Usage: ./scripts/paper/smoke_test.sh [build_dir]
#   build_dir: path to GPU build directory (default: build_paper)
#
# Run this BEFORE submitting the full experiment matrix!

set -euo pipefail

BUILD_DIR="${1:-build_paper}"
RESULTS_DIR="results/paper/smoke_test"
CHANNEL_BIN="$BUILD_DIR/channel"
HILLS_BIN="$BUILD_DIR/hills"

# Models and their NN preset flags
declare -A CHANNEL_FLAGS=(
    [baseline]=""
    [sst]=""
    [earsm_pope]=""
    [nn_mlp]="--nn_preset mlp_channel_caseholdout"
    [nn_tbnn]="--nn_preset tbnn_channel_caseholdout"
)

declare -A HILLS_FLAGS=(
    [baseline]=""
    [sst]=""
    [earsm_pope]=""
    [nn_mlp]="--nn_preset mlp_phll_caseholdout"
    [nn_tbnn]="--nn_preset tbnn_phll_caseholdout"
)

MODELS=(baseline sst earsm_pope nn_mlp nn_tbnn)

# Build if needed
if [ ! -f "$CHANNEL_BIN" ] || [ ! -f "$HILLS_BIN" ]; then
    echo "=== Building solver ==="
    module reset 2>/dev/null || true
    module load nvhpc 2>/dev/null || true
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 \
             -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j8 channel hills 2>&1 | tail -3
    cd ..
    echo ""
fi

echo "=========================================="
echo "  SMOKE TEST: Stability Check"
echo "  5 models × 2 cases × 200 steps"
echo "=========================================="
echo ""

mkdir -p "$RESULTS_DIR"

PASS=0
FAIL=0
FAILED_RUNS=""

run_test() {
    local case_name=$1
    local model=$2
    local binary=$3
    local config=$4
    local extra_flags=$5
    local outdir="$RESULTS_DIR/${case_name}_${model}"

    mkdir -p "$outdir"

    printf "  %-12s %-12s ... " "$case_name" "$model"

    # Run 200 steps on coarsest grid
    if timeout 120 "$binary" \
        --config "$config" \
        --model "$model" $extra_flags \
        --max_steps 200 \
        --output "$outdir/" \
        --no_write_fields \
        > "$outdir/stdout.log" 2>&1; then

        # Check for NaN/Inf in output
        if grep -qiE "NaN detected|Inf detected|blow.?up|FATAL|NUMERICAL STABILITY" "$outdir/stdout.log"; then
            echo "FAIL (NaN/Inf detected)"
            FAIL=$((FAIL + 1))
            FAILED_RUNS="$FAILED_RUNS  ${case_name}/${model}\n"
        else
            # Extract final residual
            local resid=$(grep "Final residual" "$outdir/stdout.log" | tail -1 | awk '{print $NF}')
            local step_ms=$(grep "solver_step" "$outdir/stdout.log" | awk '{print $4}')
            echo "PASS  (resid=${resid:-?}, step=${step_ms:-?} ms)"
            PASS=$((PASS + 1))
        fi
    else
        echo "FAIL (crashed or timed out)"
        FAIL=$((FAIL + 1))
        FAILED_RUNS="$FAILED_RUNS  ${case_name}/${model}\n"
        # Print last 5 lines of output for diagnosis
        tail -5 "$outdir/stdout.log" 2>/dev/null | sed 's/^/    /'
    fi
}

echo "--- Channel flow (32×32×32, 200 steps) ---"
for model in "${MODELS[@]}"; do
    run_test "channel" "$model" "$CHANNEL_BIN" \
        "examples/paper_experiments/channel_C.cfg" \
        "${CHANNEL_FLAGS[$model]}"
done
echo ""

echo "--- Periodic hills (32×16×16, 200 steps) ---"
for model in "${MODELS[@]}"; do
    run_test "hills" "$model" "$HILLS_BIN" \
        "examples/paper_experiments/hills_C.cfg" \
        "${HILLS_FLAGS[$model]}"
done
echo ""

echo "=========================================="
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "FAILED RUNS:"
    printf "$FAILED_RUNS"
    echo ""
    echo "Fix these before submitting the full experiment matrix!"
    exit 1
else
    echo ""
    echo "All stable. Safe to submit full experiments:"
    echo "  ./scripts/paper/submit_all.sh"
fi
