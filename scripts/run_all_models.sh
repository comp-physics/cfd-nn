#!/bin/bash
# Run all turbulence models on the 3 V100-compatible cases.
# Usage: ./scripts/run_all_models.sh [build_dir]
# Produces a summary table for each case.

set -euo pipefail

BUILD=${1:-build_gpu}
NSTEPS=${2:-1000}

if [ ! -f "$BUILD/cylinder" ]; then
    echo "ERROR: $BUILD/cylinder not found. Build first."
    exit 1
fi

export OMP_TARGET_OFFLOAD=MANDATORY

# Model lists
CLASSICAL="none baseline gep sst earsm_wj earsm_gs earsm_pope rsm"
NN_SPECS="nn_mlp:mlp_paper nn_mlp:mlp_med_paper nn_mlp:mlp_large_paper"
NN_SPECS="$NN_SPECS nn_tbnn:tbnn_paper nn_tbnn:tbnn_small_paper nn_tbnn:tbnn_large_paper"
NN_SPECS="$NN_SPECS nn_tbnn:pi_tbnn_paper nn_tbnn:pi_tbnn_small_paper nn_tbnn:pi_tbnn_large_paper"
NN_SPECS="$NN_SPECS nn_tbrf:tbrf_1t_paper nn_tbrf:tbrf_5t_paper nn_tbrf:tbrf_10t_paper"

# Skip EARSM on duct (known divergence - see CLAUDE.md)
DUCT_CLASSICAL="none baseline gep sst rsm"

run_model() {
    local binary=$1 config=$2 model=$3 weights=$4 nsteps=$5
    local cmd="$BUILD/$binary --config $config --max_steps $nsteps --model $model"
    [ -n "$weights" ] && cmd="$cmd --weights data/models/$weights"
    timeout 600 $cmd 2>&1
}

parse_cylinder_cd() {
    grep "Cd=" | tail -1 | sed 's/.*Cd=//' | cut -d' ' -f1 | tr -d ','
}

parse_duct_res() {
    grep "res=" | tail -1 | sed 's/.*res=//' | cut -d' ' -f1
}

parse_hills_ub() {
    grep "U_b=" | tail -1 | sed 's/.*U_b=//'
}

# ============================================================
echo "========== CYLINDER Re=100 ($NSTEPS steps, no warmup) =========="
printf "%-22s %12s %12s %8s\n" "Model" "Cd" "Cl" "Status"
echo "------------------------------------------------------"

for m in $CLASSICAL; do
    line=$(run_model cylinder examples/paper_experiments/cylinder_re100.cfg $m "" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1 | tr -d ',')
    cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$m" "$cd" "$cl" "$status"
done

for spec in $NN_SPECS; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    line=$(run_model cylinder examples/paper_experiments/cylinder_re100.cfg $cli "$wt" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1 | tr -d ',')
    cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$wt" "$cd" "$cl" "$status"
done

# ============================================================
echo ""
echo "========== DUCT Re_b=3500 ($NSTEPS steps, warmup=20s) =========="
printf "%-22s %12s %8s\n" "Model" "Residual" "Status"
echo "--------------------------------------------"

for m in $DUCT_CLASSICAL; do
    res=$(run_model duct examples/paper_experiments/duct_reb3500.cfg $m "" $NSTEPS | parse_duct_res)
    nan=$(run_model duct examples/paper_experiments/duct_reb3500.cfg $m "" $NSTEPS 2>&1 | grep -c "NaN" || true)
    status="OK"; [ -z "$res" ] && status="TIMEOUT"; [ "$nan" -gt 0 ] 2>/dev/null && status="NaN"
    printf "%-22s %12s %8s\n" "$m" "${res:-???}" "$status"
done

for spec in $NN_SPECS; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    res=$(run_model duct examples/paper_experiments/duct_reb3500.cfg $cli "$wt" $NSTEPS | parse_duct_res)
    status="OK"; [ -z "$res" ] && status="TIMEOUT"
    printf "%-22s %12s %8s\n" "$wt" "${res:-???}" "$status"
done

echo ""
echo "NOTE: EARSM skipped on duct (known divergence with explicit solver)"
echo "NOTE: Hills runs are too slow for V100 (~15 min/model with warmup)"
echo "NOTE: Sphere needs H200 (25M cells, 25GB)"
