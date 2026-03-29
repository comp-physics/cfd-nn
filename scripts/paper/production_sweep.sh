#!/bin/bash
# Production a posteriori sweep: 15 models × 4 cases
# SST warm-up → model stabilization → long evaluation with QoI extraction
# Run on H200 GPU
set -uo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=${1:-build_h200}
OUTDIR=results/paper/production_sweep
mkdir -p $OUTDIR

echo "=== Production A Posteriori Sweep ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo ""

ALL_MODELS=(
    "none||baseline"
    "baseline||mixing_length"
    "komega||komega"
    "sst||sst"
    "earsm_wj||earsm_wj"
    "earsm_gs||earsm_gs"
    "earsm_pope||earsm_pope"
    "gep||gep"
    "nn_mlp|mlp_paper|mlp"
    "nn_mlp|mlp_med_paper|mlp_med"
    "nn_tbnn|tbnn_small_paper|tbnn_small"
    "nn_tbnn|tbnn_paper|tbnn"
    "nn_tbnn|pi_tbnn_small_paper|pi_tbnn_small"
    "nn_tbnn|pi_tbnn_paper|pi_tbnn"
    "nn_tbrf|tbrf_1t_paper|tbrf_1t"
)

# Case definitions: name:binary:base_cfg:warmup_time:T_final:max_steps
CASES=(
    "hills_re10595:$BUILD/hills:examples/paper_experiments/hills_re10595.cfg:5.0:30.0:200000"
    "cylinder_re100:$BUILD/cylinder:examples/paper_experiments/cylinder_re100.cfg:5.0:50.0:200000"
    "duct_reb3500:$BUILD/duct:examples/paper_experiments/duct_reb3500.cfg:3.0:20.0:100000"
    "sphere_re200:$BUILD/cylinder:examples/paper_experiments/sphere_re200.cfg:5.0:20.0:50000"
)

for CASE_SPEC in "${CASES[@]}"; do
    IFS=':' read -r CASE BIN BASE_CFG WARMUP TFINAL MAXSTEPS <<< "$CASE_SPEC"

    if [ ! -f "$BIN" ]; then
        echo "=== $CASE === SKIP (binary $BIN not found)"
        continue
    fi

    echo ""
    echo "=== $CASE (warmup=$WARMUP, T_final=$TFINAL) ==="
    printf "%-16s %9s %6s %9s %7s %-12s %s\n" \
        "MODEL" "eval_wall" "steps" "ms/step" "turb%" "residual" "QoI"

    for MODEL_SPEC in "${ALL_MODELS[@]}"; do
        IFS='|' read -r TURB PRESET LABEL <<< "$MODEL_SPEC"

        DIR=$OUTDIR/${CASE}/${LABEL}
        mkdir -p $DIR
        CFG=$DIR/run.cfg
        sed "s/T_final = .*/T_final = $TFINAL/;s/max_steps = .*/max_steps = $MAXSTEPS/" $BASE_CFG > $CFG
        echo "turb_model = $TURB" >> $CFG
        echo "warmup_model = sst" >> $CFG
        echo "warmup_time = $WARMUP" >> $CFG
        echo "warmup_steps = 200" >> $CFG
        echo "qoi_output_dir = $DIR/qoi" >> $CFG
        echo "output_dir = $DIR/" >> $CFG
        [ -n "$PRESET" ] && echo "nn_preset = $PRESET" >> $CFG

        OUTPUT=$($BIN --config $CFG 2>&1) || true
        echo "$OUTPUT" > $DIR/output.log

        # Extract metrics
        WALL=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $2}')
        STEPS=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $3}')
        MSTP=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $4}')
        TURBT=$(echo "$OUTPUT" | grep "^turbulence_update" | awk '{print $2}')
        LAST_RES=$(echo "$OUTPUT" | grep -oP 'res=\K\S+' | tail -1)

        TPCT="0"
        [ -n "$TURBT" ] && [ -n "$WALL" ] && TPCT=$(python3 -c "print(f'{100*float(\"$TURBT\")/max(0.001,float(\"$WALL\")):.0f}')" 2>/dev/null || echo "?")

        # QoI
        QOI=""
        if [ -f "$DIR/qoi/qoi_summary.dat" ]; then
            CD=$(grep "Cd_mean" $DIR/qoi/qoi_summary.dat 2>/dev/null | awk '{print $2}')
            ST=$(grep "^St " $DIR/qoi/qoi_summary.dat 2>/dev/null | awk '{print $2}')
            SEP=$(grep "sep_angle" $DIR/qoi/qoi_summary.dat 2>/dev/null | awk '{print $2}')
            [ -n "$CD" ] && QOI="Cd=$CD"
            [ -n "$ST" ] && [ "$ST" != "-1" ] && QOI="$QOI St=$ST"
            [ -n "$SEP" ] && [ "$SEP" != "-1" ] && QOI="$QOI sep=$SEP"
        fi
        XSEP=$(echo "$OUTPUT" | grep "Separation:" | sed 's/.*x\/H = //' | head -1)
        [ -n "$XSEP" ] && QOI="sep=$XSEP"
        XREA=$(echo "$OUTPUT" | grep "Reattachment:" | sed 's/.*x\/H = //' | head -1)
        [ -n "$XREA" ] && QOI="$QOI reat=$XREA"

        STOP=""
        echo "$OUTPUT" | grep -q "T_final" && STOP="[T]"
        echo "$OUTPUT" | grep -q "STOPPING" && STOP="[DIV]"

        printf "%-16s %9s %6s %9s %6s%% %-12s %s %s\n" \
            "$LABEL" "${WALL:-?}" "${STEPS:-?}" "${MSTP:-?}" "$TPCT" "${LAST_RES:-?}" "$QOI" "$STOP"
    done
done

echo ""
echo "=== Done $(date) ==="
echo "Results in: $OUTDIR/"
