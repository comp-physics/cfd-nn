#!/bin/bash
# Smoke test: 20 models × 8 configs = 160 runs
# Validates: runs without crash, residuals finite, QoI files written,
# models actually affect the flow (not silently ignored)
set -uo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=${1:-build_gpu}
NSTEPS=50
OUTDIR=results/paper/smoke_test
mkdir -p $OUTDIR

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== Smoke Test: 20 models × 8 configs ==="
echo "GPU: ${GPU:-none}"
echo "Build: $BUILD"
echo "Steps: $NSTEPS"
echo "Date: $(date)"
echo ""

PASS=0
FAIL=0
SUMMARY_FILE=$OUTDIR/summary.csv
echo "case,model,ms_per_step,turb_ms,final_residual,status,notes" > $SUMMARY_FILE

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
    "nn_mlp|mlp_large_paper|mlp_large"
    "nn_tbnn|tbnn_small_paper|tbnn_small"
    "nn_tbnn|tbnn_paper|tbnn"
    "nn_tbnn|tbnn_large_paper|tbnn_large"
    "nn_tbnn|pi_tbnn_small_paper|pi_tbnn_small"
    "nn_tbnn|pi_tbnn_paper|pi_tbnn"
    "nn_tbnn|pi_tbnn_large_paper|pi_tbnn_large"
    "nn_tbrf|tbrf_1t_paper|tbrf_1t"
    "nn_tbrf|tbrf_5t_paper|tbrf_5t"
    "nn_tbrf|tbrf_10t_paper|tbrf_10t"
)

CASES=(
    "hills_re10595:$BUILD/hills:examples/paper_experiments/hills_re10595.cfg"
    "cylinder_re100:$BUILD/cylinder:examples/paper_experiments/cylinder_re100.cfg"
    "cylinder_re300:$BUILD/cylinder:examples/paper_experiments/cylinder_re300.cfg"
    "cylinder_re3900:$BUILD/cylinder:examples/paper_experiments/cylinder_re3900.cfg"
    "duct_reb3500:$BUILD/duct:examples/paper_experiments/duct_reb3500.cfg"
    "sphere_re100:$BUILD/cylinder:examples/paper_experiments/sphere_re100.cfg"
    "sphere_re200:$BUILD/cylinder:examples/paper_experiments/sphere_re200.cfg"
    "sphere_re300:$BUILD/cylinder:examples/paper_experiments/sphere_re300.cfg"
)

for CASE_SPEC in "${CASES[@]}"; do
    IFS=':' read -r CASE BIN BASE_CFG <<< "$CASE_SPEC"

    if [ ! -f "$BIN" ]; then
        echo "=== $CASE === SKIP (binary $BIN not found)"
        continue
    fi

    echo ""
    echo "=== $CASE ==="

    for MODEL_SPEC in "${ALL_MODELS[@]}"; do
        IFS='|' read -r TURB PRESET LABEL <<< "$MODEL_SPEC"

        RUN_DIR=$OUTDIR/${CASE}/${LABEL}
        mkdir -p $RUN_DIR
        CFG=$RUN_DIR/run.cfg
        cp $BASE_CFG $CFG
        echo "turb_model = $TURB" >> $CFG
        echo "max_steps = $NSTEPS" >> $CFG
        echo "qoi_output_dir = $RUN_DIR/qoi" >> $CFG
        echo "output_dir = $RUN_DIR/" >> $CFG
        [ -n "$PRESET" ] && echo "nn_preset = $PRESET" >> $CFG

        printf "  %-20s" "$LABEL"
        OUTPUT=$($BIN --config $CFG 2>&1) || true
        echo "$OUTPUT" > $RUN_DIR/output.log

        # Extract metrics
        STEP_MS=$(echo "$OUTPUT" | grep "solver_step" | head -1 | awk '{print $4}')
        TURB_MS=$(echo "$OUTPUT" | grep "turbulence_update" | head -1 | awk '{print $4}')
        LAST_RES=$(echo "$OUTPUT" | grep -oP 'res=\K\S+' | tail -1)
        ERR=$(echo "$OUTPUT" | grep -iE "terminate|FATAL|Segmentation|Abort" | head -1)

        # Check for REAL nan/inf: in residual values, Cd/Cl values, or QoI files
        REAL_NAN=0
        # Check residuals for nan/inf (standalone word, not substring)
        if echo "$OUTPUT" | grep -qP 'res=\s*-?nan|res=\s*-?inf'; then
            REAL_NAN=1
        fi
        # Check Cd/Cl for nan/inf
        if echo "$OUTPUT" | grep -qP 'Cd=\s*-?nan|Cl=\s*-?nan|Cd=\s*-?inf|Cl=\s*-?inf'; then
            REAL_NAN=1
        fi
        # Check QoI summary for nan/inf
        if [ -f "$RUN_DIR/qoi/qoi_summary.dat" ]; then
            if grep -qP '\bnan\b|\binf\b' "$RUN_DIR/qoi/qoi_summary.dat" 2>/dev/null; then
                REAL_NAN=1
            fi
        fi
        # Check wake profiles for nan/inf
        for wf in $RUN_DIR/qoi/wake_*.dat; do
            [ -f "$wf" ] && grep -qP '\bnan\b|\binf\b' "$wf" 2>/dev/null && REAL_NAN=1
        done
        # Check forces.dat for nan/inf
        if [ -f "$RUN_DIR/forces.dat" ]; then
            if grep -qP '\bnan\b|\binf\b' "$RUN_DIR/forces.dat" 2>/dev/null; then
                REAL_NAN=1
            fi
        fi

        # Determine status
        STATUS="ok"
        NOTES=""
        if [ -n "$ERR" ]; then
            STATUS="CRASH"
            NOTES="$ERR"
        elif [ -z "$STEP_MS" ]; then
            STATUS="NO_OUTPUT"
            NOTES="no timing data"
        elif [ "$REAL_NAN" -eq 1 ]; then
            STATUS="NaN"
            NOTES="nan/inf in output or QoI"
        fi

        # Print
        if [ "$STATUS" = "ok" ]; then
            printf "%6s ms/step  turb=%6s ms  res=%-12s OK\n" "$STEP_MS" "${TURB_MS:-0}" "${LAST_RES:-?}"
            PASS=$((PASS + 1))
        else
            printf "%6s ms/step  turb=%6s ms  res=%-12s **%s** %s\n" \
                "${STEP_MS:-?}" "${TURB_MS:-?}" "${LAST_RES:-?}" "$STATUS" "$NOTES"
            FAIL=$((FAIL + 1))
        fi

        echo "$CASE,$LABEL,${STEP_MS:-?},${TURB_MS:-0},${LAST_RES:-?},$STATUS,$NOTES" >> $SUMMARY_FILE
    done
done

echo ""
echo "========================================="
echo "Results: $PASS ok, $FAIL problems (of $((PASS + FAIL)) total)"
echo "========================================="

# Final fidelity check: do different models produce different residuals?
echo ""
echo "=== Fidelity Check: Do models affect the flow? ==="
for case_dir in $OUTDIR/*/; do
    CASE=$(basename $case_dir)
    RESIDS=""
    for model_dir in $case_dir*/; do
        MODEL=$(basename $model_dir)
        RES=$(grep -oP 'res=\K\S+' "$model_dir/output.log" 2>/dev/null | tail -1)
        RESIDS="$RESIDS $RES"
    done
    # Count unique residuals
    N_UNIQUE=$(echo $RESIDS | tr ' ' '\n' | sort -u | wc -l)
    N_TOTAL=$(echo $RESIDS | tr ' ' '\n' | grep -c . || true)
    if [ "$N_UNIQUE" -le 2 ] && [ "$N_TOTAL" -gt 5 ]; then
        echo "  WARNING $CASE: only $N_UNIQUE unique residuals across $N_TOTAL models — models may not be loading"
    else
        echo "  $CASE: $N_UNIQUE unique residuals across $N_TOTAL models — OK"
    fi
done

# QoI file check
echo ""
echo "=== QoI File Check ==="
for CASE in hills_re10595 cylinder_re100 duct_reb3500 sphere_re200; do
    qoi_dir="$OUTDIR/$CASE/baseline/qoi"
    if [ -d "$qoi_dir" ]; then
        nfiles=$(ls "$qoi_dir" 2>/dev/null | wc -l)
        echo "  $CASE/baseline/qoi: $nfiles files"
        ls "$qoi_dir" 2>/dev/null | sed 's/^/    /'
    else
        echo "  $CASE/baseline/qoi: MISSING"
    fi
done

echo ""
echo "Summary CSV: $OUTDIR/summary.csv"
echo "Done: $(date)"
exit $FAIL
