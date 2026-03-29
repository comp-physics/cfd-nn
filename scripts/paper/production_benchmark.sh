#!/bin/bash
# Production benchmark: 20 models × 8 configs
# Reports wall-time-per-physical-time with component breakdown
# Cost metric: how many wall-clock seconds to simulate 1 second of physics
set -uo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=${1:-build_h200}
OUTDIR=results/paper/production_benchmark
mkdir -p $OUTDIR

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== Production Benchmark: 20 models × 8 configs ==="
echo "GPU: ${GPU:-none}"
echo "Build: $BUILD"
echo "Date: $(date)"
echo ""

SUMMARY=$OUTDIR/summary.csv
echo "case,model,wall_per_phys,total_wall_s,final_t,steps,avg_dt,turb_ms,turb_pct,poisson_pct,convect_pct,diffuse_pct,final_res,qoi" > $SUMMARY

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

# Extract timing metrics from a run output log
extract_metrics() {
    local LOG=$1
    local DIR=$(dirname $LOG)

    # Timing categories from TimingStats (Total column, Calls column, Avg column)
    local solver_total=$(grep "solver_step" "$LOG" | head -1 | awk '{print $2}')
    local solver_calls=$(grep "solver_step" "$LOG" | head -1 | awk '{print $3}')
    local turb_total=$(grep "turbulence_update" "$LOG" | head -1 | awk '{print $2}')
    local turb_avg=$(grep "turbulence_update" "$LOG" | head -1 | awk '{print $4}')
    local poisson_total=$(grep "poisson_solve" "$LOG" | head -1 | awk '{print $2}')
    local convect_total=$(grep "convective_term" "$LOG" | head -1 | awk '{print $2}')
    local diffuse_total=$(grep "diffusive_term" "$LOG" | head -1 | awk '{print $2}')

    # Physical time: get from the last "Step N  t=X.XXXX" line (not T_final line which has scientific notation)
    local final_t=$(grep -oP 'Step\s+\d+\s+t=\K[\d.]+' "$LOG" | tail -1)
    local final_res=$(grep -oP 'res=\K\S+' "$LOG" | tail -1)

    # QoI
    local qoi=""
    if [ -f "$DIR/qoi/qoi_summary.dat" ]; then
        local cd=$(grep "Cd_mean" "$DIR/qoi/qoi_summary.dat" 2>/dev/null | awk '{print $2}')
        local st=$(grep "^St " "$DIR/qoi/qoi_summary.dat" 2>/dev/null | awk '{print $2}')
        local sep=$(grep "sep_angle" "$DIR/qoi/qoi_summary.dat" 2>/dev/null | awk '{print $2}')
        [ -n "$cd" ] && qoi="Cd=$cd"
        [ -n "$st" ] && [ "$st" != "-1" ] && qoi="$qoi St=$st"
        [ -n "$sep" ] && [ "$sep" != "-1" ] && qoi="$qoi sep=${sep}d"
    fi
    # Hills separation from stdout
    local xsep=$(grep "Separation:" "$LOG" | grep -oP '[\d.e+-]+' | head -1)
    local xrea=$(grep "Reattachment:" "$LOG" | grep -oP '[\d.e+-]+' | head -1)
    [ -n "$xsep" ] && qoi="sep=$xsep reat=$xrea"

    # Get evaluation start time (from warm-up output, if present)
    local eval_start_t=$(grep -oP 'Evaluation phase from t=\K[\d.eE+-]+' "$LOG" | head -1)
    [ -z "$eval_start_t" ] && eval_start_t="0"

    # Compute derived metrics
    local wall_per_phys="?" avg_dt="?" turb_pct="?" poisson_pct="?" convect_pct="?" diffuse_pct="?"
    if [ -n "$solver_total" ] && [ -n "$final_t" ] && [ "$final_t" != "0" ]; then
        wall_per_phys=$(python3 -c "
s=float('$solver_total'); t_end=float('$final_t'); t_start=float('$eval_start_t')
dt_phys = t_end - t_start
print(f'{s/dt_phys:.3f}' if dt_phys > 0 else '?')" 2>/dev/null || echo "?")
        avg_dt=$(python3 -c "
t_end=float('$final_t'); t_start=float('$eval_start_t'); n=int('${solver_calls:-1}')
dt_phys = t_end - t_start
print(f'{dt_phys/n:.2e}' if n > 0 and dt_phys > 0 else '?')" 2>/dev/null || echo "?")
    fi
    if [ -n "$solver_total" ] && [ "$solver_total" != "0" ]; then
        turb_pct=$(python3 -c "print(f'{100*float(\"${turb_total:-0}\")/float(\"$solver_total\"):.1f}')" 2>/dev/null || echo "0")
        poisson_pct=$(python3 -c "print(f'{100*float(\"${poisson_total:-0}\")/float(\"$solver_total\"):.1f}')" 2>/dev/null || echo "?")
        convect_pct=$(python3 -c "print(f'{100*float(\"${convect_total:-0}\")/float(\"$solver_total\"):.1f}')" 2>/dev/null || echo "?")
        diffuse_pct=$(python3 -c "print(f'{100*float(\"${diffuse_total:-0}\")/float(\"$solver_total\"):.1f}')" 2>/dev/null || echo "?")
    fi

    # Check for early termination
    local stop=""
    grep -q "STOPPING" "$LOG" && stop="STOPPED"
    grep -q "Reached T_final" "$LOG" && stop="T_FINAL"

    # Export results
    R_WALL_PER_PHYS=$wall_per_phys
    R_STEPS=${solver_calls:-?}
    R_AVG_DT=$avg_dt
    R_TURB_MS=${turb_avg:-0}
    R_TURB_PCT=${turb_pct:-0}
    R_POISSON_PCT=${poisson_pct:-?}
    R_CONVECT_PCT=${convect_pct:-?}
    R_DIFFUSE_PCT=${diffuse_pct:-?}
    R_FINAL_T=${final_t:-?}
    R_FINAL_RES=${final_res:-?}
    R_TOTAL_WALL=${solver_total:-?}
    R_QOI="$qoi"
    R_STOP="$stop"
}

for CASE_SPEC in "${CASES[@]}"; do
    IFS=':' read -r CASE BIN BASE_CFG <<< "$CASE_SPEC"

    if [ ! -f "$BIN" ]; then
        echo "=== $CASE === SKIP (binary not found)"
        continue
    fi

    echo "=== $CASE ==="
    printf "  %-20s %9s %6s %9s %6s %6s %6s %6s %-12s %s\n" \
        "MODEL" "wall/phys" "steps" "avg_dt" "turb%" "pois%" "conv%" "diff%" "residual" "QoI"
    printf "  %-20s %9s %6s %9s %6s %6s %6s %6s %-12s %s\n" \
        "-----" "---------" "-----" "------" "-----" "-----" "-----" "-----" "--------" "---"

    for MODEL_SPEC in "${ALL_MODELS[@]}"; do
        IFS='|' read -r TURB PRESET LABEL <<< "$MODEL_SPEC"

        RUN_DIR=$OUTDIR/${CASE}/${LABEL}
        mkdir -p $RUN_DIR
        CFG=$RUN_DIR/run.cfg
        cp $BASE_CFG $CFG
        echo "turb_model = $TURB" >> $CFG
        echo "qoi_output_dir = $RUN_DIR/qoi" >> $CFG
        echo "output_dir = $RUN_DIR/" >> $CFG
        [ -n "$PRESET" ] && echo "nn_preset = $PRESET" >> $CFG

        # Run
        OUTPUT=$($BIN --config $CFG 2>&1) || true
        echo "$OUTPUT" > $RUN_DIR/output.log

        # Extract metrics
        extract_metrics $RUN_DIR/output.log

        # Print
        printf "  %-20s %9s %6s %9s %5s%% %5s%% %5s%% %5s%% %-12s %s %s\n" \
            "$LABEL" "$R_WALL_PER_PHYS" "$R_STEPS" "$R_AVG_DT" \
            "$R_TURB_PCT" "$R_POISSON_PCT" "$R_CONVECT_PCT" "$R_DIFFUSE_PCT" \
            "$R_FINAL_RES" "$R_QOI" "$R_STOP"

        # CSV
        echo "$CASE,$LABEL,$R_WALL_PER_PHYS,$R_TOTAL_WALL,$R_FINAL_T,$R_STEPS,$R_AVG_DT,$R_TURB_MS,$R_TURB_PCT,$R_POISSON_PCT,$R_CONVECT_PCT,$R_DIFFUSE_PCT,$R_FINAL_RES,$R_QOI $R_STOP" >> $SUMMARY
    done
    echo ""
done

echo "========================================="
echo "Summary CSV: $SUMMARY"
echo "Done: $(date)"
