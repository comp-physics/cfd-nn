#!/bin/bash
# Submit one SLURM job per model × case (10 min each)
# Short validation to verify: builds, runs, produces QoI files, no NaN
set -euo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
OUTDIR=results/paper/validation_grid
mkdir -p $OUTDIR

MODELS=(
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

# case:binary_name:base_cfg:warmup_time:T_final_short
CASES=(
    "hills_re10595:hills:examples/paper_experiments/hills_re10595.cfg:2.0:5.0"
    "cylinder_re100:cylinder:examples/paper_experiments/cylinder_re100.cfg:2.0:5.0"
    "duct_reb3500:duct:examples/paper_experiments/duct_reb3500.cfg:1.0:3.0"
    "sphere_re200:cylinder:examples/paper_experiments/sphere_re200.cfg:2.0:5.0"
)

NJOBS=0

for CASE_SPEC in "${CASES[@]}"; do
    IFS=':' read -r CASE BINNAME BASE_CFG WARMUP TFINAL <<< "$CASE_SPEC"

    for MODEL_SPEC in "${MODELS[@]}"; do
        IFS='|' read -r TURB PRESET LABEL <<< "$MODEL_SPEC"

        DIR=$OUTDIR/${CASE}/${LABEL}
        mkdir -p $DIR

        SCRIPT=$DIR/job.sbatch
        cat > $SCRIPT << SBEOF
#!/bin/bash
#SBATCH -J val-${CASE:0:4}-${LABEL}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:10:00
#SBATCH -o ${DIR}/slurm_%j.out
#SBATCH -e ${DIR}/slurm_%j.err

set -uo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=build_h200
if [ ! -f \$BUILD/$BINNAME ]; then
    mkdir -p \$BUILD && cd \$BUILD
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF 2>&1 | tail -2
    make -j\$(nproc) hills duct cylinder 2>&1 | tail -3
    cd ..
fi

DIR_RUN=${DIR}
CFG=\$DIR_RUN/run.cfg
sed "s/T_final = .*/T_final = ${TFINAL}/;s/max_steps = .*/max_steps = 50000/" ${BASE_CFG} > \$CFG
echo "turb_model = ${TURB}" >> \$CFG
echo "warmup_model = sst" >> \$CFG
echo "warmup_time = ${WARMUP}" >> \$CFG
echo "warmup_steps = 50" >> \$CFG
echo "qoi_output_dir = \$DIR_RUN/qoi" >> \$CFG
echo "output_dir = \$DIR_RUN/" >> \$CFG
SBEOF
        [ -n "$PRESET" ] && echo "echo \"nn_preset = $PRESET\" >> \$CFG" >> $SCRIPT

        cat >> $SCRIPT << 'SBEOF'

echo "=== Config ==="
grep -E "turb_model|warmup|T_final|max_steps|qoi_freq|ibm_eta|dt_min|output_freq" $CFG

echo ""
echo "=== Running ==="
OUTPUT=$($BUILD/${BINNAME} --config $CFG 2>&1) || true
echo "$OUTPUT" > $DIR_RUN/output.log

# Report key metrics
WALL=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $2}')
STEPS=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $3}')
MSTP=$(echo "$OUTPUT" | grep "^solver_step" | awk '{print $4}')
LAST_RES=$(echo "$OUTPUT" | grep -oP 'res=\K\S+' | tail -1)
echo ""
echo "=== Results ==="
echo "wall=$WALL steps=$STEPS ms/step=$MSTP res=$LAST_RES"

# Check QoI files
echo ""
echo "=== QoI files ==="
ls -la $DIR_RUN/qoi/ 2>/dev/null || echo "NO QOI DIR"
ls -la $DIR_RUN/forces.dat 2>/dev/null || echo "no forces.dat (expected for non-IBM)"

# Check for NaN/divergence
echo ""
echo "=== Health ==="
echo "$OUTPUT" | grep -i "STOPPING\|diverged\|NaN" | head -3 || echo "OK - no divergence"

# Print QoI summary if it exists
[ -f "$DIR_RUN/qoi/qoi_summary.dat" ] && { echo ""; echo "=== QoI Summary ==="; cat $DIR_RUN/qoi/qoi_summary.dat; }
echo "$OUTPUT" | grep -E "Separation|Reattachment" | head -2

echo ""
echo "=== Done $(date) ==="
SBEOF
        # Fix the BINNAME variable inside the heredoc
        sed -i "s|\$BUILD/\${BINNAME}|\$BUILD/$BINNAME|g" $SCRIPT

        sleep 0.5
        JOBID=$(sbatch $SCRIPT 2>&1 | awk '{print $4}')
        NJOBS=$((NJOBS + 1))
        # Only print every 15th to avoid spam
        if [ $((NJOBS % 15)) -eq 0 ]; then
            echo "  Submitted $NJOBS jobs... (latest: $CASE/$LABEL = $JOBID)"
        fi
    done
done

echo ""
echo "Total: $NJOBS jobs submitted (10 min each)"
echo "Monitor: squeue -u \$USER | grep val-"
echo "Results: $OUTDIR/{case}/{model}/"
echo ""
echo "After jobs complete, check all results:"
echo "  for d in $OUTDIR/*/*; do echo \"\$(basename \$(dirname \$d))/\$(basename \$d): \$(ls \$d/qoi/ 2>/dev/null | wc -l) qoi files, res=\$(grep -oP 'res=\K\S+' \$d/output.log 2>/dev/null | tail -1)\"; done"
