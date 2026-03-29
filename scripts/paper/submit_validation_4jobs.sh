#!/bin/bash
# Submit 4 validation jobs (one per case), each runs all 15 models sequentially
# 10 min wall time, short T_final — just verify everything works
set -euo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
OUTDIR=results/paper/validation_grid
rm -rf $OUTDIR
mkdir -p $OUTDIR

ALL_MODELS="none||baseline baseline||mixing_length komega||komega sst||sst earsm_wj||earsm_wj earsm_gs||earsm_gs earsm_pope||earsm_pope gep||gep nn_mlp|mlp_paper|mlp nn_mlp|mlp_med_paper|mlp_med nn_tbnn|tbnn_small_paper|tbnn_small nn_tbnn|tbnn_paper|tbnn nn_tbnn|pi_tbnn_small_paper|pi_tbnn_small nn_tbnn|pi_tbnn_paper|pi_tbnn nn_tbrf|tbrf_1t_paper|tbrf_1t"

# case:binary:cfg:warmup:tfinal
CASES=(
    "hills_re10595:hills:examples/paper_experiments/hills_re10595.cfg:2.0:3.0"
    "cylinder_re100:cylinder:examples/paper_experiments/cylinder_re100.cfg:2.0:3.0"
    "duct_reb3500:duct:examples/paper_experiments/duct_reb3500.cfg:1.0:2.0"
    "sphere_re200:cylinder:examples/paper_experiments/sphere_re200.cfg:2.0:3.0"
)

for CASE_SPEC in "${CASES[@]}"; do
    IFS=':' read -r CASE BINNAME BASE_CFG WARMUP TFINAL <<< "$CASE_SPEC"

    SCRIPT=$OUTDIR/${CASE}_job.sbatch
    cat > $SCRIPT << SBEOF
#!/bin/bash
#SBATCH -J val-${CASE}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 00:20:00
#SBATCH -o ${OUTDIR}/${CASE}_%j.out
#SBATCH -e ${OUTDIR}/${CASE}_%j.err

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

echo "=== Validation: ${CASE} (15 models, T_final=${TFINAL}) ==="
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: \$(date)"
echo ""

printf "%-16s %6s %6s %7s %6s %-12s %-6s %s\n" "MODEL" "wall_s" "steps" "ms/step" "turb%" "residual" "qoi#" "status"

for MODEL_SPEC in ${ALL_MODELS}; do
    IFS='|' read -r TURB PRESET LABEL <<< "\$MODEL_SPEC"

    DIR=${OUTDIR}/${CASE}/\$LABEL
    mkdir -p \$DIR
    CFG=\$DIR/run.cfg
    sed "s/T_final = .*/T_final = ${TFINAL}/;s/max_steps = .*/max_steps = 20000/" ${BASE_CFG} > \$CFG
    echo "turb_model = \$TURB" >> \$CFG
    echo "warmup_model = sst" >> \$CFG
    echo "warmup_time = ${WARMUP}" >> \$CFG
    echo "warmup_steps = 50" >> \$CFG
    echo "qoi_output_dir = \$DIR/qoi" >> \$CFG
    echo "output_dir = \$DIR/" >> \$CFG
    [ -n "\$PRESET" ] && echo "nn_preset = \$PRESET" >> \$CFG

    OUTPUT=\$(\$BUILD/$BINNAME --config \$CFG 2>&1) || true
    echo "\$OUTPUT" > \$DIR/output.log

    WALL=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$2}')
    STEPS=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$3}')
    MSTP=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$4}')
    TURBT=\$(echo "\$OUTPUT" | grep "^turbulence_update" | awk '{print \$2}')
    LAST_RES=\$(echo "\$OUTPUT" | grep -oP 'res=\K\S+' | tail -1)
    TPCT="0"
    [ -n "\$TURBT" ] && [ -n "\$WALL" ] && TPCT=\$(python3 -c "print(f'{100*float(\"\$TURBT\")/max(0.001,float(\"\$WALL\")):.0f}')" 2>/dev/null)

    # Count QoI files
    NQOI=\$(ls \$DIR/qoi/ 2>/dev/null | wc -l)
    NFORCE=0
    [ -f "\$DIR/forces.dat" ] && NFORCE=\$(wc -l < \$DIR/forces.dat)

    # Status
    STATUS="OK"
    echo "\$OUTPUT" | grep -q "STOPPING\|diverged" && STATUS="DIV"
    [ -z "\$WALL" ] && STATUS="FAIL"

    printf "%-16s %6s %6s %7s %5s%% %-12s %s+%sF %s\n" \
        "\$LABEL" "\${WALL:-?}" "\${STEPS:-?}" "\${MSTP:-?}" "\$TPCT" "\${LAST_RES:-?}" "\$NQOI" "\$NFORCE" "\$STATUS"
done

echo ""
echo "=== Done \$(date) ==="
SBEOF

    JOBID=$(sbatch $SCRIPT 2>&1 | awk '{print $4}')
    echo "Submitted $CASE: job $JOBID (20 min)"
done

echo ""
echo "4 jobs submitted. Monitor: squeue -u \$USER | grep val-"
