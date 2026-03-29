#!/bin/bash
# Submit production a posteriori sweep as 10 SLURM jobs
# Split by case (4) × model group (fast vs slow NN) + individual expensive runs
set -euo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
mkdir -p results/paper/production_sweep

FAST_MODELS="none||baseline baseline||mixing_length komega||komega sst||sst earsm_wj||earsm_wj earsm_gs||earsm_gs earsm_pope||earsm_pope gep||gep nn_tbrf|tbrf_1t_paper|tbrf_1t nn_mlp|mlp_paper|mlp"
SLOW_MODELS="nn_mlp|mlp_med_paper|mlp_med nn_tbnn|tbnn_small_paper|tbnn_small nn_tbnn|tbnn_paper|tbnn nn_tbnn|pi_tbnn_small_paper|pi_tbnn_small nn_tbnn|pi_tbnn_paper|pi_tbnn"

submit_job() {
    local NAME=$1 CASE=$2 BIN=$3 BASE_CFG=$4 WARMUP=$5 TFINAL=$6 MAXSTEPS=$7 TIME=$8 MODELS=$9

    SCRIPT=$(mktemp -p . job_XXXXXX.sbatch)
    cat > $SCRIPT << SBEOF
#!/bin/bash
#SBATCH -J prod-${NAME}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t ${TIME}
#SBATCH -o results/paper/production_sweep/${NAME}_%j.out
#SBATCH -e results/paper/production_sweep/${NAME}_%j.err

set -uo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=build_h200
if [ ! -f \$BUILD/${BIN##*/} ]; then
    mkdir -p \$BUILD && cd \$BUILD
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF 2>&1 | tail -2
    make -j\$(nproc) hills duct cylinder 2>&1 | tail -3
    cd ..
fi

echo "=== ${NAME} ==="
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: \$(date)"

OUTDIR=results/paper/production_sweep/${CASE}
printf "%-16s %9s %6s %9s %7s %-12s %s\n" "MODEL" "eval_wall" "steps" "ms/step" "turb%" "residual" "QoI"

for MODEL_SPEC in ${MODELS}; do
    IFS='|' read -r TURB PRESET LABEL <<< "\$MODEL_SPEC"
    DIR=\$OUTDIR/\$LABEL
    mkdir -p \$DIR
    CFG=\$DIR/run.cfg
    sed "s/T_final = .*/T_final = ${TFINAL}/;s/max_steps = .*/max_steps = ${MAXSTEPS}/" ${BASE_CFG} > \$CFG
    echo "turb_model = \$TURB" >> \$CFG
    echo "warmup_model = sst" >> \$CFG
    echo "warmup_time = ${WARMUP}" >> \$CFG
    echo "warmup_steps = 200" >> \$CFG
    echo "qoi_output_dir = \$DIR/qoi" >> \$CFG
    echo "output_dir = \$DIR/" >> \$CFG
    [ -n "\$PRESET" ] && echo "nn_preset = \$PRESET" >> \$CFG

    OUTPUT=\$(\$BUILD/${BIN##*/} --config \$CFG 2>&1) || true
    echo "\$OUTPUT" > \$DIR/output.log

    WALL=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$2}')
    STEPS=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$3}')
    MSTP=\$(echo "\$OUTPUT" | grep "^solver_step" | awk '{print \$4}')
    TURBT=\$(echo "\$OUTPUT" | grep "^turbulence_update" | awk '{print \$2}')
    LAST_RES=\$(echo "\$OUTPUT" | grep -oP 'res=\K\S+' | tail -1)
    TPCT="0"
    [ -n "\$TURBT" ] && [ -n "\$WALL" ] && TPCT=\$(python3 -c "print(f'{100*float(\"\$TURBT\")/max(0.001,float(\"\$WALL\")):.0f}')" 2>/dev/null)

    QOI=""
    [ -f "\$DIR/qoi/qoi_summary.dat" ] && {
        CD=\$(grep "Cd_mean" \$DIR/qoi/qoi_summary.dat 2>/dev/null | awk '{print \$2}')
        ST=\$(grep "^St " \$DIR/qoi/qoi_summary.dat 2>/dev/null | awk '{print \$2}')
        [ -n "\$CD" ] && QOI="Cd=\$CD"
        [ -n "\$ST" ] && [ "\$ST" != "-1" ] && QOI="\$QOI St=\$ST"
    }
    XSEP=\$(echo "\$OUTPUT" | grep "Separation:" | sed 's/.*x\/H = //' | head -1)
    [ -n "\$XSEP" ] && QOI="sep=\$XSEP"

    printf "%-16s %9s %6s %9s %6s%% %-12s %s\n" "\$LABEL" "\${WALL:-?}" "\${STEPS:-?}" "\${MSTP:-?}" "\$TPCT" "\${LAST_RES:-?}" "\$QOI"
done

echo "=== Done \$(date) ==="
SBEOF

    JOBID=$(sbatch $SCRIPT 2>&1 | awk '{print $4}')
    echo "  $NAME: job $JOBID ($TIME)"
    rm -f $SCRIPT
}

echo "Submitting 10 production jobs..."

# Fast models (10 models each, ~5-40 min per case)
submit_job "hills-fast" "hills_re10595" "build_h200/hills" \
    "examples/paper_experiments/hills_re10595.cfg" "5.0" "30.0" "200000" "01:00:00" "$FAST_MODELS"

submit_job "cyl100-fast" "cylinder_re100" "build_h200/cylinder" \
    "examples/paper_experiments/cylinder_re100.cfg" "5.0" "50.0" "200000" "01:00:00" "$FAST_MODELS"

submit_job "duct-fast" "duct_reb3500" "build_h200/duct" \
    "examples/paper_experiments/duct_reb3500.cfg" "3.0" "20.0" "100000" "01:00:00" "$FAST_MODELS"

submit_job "sph200-fast" "sphere_re200" "build_h200/cylinder" \
    "examples/paper_experiments/sphere_re200.cfg" "5.0" "20.0" "50000" "01:00:00" "$FAST_MODELS"

# Slow NN models (5 models each, up to 4h for sphere)
submit_job "hills-nn" "hills_re10595" "build_h200/hills" \
    "examples/paper_experiments/hills_re10595.cfg" "5.0" "30.0" "200000" "02:00:00" "$SLOW_MODELS"

submit_job "cyl100-nn" "cylinder_re100" "build_h200/cylinder" \
    "examples/paper_experiments/cylinder_re100.cfg" "5.0" "50.0" "200000" "02:00:00" "$SLOW_MODELS"

submit_job "duct-nn" "duct_reb3500" "build_h200/duct" \
    "examples/paper_experiments/duct_reb3500.cfg" "3.0" "20.0" "100000" "04:00:00" "$SLOW_MODELS"

# Sphere NN: split TBNN/PI-TBNN (very slow) from smaller models
submit_job "sph200-nn-small" "sphere_re200" "build_h200/cylinder" \
    "examples/paper_experiments/sphere_re200.cfg" "5.0" "10.0" "50000" "04:00:00" \
    "nn_mlp|mlp_med_paper|mlp_med nn_tbnn|tbnn_small_paper|tbnn_small nn_tbnn|pi_tbnn_small_paper|pi_tbnn_small"

submit_job "sph200-nn-large" "sphere_re200" "build_h200/cylinder" \
    "examples/paper_experiments/sphere_re200.cfg" "5.0" "5.0" "30000" "06:00:00" \
    "nn_tbnn|tbnn_paper|tbnn nn_tbnn|pi_tbnn_paper|pi_tbnn"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
