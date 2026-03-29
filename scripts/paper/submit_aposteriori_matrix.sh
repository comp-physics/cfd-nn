#!/bin/bash
# Submit a posteriori matrix: 5 cases × 6 models = 5 batched jobs
# Each job runs all 6 models sequentially for one case
set -euo pipefail

cd /storage/scratch1/6/sbryngelson3/cfd-nn
mkdir -p results/paper/aposteriori

MODELS="none||baseline sst||sst earsm_wj||earsm_wj nn_mlp|mlp_paper|mlp nn_tbnn|tbnn_small_paper|tbnn_small nn_tbrf|tbrf_1t_paper|tbrf_1t"

submit_case() {
    local NAME=$1 CASE=$2 BIN=$3 NX=$4 NY=$5 NZ=$6 \
          XMIN=$7 XMAX=$8 YMIN=$9 YMAX=${10} ZMIN=${11} ZMAX=${12} \
          NU=${13} DPDX=${14} BVT=${15} BODY=${16} \
          WARMUP=${17} TFINAL=${18} MAXSTEPS=${19} TIME=${20} \
          EXTRA_CFG="${21:-}"

    local SCRIPT=results/paper/aposteriori/${CASE}_job.sbatch
    cat > $SCRIPT << SBEOF
#!/bin/bash
#SBATCH -J ap-${NAME}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t ${TIME}
#SBATCH -o results/paper/aposteriori/${CASE}_%j.out
#SBATCH -e results/paper/aposteriori/${CASE}_%j.err

set -uo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY
export XALT_EXECUTABLE_TRACKING=no

BUILD=build_h200
if [ ! -f \$BUILD/$BIN ]; then
    rm -rf \$BUILD && mkdir -p \$BUILD && cd \$BUILD
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF 2>&1 | tail -2
    make -j\$(nproc) hills duct cylinder 2>&1 | tail -3
    cd ..
fi

echo "=== ${CASE} on \$(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "Date: \$(date)"
echo ""
printf "%-16s %9s %6s %7s %6s %-12s %s\n" "MODEL" "eval_wall" "steps" "ms/step" "turb%" "residual" "QoI"

for MODEL_SPEC in ${MODELS}; do
    IFS='|' read -r TURB PRESET LABEL <<< "\$MODEL_SPEC"
    DIR=results/paper/aposteriori/${CASE}/\$LABEL
    mkdir -p \$DIR
    CFG=\$DIR/run.cfg

    cat > \$CFG << CFGEOF
Nx = $NX
Ny = $NY
Nz = $NZ
x_min = $XMIN
x_max = $XMAX
y_min = $YMIN
y_max = $YMAX
z_min = $ZMIN
z_max = $ZMAX
nu = $NU
dp_dx = $DPDX
bulk_velocity_target = $BVT
max_steps = $MAXSTEPS
T_final = $TFINAL
CFL_max = 0.3
dt_min = 1e-4
dt_safety = 0.85
adaptive_dt = true
simulation_mode = unsteady
convective_scheme = skew
time_integrator = rk3
poisson_tol = 1e-6
output_freq = 50
gpu_only_mode = true
ibm_eta = 0
perturbation_amplitude = 0.01
turb_model = \$TURB
warmup_model = sst
warmup_time = $WARMUP
warmup_steps = 200
qoi_freq = 1
qoi_output_dir = \$DIR/qoi
output_dir = \$DIR/
CFGEOF

    [ "$BODY" != "none" ] && echo "ibm_body = $BODY" >> \$CFG
    [ -n "\$PRESET" ] && echo "nn_preset = \$PRESET" >> \$CFG
    $EXTRA_CFG

    OUTPUT=\$(\$BUILD/$BIN --config \$CFG 2>&1) || true
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

    printf "%-16s %9s %6s %7s %5s%% %-12s %s\n" \
        "\$LABEL" "\${WALL:-?}" "\${STEPS:-?}" "\${MSTP:-?}" "\$TPCT" "\${LAST_RES:-?}" "\$QOI"
done

echo ""
echo "=== Done \$(date) ==="
SBEOF

    JOBID=$(sbatch $SCRIPT 2>&1 | awk '{print $4}')
    echo "  $NAME: job $JOBID ($TIME)"
}

echo "=== Submitting a posteriori matrix (5 batched jobs) ==="
echo ""

# Cylinder Re=100 (384×288, proven Cd=1.47 at 9% error)
submit_case "cyl100" "cylinder_re100" "cylinder" \
    384 288 1 -3.0 13.0 -6.0 6.0 0.0 1.0 \
    0.01 0.0 1.0 "none" \
    5.0 100.0 100000 "02:00:00"

# Cylinder Re=3900 (384×288, turbulent)
submit_case "cyl3900" "cylinder_re3900" "cylinder" \
    384 288 1 -3.0 13.0 -6.0 6.0 0.0 1.0 \
    0.000256 0.0 1.0 "none" \
    5.0 50.0 200000 "04:00:00"

# Sphere Re=200 (512×384×384, 34 cells/D — higher res)
submit_case "sph200" "sphere_re200" "cylinder" \
    512 384 384 -4.0 11.0 -4.0 4.0 -4.0 4.0 \
    0.005 0.0 1.0 "sphere" \
    5.0 30.0 30000 "06:00:00"

# Hills Re=10595 (384×192, warm-up→ghost-cell)
# Extra: steady mode, stretch, penalization for warm-up, lower CFL
submit_case "hills" "hills" "hills" \
    384 192 1 0.0 9.0 0.0 3.035 0.0 1.0 \
    0.0000944 -1.0 0.0 "periodic_hills" \
    3.0 30.0 200000 "04:00:00" \
    "sed -i 's/simulation_mode = unsteady/simulation_mode = steady/' \$CFG; echo 'tol = 1e-8' >> \$CFG; sed -i 's/ibm_eta = 0/ibm_eta = 0.1/' \$CFG; sed -i 's/CFL_max = 0.3/CFL_max = 0.15/' \$CFG; echo 'stretch_y = true' >> \$CFG; echo 'stretch_beta = 2.0' >> \$CFG"

# Duct Re_b=3500 (96³, no IBM)
submit_case "duct" "duct" "duct" \
    96 96 96 0.0 6.283185 -1.0 1.0 -1.0 1.0 \
    0.006 -1.0 0.0 "none" \
    3.0 30.0 100000 "02:00:00"

echo ""
echo "5 jobs submitted. Monitor: squeue -u \$USER | grep ap-"
