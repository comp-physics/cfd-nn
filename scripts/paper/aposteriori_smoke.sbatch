#!/bin/bash
#SBATCH -J aposteriori-smoke
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 01:00:00
#SBATCH -o results/paper/aposteriori/smoke_%j.out
#SBATCH -e results/paper/aposteriori/smoke_%j.err

set -uo pipefail
cd /storage/scratch1/6/sbryngelson3/cfd-nn
module load nvhpc/24.5
export OMP_TARGET_OFFLOAD=MANDATORY

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"
echo "$(date)"
echo ""

BUILD=build_h200
if [ ! -f $BUILD/hills ] || [ ! -f $BUILD/duct ] || [ ! -f $BUILD/cylinder ]; then
    echo "Building..."
    mkdir -p $BUILD && cd $BUILD
    cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON \
             -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90 2>&1 | tail -2
    make -j$(nproc) hills duct cylinder 2>&1 | tail -3
    cd ..
fi

# Case configs
cat > /tmp/apost_hills.cfg << 'CFGEOF'
Nx = 196
Ny = 96
Nz = 1
x_min = 0.0
x_max = 9.0
y_min = 0.0
y_max = 3.035
nu = 0.0001786
dp_dx = -1.0
ibm_body = periodic_hills
convective_scheme = skew
time_integrator = rk3
adaptive_dt = true
CFL_max = 0.5
stretch_y = true
stretch_beta = 2.0
simulation_mode = steady
max_steps = 50
CFGEOF

cat > /tmp/apost_cyl.cfg << 'CFGEOF'
Nx = 256
Ny = 192
Nz = 1
x_min = -3.0
x_max = 13.0
y_min = -6.0
y_max = 6.0
nu = 0.01
dp_dx = 0.0
bulk_velocity_target = 1.0
ibm_body = cylinder
convective_scheme = upwind2
time_integrator = rk3
adaptive_dt = true
CFL_max = 0.3
simulation_mode = steady
max_steps = 50
CFGEOF

cat > /tmp/apost_duct.cfg << 'CFGEOF'
Nx = 48
Ny = 64
Nz = 64
x_min = 0.0
x_max = 6.283185
y_min = -1.0
y_max = 1.0
stretch_y = true
stretch_beta = 2.0
stretch_z = true
stretch_beta_z = 2.0
nu = 0.001
dp_dx = -1.0
convective_scheme = skew
time_integrator = rk3
adaptive_dt = true
CFL_max = 0.5
simulation_mode = steady
max_steps = 50
CFGEOF

cat > /tmp/apost_sphere.cfg << 'CFGEOF'
Nx = 128
Ny = 96
Nz = 96
x_min = -4.0
x_max = 11.0
y_min = -4.0
y_max = 4.0
nu = 0.01
dp_dx = 0.0
bulk_velocity_target = 1.0
ibm_body = sphere
convective_scheme = upwind2
time_integrator = rk3
adaptive_dt = true
CFL_max = 0.3
simulation_mode = steady
max_steps = 50
CFGEOF

run_case() {
    local CASE=$1 BINARY=$2 CFG=$3 TURB=$4 PRESET=$5 LABEL=$6
    MCFG=/tmp/${CASE}_${LABEL}.cfg
    cp $CFG $MCFG
    echo "turb_model = $TURB" >> $MCFG
    [ -n "$PRESET" ] && echo "nn_preset = $PRESET" >> $MCFG

    printf "  %-15s" "$LABEL"
    OUTPUT=$($BINARY --config $MCFG 2>&1) || true
    STEP=$(echo "$OUTPUT" | grep "solver_step" | head -1 | awk '{print $4}')
    ERR=$(echo "$OUTPUT" | grep -iE "terminate|FATAL|error while|Invalid config" | head -1)

    if [ -n "$ERR" ]; then
        echo "FAILED: $ERR"
    elif [ -z "$STEP" ]; then
        echo "NO TIMING"
    else
        printf "%8s ms/step\n" "$STEP"
    fi
}

# All 11 RANS models
ALL_MODELS=(
    "none||baseline"
    "baseline||mixing_length"
    "komega||k-omega"
    "sst||sst"
    "earsm_wj||earsm_wj"
    "earsm_gs||earsm_gs"
    "earsm_pope||earsm_pope"
    "gep||gep"
    "nn_mlp|mlp_paper|mlp"
    "nn_tbnn|tbnn_paper|tbnn"
    "nn_tbrf|tbrf_1t_paper|tbrf1t"
)

for CASE_SPEC in \
    "hills:$BUILD/hills:/tmp/apost_hills.cfg" \
    "cylinder:$BUILD/cylinder:/tmp/apost_cyl.cfg" \
    "duct:$BUILD/duct:/tmp/apost_duct.cfg" \
    "sphere:$BUILD/cylinder:/tmp/apost_sphere.cfg"; do

    IFS=':' read -r CASE BIN CFG <<< "$CASE_SPEC"
    echo ""
    echo "=== $CASE ==="
    for MODEL_SPEC in "${ALL_MODELS[@]}"; do
        IFS='|' read -r TURB PRESET LABEL <<< "$MODEL_SPEC"
        run_case $CASE $BIN $CFG "$TURB" "$PRESET" "$LABEL"
    done
done

echo ""
echo "=== DONE $(date) ==="
