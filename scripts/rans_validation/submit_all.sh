#!/bin/bash
# Submit all RANS validation models as parallel SLURM jobs
set -euo pipefail

PROJ_DIR=/storage/scratch1/6/sbryngelson3/cfd-nn
SCRIPT_DIR=${PROJ_DIR}/scripts/rans_validation
CFG_DIR=${SCRIPT_DIR}/configs
TEMPLATE=${SCRIPT_DIR}/run_single_model.sbatch
OUT_DIR=${PROJ_DIR}/output/rans_validation

# Clean old output
rm -rf ${OUT_DIR}/data/* ${OUT_DIR}/logs/*
mkdir -p ${OUT_DIR}/logs ${OUT_DIR}/data

echo "Submitting RANS validation jobs..."
echo ""

submit_model() {
    local label="$1"
    local model="$2"
    local config="$3"
    local steps="$4"

    JOB_ID=$(sbatch \
        --export=MODEL=${model},LABEL=${label},MAX_STEPS=${steps},CONFIG=${config} \
        -J "rans_${label}" \
        -o "${OUT_DIR}/slurm-${label}-%j.out" \
        -e "${OUT_DIR}/slurm-${label}-%j.err" \
        "${TEMPLATE}" | awk '{print $NF}')
    echo "  ${label}: job ${JOB_ID}"
}

# Phase 1: All models on base grid (64x128), 50000 steps
BASE_CFG="${CFG_DIR}/rans_retau180_base.cfg"
for model in none baseline gep earsm_wj sst komega nn_mlp nn_tbnn; do
    submit_model "${model}" "${model}" "${BASE_CFG}" 50000
done

# Phase 2: SST grid convergence
submit_model "sst_grid_32x64"   "sst" "${CFG_DIR}/grid_32x64.cfg"   50000
submit_model "sst_grid_128x256" "sst" "${CFG_DIR}/grid_128x256.cfg" 50000

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$(whoami) | grep rans"
echo "Check output: ls ${OUT_DIR}/data/*/velocity_profile.dat"
