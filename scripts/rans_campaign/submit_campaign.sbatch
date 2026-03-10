#!/bin/bash
#SBATCH -J rans_campaign
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -p gpu-h100,gpu-h200,gpu-a100,gpu-v100
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 02:00:00
#SBATCH -o output/rans_campaign/slurm-%A_%a.out
#SBATCH -e output/rans_campaign/slurm-%A_%a.err
#SBATCH --array=0-37

set -euo pipefail

# Resolve project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build_gpu_campaign"
JOB_LIST="${SCRIPT_DIR}/job_list.txt"

# Read job info for this array task
LINE=$(grep "^${SLURM_ARRAY_TASK_ID} " "$JOB_LIST") || {
    echo "ERROR: no entry for array index ${SLURM_ARRAY_TASK_ID} in $JOB_LIST"
    exit 1
}
EXE=$(echo "$LINE" | awk '{print $2}')
CONFIG=$(echo "$LINE" | awk '{print $3}')

# Environment for GPU offload
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_THREADS=1

# Print job info
echo "=== RANS Campaign Job ==="
echo "Array task ID : ${SLURM_ARRAY_TASK_ID}"
echo "SLURM job ID  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Executable    : ${EXE}"
echo "Config        : ${CONFIG}"
echo "Hostname      : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================="

# Create output directory for this config
CONFIG_BASENAME=$(basename "$CONFIG" .cfg)
OUTPUT_DIR="${PROJECT_DIR}/output/rans_campaign/${CONFIG_BASENAME}"
mkdir -p "$OUTPUT_DIR"

# Run the solver
cd "$PROJECT_DIR"
START_SEC=$SECONDS

set +e
"${BUILD_DIR}/${EXE}" --config "${CONFIG}"
EXIT_CODE=$?
set -e

ELAPSED=$((SECONDS - START_SEC))
WALL_MS=$((ELAPSED * 1000))

echo ""
echo "=== Job Complete ==="
echo "Exit code : ${EXIT_CODE}"
echo "Wall time : ${WALL_MS}ms (${ELAPSED}s)"
echo "End time  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================="

exit ${EXIT_CODE}
