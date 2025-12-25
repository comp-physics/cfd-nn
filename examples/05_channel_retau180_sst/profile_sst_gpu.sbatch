#!/bin/bash
#SBATCH --job-name=profile_sst
#SBATCH --account=gts-sbryngelson3
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-profile-%j.out
#SBATCH --error=slurm-profile-%j.err

set -euo pipefail

echo "=========================================="
echo "GPU Profiling: SST Channel (200 steps)"
echo "=========================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo ""

# Load modules
module purge
module load nvhpc/24.5

# Set OpenMP environment
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_TARGET_OFFLOAD=MANDATORY
export NVCOMPILER_ACC_NOTIFY=0

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Set paths
ROOT="/storage/home/hcoda1/6/sbryngelson3/cfd-nn"
CASE_DIR="${ROOT}/examples/05_channel_retau180_sst"
BUILD_DIR="${ROOT}/build_gpu_examples"
OUT_DIR="${CASE_DIR}/output_profile"

mkdir -p "${OUT_DIR}"
cd "${BUILD_DIR}"

echo "Running nsys profile on SST channel (200 iterations)..."
echo "Config: ${CASE_DIR}/channel.cfg"
echo "Output: ${OUT_DIR}/"
echo ""

# Run with nsys profiler
nsys profile \
  -o "${CASE_DIR}/sst_gpu_profile_200" \
  --stats=true \
  --force-overwrite=true \
  ./channel \
    --config "${CASE_DIR}/channel.cfg" \
    --output "${OUT_DIR}/" \
    --max_iter 200 \
    --tol 1e-4 \
    --poisson_max_iter 10 \
    --no_postprocess \
    --no_write_fields \
    --verbose

echo ""
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Profile files:"
echo "  ${CASE_DIR}/sst_gpu_profile_200.nsys-rep"
echo "  ${CASE_DIR}/sst_gpu_profile_200.sqlite"
echo ""
echo "To view the profile:"
echo "  nsys-ui ${CASE_DIR}/sst_gpu_profile_200.nsys-rep"
echo ""


