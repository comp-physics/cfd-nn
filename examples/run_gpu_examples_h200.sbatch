#!/bin/bash
#SBATCH --job-name=cfdnn_examples_h200
#SBATCH --account=gts-sbryngelson3
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-examples-%j.out
#SBATCH --error=slurm-examples-%j.err

set -euo pipefail

echo "=========================================="
echo "CFD-NN GPU examples on H200"
echo "=========================================="
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo ""

# Load compiler / offload toolchain (match existing H200 script)
module purge
module load nvhpc/24.5

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_TARGET_OFFLOAD=MANDATORY
export NVCOMPILER_ACC_NOTIFY=0

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo ""

ROOT="/storage/home/hcoda1/6/sbryngelson3/cfd-nn"
BUILD_GPU="${ROOT}/build_gpu_examples"

mkdir -p "${BUILD_GPU}"
cd "${BUILD_GPU}"

echo "Configuring GPU build in ${BUILD_GPU} ..."
cmake .. -DUSE_GPU_OFFLOAD=ON

echo "Building channel and periodic_hills ..."
make -j4 channel periodic_hills

echo ""
echo "=========================================="
echo "Running channel / Re_tau ~ 180 (SST, steady) [GPU PHYSICS TEST]"
echo "=========================================="

SST_CASE_DIR="${ROOT}/examples/05_channel_retau180_sst"
SST_OUT_DIR="${SST_CASE_DIR}/output_gpu"
mkdir -p "${SST_OUT_DIR}"

time stdbuf -oL -eL ./channel \
  --config "${SST_CASE_DIR}/channel.cfg" \
  --output "${SST_OUT_DIR}/" \
  --max_iter 20000 \
  --tol 1e-6 \
  --poisson_max_iter 10 \
  --verbose

echo ""
echo "DNS comparison for GPU SST channel..."
python3 "${SST_CASE_DIR}/compare_dns.py" \
  --output_dir "${SST_OUT_DIR}" \
  --max_rel_error 25.0 \
  --min_retau 150.0 \
  --max_retau 210.0

echo ""
echo "=========================================="
echo "Running channel / Re_tau ~ 180 (unsteady laminar) [SHORT RUN]"
echo "=========================================="

CHAN_EX="${ROOT}/examples/channel"
mkdir -p "${CHAN_EX}/output_gpu/retau180_unsteady_laminar"

time stdbuf -oL -eL ./channel \
  --config "${CHAN_EX}/cases/retau180_unsteady_laminar.cfg" \
  --output "${CHAN_EX}/output_gpu/retau180_unsteady_laminar/" \
  --max_iter 2000 \
  --poisson_max_iter 10 \
  --no_postprocess

echo ""
echo "=========================================="
echo "Running periodic hills / SST [QUICK GPU CHECK]"
echo "=========================================="

HILLS_EX="${ROOT}/examples/periodic_hills"
mkdir -p "${HILLS_EX}/output_gpu/hills_sst"

time stdbuf -oL -eL ./periodic_hills \
  --config "${HILLS_EX}/cases/hills_sst.cfg" \
  --output "${HILLS_EX}/output_gpu/hills_sst/" \
  --max_iter 20000 \
  --tol 1e-5 \
  --poisson_max_iter 10 \
  --verbose

echo ""
echo "=========================================="
echo "GPU examples complete"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Outputs:"
echo "  Channel SST (Re_tau~180):       ${CHAN_EX}/output_gpu/retau180_sst/"
echo "  Channel unsteady laminar:       ${CHAN_EX}/output_gpu/retau180_unsteady_laminar/"
echo "  Periodic hills SST:             ${HILLS_EX}/output_gpu/hills_sst/"
echo ""


