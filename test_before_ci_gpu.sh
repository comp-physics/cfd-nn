#!/bin/bash
# Pre-CI GPU validation script
# Run this before pushing to ensure GPU CI will pass
# This script submits a job to a GPU node and runs the full GPU CI test suite

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "==================================================================="
echo -e "  ${BLUE}Pre-CI GPU Testing - Full GPU CI Validation${NC}"
echo "==================================================================="
echo ""
echo "This script runs the COMPLETE GPU CI test suite locally."
echo "It submits a job to a GPU node and waits for completion."
echo ""
echo "Test suite includes:"
echo "  1. All unit tests (ctest)"
echo "  2. Algebraic models (Baseline, GEP) - large grids"
echo "  3. Transport models (SST, k-omega)"
echo "  4. EARSM models (3 variants)"
echo "  5. Periodic Hills test cases"
echo ""
echo -e "${YELLOW}[WARNING] This will take approximately 10-15 minutes to complete.${NC}"
echo ""

# Ask for confirmation
read -p "Continue with full GPU CI test? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="pre_ci_gpu_full_${TIMESTAMP}"
LOG_FILE="${JOB_NAME}.log"
ERR_FILE="${JOB_NAME}.err"
SBATCH_SCRIPT="/tmp/${JOB_NAME}.sbatch"

echo ""
echo "==================================================================="
echo "  Step 1: Building GPU version"
echo "==================================================================="
echo ""

# Clean and build GPU version
rm -rf build_gpu_ci
mkdir -p build_gpu_ci
cd build_gpu_ci

echo "--- Configuring CMake (GPU offload enabled) ---"
module load nvhpc/25.5
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON

echo ""
echo "--- Building with GPU offload ---"
make -j8

if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL] Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Build successful${NC}"
echo ""

cd "$PROJECT_ROOT"

# Create SLURM submission script (matches GPU CI configuration)
echo "==================================================================="
echo "  Step 2: Submitting GPU test job to Slurm"
echo "==================================================================="
echo ""

cat > "$SBATCH_SCRIPT" <<'EOFSBATCH'
#!/bin/bash
#SBATCH -J pre_ci_gpu_full
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-l40s,gpu-h200
#SBATCH -G1
#SBATCH --qos=embers
#SBATCH -t 00:20:00
#SBATCH -o PRE_CI_GPU_LOG_FILE
#SBATCH -e PRE_CI_GPU_ERR_FILE

set -e

module reset
module load nvhpc/25.5

echo "==================================================================="
echo "  GPU Test Environment"
echo "==================================================================="
echo ""
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

cd PRE_CI_WORK_DIR/build_gpu_ci

echo "==================================================================="
echo "  1. Unit Tests"
echo "==================================================================="
echo ""
ctest --output-on-failure

echo ""
echo "==================================================================="
echo "  2. Algebraic Models (Large Grid Validation)"
echo "==================================================================="
echo ""

mkdir -p output/gpu_validation

echo "--- Testing Baseline Model ---"
../.github/scripts/test_turbulence_model_gpu.sh baseline "Baseline" 64 128 5000 output/gpu_validation/baseline

echo ""
echo "--- Testing GEP Model ---"
../.github/scripts/test_turbulence_model_gpu.sh gep "GEP" 64 128 5000 output/gpu_validation/gep

echo ""
echo "==================================================================="
echo "  3. Transport Equation Models"
echo "==================================================================="
echo ""

echo "--- Testing SST k-omega ---"
../.github/scripts/test_turbulence_model_gpu.sh sst "SST k-omega" 64 128 500 output/gpu_validation/sst

echo ""
echo "--- Testing k-omega (Wilcox) ---"
../.github/scripts/test_turbulence_model_gpu.sh komega "k-omega (Wilcox)" 64 128 500 output/gpu_validation/komega

echo ""
echo "==================================================================="
echo "  4. EARSM Models"
echo "==================================================================="
echo ""

echo "--- Testing Wallin-Johansson EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_wj "Wallin-Johansson EARSM" 256 512 1000 output/gpu_validation/earsm_wj

echo ""
echo "--- Testing Gatski-Speziale EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_gs "Gatski-Speziale EARSM" 256 512 1000 output/gpu_validation/earsm_gs

echo ""
echo "--- Testing Pope Quadratic EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_pope "Pope Quadratic EARSM" 256 512 1000 output/gpu_validation/earsm_pope

echo ""
echo "==================================================================="
echo "  5. Periodic Hills - Complex Geometry"
echo "==================================================================="
echo ""

echo "--- Testing Periodic Hills with Baseline ---"
./periodic_hills --Nx 64 --Ny 48 --nu 0.001 --max_iter 200 --model baseline --num_snapshots 0

echo ""
echo "==================================================================="
echo "  6. CPU/GPU Consistency Validation"
echo "==================================================================="
echo ""
echo "This test validates that CPU and GPU produce IDENTICAL results."
echo ""

./test_cpu_gpu_consistency

echo ""
echo "==================================================================="
echo "  [PASS] All GPU CI Tests PASSED!"
echo "==================================================================="
echo ""
EOFSBATCH

# Replace placeholders
sed -i "s|PRE_CI_GPU_LOG_FILE|${LOG_FILE}|g" "$SBATCH_SCRIPT"
sed -i "s|PRE_CI_GPU_ERR_FILE|${ERR_FILE}|g" "$SBATCH_SCRIPT"
sed -i "s|PRE_CI_WORK_DIR|${PROJECT_ROOT}|g" "$SBATCH_SCRIPT"

# Submit job and wait
echo "Submitting job to GPU queue..."
echo "Job name: ${JOB_NAME}"
echo ""
echo -e "${YELLOW}⏳ Waiting for job to complete (this may take 45-60 minutes)...${NC}"
echo ""

sbatch -W "$SBATCH_SCRIPT"
EXIT_CODE=$?

# Wait for files to be written
sleep 2

echo ""
echo "==================================================================="
echo "  Job Output"
echo "==================================================================="
echo ""

if [ -f "${LOG_FILE}" ]; then
    cat "${LOG_FILE}"
else
    echo -e "${RED}ERROR: Log file not found: ${LOG_FILE}${NC}"
fi

echo ""

if [ -f "${ERR_FILE}" ] && [ -s "${ERR_FILE}" ]; then
    echo "==================================================================="
    echo "  Job Errors/Warnings"
    echo "==================================================================="
    echo ""
    cat "${ERR_FILE}"
    echo ""
fi

# Cleanup
rm -f "$SBATCH_SCRIPT"

echo ""
echo "==================================================================="
echo "  Results"
echo "==================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}[PASS] ALL GPU CI TESTS PASSED!${NC}"
    echo ""
    echo "Your code has been validated with the complete GPU CI test suite."
    echo "You are safe to push to the repository!"
    echo ""
    
    # Cleanup
    read -p "Clean up build directory and logs? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf build_gpu_ci "${LOG_FILE}" "${ERR_FILE}"
        echo "Cleaned up."
    fi
    
    exit 0
else
    echo -e "${RED}[FAIL] GPU CI TESTS FAILED!${NC}"
    echo ""
    echo "Fix the issues above before pushing to avoid CI failures."
    echo "Log files retained for debugging:"
    echo "  - ${LOG_FILE}"
    echo "  - ${ERR_FILE}"
    echo ""
    exit 1
fi

