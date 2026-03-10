#!/bin/bash
# Convenience wrapper: validate prerequisites and submit the RANS campaign SLURM array job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build_gpu_campaign"
CONFIG_DIR="${PROJECT_DIR}/examples/13_rans_campaign"
OUTPUT_DIR="${PROJECT_DIR}/output/rans_campaign"

# ---------- Pre-flight checks ----------

ERRORS=0

# Check build directory and executables
EXES=(cylinder airfoil step hills)
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: build directory not found: $BUILD_DIR"
    echo "       Build with: mkdir -p build_gpu_campaign && cd build_gpu_campaign && cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
    ERRORS=$((ERRORS + 1))
else
    for exe in "${EXES[@]}"; do
        if [ ! -x "${BUILD_DIR}/${exe}" ]; then
            echo "ERROR: executable not found: ${BUILD_DIR}/${exe}"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

# Check config files
CFG_COUNT=$(find "$CONFIG_DIR" -name '*.cfg' 2>/dev/null | wc -l)
if [ "$CFG_COUNT" -ne 38 ]; then
    echo "ERROR: expected 38 config files in $CONFIG_DIR, found $CFG_COUNT"
    echo "       Run: python3 scripts/rans_campaign/generate_configs.py"
    ERRORS=$((ERRORS + 1))
fi

# Check job list
if [ ! -f "${SCRIPT_DIR}/job_list.txt" ]; then
    echo "ERROR: job_list.txt not found in $SCRIPT_DIR"
    echo "       Run: python3 scripts/rans_campaign/generate_configs.py"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "Found $ERRORS error(s). Fix them before submitting."
    exit 1
fi

# ---------- Submit ----------

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"
export RANS_PROJECT_DIR="$PROJECT_DIR"
JOB_ID=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/submit_campaign.sbatch")

echo "Submitted RANS campaign as SLURM array job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  sacct -j $JOB_ID --format=JobID,State,Elapsed,ExitCode"
echo ""
echo "Logs in: $OUTPUT_DIR/slurm-${JOB_ID}_*.out"
echo ""
echo "After completion, analyze results with:"
echo "  python3 scripts/rans_campaign/analyze_campaign.py"
