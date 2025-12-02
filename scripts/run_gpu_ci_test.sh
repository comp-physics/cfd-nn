#!/bin/bash
# GPU CI Test Runner for SLURM
# Uses sbatch -W to wait for completion

set -e

TEST_NAME="${1:-gpu_test}"
TEST_COMMAND="${2:-./test_gpu_execution}"
TIMEOUT_MINUTES="${3:-60}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="ci_${TEST_NAME}_${TIMESTAMP}"
LOG_FILE="${JOB_NAME}.log"
ERR_FILE="${JOB_NAME}.err"

echo "========================================"
echo "GPU CI Test: ${TEST_NAME}"
echo "Command: ${TEST_COMMAND}"
echo "========================================"
echo ""

# Create SLURM submission script
cat > "/tmp/${JOB_NAME}.sbatch" <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-l40s,gpu-h200
#SBATCH -G1
#SBATCH --qos=embers
#SBATCH -t 01:00:00
#SBATCH -o ${LOG_FILE}
#SBATCH -e ${ERR_FILE}

module reset
module load nvhpc/25.5

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

echo "=== Running: ${TEST_NAME} ==="
${TEST_COMMAND}
EXIT_CODE=\$?

echo ""
echo "=== Completed with exit code: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF

# Submit and wait for completion
echo "Submitting job (will wait for completion)..."
set +e  # Don't exit on error - we need to print logs first
sbatch -W "/tmp/${JOB_NAME}.sbatch"
SBATCH_EXIT=$?
set -e

echo ""
echo "Job finished with exit code: ${SBATCH_EXIT}"
echo ""

# Wait a moment for files to be written
sleep 2

echo "========================================"
echo "Job Output:"
echo "========================================"
if [ -f "${LOG_FILE}" ]; then
    cat "${LOG_FILE}"
else
    echo "ERROR: Log file not found: ${LOG_FILE}"
    echo "Checking current directory..."
    ls -lh *.log 2>/dev/null || echo "No log files found"
fi

echo ""
if [ -f "${ERR_FILE}" ]; then
    if [ -s "${ERR_FILE}" ]; then
        echo "========================================"
        echo "Job Errors:"
        echo "========================================"
        cat "${ERR_FILE}"
        echo ""
    fi
fi

# Cleanup
rm -f "/tmp/${JOB_NAME}.sbatch" "${LOG_FILE}" "${ERR_FILE}"

echo "========================================"
echo "Final exit code: ${SBATCH_EXIT}"
echo "========================================"
exit ${SBATCH_EXIT}

