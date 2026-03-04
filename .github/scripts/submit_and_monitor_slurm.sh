#!/bin/bash
# Submit a SLURM job and monitor its output periodically
# Usage: submit_and_monitor_slurm.sh <template> <workdir> <output_file> <error_file> <final_script>

set -euo pipefail

TEMPLATE="$1"
WORKDIR="$2"
SLURM_OUT="$3"
SLURM_ERR="$4"
SBATCH_SCRIPT="$5"

# Generate sbatch script from template
sed -e "s|__WORKDIR__|${WORKDIR}|g" \
    -e "s|__SLURM_OUT__|${SLURM_OUT}|g" \
    -e "s|__SLURM_ERR__|${SLURM_ERR}|g" \
    "${TEMPLATE}" > "${SBATCH_SCRIPT}"

chmod +x "${SBATCH_SCRIPT}"

echo "Submitting Slurm job from ${TEMPLATE}..."
JOB_ID=$(sbatch --parsable "${SBATCH_SCRIPT}")
echo "Submitted job ID: ${JOB_ID}"

# Cancel SLURM job if this script is killed (e.g., by CI timeout)
cleanup() {
  echo "Monitoring script interrupted — cancelling SLURM job ${JOB_ID}..."
  scancel "${JOB_ID}" 2>/dev/null || true
  exit 1
}
trap cleanup SIGTERM SIGINT SIGHUP

# Monitor job with periodic output
LAST_OUT_SIZE=0
while true; do
  JOB_STATE=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null || echo "COMPLETED")
  
  if [ "$JOB_STATE" = "COMPLETED" ] || [ "$JOB_STATE" = "" ]; then
    echo ""
    echo "Job ${JOB_ID} completed"
    break
  fi
  
  # Print new output if files exist and have grown
  if [ -f "${SLURM_OUT}" ]; then
    CURR_SIZE=$(stat -c%s "${SLURM_OUT}" 2>/dev/null || echo "0")
    if [ "$CURR_SIZE" -gt "$LAST_OUT_SIZE" ]; then
      tail -c +$((LAST_OUT_SIZE + 1)) "${SLURM_OUT}"
      LAST_OUT_SIZE=$CURR_SIZE
    fi
  fi
  
  sleep 30
done

# Wait for sacct to report a terminal state (COMPLETED, FAILED, TIMEOUT, etc.)
# sacct can lag behind squeue on some SLURM clusters, so poll until terminal
echo "Waiting for sacct to report terminal state..."
for i in $(seq 1 30); do
  JOB_STATE=$(sacct -j ${JOB_ID} --format=State --noheader | head -n1 | tr -d ' ')
  case "$JOB_STATE" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY)
      break
      ;;
    *)
      echo "  sacct state: ${JOB_STATE} (attempt $i/30, waiting 10s...)"
      sleep 10
      ;;
  esac
done

echo ""
echo "=== Final Slurm STDOUT ==="
cat "${SLURM_OUT}" || true
echo ""
echo "=== Final Slurm STDERR ==="
cat "${SLURM_ERR}" || true

# Check job state and exit code
EXIT_CODE_FULL=$(sacct -j ${JOB_ID} --format=ExitCode --noheader | head -n1 | tr -d ' ')
EXIT_CODE=$(echo "$EXIT_CODE_FULL" | cut -d: -f1)
SIGNAL=$(echo "$EXIT_CODE_FULL" | cut -d: -f2)

echo "Job state: ${JOB_STATE}, exit code: ${EXIT_CODE_FULL}"

# Fail on non-COMPLETED states (TIMEOUT, CANCELLED, FAILED, etc.)
if [ "$JOB_STATE" != "COMPLETED" ]; then
  echo "Job did not complete successfully (state: ${JOB_STATE})"
  exit 1
fi

# Fail on non-zero exit code or signal
if [ "$EXIT_CODE" != "0" ] || [ "$SIGNAL" != "0" ]; then
  echo "Job failed with exit code: ${EXIT_CODE_FULL}"
  exit 1
fi

