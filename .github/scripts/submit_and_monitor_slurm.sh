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

# Wait for job to fully complete
scontrol show job ${JOB_ID} >/dev/null 2>&1 || true

echo ""
echo "=== Final Slurm STDOUT ==="
cat "${SLURM_OUT}" || true
echo ""
echo "=== Final Slurm STDERR ==="
cat "${SLURM_ERR}" || true

# Check job exit code
EXIT_CODE=$(sacct -j ${JOB_ID} --format=ExitCode --noheader | head -n1 | cut -d: -f1 | tr -d ' ')
if [ "$EXIT_CODE" != "0" ]; then
  echo "Job failed with exit code: ${EXIT_CODE}"
  exit 1
fi
