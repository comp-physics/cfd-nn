#!/bin/bash
# Collect results from all paper experiment runs into a CSV
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$ROOT_DIR/results/paper"
CSV_FILE="$RESULTS_BASE/experiment_results.csv"

CASES=(channel hills)
MODELS=(baseline sst earsm_pope nn_mlp nn_tbnn)
GRIDS=(D A B C)

# Grid cell counts
declare -A NCELLS_CHANNEL
NCELLS_CHANNEL[D]=$((256 * 192 * 256))
NCELLS_CHANNEL[A]=$((128 * 96 * 128))
NCELLS_CHANNEL[B]=$((64 * 64 * 64))
NCELLS_CHANNEL[C]=$((32 * 32 * 32))

declare -A NCELLS_HILLS
NCELLS_HILLS[D]=$((256 * 128 * 128))
NCELLS_HILLS[A]=$((128 * 64 * 64))
NCELLS_HILLS[B]=$((64 * 32 * 32))
NCELLS_HILLS[C]=$((32 * 16 * 16))

echo "=== Collecting Paper Experiment Results ==="
echo ""

# CSV header
echo "case,model,grid,ncells,status,steps,step_time_ms,total_time_s,final_residual,u_tau,re_tau" \
    > "$CSV_FILE"

total=0
completed=0
failed=0

for case_name in "${CASES[@]}"; do
    for model in "${MODELS[@]}"; do
        for grid in "${GRIDS[@]}"; do
            total=$((total + 1))
            outdir="$RESULTS_BASE/${case_name}/${model}_${grid}"

            # Get cell count
            if [ "$case_name" = "channel" ]; then
                ncells="${NCELLS_CHANNEL[$grid]}"
            else
                ncells="${NCELLS_HILLS[$grid]}"
            fi

            # Find latest slurm output
            slurm_out=$(ls -t "$outdir"/slurm_*.out 2>/dev/null | head -1)

            if [ -z "$slurm_out" ]; then
                echo "  MISSING: ${case_name}/${model}_${grid}"
                echo "${case_name},${model},${grid},${ncells},missing,,,,,," >> "$CSV_FILE"
                failed=$((failed + 1))
                continue
            fi

            # Check if job completed
            if grep -q "TIMING SUMMARY" "$slurm_out" 2>/dev/null; then
                status="completed"
                completed=$((completed + 1))
            else
                status="incomplete"
                failed=$((failed + 1))
            fi

            # Extract step count
            steps=$(grep -oP 'Step\s+\K[0-9]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")

            # Extract timing (ms/step from TimingStats summary)
            step_time_ms=$(grep -oP 'step\s*:\s*\K[0-9.]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")

            # Extract total time
            total_time_s=$(grep -oP 'Total wall time\s*:\s*\K[0-9.]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")

            # Extract final residual
            final_residual=$(grep -oP 'residual\s*=\s*\K[0-9.eE+-]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")

            # Extract u_tau / Re_tau
            u_tau=$(grep -oP 'u_tau\s*=\s*\K[0-9.eE+-]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")
            re_tau=$(grep -oP 'Re_tau\s*=\s*\K[0-9.eE+-]+' "$slurm_out" 2>/dev/null | tail -1 || echo "")

            echo "  ${status}: ${case_name}/${model}_${grid} (${steps:-?} steps, ${step_time_ms:-?} ms/step)"
            echo "${case_name},${model},${grid},${ncells},${status},${steps},${step_time_ms},${total_time_s},${final_residual},${u_tau},${re_tau}" \
                >> "$CSV_FILE"
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total runs:  ${total}"
echo "Completed:   ${completed}"
echo "Failed/miss: ${failed}"
echo ""
echo "CSV written: ${CSV_FILE}"
