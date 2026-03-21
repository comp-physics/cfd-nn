#!/bin/bash
# Submit all paper experiments: 2 cases x 5 models x 4 grids = 40 jobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== NN vs RANS Paper Experiment Submission ==="
echo "Total runs: 40 (2 cases x 5 models x 4 grids)"
echo ""

echo "--- Channel flow (20 jobs) ---"
"$SCRIPT_DIR/submit_channel.sh"

echo ""
echo "--- Periodic hills (20 jobs) ---"
"$SCRIPT_DIR/submit_hills.sh"

echo ""
echo "=== All 40 jobs submitted ==="
echo "Monitor with: squeue -u \$USER -n nn-paper"
echo "Results in:   results/paper/{channel,hills}/<model>_<grid>/"
