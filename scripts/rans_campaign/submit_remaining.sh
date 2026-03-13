#!/bin/bash
# Submit remaining array tasks 36-37 once QOS limit allows
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RANS_PROJECT_DIR="$(pwd)"
echo "Submitting remaining tasks 36-37..."
sbatch --parsable --export=ALL --array=36-37 scripts/rans_campaign/submit_campaign.sbatch
