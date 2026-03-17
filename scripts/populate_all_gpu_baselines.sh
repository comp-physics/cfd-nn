#!/bin/bash
# Submit GPU baseline population jobs to all CI GPU partitions.
# Each job builds, runs tests, and writes a GPU-specific baseline file.
# Usage: ./scripts/populate_all_gpu_baselines.sh

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASELINES_DIR="${PROJ_DIR}/tests/baselines"

# CI partitions (must match gpu-ci.yml sbatch -p list)
PARTITIONS=(gpu-h200 gpu-h100 gpu-a100 gpu-l40s)

# GPU compute capabilities per partition
declare -A GPU_CC
GPU_CC[gpu-h200]=90
GPU_CC[gpu-h100]=90
GPU_CC[gpu-a100]=80
GPU_CC[gpu-l40s]=89

echo "=== Submitting GPU baseline jobs ==="
echo "Project: ${PROJ_DIR}"
echo ""

for part in "${PARTITIONS[@]}"; do
    # Check partition has any non-drained nodes
    avail=$(sinfo -p "$part" --noheader --format="%T" 2>/dev/null | grep -cvE "drained|down" || true)
    if [ "$avail" -eq 0 ]; then
        echo "SKIP: $part (no available nodes)"
        continue
    fi

    cc=${GPU_CC[$part]}
    job_name="baseline-${part}"

    # Submit inline sbatch
    JOBID=$(sbatch --parsable <<SBATCH
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${part}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=embers
#SBATCH -A gts-sbryngelson3
#SBATCH --output=${PROJ_DIR}/gpu_baseline_${part}_%j.log

set -euo pipefail

module load nvhpc/25.5 cmake

PROJ_DIR="${PROJ_DIR}"
BUILD_DIR="\${PROJ_DIR}/build_gpu_baseline_${part}"

echo "=== GPU Baseline: ${part} ==="
echo "Node: \$(hostname)"
echo "Date: \$(date)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

mkdir -p "\$BUILD_DIR" && cd "\$BUILD_DIR"
cmake "\$PROJ_DIR" \\
    -DCMAKE_CXX_COMPILER=nvc++ \\
    -DUSE_GPU_OFFLOAD=ON \\
    -DGPU_CC=${cc} \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DBUILD_TESTS=ON \\
    2>&1 | tail -5

make -j4 2>&1 | tail -5

export OMP_TARGET_OFFLOAD=MANDATORY

TESTS=(
    test_poiseuille_steady
    test_energy_budget_channel
    test_projection_effectiveness
    test_conservation_audit
    test_advection_rotation
    test_poiseuille_refinement
    test_rans_frame_invariance
    test_galilean_invariance
    test_projection_galilean
    test_galilean_stage_breakdown
)

QOI_FILE="\${PROJ_DIR}/gpu_qoi_${part}.txt"
> "\$QOI_FILE"

for t in "\${TESTS[@]}"; do
    if [ -x "./\$t" ]; then
        echo "--- \$t ---"
        ./\$t 2>&1 | grep "QOI_JSON:" >> "\$QOI_FILE" || true
    fi
done

GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
OUT_FILE="\${PROJ_DIR}/tests/baselines/baseline_gpu_\${GPU_NAME}.json"

cd "\$PROJ_DIR"
python3 - "\$QOI_FILE" "\$OUT_FILE" <<'PYEOF'
import json, sys, subprocess
from datetime import date

qoi_file, out_file = sys.argv[1], sys.argv[2]

qois = []
with open(qoi_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith("QOI_JSON:"): line = line[len("QOI_JSON:"):].strip()
        if line:
            try: qois.append(json.loads(line))
            except: pass

mapping = {k: k for k in [
    "poiseuille_steady", "energy_budget", "projection_effectiveness",
    "op_adjoint", "projection_divfree", "divergence_reduction",
    "advection_rotation", "poiseuille_refine", "rans_frame_invariance",
    "galilean_invariance", "projection_galilean", "galilean_breakdown",
]}

baseline = {"schema_version": "2", "build_type": "gpu", "created": str(date.today()), "commit": "TODO"}
for q in qois:
    section = mapping.get(q.get("test", ""))
    if section:
        baseline.setdefault(section, {}).update({k: v for k, v in q.items() if k != "test"})

try:
    baseline["commit"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=out_file.rsplit("/", 2)[0]
    ).decode().strip()
except: pass

with open(out_file, "w") as f:
    json.dump(baseline, f, indent=2)
    f.write("\\n")
print(f"Wrote {out_file} with {len([k for k in baseline if isinstance(baseline[k], dict)])} sections")
PYEOF

rm -f "\$QOI_FILE"
echo "=== DONE: \${GPU_NAME} ==="
cat "\$OUT_FILE"
SBATCH
)
    echo "Submitted ${part}: job ${JOBID}"
done

echo ""
echo "Monitor with: squeue -u \$USER -n baseline-gpu-h200,baseline-gpu-h100,baseline-gpu-a100,baseline-gpu-l40s"
echo "After completion, commit the new files in tests/baselines/"
