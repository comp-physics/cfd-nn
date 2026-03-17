#!/bin/bash
#SBATCH --job-name=gpu-baseline
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=embers
#SBATCH -A gts-sbryngelson3
#SBATCH --output=gpu_baseline_%j.log

set -euo pipefail

PROJ_DIR="/storage/scratch1/6/sbryngelson3/cfd-nn"
BUILD_DIR="${PROJ_DIR}/build_gpu_baseline"

module load nvhpc/25.5 cmake

echo "=== GPU Baseline Population ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Build with GPU offload
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake "$PROJ_DIR" \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DUSE_GPU_OFFLOAD=ON \
    -DGPU_CC=75 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    2>&1 | tail -5

make -j4 2>&1 | tail -5

export OMP_TARGET_OFFLOAD=MANDATORY

# Tests that emit QOI_JSON matching baseline_gpu.json sections
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

QOI_FILE="${PROJ_DIR}/gpu_qoi_raw.txt"
> "$QOI_FILE"

echo ""
echo "=== Running tests ==="
for t in "${TESTS[@]}"; do
    if [ -x "./$t" ]; then
        echo "--- $t ---"
        ./$t 2>&1 | tee -a "${PROJ_DIR}/gpu_test_output.log" | grep "QOI_JSON:" >> "$QOI_FILE" || true
        echo ""
    else
        echo "SKIP: $t not found"
    fi
done

echo ""
echo "=== QOI Results ==="
cat "$QOI_FILE"

# Parse QOI into baseline JSON
cd "$PROJ_DIR"
python3 - "$QOI_FILE" "${PROJ_DIR}/tests/baselines/baseline_gpu.json" <<'PYEOF'
import json, sys, re
from datetime import date

qoi_file = sys.argv[1]
out_file = sys.argv[2]

# Read all QOI_JSON lines
qois = []
with open(qoi_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith("QOI_JSON:"):
            line = line[len("QOI_JSON:"):].strip()
        if line:
            try:
                qois.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARN: bad JSON: {line}", file=sys.stderr)

# Map test names to baseline sections
mapping = {
    "poiseuille_steady": "poiseuille_steady",
    "energy_budget": "energy_budget",
    "projection_effectiveness": "projection_effectiveness",
    "op_adjoint": "op_adjoint",
    "projection_divfree": "projection_divfree",
    "divergence_reduction": "divergence_reduction",
    "advection_rotation": "advection_rotation",
    "poiseuille_refine": "poiseuille_refine",
    "rans_frame_invariance": "rans_frame_invariance",
    "galilean_invariance": "galilean_invariance",
    "projection_galilean": "projection_galilean",
    "galilean_breakdown": "galilean_breakdown",
}

baseline = {
    "schema_version": "2",
    "build_type": "gpu",
    "created": str(date.today()),
    "commit": "TODO",
}

# Group QOI by test name
for q in qois:
    test_name = q.get("test", "")
    section = mapping.get(test_name)
    if section:
        if section not in baseline:
            baseline[section] = {}
        for k, v in q.items():
            if k != "test":
                baseline[section][k] = v
    else:
        print(f"INFO: unmapped test '{test_name}'", file=sys.stderr)

# Fill commit from git
import subprocess
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=sys.argv[2].rsplit("/", 2)[0]).decode().strip()
    baseline["commit"] = commit
except:
    pass

with open(out_file, "w") as f:
    json.dump(baseline, f, indent=2)
    f.write("\n")

print(f"Wrote {out_file} with {len([k for k in baseline if isinstance(baseline[k], dict)])} sections")
PYEOF

echo ""
echo "=== Generated baseline ==="
cat "${PROJ_DIR}/tests/baselines/baseline_gpu.json"
echo ""
echo "=== DONE ==="
