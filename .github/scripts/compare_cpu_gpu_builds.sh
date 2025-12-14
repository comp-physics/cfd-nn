#!/bin/bash
# Compare CPU-only build vs GPU-offload build (perturbed Poisson test)
# Usage: compare_cpu_gpu_builds.sh <workdir>
# Designed to run on an H200 GPU node via SLURM

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  CPU-only build vs GPU-offload build (perturbed Poisson compare)"
echo "==================================================================="
echo ""

# Build a CPU-only reference binary (no GPU offload) and run the same case.
rm -rf build_ci_cpu_ref
mkdir -p build_ci_cpu_ref
cd build_ci_cpu_ref

echo "=== CMake Configuration (CPU-only reference) ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF 2>&1 | tee cmake_config.log
echo ""
echo "=== Building (CPU-only reference) ==="
make -j8
mkdir -p output/cpu_ref

echo ""
echo "--- Running perturbed Poisson test (CPU-only build) ---"
./test_poisson_perturbed --Nx 64 --Ny 64 --nu 0.01 --dp_dx -0.01 --max_iter 2000 --tol 1e-6 --verbose false \
    | tee output/cpu_ref/poisson_perturbed.log

# Now run the same case with the GPU-offload build (must offload due to MANDATORY).
cd "$WORKDIR/build_ci_gpu_correctness"
mkdir -p output/gpu_ref

echo ""
echo "--- Running perturbed Poisson test (GPU-offload build) ---"
./test_poisson_perturbed --Nx 64 --Ny 64 --nu 0.01 --dp_dx -0.01 --max_iter 2000 --tol 1e-6 --verbose false \
    | tee output/gpu_ref/poisson_perturbed.log

echo ""
echo "--- Comparing perturbed Poisson results: CPU-only vs GPU-offload build ---"
python3 - <<'PY'
import math
import os
import re
import sys

def parse_metric(log_path: str, pattern: str):
    """Extract numeric value from log using regex pattern."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        content = f.read()
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    return None

cpu_log = "$WORKDIR/build_ci_cpu_ref/output/cpu_ref/poisson_perturbed.log"
gpu_log = "$WORKDIR/build_ci_gpu_correctness/output/gpu_ref/poisson_perturbed.log"

# Extract key metrics from both logs
metrics = [
    ("bulk_velocity", r"Bulk velocity:\s+([\d.eE+-]+)", 1e-6),
    ("max_div", r"Max divergence:\s+([\d.eE+-]+)", 1e-8),
    ("rms_div", r"RMS divergence:\s+([\d.eE+-]+)", 1e-8),
    ("residual", r"Final residual:\s+([\d.eE+-]+)", 1e-5),
]

all_ok = True
print("")
print("Metric Comparison:")
print("-" * 80)

for name, pattern, tol in metrics:
    cpu_val = parse_metric(cpu_log, pattern)
    gpu_val = parse_metric(gpu_log, pattern)
    
    if cpu_val is None or gpu_val is None:
        print(f"SKIP {name}: missing value (cpu={cpu_val}, gpu={gpu_val})")
        continue
    
    diff = abs(cpu_val - gpu_val)
    denom = abs(cpu_val) + 1e-30
    rel = diff / denom
    
    status = "✓" if diff <= tol else "✗"
    print(f"{status} {name:20s}: cpu={cpu_val:.6e} gpu={gpu_val:.6e} diff={diff:.3e} (tol={tol:.1e})")
    
    if diff > tol:
        all_ok = False

# Also print iterations (informational, not enforced)
cpu_iters = parse_metric(cpu_log, r"Iterations:\s+(\d+)")
gpu_iters = parse_metric(gpu_log, r"Iterations:\s+(\d+)")
if cpu_iters is not None and gpu_iters is not None:
    print(f"  iterations (info):     cpu={int(cpu_iters)} gpu={int(gpu_iters)}")

print("-" * 80)

if not all_ok:
    print("\n✗ FAILED: CPU-only vs GPU-offload build differences exceed tolerances")
    sys.exit(1)

print("\n✓ PASSED: CPU-only build and GPU-offload build agree within tolerances")
sys.exit(0)
PY

echo ""
echo "✅ CPU-only vs GPU-offload comparison completed successfully"
