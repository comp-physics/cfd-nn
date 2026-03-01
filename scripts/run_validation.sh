#!/usr/bin/env bash
# Run Tier 2 turbulence validation suite on GPU via SLURM
#
# Submits long-running DNS, RANS, TGV, and Poiseuille jobs, then generates
# a validation report with plots and error metrics.
#
# Usage:
#   ./scripts/run_validation.sh [--build] [--report-only]
#
# Prerequisites:
#   1. GPU build exists in build/
#   2. Reference data downloaded: scripts/download_reference_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
OUTPUT_DIR="$PROJECT_DIR/output/validation"
REPORT_DIR="$PROJECT_DIR/output/validation_report"
DATA_DIR="$PROJECT_DIR/data/reference"

DO_BUILD=false
REPORT_ONLY=false

for arg in "$@"; do
    case $arg in
        --build) DO_BUILD=true ;;
        --report-only) REPORT_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

# Download reference data if not present
if [ ! -d "$DATA_DIR/mkm_retau180" ]; then
    echo "Downloading reference data..."
    "$SCRIPT_DIR/download_reference_data.sh" "$DATA_DIR"
fi

# Build if requested
if $DO_BUILD; then
    echo "Building GPU release..."
    "$SCRIPT_DIR/../make.sh" gpu
fi

if $REPORT_ONLY; then
    echo "=== Report-only mode ==="
    python3 "$SCRIPT_DIR/generate_validation_report.py" \
        --output-dir "$REPORT_DIR" \
        --data-dir "$DATA_DIR" \
        --sim-dir "$OUTPUT_DIR"
    exit 0
fi

# Check build exists
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: build/channel not found. Run with --build or build first."
    exit 1
fi

echo "=== Tier 2 Turbulence Validation Suite ==="
echo "  Build:  $BUILD_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create SLURM job script
JOB_SCRIPT="$OUTPUT_DIR/validation_jobs.sbatch"
cat > "$JOB_SCRIPT" << 'JOBEOF'
#!/bin/bash
#SBATCH -J cfd-nn-validation
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 04:00:00
#SBATCH -o __OUTPUT_DIR__/slurm-%j.out
#SBATCH -e __OUTPUT_DIR__/slurm-%j.err

set -euo pipefail

module reset
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY

cd __PROJECT_DIR__
BUILD=__BUILD_DIR__
OUT=__OUTPUT_DIR__

echo "=== Turbulence Validation Suite ==="
echo "  Started: $(date)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo ""

# ---- DNS Channel Re_tau=180 ----
echo "=== DNS Channel (long run) ==="
$BUILD/channel --config examples/02_dns_channel/dns_channel_re180.cfg \
    --Nx 192 --Ny 96 --Nz 192 \
    --max_steps 20000 \
    --output_prefix "$OUT/dns_channel" \
    --profile_output "$OUT/dns_channel_u_plus.dat" \
    --stress_output "$OUT/dns_channel_stresses.dat" \
    2>&1 | tee "$OUT/dns_channel.log"
echo ""

# ---- RANS models (all 10, converged) ----
echo "=== RANS Model Sweep ==="
MODELS="none baseline gep sst komega earsm_wj earsm_gs earsm_pope nn_mlp nn_tbnn"
for model in $MODELS; do
    echo "  Running $model..."
    cfg="examples/04_rans_validation/${model}_re180.cfg"
    if [ -f "$cfg" ]; then
        $BUILD/channel --config "$cfg" \
            --max_steps 5000 \
            --output_prefix "$OUT/rans_${model}" \
            --profile_output "$OUT/${model}_u_plus.dat" \
            2>&1 | tee "$OUT/rans_${model}.log" || \
            echo "  [WARN] $model failed"
    else
        echo "  [SKIP] Config not found: $cfg"
    fi
done
echo ""

# ---- TGV Re=1600 ----
echo "=== TGV Re=1600 ==="
if [ -f "$BUILD/tgv" ]; then
    $BUILD/tgv --Re 1600 --N 128 --max_steps 5000 \
        --energy_output "$OUT/tgv_re1600_energy.dat" \
        2>&1 | tee "$OUT/tgv.log"
elif [ -f examples/05_tgv/tgv_re1600.cfg ]; then
    $BUILD/channel --config examples/05_tgv/tgv_re1600.cfg \
        --max_steps 5000 \
        --output_prefix "$OUT/tgv" \
        2>&1 | tee "$OUT/tgv.log"
else
    echo "  [SKIP] No TGV config found"
fi
echo ""

# ---- Poiseuille (grid convergence) ----
echo "=== Poiseuille Grid Convergence ==="
for ny in 32 64 128 256; do
    nx=$((ny / 2))
    echo "  Grid ${nx}x${ny}..."
    $BUILD/channel --config examples/01_laminar_channel/poiseuille.cfg \
        --Nx $nx --Ny $ny \
        --max_steps 50000 \
        --output_prefix "$OUT/poiseuille_${ny}" \
        --profile_output "$OUT/poiseuille_${ny}_profile.dat" \
        2>&1 | tee "$OUT/poiseuille_${ny}.log" || \
        echo "  [WARN] Ny=$ny failed"
done
echo ""

echo "=== All validation jobs complete ==="
echo "  Finished: $(date)"
JOBEOF

# Substitute paths
sed -i "s|__OUTPUT_DIR__|$OUTPUT_DIR|g" "$JOB_SCRIPT"
sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$JOB_SCRIPT"
sed -i "s|__BUILD_DIR__|$BUILD_DIR|g" "$JOB_SCRIPT"

echo "Submitting SLURM job..."
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
echo "  Job ID: $JOB_ID"
echo "  Output: $OUTPUT_DIR/slurm-${JOB_ID}.out"
echo ""
echo "After job completes, generate report:"
echo "  python3 scripts/generate_validation_report.py --sim-dir $OUTPUT_DIR"
