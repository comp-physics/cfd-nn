#!/usr/bin/env bash
# Run Tier 2 turbulence validation suite on GPU via SLURM
#
# Submits DNS as a dedicated long job and RANS/TGV/Poiseuille as a shorter job.
# Results are validated by parsing solver output logs.
#
# Usage:
#   ./scripts/run_validation.sh [--build] [--report-only] [--dns-only] [--fast-only]
#
# Prerequisites:
#   1. GPU build exists in build_gpu_validation/ (or use --build)
#   2. Reference data: scripts/download_reference_data.sh (auto-downloaded)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_gpu_validation"
OUTPUT_DIR="$PROJECT_DIR/output/validation_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="$PROJECT_DIR/data/reference"

DO_BUILD=false
REPORT_ONLY=false
DNS_ONLY=false
FAST_ONLY=false

for arg in "$@"; do
    case $arg in
        --build) DO_BUILD=true ;;
        --report-only) REPORT_ONLY=true ;;
        --dns-only) DNS_ONLY=true ;;
        --fast-only) FAST_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Download reference data if not present
if [ ! -d "$DATA_DIR/mkm_retau180" ]; then
    echo "Downloading reference data..."
    "$SCRIPT_DIR/download_reference_data.sh" "$DATA_DIR"
fi

if $REPORT_ONLY; then
    echo "=== Report-only mode ==="
    echo "Looking for latest output in output/validation_*..."
    LATEST=$(ls -d "$PROJECT_DIR/output/validation_"* 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No validation output found"
        exit 1
    fi
    echo "Using: $LATEST"
    bash "$SCRIPT_DIR/parse_validation_results.sh" "$LATEST"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Tier 2 Turbulence Validation Suite ==="
echo "  Build:  $BUILD_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Common SLURM preamble
make_preamble() {
    local job_name=$1 wall_time=$2 out_dir=$3
    cat << PREAMBLE
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t ${wall_time}
#SBATCH -o ${out_dir}/slurm-${job_name}-%j.out
#SBATCH -e ${out_dir}/slurm-${job_name}-%j.err

set -uo pipefail
module reset
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY

BUILD="${BUILD_DIR}"
OUT="${out_dir}"
PROJECT="${PROJECT_DIR}"

echo "=== \${SLURM_JOB_NAME} ==="
echo "  Started: \$(date)"
echo "  Node: \$(hostname)"
echo "  GPU: \$(nvidia-smi -L 2>/dev/null | head -1)"
echo ""

# Ensure GPU build exists
if [ ! -f "\$BUILD/channel" ]; then
    echo "Building GPU release in \$BUILD..."
    mkdir -p \$BUILD && cd \$BUILD
    cmake \$PROJECT -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90
    make -j\$(nproc) channel
    cd \$PROJECT
    echo "Build complete"
    echo ""
else
    echo "Using existing GPU build: \$BUILD/channel"
fi
PREAMBLE
}

# ================================================================
# Job 1: DNS Channel (long — needs 6 hours for 5000 steps at 192x96x192)
# ================================================================
if ! $FAST_ONLY; then
    DNS_SCRIPT="$OUTPUT_DIR/dns.sbatch"
    {
        make_preamble "val-dns" "06:00:00" "$OUTPUT_DIR"
        cat << 'DNSEOF'

echo "================================================================"
echo "  DNS Channel Re_tau~180 (192x96x192, v13 recipe, 5000 steps)"
echo "================================================================"
$BUILD/channel --config $PROJECT/examples/07_unsteady_developing_channel/dns_retau180_3d_v13.cfg \
    --output $OUT/dns/ \
    2>&1 | tee $OUT/dns_channel.log || echo "  [WARN] DNS channel exited with code $?"

echo ""
echo "  DNS job finished: $(date)"
DNSEOF
    } > "$DNS_SCRIPT"

    echo "Submitting DNS job (6h wall time)..."
    DNS_ID=$(sbatch --parsable "$DNS_SCRIPT")
    echo "  Job ID: $DNS_ID"
    echo ""
fi

# ================================================================
# Job 2: RANS + TGV + Poiseuille (fast — should finish in <2 hours)
# ================================================================
if ! $DNS_ONLY; then
    FAST_SCRIPT="$OUTPUT_DIR/fast_validation.sbatch"
    {
        make_preamble "val-fast" "02:00:00" "$OUTPUT_DIR"
        cat << 'FASTEOF'

# ============================================================
# 1. RANS Steady Channel (all available models)
# ============================================================
echo "================================================================"
echo "  RANS Model Sweep (steady, 64x128, stretched)"
echo "================================================================"

for model in none baseline gep sst komega earsm_wj earsm_gs earsm_pope; do
    echo "--- RANS model: $model ---"
    $BUILD/channel --config $PROJECT/examples/06_steady_rans_channel/baseline.cfg \
        --model $model \
        --nu 0.005556 --dp_dx -1.0 \
        --max_steps 20000 --tol 1e-8 \
        --output $OUT/rans_${model}/ \
        2>&1 | tee $OUT/rans_${model}.log || \
        echo "  [WARN] $model failed or did not converge"
    echo ""
done

# ============================================================
# 2. TGV Re=1600 (DNS, 64^3, 5000 steps)
# ============================================================
echo "================================================================"
echo "  TGV Re=1600 (64^3, 5000 steps)"
echo "================================================================"
$BUILD/channel --config $PROJECT/examples/09_taylor_green_3d/tg_re1600.cfg \
    --max_steps 5000 \
    --adaptive_dt --CFL 0.3 \
    --scheme central \
    --output $OUT/tgv/ \
    2>&1 | tee $OUT/tgv_re1600.log || echo "  [WARN] TGV exited with code $?"
echo ""

# ============================================================
# 3. Poiseuille Grid Convergence (4 grids)
# ============================================================
echo "================================================================"
echo "  Poiseuille Grid Convergence"
echo "================================================================"
for ny in 32 64 128 256; do
    nx=$((ny / 2))
    echo "--- Poiseuille grid ${nx}x${ny} ---"
    $BUILD/channel --config $PROJECT/examples/01_laminar_channel/poiseuille.cfg \
        --Nx $nx --Ny $ny \
        --max_steps 50000 --tol 1e-10 \
        --output $OUT/poiseuille_${ny}/ \
        2>&1 | tee $OUT/poiseuille_${ny}.log || \
        echo "  [WARN] Ny=$ny failed"
    echo ""
done

echo "================================================================"
echo "  Fast validation jobs complete: $(date)"
echo "================================================================"
FASTEOF
    } > "$FAST_SCRIPT"

    echo "Submitting fast validation job (2h wall time)..."
    FAST_ID=$(sbatch --parsable "$FAST_SCRIPT")
    echo "  Job ID: $FAST_ID"
    echo ""
fi

echo "Output: $OUTPUT_DIR"
echo ""
echo "After all jobs complete, parse results:"
echo "  bash scripts/parse_validation_results.sh $OUTPUT_DIR"
