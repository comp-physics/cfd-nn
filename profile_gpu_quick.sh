#!/bin/bash
# Quick local GPU profiling script (no SLURM, run directly on GPU node)

set -e

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected - run this on a GPU node"
    exit 1
fi

echo "========================================"
echo "Quick GPU Profiling"
echo "========================================"
echo ""

# Check if build exists
if [ ! -d "build_refactor" ]; then
    echo "ERROR: build_refactor directory not found!"
    echo "Please build first with GPU offload enabled"
    exit 1
fi

cd build_refactor

# Test parameters (small for quick test)
NX=128
NY=256
ITERS=50

echo "Test: ${NX}x${NY} grid, $ITERS iterations"
echo ""

# ============================================
# Part 1: Data Movement Check
# ============================================
echo "=== Part 1: Data Movement Check ==="
echo ""

export LIBOMPTARGET_INFO=8
export OMP_TARGET_OFFLOAD=MANDATORY

./channel --Nx $NX --Ny $NY --nu 0.001 --max_iter $ITERS \
         --model baseline --dp_dx -0.0001 \
         --output quick_profile --num_snapshots 0 \
         2>&1 | tee quick_profile.log

# Count transfers
UPDATE_TO=$(grep -c "target update to" quick_profile.log || echo "0")
UPDATE_FROM=$(grep -c "target update from" quick_profile.log || echo "0")

echo ""
echo "Data Transfer Summary:"
echo "  H→D updates: $UPDATE_TO"
echo "  D→H updates: $UPDATE_FROM"

if [ "$UPDATE_TO" -gt 10 ]; then
    echo "  ⚠️  Too many H→D transfers!"
elif [ "$UPDATE_TO" -le 5 ]; then
    echo "  ✓ H→D transfers look good"
fi

if [ "$UPDATE_FROM" -gt 5 ]; then
    echo "  ⚠️  Too many D→H transfers!"
elif [ "$UPDATE_FROM" -le 2 ]; then
    echo "  ✓ D→H transfers look good"
fi

echo ""
echo "Full log: build_refactor/quick_profile.log"
echo ""

# ============================================
# Part 2: Basic Performance
# ============================================
echo "=== Part 2: GPU Utilization ==="
echo ""

unset LIBOMPTARGET_INFO
unset LIBOMPTARGET_DEBUG

# Monitor GPU while running
echo "Starting GPU monitoring (10 seconds)..."
nvidia-smi dmon -c 10 &
MONITOR_PID=$!

sleep 1

# Run a test
./channel --Nx $NX --Ny $NY --nu 0.001 --max_iter 100 \
         --model baseline --dp_dx -0.0001 \
         --output perf_test --num_snapshots 0 --quiet

wait $MONITOR_PID || true

echo ""
echo "✓ Quick profiling complete"
echo ""
echo "Next steps:"
echo "  - Full profiling: sbatch profile_gpu_performance.sh"
echo "  - Nsight trace:  nsys profile ./channel ..."


