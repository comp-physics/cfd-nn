#!/bin/bash
#SBATCH -J gpu_profile_debug
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -t0:30:00
#SBATCH -qembers
#SBATCH -Agts-sbryngelson3-paid
#SBATCH --gres=gpu:1

# ============================================================================
# GPU Data Movement Profiling
# Tracks all CPU<->GPU transfers to ensure minimal data movement
# ============================================================================

module load nvhpc

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_refactor

echo "========================================"
echo "GPU Data Movement Profiling"
echo "========================================"
echo ""

# Environment variables for OpenMP GPU debugging
export LIBOMPTARGET_INFO=8           # Max verbosity for data transfers
export LIBOMPTARGET_DEBUG=1          # Enable debug output
export LIBOMPTARGET_KERNEL_TRACE=1   # Trace kernel launches
export OMP_TARGET_OFFLOAD=MANDATORY  # Fail if GPU not available

echo "Environment Variables:"
echo "  LIBOMPTARGET_INFO=$LIBOMPTARGET_INFO (8 = max verbosity)"
echo "  LIBOMPTARGET_DEBUG=$LIBOMPTARGET_DEBUG"
echo "  LIBOMPTARGET_KERNEL_TRACE=$LIBOMPTARGET_KERNEL_TRACE"
echo "  OMP_TARGET_OFFLOAD=$OMP_TARGET_OFFLOAD"
echo ""

# Run a SHORT test to see data movement
echo "Running SHORT test (50 iterations)..."
echo "Watch for 'target update to/from' - these are data transfers"
echo ""

./channel --Nx 128 --Ny 256 --nu 0.001 --max_iter 50 \
         --model baseline --dp_dx -0.0001 \
         --output debug_profile --num_snapshots 0 --verbose \
         2>&1 | tee gpu_data_movement.log

echo ""
echo "========================================"
echo "Data Movement Analysis"
echo "========================================"
echo ""

# Analyze the log
echo "Counting data transfers..."
ENTER_DATA=$(grep -c "target enter data" gpu_data_movement.log || echo "0")
EXIT_DATA=$(grep -c "target exit data" gpu_data_movement.log || echo "0")
UPDATE_TO=$(grep -c "target update to" gpu_data_movement.log || echo "0")
UPDATE_FROM=$(grep -c "target update from" gpu_data_movement.log || echo "0")
KERNEL_LAUNCH=$(grep -c "Launching kernel" gpu_data_movement.log || echo "0")

echo "Data Transfer Summary:"
echo "  enter data:   $ENTER_DATA  (initial allocation + upload)"
echo "  exit data:    $EXIT_DATA   (final download + free)"
echo "  update to:    $UPDATE_TO   (H->D copies during iteration)"
echo "  update from:  $UPDATE_FROM (D->H copies during iteration)"
echo "  kernel launches: $KERNEL_LAUNCH"
echo ""

# Expected behavior
echo "Expected Behavior:"
echo "  enter data:   ~16  (once at init for all fields)"
echo "  exit data:    ~16  (once at end)"
echo "  update to:    2-4  (only for critical sync points)"
echo "  update from:  1-2  (only for I/O)"
echo ""

if [ "$UPDATE_TO" -gt 10 ]; then
    echo "⚠️  WARNING: Too many H->D transfers!"
    echo "   Should be <5 for a 50-iteration run"
    echo "   Check for unnecessary sync_to_gpu() calls"
fi

if [ "$UPDATE_FROM" -gt 5 ]; then
    echo "⚠️  WARNING: Too many D->H transfers!"
    echo "   Should be <3 for a 50-iteration run (only for I/O)"
    echo "   Check for unnecessary sync_from_gpu() calls"
fi

echo ""
echo "Full log saved to: gpu_data_movement.log"
echo "Search for 'Copying data' to see transfer sizes"
echo ""
echo "To view detailed transfers:"
echo "  grep 'Copying data' gpu_data_movement.log"
echo "  grep 'target update' gpu_data_movement.log"




