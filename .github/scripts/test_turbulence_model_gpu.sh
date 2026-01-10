#!/bin/bash
# Test a single turbulence model on GPU with validation
# 
# Usage: test_turbulence_model_gpu.sh <model> <name> <Nx> <Ny> <max_iter> <output_prefix> [min_vel_tolerance]

set -e

MODEL="$1"
NAME="$2"
NX="${3:-256}"
NY="${4:-512}"
MAX_ITER="${5:-20}"
OUTPUT_PREFIX="${6:-output/test}"
MIN_VEL_TOL="${7:-}"

echo "========================================"
echo "Testing: $NAME"
echo "Model: $MODEL"
echo "Grid: ${NX}x${NY}"
echo "Max iterations: $MAX_ITER"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_PREFIX"

# Run the model
echo "Running simulation..."
./channel --Nx $NX --Ny $NY --nu 0.001 --max_steps $MAX_ITER \
         --model $MODEL --dp_dx -0.0001 \
         --output $OUTPUT_PREFIX --num_snapshots 0 --quiet

echo "Simulation completed."
echo ""

# Validate output
SCRIPT_DIR="$(dirname "$0")"
if [ -n "$MIN_VEL_TOL" ]; then
    "$SCRIPT_DIR/validate_turbulence_model.sh" "$NAME" "$OUTPUT_PREFIX" "$MIN_VEL_TOL"
else
    "$SCRIPT_DIR/validate_turbulence_model.sh" "$NAME" "$OUTPUT_PREFIX"
fi

echo "[OK] $NAME validated successfully"
echo ""

