#!/bin/bash
# Test script to validate GPU refactor by comparing CPU vs GPU results

set -e

echo "========================================="
echo "GPU Refactor Validation Test"
echo "========================================="
echo ""

# Check if we're on a GPU node
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi -L
    echo ""
else
    echo "WARNING: No GPU detected - will only test CPU path"
    echo ""
fi

# Test case parameters (run from build directory)
NX=64
NY=128
NU=0.001
MAX_ITER=100
OUTPUT_PREFIX="test_refactor"

echo "Test parameters:"
echo "  Grid: ${NX}x${NY}"
echo "  Viscosity: $NU"
echo "  Max iterations: $MAX_ITER"
echo ""

# Run a simple laminar channel flow case
echo "========================================="
echo "Running channel flow test..."
echo "========================================="
echo ""

./channel \
    --Nx $NX \
    --Ny $NY \
    --nu $NU \
    --model baseline \
    --adaptive_dt \
    --max_iter $MAX_ITER \
    --output_prefix $OUTPUT_PREFIX \
    --verbose

echo ""
echo "========================================="
echo "Test completed successfully!"
echo "========================================="
echo ""
echo "Output files written to: ${OUTPUT_PREFIX}_*"
echo ""

# Check if GPU was actually used
if [ -f "${OUTPUT_PREFIX}_final.vtk" ]; then
    echo "✓ VTK output generated"
else
    echo "✗ VTK output NOT found"
    exit 1
fi

echo ""
echo "To compare CPU vs GPU results:"
echo "  1. Run this script on CPU node"
echo "  2. Run this script on GPU node"  
echo "  3. Compare the output files"
echo ""

