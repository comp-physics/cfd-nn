#!/bin/bash
#
# Run all turbulence models and compare results
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=========================================================="
echo "Turbulent Channel Flow - Turbulence Model Comparison"
echo "=========================================================="
echo ""
echo "This will run 5 turbulence models:"
echo "  1. None        - No model (laminar-like, for reference)"
echo "  2. Baseline    - Mixing length with van Driest damping"
echo "  3. GEP         - Gene expression programming"
echo "  4. NN-MLP      - Neural network (scalar eddy viscosity)"
echo "  5. NN-TBNN     - Tensor basis neural network"
echo ""
echo "Note: NN models use example weights (random) unless you've"
echo "      trained your own. See README.md for training instructions."
echo ""

# Check if solver is built
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: Solver not found at $BUILD_DIR/channel"
    echo "Please build the project first."
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{no_model,baseline,gep,nn_mlp,nn_tbnn}

# Array of models and their configs
models=(
    "no_model:01_no_model.cfg"
    "baseline:02_baseline.cfg"
    "gep:03_gep.cfg"
    "nn_mlp:04_nnmlp.cfg"
    "nn_tbnn:05_nntbnn.cfg"
)

cd "$BUILD_DIR"

# Run each model
for model_config in "${models[@]}"; do
    IFS=':' read -r model cfg <<< "$model_config"
    
    echo ""
    echo "=========================================="
    echo "Running: $model"
    echo "=========================================="
    
    ./channel --config "$EXAMPLE_DIR/$cfg" \
              --output "$OUTPUT_DIR/$model" \
        || echo "WARNING: $model failed or didn't converge fully"
    
    echo "[OK] $model complete"
done

echo ""
echo "=========================================================="
echo "All simulations complete!"
echo "=========================================================="
echo ""
echo "Output saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Compare results:"
echo "     python $EXAMPLE_DIR/compare_models.py"
echo ""
echo "  2. View in ParaView:"
echo "     paraview $OUTPUT_DIR/*/velocity_final.vtk"
echo ""

# Run comparison if Python available
if command -v python3 &> /dev/null; then
    echo "Running automated comparison..."
    python3 "$EXAMPLE_DIR/compare_models.py"
else
    echo "Python3 not found - skipping automated comparison"
fi

