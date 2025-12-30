#!/bin/bash
#
# Run turbulent channel simulation with selected model
#
# Usage: ./run.sh <config>
#   ./run.sh 01_no_model     (no turbulence model)
#   ./run.sh 02_baseline     (mixing length)
#   ./run.sh 03_gep          (GEP algebraic)
#   ./run.sh 04_nnmlp        (neural network MLP)
#   ./run.sh 05_nntbnn       (tensor basis NN)
#
# Or run all models:
#   ./run_all.sh
#
# Or run directly:
#   ./channel --config 02_baseline.cfg

set -euo pipefail

CASE="${1:-02_baseline}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

echo "=========================================================="
echo "Turbulent Channel Flow"
echo "=========================================================="
echo ""

# List available configs if none found
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config not found: $CFG"
    echo ""
    echo "Available configs:"
    ls -1 "$EXAMPLE_DIR"/*.cfg 2>/dev/null | xargs -n1 basename | sed 's/\.cfg$//' | grep -v config_base
    exit 2
fi

# Check if solver is built
if [[ ! -x "$BUILD_DIR/channel" ]]; then
    echo "ERROR: channel executable not found at $BUILD_DIR/channel"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j4"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

echo "Config: $CFG"
echo "Output: $OUT"
echo ""

cd "$BUILD_DIR"
./channel --config "$CFG" --output "$OUT" "${@:2}"

echo ""
echo "=========================================================="
echo "  Simulation complete!"
echo "=========================================================="
echo ""
echo "Results saved to: $OUT"
echo ""
echo "To compare all models:"
echo "  python $EXAMPLE_DIR/compare_models.py"
echo ""
echo "To visualize:"
echo "  paraview $OUT/velocity_final.vtk"
echo ""
