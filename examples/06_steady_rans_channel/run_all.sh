#!/bin/bash
#
# Run all three RANS models and compare results
#
# Usage: ./run_all.sh
#
# Or run individual models:
#   ./run.sh baseline
#   ./run.sh gep
#   ./run.sh sst

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "======================================"
echo "  Steady RANS Channel Comparison"
echo "======================================"
echo ""
echo "Running three turbulence models:"
echo "  1. Baseline (mixing length)"
echo "  2. GEP (algebraic)"
echo "  3. SST k-omega (transport)"
echo ""

# Check if solver is built
if [[ ! -x "$BUILD_DIR/channel" ]]; then
    echo "ERROR: channel executable not found at $BUILD_DIR/channel"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j4"
    exit 1
fi

# Create output directories
mkdir -p "$EXAMPLE_DIR/output/baseline" "$EXAMPLE_DIR/output/gep" "$EXAMPLE_DIR/output/sst"

cd "$BUILD_DIR"

# Run baseline model
echo "======================================"
echo "Running Baseline Model..."
echo "======================================"
./channel --config "$EXAMPLE_DIR/baseline.cfg" --output "$EXAMPLE_DIR/output/baseline/"

# Run GEP model
echo ""
echo "======================================"
echo "Running GEP Model..."
echo "======================================"
./channel --config "$EXAMPLE_DIR/gep.cfg" --output "$EXAMPLE_DIR/output/gep/"

# Run SST model
echo ""
echo "======================================"
echo "Running SST k-omega Model..."
echo "======================================"
./channel --config "$EXAMPLE_DIR/sst.cfg" --output "$EXAMPLE_DIR/output/sst/"

echo ""
echo "======================================"
echo "  All simulations complete!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  $EXAMPLE_DIR/output/baseline/"
echo "  $EXAMPLE_DIR/output/gep/"
echo "  $EXAMPLE_DIR/output/sst/"
echo ""
echo "To visualize:"
echo "  paraview $EXAMPLE_DIR/output/*/channel_final.vtk"
echo ""
