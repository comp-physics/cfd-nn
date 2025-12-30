#!/bin/bash
#
# Run steady RANS channel simulation
#
# Usage: ./run.sh <config>
#   ./run.sh baseline   (mixing length model)
#   ./run.sh gep        (GEP algebraic model)
#   ./run.sh sst        (SST k-omega transport model)
#
# Or run directly:
#   ./channel --config baseline.cfg

set -euo pipefail

CASE="${1:-baseline}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

echo "======================================"
echo "  Steady RANS Channel Flow"
echo "======================================"
echo ""

# List available configs if none found
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config not found: $CFG"
    echo ""
    echo "Available configs:"
    ls -1 "$EXAMPLE_DIR"/*.cfg 2>/dev/null | xargs -n1 basename | sed 's/\.cfg$//'
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
echo "======================================"
echo "  Simulation complete!"
echo "======================================"
echo ""
echo "Results saved to: $OUT"
echo ""
echo "To visualize:"
echo "  paraview $OUT/channel_final.vtk"
echo ""
