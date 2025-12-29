#!/bin/bash
#
# Run laminar channel flow (Poiseuille) validation case
#
# Usage: ./run.sh <config>
#   ./run.sh poiseuille    (default)
#
# Or run directly:
#   ./channel --config poiseuille.cfg

set -euo pipefail

CASE="${1:-poiseuille}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

echo "=============================================="
echo "Laminar Channel Flow - Poiseuille Validation"
echo "=============================================="
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
    echo "ERROR: Solver not found at $BUILD_DIR/channel"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

# Run the simulation
echo "Config: $CFG"
echo "Output: $OUT"
echo ""

cd "$BUILD_DIR"
./channel --config "$CFG" --output "$OUT" "${@:2}"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUT"
echo ""
echo "Next steps:"
echo "  1. Visualize results:"
echo "     python $EXAMPLE_DIR/analyze.py"
echo ""
echo "  2. View VTK files in ParaView:"
echo "     paraview $OUT/velocity_*.vtk"
echo ""

# Run analysis if Python is available
if command -v python3 &> /dev/null && [[ -f "$EXAMPLE_DIR/analyze.py" ]]; then
    echo "Running automated analysis..."
    python3 "$EXAMPLE_DIR/analyze.py" "$OUT"
fi
