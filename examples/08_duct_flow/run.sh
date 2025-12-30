#!/bin/bash
#
# Run 3D square duct flow simulation
#
# Usage: ./run.sh <config>
#   ./run.sh laminar_square      (default, coarse grid laminar)
#   ./run.sh laminar_fine        (fine grid laminar)
#   ./run.sh turbulent_sst       (turbulent with SST model)
#
# Or run directly:
#   ./duct --config laminar_square.cfg

set -euo pipefail

CASE="${1:-laminar_square}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

echo "=============================================="
echo "3D Square Duct Flow"
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
if [[ ! -x "$BUILD_DIR/duct" ]]; then
    echo "ERROR: Solver not found at $BUILD_DIR/duct"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DUSE_GPU_OFFLOAD=ON && make duct"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

echo "Config: $CFG"
echo "Output: $OUT"
echo ""

cd "$BUILD_DIR"
./duct --config "$CFG" --output "$OUT" "${@:2}"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUT"
echo ""
echo "View VTK files in ParaView:"
echo "  paraview $OUT/duct_final.vtk"
echo ""
