#!/bin/bash
#
# Run 3D Taylor-Green vortex simulation
#
# Usage: ./run.sh <config> [options]
#   ./run.sh tg_re100          (default, Re=100 on 32^3)
#   ./run.sh tg_re100_fine     (Re=100 on 64^3)
#   ./run.sh tg_re1600         (Re=1600 DNS on 64^3)
#
# Override parameters from command line:
#   ./run.sh tg_re100 --Re 200 --T 20.0
#
# Or run directly:
#   ./taylor_green_3d --config tg_re100.cfg

set -euo pipefail

CASE="${1:-tg_re100}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

echo "=============================================="
echo "3D Taylor-Green Vortex"
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
if [[ ! -x "$BUILD_DIR/taylor_green_3d" ]]; then
    echo "ERROR: Solver not found at $BUILD_DIR/taylor_green_3d"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DUSE_GPU_OFFLOAD=ON && make taylor_green_3d"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

echo "Config: $CFG"
echo "Output: $OUT"
echo ""

cd "$BUILD_DIR"
./taylor_green_3d --config "$CFG" --output "$OUT" "${@:2}"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUT"
echo ""
echo "Next steps:"
echo "  1. Plot kinetic energy decay:"
echo "     python $EXAMPLE_DIR/plot_energy.py"
echo ""
echo "  2. View VTK files in ParaView:"
echo "     paraview $OUT/tg3d_*.vtk"
echo ""

# Run analysis if Python is available
if command -v python3 &> /dev/null && [[ -f "$EXAMPLE_DIR/plot_energy.py" ]]; then
    echo "Running energy analysis..."
    python3 "$EXAMPLE_DIR/plot_energy.py" "$OUT"
fi
