#!/bin/bash
#
# Run 3D Taylor-Green vortex simulation
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=============================================="
echo "3D Taylor-Green Vortex"
echo "=============================================="
echo ""

# Check if solver is built
if [ ! -f "$BUILD_DIR/taylor_green_3d" ]; then
    echo "ERROR: Solver not found at $BUILD_DIR/taylor_green_3d"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DUSE_GPU_OFFLOAD=ON && make taylor_green_3d"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Default parameters (can be overridden via command line)
N=${N:-32}
RE=${RE:-100}
T_FINAL=${T_FINAL:-10.0}
DT=${DT:-0.01}
NUM_SNAPSHOTS=${NUM_SNAPSHOTS:-10}

echo "Configuration:"
echo "  Grid: ${N}³ cells"
echo "  Reynolds number: ${RE}"
echo "  Final time: ${T_FINAL}"
echo "  Time step: ${DT}"
echo "  Output: $OUTPUT_DIR/"
echo ""

cd "$BUILD_DIR"
./taylor_green_3d --N "$N" \
                  --Re "$RE" \
                  --T "$T_FINAL" \
                  --dt "$DT" \
                  --num_snapshots "$NUM_SNAPSHOTS" \
                  --output "$OUTPUT_DIR/"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Plot kinetic energy decay:"
echo "     python $EXAMPLE_DIR/plot_energy.py"
echo ""
echo "  2. View VTK files in ParaView:"
echo "     paraview $OUTPUT_DIR/tg3d_*.vtk"
echo ""

# Run analysis if Python is available
if command -v python3 &> /dev/null && [ -f "$EXAMPLE_DIR/plot_energy.py" ]; then
    echo "Running energy analysis..."
    python3 "$EXAMPLE_DIR/plot_energy.py"
fi
