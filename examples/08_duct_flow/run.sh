#!/bin/bash
#
# Run 3D square duct flow simulation
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=============================================="
echo "3D Square Duct Flow"
echo "=============================================="
echo ""

# Check if solver is built
if [ ! -f "$BUILD_DIR/duct" ]; then
    echo "ERROR: Solver not found at $BUILD_DIR/duct"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DUSE_GPU_OFFLOAD=ON && make duct"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Default parameters (can be overridden via command line)
NX=${NX:-16}
NY=${NY:-32}
NZ=${NZ:-32}
MAX_ITER=${MAX_ITER:-10000}
NU=${NU:-0.01}

echo "Configuration:"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Max iterations: ${MAX_ITER}"
echo "  Viscosity: ${NU}"
echo "  Output: $OUTPUT_DIR/"
echo ""

cd "$BUILD_DIR"
./duct --Nx "$NX" --Ny "$NY" --Nz "$NZ" \
       --max_iter "$MAX_ITER" \
       --nu "$NU" \
       --output "$OUTPUT_DIR/"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  View VTK files in ParaView:"
echo "    paraview $OUTPUT_DIR/duct_final.vtk"
echo ""
