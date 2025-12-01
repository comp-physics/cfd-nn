#!/bin/bash
#
# Run laminar channel flow (Poiseuille) validation case
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=============================================="
echo "Laminar Channel Flow - Poiseuille Validation"
echo "=============================================="
echo ""

# Check if solver is built
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: Solver not found at $BUILD_DIR/channel"
    echo "Please build the project first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the simulation
echo "Running simulation..."
echo "Config: poiseuille.cfg"
echo "Output: $OUTPUT_DIR/"
echo ""

cd "$BUILD_DIR"
./channel --config "$EXAMPLE_DIR/poiseuille.cfg" \
          --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Simulation complete!"
echo "=============================================="
echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Visualize results:"
echo "     python $EXAMPLE_DIR/analyze.py"
echo ""
echo "  2. View VTK files in ParaView:"
echo "     paraview $OUTPUT_DIR/velocity_*.vtk"
echo ""

# Run analysis if Python is available
if command -v python3 &> /dev/null; then
    echo "Running automated analysis..."
    python3 "$EXAMPLE_DIR/analyze.py"
fi

