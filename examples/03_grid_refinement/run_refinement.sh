#!/bin/bash
#
# Run grid refinement study for spatial convergence analysis
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=========================================================="
echo "Grid Refinement Study - Spatial Convergence Analysis"
echo "=========================================================="
echo ""
echo "Running laminar Poiseuille flow on 4 grid resolutions:"
echo "  1. Coarse:     32 x 64"
echo "  2. Medium:     64 x 128"
echo "  3. Fine:      128 x 256"
echo "  4. Very Fine: 256 x 512"
echo ""
echo "This will quantify numerical accuracy and convergence order."
echo ""

# Check if solver exists
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: Solver not found"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{coarse,medium,fine,very_fine}

# Array of grid levels
grids=(
    "coarse:coarse_32x64.cfg"
    "medium:medium_64x128.cfg"
    "fine:fine_128x256.cfg"
    "very_fine:very_fine_256x512.cfg"
)

cd "$BUILD_DIR"

# Run each grid level
for grid_config in "${grids[@]}"; do
    IFS=':' read -r grid cfg <<< "$grid_config"
    
    echo ""
    echo "=========================================="
    echo "Running: $grid grid"
    echo "=========================================="
    
    ./channel --config "$EXAMPLE_DIR/$cfg" \
              --output_dir "$OUTPUT_DIR/$grid"
    
    echo "✓ $grid complete"
done

echo ""
echo "=========================================================="
echo "All grid levels complete!"
echo "=========================================================="
echo ""
echo "Output saved to: $OUTPUT_DIR/"
echo ""

# Run convergence analysis
if command -v python3 &> /dev/null; then
    echo "Running convergence analysis..."
    python3 "$EXAMPLE_DIR/convergence_analysis.py"
else
    echo "Python3 not found - skipping automated analysis"
    echo "Run manually: python3 convergence_analysis.py"
fi

