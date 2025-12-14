#!/bin/bash
#
# Spatial Convergence Study: Fix dt, vary h
# Expected: 2nd-order convergence (p ≈ 2.0)
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output_spatial"

echo "=========================================================="
echo "SPATIAL Convergence Study: Fixed dt, Varying Grid"
echo "=========================================================="
echo ""
echo "Fixed: dt = 0.00001 (very small)"
echo "Varying: Grid spacing h"
echo ""
echo "  1. Coarse:  32 x 64   (h ≈ 0.031)"
echo "  2. Medium:  64 x 128  (h ≈ 0.016)"
echo "  3. Fine:   128 x 256  (h ≈ 0.008)"
echo ""
echo "Expected: 2nd-order spatial convergence (p ≈ 2.0)"
echo "  → Errors should decrease by ~4× per refinement"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"/{coarse,medium,fine}

# Run each grid
grids=(
    "coarse:coarse_dt_fixed.cfg"
    "medium:medium_dt_fixed.cfg"
    "fine:fine_dt_fixed.cfg"
)

cd "$BUILD_DIR"

for grid_config in "${grids[@]}"; do
    IFS=':' read -r grid cfg <<< "$grid_config"
    
    echo "=========================================="
    echo "Running: $grid"
    echo "=========================================="
    
    ./channel --config "$EXAMPLE_DIR/spatial_convergence_configs/$cfg" \
              --output "$OUTPUT_DIR/$grid/"
    
    echo "[OK] $grid complete"
    echo ""
done

echo "=========================================================="
echo "Spatial convergence study complete!"
echo "=========================================================="
echo ""

# Run analysis
cd "$EXAMPLE_DIR"
python3 << 'PYEOF'
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
output_dir = script_dir / "output_spatial"

# Import the existing analysis functions
exec(open('convergence_analysis.py').read().replace(
    'output_dir = script_dir / "output"',
    'output_dir = script_dir / "output_spatial"'
).replace(
    'output_dir / "convergence_analysis.png"',
    'output_dir / "spatial_convergence.png"'
).replace(
    'very_fine',
    ''
).replace(
    "grids = {",
    "grids = {"
).replace(
    "        'very_fine': (256, 512)",
    ""
))
PYEOF

echo ""
echo "Analysis plot saved to: output_spatial/spatial_convergence.png"












