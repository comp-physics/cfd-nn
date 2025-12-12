#!/bin/bash
#
# GPU Grid Convergence Study - All 4 grids with extended iterations
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_ci_gpu"
OUTPUT_DIR="$EXAMPLE_DIR/output_gpu"

echo "=========================================================="
echo "GPU Grid Convergence Study - Full 4-Grid Analysis"
echo "=========================================================="
echo ""
echo "Running laminar Poiseuille flow on 4 grid resolutions:"
echo "  1. Coarse:     32 x 64     (max_iter:  50k)"
echo "  2. Medium:     64 x 128    (max_iter: 100k)"
echo "  3. Fine:      128 x 256    (max_iter: 300k)"
echo "  4. Very Fine: 256 x 512    (max_iter: 600k)"
echo ""
echo "Using GPU offload for faster convergence."
echo ""

# Check if GPU solver exists
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: GPU solver not found at $BUILD_DIR/channel"
    echo "Building GPU version..."
    cd "$PROJECT_ROOT"
    mkdir -p build_ci_gpu
    cd build_ci_gpu
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON
    make -j4 channel
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{coarse,medium,fine,very_fine}

# Grid configurations
declare -A grid_iters
grid_iters[coarse]=50000
grid_iters[medium]=100000
grid_iters[fine]=300000
grid_iters[very_fine]=600000

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
    echo "Running: $grid grid (max_iter=${grid_iters[$grid]})"
    echo "=========================================="
    echo "Start time: $(date)"
    
    START_TIME=$(date +%s)
    
    ./channel --config "$EXAMPLE_DIR/$cfg" \
              --output "$OUTPUT_DIR/$grid/" \
              --max_iter ${grid_iters[$grid]}
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo "✓ $grid complete in ${ELAPSED}s"
    echo ""
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
    cd "$EXAMPLE_DIR"
    
    # Create temporary analysis script for GPU output
    cat > analyze_gpu_results.py << 'PYEOF'
import sys
sys.path.insert(0, '.')
exec(open('convergence_analysis.py').read().replace(
    'output_dir = script_dir / "output"',
    'output_dir = script_dir / "output_gpu"'
).replace(
    'output_dir / "convergence_analysis.png"',
    'output_dir / "convergence_analysis_gpu.png"'
))
PYEOF
    
    python3 analyze_gpu_results.py
    rm -f analyze_gpu_results.py
else
    echo "Python3 not found - skipping automated analysis"
    echo "Run manually: python3 convergence_analysis.py"
fi

echo ""
echo "=========================================================="
echo "GPU Convergence Study Complete!"
echo "=========================================================="











