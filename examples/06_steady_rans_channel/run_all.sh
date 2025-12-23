#!/bin/bash
# Run all three RANS models and compare results

set -e

echo "======================================"
echo "  Steady RANS Channel Comparison"
echo "======================================"
echo ""
echo "Running three turbulence models:"
echo "  1. Baseline (mixing length)"
echo "  2. GEP (algebraic)"
echo "  3. SST k-omega (transport)"
echo ""

# Build directory
BUILD_DIR="../../build"

if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "Error: channel executable not found in $BUILD_DIR"
    echo "Please build the project first:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j4"
    exit 1
fi

# Create output directories
mkdir -p output/baseline output/gep output/sst

# Run baseline model
echo "======================================"
echo "Running Baseline Model..."
echo "======================================"
$BUILD_DIR/channel --config baseline.cfg

# Run GEP model
echo ""
echo "======================================"
echo "Running GEP Model..."
echo "======================================"
$BUILD_DIR/channel --config gep.cfg

# Run SST model
echo ""
echo "======================================"
echo "Running SST k-omega Model..."
echo "======================================"
$BUILD_DIR/channel --config sst.cfg

echo ""
echo "======================================"
echo "  All simulations complete!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  output/baseline/"
echo "  output/gep/"
echo "  output/sst/"
echo ""
echo "To visualize:"
echo "  paraview output/*/channel_final.vtk"
echo ""


