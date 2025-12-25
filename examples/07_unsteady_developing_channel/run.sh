#!/bin/bash
# Run unsteady developing channel simulation

set -e

echo "======================================"
echo "  Unsteady Developing Channel"
echo "======================================"
echo ""
echo "Time-accurate laminar simulation"
echo "Initial condition: Divergence-free perturbation"
echo "Turbulence model: None"
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

# Create output directory
mkdir -p output

# Run simulation
echo "======================================"
echo "Running unsteady simulation..."
echo "======================================"
$BUILD_DIR/channel --config laminar.cfg

echo ""
echo "======================================"
echo "  Simulation complete!"
echo "======================================"
echo ""
echo "Results saved to: output/"
echo ""
echo "To visualize time evolution:"
echo "  paraview output/developing_channel_*.vtk"
echo ""
echo "To run high-resolution version:"
echo "  $BUILD_DIR/channel --config laminar_fine.cfg"
echo ""


