#!/bin/bash
# Rebuild script for CFD-NN project
# Usage: ./rebuild.sh [clean|debug|release]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default build type
BUILD_TYPE="Release"
BUILD_DIR="build"
CLEAN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        clean)
            CLEAN=true
            ;;
        debug)
            BUILD_TYPE="Debug"
            BUILD_DIR="build_debug"
            ;;
        release)
            BUILD_TYPE="Release"
            BUILD_DIR="build"
            ;;
        *)
            echo "Usage: $0 [clean|debug|release]"
            echo "  clean   - Clean build directory before building"
            echo "  debug   - Build in Debug mode (default: Release)"
            echo "  release - Build in Release mode"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== CFD-NN Rebuild Script ===${NC}"
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build
echo -e "${GREEN}Building (using all available cores)...${NC}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j"$NPROC"

# Success message
echo -e "${GREEN}=== Build completed successfully! ===${NC}"
echo ""
echo "Executables built:"
ls -lh channel periodic_hills test_* 2>/dev/null | grep -v ".o$" || true
echo ""
echo "To run tests: cd $BUILD_DIR && ctest --output-on-failure"
echo "To run channel: cd $BUILD_DIR && ./channel --help"


