#!/bin/bash
# Pre-CI validation script
# Run this before pushing to ensure CI will pass

set -e

echo "==================================================================="
echo "  Pre-CI Testing - Validates Both Debug and Release Builds"
echo "==================================================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall success
TESTS_PASSED=true

cleanup_on_exit() {
    if [ "$TESTS_PASSED" = false ]; then
        echo -e "${RED}[FAIL] Pre-CI tests FAILED!${NC}"
        echo "Fix the issues above before pushing to avoid CI failures."
        exit 1
    fi
}
trap cleanup_on_exit EXIT

# Function to test a build type
test_build_type() {
    local BUILD_TYPE=$1
    local BUILD_DIR="build_${BUILD_TYPE,,}"
    
    echo ""
    echo "==================================================================="
    echo -e "  Testing ${YELLOW}${BUILD_TYPE}${NC} Build"
    echo "==================================================================="
    
    # Clean build directory
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure
    echo "--- Configuring CMake ---"
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > cmake_output.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] CMake configuration failed!${NC}"
        cat cmake_output.log
        TESTS_PASSED=false
        return 1
    fi
    
    # Build
    echo "--- Building ($BUILD_TYPE) ---"
    make -j4 2>&1 | tee build.log
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] Build failed!${NC}"
        TESTS_PASSED=false
        return 1
    fi
    
    # Check for warnings (Release only to match CI)
    if [ "$BUILD_TYPE" = "Release" ]; then
        echo "--- Checking for compiler warnings ---"
        if grep -i "warning:" build.log; then
            echo -e "${RED}[FAIL] Compiler warnings detected!${NC}"
            echo "Fix all warnings before pushing."
            TESTS_PASSED=false
            return 1
        else
            echo -e "${GREEN}[OK] No warnings${NC}"
        fi
    fi
    
    # Run tests
    echo "--- Running tests ($BUILD_TYPE) ---"
    ctest --output-on-failure
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] Tests failed in $BUILD_TYPE build!${NC}"
        TESTS_PASSED=false
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}[PASS] $BUILD_TYPE build passed all tests${NC}"
}

# Test both build types (matching CI)
test_build_type "Release"
test_build_type "Debug"

# Summary
echo ""
echo "==================================================================="
echo -e "${GREEN}[PASS] All Pre-CI Tests PASSED!${NC}"
echo "==================================================================="
echo ""
echo "Both Debug and Release builds succeeded."
echo "All tests passed in both configurations."
echo "No compiler warnings detected."
echo ""
echo "You are safe to push to the repository!"
echo ""

# Cleanup build directories (optional)
read -p "Clean up build directories? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf build_release build_debug
    echo "Build directories cleaned."
fi

exit 0

