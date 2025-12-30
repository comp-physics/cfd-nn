#!/bin/bash
# regenerate_golden.sh - Regenerate all golden reference files
#
# Run this when intentionally changing numerics (e.g., fixing a bug)
# After regeneration, review changes with: git diff tests/golden/reference/
#
# Usage:
#   ./tests/golden/regenerate_golden.sh              # Use default build dir
#   ./tests/golden/regenerate_golden.sh build_gpu    # Specify build dir

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../.."

# Find build directory
BUILD_DIR="${1:-}"
if [ -z "$BUILD_DIR" ]; then
    if [ -f "${PROJECT_DIR}/build_gpu/test_golden" ]; then
        BUILD_DIR="${PROJECT_DIR}/build_gpu"
    elif [ -f "${PROJECT_DIR}/build_cpu/test_golden" ]; then
        BUILD_DIR="${PROJECT_DIR}/build_cpu"
    elif [ -f "${PROJECT_DIR}/build/test_golden" ]; then
        BUILD_DIR="${PROJECT_DIR}/build"
    else
        echo "Error: Cannot find test_golden binary. Please specify build directory."
        echo "Usage: $0 <build_dir>"
        exit 1
    fi
fi

TEST_GOLDEN="${BUILD_DIR}/test_golden"
if [ ! -f "$TEST_GOLDEN" ]; then
    echo "Error: test_golden not found at $TEST_GOLDEN"
    echo "Please build with: cmake --build $BUILD_DIR --target test_golden"
    exit 1
fi

echo "Using build directory: $BUILD_DIR"
echo "Regenerating golden reference files..."
echo ""

# List of all golden test cases
CASES="channel_komega channel_earsm channel_mlp channel_tbnn mixing_length laminar_3d"

PASSED=0
SKIPPED=0
FAILED=0

for case in $CASES; do
    echo "----------------------------------------"
    echo "Regenerating: $case"
    if "$TEST_GOLDEN" "$case" --regenerate; then
        PASSED=$((PASSED + 1))
    else
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            SKIPPED=$((SKIPPED + 1))
        else
            echo "[FAIL] $case failed to regenerate"
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo ""
echo "========================================"
echo "REGENERATION COMPLETE"
echo "========================================"
echo "Passed:  $PASSED"
echo "Skipped: $SKIPPED"
echo "Failed:  $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "Review changes with:"
    echo "  git diff tests/golden/reference/"
    echo ""
    echo "If changes are intentional, commit them:"
    echo "  git add tests/golden/reference/"
    echo "  git commit -m 'Regenerate golden files after <reason>'"
    exit 0
else
    echo "Some cases failed to regenerate. Check errors above."
    exit 1
fi
