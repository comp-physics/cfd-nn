#!/bin/bash
#
# GPU Utilization Profiling Script
#
# This script profiles the solver to measure GPU vs CPU utilization.
# It can use either:
#   1. Built-in timing (fast, for CI)
#   2. NVIDIA Nsight Systems (detailed, for investigation)
#
# Usage:
#   ./scripts/profile_gpu_utilization.sh [--nsys] [--threshold N]
#
# Options:
#   --nsys      Use Nsight Systems for detailed profiling (default: built-in timing)
#   --threshold Set GPU utilization threshold percentage (default: 70)
#   --verbose   Show detailed output
#   --help      Show this help message
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build_gpu"

# Default options
USE_NSYS=false
THRESHOLD=70
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nsys)
            USE_NSYS=true
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================"
echo "  GPU Utilization Profiler"
echo "================================================================"
echo ""

# Check for GPU build
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: GPU build directory not found: $BUILD_DIR"
    echo "Run: cmake -B build_gpu -DUSE_GPU_OFFLOAD=ON && cmake --build build_gpu"
    exit 1
fi

# Check for test binary
TEST_BIN="${BUILD_DIR}/test_gpu_utilization"
if [[ ! -x "$TEST_BIN" ]]; then
    echo "ERROR: test_gpu_utilization not found. Building..."
    cd "$PROJECT_DIR"
    cmake --build build_gpu --target test_gpu_utilization
fi

cd "$BUILD_DIR"

if $USE_NSYS; then
    echo "Mode: NVIDIA Nsight Systems (detailed profiling)"
    echo ""

    # Check for nsys
    if ! command -v nsys &> /dev/null; then
        echo "ERROR: nsys not found. Load nvhpc module: module load nvhpc"
        exit 1
    fi

    PROFILE_OUT="gpu_utilization_profile"

    echo "Running nsys profiler..."
    echo ""

    # Run nsys with stats output
    nsys profile \
        --stats=true \
        --force-overwrite=true \
        --output="${PROFILE_OUT}" \
        --capture-range=cudaProfilerApi \
        ./test_gpu_utilization "$THRESHOLD" 2>&1 | tee nsys_output.txt

    echo ""
    echo "================================================================"
    echo "  Nsight Systems Analysis"
    echo "================================================================"

    # Parse nsys output for key metrics
    if [[ -f nsys_output.txt ]]; then
        echo ""
        echo "Key Metrics from Nsight Systems:"
        echo "---------------------------------"

        # Look for CUDA kernel time
        grep -E "cuda|GPU|kernel|Time" nsys_output.txt 2>/dev/null | head -20 || true

        echo ""
        echo "Profile saved to: ${PROFILE_OUT}.nsys-rep"
        echo "Open in Nsight Systems GUI for detailed analysis:"
        echo "  nsys-ui ${PROFILE_OUT}.nsys-rep"
    fi

    # Additional detailed stats
    if [[ -f "${PROFILE_OUT}.sqlite" ]] || [[ -f "${PROFILE_OUT}.nsys-rep" ]]; then
        echo ""
        echo "Generating detailed GPU/CPU breakdown..."

        # Export stats if possible
        nsys stats "${PROFILE_OUT}.nsys-rep" 2>/dev/null | head -50 || true
    fi

else
    echo "Mode: Built-in timing (fast, for CI)"
    echo "Threshold: ${THRESHOLD}% GPU utilization"
    echo ""

    # Run the test with threshold
    threshold_decimal=$(echo "scale=2; $THRESHOLD/100" | bc)
    export GPU_UTIL_THRESHOLD="$threshold_decimal"

    # Run test and capture exit code
    # In non-verbose mode, we still show all output (test already has reasonable output)
    EXIT_CODE=0
    ./test_gpu_utilization 2>&1 || EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "[PASS] GPU utilization check passed"
    else
        echo ""
        echo "[FAIL] GPU utilization check failed"
        echo ""
        echo "To investigate, run with --nsys for detailed profiling:"
        echo "  $0 --nsys"
    fi

    exit $EXIT_CODE
fi

echo ""
echo "================================================================"
echo "  Profiling Complete"
echo "================================================================"
