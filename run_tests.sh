#!/bin/bash
# Run all unit tests

cd build

echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="

FAILED=0
PASSED=0

run_test() {
    TEST_NAME=$1
    TEST_BIN=$2
    
    echo ""
    echo "Running $TEST_NAME..."
    echo "------------------------------------------"
    
    if ./$TEST_BIN; then
        echo "RESULT: PASSED"
        ((PASSED++))
    else
        echo "RESULT: FAILED"
        ((FAILED++))
    fi
}

run_test "Mesh Tests" "test_mesh"
run_test "Poisson Tests" "test_poisson"
run_test "Solver Tests" "test_solver"
run_test "Features Tests" "test_features"
run_test "NN Core Tests" "test_nn_core"
run_test "Turbulence Tests" "test_turbulence"

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi

