#!/bin/bash
# Test script for model zoo functionality

echo "================================================================"
echo "Model Zoo Infrastructure Test"
echo "================================================================"
echo ""

cd build

echo "1. Testing --help shows preset option..."
./channel --help | grep -A1 "nn_preset" || { echo "ERROR: preset option not found"; exit 1; }
echo "   ✓ Preset option documented in help"
echo ""

echo "2. Testing preset loading with MLP model..."
./channel --model nn_mlp --nn_preset example_scalar_nut --max_iter 1 2>&1 | grep "NN preset:" || { echo "ERROR: preset not loaded"; exit 1; }
echo "   ✓ MLP preset loads correctly"
echo ""

echo "3. Testing preset loading with TBNN model..."
./channel --model nn_tbnn --nn_preset example_tbnn --max_iter 1 2>&1 | grep "NN preset:" || { echo "ERROR: preset not loaded"; exit 1; }
echo "   ✓ TBNN preset loads correctly"
echo ""

echo "4. Verifying model directory structure..."
test -f ../data/models/example_scalar_nut/metadata.json || { echo "ERROR: metadata missing"; exit 1; }
test -f ../data/models/example_scalar_nut/layer0_W.txt || { echo "ERROR: weights missing"; exit 1; }
test -f ../data/models/example_tbnn/metadata.json || { echo "ERROR: metadata missing"; exit 1; }
echo "   ✓ Model directories complete"
echo ""

echo "5. Testing NN loading diagnostic..."
./test_nn_simple 2>&1 | grep "All tests passed" || { echo "ERROR: NN test failed"; exit 1; }
echo "   ✓ NN loading works correctly"
echo ""

echo "================================================================"
echo "All Model Zoo Tests Passed!"
echo "================================================================"
echo ""
echo "Infrastructure is ready to use published models!"
echo ""
echo "Next steps:"
echo "  1. Find published model (e.g., Ling TBNN, Weatheritt GEP)"
echo "  2. Export weights using scripts/export_*.py"
echo "  3. Place in data/models/<name>/"
echo "  4. Run with: ./channel --model <type> --nn_preset <name>"
echo ""

