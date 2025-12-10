#!/bin/bash
module load nvhpc/25.5

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_ci_gpu

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Testing turbulence test directly vs via ctest         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

echo "==== Test 1: Direct execution ===="
if timeout 60 ./test_turbulence; then
    echo "✓ Direct execution PASSED"
else
    echo "✗ Direct execution FAILED ($?)"
fi

echo ""
echo "==== Test 2: Via ctest ===="
if timeout 60 ctest -R TurbulenceTest --output-on-failure; then
    echo "✓ ctest PASSED"
else
    echo "✗ ctest FAILED ($?)"
fi

echo ""
echo "==== Test 3: Direct with NVCOMPILER_ACC_NOTIFY ===="
export NVCOMPILER_ACC_NOTIFY=3
if timeout 60 ./test_turbulence 2>&1 | tail -100; then
    echo "✓ Direct with verbose PASSED"
else
    echo "✗ Direct with verbose FAILED ($?)"
fi

