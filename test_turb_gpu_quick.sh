#!/bin/bash
module load nvhpc/25.5

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_test_partial

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Testing Turbulence Models with Persistent Mapping     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

timeout 60 ./test_turbulence
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ TEST PASSED! ✓✓✓"
else
    echo "✗ FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
