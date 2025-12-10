#!/bin/bash
module load nvhpc/25.5

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn
rm -rf build_sst_disabled
mkdir -p build_sst_disabled
cd build_sst_disabled

CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON
make test_turbulence -j4

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Testing with SST GPU path disabled                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

for i in {1..5}; do
    echo "==== Run #$i ===="
    if timeout 60 ./test_turbulence; then
        echo "✓ Run #$i PASSED"
    else
        echo "✗ Run #$i FAILED ($?)"
    fi
done

echo ""
echo "==== Via ctest ===="
if ctest -R TurbulenceTest --output-on-failure; then
    echo "✓ ctest PASSED"
else
    echo "✗ ctest FAILED"
fi

