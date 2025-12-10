#!/bin/bash
module load nvhpc/25.5

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn
rm -rf build_sst_release
mkdir -p build_sst_release
cd build_sst_release

CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON
make test_turbulence -j4

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Testing with map(release:) instead of map(delete:)    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

for i in {1..3}; do
    echo "==== Run #$i ===="
    if timeout 60 ./test_turbulence; then
        echo "✓ Run #$i PASSED"
    else
        echo "✗ Run #$i FAILED ($?)"
    fi
    echo ""
done

