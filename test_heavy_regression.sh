#!/bin/bash
# Heavy regression test for pre-push validation
# Larger grid, more iterations, comprehensive checks

set -e

echo "========================================"
echo "Heavy Regression Test (Pre-Push)"
echo "Date: $(date)"
echo "========================================"

# Load modules
module reset
module load nvhpc

# Heavy test parameters (realistic production case)
NX=256
NY=512
NU=0.0005
DP_DX=-0.0001
ITERS=1000
MODEL="baseline"

# Tolerances
TOL_LINF=1e-7
TOL_L2=1e-8

echo ""
echo "Configuration:"
echo "  Grid: ${NX}x${NY}"
echo "  Reynolds number: ~$(echo "scale=0; 1/$NU" | bc)"
echo "  Iterations: $ITERS"
echo "  Model: $MODEL"
echo "  Tolerances: L∞ < $TOL_LINF, L2 < $TOL_L2"
echo "========================================"

# Check if builds exist, otherwise build
if [ ! -d "build_heavy_cpu" ] || [ ! -f "build_heavy_cpu/channel" ]; then
    echo ""
    echo "=== Building CPU version ==="
    rm -rf build_heavy_cpu
    mkdir -p build_heavy_cpu
    cd build_heavy_cpu
    CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF > cmake.log 2>&1
    make channel -j4 > build.log 2>&1
    echo "✓ CPU build complete"
    cd ..
fi

if [ ! -d "build_heavy_gpu" ] || [ ! -f "build_heavy_gpu/channel" ]; then
    echo ""
    echo "=== Building GPU version ==="
    rm -rf build_heavy_gpu
    mkdir -p build_heavy_gpu
    cd build_heavy_gpu
    CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON > cmake.log 2>&1
    make channel -j4 > build.log 2>&1
    echo "✓ GPU build complete"
    cd ..
fi

# Create output directories
mkdir -p output_heavy_cpu
mkdir -p output_heavy_gpu

# Run CPU
echo ""
echo "=== Running CPU solver (this may take a few minutes) ==="
cd build_heavy_cpu
time ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model $MODEL --dp_dx $DP_DX \
          --output ../output_heavy_cpu/result \
          --num_snapshots 0 > ../output_heavy_cpu/run.log 2>&1
cd ..
echo "✓ CPU solver complete"

# Run GPU
echo ""
echo "=== Running GPU solver ==="
cd build_heavy_gpu
time ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model $MODEL --dp_dx $DP_DX \
          --output ../output_heavy_gpu/result \
          --num_snapshots 0 > ../output_heavy_gpu/run.log 2>&1
cd ..
echo "✓ GPU solver complete"

# Compare results
echo ""
echo "========================================"
echo "Comparing Results"
echo "========================================"

CPU_FILE="output_heavy_cpu/resultchannel_velocity.dat"
GPU_FILE="output_heavy_gpu/resultchannel_velocity.dat"

if [[ ! -f "$CPU_FILE" || ! -f "$GPU_FILE" ]]; then
    echo "✗ ERROR: Output files not found"
    echo "  CPU file: $CPU_FILE"
    echo "  GPU file: $GPU_FILE"
    ls -lh output_heavy_cpu/ output_heavy_gpu/
    exit 1
fi

echo ""
echo "Computing field norms..."

paste <(tail -n +2 "$CPU_FILE") <(tail -n +2 "$GPU_FILE") | \
awk -v tol_linf=$TOL_LINF -v tol_l2=$TOL_L2 '
BEGIN {
    max_u = 0; max_v = 0; max_mag = 0;
    sum_sq_u = 0; sum_sq_v = 0; sum_sq_mag = 0;
    n = 0;
    
    # Track some sample points for detailed output
    sample_count = 0;
    max_sample = 5;
}
{
    u_cpu = $3; v_cpu = $4;
    u_gpu = $8; v_gpu = $9;
    
    du = u_cpu - u_gpu;
    dv = v_cpu - v_gpu;
    dmag = sqrt(du*du + dv*dv);
    
    du_abs = (du < 0) ? -du : du;
    dv_abs = (dv < 0) ? -dv : dv;
    
    if (du_abs > max_u) max_u = du_abs;
    if (dv_abs > max_v) max_v = dv_abs;
    if (dmag > max_mag) max_mag = dmag;
    
    sum_sq_u += du*du;
    sum_sq_v += dv*dv;
    sum_sq_mag += dmag*dmag;
    n++;
    
    # Store some large-difference samples
    if (dmag > 1e-10 && sample_count < max_sample) {
        samples[sample_count++] = sprintf("  Point %d: u_cpu=%.6e, u_gpu=%.6e, diff=%.3e", n, u_cpu, u_gpu, dmag);
    }
}
END {
    if (n == 0) {
        print "ERROR: No data points found";
        exit 1;
    }
    
    l2_u = sqrt(sum_sq_u / n);
    l2_v = sqrt(sum_sq_v / n);
    l2_mag = sqrt(sum_sq_mag / n);
    
    printf "\nVelocity field comparison (N=%d points):\n", n;
    printf "  L∞(u_diff):      %.6e\n", max_u;
    printf "  L∞(v_diff):      %.6e\n", max_v;
    printf "  L∞(|vel|_diff):  %.6e\n", max_mag;
    printf "  L2(u_diff):      %.6e\n", l2_u;
    printf "  L2(v_diff):      %.6e\n", l2_v;
    printf "  L2(|vel|_diff):  %.6e\n", l2_mag;
    
    if (sample_count > 0) {
        printf "\nSample points with largest differences:\n";
        for (i = 0; i < sample_count; i++) {
            print samples[i];
        }
    }
    
    printf "\nTolerance check:\n";
    printf "  L∞ tolerance: %.1e\n", tol_linf;
    printf "  L2 tolerance: %.1e\n", tol_l2;
    
    failed = 0;
    if (max_mag > tol_linf) {
        printf "  ✗ L∞ FAILED: %.6e > %.6e\n", max_mag, tol_linf;
        failed = 1;
    } else {
        printf "  ✓ L∞ PASSED: %.6e <= %.6e\n", max_mag, tol_linf;
    }
    
    if (l2_mag > tol_l2) {
        printf "  ✗ L2 FAILED: %.6e > %.6e\n", l2_mag, tol_l2;
        failed = 1;
    } else {
        printf "  ✓ L2 PASSED: %.6e <= %.6e\n", l2_mag, tol_l2;
    }
    
    if (failed) {
        printf "\n✗ HEAVY REGRESSION TEST FAILED\n";
        exit 1;
    } else {
        printf "\n✓ CPU/GPU results match within tolerance\n";
    }
}
' || {
    echo ""
    echo "========================================"
    echo "✗ TEST FAILED"
    echo "========================================"
    exit 1
}

echo ""
echo "========================================"
echo "✓ HEAVY REGRESSION TEST PASSED"
echo "========================================"
echo ""
echo "Summary:"
echo "  - Grid: ${NX}x${NY}"
echo "  - Iterations: $ITERS"
echo "  - CPU and GPU results agree within tolerance"
echo "  - Safe to push!"

