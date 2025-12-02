#!/bin/bash
# CI-friendly CPU vs GPU comparison test
# Lighter than full test_cpu_vs_gpu.sh for faster CI turnaround

set -e

echo "========================================"
echo "CPU vs GPU Consistency Test (CI)"
echo "Date: $(date)"
echo "========================================"

# Load modules
module reset
module load nvhpc

# Smaller test parameters for CI speed
NX=64
NY=128
NU=0.001
DP_DX=-0.0001
ITERS=200
TURB_MODEL="baseline"

echo ""
echo "Test Configuration:"
echo "  Grid: ${NX}x${NY}"
echo "  nu: ${NU}"
echo "  dp/dx: ${DP_DX}"
echo "  Iterations: ${ITERS}"
echo "  Turbulence model: ${TURB_MODEL}"
echo "========================================"

# Build CPU version
echo ""
echo "=== Building CPU version ==="
rm -rf build_cpu_ci
mkdir -p build_cpu_ci
cd build_cpu_ci
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF > cmake_cpu.log 2>&1
make channel -j4 > build_cpu.log 2>&1
echo "✓ CPU build complete"
cd ..

# Build GPU version
echo ""
echo "=== Building GPU version ==="
rm -rf build_gpu_ci
mkdir -p build_gpu_ci
cd build_gpu_ci
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON > cmake_gpu.log 2>&1
make channel test_cpu_gpu_consistency -j4 > build_gpu.log 2>&1
echo "✓ GPU build complete"
cd ..

# Run GPU unit tests first
echo ""
echo "=== Running GPU unit tests ==="
cd build_gpu_ci
if [ -f "test_cpu_gpu_consistency" ]; then
    ./test_cpu_gpu_consistency || {
        echo "✗ GPU unit tests FAILED"
        exit 1
    }
    echo "✓ GPU unit tests PASSED"
else
    echo "⚠ test_cpu_gpu_consistency not built, skipping"
fi
cd ..

# Run CPU solver
echo ""
echo "=== Running CPU solver ==="
mkdir -p output_cpu_ci
cd build_cpu_ci
./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model $TURB_MODEL --dp_dx $DP_DX \
          --output ../output_cpu_ci/cpu --num_snapshots 0 > ../output_cpu_ci/run.log 2>&1
cd ..
echo "✓ CPU solver complete"

# Run GPU solver
echo ""
echo "=== Running GPU solver ==="
mkdir -p output_gpu_ci
cd build_gpu_ci
./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
          --model $TURB_MODEL --dp_dx $DP_DX \
          --output ../output_gpu_ci/gpu --num_snapshots 0 > ../output_gpu_ci/run.log 2>&1
cd ..
echo "✓ GPU solver complete"

# Compare velocity fields
echo ""
echo "========================================"
echo "Comparing CPU vs GPU Results"
echo "========================================"

CPU_VEL_FILE="output_cpu_ci/cpuchannel_velocity.dat"
GPU_VEL_FILE="output_gpu_ci/gpuchannel_velocity.dat"

if [[ -f "$CPU_VEL_FILE" && -f "$GPU_VEL_FILE" ]]; then
    echo ""
    echo "Computing field norms..."
    
    # Compute L2 and Linf norms of velocity difference
    paste <(tail -n +2 "$CPU_VEL_FILE") <(tail -n +2 "$GPU_VEL_FILE") | \
    awk '
    BEGIN {
        max_u = 0; max_v = 0; max_mag = 0;
        sum_sq_u = 0; sum_sq_v = 0; sum_sq_mag = 0;
        n = 0;
    }
    {
        # Extract u, v from CPU (columns 3,4) and GPU (columns 8,9)
        # Adjust if your output format differs
        u_cpu = $3; v_cpu = $4;
        u_gpu = $8; v_gpu = $9;
        
        du = u_cpu - u_gpu;
        dv = v_cpu - v_gpu;
        dmag = sqrt(du*du + dv*dv);
        
        # Track maximums
        du_abs = (du < 0) ? -du : du;
        dv_abs = (dv < 0) ? -dv : dv;
        if (du_abs > max_u) max_u = du_abs;
        if (dv_abs > max_v) max_v = dv_abs;
        if (dmag > max_mag) max_mag = dmag;
        
        # Accumulate for RMS
        sum_sq_u += du*du;
        sum_sq_v += dv*dv;
        sum_sq_mag += dmag*dmag;
        n++;
    }
    END {
        if (n == 0) {
            print "ERROR: No data points found";
            exit 1;
        }
        
        l2_u = sqrt(sum_sq_u / n);
        l2_v = sqrt(sum_sq_v / n);
        l2_mag = sqrt(sum_sq_mag / n);
        
        printf "Velocity field comparison (N=%d points):\n", n;
        printf "  L∞(u_diff): %.6e\n", max_u;
        printf "  L∞(v_diff): %.6e\n", max_v;
        printf "  L∞(|vel|_diff): %.6e\n", max_mag;
        printf "  L2(u_diff): %.6e\n", l2_u;
        printf "  L2(v_diff): %.6e\n", l2_v;
        printf "  L2(|vel|_diff): %.6e\n", l2_mag;
        
        # Algorithm-based tolerances (not platform-based)
        tol_linf = 1e-7;
        tol_l2 = 1e-8;
        
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
            printf "\n✗ CPU/GPU MISMATCH DETECTED\n";
            exit 1;
        } else {
            printf "\n✓ CPU/GPU velocity fields match within tolerance\n";
        }
    }
    ' || {
        echo ""
        echo "========================================"
        echo "✗ TEST FAILED"
        echo "========================================"
        exit 1
    }
    
else
    echo "✗ ERROR: Could not find velocity output files"
    echo "  Looking for:"
    echo "    CPU: $CPU_VEL_FILE"
    echo "    GPU: $GPU_VEL_FILE"
    ls -l output_cpu_ci/ output_gpu_ci/ 2>/dev/null || true
    exit 1
fi

# Sample a few velocity values for manual inspection
echo ""
echo "Sample velocity values (first 5 interior points):"
echo ""
echo "CPU:"
head -6 "$CPU_VEL_FILE" | tail -5
echo ""
echo "GPU:"
head -6 "$GPU_VEL_FILE" | tail -5

echo ""
echo "========================================"
echo "✓ ALL TESTS PASSED"
echo "========================================"

