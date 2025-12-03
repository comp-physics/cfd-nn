#!/bin/bash
# Systematic parameter sweep: CPU vs GPU consistency
# Tests multiple grids, Reynolds numbers, and turbulence models

set -e

echo "========================================"
echo "CPU vs GPU Parameter Sweep Test"
echo "Date: $(date)"
echo "========================================"

# Load modules
module reset
module load nvhpc

# Test configurations (grid, nu, model)
# Format: "Nx Ny nu model_name"
declare -a TEST_CASES=(
    "64 128 0.001 baseline"
    "128 256 0.0005 baseline"
    "63 129 0.001 baseline"      # Odd grid
    "64 128 0.0001 baseline"     # Higher Re
    "64 128 0.001 none"          # Laminar
)

ITERS=300
DP_DX=-0.0001

# Tolerances (algorithm-based)
TOL_LINF=1e-7
TOL_L2=1e-8

echo ""
echo "Test matrix:"
echo "  Cases: ${#TEST_CASES[@]}"
echo "  Iterations per case: $ITERS"
echo "  Tolerances: L∞ < $TOL_LINF, L2 < $TOL_L2"
echo "========================================"

# Build once (both CPU and GPU)
echo ""
echo "=== Building CPU version ==="
rm -rf build_sweep_cpu
mkdir -p build_sweep_cpu
cd build_sweep_cpu
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF > cmake.log 2>&1
make channel -j4 > build.log 2>&1
echo "✓ CPU build complete"
cd ..

echo ""
echo "=== Building GPU version ==="
rm -rf build_sweep_gpu
mkdir -p build_sweep_gpu
cd build_sweep_gpu
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON > cmake.log 2>&1
make channel -j4 > build.log 2>&1
echo "✓ GPU build complete"
cd ..

# Run parameter sweep
FAILED_CASES=()
PASSED_CASES=0

for test_case in "${TEST_CASES[@]}"; do
    read -r NX NY NU MODEL <<< "$test_case"
    
    echo ""
    echo "========================================"
    echo "Test case: ${NX}x${NY}, nu=${NU}, model=${MODEL}"
    echo "========================================"
    
    # Create output directories
    mkdir -p output_sweep_cpu_${NX}_${NY}_${MODEL}
    mkdir -p output_sweep_gpu_${NX}_${NY}_${MODEL}
    
    # Run CPU
    echo "  Running CPU solver..."
    cd build_sweep_cpu
    ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
              --model $MODEL --dp_dx $DP_DX \
              --output ../output_sweep_cpu_${NX}_${NY}_${MODEL}/result \
              --num_snapshots 0 > ../output_sweep_cpu_${NX}_${NY}_${MODEL}/run.log 2>&1
    cd ..
    echo "  ✓ CPU complete"
    
    # Run GPU
    echo "  Running GPU solver..."
    cd build_sweep_gpu
    ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
              --model $MODEL --dp_dx $DP_DX \
              --output ../output_sweep_gpu_${NX}_${NY}_${MODEL}/result \
              --num_snapshots 0 > ../output_sweep_gpu_${NX}_${NY}_${MODEL}/run.log 2>&1
    cd ..
    echo "  ✓ GPU complete"
    
    # Compare results
    CPU_FILE="output_sweep_cpu_${NX}_${NY}_${MODEL}/resultchannel_velocity.dat"
    GPU_FILE="output_sweep_gpu_${NX}_${NY}_${MODEL}/resultchannel_velocity.dat"
    
    if [[ ! -f "$CPU_FILE" || ! -f "$GPU_FILE" ]]; then
        echo "  ✗ ERROR: Output files not found"
        FAILED_CASES+=("${NX}x${NY}_${MODEL} (missing output)")
        continue
    fi
    
    echo "  Comparing velocity fields..."
    
    # Compute L2 and Linf norms
    paste <(tail -n +2 "$CPU_FILE") <(tail -n +2 "$GPU_FILE") | \
    awk -v tol_linf=$TOL_LINF -v tol_l2=$TOL_L2 -v case_name="${NX}x${NY}_${MODEL}" '
    BEGIN {
        max_mag = 0;
        sum_sq_mag = 0;
        n = 0;
    }
    {
        u_cpu = $3; v_cpu = $4;
        u_gpu = $8; v_gpu = $9;
        
        du = u_cpu - u_gpu;
        dv = v_cpu - v_gpu;
        dmag = sqrt(du*du + dv*dv);
        
        if (dmag > max_mag) max_mag = dmag;
        sum_sq_mag += dmag*dmag;
        n++;
    }
    END {
        if (n == 0) {
            print "    ✗ ERROR: No data points";
            exit 1;
        }
        
        l2_mag = sqrt(sum_sq_mag / n);
        
        printf "    L∞(vel_diff): %.6e\n", max_mag;
        printf "    L2(vel_diff): %.6e\n", l2_mag;
        
        failed = 0;
        if (max_mag > tol_linf) {
            printf "    ✗ L∞ FAILED: %.6e > %.6e\n", max_mag, tol_linf;
            failed = 1;
        } else {
            printf "    ✓ L∞ PASSED\n";
        }
        
        if (l2_mag > tol_l2) {
            printf "    ✗ L2 FAILED: %.6e > %.6e\n", l2_mag, tol_l2;
            failed = 1;
        } else {
            printf "    ✓ L2 PASSED\n";
        }
        
        if (failed) {
            print "  CASE_FAILED:" case_name;
            exit 1;
        } else {
            print "  CASE_PASSED:" case_name;
        }
    }
    ' && {
        PASSED_CASES=$((PASSED_CASES + 1))
    } || {
        FAILED_CASES+=("${NX}x${NY}_${MODEL}")
    }
done

# Summary
echo ""
echo "========================================"
echo "Parameter Sweep Summary"
echo "========================================"
echo "Total cases: ${#TEST_CASES[@]}"
echo "Passed: $PASSED_CASES"
echo "Failed: ${#FAILED_CASES[@]}"

if [ ${#FAILED_CASES[@]} -gt 0 ]; then
    echo ""
    echo "Failed cases:"
    for case in "${FAILED_CASES[@]}"; do
        echo "  - $case"
    done
    echo ""
    echo "✗ PARAMETER SWEEP FAILED"
    exit 1
else
    echo ""
    echo "✓ ALL PARAMETER SWEEP TESTS PASSED"
fi

