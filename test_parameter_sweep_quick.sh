#!/bin/bash
# Quick parameter sweep for testing (smaller than full sweep)

set -e

echo "========================================"
echo "Quick Parameter Sweep Test"
echo "========================================"

module reset
module load nvhpc

# Smaller test cases for quick validation
declare -a TEST_CASES=(
    "64 128 0.001 baseline"
    "63 129 0.001 baseline"  # Odd grid
)

ITERS=500  # Increased to ensure convergence
DP_DX=-0.0001
TOL_LINF=1e-7
TOL_L2=1e-8

echo "Test cases: ${#TEST_CASES[@]}"
echo "Iterations: $ITERS"

# Build
echo ""
echo "Building..."
rm -rf build_sweep_quick_cpu build_sweep_quick_gpu
mkdir -p build_sweep_quick_cpu build_sweep_quick_gpu

cd build_sweep_quick_cpu
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=OFF > /dev/null 2>&1
make channel -j4 > /dev/null 2>&1
cd ..

cd build_sweep_quick_gpu
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON > /dev/null 2>&1
make channel -j4 > /dev/null 2>&1
cd ..

echo "✓ Builds complete"

# Run tests
PASSED=0
for test_case in "${TEST_CASES[@]}"; do
    read -r NX NY NU MODEL <<< "$test_case"
    echo ""
    echo "Testing ${NX}x${NY}, model=${MODEL}..."
    
    mkdir -p out_qcpu_${NX}_${NY} out_qgpu_${NX}_${NY}
    
    cd build_sweep_quick_cpu
    ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
              --model $MODEL --dp_dx $DP_DX \
              --output ../out_qcpu_${NX}_${NY}/r --num_snapshots 0 > ../out_qcpu_${NX}_${NY}/run.log 2>&1
    cd ..
    
    echo "  CPU summary:"
    grep -E "Final residual|Iterations|Converged" out_qcpu_${NX}_${NY}/run.log | head -3 || echo "    (no summary found)"
    
    cd build_sweep_quick_gpu
    # Run GPU build on actual GPU node
    srun -A gts-sbryngelson3 --qos=embers --partition=gpu-v100 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=0:05:00 \
        ./channel --Nx $NX --Ny $NY --nu $NU --max_iter $ITERS \
                  --model $MODEL --dp_dx $DP_DX \
                  --output ../out_qgpu_${NX}_${NY}/r --num_snapshots 0 > ../out_qgpu_${NX}_${NY}/run.log 2>&1
    cd ..
    
    echo "  GPU summary:"
    grep -E "Final residual|Iterations|Converged" out_qgpu_${NX}_${NY}/run.log | head -3 || echo "    (no summary found)"
    
    CPU_F="out_qcpu_${NX}_${NY}/rchannel_velocity.dat"
    GPU_F="out_qgpu_${NX}_${NY}/rchannel_velocity.dat"
    CPU_NUT="out_qcpu_${NX}_${NY}/rchannel_nu_t.dat"
    GPU_NUT="out_qgpu_${NX}_${NY}/rchannel_nu_t.dat"
    
    echo "  Velocity comparison:"
    paste <(tail -n +2 "$CPU_F") <(tail -n +2 "$GPU_F") | \
    awk -v tol_linf=$TOL_LINF -v tol_l2=$TOL_L2 '
    BEGIN { max_mag = 0; sum_sq = 0; n = 0; }
    {
        du = $3 - $8; dv = $4 - $9;
        dmag = sqrt(du*du + dv*dv);
        if (dmag > max_mag) max_mag = dmag;
        sum_sq += dmag*dmag; n++;
    }
    END {
        l2 = sqrt(sum_sq / n);
        printf "    L∞=%.2e, L2=%.2e ", max_mag, l2;
        if (max_mag <= tol_linf && l2 <= tol_l2) {
            print "✓";
            exit 0;
        } else {
            print "✗ FAILED";
            exit 1;
        }
    }
    ' 
    
    VEL_RESULT=$?
    
    # Also compare nu_t if model is not "none"
    if [[ "$MODEL" != "none" && -f "$CPU_NUT" && -f "$GPU_NUT" ]]; then
        echo "  nu_t comparison:"
        paste <(tail -n +2 "$CPU_NUT") <(tail -n +2 "$GPU_NUT") | \
        awk -v tol_linf=$TOL_LINF -v tol_l2=$TOL_L2 '
        BEGIN { maxd=0; sum2=0; n=0; }
        { d=$3-$6; ad=(d<0)?-d:d; if(ad>maxd) maxd=ad; sum2+=d*d; n++; }
        END {
            l2=sqrt(sum2/n);
            printf "    L∞=%.2e, L2=%.2e ", maxd, l2;
            if(maxd<=tol_linf && l2<=tol_l2) {
                print "✓";
            } else {
                print "✗ FAILED";
            }
        }
        '
    fi
    
    if [ $VEL_RESULT -eq 0 ]; then
        PASSED=$((PASSED + 1))
    fi
done

echo ""
if [ $PASSED -eq ${#TEST_CASES[@]} ]; then
    echo "✓ ALL QUICK SWEEP TESTS PASSED ($PASSED/${#TEST_CASES[@]})"
else
    echo "✗ SOME TESTS FAILED"
    exit 1
fi

