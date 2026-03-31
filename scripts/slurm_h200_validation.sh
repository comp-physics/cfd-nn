#!/bin/bash
#SBATCH --job-name=cfd-nn-validate
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=gts-sbryngelson3
#SBATCH --qos=embers
#SBATCH --output=validation_%j.log
#SBATCH --error=validation_%j.err

# H200 validation: build + run all models on all 4 cases (short runs)
# This verifies everything works on H200 before long production runs.

set -euo pipefail

module load nvhpc/24.5
export OMP_TARGET_OFFLOAD=MANDATORY

cd /storage/scratch1/6/sbryngelson3/cfd-nn

# Build for H200 (CC=90)
echo "=== Building for H200 ==="
mkdir -p build_h200
cd build_h200
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90 \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF 2>&1 | tail -2
make -j8 cylinder hills duct 2>&1 | tail -3
echo "Build complete."
cd ..

B=build_h200
NSTEPS=500

# Model specs
CLASSICAL="none baseline gep sst earsm_wj earsm_gs earsm_pope rsm"
DUCT_CLASSICAL="none baseline gep sst rsm"  # skip EARSM on duct
NN="nn_mlp:mlp_paper nn_mlp:mlp_med_paper nn_mlp:mlp_large_paper"
NN="$NN nn_tbnn:tbnn_paper nn_tbnn:tbnn_small_paper nn_tbnn:tbnn_large_paper"
NN="$NN nn_tbnn:pi_tbnn_paper nn_tbnn:pi_tbnn_small_paper nn_tbnn:pi_tbnn_large_paper"
NN="$NN nn_tbrf:tbrf_1t_paper nn_tbrf:tbrf_5t_paper nn_tbrf:tbrf_10t_paper"

run_model() {
    local bin=$1 cfg=$2 model=$3 weights=$4 steps=$5
    local cmd="$B/$bin --config $cfg --max_steps $steps --model $model"
    [ -n "$weights" ] && cmd="$cmd --weights data/models/$weights"
    timeout 600 $cmd 2>&1
}

echo ""
echo "=========================================="
echo "=== CYLINDER Re=100 ($NSTEPS steps)    ==="
echo "=========================================="
printf "%-22s %12s %12s %8s\n" "Model" "Cd" "Cl" "Status"
echo "------------------------------------------------------"
for m in $CLASSICAL; do
    line=$(run_model cylinder examples/test_configs/cylinder_quick.cfg $m "" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1); cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$m" "$cd" "$cl" "$status"
done
for spec in $NN; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    line=$(run_model cylinder examples/test_configs/cylinder_quick.cfg $cli "$wt" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1); cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$wt" "$cd" "$cl" "$status"
done

echo ""
echo "=========================================="
echo "=== DUCT Re_b=3500 ($NSTEPS steps)     ==="
echo "=========================================="
printf "%-22s %12s %8s\n" "Model" "Residual" "Status"
echo "--------------------------------------------"
for m in $DUCT_CLASSICAL; do
    res=$(run_model duct examples/test_configs/duct_quick.cfg $m "" $NSTEPS | grep "res=" | tail -1 | sed 's/.*res=//' | cut -d' ' -f1)
    status="OK"; [ -z "$res" ] && status="FAIL" && res="---"
    printf "%-22s %12s %8s\n" "$m" "$res" "$status"
done
for spec in $NN; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    res=$(run_model duct examples/test_configs/duct_quick.cfg $cli "$wt" $NSTEPS | grep "res=" | tail -1 | sed 's/.*res=//' | cut -d' ' -f1)
    status="OK"; [ -z "$res" ] && status="FAIL" && res="---"
    printf "%-22s %12s %8s\n" "$wt" "$res" "$status"
done

echo ""
echo "=========================================="
echo "=== HILLS Re=5600 ($NSTEPS steps, warmup=2s) ==="
echo "=========================================="
printf "%-22s %12s %8s\n" "Model" "U_b" "Status"
echo "--------------------------------------------"
for m in none sst earsm_wj rsm; do
    ub=$(run_model hills examples/test_configs/hills_quick.cfg $m "" $NSTEPS | grep "U_b=" | tail -1 | sed 's/.*U_b=//')
    status="OK"; [ -z "$ub" ] && status="FAIL" && ub="---"
    printf "%-22s %12s %8s\n" "$m" "$ub" "$status"
done
for spec in nn_mlp:mlp_paper nn_tbnn:tbnn_paper nn_tbrf:tbrf_1t_paper; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    ub=$(run_model hills examples/test_configs/hills_quick.cfg $cli "$wt" $NSTEPS | grep "U_b=" | tail -1 | sed 's/.*U_b=//')
    status="OK"; [ -z "$ub" ] && status="FAIL" && ub="---"
    printf "%-22s %12s %8s\n" "$wt" "$ub" "$status"
done

echo ""
echo "=========================================="
echo "=== SPHERE Re=200 (384x256x256, $NSTEPS steps) ==="
echo "=========================================="
printf "%-22s %12s %12s %8s\n" "Model" "Cd" "Cl" "Status"
echo "------------------------------------------------------"
for m in none sst rsm; do
    line=$(run_model cylinder examples/paper_experiments/sphere_re200.cfg $m "" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1); cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$m" "$cd" "$cl" "$status"
done
for spec in nn_mlp:mlp_paper nn_tbnn:tbnn_paper nn_tbrf:tbrf_1t_paper; do
    cli=$(echo $spec | cut -d: -f1); wt=$(echo $spec | cut -d: -f2)
    line=$(run_model cylinder examples/paper_experiments/sphere_re200.cfg $cli "$wt" $NSTEPS | grep "Cd=" | tail -1)
    cd=$(echo "$line" | sed 's/.*Cd=//' | cut -d' ' -f1); cl=$(echo "$line" | sed 's/.*Cl=//')
    status="OK"; [ -z "$cd" ] && status="FAIL" && cd="---" && cl="---"
    printf "%-22s %12s %12s %8s\n" "$wt" "$cd" "$cl" "$status"
done

echo ""
echo "=========================================="
echo "=== VALIDATION COMPLETE ==="
echo "=========================================="
