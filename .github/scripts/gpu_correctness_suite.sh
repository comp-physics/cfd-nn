#!/bin/bash
# GPU Correctness Suite - Build and run all correctness tests on GPU
# Usage: gpu_correctness_suite.sh <workdir>
# Designed to run on an H200 GPU node via SLURM

set -euo pipefail

WORKDIR="${1:-.}"
cd "$WORKDIR"

echo "==================================================================="
echo "  GPU Correctness Suite"
echo "==================================================================="
echo ""
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Hard-require OpenMP target offload (fail if it falls back to CPU)
export OMP_TARGET_OFFLOAD=MANDATORY

chmod +x scripts/run_gpu_ci_test.sh
chmod +x .github/scripts/*.sh

# Clean rebuild (correctness suite must rebuild first)
rm -rf build_ci_gpu_correctness
mkdir -p build_ci_gpu_correctness
cd build_ci_gpu_correctness

echo "==================================================================="
echo "  Building GPU-offload binary (Release)"
echo "==================================================================="
echo ""

echo "=== CMake Configuration ==="
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON 2>&1 | tee cmake_config.log
echo ""
echo "=== CMake Configuration Summary ==="
grep -E "GPU offload|CXX compiler|OpenMP|NVIDIA" cmake_config.log || echo "No GPU-related config found"
echo ""
echo "=== Building ==="
make -j8
mkdir -p output

echo ""
echo "==================================================================="
echo "  1. CPU-only vs GPU-offload build consistency (two binaries)"
echo "==================================================================="
echo ""
echo "Build a CPU-only reference binary and compare against the GPU-offload binary."
../.github/scripts/compare_cpu_gpu_builds.sh "$WORKDIR"

echo ""
echo "==================================================================="
echo "  2. Unit Tests (ctest)"
echo "==================================================================="
echo ""
ctest --output-on-failure

echo ""
echo "==================================================================="
echo "  3. Algebraic Models (Fast Validation)"
echo "==================================================================="
echo ""
mkdir -p output/gpu_validation

echo "--- Testing Baseline Model ---"
../.github/scripts/test_turbulence_model_gpu.sh baseline "Baseline" 64 128 5000 output/gpu_validation/baseline

echo ""
echo "--- Testing GEP Model ---"
../.github/scripts/test_turbulence_model_gpu.sh gep "GEP" 64 128 5000 output/gpu_validation/gep

echo ""
echo "==================================================================="
echo "  3. Transport Equation Models (Fast Validation)"
echo "==================================================================="
echo ""

echo "--- Testing SST k-omega ---"
../.github/scripts/test_turbulence_model_gpu.sh sst "SST k-omega" 64 128 500 output/gpu_validation/sst 0.001

echo ""
echo "--- Testing k-omega (Wilcox) ---"
../.github/scripts/test_turbulence_model_gpu.sh komega "k-omega (Wilcox)" 64 128 500 output/gpu_validation/komega 0.001

echo ""
echo "==================================================================="
echo "  5. EARSM Models"
echo "==================================================================="
echo ""

echo "--- Testing Wallin-Johansson EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_wj "Wallin-Johansson EARSM" 256 512 1000 output/gpu_validation/earsm_wj 0.001

echo ""
echo "--- Testing Gatski-Speziale EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_gs "Gatski-Speziale EARSM" 256 512 1000 output/gpu_validation/earsm_gs 0.001

echo ""
echo "--- Testing Pope Quadratic EARSM ---"
../.github/scripts/test_turbulence_model_gpu.sh earsm_pope "Pope Quadratic EARSM" 256 512 1000 output/gpu_validation/earsm_pope 0.001

echo ""
echo "==================================================================="
echo "  5. Periodic Hills - Complex Geometry"
echo "==================================================================="
echo ""
echo "--- Testing with Baseline model ---"
./periodic_hills --Nx 64 --Ny 48 --nu 0.001 --max_iter 200 --model baseline --num_snapshots 0

echo ""
echo "==================================================================="
echo "  7. CPU/GPU Consistency Validation (Critical)"
echo "==================================================================="
echo ""
./test_cpu_gpu_consistency

echo ""
echo "==================================================================="
echo "  8. GPU Utilization Validation (Critical)"
echo "==================================================================="
echo ""
echo "Verifying that compute runs on GPU, not CPU..."
./test_gpu_utilization

echo ""
echo "==================================================================="
echo "  9. Physics Validation (Comprehensive)"
echo "==================================================================="
echo ""
./test_physics_validation
./test_tg_validation

echo ""
echo "==================================================================="
echo " 10. Perturbed Channel Flow Test (Poisson validation)"
echo "==================================================================="
echo ""
echo "Testing 2D perturbed channel (validates Poisson solver with non-trivial flow)"
echo "Running PerturbedChannelTest (already executed in ctest above)"
echo "[PASS] PerturbedChannelTest passed in unit tests"

echo ""
echo "[PASS] GPU Correctness Suite completed successfully"

