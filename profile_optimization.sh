#!/bin/bash
#SBATCH --job-name=gpu_opt_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_opt_profile_%j.out
#SBATCH --error=gpu_opt_profile_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --account=gts-sbryngelson3

set -e

echo "========================================"
echo "GPU Optimization Profiling"
echo "========================================"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "========================================"
echo ""

# Load required modules
module load nvhpc/24.5
module load cuda/11.8.0 2>/dev/null || echo "CUDA module not available"

# Create output directory
WORKSPACE="/storage/home/hcoda1/6/sbryngelson3/cfd-nn"
PROFILE_DIR="${WORKSPACE}/optimization_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${PROFILE_DIR}

echo "Results will be saved to: ${PROFILE_DIR}"
echo ""

# Build with GPU offloading
echo "========================================"
echo "Building with GPU Offloading (Release)"
echo "========================================"

cd ${WORKSPACE}
rm -rf build_profile
mkdir -p build_profile
cd build_profile

CC=nvc CXX=nvc++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF

make -j4 test_solver 2>&1 | tee ${PROFILE_DIR}/build.log

if [ $? -ne 0 ]; then
    echo "❌ Build FAILED!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# =================================================================
# PHASE 1: KERNEL LAUNCH COUNTING (Main metric for optimization)
# =================================================================
echo "========================================"
echo "Phase 1: Kernel Launch Analysis"
echo "========================================"

export NVCOMPILER_ACC_NOTIFY=3
export NV_ACC_CUDA_HEAPSIZE=1G
export OMP_TARGET_OFFLOAD=MANDATORY

echo "Running test_solver with full kernel tracking..."
echo "This will take a few minutes..."
echo ""

./test_solver > ${PROFILE_DIR}/kernel_launches.log 2>&1

# Count kernels
echo "Analyzing kernel launches..."
TOTAL_KERNELS=$(grep -c "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log || echo "0")
BC_KERNELS=$(grep "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log | grep -c "apply_bc" || echo "0")
SMOOTH_KERNELS=$(grep "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log | grep -c "smooth" || echo "0")
UPLOAD_COUNT=$(grep -c "upload" ${PROFILE_DIR}/kernel_launches.log || echo "0")
DOWNLOAD_COUNT=$(grep -c "download" ${PROFILE_DIR}/kernel_launches.log || echo "0")
TOTAL_TRANSFERS=$((UPLOAD_COUNT + DOWNLOAD_COUNT))

echo ""
echo "=== Kernel Launch Summary ==="
echo "Total CUDA kernels:        $TOTAL_KERNELS"
echo "  - apply_bc kernels:      $BC_KERNELS ($(echo "scale=1; 100 * $BC_KERNELS / $TOTAL_KERNELS" | bc)%)"
echo "  - smooth kernels:        $SMOOTH_KERNELS ($(echo "scale=1; 100 * $SMOOTH_KERNELS / $TOTAL_KERNELS" | bc)%)"
echo "Upload operations:         $UPLOAD_COUNT"
echo "Download operations:       $DOWNLOAD_COUNT"
echo "Total transfers:           $TOTAL_TRANSFERS"
echo ""

# Extract top kernels
echo "=== Top 15 Most Called Kernels ==="
grep "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log | \
    sed 's/.*function=//; s/ line=.*//; s/ .*//' | \
    sort | uniq -c | sort -rn | head -15 | tee ${PROFILE_DIR}/kernel_stats.txt
echo ""

# Compare to baseline
echo "=== Comparison to Baseline (Pre-Optimization) ==="
echo "Baseline BC kernels:       6,331,608 (72% of total)"
echo "Baseline total kernels:    8,769,713"
echo ""
echo "Current BC kernels:        $BC_KERNELS"
echo "Current total kernels:     $TOTAL_KERNELS"
echo ""

if [ "$BC_KERNELS" -lt 1500000 ]; then
    REDUCTION=$(echo "scale=1; (6331608 - $BC_KERNELS) * 100 / 6331608" | bc)
    echo "✅ BC REDUCTION: ${REDUCTION}% (Expected: ~85%)"
else
    echo "⚠️  BC count still high (expected <1.5M)"
fi

if [ "$TOTAL_KERNELS" -lt 4500000 ]; then
    REDUCTION=$(echo "scale=1; (8769713 - $TOTAL_KERNELS) * 100 / 8769713" | bc)
    echo "✅ TOTAL REDUCTION: ${REDUCTION}% (Expected: ~60%)"
else
    echo "⚠️  Total kernel count still high (expected <4.5M)"
fi

echo ""
echo "✅ Phase 1 complete"
echo ""

# =================================================================
# PHASE 2: CORRECTNESS VALIDATION
# =================================================================
echo "========================================"
echo "Phase 2: Correctness Validation"
echo "========================================"

export NVCOMPILER_ACC_NOTIFY=0

echo "Running full test suite on GPU..."
./test_solver 2>&1 | tee ${PROFILE_DIR}/all_tests.log

# Extract results
PASS_COUNT=$(grep -c "PASSED" ${PROFILE_DIR}/all_tests.log || echo "0")
FAIL_COUNT=$(grep -c "FAILED" ${PROFILE_DIR}/all_tests.log || echo "0")

echo ""
echo "=== Test Results ==="
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "❌ Some tests FAILED!"
    grep "FAILED" ${PROFILE_DIR}/all_tests.log | head -10
else
    echo "✅ All tests PASSED!"
fi

echo ""
echo "✅ Phase 2 complete"
echo ""

# =================================================================
# PHASE 3: PERFORMANCE TIMING
# =================================================================
echo "========================================"
echo "Phase 3: Performance Timing"
echo "========================================"

echo "Measuring GPU execution time (5 runs for averaging)..."

TIMES=()
for i in {1..5}; do
    echo -n "  Run $i: "
    START=$(date +%s.%N)
    ./test_solver > /dev/null 2>&1
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    TIMES+=($TIME)
    echo "${TIME}s"
done

# Calculate average
TOTAL_TIME=0
for t in "${TIMES[@]}"; do
    TOTAL_TIME=$(echo "$TOTAL_TIME + $t" | bc)
done
AVG_TIME=$(echo "scale=2; $TOTAL_TIME / 5" | bc)

echo ""
echo "Average execution time: ${AVG_TIME}s"
echo ""
echo "✅ Phase 3 complete"
echo ""

# =================================================================
# GENERATE REPORT
# =================================================================
echo "========================================"
echo "Generating Report"
echo "========================================"

cat > ${PROFILE_DIR}/OPTIMIZATION_REPORT.md <<EOF
# GPU Optimization Impact Report

**Date:** $(date)
**Branch:** $(git branch --show-current)
**Commit:** $(git rev-parse --short HEAD)
**Node:** $(hostname)
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')

---

## Executive Summary

### Kernel Launch Reduction

| Metric | Baseline | After Optimization | Reduction |
|--------|----------|-------------------|-----------|
| **BC kernels** | 6,331,608 (72%) | $BC_KERNELS ($(echo "scale=1; 100 * $BC_KERNELS / $TOTAL_KERNELS" | bc)%) | $(echo "scale=1; (6331608 - $BC_KERNELS) * 100 / 6331608" | bc)% |
| **Total kernels** | 8,769,713 | $TOTAL_KERNELS | $(echo "scale=1; (8769713 - $TOTAL_KERNELS) * 100 / 8769713" | bc)% |
| **Memory transfers** | 536,495 | $TOTAL_TRANSFERS | $(echo "scale=1; (536495 - $TOTAL_TRANSFERS) * 100 / 536495" | bc)% |

### Correctness Validation

- **Tests passed:** $PASS_COUNT
- **Tests failed:** $FAIL_COUNT

$(if [ "$FAIL_COUNT" -eq 0 ]; then echo "✅ All physics validation tests pass"; else echo "❌ Some tests failed"; fi)

### Performance

- **Average execution time:** ${AVG_TIME}s (5 runs)

---

## Detailed Analysis

### Top 15 Kernel Calls

\`\`\`
$(cat ${PROFILE_DIR}/kernel_stats.txt)
\`\`\`

### Optimization Impact Assessment

EOF

# Assess success
BC_REDUCTION=$(echo "scale=1; (6331608 - $BC_KERNELS) * 100 / 6331608" | bc)
TOTAL_REDUCTION=$(echo "scale=1; (8769713 - $TOTAL_KERNELS) * 100 / 8769713" | bc)

BC_SUCCESS=$(echo "$BC_REDUCTION > 70" | bc)
TOTAL_SUCCESS=$(echo "$TOTAL_REDUCTION > 50" | bc)
TESTS_SUCCESS=$([ "$FAIL_COUNT" -eq 0 ] && echo 1 || echo 0)

if [ "$BC_SUCCESS" -eq 1 ] && [ "$TOTAL_SUCCESS" -eq 1 ] && [ "$TESTS_SUCCESS" -eq 1 ]; then
    cat >> ${PROFILE_DIR}/OPTIMIZATION_REPORT.md <<EOF
✅ **OPTIMIZATION SUCCESSFUL!**

The Priority 1 optimization achieved its goals:
- BC kernel launches reduced by ${BC_REDUCTION}% (target: >70%)
- Total kernel launches reduced by ${TOTAL_REDUCTION}% (target: >50%)
- All correctness tests pass

**Recommendation:** Proceed with merging to main branch and implementing Priority 2 optimizations.
EOF
else
    cat >> ${PROFILE_DIR}/OPTIMIZATION_REPORT.md <<EOF
⚠️ **OPTIMIZATION RESULTS NEED REVIEW**

Results:
- BC reduction: ${BC_REDUCTION}% (target: >70%) - $([ "$BC_SUCCESS" -eq 1 ] && echo "✅" || echo "❌")
- Total reduction: ${TOTAL_REDUCTION}% (target: >50%) - $([ "$TOTAL_SUCCESS" -eq 1 ] && echo "✅" || echo "❌")
- Tests passing: $([ "$TESTS_SUCCESS" -eq 1 ] && echo "✅" || echo "❌")

**Recommendation:** Review logs for unexpected behavior.
EOF
fi

cat >> ${PROFILE_DIR}/OPTIMIZATION_REPORT.md <<EOF

---

## Files in This Report

- \`build.log\` - Build output
- \`kernel_launches.log\` - Full kernel trace (~2GB)
- \`kernel_stats.txt\` - Kernel statistics summary
- \`all_tests.log\` - Complete test output
- \`OPTIMIZATION_REPORT.md\` - This report

---

**Completed:** $(date)
EOF

echo ""
echo "Report saved to: ${PROFILE_DIR}/OPTIMIZATION_REPORT.md"
echo ""
cat ${PROFILE_DIR}/OPTIMIZATION_REPORT.md

echo ""
echo "========================================"
echo "Profiling Complete!"
echo "========================================"
echo ""
echo "Results: ${PROFILE_DIR}"
echo ""
echo "Key files:"
echo "  - OPTIMIZATION_REPORT.md (executive summary)"
echo "  - kernel_stats.txt (top kernels)"
echo "  - all_tests.log (validation results)"
echo ""
echo "Completed: $(date)"
