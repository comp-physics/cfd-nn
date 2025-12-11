#!/bin/bash
#SBATCH --job-name=gpu_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gpu_profile_%j.out
#SBATCH --error=gpu_profile_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --qos=inferno
#SBATCH --account=gts-sbryngelson3

set -e

echo "========================================"
echo "Comprehensive GPU Profiling & Validation"
echo "========================================"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"
echo ""

# Load required modules
module load cuda/11.8.0

# Create output directory in workspace
WORKSPACE="/storage/home/hcoda1/6/sbryngelson3/cfd-nn"
PROFILE_DIR="${WORKSPACE}/gpu_profile_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${PROFILE_DIR}

echo "Results will be saved to: ${PROFILE_DIR}"
echo ""

# Build with GPU offloading
echo "========================================"
echo "Phase 0: Building with GPU Offloading"
echo "========================================"

cd ${WORKSPACE}
rm -rf build_gpu_profile
mkdir -p build_gpu_profile
cd build_gpu_profile

CC=nvc CXX=nvc++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF

make -j4 test_solver channel 2>&1 | tee ${PROFILE_DIR}/build.log

if [ $? -ne 0 ]; then
    echo "❌ Build FAILED!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# =================================================================
# PHASE 1: KERNEL LAUNCH VERIFICATION
# =================================================================
echo "========================================"
echo "Phase 1: Kernel Launch Verification"
echo "========================================"

export NVCOMPILER_ACC_NOTIFY=3
export NV_ACC_CUDA_HEAPSIZE=1G
export OMP_TARGET_OFFLOAD=MANDATORY

echo "Running test_solver with kernel launch tracking..."
./test_solver > ${PROFILE_DIR}/kernel_launches.log 2>&1

# Analyze kernel launches
KERNEL_COUNT=$(grep -c "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log || echo "0")
UPLOAD_COUNT=$(grep -c "upload" ${PROFILE_DIR}/kernel_launches.log || echo "0")
DOWNLOAD_COUNT=$(grep -c "download" ${PROFILE_DIR}/kernel_launches.log || echo "0")

echo ""
echo "=== Kernel Launch Summary ==="
echo "Total CUDA kernels launched: $KERNEL_COUNT"
echo "Upload operations: $UPLOAD_COUNT"
echo "Download operations: $DOWNLOAD_COUNT"
echo ""

if [ "$KERNEL_COUNT" -lt 50 ]; then
    echo "⚠️  WARNING: Expected >50 kernel launches, found $KERNEL_COUNT"
else
    echo "✅ GPU kernels are launching ($KERNEL_COUNT launches detected)"
fi

# Extract unique kernels
echo ""
echo "=== Top 20 Most Called Kernels ==="
grep "launch CUDA kernel" ${PROFILE_DIR}/kernel_launches.log | \
    sed 's/.*function=//; s/ line=.*//; s/ .*//' | \
    sort | uniq -c | sort -rn | head -20 | tee ${PROFILE_DIR}/unique_kernels.txt
echo ""

echo "✅ Phase 1 complete"
echo ""

# =================================================================
# PHASE 2: GPU vs CPU TIMING (32x64 mesh only for speed)
# =================================================================
echo "========================================"
echo "Phase 2: GPU vs CPU Timing"
echo "========================================"

echo "Test,MeshSize,Device,Time(s),Status" > ${PROFILE_DIR}/timing_comparison.csv

echo "Testing with default mesh (32x64)..."

# CPU timing
echo -n "  CPU: "
export OMP_TARGET_OFFLOAD=DISABLED
START=$(date +%s.%N)
./test_solver 2>&1 | grep -A1 "Testing laminar Poiseuille" | grep "PASSED" > ${PROFILE_DIR}/cpu_result.txt
END=$(date +%s.%N)
CPU_TIME=$(echo "$END - $START" | bc)
CPU_STATUS=$(cat ${PROFILE_DIR}/cpu_result.txt | grep -o "PASSED\|FAILED" || echo "UNKNOWN")
echo "$CPU_TIME seconds - $CPU_STATUS"

echo "Poiseuille,32x64,CPU,$CPU_TIME,$CPU_STATUS" >> ${PROFILE_DIR}/timing_comparison.csv

# GPU timing
echo -n "  GPU: "
export OMP_TARGET_OFFLOAD=MANDATORY
export NVCOMPILER_ACC_NOTIFY=0
START=$(date +%s.%N)
./test_solver 2>&1 | grep -A1 "Testing laminar Poiseuille" | grep "PASSED" > ${PROFILE_DIR}/gpu_result.txt
END=$(date +%s.%N)
GPU_TIME=$(echo "$END - $START" | bc)
GPU_STATUS=$(cat ${PROFILE_DIR}/gpu_result.txt | grep -o "PASSED\|FAILED" || echo "UNKNOWN")
echo "$GPU_TIME seconds - $GPU_STATUS"

echo "Poiseuille,32x64,GPU,$GPU_TIME,$GPU_STATUS" >> ${PROFILE_DIR}/timing_comparison.csv

# Calculate speedup
if [ "$GPU_TIME" != "0" ] && [ "$GPU_TIME" != "" ]; then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo "  Speedup: ${SPEEDUP}x"
else
    SPEEDUP="N/A"
fi

echo ""
echo "=== Timing Summary ==="
cat ${PROFILE_DIR}/timing_comparison.csv
echo ""
echo "✅ Phase 2 complete"
echo ""

# =================================================================
# PHASE 3: ALL TESTS VALIDATION
# =================================================================
echo "========================================"
echo "Phase 3: Full Test Suite on GPU"
echo "========================================"

export OMP_TARGET_OFFLOAD=MANDATORY
export NVCOMPILER_ACC_NOTIFY=1

echo "Running all solver tests on GPU..."
./test_solver 2>&1 | tee ${PROFILE_DIR}/all_tests.log

# Extract results
echo ""
echo "=== Test Summary ==="
PASS_COUNT=$(grep -c "PASSED" ${PROFILE_DIR}/all_tests.log || echo "0")
FAIL_COUNT=$(grep -c "FAILED" ${PROFILE_DIR}/all_tests.log || echo "0")

echo "Passes: $PASS_COUNT"
echo "Failures: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "⚠️  Some tests FAILED!"
else
    echo "✅ All tests PASSED!"
fi

echo ""
echo "✅ Phase 3 complete"
echo ""

# =================================================================
# PHASE 4: MEMORY TRANSFER ANALYSIS
# =================================================================
echo "========================================"
echo "Phase 4: Memory Transfer Analysis"
echo "========================================"

export NVCOMPILER_ACC_NOTIFY=2
export OMP_TARGET_OFFLOAD=MANDATORY

echo "Running with detailed transfer tracking..."
./test_solver 2>&1 > ${PROFILE_DIR}/memory_transfers.log

# Analyze transfers
TOTAL_UPLOADS=$(grep -c "upload" ${PROFILE_DIR}/memory_transfers.log || echo "0")
TOTAL_DOWNLOADS=$(grep -c "download" ${PROFILE_DIR}/memory_transfers.log || echo "0")
TOTAL_TRANSFERS=$((TOTAL_UPLOADS + TOTAL_DOWNLOADS))

echo ""
echo "=== Memory Transfer Summary ==="
echo "Total upload operations: $TOTAL_UPLOADS"
echo "Total download operations: $TOTAL_DOWNLOADS"
echo "Total transfers: $TOTAL_TRANSFERS"

if [ "$KERNEL_COUNT" -gt 0 ]; then
    TRANSFERS_PER_KERNEL=$(echo "scale=2; $TOTAL_TRANSFERS / $KERNEL_COUNT" | bc)
    echo "Transfers per kernel: $TRANSFERS_PER_KERNEL"
    
    THRESHOLD=$(echo "$TRANSFERS_PER_KERNEL < 0.5" | bc)
    if [ "$THRESHOLD" -eq 1 ]; then
        echo "✅ GOOD: Low transfer rate (persistent mapping working)"
    else
        echo "⚠️  WARNING: High transfer rate"
    fi
fi

echo ""
echo "✅ Phase 4 complete"
echo ""

# =================================================================
# FINAL REPORT
# =================================================================
echo "========================================"
echo "Generating Final Report"
echo "========================================"

cat > ${PROFILE_DIR}/PROFILING_REPORT.md <<EOF
# GPU Profiling Report

**Date:** $(date)
**Node:** $(hostname)
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader)
**CUDA Version:** $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
**Compiler:** NVHPC 24.5

---

## Executive Summary

### Kernel Launch Statistics
- **Total CUDA kernels launched:** $KERNEL_COUNT
- **Upload operations:** $TOTAL_UPLOADS
- **Download operations:** $TOTAL_DOWNLOADS
- **Transfers per kernel:** ${TRANSFERS_PER_KERNEL:-N/A}

### Performance
- **CPU time:** ${CPU_TIME}s
- **GPU time:** ${GPU_TIME}s
- **Speedup:** ${SPEEDUP}x

### Correctness
- **Tests passed:** $PASS_COUNT
- **Tests failed:** $FAIL_COUNT

---

## Detailed Analysis

### Phase 1: Kernel Launch Verification
EOF

if [ "$KERNEL_COUNT" -lt 50 ]; then
    echo "⚠️ **WARNING:** Low kernel count detected" >> ${PROFILE_DIR}/PROFILING_REPORT.md
else
    echo "✅ **SUCCESS:** GPU kernels launching correctly" >> ${PROFILE_DIR}/PROFILING_REPORT.md
fi

cat >> ${PROFILE_DIR}/PROFILING_REPORT.md <<EOF

Top kernels:
\`\`\`
$(head -10 ${PROFILE_DIR}/unique_kernels.txt)
\`\`\`

### Phase 2: Performance

\`\`\`
$(cat ${PROFILE_DIR}/timing_comparison.csv)
\`\`\`

EOF

if [ "$SPEEDUP" != "N/A" ]; then
    SPEEDUP_NUM=$(echo "$SPEEDUP" | bc)
    SPEEDUP_CHECK=$(echo "$SPEEDUP_NUM > 1.0" | bc 2>/dev/null || echo "0")
    
    if [ "$SPEEDUP_CHECK" -eq 1 ]; then
        echo "✅ GPU provides speedup over CPU" >> ${PROFILE_DIR}/PROFILING_REPORT.md
    else
        echo "⚠️ GPU slower than CPU (may have overhead for small problems)" >> ${PROFILE_DIR}/PROFILING_REPORT.md
    fi
fi

cat >> ${PROFILE_DIR}/PROFILING_REPORT.md <<EOF

### Phase 3: Correctness
- **$PASS_COUNT tests passed**
- **$FAIL_COUNT tests failed**

### Phase 4: Memory Efficiency
- Total memory transfers: $TOTAL_TRANSFERS
- Transfers per kernel: ${TRANSFERS_PER_KERNEL:-N/A}

EOF

if [ "$TOTAL_TRANSFERS" -lt 500 ]; then
    echo "✅ Low transfer count indicates efficient persistent mapping" >> ${PROFILE_DIR}/PROFILING_REPORT.md
fi

cat >> ${PROFILE_DIR}/PROFILING_REPORT.md <<EOF

---

## Overall Assessment

EOF

# Determine overall status
ISSUES=0

if [ "$KERNEL_COUNT" -lt 50 ]; then
    echo "- ⚠️ Low kernel launch count" >> ${PROFILE_DIR}/PROFILING_REPORT.md
    ISSUES=$((ISSUES + 1))
fi

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "- ⚠️ Some tests failing" >> ${PROFILE_DIR}/PROFILING_REPORT.md
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ **NO ISSUES DETECTED - GPU implementation is working correctly!**" >> ${PROFILE_DIR}/PROFILING_REPORT.md
else
    echo "⚠️ **$ISSUES issue(s) detected - review logs for details**" >> ${PROFILE_DIR}/PROFILING_REPORT.md
fi

cat >> ${PROFILE_DIR}/PROFILING_REPORT.md <<EOF

---

## Files in this Report
- \`build.log\` - Build output
- \`kernel_launches.log\` - Detailed kernel launch trace
- \`unique_kernels.txt\` - Kernel statistics
- \`timing_comparison.csv\` - Performance data
- \`all_tests.log\` - Complete test output
- \`memory_transfers.log\` - Memory transfer trace

EOF

echo ""
echo "✅ Report generated!"
echo ""
cat ${PROFILE_DIR}/PROFILING_REPORT.md

echo ""
echo "========================================"
echo "Profiling Complete!"
echo "========================================"
echo "Results directory: ${PROFILE_DIR}"
echo ""
echo "Completed: $(date)"

