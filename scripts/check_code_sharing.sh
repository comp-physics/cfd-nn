#!/bin/bash
# check_code_sharing.sh - Enforce GPU/CPU code sharing paradigm
#
# This script verifies that the codebase follows the paradigm where:
# - GPU and CPU use identical compute kernels
# - The only allowed differences are:
#   1. Memory sync operations (sync_to_gpu, sync_from_gpu, etc.)
#   2. Buffer initialization/cleanup (initialize_gpu_buffers, cleanup_gpu_buffers)
#   3. OpenMP pragma annotations (#pragma omp target)
#   4. Device data mapping (#pragma omp target enter/exit data)
#
# VIOLATIONS to detect:
# - Runtime branching: if (use_gpu) { ... } else { ... } in compute code
# - Separate CPU/GPU kernel implementations (other than pragma-annotated)
# - GPU-specific compute logic not guarded by compile-time #ifdef

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"

echo "=== Code Sharing Paradigm Check ==="
echo ""

VIOLATIONS=0

# --- Check 1: No runtime GPU/CPU branching in compute code ---
echo "Check 1: No runtime GPU/CPU branching..."

# Pattern: look for runtime boolean checks that switch between GPU and CPU logic
# These are NOT allowed in compute kernels
RUNTIME_PATTERNS=(
    'if\s*\(\s*use_gpu'
    'if\s*\(\s*gpu_enabled'
    'if\s*\(\s*is_gpu'
    'if\s*\(\s*on_gpu'
    '\?\s*gpu_kernel\s*:'
    '\?\s*cpu_kernel\s*:'
)

for pattern in "${RUNTIME_PATTERNS[@]}"; do
    matches=$(grep -rn -E "$pattern" "$SRC_DIR" 2>/dev/null | grep -v '^\s*//' || true)
    if [ -n "$matches" ]; then
        echo "  [VIOLATION] Found runtime GPU branching:"
        echo "$matches" | head -5
        ((VIOLATIONS++))
    fi
done

if [ $VIOLATIONS -eq 0 ]; then
    echo "  [PASS] No runtime GPU branching detected"
fi

# --- Check 2: All #ifdef USE_GPU_OFFLOAD blocks are for allowed purposes ---
echo ""
echo "Check 2: Verifying #ifdef USE_GPU_OFFLOAD blocks are for allowed purposes..."

# Allowed patterns within GPU blocks:
# - sync_to_gpu, sync_from_gpu, sync_solution_from_gpu
# - initialize_gpu_buffers, cleanup_gpu_buffers
# - #pragma omp target
# - map(to:, map(from:, map(tofrom:
# - gpu_ready_, device_ptr, DeviceArray

# We'll do a heuristic check: look for function definitions inside #ifdef blocks
# that don't match the allowed patterns

# This is a soft check - just warn, don't fail
echo "  [INFO] Scanning for potentially duplicated compute logic..."
echo "  (Manual review recommended for complex cases)"

# Count GPU ifdef blocks in key files
for file in solver.cpp gpu_kernels.cpp; do
    if [ -f "$SRC_DIR/$file" ]; then
        count=$(grep -c "#ifdef USE_GPU_OFFLOAD" "$SRC_DIR/$file" 2>/dev/null || echo "0")
        echo "  $file: $count #ifdef USE_GPU_OFFLOAD blocks"
    fi
done

# --- Check 3: Verify kernel functions have declare target ---
echo ""
echo "Check 3: Checking kernel declarations..."

# Key kernel patterns that should have declare target (when GPU is enabled)
KERNEL_FUNCTIONS=(
    "convective_kernel"
    "diffusive_kernel"
    "pressure_gradient_kernel"
    "divergence_kernel"
)

for kernel in "${KERNEL_FUNCTIONS[@]}"; do
    if grep -q "$kernel" "$SRC_DIR"/*.cpp 2>/dev/null; then
        # Check if it's properly wrapped with declare target
        if grep -B5 "$kernel" "$SRC_DIR"/*.cpp 2>/dev/null | grep -q "declare target"; then
            echo "  [PASS] $kernel has declare target"
        else
            # Not a violation if it's only called within target regions
            echo "  [INFO] $kernel - verify it runs in target region"
        fi
    fi
done

# --- Check 4: No CPU-only compute implementations ---
echo ""
echo "Check 4: Checking for CPU-only compute paths..."

# Look for patterns like "// CPU version" or "// CPU fallback" followed by compute code
CPU_ONLY_PATTERNS=(
    'CPU.only'
    'CPU.version'
    'CPU.fallback'
    'cpu_compute'
    'host_compute'
)

for pattern in "${CPU_ONLY_PATTERNS[@]}"; do
    matches=$(grep -rn -i "$pattern" "$SRC_DIR" 2>/dev/null | grep -v '^\s*//' || true)
    if [ -n "$matches" ]; then
        echo "  [WARNING] Found CPU-specific compute reference:"
        echo "$matches" | head -3
        echo "  (Verify this is not duplicated compute logic)"
    fi
done

# --- Summary ---
echo ""
echo "=== Summary ==="
if [ $VIOLATIONS -eq 0 ]; then
    echo "[PASS] Code sharing paradigm check passed"
    echo ""
    echo "Reminder: The paradigm requires that:"
    echo "  - All compute kernels run identical code on CPU and GPU"
    echo "  - Only memory transfers and pragma annotations differ"
    echo "  - #ifdef USE_GPU_OFFLOAD should only guard:"
    echo "    * sync_to_gpu(), sync_from_gpu()"
    echo "    * initialize_gpu_buffers(), cleanup_gpu_buffers()"
    echo "    * #pragma omp target annotations"
    exit 0
else
    echo "[FAIL] Found $VIOLATIONS paradigm violations"
    echo "Please fix the violations above before committing."
    exit 1
fi
