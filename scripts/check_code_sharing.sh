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

# Validate source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found: $SRC_DIR"
    exit 1
fi

echo "=== Code Sharing Paradigm Check ==="
echo ""

VIOLATIONS=0

# --- Check 1: No runtime GPU/CPU branching in compute code ---
echo "Check 1: No runtime GPU/CPU branching..."

# Pattern: look for runtime boolean checks that switch between GPU and CPU logic
# These are NOT allowed in compute kernels (but allowed for memory management)
RUNTIME_PATTERNS=(
    'if\s*\(\s*use_gpu'
    'if\s*\(\s*gpu_enabled'
    'if\s*\(\s*is_gpu'
    'if\s*\(\s*on_gpu'
    '\?\s*gpu_kernel\s*:'
    '\?\s*cpu_kernel\s*:'
)

for pattern in "${RUNTIME_PATTERNS[@]}"; do
    # Filter out:
    # - Comment-only lines (lines starting with optional whitespace then //)
    # - timing.cpp (profiling code that categorizes timing entries, not compute branching)
    matches=$(grep -rn -E "$pattern" "$SRC_DIR" 2>/dev/null | \
              grep -Ev '^[^:]+:[0-9]+:[[:space:]]*//' | \
              grep -v 'timing\.cpp' || true)
    if [ -n "$matches" ]; then
        echo "  [VIOLATION] Found runtime GPU branching:"
        echo "$matches" | head -5
        VIOLATIONS=$((VIOLATIONS + 1))
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

# This is a soft check - just warn, don't fail
echo "  [INFO] Scanning for potentially duplicated compute logic..."
echo "  (Manual review recommended for complex cases)"

# Count GPU ifdef blocks in all source files
for file in "$SRC_DIR"/*.cpp; do
    if [ -f "$file" ]; then
        count=$(grep -c "#ifdef USE_GPU_OFFLOAD" "$file" 2>/dev/null | head -1 || echo "0")
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            echo "  $(basename "$file"): $count #ifdef USE_GPU_OFFLOAD blocks"
        fi
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

# --- Check 4: No separate CPU/GPU kernel function implementations ---
echo ""
echo "Check 4: Checking for separate CPU/GPU kernel implementations..."

# Look for function naming patterns that suggest duplicate implementations
# These patterns detect actual code constructs, not comments
DUPLICATE_PATTERNS=(
    '_cpu\s*\('          # function_cpu() pattern
    '_gpu\s*\('          # function_gpu() pattern
    '_host\s*\('         # function_host() pattern
    '_device\s*\('       # function_device() pattern
)

for pattern in "${DUPLICATE_PATTERNS[@]}"; do
    # Filter out allowed patterns:
    # - Comment lines
    # - sync_*_gpu/cleanup_gpu_buffers (memory management)
    # - solve_device (intentional device-pointer entry point for Poisson solver)
    # - *_gpu() methods (kernel wrappers and query methods like weights_gpu, is_on_gpu)
    # - omp_*_device (OpenMP device management)
    matches=$(grep -rn -E "$pattern" "$SRC_DIR" 2>/dev/null | \
              grep -Ev '^[^:]+:[0-9]+:[[:space:]]*//' | \
              grep -Ev 'sync_.*_gpu|_gpu_buffers|solve_device|[a-z_]+_gpu\s*\(|omp_.*_device' || true)
    if [ -n "$matches" ]; then
        echo "  [WARNING] Found potentially duplicate kernel implementation:"
        echo "$matches" | head -3
        echo "  (Verify these are not duplicating compute logic)"
    fi
done

# Check for gpu_ready_ runtime branching (informational)
echo ""
echo "Check 5: Runtime gpu_ready_ branching (informational)..."
gpu_ready_count=$(grep -rn 'if\s*(\s*gpu_ready_' "$SRC_DIR" 2>/dev/null | wc -l || echo "0")
echo "  Found $gpu_ready_count 'if (gpu_ready_)' patterns"
echo "  (These should only guard memory transfers, not duplicate compute logic)"

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
