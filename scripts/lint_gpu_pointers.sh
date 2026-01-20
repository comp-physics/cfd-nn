#!/bin/bash
# lint_gpu_pointers.sh - CI gate to prevent unsafe GPU pointer patterns
#
# For NVHPC: raw mapped pointers passed directly into target regions can
# silently use wrong memory. This script catches common violations.
#
# Exit code: 0 = clean, 1 = violations found

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0

echo "=========================================="
echo "  GPU Pointer Safety Lint"
echo "=========================================="
echo ""

# Pattern 1: firstprivate with raw pointer variables in target regions
# This is the most dangerous pattern - NVHPC may pass host address
echo "Checking for firstprivate with raw pointers..."
pattern1_matches=$(grep -rn --include="*.cpp" --include="*.hpp" \
    'firstprivate.*_ptr_\|firstprivate.*_ptr)' \
    "$REPO_ROOT/src" "$REPO_ROOT/include" 2>/dev/null | \
    grep -v "is_device_ptr" | grep -v "// SAFE:" || true)

if [ -n "$pattern1_matches" ]; then
    echo "$pattern1_matches" | head -20
    echo -e "${RED}ERROR: Found firstprivate with raw pointer variables${NC}"
    echo "       Use gpu::dev_ptr() + is_device_ptr() instead"
    ((errors++))
else
    echo -e "${GREEN}OK${NC}"
fi
echo ""

# Pattern 2: target teams without is_device_ptr when pointers are used
# Check for target regions that use pointers but don't declare is_device_ptr
echo "Checking for target regions missing is_device_ptr..."
# This is a heuristic - look for target teams with pointer dereferences but no is_device_ptr
violations=$(grep -rn --include="*.cpp" \
    '#pragma omp target teams' \
    "$REPO_ROOT/src" 2>/dev/null | \
    grep -v "is_device_ptr" | \
    grep -v "enter data\|exit data\|update" | \
    grep -v "// NO_POINTERS" || true)

if [ -n "$violations" ]; then
    # Check if these are actually using pointers
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        linenum=$(echo "$line" | cut -d: -f2)
        # Look at the next 10 lines for pointer usage
        context=$(sed -n "${linenum},$((linenum+10))p" "$file" 2>/dev/null || true)
        if echo "$context" | grep -q '_dev\[' && ! echo "$context" | grep -q 'is_device_ptr'; then
            echo -e "${YELLOW}WARNING: $file:$linenum - target region may use pointers without is_device_ptr${NC}"
            ((warnings++))
        fi
    done <<< "$violations"
fi
if [ $warnings -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
fi
echo ""

# Pattern 3: Local pointer aliases used in target regions (in GPU code paths only)
# e.g., double* u = velocity_u_ptr_; then u used in target
# Exclude #else blocks (CPU fallback) and lines with dev_ptr/omp_get_mapped_ptr
echo "Checking for local pointer aliases in GPU code paths..."
pattern3_matches=$(grep -rn --include="*.cpp" \
    'double\* [a-z_]* = [a-z_]*_ptr_' \
    "$REPO_ROOT/src" 2>/dev/null | \
    grep -v "dev_ptr\|omp_get_mapped_ptr\|// ALIAS_OK\|#else" || true)

if [ -n "$pattern3_matches" ]; then
    # Filter out lines that are clearly in CPU-only blocks
    real_issues=""
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        linenum=$(echo "$line" | cut -d: -f2)
        # Check if this line is inside an #else block (CPU path)
        # Look backwards for #ifdef USE_GPU_OFFLOAD ... #else
        context_before=$(head -n "$linenum" "$file" 2>/dev/null | tail -20 || true)
        if echo "$context_before" | grep -q "#else" && ! echo "$context_before" | grep -q "#endif"; then
            # Inside #else block - this is CPU path, skip
            continue
        fi
        real_issues="$real_issues\n$line"
    done <<< "$pattern3_matches"

    if [ -n "$real_issues" ] && [ "$real_issues" != "\n" ]; then
        echo -e "$real_issues" | grep -v "^$" | head -10
        echo -e "${YELLOW}WARNING: Found local pointer aliases - verify they use dev_ptr() before target regions${NC}"
        ((warnings++))
    else
        echo -e "${GREEN}OK${NC}"
    fi
else
    echo -e "${GREEN}OK${NC}"
fi
echo ""

# Pattern 4: map(present:) with local variables (not member/parameter pointers)
# Member pointers (*_ptr_) and parameter pointers (*_ptr) with map(present:) are safe
echo "Checking for map(present:) with local aliases..."
# Look for map(present: in actual pragmas (not comments)
pattern4_matches=$(grep -rn --include="*.cpp" \
    '#pragma.*map(present:' \
    "$REPO_ROOT/src" 2>/dev/null | \
    grep -v '_ptr' | \
    grep -v '// PTR_OK' || true)

if [ -n "$pattern4_matches" ]; then
    echo "$pattern4_matches" | head -5
    echo -e "${YELLOW}WARNING: Found map(present:) with possible local aliases${NC}"
    ((warnings++))
else
    echo -e "${GREEN}OK${NC}"
fi
echo ""

# Pattern 5: Verify dev_ptr() is used before is_device_ptr
echo "Checking that dev_ptr() precedes is_device_ptr usage..."
# This is informational - we expect to see dev_ptr calls
dev_ptr_count=$(grep -rc "gpu::dev_ptr\|dev_ptr(" "$REPO_ROOT/src" 2>/dev/null | \
    awk -F: '{sum+=$2} END {print sum}' || echo 0)
is_device_ptr_count=$(grep -rc "is_device_ptr" "$REPO_ROOT/src" 2>/dev/null | \
    awk -F: '{sum+=$2} END {print sum}' || echo 0)

echo "  dev_ptr() calls: $dev_ptr_count"
echo "  is_device_ptr clauses: $is_device_ptr_count"

# Some variance is expected (multiple is_device_ptr vars per kernel, etc.)
# Only warn if dev_ptr count is significantly lower (< 70% of is_device_ptr)
threshold=$((is_device_ptr_count * 70 / 100))
if [ "$dev_ptr_count" -lt "$threshold" ]; then
    echo -e "${YELLOW}WARNING: Significantly fewer dev_ptr() calls than is_device_ptr - some may use raw pointers${NC}"
    ((warnings++))
else
    echo -e "${GREEN}OK - dev_ptr() usage is consistent with is_device_ptr${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
if [ $errors -gt 0 ]; then
    echo -e "${RED}FAILED: $errors error(s), $warnings warning(s)${NC}"
    echo ""
    echo "The GPU pointer pattern must be:"
    echo "  1. Get device pointer: double* u_dev = gpu::dev_ptr(host_ptr);"
    echo "  2. Use in kernel:      #pragma omp target ... is_device_ptr(u_dev)"
    echo ""
    echo "Never pass raw mapped pointers directly into target regions."
    exit 1
elif [ $warnings -gt 0 ]; then
    echo -e "${YELLOW}PASSED with $warnings warning(s)${NC}"
    echo "Review warnings to ensure GPU pointer safety."
    echo ""
    echo "NOTE: Warnings are informational and don't block CI."
    echo "      Most are in debug code or CPU fallback paths."
    exit 0
else
    echo -e "${GREEN}PASSED: No GPU pointer safety issues found${NC}"
    exit 0
fi
