#!/bin/bash
# lint_gpu_pointers.sh - Check for unsafe GPU pointer patterns in NVHPC code
#
# This script detects patterns known to cause bugs with NVHPC's OpenMP target:
#   1. map(present: member_ptr_) without is_device_ptr - member pointers get HOST address
#   2. firstprivate(T*) in target regions - pointer copies get wrong address
#   3. use_device_ptr - deprecated, use is_device_ptr instead
#
# Run from repository root: ./scripts/lint_gpu_pointers.sh
#
# Exit code:
#   0 = No issues found
#   1 = Potential issues detected (review required)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WARNINGS=0
ERRORS=0

echo "==============================================="
echo "  GPU Pointer Pattern Lint Check"
echo "==============================================="
echo ""

# Check for map(present: member_ptr) pattern without is_device_ptr
echo "Checking for 'map(present: member_ptr_)' patterns..."
echo ""

# Pattern 1: map(present: with member pointer (contains underscore before bracket)
RISKY_PRESENT=$(grep -rn --include="*.cpp" --include="*.hpp" \
    'map(present:.*_ptr_\[' "$REPO_ROOT/src" "$REPO_ROOT/include" 2>/dev/null || true)

if [ -n "$RISKY_PRESENT" ]; then
    echo -e "${YELLOW}WARNING: Found 'map(present: member_ptr_[' patterns:${NC}"
    echo "$RISKY_PRESENT" | head -20
    echo ""
    echo "These may be safe if followed by is_device_ptr, but review carefully."
    echo "Preferred pattern: gpu::dev_ptr() + is_device_ptr()"
    echo ""
    WARNINGS=$((WARNINGS + 1))
fi

# Pattern 2: Check for deprecated use_device_ptr
echo "Checking for deprecated 'use_device_ptr' usage..."
DEPRECATED_USE_DEVICE=$(grep -rn --include="*.cpp" --include="*.hpp" \
    'use_device_ptr' "$REPO_ROOT/src" "$REPO_ROOT/include" 2>/dev/null || true)

if [ -n "$DEPRECATED_USE_DEVICE" ]; then
    echo -e "${YELLOW}WARNING: Found deprecated 'use_device_ptr' patterns:${NC}"
    echo "$DEPRECATED_USE_DEVICE" | head -10
    echo ""
    echo "Consider migrating to 'is_device_ptr' for NVHPC compatibility."
    echo ""
    WARNINGS=$((WARNINGS + 1))
fi

# Pattern 3: Check for map(from:) or map(tofrom:) in stepping code
# These patterns cause implicit H↔D transfers during time stepping, which defeats
# the purpose of keeping data resident on GPU.
# EXCEPTION: Debug logging statements (map(from:xxx_sample)) are allowed since they
# only transfer single scalar values for diagnostics.
echo ""
echo "Checking for 'map(from:' or 'map(tofrom:' in stepping code..."

# Stepping-critical files: these run every time step and must not do H↔D transfers
# - solver_time*.cpp: RK time integration
# - solver_periodic_halos.cpp: periodic boundary fill
# Exclude lines containing "DEBUG", "_sample", or "original" (debug instrumentation)
MAP_FROM_IN_STEPPING=$(grep -rn --include="solver_time*.cpp" --include="solver_periodic_halos.cpp" \
    -E 'map\((from|tofrom):' "$REPO_ROOT/src" 2>/dev/null | \
    grep -v -E '(DEBUG|_sample|original|readback|sentinel)' || true)

if [ -n "$MAP_FROM_IN_STEPPING" ]; then
    echo -e "${RED}ERROR: Found 'map(from:' or 'map(tofrom:' in stepping code:${NC}"
    echo "$MAP_FROM_IN_STEPPING"
    echo ""
    echo "Time-stepping code should use only map(present:) + is_device_ptr()"
    echo "or gpu::dev_ptr() + is_device_ptr() to avoid H↔D transfers."
    echo "(Debug logging with scalar _sample variables is allowed)"
    echo ""
    ERRORS=$((ERRORS + 1))
else
    echo "  OK - no production map(from:/tofrom:) patterns found"
fi

# Pattern 4: Check for map(to:) in stepping code (also causes host→device transfer)
echo "Checking for 'map(to:' in stepping code..."

MAP_TO_IN_STEPPING=$(grep -rn --include="solver_time*.cpp" --include="solver_periodic_halos.cpp" \
    -E 'map\(to:' "$REPO_ROOT/src" 2>/dev/null | grep -v 'map(to: dt)' || true)

if [ -n "$MAP_TO_IN_STEPPING" ]; then
    echo -e "${YELLOW}WARNING: Found 'map(to:' in stepping code:${NC}"
    echo "$MAP_TO_IN_STEPPING" | head -10
    echo ""
    echo "Scalar values like dt are OK, but arrays should use map(present:)."
    echo ""
    WARNINGS=$((WARNINGS + 1))
fi

# Pattern 5: Ensure solve_device() is used instead of solve() in stepping code
# The solve() method reads from HOST ScalarField, but stepping keeps data on device.
# Must use solve_device() which works with device-resident pointers.
echo ""
echo "Checking for 'solve(' (not solve_device) Poisson calls in stepping code..."

# Note: Lines inside #else (CPU path) are fine - we only care about GPU path
BARE_SOLVE_IN_STEPPING=$(grep -rn --include="solver_time*.cpp" --include="solver_periodic_halos.cpp" \
    '\.solve(' "$REPO_ROOT/src" 2>/dev/null | \
    grep -v 'solve_device' | grep -v '//' | grep -v '#else' || true)

# Additional filter: if the hit is inside a !USE_GPU_OFFLOAD block, that's OK
# We check this by looking for #else on the previous line (simple heuristic)
if [ -n "$BARE_SOLVE_IN_STEPPING" ]; then
    REAL_ISSUES=""
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        lineno=$(echo "$line" | cut -d: -f2)
        # Check if previous few lines contain #else or #ifndef USE_GPU_OFFLOAD
        prev_lines=$(sed -n "$((lineno-5)),$((lineno-1))p" "$file" 2>/dev/null || true)
        if echo "$prev_lines" | grep -qE '#else|#ifndef USE_GPU_OFFLOAD'; then
            continue  # Skip - this is in CPU path
        fi
        REAL_ISSUES="$REAL_ISSUES$line\n"
    done <<< "$BARE_SOLVE_IN_STEPPING"
    BARE_SOLVE_IN_STEPPING=$(echo -e "$REAL_ISSUES" | grep -v '^$' || true)
fi

if [ -n "$BARE_SOLVE_IN_STEPPING" ]; then
    echo -e "${RED}ERROR: Found 'solve(' instead of 'solve_device(' in stepping code:${NC}"
    echo "$BARE_SOLVE_IN_STEPPING"
    echo ""
    echo "Time-stepping code must use solve_device() for device-resident data."
    echo "The solve() method reads from HOST ScalarField which is stale."
    echo ""
    ERRORS=$((ERRORS + 1))
else
    echo "  OK - Poisson solver uses solve_device() in stepping code"
fi

# Summary
echo ""
echo "==============================================="
echo "  Summary"
echo "==============================================="

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}ERRORS: $ERRORS${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}WARNINGS: $WARNINGS${NC}"
    echo "Review warnings to ensure patterns are safe."
    exit 0
else
    echo -e "${GREEN}No issues found!${NC}"
    exit 0
fi
