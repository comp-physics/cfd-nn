#!/bin/bash
# check_symbol_uniqueness.sh - Verify no duplicate RK function definitions
#
# This script ensures the RK time stepping functions are defined exactly once.
# If duplicates are found, it means solver.cpp accidentally contains copies
# of the functions that should only exist in solver_time.cpp.
#
# Usage: ./scripts/check_symbol_uniqueness.sh <path-to-library>
# Example: ./scripts/check_symbol_uniqueness.sh build-gpu/libnn_cfd_core.a

set -e

LIB="${1:-build-gpu/libnn_cfd_core.a}"

if [[ ! -f "$LIB" ]]; then
    echo "ERROR: Library not found: $LIB"
    exit 1
fi

echo "Checking symbol uniqueness in: $LIB"
echo "============================================"

CRITICAL_FUNCS=(
    "RANSSolver::euler_substep"
    "RANSSolver::project_velocity"
    "RANSSolver::ssprk2_step"
    "RANSSolver::ssprk3_step"
)

ERRORS=0

for func in "${CRITICAL_FUNCS[@]}"; do
    # Count defined symbols (T = text section = function definition)
    count=$(nm -C "$LIB" 2>/dev/null | grep "$func" | grep " T " | wc -l)

    if [[ $count -eq 0 ]]; then
        echo "ERROR: $func - NOT DEFINED (missing implementation)"
        ERRORS=$((ERRORS + 1))
    elif [[ $count -eq 1 ]]; then
        echo "  OK: $func - exactly 1 definition"
    else
        echo "ERROR: $func - $count definitions (DUPLICATE!)"
        echo "  Definitions found:"
        nm -C "$LIB" 2>/dev/null | grep "$func" | grep " T " | sed 's/^/    /'
        ERRORS=$((ERRORS + 1))
    fi
done

echo "============================================"

if [[ $ERRORS -gt 0 ]]; then
    echo "FAILED: $ERRORS symbol uniqueness violations"
    echo ""
    echo "If you see DUPLICATE errors, check that solver.cpp does NOT contain"
    echo "implementations of euler_substep, project_velocity, ssprk2_step, or"
    echo "ssprk3_step. These functions must only be defined in solver_time.cpp."
    exit 1
else
    echo "PASSED: All critical RK functions have exactly one definition"
    exit 0
fi
