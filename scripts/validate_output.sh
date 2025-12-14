#!/bin/bash
# Validate CFD solver output files for physical validity
# Checks for NaN, Inf, empty files, and basic physical constraints

set -e

OUTPUT_DIR="${1:-output}"
VERBOSE="${2:-1}"

echo "========================================"
echo "Validating Output Files"
echo "========================================"
echo "Directory: ${OUTPUT_DIR}"
echo ""

FAILURES=0
CHECKS=0

# Function to check a VTK file
check_vtk_file() {
    local file=$1
    local filename=$(basename "$file")
    
    if [ ! -f "$file" ]; then
        echo "[FAIL] ${filename}: File not found"
        return 1
    fi
    
    # Check if file is empty
    if [ ! -s "$file" ]; then
        echo "[FAIL] ${filename}: File is empty"
        return 1
    fi
    
    # Check for NaN values (case insensitive)
    if grep -qi "nan" "$file"; then
        echo "[FAIL] ${filename}: Contains NaN values"
        if [ "$VERBOSE" = "1" ]; then
            echo "  First NaN occurrence:"
            grep -n -i "nan" "$file" | head -3 | sed 's/^/    /'
        fi
        return 1
    fi
    
    # Check for Inf values (case insensitive)
    if grep -qi "inf" "$file"; then
        echo "[FAIL] ${filename}: Contains Inf values"
        if [ "$VERBOSE" = "1" ]; then
            echo "  First Inf occurrence:"
            grep -n -i "inf" "$file" | head -3 | sed 's/^/    /'
        fi
        return 1
    fi
    
    # Check for very large values that might indicate numerical instability
    # Look for numbers with magnitude > 1e10
    if grep -qE '[0-9]\.[0-9]+e\+[0-9]{2,}' "$file"; then
        echo "[WARNING] ${filename}: Contains very large values (>1e10)"
        if [ "$VERBOSE" = "1" ]; then
            echo "  Sample large values:"
            grep -oE '[0-9]\.[0-9]+e\+[0-9]{2,}' "$file" | head -3 | sed 's/^/    /'
        fi
        # This is a warning, not a failure
    fi
    
    # Check that file has actual data (POINT_DATA or CELL_DATA)
    if ! grep -q "POINT_DATA\|CELL_DATA" "$file"; then
        echo "[FAIL] ${filename}: No data section found"
        return 1
    fi
    
    # Check for velocity field
    if grep -q "velocity" "$file"; then
        local n_velocities=$(grep -A 1000000 "velocity" "$file" | grep -E "^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?" | wc -l)
        if [ "$n_velocities" -lt 10 ]; then
            echo "[FAIL] ${filename}: Too few velocity values ($n_velocities)"
            return 1
        fi
    fi
    
    echo "[OK] ${filename}: Valid"
    return 0
}

# Function to check a data file (like velocity_profile.dat)
check_dat_file() {
    local file=$1
    local filename=$(basename "$file")
    
    if [ ! -f "$file" ]; then
        echo "[WARNING] ${filename}: File not found (may be optional)"
        return 0
    fi
    
    # Check if file is empty
    if [ ! -s "$file" ]; then
        echo "[FAIL] ${filename}: File is empty"
        return 1
    fi
    
    # Count data lines (excluding comments and empty lines)
    local n_lines=$(grep -v "^#" "$file" | grep -v "^$" | wc -l)
    if [ "$n_lines" -lt 2 ]; then
            echo "[FAIL] ${filename}: Too few data lines ($n_lines)"
        return 1
    fi
    
    # Check for NaN - NO NaNs allowed in valid output!
    if grep -qi "nan" "$file"; then
        echo "[FAIL] ${filename}: Contains NaN values"
        return 1
    fi
    
    # Check for Inf
    if grep -qi "inf" "$file"; then
        echo "[FAIL] ${filename}: Contains Inf values"
        return 1
    fi
    
    echo "[OK] ${filename}: Valid ($n_lines data lines)"
    return 0
}

# Check VTK files
echo "Checking VTK files..."
VTK_FILES=$(find "$OUTPUT_DIR" -name "*.vtk" 2>/dev/null)

if [ -z "$VTK_FILES" ]; then
    echo "[WARNING] No VTK files found in ${OUTPUT_DIR}"
else
    for file in $VTK_FILES; do
        CHECKS=$((CHECKS + 1))
        if ! check_vtk_file "$file"; then
            FAILURES=$((FAILURES + 1))
        fi
    done
fi

echo ""
echo "Checking data files..."

# Check common output files (these may be optional)
DAT_FILES=$(find "$OUTPUT_DIR" -name "*.dat" 2>/dev/null)

if [ -n "$DAT_FILES" ]; then
    for file in $DAT_FILES; do
        CHECKS=$((CHECKS + 1))
        if ! check_dat_file "$file"; then
            FAILURES=$((FAILURES + 1))
        fi
    done
else
    echo "  No .dat files found (may be expected)"
fi

echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"
echo "Checks performed: ${CHECKS}"
echo "Failures: ${FAILURES}"
echo ""

if [ $CHECKS -eq 0 ]; then
    echo "[WARNING] No files were validated!"
    echo "  This may indicate the output directory is empty or misconfigured."
    exit 1
fi

if [ $FAILURES -eq 0 ]; then
    echo "[PASS] All validations passed!"
    exit 0
else
    echo "[FAIL] Validation failed with ${FAILURES} error(s)"
    exit 1
fi

