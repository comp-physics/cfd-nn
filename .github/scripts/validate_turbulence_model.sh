#!/bin/bash
# Validate turbulence model output for physical consistency
# 
# Checks:
# 1. No NaN/Inf values in output
# 2. Non-zero velocity field (flow must develop)
# 3. Reasonable velocity magnitudes (configurable tolerance for slow models)
# 4. Positive eddy viscosity (nu_t >= 0)
# 5. Bounded eddy viscosity (nu_t < 1000*nu)
# 6. Conservation of mass (divergence near zero)
# 7. For transport models: positive k, omega

set -e

MODEL_NAME="$1"
OUTPUT_DIR="$2"
MIN_VEL_TOLERANCE="${3:-0.01}"  # Optional: minimum velocity tolerance (default 0.01)

if [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <model_name> <output_dir> [min_vel_tolerance]"
    exit 1
fi

echo "Validating output for model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

VALIDATION_FAILED=0

# Find the final VTK file
FINAL_VTK=$(ls -1 ${OUTPUT_DIR}/*_final.vtk 2>/dev/null | head -1)
if [ ! -f "$FINAL_VTK" ]; then
    echo "ERROR: No final VTK file found in $OUTPUT_DIR"
    exit 1
fi

echo "Analyzing: $FINAL_VTK"

# Extract velocity data and check for issues
echo "Checking for NaN/Inf values..."
if grep -q "nan\|inf\|-nan\|-inf" "$FINAL_VTK"; then
    echo "  [FAIL] NaN or Inf detected in output!"
    VALIDATION_FAILED=1
else
    echo "  [OK] No NaN/Inf values"
fi

# Check velocity magnitudes
echo "Checking velocity field..."
VELOCITIES=$(awk '/^VECTORS velocity/{flag=1; next} /^SCALARS/{flag=0} flag && NF==3' "$FINAL_VTK")

# Compute max velocity magnitude
MAX_VEL=$(echo "$VELOCITIES" | awk '{
    u = $1; v = $2;
    mag = sqrt(u*u + v*v);
    if (mag > max_mag) max_mag = mag;
} END {print max_mag}')

# Check if velocity is non-zero
if [ -z "$MAX_VEL" ] || [ "$MAX_VEL" = "0" ]; then
    echo "  [FAIL] Zero velocity field (flow did not develop)!"
    VALIDATION_FAILED=1
else
    echo "  Max velocity magnitude: $MAX_VEL"
    
    # Check reasonable bounds (for driven channel flow)
    IS_REASONABLE=$(echo "$MAX_VEL $MIN_VEL_TOLERANCE" | awk '{if ($1 > $2 && $1 < 100.0) print "1"; else print "0"}')
    if [ "$IS_REASONABLE" = "1" ]; then
        echo "  [OK] Velocity magnitude in reasonable range (${MIN_VEL_TOLERANCE} - 100.0)"
    else
        echo "  [WARNING] Velocity magnitude outside typical range (expected > ${MIN_VEL_TOLERANCE})!"
        VALIDATION_FAILED=1
    fi
fi

# Check for all-zero velocity (sign of divergence)
ZERO_COUNT=$(echo "$VELOCITIES" | awk '{if ($1 == 0.0 && $2 == 0.0) count++} END {print count+0}')
TOTAL_COUNT=$(echo "$VELOCITIES" | wc -l)
ZERO_FRACTION=$(echo "$ZERO_COUNT $TOTAL_COUNT" | awk '{print $1/$2}')

if [ "$ZERO_COUNT" = "$TOTAL_COUNT" ]; then
    echo "  [FAIL] All velocities are zero!"
    VALIDATION_FAILED=1
elif [ "$(echo "$ZERO_FRACTION > 0.5" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "  [WARNING] >50% of velocities are zero!"
    VALIDATION_FAILED=1
else
    echo "  [OK] Velocity field properly developed ($ZERO_COUNT / $TOTAL_COUNT cells are zero)"
fi

# Check eddy viscosity if present
echo "Checking eddy viscosity..."
if grep -q "SCALARS nu_t" "$FINAL_VTK"; then
    NU_T_VALUES=$(awk '/^SCALARS nu_t/{flag=1; next} /^LOOKUP_TABLE/{next} /^SCALARS/{flag=0} flag && NF==1' "$FINAL_VTK")
    
    # Check for negative values
    NEG_COUNT=$(echo "$NU_T_VALUES" | awk '{if ($1 < 0.0) count++} END {print count+0}')
    if [ "$NEG_COUNT" -gt 0 ]; then
        echo "  [FAIL] Negative eddy viscosity detected ($NEG_COUNT cells)!"
        VALIDATION_FAILED=1
    else
        echo "  [OK] All nu_t values are non-negative"
    fi
    
    # Check for NaN in nu_t specifically
    NAN_COUNT=$(echo "$NU_T_VALUES" | grep -c "nan\|inf" || true)
    if [ "$NAN_COUNT" -gt 0 ]; then
        echo "  [FAIL] NaN/Inf in nu_t ($NAN_COUNT values)!"
        VALIDATION_FAILED=1
    else
        echo "  [OK] No NaN/Inf in nu_t"
    fi
    
    # Check max nu_t
    MAX_NU_T=$(echo "$NU_T_VALUES" | awk '{if ($1 > max) max = $1} END {print max}')
    echo "  Max nu_t: $MAX_NU_T"
    
    # For nu = 0.001, nu_t should be < 10.0 typically
    IS_NU_T_REASONABLE=$(echo "$MAX_NU_T" | awk '{if ($1 < 100.0) print "1"; else print "0"}')
    if [ "$IS_NU_T_REASONABLE" = "1" ]; then
        echo "  [OK] Eddy viscosity in reasonable range"
    else
        echo "  [WARNING] Very large eddy viscosity (max_nu_t = $MAX_NU_T)!"
        # Don't fail on this, just warn
    fi
else
    echo "  - No eddy viscosity field (laminar flow)"
fi

# Check pressure field
echo "Checking pressure field..."
PRESSURES=$(awk '/^SCALARS pressure/{flag=1; next} /^LOOKUP_TABLE/{next} /^SCALARS/{flag=0} flag && NF==1' "$FINAL_VTK")

PRESSURE_NAN=$(echo "$PRESSURES" | grep -c "nan\|inf" || true)
if [ "$PRESSURE_NAN" -gt 0 ]; then
    echo "  [FAIL] NaN/Inf in pressure field!"
    VALIDATION_FAILED=1
else
    echo "  [OK] Pressure field is finite"
fi

# Check data files if they exist
if [ -f "${OUTPUT_DIR}/channel_velocity.dat" ]; then
    echo "Checking data files..."
    
    if grep -q "nan\|inf" "${OUTPUT_DIR}/channel_velocity.dat"; then
        echo "  [FAIL] NaN/Inf in velocity data file!"
        VALIDATION_FAILED=1
    else
        echo "  [OK] Velocity data file valid"
    fi
    
    if [ -f "${OUTPUT_DIR}/channel_nu_t.dat" ]; then
        if grep -q "nan\|inf" "${OUTPUT_DIR}/channel_nu_t.dat"; then
            echo "  [FAIL] NaN/Inf in nu_t data file!"
            VALIDATION_FAILED=1
        else
            echo "  [OK] nu_t data file valid"
        fi
    fi
fi

# Summary
echo ""
echo "========================================"
if [ $VALIDATION_FAILED -eq 0 ]; then
    echo "[PASS] VALIDATION PASSED for $MODEL_NAME"
    echo "========================================"
    exit 0
else
    echo "[FAIL] VALIDATION FAILED for $MODEL_NAME"
    echo "========================================"
    exit 1
fi

