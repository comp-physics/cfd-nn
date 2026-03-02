#!/usr/bin/env bash
# Parse Tier 2 validation results from solver logs
# Usage: bash scripts/parse_validation_results.sh <output_dir>
set -euo pipefail

OUT="${1:?Usage: $0 <output_dir>}"
PASS=0
FAIL=0
SKIP=0

check() {
    local name="$1" condition="$2"
    if eval "$condition"; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    echo "  [SKIP] $1"
    SKIP=$((SKIP + 1))
}

echo "================================================================"
echo "  Tier 2 Validation Report"
echo "  Output: $OUT"
echo "================================================================"
echo ""

# ---- 1. DNS Channel ----
echo "--- 1. DNS Channel Re_tau~180 ---"
if [ -f "$OUT/dns_channel.log" ]; then
    re_tau=$(grep -oP 'Re_tau: \K[0-9.]+' "$OUT/dns_channel.log" | tail -1)
    u_tau=$(grep -oP 'Friction velocity u_tau: \K[0-9.]+' "$OUT/dns_channel.log" | tail -1)
    bulk_u=$(grep -oP 'Bulk velocity: \K[0-9.]+' "$OUT/dns_channel.log" | tail -1)
    converged=$(grep -c "TURBULENT" "$OUT/dns_channel.log" || true)
    mom_balance=$(grep -oP 'max residual.*: \K[0-9.]+' "$OUT/dns_channel.log" | tail -1)
    has_nan=$(grep -c "NaN\|Inf\|SAFETY-VEL" "$OUT/dns_channel.log" || true)

    echo "  Re_tau = ${re_tau:-N/A}"
    echo "  u_tau  = ${u_tau:-N/A}"
    echo "  U_bulk = ${bulk_u:-N/A}"
    echo "  Turbulent steps: $converged"
    echo "  Momentum balance residual: ${mom_balance:-N/A}%"
    echo ""

    check "No NaN/Inf/safety triggers" "[ '${has_nan:-0}' -eq 0 ]"
    check "Re_tau > 100" "[ -n '${re_tau:-}' ] && python3 -c 'exit(0 if ${re_tau} > 100 else 1)' 2>/dev/null"
    check "Re_tau < 300 (not overdriven)" "[ -n '${re_tau:-}' ] && python3 -c 'exit(0 if ${re_tau} < 300 else 1)' 2>/dev/null"
    check "Reached turbulent state" "[ '${converged:-0}' -gt 10 ]"
    if [ -n "${mom_balance:-}" ]; then
        check "Momentum balance < 10%" "python3 -c 'exit(0 if ${mom_balance} < 10 else 1)' 2>/dev/null"
    else
        skip "Momentum balance (stats may not have accumulated)"
    fi
else
    skip "DNS channel (log not found)"
fi
echo ""

# ---- 2. RANS Models ----
echo "--- 2. RANS Model Sweep ---"
for model in none baseline gep sst komega earsm_wj earsm_gs earsm_pope nn_mlp nn_tbnn; do
    log="$OUT/rans_${model}.log"
    if [ -f "$log" ]; then
        re_tau=$(grep -oP 'Re_tau: \K[0-9.]+' "$log" | tail -1)
        converged=$(grep -c "Converged: YES" "$log" || true)
        has_error=$(grep -c "SAFETY-VEL\|NaN\|Inf\|abort\|SIGABRT" "$log" || true)
        l2_err=$(grep -oP 'L2 error = \K[0-9.]+' "$log" | tail -1)

        echo "  [$model] Re_tau=${re_tau:-N/A} converged=${converged} L2=${l2_err:-N/A}%"
        check "[$model] No crash/NaN" "[ '${has_error:-0}' -eq 0 ]"
        if [ "$model" != "none" ]; then
            check "[$model] Re_tau > 50" "[ -n '${re_tau:-}' ] && python3 -c 'exit(0 if ${re_tau} > 50 else 1)' 2>/dev/null"
        fi
    else
        skip "[$model] (log not found)"
    fi
done
echo ""

# ---- 3. TGV Re=1600 ----
echo "--- 3. TGV Re=1600 ---"
if [ -f "$OUT/tgv_re1600.log" ]; then
    has_nan=$(grep -c "NaN\|Inf" "$OUT/tgv_re1600.log" || true)
    final_ke=$(grep -oP 'Kinetic energy: \K[0-9.e+-]+' "$OUT/tgv_re1600.log" | tail -1)

    echo "  Final KE: ${final_ke:-N/A}"
    check "TGV No NaN/Inf" "[ '${has_nan:-0}' -eq 0 ]"
    check "TGV completed 10000 steps" "grep -q 'Step.*10000\|Unsteady simulation complete' '$OUT/tgv_re1600.log'"
else
    skip "TGV Re=1600 (log not found)"
fi
echo ""

# ---- 4. Poiseuille ----
echo "--- 4. Poiseuille Grid Convergence ---"
declare -a pois_errors
for ny in 32 64 128 256; do
    log="$OUT/poiseuille_${ny}.log"
    if [ -f "$log" ]; then
        l2=$(grep -oP 'L2 error = \K[0-9.]+' "$log" | tail -1)
        converged=$(grep -c "Converged: YES" "$log" || true)
        passed=$(grep -c "VALIDATION PASSED" "$log" || true)
        echo "  Ny=$ny: L2=${l2:-N/A}% converged=$converged"
        check "Poiseuille Ny=$ny converged" "[ '${converged:-0}' -gt 0 ]"
        if [ -n "${l2:-}" ]; then
            check "Poiseuille Ny=$ny L2 < 5%" "python3 -c 'exit(0 if ${l2} < 5 else 1)' 2>/dev/null"
            pois_errors+=("$l2")
        fi
    else
        skip "Poiseuille Ny=$ny (log not found)"
    fi
done

# Check grid convergence rate if we have at least 2 errors
if [ ${#pois_errors[@]} -ge 2 ]; then
    e1=${pois_errors[0]}
    e2=${pois_errors[1]}
    rate=$(python3 -c "import math; print(f'{math.log($e1/$e2)/math.log(2):.2f}')" 2>/dev/null || echo "N/A")
    echo "  Convergence rate (Ny=32→64): $rate"
    check "Grid convergence rate > 1.5" "python3 -c 'exit(0 if $rate > 1.5 else 1)' 2>/dev/null"
fi
echo ""

# ---- Summary ----
echo "================================================================"
echo "  SUMMARY: $PASS passed, $FAIL failed, $SKIP skipped"
echo "================================================================"
if [ "$FAIL" -gt 0 ]; then
    echo "  STATUS: FAIL"
    exit 1
else
    echo "  STATUS: PASS"
fi
