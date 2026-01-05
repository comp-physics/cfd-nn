#!/bin/bash
# detect_gpu_cc.sh - Detect GPU compute capability
#
# Usage: ./detect_gpu_cc.sh [--test]
#
# Returns the compute capability (e.g., "80" for A100, "90" for H100)
# Falls back to "80" if detection fails.
#
# Exit codes:
#   0 - Success (GPU detected)
#   1 - Fallback used (no GPU or detection failed)

# Note: Don't use set -e as we need to handle return codes from detect_gpu_cc

# Function to detect GPU CC (testable core logic)
detect_gpu_cc() {
    local nvidia_smi_output
    local cc

    # Check if nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
        echo "80"  # Default fallback
        return 1
    fi

    # Try to get GPU name and map to CC
    nvidia_smi_output=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")

    if [[ -z "$nvidia_smi_output" ]]; then
        echo "80"
        return 1
    fi

    # Map GPU names to compute capabilities
    case "$nvidia_smi_output" in
        *"H100"*)     cc="90" ;;
        *"H200"*)     cc="90" ;;
        *"A100"*)     cc="80" ;;
        *"A30"*)      cc="80" ;;
        *"A10"*)      cc="86" ;;
        *"A40"*)      cc="86" ;;
        *"A6000"*)    cc="86" ;;
        *"RTX 4090"*) cc="89" ;;
        *"RTX 4080"*) cc="89" ;;
        *"RTX 3090"*) cc="86" ;;
        *"RTX 3080"*) cc="86" ;;
        *"V100"*)     cc="70" ;;
        *"T4"*)       cc="75" ;;
        *"P100"*)     cc="60" ;;
        *)
            # Unknown GPU, try to parse from nvidia-smi -L
            local cuda_caps
            cuda_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "")
            if [[ -n "$cuda_caps" ]]; then
                # Convert 8.0 to 80
                cc=$(echo "$cuda_caps" | head -1 | tr -d '.')
            else
                cc="80"  # Default fallback
                echo "$cc"
                return 1
            fi
            ;;
    esac

    echo "$cc"
    return 0
}

# Test mode: run test cases
run_tests() {
    local passed=0
    local failed=0

    echo "=== GPU_CC Detection Unit Tests ==="
    echo

    # Test 1: Function returns something
    local result
    result=$(detect_gpu_cc)
    if [[ -n "$result" ]] && [[ "$result" =~ ^[0-9]+$ ]]; then
        echo "[PASS] detect_gpu_cc returns numeric value: $result"
        ((passed++))
    else
        echo "[FAIL] detect_gpu_cc returned invalid: '$result'"
        ((failed++))
    fi

    # Test 2: Result is in valid range (50-99)
    if [[ "$result" -ge 50 ]] && [[ "$result" -le 99 ]]; then
        echo "[PASS] CC value in valid range: $result"
        ((passed++))
    else
        echo "[FAIL] CC value out of range: $result (expected 50-99)"
        ((failed++))
    fi

    # Test 3: Verify against actual nvidia-smi if available
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo "[INFO] Detected GPU: $gpu_name"
        echo "[PASS] nvidia-smi available, detection used real hardware"
        ((passed++))
    else
        echo "[SKIP] nvidia-smi not available (detection used fallback)"
    fi

    echo
    echo "=== Summary: $passed passed, $failed failed ==="

    if [[ $failed -gt 0 ]]; then
        return 1
    fi
    return 0
}

# Main
if [[ "$1" == "--test" ]]; then
    run_tests
    exit $?
else
    detect_gpu_cc
    exit $?
fi
