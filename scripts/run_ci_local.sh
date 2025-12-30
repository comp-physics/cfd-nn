#!/bin/bash
# run_ci_local.sh - Run CI tests locally
#
# This script runs the same tests that would run in CI, allowing developers
# to verify their changes before pushing.
#
# Usage:
#   ./scripts/run_ci_local.sh              # Run all tests (fast + medium + golden), auto-detect GPU
#   ./scripts/run_ci_local.sh fast         # Run only fast tests (~1 minute)
#   ./scripts/run_ci_local.sh golden       # Run golden file regression tests only
#   ./scripts/run_ci_local.sh full         # Run all tests including slow ones
#   ./scripts/run_ci_local.sh gpu          # Run GPU-specific tests only
#   ./scripts/run_ci_local.sh paradigm     # Run code sharing paradigm checks
#   ./scripts/run_ci_local.sh --cpu        # Force CPU-only build (no GPU offload)
#   ./scripts/run_ci_local.sh --cpu fast   # CPU-only with fast tests
#   ./scripts/run_ci_local.sh -v           # Verbose output (show full test output)
#   ./scripts/run_ci_local.sh --verbose    # Same as -v

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# Parse flags
USE_GPU=ON
VERBOSE=0
while [[ "$1" == --* || "$1" == -* ]]; do
    case "$1" in
        --cpu)
            USE_GPU=OFF
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set build directory based on mode
if [[ "$USE_GPU" == "ON" ]]; then
    BUILD_DIR="${PROJECT_DIR}/build_gpu"
else
    BUILD_DIR="${PROJECT_DIR}/build_cpu"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timer
SECONDS=0

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Parse arguments
TEST_SUITE="${1:-all}"

# Track results
PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=""

# Known flaky tests on GPU (pre-existing issues, not related to 3D work)
# These will be skipped when USE_GPU=ON until root causes are addressed.
# Note: test_solver and test_physics_validation were slow (not flaky) - fixed by increasing timeouts
# Note: test_turbulence_guard was flaky - fixed by calling check_for_nan_inf directly instead of step()
GPU_FLAKY_TESTS=""

is_gpu_flaky() {
    local test_binary=$1
    local test_name
    test_name=$(basename "$test_binary")

    if [[ "$USE_GPU" == "ON" ]]; then
        for flaky in $GPU_FLAKY_TESTS; do
            if [[ "$test_name" == "$flaky" ]]; then
                return 0  # true - is flaky
            fi
        done
    fi
    return 1  # false - not flaky
}

run_test() {
    local test_name=$1
    local test_binary=$2
    local timeout_secs=${3:-120}

    if [ ! -f "$test_binary" ]; then
        log_skip "$test_name (not built)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Skip known flaky GPU tests
    if is_gpu_flaky "$test_binary"; then
        log_skip "$test_name (known GPU flaky - see GPU_FLAKY_TESTS)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo ""
    log_info "Running $test_name..."

    local output_file="/tmp/test_output_$$.txt"
    local exit_code=0
    timeout "$timeout_secs" "$test_binary" > "$output_file" 2>&1 || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "$test_name"
        PASSED=$((PASSED + 1))
        if [ $VERBOSE -eq 1 ]; then
            echo "  Output:"
            cat "$output_file" | sed 's/^/    /'
        else
            # Show summary lines (PASSED/FAILED counts, key results, metrics)
            # Patterns: [PASS], [FAIL], [OK], PASSED, FAILED, Results:, ===...===,
            #           max_diff=, max_div=, L2/Linf norms, Test N:, scientific notation
            local summary
            summary=$(grep -E '(\[PASS\]|\[FAIL\]|\[OK\]|\[SUCCESS\]|PASSED|FAILED|passed|failed|Results:|===.*===|error=|Error|SUCCESS|max_diff|max_div|L2|Linf|Test [0-9]+:|[0-9]+\.[0-9]+e[-+]?[0-9]+)' "$output_file" | head -15) || true
            if [ -n "$summary" ]; then
                echo "$summary" | sed 's/^/    /'
            fi
        fi
    else
        log_failure "$test_name (exit code: $exit_code)"
        echo "  Output (last 30 lines):"
        tail -30 "$output_file" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name"
    fi
    rm -f "$output_file"
}

# Run a golden file regression test
# These tests take a case name as argument to the test_golden binary
run_golden_test() {
    local test_name=$1
    local case_name=$2
    local timeout_secs=${3:-60}

    local test_binary="${BUILD_DIR}/test_golden"

    if [ ! -f "$test_binary" ]; then
        log_skip "$test_name (test_golden not built)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo ""
    log_info "Running $test_name..."

    local output_file="/tmp/test_output_$$.txt"
    local exit_code=0
    timeout "$timeout_secs" "$test_binary" "$case_name" > "$output_file" 2>&1 || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Check if test was skipped (NN weights not found)
        if grep -q "\[SKIP\]" "$output_file"; then
            log_skip "$test_name (NN weights not available)"
            SKIPPED=$((SKIPPED + 1))
        else
            log_success "$test_name"
            PASSED=$((PASSED + 1))
        fi
        if [ $VERBOSE -eq 1 ]; then
            echo "  Output:"
            cat "$output_file" | sed 's/^/    /'
        else
            # Show max_diff lines for golden tests
            local summary
            summary=$(grep -E '(max_diff|SUCCESS|FAILURE|\[OK\]|\[FAIL\]|\[SKIP\])' "$output_file" | head -10) || true
            if [ -n "$summary" ]; then
                echo "$summary" | sed 's/^/    /'
            fi
        fi
    else
        log_failure "$test_name (exit code: $exit_code)"
        echo "  Output (last 30 lines):"
        tail -30 "$output_file" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name"
    fi
    rm -f "$output_file"
}

# Run a CPU/GPU cross-build comparison test
# These tests require both CPU and GPU builds and compare outputs between them
run_cross_build_test() {
    local test_name=$1
    local test_binary_name=$2
    local timeout_secs=${3:-120}
    local ref_prefix=$4

    local cpu_binary="${PROJECT_DIR}/build_cpu/${test_binary_name}"
    local gpu_binary="${PROJECT_DIR}/build_gpu/${test_binary_name}"

    # In GPU mode, compare GPU build against CPU reference
    if [[ "$USE_GPU" == "ON" ]]; then
        if [ ! -f "$gpu_binary" ]; then
            log_skip "$test_name (GPU binary not built)"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi

        # Check if CPU build exists for reference generation
        if [ ! -f "$cpu_binary" ]; then
            log_skip "$test_name (CPU build not available for reference)"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi

        echo ""
        log_info "Running $test_name (cross-build comparison)..."

        # Generate CPU reference if it doesn't exist
        local ref_dir="${PROJECT_DIR}/build_gpu/cpu_reference"
        mkdir -p "$ref_dir"

        local output_file="/tmp/test_output_$$.txt"

        # Check if reference already exists
        if [ ! -f "${ref_dir}/${ref_prefix}_u.dat" ] && [ ! -f "${ref_dir}/${ref_prefix}_pressure.dat" ]; then
            log_info "  Generating CPU reference..."
            timeout "$timeout_secs" "$cpu_binary" --dump-prefix "${ref_dir}/${ref_prefix}" > "$output_file" 2>&1 || {
                log_failure "$test_name (CPU reference generation failed)"
                tail -20 "$output_file" | sed 's/^/    /'
                FAILED=$((FAILED + 1))
                FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (CPU ref)"
                rm -f "$output_file"
                return 0
            }
        fi

        # Run GPU comparison
        log_info "  Running GPU and comparing against CPU reference..."
        local exit_code=0
        timeout "$timeout_secs" "$gpu_binary" --compare-prefix "${ref_dir}/${ref_prefix}" > "$output_file" 2>&1 || exit_code=$?

        if [ $exit_code -eq 0 ]; then
            log_success "$test_name"
            PASSED=$((PASSED + 1))
            if [ $VERBOSE -eq 1 ]; then
                echo "  Output:"
                cat "$output_file" | sed 's/^/    /'
            else
                local summary
                summary=$(grep -E '(\[PASS\]|\[FAIL\]|\[OK\]|\[SUCCESS\]|\[WARN\]|PASSED|FAILED|Max abs diff|Max rel diff|RMS diff)' "$output_file" | head -10) || true
                if [ -n "$summary" ]; then
                    echo "$summary" | sed 's/^/    /'
                fi
            fi
        else
            log_failure "$test_name (exit code: $exit_code)"
            echo "  Output (last 30 lines):"
            tail -30 "$output_file" | sed 's/^/    /'
            FAILED=$((FAILED + 1))
            FAILED_TESTS="${FAILED_TESTS}\n  - $test_name"
        fi
        rm -f "$output_file"
    else
        # In CPU mode, just generate reference (useful for pre-generating)
        if [ ! -f "$cpu_binary" ]; then
            log_skip "$test_name (CPU binary not built)"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi

        echo ""
        log_info "Running $test_name (CPU reference generation)..."

        local ref_dir="${PROJECT_DIR}/build_cpu/cpu_reference"
        mkdir -p "$ref_dir"

        local output_file="/tmp/test_output_$$.txt"
        local exit_code=0
        timeout "$timeout_secs" "$cpu_binary" --dump-prefix "${ref_dir}/${ref_prefix}" > "$output_file" 2>&1 || exit_code=$?

        if [ $exit_code -eq 0 ]; then
            log_success "$test_name (reference generated)"
            PASSED=$((PASSED + 1))
            if [ $VERBOSE -eq 1 ]; then
                cat "$output_file" | sed 's/^/    /'
            fi
        else
            log_failure "$test_name (exit code: $exit_code)"
            tail -20 "$output_file" | sed 's/^/    /'
            FAILED=$((FAILED + 1))
            FAILED_TESTS="${FAILED_TESTS}\n  - $test_name"
        fi
        rm -f "$output_file"
    fi
}

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    log_info "Build directory not found. Creating and building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DUSE_GPU_OFFLOAD=${USE_GPU} -DBUILD_TESTS=ON
    make -j"$(nproc)"
    cd "$PROJECT_DIR"
fi

# Display mode
if [[ "$USE_GPU" == "ON" ]]; then
    log_info "Running in GPU mode (USE_GPU_OFFLOAD=ON)"
else
    log_info "Running in CPU-only mode (USE_GPU_OFFLOAD=OFF)"
fi

log_section "CI Test Suite: $TEST_SUITE"
echo "Build directory: $BUILD_DIR"
echo ""

# Run paradigm check first (always)
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "paradigm" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Code Sharing Paradigm Check"
    if "${SCRIPT_DIR}/check_code_sharing.sh"; then
        log_success "Code sharing paradigm"
        PASSED=$((PASSED + 1))
    else
        log_failure "Code sharing paradigm"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - check_code_sharing.sh"
    fi
fi

# Fast tests (~1-2 minutes total)
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "fast" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Fast Tests (~1-2 minutes)"

    # Core 3D tests (longer timeouts for CPU Debug builds)
    run_test "3D Quick Validation" "$BUILD_DIR/test_3d_quick_validation" 120
    run_test "3D Gradients" "$BUILD_DIR/test_3d_gradients" 60
    run_test "3D W-Velocity" "$BUILD_DIR/test_3d_w_velocity" 60
    run_test "3D BC Application" "$BUILD_DIR/test_3d_bc_application" 180

    # Existing fast tests
    run_test "Mesh" "$BUILD_DIR/test_mesh" 30
    run_test "Features" "$BUILD_DIR/test_features" 30
    run_test "NN Core" "$BUILD_DIR/test_nn_core" 30
fi

# Golden file regression tests - verify numerical reproducibility
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "golden" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Golden File Regression Tests"

    # These tests verify exact numerical reproducibility against reference files
    # Each runs 10 time steps and compares output to golden files (tol=1e-10)
    run_golden_test "Golden: Channel k-omega" "channel_komega" 60
    run_golden_test "Golden: Channel EARSM" "channel_earsm" 60
    run_golden_test "Golden: Mixing Length" "mixing_length" 60
    run_golden_test "Golden: 3D Laminar" "laminar_3d" 60
    # NN models may be skipped if weights not available
    run_golden_test "Golden: Channel MLP" "channel_mlp" 60
    run_golden_test "Golden: Channel TBNN" "channel_tbnn" 60
fi

# Medium tests (~2-5 minutes total)
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Medium Tests (~2-5 minutes)"

    run_test "3D Poiseuille Fast" "$BUILD_DIR/test_3d_poiseuille_fast" 300
    run_test "Poisson" "$BUILD_DIR/test_poisson" 120
    run_test "Poisson Solvers 2D/3D" "$BUILD_DIR/test_poisson_solvers" 300
    run_test "Stability" "$BUILD_DIR/test_stability" 120
    run_test "Turbulence" "$BUILD_DIR/test_turbulence" 120
    run_test "Turbulence Features" "$BUILD_DIR/test_turbulence_features" 120
    run_test "Turbulence Guard" "$BUILD_DIR/test_turbulence_guard" 60
    run_test "All Turbulence Models Smoke" "$BUILD_DIR/test_all_turbulence_models_smoke" 300
fi

# GPU-specific tests
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "gpu" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "GPU-Specific Tests"

    # Cross-build comparison tests (require both CPU and GPU builds)
    # These tests compare CPU-built outputs against GPU-built outputs
    run_cross_build_test "CPU/GPU Bitwise" "test_cpu_gpu_bitwise" 180 "bitwise"
    run_cross_build_test "Poisson CPU/GPU 3D" "test_poisson_cpu_gpu_3d" 180 "poisson3d"
    run_cross_build_test "CPU/GPU Consistency" "test_cpu_gpu_consistency" 180 "consistency"
    run_cross_build_test "Solver CPU/GPU" "test_solver_cpu_gpu" 180 "solver"
    run_cross_build_test "Time History Consistency" "test_time_history_consistency" 180 "timehistory"

    # Non-comparison GPU tests
    run_test "Backend Execution" "$BUILD_DIR/test_backend_execution" 60

    # GPU utilization test - ensures compute runs on GPU, not CPU
    # Only meaningful for GPU builds (skips gracefully on CPU builds)
    if [[ "$USE_GPU" == "ON" ]]; then
        run_test "GPU Utilization" "$BUILD_DIR/test_gpu_utilization" 300
    fi
fi

# Longer tests (~3-5 minutes)
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Longer Tests (~3-5 minutes)"

    run_test "2D/3D Comparison" "$BUILD_DIR/test_2d_3d_comparison" 600
    run_test "Solver" "$BUILD_DIR/test_solver" 900
    run_test "Divergence All BCs" "$BUILD_DIR/test_divergence_all_bcs" 180
    run_test "Physics Validation" "$BUILD_DIR/test_physics_validation" 600
    run_test "Taylor-Green" "$BUILD_DIR/test_tg_validation" 120
    run_test "NN Integration" "$BUILD_DIR/test_nn_integration" 180
fi

# Slow tests (only with 'full' flag)
if [ "$TEST_SUITE" = "full" ]; then
    log_section "Slow Tests (full suite only)"

    run_test "Perturbed Channel" "$BUILD_DIR/test_perturbed_channel" 600
fi

# Summary
log_section "Test Summary"

ELAPSED=$SECONDS
MINUTES=$((ELAPSED / 60))
SECONDS_REMAINING=$((ELAPSED % 60))

echo ""
echo "Results:"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo ""
echo "Total time: ${MINUTES}m ${SECONDS_REMAINING}s"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}FAILED TESTS:${NC}"
    echo -e "$FAILED_TESTS"
    echo ""
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
