#!/bin/bash
# ci.sh - Run CI tests (locally or in CI pipelines)
#
# This script runs the test suite used by both local development and GitHub CI.
#
# Usage:
#   ./scripts/ci.sh              # Run all tests with GPU+HYPRE (default)
#   ./scripts/ci.sh fast         # Run only fast tests (~1 minute)
#   ./scripts/ci.sh full         # Run all tests including slow ones
#   ./scripts/ci.sh gpu          # Run GPU-specific tests only
#   ./scripts/ci.sh hypre        # Run HYPRE-specific tests only
#   ./scripts/ci.sh paradigm     # Run code sharing paradigm checks
#   ./scripts/ci.sh --cpu        # Force CPU-only build (no GPU, no HYPRE)
#   ./scripts/ci.sh --cpu fast   # CPU-only with fast tests
#   ./scripts/ci.sh --no-hypre   # Disable HYPRE (use multigrid only)
#   ./scripts/ci.sh --debug      # Debug build mode (4x timeout multiplier)
#   ./scripts/ci.sh -v           # Verbose output (show full test output)
#   ./scripts/ci.sh --verbose    # Same as -v
#
# HYPRE Poisson Solver:
#   HYPRE is ENABLED by default for GPU builds (provides 8-10x speedup).
#   Use --no-hypre to test with multigrid-only build.
#   HYPRE tests validate both solvers produce consistent results.
#
# Cross-build tests (CPU vs GPU comparison):
#   When running in GPU mode (default), the script will:
#   1. Build both CPU (USE_GPU_OFFLOAD=OFF) and GPU (USE_GPU_OFFLOAD=ON) versions
#   2. Generate CPU reference outputs
#   3. Run GPU and compare against CPU reference
#   4. FAIL if GPU is not available (not skip!)
#
#   This ensures we actually test CPU/GPU consistency, not just CPU/CPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# GPU compute capability detection (precedence):
#   1. GPU_CC env var if set (e.g., GPU_CC=90 ./scripts/ci.sh)
#   2. Auto-detect from nvidia-smi (first GPU's compute_cap)
#   3. Fallback to 80 (A100/Ampere) - conservative default
# Examples: A100=80, H100/H200=90
GPU_CC_SOURCE="env"
if [[ -z "${GPU_CC:-}" ]]; then
    GPU_CC_SOURCE="auto"
    if command -v nvidia-smi &> /dev/null; then
        # Query compute capability (e.g., "8.0" -> "80", "9.0" -> "90")
        GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [[ -z "$GPU_CC" ]]; then
            # nvidia-smi exists but returned empty (MIG mode, no GPU visible, etc.)
            GPU_CC_SOURCE="fallback"
        fi
    else
        GPU_CC_SOURCE="fallback"
    fi
    # Fallback to 80 (A100/Ampere) if detection fails
    GPU_CC=${GPU_CC:-80}
fi

# Validate GPU_CC is a reasonable value (2-digit number in range 60-100)
if ! [[ "$GPU_CC" =~ ^[0-9]+$ ]] || [[ "$GPU_CC" -lt 60 ]] || [[ "$GPU_CC" -gt 100 ]]; then
    echo "ERROR: Invalid GPU_CC value: '$GPU_CC' (expected 60-100, e.g., 80 for A100, 90 for H100)"
    echo "       Set GPU_CC explicitly: GPU_CC=90 ./scripts/ci.sh"
    exit 1
fi

# Parse flags
USE_GPU=ON
# HYPRE is enabled by default for GPU builds (best performance)
# Use --no-hypre to disable
USE_HYPRE=AUTO
VERBOSE=0
DEBUG_BUILD=0
TIMEOUT_MULTIPLIER=1
while [[ "$1" == --* || "$1" == -* ]]; do
    case "$1" in
        --cpu)
            USE_GPU=OFF
            shift
            ;;
        --hypre)
            USE_HYPRE=ON
            shift
            ;;
        --no-hypre)
            USE_HYPRE=OFF
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --debug)
            # Debug builds are 3-10x slower, use 4x timeout multiplier
            DEBUG_BUILD=1
            TIMEOUT_MULTIPLIER=4
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve AUTO: HYPRE is enabled by default for GPU builds
if [[ "$USE_HYPRE" == "AUTO" ]]; then
    if [[ "$USE_GPU" == "ON" ]]; then
        USE_HYPRE=ON
    else
        USE_HYPRE=OFF
    fi
fi

# Set build directory based on mode
if [[ "$USE_GPU" == "ON" ]]; then
    if [[ "$USE_HYPRE" == "ON" ]]; then
        BUILD_DIR="${PROJECT_DIR}/build_gpu_hypre"
    else
        BUILD_DIR="${PROJECT_DIR}/build_gpu"
    fi
else
    if [[ "$USE_HYPRE" == "ON" ]]; then
        BUILD_DIR="${PROJECT_DIR}/build_cpu_hypre"
    else
        BUILD_DIR="${PROJECT_DIR}/build_cpu"
    fi
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

# Check if GPU is available (for GPU builds)
check_gpu_available() {
    # Check if nvidia-smi exists and reports a GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0  # GPU available
        fi
    fi
    return 1  # No GPU
}

# Build a specific configuration, ensuring all tests are built
# Usage: ensure_build <build_dir> <use_gpu_offload> [use_hypre]
ensure_build() {
    local build_dir=$1
    local gpu_offload=$2
    local use_hypre=${3:-OFF}
    local build_name="CPU"
    if [[ "$gpu_offload" == "ON" ]]; then
        build_name="GPU"
    fi
    if [[ "$use_hypre" == "ON" ]]; then
        build_name="${build_name}+HYPRE"
    fi

    log_info "Ensuring $build_name build in $build_dir..."
    mkdir -p "$build_dir"

    # Save current directory (SC2155: declare and assign separately)
    local orig_dir
    orig_dir=$(pwd)
    cd "$build_dir"

    # Configure if not already configured
    if [ ! -f "CMakeCache.txt" ]; then
        log_info "  Configuring $build_name build..."
        local gpu_cc_flag=""
        if [[ "$gpu_offload" == "ON" ]]; then
            gpu_cc_flag="-DGPU_CC=${GPU_CC}"
        fi
        if ! cmake "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=${gpu_offload} -DUSE_HYPRE=${use_hypre} ${gpu_cc_flag} -DBUILD_TESTS=ON > cmake_output.log 2>&1; then
            log_failure "$build_name cmake configuration failed"
            cat cmake_output.log | tail -20 | sed 's/^/    /'
            cd "$orig_dir"
            return 1
        fi
    fi

    # Always run make to ensure all targets are built
    # (make is smart and will skip already-built targets)
    log_info "  Building $build_name..."
    if ! make -j"$(nproc)" > build_output.log 2>&1; then
        log_failure "$build_name build failed"
        cat build_output.log | tail -30 | sed 's/^/    /'
        cd "$orig_dir"
        return 1
    fi

    log_success "$build_name build completed"
    cd "$orig_dir"
    return 0
}

# Parse arguments
TEST_SUITE="${1:-all}"

# Track results
PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=""

# Track build status to avoid redundant ensure_build calls
CPU_BUILD_ENSURED=0
GPU_BUILD_ENSURED=0

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
    local base_timeout=${3:-120}
    local env_prefix=${4:-""}  # Optional env vars (e.g., "OMP_TARGET_OFFLOAD=MANDATORY")

    # Apply timeout multiplier for Debug builds
    local timeout_secs=$((base_timeout * TIMEOUT_MULTIPLIER))

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
    if [ $DEBUG_BUILD -eq 1 ]; then
        log_info "Running $test_name... (timeout: ${timeout_secs}s, debug 4x)"
    else
        log_info "Running $test_name..."
    fi

    local output_file="/tmp/test_output_$$.txt"
    local exit_code=0
    if [ -n "$env_prefix" ]; then
        env $env_prefix timeout "$timeout_secs" "$test_binary" > "$output_file" 2>&1 || exit_code=$?
    else
        timeout "$timeout_secs" "$test_binary" > "$output_file" 2>&1 || exit_code=$?
    fi

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

# Run a CPU/GPU cross-build comparison test
# These tests require both CPU and GPU builds and compare outputs between them
# This function will build both versions if they don't exist
run_cross_build_test() {
    local test_name=$1
    local test_binary_name=$2
    local timeout_secs=${3:-120}
    local ref_prefix=$4

    local cpu_build_dir="${PROJECT_DIR}/build_cpu"
    local gpu_build_dir="${PROJECT_DIR}/build_gpu"
    local cpu_binary="${cpu_build_dir}/${test_binary_name}"
    local gpu_binary="${gpu_build_dir}/${test_binary_name}"

    echo ""
    log_info "Running $test_name (cross-build CPU vs GPU comparison)..."

    # For cross-build tests, we need a GPU available
    if ! check_gpu_available; then
        log_failure "$test_name (GPU required but not available)"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (no GPU)"
        return 0
    fi

    # Ensure both CPU and GPU builds exist and are up to date
    # Use caching to avoid redundant builds across multiple cross-build tests
    if [ $CPU_BUILD_ENSURED -eq 0 ]; then
        if ! ensure_build "$cpu_build_dir" "OFF"; then
            log_failure "$test_name (CPU build failed)"
            FAILED=$((FAILED + 1))
            FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (CPU build)"
            return 0
        fi
        CPU_BUILD_ENSURED=1
    fi

    if [ $GPU_BUILD_ENSURED -eq 0 ]; then
        if ! ensure_build "$gpu_build_dir" "ON"; then
            log_failure "$test_name (GPU build failed)"
            FAILED=$((FAILED + 1))
            FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (GPU build)"
            return 0
        fi
        GPU_BUILD_ENSURED=1
    fi

    # Verify binaries exist after build
    if [ ! -f "$cpu_binary" ]; then
        log_failure "$test_name (CPU binary missing after build: $cpu_binary)"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (CPU binary missing)"
        return 0
    fi

    if [ ! -f "$gpu_binary" ]; then
        log_failure "$test_name (GPU binary missing after build: $gpu_binary)"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (GPU binary missing)"
        return 0
    fi

    # Create reference directory
    local ref_dir="${PROJECT_DIR}/build_gpu/cpu_reference"
    mkdir -p "$ref_dir"

    local output_file="/tmp/test_output_$$.txt"

    # Always regenerate CPU reference to ensure consistency
    # (reference files might be stale from a previous build)
    log_info "  Step 1: Generating CPU reference..."
    local cpu_exit_code=0
    timeout "$timeout_secs" "$cpu_binary" --dump-prefix "${ref_dir}/${ref_prefix}" > "$output_file" 2>&1 || cpu_exit_code=$?

    if [ $cpu_exit_code -ne 0 ]; then
        log_failure "$test_name (CPU reference generation failed, exit code: $cpu_exit_code)"
        echo "  Output (last 30 lines):"
        tail -30 "$output_file" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name (CPU ref generation)"
        rm -f "$output_file"
        return 0
    fi

    if [ $VERBOSE -eq 1 ]; then
        echo "  CPU reference output:"
        cat "$output_file" | sed 's/^/    /'
    fi

    # Run GPU comparison against CPU reference
    # MANDATORY ensures we fail if GPU offload doesn't work (no silent CPU fallback)
    log_info "  Step 2: Running GPU and comparing against CPU reference..."
    local gpu_exit_code=0
    OMP_TARGET_OFFLOAD=MANDATORY timeout "$timeout_secs" "$gpu_binary" --compare-prefix "${ref_dir}/${ref_prefix}" > "$output_file" 2>&1 || gpu_exit_code=$?

    if [ $gpu_exit_code -eq 0 ]; then
        log_success "$test_name"
        PASSED=$((PASSED + 1))
        if [ $VERBOSE -eq 1 ]; then
            echo "  GPU comparison output:"
            cat "$output_file" | sed 's/^/    /'
        else
            local summary
            summary=$(grep -E '(\[PASS\]|\[FAIL\]|\[OK\]|\[SUCCESS\]|\[WARN\]|PASSED|FAILED|Max abs diff|Max rel diff|RMS diff)' "$output_file" | head -10) || true
            if [ -n "$summary" ]; then
                echo "$summary" | sed 's/^/    /'
            fi
        fi
    else
        log_failure "$test_name (GPU comparison failed, exit code: $gpu_exit_code)"
        echo "  Output (last 30 lines):"
        tail -30 "$output_file" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_TESTS="${FAILED_TESTS}\n  - $test_name"
    fi
    rm -f "$output_file"
}

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    log_info "Build directory not found. Creating and building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    GPU_CC_FLAG=""
    if [[ "$USE_GPU" == "ON" ]]; then
        GPU_CC_FLAG="-DGPU_CC=${GPU_CC}"
    fi
    cmake .. -DUSE_GPU_OFFLOAD=${USE_GPU} -DUSE_HYPRE=${USE_HYPRE} ${GPU_CC_FLAG} -DBUILD_TESTS=ON
    make -j"$(nproc)"
    cd "$PROJECT_DIR"
fi

# Display mode
if [[ "$USE_GPU" == "ON" ]]; then
    if [[ "$USE_HYPRE" == "ON" ]]; then
        log_info "Running in GPU+HYPRE mode (USE_GPU_OFFLOAD=ON, USE_HYPRE=ON)"
    else
        log_info "Running in GPU mode (USE_GPU_OFFLOAD=ON)"
    fi
else
    if [[ "$USE_HYPRE" == "ON" ]]; then
        log_info "Running in CPU+HYPRE mode (USE_GPU_OFFLOAD=OFF, USE_HYPRE=ON)"
    else
        log_info "Running in CPU-only mode (USE_GPU_OFFLOAD=OFF)"
    fi
fi

log_section "CI Test Suite: $TEST_SUITE"
echo "Build directory: $BUILD_DIR"

# Report GPU_CC detection status
if [[ "$USE_GPU" == "ON" ]]; then
    echo "GPU compute capability: cc$GPU_CC (source: $GPU_CC_SOURCE)"
    if [[ "$GPU_CC_SOURCE" == "fallback" ]]; then
        log_info "WARNING: GPU_CC auto-detection failed (no GPU visible?)"
        log_info "         Using fallback cc$GPU_CC - may cause runtime errors if wrong"
        log_info "         Set GPU_CC explicitly for cross-compile: GPU_CC=90 ./scripts/ci.sh"
    fi
fi
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
    # Skip these if --cpu flag was passed (no GPU comparison possible)
    if [[ "$USE_GPU" == "OFF" ]]; then
        log_info "Skipping cross-build tests in CPU-only mode (--cpu flag)"
        log_info "Cross-build tests require GPU to compare CPU vs GPU outputs"
    else
        run_cross_build_test "CPU/GPU Bitwise" "test_cpu_gpu_bitwise" 180 "bitwise"
        run_cross_build_test "Poisson CPU/GPU 3D" "test_poisson_cpu_gpu_3d" 180 "poisson3d"
        run_cross_build_test "CPU/GPU Consistency" "test_cpu_gpu_consistency" 180 "consistency"
        run_cross_build_test "Solver CPU/GPU" "test_solver_cpu_gpu" 180 "solver"
        run_cross_build_test "Time History Consistency" "test_time_history_consistency" 180 "timehistory"
    fi

    # Non-comparison GPU tests
    run_test "Backend Execution" "$BUILD_DIR/test_backend_execution" 60

    # GPU utilization test - ensures compute runs on GPU, not CPU
    # Only meaningful for GPU builds (skips gracefully on CPU builds)
    # MANDATORY ensures we fail if GPU offload doesn't work (no silent CPU fallback)
    if [[ "$USE_GPU" == "ON" ]]; then
        run_test "GPU Utilization" "$BUILD_DIR/test_gpu_utilization" 300 "OMP_TARGET_OFFLOAD=MANDATORY"
    fi
fi

# HYPRE-specific tests (only when --hypre flag is used)
if [[ "$USE_HYPRE" == "ON" ]]; then
    if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "hypre" ] || [ "$TEST_SUITE" = "full" ]; then
        log_section "HYPRE Poisson Solver Tests"

        # HYPRE initialization test (fast)
        run_test "HYPRE All BCs Init" "$BUILD_DIR/test_hypre_all_bcs" 60

        # HYPRE vs Multigrid validation test
        # This compares HYPRE and Multigrid results on the same problem
        run_test "HYPRE Validation" "$BUILD_DIR/test_hypre_validation" 300

        # Cross-build HYPRE test (CPU HYPRE vs GPU HYPRE)
        # Only run if GPU is available and enabled
        if [[ "$USE_GPU" == "ON" ]] && check_gpu_available; then
            log_info "Running HYPRE cross-build comparison (requires both CPU and GPU HYPRE builds)..."

            cpu_hypre_dir="${PROJECT_DIR}/build_cpu_hypre"
            gpu_hypre_dir="${PROJECT_DIR}/build_gpu_hypre"
            ref_dir="${PROJECT_DIR}/build_gpu_hypre/hypre_reference"
            mkdir -p "$ref_dir"

            # Ensure CPU HYPRE build exists
            if ! ensure_build "$cpu_hypre_dir" "OFF" "ON"; then
                log_failure "CPU HYPRE build failed"
                FAILED=$((FAILED + 1))
                FAILED_TESTS="${FAILED_TESTS}\n  - HYPRE CPU build"
            else
                # Generate CPU HYPRE reference
                output_file="/tmp/hypre_ref_$$.txt"
                if timeout 180 "$cpu_hypre_dir/test_hypre_validation" --dump-prefix "${ref_dir}/hypre" > "$output_file" 2>&1; then
                    log_success "HYPRE CPU reference generated"

                    # Ensure GPU HYPRE build exists
                    if ! ensure_build "$gpu_hypre_dir" "ON" "ON"; then
                        log_failure "GPU HYPRE build failed"
                        FAILED=$((FAILED + 1))
                        FAILED_TESTS="${FAILED_TESTS}\n  - HYPRE GPU build"
                    else
                        # Run GPU HYPRE comparison
                        if OMP_TARGET_OFFLOAD=MANDATORY timeout 180 "$gpu_hypre_dir/test_hypre_validation" --compare-prefix "${ref_dir}/hypre" > "$output_file" 2>&1; then
                            log_success "HYPRE Cross-Build (CPU vs GPU)"
                            PASSED=$((PASSED + 1))
                            if [ $VERBOSE -eq 1 ]; then
                                cat "$output_file" | sed 's/^/    /'
                            fi
                        else
                            log_failure "HYPRE Cross-Build (CPU vs GPU)"
                            echo "  Output (last 30 lines):"
                            tail -30 "$output_file" | sed 's/^/    /'
                            FAILED=$((FAILED + 1))
                            FAILED_TESTS="${FAILED_TESTS}\n  - HYPRE Cross-Build"
                        fi
                    fi
                else
                    log_failure "HYPRE CPU reference generation failed"
                    tail -20 "$output_file" | sed 's/^/    /'
                    FAILED=$((FAILED + 1))
                    FAILED_TESTS="${FAILED_TESTS}\n  - HYPRE CPU reference"
                fi
                rm -f "$output_file"
            fi
        fi
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
