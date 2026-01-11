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

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
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

# QoI metrics aggregation file (JSON fragments)
QOI_METRICS_FILE="/tmp/ci_qoi_metrics_$$.json"
echo "{" > "$QOI_METRICS_FILE"
QOI_COUNT=0

# Track which tests should have QoI (for parser health check)
declare -A QOI_EXPECTED
declare -A QOI_EXTRACTED

# Function to extract QoI values from test output
# Parses QOI_JSON: lines emitted by tests (robust, machine-readable)
# Falls back to regex parsing for legacy tests
extract_qoi_metrics() {
    local test_name=$1
    local output_file=$2

    # Skip if no output
    [ ! -f "$output_file" ] && return

    # Mark that we expected QoI from this test
    # These are the tests that MUST emit QOI_JSON lines
    case "$test_name" in
        "TGV 2D Invariants"|"TGV 3D Invariants"|"TGV Repeatability"|"CPU/GPU Bitwise"|"HYPRE Validation"|"MMS Convergence"|"RANS Channel Sanity"|"Perf Sentinel"|"Solver Selection"|"Stability Sentinel"|"Poiseuille Steady"|"Energy Budget"|"Operator Consistency"|"Advection Rotation"|"Projection Effectiveness"|"Poiseuille Refinement")
            QOI_EXPECTED["$test_name"]=1
            ;;
    esac

    # Try to extract QOI_JSON lines first (preferred, robust method)
    local qoi_lines
    qoi_lines=$(grep '^QOI_JSON: ' "$output_file" 2>/dev/null || true)

    if [ -n "$qoi_lines" ]; then
        # Parse QOI_JSON lines and merge into metrics
        while IFS= read -r line; do
            # Extract JSON after "QOI_JSON: "
            local json="${line#QOI_JSON: }"
            # Extract test name from JSON
            local test_id
            test_id=$(echo "$json" | grep -oP '"test":"\K[^"]+' || true)

            if [ -n "$test_id" ]; then
                # Handle tests with case IDs: use composite keys (test_id.$case_id)
                # {"test":"perf_gate","case":"foo",...} -> "perf_gate.foo": {...}
                # {"test":"solver_select","case":"2D_channel_auto",...} -> "solver_select.2D_channel_auto": {...}
                if [ "$test_id" = "perf_gate" ] || [ "$test_id" = "solver_select" ]; then
                    local case_id
                    case_id=$(echo "$json" | grep -oP '"case":"\K[^"]+' || true)
                    if [ -n "$case_id" ]; then
                        # Remove test and case keys, keep the rest
                        local metrics
                        metrics=$(echo "$json" | sed 's/"test":"[^"]*",//' | sed 's/"case":"[^"]*",//' | sed 's/^{//' | sed 's/}$//')
                        # Use composite key: $test_id.$case_id
                        if [ $QOI_COUNT -gt 0 ]; then
                            echo "," >> "$QOI_METRICS_FILE"
                        fi
                        echo "    \"$test_id.$case_id\": {$metrics}" >> "$QOI_METRICS_FILE"
                        QOI_COUNT=$((QOI_COUNT + 1))
                    fi
                else
                    # Standard case: use test_id as key
                    local metrics
                    metrics=$(echo "$json" | sed 's/"test":"[^"]*",//' | sed 's/^{//' | sed 's/}$//')

                    if [ $QOI_COUNT -gt 0 ]; then
                        echo "," >> "$QOI_METRICS_FILE"
                    fi
                    echo "    \"$test_id\": {$metrics}" >> "$QOI_METRICS_FILE"
                    QOI_COUNT=$((QOI_COUNT + 1))
                fi
                QOI_EXTRACTED["$test_name"]=1
            fi
        done <<< "$qoi_lines"
        return
    fi

    # Fallback: legacy regex parsing for tests not yet updated
    # NOTE: These should match the key names in emit_qoi_* functions
    local metrics=""

    case "$test_name" in
        "TGV 2D Invariants")
            local div_max=$(grep -oP 'Divergence-free.*val=\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            local e_final=$(grep -oP 'KE_final=\K[0-9.]+' "$output_file" 2>/dev/null | head -1)
            local e_ratio=$(grep -oP 'ratio=\K[0-9.]+' "$output_file" 2>/dev/null | head -1)
            local const_vel=$(grep -oP 'Constant velocity.*val=\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$div_max" ] || [ -n "$e_final" ]; then
                metrics="\"tgv_2d\": {\"div_Linf\": ${div_max:-null}, \"ke_final\": ${e_final:-null}, \"ke_ratio\": ${e_ratio:-null}, \"const_vel_Linf\": ${const_vel:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "TGV 3D Invariants")
            local div_max=$(grep -oP '3D Divergence-free.*val=\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$div_max" ]; then
                metrics="\"tgv_3d\": {\"div_Linf\": ${div_max:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "TGV Repeatability")
            local rel_e=$(grep -oP 'relE = \|E1-E2\|/E1 = \K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$rel_e" ]; then
                metrics="\"repeatability\": {\"ke_rel_diff\": ${rel_e:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "CPU/GPU Bitwise")
            local u_rel=$(grep -oP 'U-velocity.*Rel L2:\s*\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            local p_rel=$(grep -oP 'Pressure.*Rel L2:\s*\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$u_rel" ] || [ -n "$p_rel" ]; then
                metrics="\"cpu_gpu\": {\"u_rel_L2\": ${u_rel:-null}, \"p_rel_L2\": ${p_rel:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "HYPRE Validation")
            local p_rel=$(grep -oP 'Pressure.*Rel L2:\s*\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$p_rel" ]; then
                metrics="\"hypre_vs_mg\": {\"p_prime_rel_L2\": ${p_rel:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "MMS Convergence")
            local rate=$(grep -oP 'rate=\K[0-9.]+' "$output_file" 2>/dev/null | tail -1)
            if [ -n "$rate" ]; then
                metrics="\"mms\": {\"spatial_order\": ${rate:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
        "RANS Channel Sanity")
            local u_bulk=$(grep -oP 'U_bulk=\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            local max_nut=$(grep -oP 'max_nut_ratio=\K[0-9.e+-]+' "$output_file" 2>/dev/null | head -1)
            if [ -n "$u_bulk" ] || [ -n "$max_nut" ]; then
                metrics="\"rans_channel\": {\"u_bulk\": ${u_bulk:-null}, \"nut_ratio_max\": ${max_nut:-null}}"
                QOI_EXTRACTED["$test_name"]=1
            fi
            ;;
    esac

    # Append to QoI file if we extracted any metrics
    if [ -n "$metrics" ]; then
        if [ $QOI_COUNT -gt 0 ]; then
            echo "," >> "$QOI_METRICS_FILE"
        fi
        echo "    $metrics" >> "$QOI_METRICS_FILE"
        QOI_COUNT=$((QOI_COUNT + 1))
    fi
}

# Parser health check: FAIL if expected QoIs weren't extracted
# This catches output format changes or binary mismatches early
check_qoi_health() {
    local missing=0
    local missing_tests=""
    for test_name in "${!QOI_EXPECTED[@]}"; do
        if [ -z "${QOI_EXTRACTED[$test_name]:-}" ]; then
            log_failure "Ran $test_name but extracted 0 QoI fields. Output format changed?"
            missing=$((missing + 1))
            missing_tests="${missing_tests}\n  - $test_name"
        fi
    done
    if [ $missing -gt 0 ]; then
        log_failure "$missing test(s) missing QoI extraction - trend analysis will be incomplete"
        echo -e "Missing QoI from:${missing_tests}"
        # Count as failures so CI fails
        FAILED=$((FAILED + missing))
        FAILED_TESTS="${FAILED_TESTS}${missing_tests} (QoI extraction failed)"
    fi
}

# Track build status to avoid redundant ensure_build calls
CPU_BUILD_ENSURED=0
GPU_BUILD_ENSURED=0

# Known flaky tests on GPU (pre-existing issues, not related to 3D work)
# These will be skipped when USE_GPU=ON until root causes are addressed.
# Note: test_solver and test_physics_validation were slow (not flaky) - fixed by increasing timeouts
# Note: turbulence guard (now in test_turbulence_unified) uses check_for_nan_inf directly instead of step()
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
            # Show summary lines (PASSED/FAILED counts, key results, metrics, QOI)
            # Patterns: [PASS], [FAIL], [OK], PASSED, FAILED, Results:, ===...===,
            #           max_diff=, max_div=, L2/Linf norms, Test N:, scientific notation, QOI_JSON
            local summary
            summary=$(grep -E '(\[PASS\]|\[FAIL\]|\[OK\]|\[SUCCESS\]|PASSED|FAILED|passed|failed|Results:|Result:|===.*===|error=|Error|SUCCESS|max_diff|max_div|L2|Linf|Test [0-9]+:|[0-9]+\.[0-9]+e[-+]?[0-9]+|QOI_JSON)' "$output_file" | head -25) || true
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

    # Extract QoI metrics before deleting output
    extract_qoi_metrics "$test_name" "$output_file"
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

# Note: run_cross_build_canary_test removed - functionality consolidated into test_backend_unified
# The unified test includes an internal canary that verifies CPU/GPU FP differences

# Check if build is needed (library doesn't exist or directory is fresh from cache)
mkdir -p "$BUILD_DIR"
if [ ! -f "$BUILD_DIR/libnn_cfd_core.a" ]; then
    log_info "Building project (libnn_cfd_core.a not found)..."
    cd "$BUILD_DIR"
    # Force fresh configure: remove CMake cache but preserve _deps (HYPRE cache)
    rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile lib*.a
    GPU_CC_FLAG=""
    if [[ "$USE_GPU" == "ON" ]]; then
        GPU_CC_FLAG="-DGPU_CC=${GPU_CC}"
    fi
    cmake .. -DUSE_GPU_OFFLOAD=${USE_GPU} -DUSE_HYPRE=${USE_HYPRE} ${GPU_CC_FLAG} -DBUILD_TESTS=ON
    make -j"$(nproc)"
    cd "$PROJECT_DIR"
else
    log_info "Using existing build in $BUILD_DIR"
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

# ============================================================================
# Discovery check: Print what test suites will run
# This makes it immediately obvious if a suite is missing from the runner
# ============================================================================
echo "Test suites to run:"
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "fast" ] || [ "$TEST_SUITE" = "full" ]; then
    echo "  [FAST] 3D-Quick, 3D-Gradients, 3D-W, 3D-BC, Mesh, Features, NN-Core,"
    echo "         Data-Driven, Config, TGV-2D-Invariants, TGV-Repeatability"
fi
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "full" ]; then
    echo "  [MEDIUM] 3D-Poiseuille, Poisson, Stability, Turbulence, Error-Recovery,"
    echo "           Adaptive-Dt, Mesh-Edge, 3D-BC-Corners, VTK, TGV-3D-Invariants,"
    echo "           MMS-Convergence, RANS-Channel-Sanity"
fi
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "gpu" ] || [ "$TEST_SUITE" = "full" ]; then
    echo "  [GPU] CPU/GPU-Bitwise, Backend-Unified, GPU-Utilization, V-cycle-Graph"
fi
if [[ "$USE_HYPRE" == "ON" ]]; then
    if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "hypre" ] || [ "$TEST_SUITE" = "full" ]; then
        echo "  [HYPRE] HYPRE-All-BCs, HYPRE-Validation, HYPRE-Cross-Build"
    fi
fi
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "full" ]; then
    echo "  [LONG] 2D/3D-Comparison, Solver, Divergence-All-BCs, Physics-Validation, NN-Integration"
fi
if [ "$TEST_SUITE" = "full" ]; then
    echo "  [SLOW] Perturbed-Channel"
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

    # Data-driven test framework demo (24 tests x 2 runs = ~90s)
    run_test "Data-Driven Demo" "$BUILD_DIR/test_data_driven_demo" 180

    # Configuration and I/O tests (very fast)
    run_test "Config" "$BUILD_DIR/test_config" 30

    # New CI invariant tests (fast: 2D, small grids)
    run_test "TGV 2D Invariants" "$BUILD_DIR/test_tgv_2d_invariants" 120
    run_test "TGV Repeatability" "$BUILD_DIR/test_tgv_repeatability" 60
fi

# Medium tests (~2-5 minutes total)
if [ "$TEST_SUITE" = "all" ] || [ "$TEST_SUITE" = "full" ]; then
    log_section "Medium Tests (~2-5 minutes)"

    run_test "3D Poiseuille Fast" "$BUILD_DIR/test_3d_poiseuille_fast" 300
    run_test "Poisson Unified" "$BUILD_DIR/test_poisson_unified" 180
    run_test "Stability" "$BUILD_DIR/test_stability" 120
    # Unified turbulence test (consolidates 6 turbulence test files)
    run_test "Turbulence Unified" "$BUILD_DIR/test_turbulence_unified" 300

    # New tests: error handling, adaptive dt, mesh edge cases, 3D BCs, VTK output
    run_test "Error Recovery" "$BUILD_DIR/test_error_recovery" 120
    run_test "Adaptive Dt" "$BUILD_DIR/test_adaptive_dt" 120
    run_test "Mesh Edge Cases" "$BUILD_DIR/test_mesh_edge_cases" 120
    run_test "3D BC Corners" "$BUILD_DIR/test_3d_bc_corners" 180
    run_test "VTK Output" "$BUILD_DIR/test_vtk_output" 60

    # New CI invariant tests (medium: 3D, convergence, RANS)
    run_test "TGV 3D Invariants" "$BUILD_DIR/test_tgv_3d_invariants" 300
    run_test "MMS Convergence" "$BUILD_DIR/test_mms_convergence" 300
    run_test "RANS Channel Sanity" "$BUILD_DIR/test_rans_channel_sanity" 600

    # High-trust physics validation tests
    run_test "Poiseuille Steady" "$BUILD_DIR/test_poiseuille_steady" 600
    run_test "Energy Budget" "$BUILD_DIR/test_energy_budget_channel" 120
    run_test "Operator Consistency" "$BUILD_DIR/test_operator_consistency" 120
    run_test "Advection Rotation" "$BUILD_DIR/test_advection_rotation" 60
    run_test "Projection Effectiveness" "$BUILD_DIR/test_projection_effectiveness" 60
    run_test "Poiseuille Refinement" "$BUILD_DIR/test_poiseuille_refinement" 300

    # Solver selection matrix test (verifies auto-selection logic)
    run_test "Solver Selection" "$BUILD_DIR/test_solver_selection" 120

    # Stability sentinel test (aggressive CFL, verifies dt limiter works)
    run_test "Stability Sentinel" "$BUILD_DIR/test_stability_sentinel" 120
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

        # Note: test_cpu_gpu_consistency, test_solver_cpu_gpu, test_time_history_consistency
        # were consolidated into test_cpu_gpu_unified (runs via test_unified_suite)
    fi

    # Non-comparison GPU tests
    # Backend unified test - consolidates backend_execution and backend_canary
    # Includes canary test that verifies CPU and GPU produce different FP results
    run_test "Backend Unified" "$BUILD_DIR/test_backend_unified" 60

    # GPU utilization test - ensures compute runs on GPU, not CPU
    # Only meaningful for GPU builds (skips gracefully on CPU builds)
    # MANDATORY ensures we fail if GPU offload doesn't work (no silent CPU fallback)
    if [[ "$USE_GPU" == "ON" ]]; then
        run_test "GPU Utilization" "$BUILD_DIR/test_gpu_utilization" 300 "OMP_TARGET_OFFLOAD=MANDATORY"

        # V-cycle CUDA Graph stress test - validates graphed MG solver
        # Tests BC alternation (graph recapture), convergence parity, anisotropic grids
        run_test "V-cycle Graph Stress" "$BUILD_DIR/test_vcycle_graph_stress" 120 "MG_USE_VCYCLE_GRAPH=1"
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
    run_test "Physics Validation Advanced" "$BUILD_DIR/test_physics_validation_advanced" 600
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

# ============================================================================
# Write CI metrics JSON artifact
# ============================================================================
METRICS_DIR="${PROJECT_DIR}/artifacts"
METRICS_FILE="${METRICS_DIR}/ci_metrics.json"
mkdir -p "$METRICS_DIR"

# Get git SHA (short form)
GIT_SHA=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Get build type and metadata
if [[ "$USE_GPU" == "ON" ]]; then
    if [[ "$USE_HYPRE" == "ON" ]]; then
        BUILD_TYPE="gpu+hypre"
    else
        BUILD_TYPE="gpu"
    fi
else
    if [[ "$USE_HYPRE" == "ON" ]]; then
        BUILD_TYPE="cpu+hypre"
    else
        BUILD_TYPE="cpu"
    fi
fi

# Get git branch
GIT_BRANCH=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Get GPU info if available
GPU_NAME="null"
GPU_CC="null"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/"/;s/$/"/' || echo "null")
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//;s/^/"/;s/$/"/' || echo "null")
fi

# Get OMP offload mode
OFFLOAD_MODE="${OMP_TARGET_OFFLOAD:-default}"

# Run parser health check
check_qoi_health

# Finalize QoI metrics file
echo "}" >> "$QOI_METRICS_FILE"

# Read QoI metrics content (remove outer braces for embedding)
QOI_CONTENT=""
if [ $QOI_COUNT -gt 0 ]; then
    # Extract content between braces
    QOI_CONTENT=$(sed '1d;$d' "$QOI_METRICS_FILE")
fi

# Write JSON metrics with stable schema (v2)
# Schema v2 changes from v1:
#   - ke_final_J -> ke_final (nondimensional)
#   - u_bulk_m_s -> u_bulk (nondimensional)
#   - Added fourier_mode and perf_gate QoIs
#   - QoI extraction failures now cause CI to fail
# Naming conventions:
#   - All keys use snake_case
#   - Norms: _Linf (L-infinity), _L2 (L2 norm)
#   - Relative values: _rel_ prefix
#   - Only add unit suffix when value is dimensional (rare)
cat > "$METRICS_FILE" << EOF
{
  "schema_version": "2",
  "metadata": {
    "git_sha": "$GIT_SHA",
    "branch": "$GIT_BRANCH",
    "build_type": "$BUILD_TYPE",
    "test_suite": "$TEST_SUITE",
    "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "elapsed_s": $ELAPSED,
    "offload": "$OFFLOAD_MODE",
    "gpu_name": $GPU_NAME,
    "gpu_cc": $GPU_CC
  },
  "summary": {
    "passed": $PASSED,
    "failed": $FAILED,
    "skipped": $SKIPPED
  },
  "qoi": {
$QOI_CONTENT
  }
}
EOF

# Cleanup temp file
rm -f "$QOI_METRICS_FILE"

log_info "Metrics written to: $METRICS_FILE"

# ============================================================================
# Baseline trend comparison (warning-only)
# Compares extracted QoI against stored baseline to catch regressions early
# Uses build-specific baselines: tests/baselines/baseline_${BUILD_TYPE}.json
# ============================================================================

BASELINE_DIR="${PROJECT_DIR}/tests/baselines"
BASELINE_FILE="${BASELINE_DIR}/baseline_${BUILD_TYPE}.json"

# Helper: compare value against ratio threshold AND absolute ceiling
# Usage: check_threshold <name> <current> <baseline> <ratio_thresh> <abs_ceiling> <higher_is_worse>
check_threshold() {
    local name=$1
    local current=$2
    local baseline=$3
    local ratio_thresh=$4
    local abs_ceiling=$5
    local higher_is_worse=${6:-1}  # 1 = higher is bad (default), 0 = lower is bad

    [ -z "$current" ] && return 0

    local warned=0

    # Absolute ceiling check (catches bad baselines)
    if [ -n "$abs_ceiling" ]; then
        local exceeds
        exceeds=$(echo "$current $abs_ceiling" | awk '{if($1>$2) print 1; else print 0}')
        if [ "$exceeds" = "1" ]; then
            log_warning "$name = $current exceeds absolute ceiling $abs_ceiling"
            warned=1
        fi
    fi

    # Ratio comparison (if baseline exists)
    if [ -n "$baseline" ]; then
        local ratio
        ratio=$(echo "$current $baseline" | awk '{if($2>1e-30) printf "%.3f", $1/$2; else print "0"}')

        local bad_ratio=0
        if [ "$higher_is_worse" = "1" ]; then
            bad_ratio=$(echo "$ratio $ratio_thresh" | awk '{if($1>$2) print 1; else print 0}')
        else
            bad_ratio=$(echo "$ratio $ratio_thresh" | awk '{if($1<$2) print 1; else print 0}')
        fi

        if [ "$bad_ratio" = "1" ]; then
            log_warning "$name = $current (${ratio}x baseline $baseline, threshold ${ratio_thresh}x)"
            warned=1
        fi
    fi

    return $warned
}

compare_to_baseline() {
    # Skip if no baseline file exists
    if [ ! -f "$BASELINE_FILE" ]; then
        log_info "No baseline file found at $BASELINE_FILE - skipping trend comparison"
        log_info "To create baseline: cp $METRICS_FILE $BASELINE_FILE"
        return 0
    fi

    log_info "Comparing QoI against baseline ($BUILD_TYPE)..."

    # Verify schema version matches
    local current_schema
    local baseline_schema
    current_schema=$(grep -oP '"schema_version":\s*"\K[^"]+' "$METRICS_FILE" 2>/dev/null | head -1 || echo "unknown")
    baseline_schema=$(grep -oP '"schema_version":\s*"\K[^"]+' "$BASELINE_FILE" 2>/dev/null | head -1 || echo "unknown")
    if [ "$current_schema" != "$baseline_schema" ]; then
        log_warning "Schema version mismatch: current=$current_schema, baseline=$baseline_schema"
        log_warning "Baseline may need regeneration after schema changes"
    fi

    local warnings=0

    # Thresholds: ratio + absolute ceiling (catches bad baselines)
    # Format: check_threshold <name> <current> <baseline> <ratio_thresh> <abs_ceiling> <higher_is_worse>

    # Performance gate metrics (warn if >1.25x baseline OR >500ms absolute)
    for key in "3D_channel_MG" "2D_channel_MG" "3D_channel_HYPRE" "3D_channel_FFT" "3D_duct_FFT1D"; do
        local current_val baseline_val
        current_val=$(grep -oP "\"perf_gate\.$key\".*?\"ms_per_step\":\s*\K[0-9.e+-]+" "$METRICS_FILE" 2>/dev/null | head -1 || true)
        baseline_val=$(grep -oP "\"perf_gate\.$key\".*?\"ms_per_step\":\s*\K[0-9.e+-]+" "$BASELINE_FILE" 2>/dev/null | head -1 || true)
        check_threshold "perf_gate.$key" "$current_val" "$baseline_val" 1.25 500 1 || warnings=$((warnings + 1))
    done

    # MMS convergence rate (warn if <0.9x baseline OR <1.5 absolute - should be ~2)
    local current_rate baseline_rate
    current_rate=$(grep -oP '"mms".*?"spatial_order":\s*\K[0-9.e+-]+' "$METRICS_FILE" 2>/dev/null | head -1 || true)
    baseline_rate=$(grep -oP '"mms".*?"spatial_order":\s*\K[0-9.e+-]+' "$BASELINE_FILE" 2>/dev/null | head -1 || true)
    # For "lower is worse", we check if ratio < threshold
    if [ -n "$current_rate" ]; then
        # Absolute floor check
        local below_floor
        below_floor=$(echo "$current_rate" | awk '{if($1<1.5) print 1; else print 0}')
        if [ "$below_floor" = "1" ]; then
            log_warning "mms.spatial_order = $current_rate below floor 1.5 (expected ~2.0)"
            warnings=$((warnings + 1))
        elif [ -n "$baseline_rate" ]; then
            local ratio
            ratio=$(echo "$current_rate $baseline_rate" | awk '{if($2>0) printf "%.3f", $1/$2; else print "0"}')
            local bad_ratio
            bad_ratio=$(echo "$ratio" | awk '{if($1<0.9) print 1; else print 0}')
            if [ "$bad_ratio" = "1" ]; then
                log_warning "mms.spatial_order = $current_rate (${ratio}x baseline $baseline_rate)"
                warnings=$((warnings + 1))
            fi
        fi
    fi

    # Divergence metrics (warn if >10x baseline OR >1e-5 absolute)
    for key in "tgv_2d" "tgv_3d"; do
        local current_div baseline_div
        current_div=$(grep -oP "\"$key\".*?\"div_Linf\":\s*\K[0-9.e+-]+" "$METRICS_FILE" 2>/dev/null | head -1 || true)
        baseline_div=$(grep -oP "\"$key\".*?\"div_Linf\":\s*\K[0-9.e+-]+" "$BASELINE_FILE" 2>/dev/null | head -1 || true)
        check_threshold "$key.div_Linf" "$current_div" "$baseline_div" 10.0 1e-5 1 || warnings=$((warnings + 1))
    done

    # CPU/GPU norms (warn if >10x baseline OR >1e-8 absolute)
    local current_u_rel baseline_u_rel
    current_u_rel=$(grep -oP '"cpu_gpu".*?"u_rel_L2":\s*\K[0-9.e+-]+' "$METRICS_FILE" 2>/dev/null | head -1 || true)
    baseline_u_rel=$(grep -oP '"cpu_gpu".*?"u_rel_L2":\s*\K[0-9.e+-]+' "$BASELINE_FILE" 2>/dev/null | head -1 || true)
    check_threshold "cpu_gpu.u_rel_L2" "$current_u_rel" "$baseline_u_rel" 10.0 1e-8 1 || warnings=$((warnings + 1))

    # Repeatability (warn if >10x baseline OR >1e-10 absolute)
    local current_rep baseline_rep
    current_rep=$(grep -oP '"repeatability".*?"ke_rel_diff":\s*\K[0-9.e+-]+' "$METRICS_FILE" 2>/dev/null | head -1 || true)
    baseline_rep=$(grep -oP '"repeatability".*?"ke_rel_diff":\s*\K[0-9.e+-]+' "$BASELINE_FILE" 2>/dev/null | head -1 || true)
    check_threshold "repeatability.ke_rel_diff" "$current_rep" "$baseline_rep" 10.0 1e-10 1 || warnings=$((warnings + 1))

    if [ $warnings -gt 0 ]; then
        log_warning "$warnings QoI trend warning(s) detected (see above)"
        log_warning "This is informational only - CI will not fail for trend regressions"
    else
        log_success "All QoI within expected ranges vs baseline"
    fi
}

# Run baseline comparison (warning-only, never fails CI)
compare_to_baseline || true

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}FAILED TESTS:${NC}"
    echo -e "$FAILED_TESTS"
    echo ""
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
