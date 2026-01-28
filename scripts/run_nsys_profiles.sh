#!/bin/bash
#
# Nsight Systems Profiling Script for NNCFD Poisson Solvers
# Profiles 4 cases with ~30M grid points each:
#   1. Taylor-Green 3D with FFT solver (fully periodic)
#   2. Channel 3D with HYPRE solver (periodic x/z, walls y)
#   3. Duct with FFT1D solver (periodic x, walls y/z)
#   4. Taylor-Green 3D with MG solver (for FFT vs MG comparison)
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BUILD_DIR="${ROOT_DIR}/build_profile"
CONFIG_DIR="${SCRIPT_DIR}/profile_configs"
OUTPUT_DIR="${ROOT_DIR}/profiles"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if build exists
check_build() {
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found: $BUILD_DIR"
        echo "Please build with profiling enabled first:"
        echo "  mkdir -p build_profile && cd build_profile"
        echo "  cmake .. -DUSE_GPU_OFFLOAD=ON -DGPU_PROFILE_KERNELS=ON -DUSE_HYPRE=ON -DCMAKE_BUILD_TYPE=Release"
        echo "  make -j8 taylor_green_3d channel duct"
        exit 1
    fi
}

# Check for required executables
check_executables() {
    local missing=0
    for exe in taylor_green_3d channel duct; do
        if [ ! -x "$BUILD_DIR/$exe" ]; then
            print_error "Executable not found: $BUILD_DIR/$exe"
            missing=1
        fi
    done
    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# Check for nsys
check_nsys() {
    if ! command -v nsys &> /dev/null; then
        print_error "nsys (Nsight Systems) not found in PATH"
        echo "Please load the NVIDIA HPC SDK or nsys module"
        exit 1
    fi
    print_status "Using nsys: $(which nsys)"
    nsys --version
}

# Run a single profile
run_profile() {
    local name="$1"
    local exe="$2"
    local config="$3"
    local solver="$4"

    print_header "Profiling: $name ($solver solver)"

    local output_base="${OUTPUT_DIR}/${name}_${solver}"
    local exe_path="${BUILD_DIR}/${exe}"
    local config_path="${CONFIG_DIR}/${config}"

    if [ ! -f "$config_path" ]; then
        print_error "Config file not found: $config_path"
        return 1
    fi

    print_status "Executable: $exe_path"
    print_status "Config: $config_path"
    print_status "Output: ${output_base}.nsys-rep"

    # Run nsys with NVTX and CUDA tracing
    # --trace=nvtx,cuda: Capture NVTX ranges and CUDA API/kernels/memory
    # --cuda-memory-usage=true: Track DtoH/HtoD transfers
    # --stats=true: Generate summary statistics
    # Note: wrap in 'if' because set -e would exit before we can check $?
    if nsys profile \
        --trace=nvtx,cuda \
        --cuda-memory-usage=true \
        --stats=true \
        --force-overwrite=true \
        --output="$output_base" \
        "$exe_path" --config "$config_path"; then

        print_status "Profile completed: ${output_base}.nsys-rep"

        # Generate stats report
        print_status "Generating stats report..."
        nsys stats "${output_base}.nsys-rep" > "${output_base}_stats.txt" 2>&1 || true

        # Extract NVTX summary if available
        print_status "Extracting NVTX summary..."
        nsys stats --report nvtx_pushpop_sum "${output_base}.nsys-rep" > "${output_base}_nvtx.txt" 2>&1 || true

        # Extract CUDA memory operations
        print_status "Extracting CUDA memory operations..."
        nsys stats --report cuda_gpu_mem_time_sum "${output_base}.nsys-rep" > "${output_base}_memops.txt" 2>&1 || true

        # Extract CUDA kernel summary
        print_status "Extracting CUDA kernel summary..."
        nsys stats --report cuda_gpu_kern_sum "${output_base}.nsys-rep" > "${output_base}_kernels.txt" 2>&1 || true
    else
        print_error "Profile failed for $name"
        return 1
    fi
}

# Generate comparison summary
generate_summary() {
    print_header "Generating Summary Report"

    local summary_file="${OUTPUT_DIR}/profiling_summary.txt"

    {
        echo "NNCFD Poisson Solver Profiling Summary"
        echo "======================================"
        echo ""
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo ""
        echo "Cases Profiled:"
        echo "  1. Taylor-Green 3D (312^3 = 30.4M) - FFT solver"
        echo "  2. Channel 3D (384x256x312 = 30.7M) - HYPRE solver"
        echo "  3. Duct (256x340x350 = 30.5M) - FFT1D solver"
        echo "  4. Taylor-Green 3D (312^3 = 30.4M) - MG solver"
        echo ""
        echo "Time Stepping: 20 iterations total (10 warmup + 10 profiled)"
        echo ""

        # Extract key metrics from each profile
        for profile in taylor_green_fft channel_hypre duct_fft1d taylor_green_mg; do
            echo "----------------------------------------"
            echo "Profile: $profile"
            echo "----------------------------------------"

            local nvtx_file="${OUTPUT_DIR}/${profile}_nvtx.txt"
            if [ -f "$nvtx_file" ]; then
                echo ""
                echo "NVTX Ranges (Poisson-related):"
                grep -i "poisson\|mg:\|fft" "$nvtx_file" 2>/dev/null | head -20 || echo "  No Poisson NVTX data"
            fi

            local memops_file="${OUTPUT_DIR}/${profile}_memops.txt"
            if [ -f "$memops_file" ]; then
                echo ""
                echo "Memory Operations (top 10):"
                head -20 "$memops_file" 2>/dev/null || echo "  No memory data"
            fi
            echo ""
        done

    } > "$summary_file"

    print_status "Summary written to: $summary_file"
    cat "$summary_file"
}

# Main execution
main() {
    print_header "NNCFD Nsight Systems Profiling"

    # Checks
    check_build
    check_executables
    check_nsys

    # Run profiles
    # Case 1: Taylor-Green with FFT
    run_profile "taylor_green" "taylor_green_3d" "profile_taylor_green_fft.cfg" "fft"

    # Case 2: Channel with HYPRE
    run_profile "channel" "channel" "profile_channel_hypre.cfg" "hypre"

    # Case 3: Duct with FFT1D
    run_profile "duct" "duct" "profile_duct_fft1d.cfg" "fft1d"

    # Case 4: Taylor-Green with MG
    run_profile "taylor_green" "taylor_green_3d" "profile_taylor_green_mg.cfg" "mg"

    # Generate summary
    generate_summary

    print_header "Profiling Complete"
    echo ""
    print_status "Profile files: ${OUTPUT_DIR}/*.nsys-rep"
    print_status "Summary: ${OUTPUT_DIR}/profiling_summary.txt"
    echo ""
    echo "To view profiles interactively:"
    echo "  nsys-ui ${OUTPUT_DIR}/taylor_green_fft.nsys-rep"
    echo ""
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help"
        echo "  --fft          Run only FFT profile"
        echo "  --hypre        Run only HYPRE profile"
        echo "  --fft1d        Run only FFT1D profile"
        echo "  --mg           Run only MG profile"
        echo ""
        echo "Without options, runs all 4 profiles."
        exit 0
        ;;
    --fft)
        check_build && check_executables && check_nsys
        run_profile "taylor_green" "taylor_green_3d" "profile_taylor_green_fft.cfg" "fft"
        ;;
    --hypre)
        check_build && check_executables && check_nsys
        run_profile "channel" "channel" "profile_channel_hypre.cfg" "hypre"
        ;;
    --fft1d)
        check_build && check_executables && check_nsys
        run_profile "duct" "duct" "profile_duct_fft1d.cfg" "fft1d"
        ;;
    --mg)
        check_build && check_executables && check_nsys
        run_profile "taylor_green" "taylor_green_3d" "profile_taylor_green_mg.cfg" "mg"
        ;;
    *)
        main
        ;;
esac
