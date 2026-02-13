#!/bin/bash
# Interactive run script for GPU turbulence verification
#
# Usage:
#   ./run.sh <case>           Run a specific case
#   ./run.sh all              Run all cases sequentially
#   ./run.sh list             List available cases
#
# For batch execution on H200, use: sbatch run_h200.sbatch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_gpu"

# Check if running interactively on GPU node
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found - are you on a GPU node?"
        echo "For H200, submit via: sbatch run_h200.sbatch"
        return 1
    fi
    return 0
}

# Build executables if needed
build_if_needed() {
    if [ ! -x "$BUILD_DIR/channel" ] || [ ! -x "$BUILD_DIR/duct" ] || [ ! -x "$BUILD_DIR/taylor_green_3d" ]; then
        echo "Building GPU executables..."
        bash "$SCRIPT_DIR/build_gpu.sh"
    fi
}

# Run a specific case
run_case() {
    local case_name="$1"
    local cfg_file=""
    local exe=""

    case "$case_name" in
        tgv_re100|tgv100)
            cfg_file="$SCRIPT_DIR/tgv_re100_validation.cfg"
            exe="$BUILD_DIR/taylor_green_3d"
            ;;
        tgv_re1600|tgv1600|tgv_dns)
            cfg_file="$SCRIPT_DIR/tgv_re1600_dns.cfg"
            exe="$BUILD_DIR/taylor_green_3d"
            ;;
        tgv_re1600_fine|tgv1600_fine)
            cfg_file="$SCRIPT_DIR/tgv_re1600_dns_fine.cfg"
            exe="$BUILD_DIR/taylor_green_3d"
            ;;
        channel_sst|channel)
            cfg_file="$SCRIPT_DIR/channel_retau180_sst.cfg"
            exe="$BUILD_DIR/channel"
            ;;
        channel_earsm)
            cfg_file="$SCRIPT_DIR/channel_retau180_earsm.cfg"
            exe="$BUILD_DIR/channel"
            ;;
        duct_sst|duct)
            cfg_file="$SCRIPT_DIR/duct_turbulent_sst.cfg"
            exe="$BUILD_DIR/duct"
            ;;
        duct_earsm)
            cfg_file="$SCRIPT_DIR/duct_turbulent_earsm.cfg"
            exe="$BUILD_DIR/duct"
            ;;
        *)
            echo "Unknown case: $case_name"
            echo "Run './run.sh list' for available cases"
            return 1
            ;;
    esac

    echo "=============================================="
    echo "Running: $case_name"
    echo "Config: $cfg_file"
    echo "=============================================="
    echo ""

    build_if_needed
    time "$exe" --config "$cfg_file"
}

# List available cases
list_cases() {
    echo "Available turbulence verification cases:"
    echo ""
    echo "  TAYLOR-GREEN VORTEX (all-periodic)"
    echo "    tgv_re100        Re=100 validation (viscous decay)"
    echo "    tgv_re1600       Re=1600 DNS (transition to turbulence)"
    echo "    tgv_re1600_fine  Re=1600 fine grid (128^3)"
    echo ""
    echo "  CHANNEL FLOW (periodic x/z, walls y)"
    echo "    channel_sst      Re_tau=180 with SST k-omega"
    echo "    channel_earsm    Re_tau=180 with EARSM (anisotropic)"
    echo ""
    echo "  SQUARE DUCT (periodic x, walls y/z)"
    echo "    duct_sst         SST k-omega (no secondary flow)"
    echo "    duct_earsm       EARSM (secondary corner vortices)"
    echo ""
    echo "Usage:"
    echo "  ./run.sh <case>           Run interactively (requires GPU node)"
    echo "  ./run.sh all              Run all cases"
    echo "  sbatch run_h200.sbatch    Submit to H200 queue"
}

# Main
case "${1:-}" in
    list|--list|-l)
        list_cases
        ;;
    all)
        check_gpu || exit 1
        shift
        for case in tgv_re100 tgv_re1600 channel_sst channel_earsm duct_sst duct_earsm; do
            run_case "$case" "$@"
            echo ""
        done
        ;;
    ""|--help|-h)
        echo "GPU Turbulence Verification Suite"
        echo ""
        list_cases
        ;;
    *)
        check_gpu || exit 1
        run_case "$@"
        ;;
esac
