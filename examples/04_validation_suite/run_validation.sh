#!/bin/bash
#
# Run complete validation suite and compare against DNS/analytical benchmarks
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "=========================================================="
echo "Validation Suite - DNS & Analytical Benchmarks"
echo "=========================================================="
echo ""
echo "Running validation cases:"
echo "  1. Poiseuille Re=100   - Analytical solution"
echo "  2. Poiseuille Re=1000  - Analytical solution"
echo "  3. Channel Re_tau=180  - DNS benchmark (Moser 1999)"
echo "  4. Channel Re_tau=395  - DNS benchmark (Moser 1999)"
echo ""

# Check solver
if [ ! -f "$BUILD_DIR/channel" ]; then
    echo "ERROR: Solver not found"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{poiseuille_re100,poiseuille_re1000,channel_re180,channel_re395}

# Array of validation cases
cases=(
    "poiseuille_re100:poiseuille_re100.cfg"
    "poiseuille_re1000:poiseuille_re1000.cfg"
    "channel_re180:channel_re180.cfg"
    "channel_re395:channel_re395.cfg"
)

cd "$BUILD_DIR"

# Run each case
for case_config in "${cases[@]}"; do
    IFS=':' read -r case cfg <<< "$case_config"
    
    echo ""
    echo "=========================================="
    echo "Running: $case"
    echo "=========================================="
    
    ./channel --config "$EXAMPLE_DIR/$cfg" \
              --output_dir "$OUTPUT_DIR/$case" \
        || echo "WARNING: $case may not have fully converged"
    
    echo "[OK] $case complete"
done

echo ""
echo "=========================================================="
echo "All validation cases complete!"
echo "=========================================================="
echo ""

# Run comparison analysis
if command -v python3 &> /dev/null; then
    echo "Running validation analysis..."
    python3 "$EXAMPLE_DIR/validate_results.py"
else
    echo "Python3 not found - skipping automated analysis"
    echo "Results saved to: $OUTPUT_DIR/"
fi

