#!/bin/bash
# Download reference data for validation
# Sources:
#   - MKM (Moser, Kim, Mansour) DNS channel data at Re_tau=180, 395, 590
#   - Del Álamo & Jiménez channel DNS at Re_tau=590

set -euo pipefail

DATA_DIR="${1:-data/reference}"
mkdir -p "$DATA_DIR"

echo "=== Downloading MKM channel DNS reference data ==="
MKM_URL="https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz"
MKM_FILE="$DATA_DIR/chandata.tar.gz"

if [ ! -f "$MKM_FILE" ]; then
    echo "Downloading from $MKM_URL ..."
    curl -fSL -o "$MKM_FILE" "$MKM_URL" || {
        echo "WARNING: Failed to download MKM data. URL may have changed."
        echo "Try: $MKM_URL"
    }
else
    echo "MKM data already downloaded: $MKM_FILE"
fi

if [ -f "$MKM_FILE" ]; then
    echo "Extracting MKM data..."
    tar -xzf "$MKM_FILE" -C "$DATA_DIR" 2>/dev/null || {
        echo "WARNING: Failed to extract MKM data."
    }
fi

echo ""
echo "=== Reference data summary ==="
echo "Directory: $DATA_DIR"
ls -la "$DATA_DIR/" 2>/dev/null || echo "(empty)"
echo ""
echo "Expected files after extraction:"
echo "  chan180/  - Re_tau=180 DNS profiles (u_mean, u_rms, uv, etc.)"
echo "  chan395/  - Re_tau=395 DNS profiles"
echo "  chan590/  - Re_tau=590 DNS profiles"
echo ""
echo "For cylinder flow validation, use:"
echo "  Re=100: Cd~1.33, St~0.164 (Park et al., JFM 1998)"
echo "  Re=300: Cd~1.38, St~0.21  (Williamson, JFM 1996)"
echo ""
echo "Done."
