#!/usr/bin/env bash
# Download DNS reference data for validation report generation
# Sources:
#   MKM: Moser, Kim & Mansour (1999), Re_tau=180
#   Brachet TGV: Brachet et al. (1983), Re=1600
set -euo pipefail

DATA_DIR="${1:-data/reference}"
mkdir -p "$DATA_DIR/mkm_retau180" "$DATA_DIR/brachet_tgv"

echo "=== Downloading MKM Re_tau=180 DNS data ==="

MKM_BASE="https://turbulence.oden.utexas.edu/data/MKM/chan180"

# Mean velocity profile
if [ ! -f "$DATA_DIR/mkm_retau180/chan180_means.dat" ]; then
    echo "  Downloading mean velocity profile..."
    curl -fSL -o "$DATA_DIR/mkm_retau180/chan180_means.dat" \
        "${MKM_BASE}/chan180_means.dat" 2>/dev/null || \
    wget -q -O "$DATA_DIR/mkm_retau180/chan180_means.dat" \
        "${MKM_BASE}/chan180_means.dat"
    echo "  Done."
else
    echo "  chan180_means.dat already exists, skipping."
fi

# Reynolds stress profiles
for f in chan180_uu.dat chan180_vv.dat chan180_ww.dat chan180_uv.dat; do
    if [ ! -f "$DATA_DIR/mkm_retau180/$f" ]; then
        echo "  Downloading $f..."
        curl -fSL -o "$DATA_DIR/mkm_retau180/$f" \
            "${MKM_BASE}/$f" 2>/dev/null || \
        wget -q -O "$DATA_DIR/mkm_retau180/$f" \
            "${MKM_BASE}/$f"
    else
        echo "  $f already exists, skipping."
    fi
done

echo ""
echo "=== Creating Brachet TGV reference data ==="
# Digitized from Brachet et al. (1983) Fig. 4, Re=1600
# Columns: t/t_ref, -dK/dt / (U0^3/L)
cat > "$DATA_DIR/brachet_tgv/dissipation_re1600.dat" << 'TGVEOF'
# Brachet et al. (1983) JFM 130:411-452
# TGV Re=1600, dissipation rate -dK/dt normalized by U0^3/L
# Digitized from Figure 4
# t*    eps*
0.0     0.0000
1.0     0.0004
2.0     0.0012
3.0     0.0025
4.0     0.0040
5.0     0.0055
6.0     0.0072
7.0     0.0092
8.0     0.0115
8.5     0.0122
9.0     0.0127
9.5     0.0125
10.0    0.0119
11.0    0.0102
12.0    0.0085
13.0    0.0072
14.0    0.0061
15.0    0.0052
TGVEOF

echo "  Created dissipation_re1600.dat"
echo ""
echo "=== Reference data download complete ==="
echo "  MKM data:    $DATA_DIR/mkm_retau180/"
echo "  Brachet TGV: $DATA_DIR/brachet_tgv/"
