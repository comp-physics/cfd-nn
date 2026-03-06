#!/usr/bin/env bash
# Download DNS reference data for validation report generation
# Sources:
#   MKM: Moser, Kim & Mansour (1999), Re_tau=180, 395, 590
#   Brachet TGV: Brachet et al. (1983), Re=1600
set -euo pipefail

DATA_DIR="${1:-data/reference}"
mkdir -p "$DATA_DIR/mkm_retau180" "$DATA_DIR/brachet_tgv"

echo "=== Downloading MKM Re_tau=180 DNS data ==="

if [ ! -f "$DATA_DIR/mkm_retau180/chan180/profiles/chan180.means" ]; then
    echo "  Downloading complete MKM database (1.2 MB)..."
    curl -fSL -o /tmp/chandata.tar.gz \
        "https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz"
    tar xzf /tmp/chandata.tar.gz -C "$DATA_DIR/mkm_retau180" chan180/profiles/
    rm /tmp/chandata.tar.gz
    echo "  Extracted profiles to $DATA_DIR/mkm_retau180/chan180/profiles/"
else
    echo "  MKM data already exists, skipping."
fi

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
echo "  MKM data:    $DATA_DIR/mkm_retau180/chan180/profiles/"
echo "  Brachet TGV: $DATA_DIR/brachet_tgv/"
echo ""
echo "For cylinder flow validation, use:"
echo "  Re=100: Cd~1.33, St~0.164 (Park et al., JFM 1998)"
echo "  Re=300: Cd~1.38, St~0.21  (Williamson, JFM 1996)"
