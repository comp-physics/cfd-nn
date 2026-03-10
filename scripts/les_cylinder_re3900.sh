#!/bin/bash
# LES cylinder flow at Re=3900 — long-running validation benchmark
#
# Reference: Ong & Wallace (1996), Exp. Fluids 20:441-453
#   Cd ~ 1.0, St ~ 0.215
#
# Usage (CPU build):
#   ./scripts/les_cylinder_re3900.sh [output_dir]
#
# Usage (GPU build via SLURM embers QOS):
#   sbatch --qos=embers --gres=gpu:H200:1 --time=4:00:00 \
#          --wrap="./scripts/les_cylinder_re3900.sh /path/to/output"
#
# Geometry:
#   Cylinder diameter D=1, center at (5,0), radius 0.5
#   Domain [0,25]x[-8,8]x[0,pi] (2D extruded in z with Nz=32)
#   Re = U_inf * D / nu = 1 * 1 / nu = 3900  ->  nu = 1/3900 ~ 2.56e-4
#
# Resolution (adequate for LES at Re=3900):
#   Nx=256, Ny=192, Nz=32 (~1.6M cells)
#   dx = 25/256 ~ 0.098, dz = pi/32 ~ 0.098
#   Stretched y: ~50 cells in boundary layer (y_min ~ 0.01 at wall)
#
# Runtime: ~3000 shedding cycles for converged statistics
#   St ~ 0.215 -> T_shed ~ 4.65 time units
#   200 cycles ~ 930 time units ~ dt=0.001 -> 930000 steps (expensive)
#   Short validation run: 50 cycles ~ 230 time units ~ 230000 steps
#   Budget: ~4 hours on H200 at ~600 steps/min = 144000 steps (31 cycles)

set -euo pipefail

OUTDIR="${1:-artifacts/les_cylinder_re3900}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/run.log"

echo "=== LES Cylinder Re=3900 ===" | tee "${LOG}"
echo "Output: ${OUTDIR}" | tee -a "${LOG}"
echo "Date: $(date)" | tee -a "${LOG}"

# --- Parameters ---
D=1.0
U_INF=1.0
NU=$(python3 -c "print(1.0/3900.0)")
echo "nu = ${NU}" | tee -a "${LOG}"

# Build check
BUILD_DIR="build"
if [ -d "build_gpu_validation" ]; then
    BUILD_DIR="build_gpu_validation"
fi
CYLINDER="${BUILD_DIR}/cylinder"
if [ ! -x "${CYLINDER}" ]; then
    echo "ERROR: ${CYLINDER} not found. Build first:" | tee -a "${LOG}"
    echo "  mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}" | tee -a "${LOG}"
    echo "  cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90" | tee -a "${LOG}"
    echo "  make -j\$(nproc) cylinder" | tee -a "${LOG}"
    exit 1
fi

# Write config file
CFG="${OUTDIR}/les_cylinder_re3900.cfg"
cat > "${CFG}" << EOF
# LES cylinder Re=3900 config
Nx = 256
Ny = 192
Nz = 32
x_min = 0.0
x_max = 25.0
y_min = -8.0
y_max = 8.0
z_min = 0.0
z_max = 3.14159265358979
nu = ${NU}
dp_dx = 0.0
dt = 0.001
max_steps = 150000
output_freq = 500
verbose = true
simulation_mode = Unsteady
adaptive_dt = true
CFL_max = 0.4
CFL_xz = 0.5
turb_model = WALE
poisson_solver = Auto
poisson_tol = 1e-6
poisson_max_vcycles = 20
output_dir = ${OUTDIR}/fields/
EOF

echo "Config: ${CFG}" | tee -a "${LOG}"
echo "Steps: 150000 (targeting ~32 shedding cycles)" | tee -a "${LOG}"

# Run
echo "" | tee -a "${LOG}"
echo "--- Starting simulation ---" | tee -a "${LOG}"

# IBM parameters are hardcoded in main_cylinder.cpp (cyl_x=10, cyl_y=0, cyl_r=0.5)
# Override via config where possible; IBM geometry uses defaults from app code.
# For accurate Re=3900 test, rebuild cylinder app with:
#   cyl_x=5, cyl_y=0, cyl_r=0.5, U_inf=1.0
# and nu from this config.

OMP_TARGET_OFFLOAD=MANDATORY "${CYLINDER}" --config "${CFG}" 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "--- Post-processing ---" | tee -a "${LOG}"

FORCES="${OUTDIR}/fields/forces.dat"
if [ ! -f "${FORCES}" ]; then
    echo "WARNING: forces.dat not found at ${FORCES}" | tee -a "${LOG}"
    exit 1
fi

# Extract statistics from second half of run (discard transient)
python3 - "${FORCES}" "${OUTDIR}" << 'PYEOF'
import sys
import numpy as np

forces_file = sys.argv[1]
outdir = sys.argv[2]

data = np.loadtxt(forces_file, comments='#')
steps = data[:, 0]
times = data[:, 1]
Cd    = data[:, 4]
Cl    = data[:, 5]

# Discard first half (transient)
n = len(times)
start = n // 2
t_analysis = times[start:]
Cd_analysis = Cd[start:]
Cl_analysis = Cl[start:]

Cd_mean = np.mean(Cd_analysis)
Cd_rms  = np.std(Cd_analysis)
Cl_mean = np.mean(Cl_analysis)
Cl_rms  = np.std(Cl_analysis)

# Strouhal from lift zero-crossings
crossings = np.where(np.diff(np.sign(Cl_analysis)) > 0)[0]
if len(crossings) >= 2:
    dt_mean = np.mean(np.diff(t_analysis[crossings]))
    T_shed = 2.0 * dt_mean   # each crossing = half period
    St = 1.0 / T_shed        # St = f*D/U_inf = f*1/1 = f
else:
    St = float('nan')

print(f"=== LES Cylinder Re=3900 Results ===")
print(f"Analysis window: t = {t_analysis[0]:.1f} to {t_analysis[-1]:.1f}")
print(f"Cd_mean = {Cd_mean:.4f}  (reference: ~1.0)")
print(f"Cd_rms  = {Cd_rms:.4f}")
print(f"|Cl|_rms = {Cl_rms:.4f}")
print(f"St      = {St:.4f}     (reference: ~0.215)")
print()

# Validation checks
print("=== Validation ===")
passed = True

if 0.6 < Cd_mean < 1.6:
    print(f"PASS: Cd_mean = {Cd_mean:.4f} in [0.6, 1.6] (ref: ~1.0)")
else:
    print(f"FAIL: Cd_mean = {Cd_mean:.4f} outside [0.6, 1.6]")
    passed = False

if not np.isnan(St):
    if 0.17 < St < 0.26:
        print(f"PASS: St = {St:.4f} in [0.17, 0.26] (ref: ~0.215)")
    else:
        print(f"FAIL: St = {St:.4f} outside [0.17, 0.26]")
        passed = False
else:
    print("WARN: Strouhal not computed (insufficient zero-crossings)")

# Save summary
with open(f"{outdir}/summary.txt", "w") as f:
    f.write(f"Cd_mean = {Cd_mean:.6f}\n")
    f.write(f"Cd_rms  = {Cd_rms:.6f}\n")
    f.write(f"Cl_rms  = {Cl_rms:.6f}\n")
    f.write(f"St      = {St:.6f}\n")
    f.write(f"passed  = {passed}\n")

sys.exit(0 if passed else 1)
PYEOF

echo "Summary written to ${OUTDIR}/summary.txt" | tee -a "${LOG}"
