#!/bin/bash
#
# Temporal Convergence Study: Fix h, vary dt
# Expected: 1st-order convergence (p ≈ 1.0)
#

set -e

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_DIR="$EXAMPLE_DIR/output_temporal"

echo "=========================================================="
echo "TEMPORAL Convergence Study: Fixed Grid, Varying dt"
echo "=========================================================="
echo ""
echo "Fixed: Grid = 128 × 256 (h ≈ 0.008)"
echo "Varying: Time step dt"
echo ""
echo "  1. dt = 0.01000  (coarse)"
echo "  2. dt = 0.00500  (medium, 2× finer)"
echo "  3. dt = 0.00250  (fine, 4× finer)"
echo "  4. dt = 0.00125  (v.fine, 8× finer)"
echo ""
echo "Expected: 1st-order temporal convergence (p ≈ 1.0)"
echo "  → Errors should decrease by ~2× per refinement"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"/{dt_coarse,dt_medium,dt_fine,dt_vfine}

# Run each time step
timesteps=(
    "dt_coarse:dt_0p01.cfg"
    "dt_medium:dt_0p005.cfg"
    "dt_fine:dt_0p0025.cfg"
    "dt_vfine:dt_0p00125.cfg"
)

cd "$BUILD_DIR"

for dt_config in "${timesteps[@]}"; do
    IFS=':' read -r dt_level cfg <<< "$dt_config"
    
    echo "=========================================="
    echo "Running: $dt_level"
    echo "=========================================="
    
    ./channel --config "$EXAMPLE_DIR/temporal_convergence_configs/$cfg" \
              --output "$OUTPUT_DIR/$dt_level/"
    
    echo "✓ $dt_level complete"
    echo ""
done

echo "=========================================================="
echo "Temporal convergence study complete!"
echo "=========================================================="
echo ""

# Run custom analysis for temporal convergence
cd "$EXAMPLE_DIR"
python3 << 'PYEOF'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path('.')
output_dir = script_dir / "output_temporal"

# Read VTK files
def read_vtk_solution(vtk_file):
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    origin = None
    spacing = None
    dims = None
    vector_start = None
    
    for i, line in enumerate(lines):
        if 'DIMENSIONS' in line:
            dims = [int(x) for x in line.split()[1:4]]
        if 'ORIGIN' in line:
            origin = [float(x) for x in line.split()[1:4]]
        if 'SPACING' in line:
            spacing = [float(x) for x in line.split()[1:4]]
        if 'VECTORS' in line:
            vector_start = i + 1
    
    Nx, Ny = dims[0], dims[1]
    n_points = Nx * Ny
    
    x = np.linspace(origin[0], origin[0] + (Nx-1)*spacing[0], Nx)
    y = np.linspace(origin[1], origin[1] + (Ny-1)*spacing[1], Ny)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    velocity = []
    for i in range(vector_start, vector_start + n_points):
        velocity.append([float(x) for x in lines[i].split()])
    velocity = np.array(velocity)
    
    u = velocity[:, 0].reshape(Ny, Nx)
    
    return yy, u, Nx, Ny

def poiseuille_exact(y, dp_dx, nu, H):
    return -(dp_dx) / (2 * nu) * (H**2 - y**2)

# Configuration
dp_dx = -0.001
nu = 0.01
H = 1.0

# Load solutions
dt_values = [0.01, 0.005, 0.0025, 0.00125]
dt_names = ['dt_coarse', 'dt_medium', 'dt_fine', 'dt_vfine']
errors = {}

print("="*70)
print("Temporal Convergence Analysis - Fixed Grid, Varying dt")
print("="*70)
print()

for dt_val, name in zip(dt_values, dt_names):
    vtk_file = output_dir / name / "channel_final.vtk"
    if vtk_file.exists():
        y, u, Nx, Ny = read_vtk_solution(vtk_file)
        
        y_profile = y[:, 0]
        u_profile = np.mean(u, axis=1)
        u_exact = poiseuille_exact(y_profile, dp_dx, nu, H)
        
        error = u_profile - u_exact
        L2_error = np.sqrt(np.mean(error**2))
        Linf_error = np.max(np.abs(error))
        
        errors[name] = {
            'dt': dt_val,
            'L2': L2_error,
            'Linf': Linf_error
        }
        
        print(f"  dt = {dt_val:.5f}:  L2 = {L2_error:.6e}  L∞ = {Linf_error:.6e}")

print()
print("Convergence Order Analysis:")
print("-" * 70)

for i in range(len(dt_names) - 1):
    n1 = dt_names[i]
    n2 = dt_names[i+1]
    
    if n1 in errors and n2 in errors:
        dt1 = errors[n1]['dt']
        dt2 = errors[n2]['dt']
        e1 = errors[n1]['L2']
        e2 = errors[n2]['L2']
        
        r = dt1 / dt2
        if e2 > 0:
            p_obs = np.log(e1 / e2) / np.log(r)
        else:
            p_obs = np.nan
        
        print(f"  dt={dt1:.5f} → dt={dt2:.5f}:  r = {r:.2f}    p = {p_obs:.2f}")

print()
print("Expected order for 1st-order Euler: p ≈ 1.0")
print()

# Save results
output_file = output_dir / "temporal_convergence.png"
plt.figure(figsize=(10, 6))

dt_vals = [errors[n]['dt'] for n in dt_names if n in errors]
L2_vals = [errors[n]['L2'] for n in dt_names if n in errors]

plt.loglog(dt_vals, L2_vals, 'bo-', linewidth=2, markersize=8, label='L2 error')

# Reference slopes
dt_ref = np.array([dt_vals[0], dt_vals[-1]])
slope1 = L2_vals[0] * (dt_ref / dt_vals[0])**1
slope2 = L2_vals[0] * (dt_ref / dt_vals[0])**2

plt.loglog(dt_ref, slope1, 'k--', alpha=0.5, label='1st order (p=1)')
plt.loglog(dt_ref, slope2, 'k:', alpha=0.5, label='2nd order (p=2)')

plt.xlabel('Time Step dt', fontsize=12)
plt.ylabel('L2 Error', fontsize=12)
plt.title('Temporal Convergence (Fixed Grid 128×256)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
print()
PYEOF

echo ""
echo "Analysis plot saved to: output_temporal/temporal_convergence.png"









