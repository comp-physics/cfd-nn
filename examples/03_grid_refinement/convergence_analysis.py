#!/usr/bin/env python3
"""
Grid convergence analysis using Richardson extrapolation.

Computes:
- Order of accuracy (should be ~2 for 2nd-order schemes)
- Grid Convergence Index (GCI) for error estimation
- Extrapolated "exact" solution
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_vtk_solution(vtk_file):
    """Read velocity field from VTK file."""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
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
    
    # Generate coordinate grid from STRUCTURED_POINTS metadata
    x = np.linspace(origin[0], origin[0] + (Nx-1)*spacing[0], Nx)
    y = np.linspace(origin[1], origin[1] + (Ny-1)*spacing[1], Ny)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    # Read velocity
    velocity = []
    for i in range(vector_start, vector_start + n_points):
        velocity.append([float(x) for x in lines[i].split()])
    velocity = np.array(velocity)
    
    u = velocity[:, 0].reshape(Ny, Nx)
    
    return yy, u, Nx, Ny

def poiseuille_exact(y, dp_dx, nu, H):
    """Analytical Poiseuille solution for channel with half-height H.
    u(y) = -(dp/dx)/(2*nu) * (H^2 - y^2) where y is measured from centerline.
    """
    return -(dp_dx) / (2 * nu) * (H**2 - y**2)

def interpolate_to_coarsest(y_fine, u_fine, y_coarse):
    """Interpolate fine solution onto coarse grid for comparison."""
    from scipy.interpolate import interp1d
    
    # Average over x direction
    y_1d = y_fine[:, 0]
    u_1d = np.mean(u_fine, axis=1)
    
    # Interpolate onto coarse grid
    f = interp1d(y_1d, u_1d, kind='cubic')
    u_interp = f(y_coarse[:, 0])
    
    return u_interp

def compute_gci(error_fine, error_medium, r, p):
    """
    Compute Grid Convergence Index.
    
    Args:
        error_fine: Error on fine grid
        error_medium: Error on medium grid
        r: Grid refinement ratio
        p: Order of accuracy
    
    Returns:
        GCI: Grid convergence index (estimated error bound)
    """
    Fs = 1.25  # Safety factor (1.25 for 3+ grids, 3.0 for 2 grids)
    GCI = Fs * abs(error_fine) / (r**p - 1)
    return GCI

def richardson_extrapolation(u_coarse, u_medium, u_fine, r, p):
    """Richardson extrapolation to estimate exact solution."""
    # Use finest two grids
    u_extrap = u_fine + (u_fine - u_medium) / (r**p - 1)
    return u_extrap

def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    print("="*70)
    print("Grid Convergence Analysis - Richardson Extrapolation")
    print("="*70)
    print()
    
    # Configuration
    dp_dx = -0.001
    nu = 0.01
    H = 1.0  # Channel half-height (domain is y ∈ [-1, 1])
    
    # Grid levels
    grids = {
        'coarse': (32, 64),
        'medium': (64, 128),
        'fine': (128, 256),
        'very_fine': (256, 512)
    }
    
    # Load solutions
    solutions = {}
    print("Loading simulation results...")
    for name, (Nx_expected, Ny_expected) in grids.items():
        vtk_file = output_dir / name / "channel_final.vtk"
        if vtk_file.exists():
            try:
                y, u, Nx, Ny = read_vtk_solution(vtk_file)
                if Nx == Nx_expected and Ny == Ny_expected:
                    solutions[name] = {'y': y, 'u': u, 'Nx': Nx, 'Ny': Ny}
                    print(f"  [OK] {name:12s} {Nx:4d} x {Ny:4d}")
                else:
                    print(f"  [FAIL] {name:12s} Wrong dimensions: {Nx} x {Ny}")
            except Exception as e:
                print(f"  [FAIL] {name:12s} Error: {e}")
        else:
            print(f"  [FAIL] {name:12s} File not found")
    
    if len(solutions) < 2:
        print("\nERROR: Need at least 2 grid levels!")
        print("Run: ./run_refinement.sh")
        sys.exit(1)
    
    print()
    
    # Compute errors against analytical solution
    errors = {}
    print("Computing errors vs analytical solution...")
    for name in solutions:
        y = solutions[name]['y']
        u = solutions[name]['u']
        
        # Centerline profile
        y_profile = y[:, 0]  # Already in physical coordinates [-H, H]
        u_profile = np.mean(u, axis=1)
        
        # Analytical solution (Poiseuille expects y relative to centerline)
        u_exact = poiseuille_exact(y_profile, dp_dx, nu, H)
        
        # Error metrics
        error = u_profile - u_exact
        L2_error = np.sqrt(np.mean(error**2))
        Linf_error = np.max(np.abs(error))
        
        errors[name] = {
            'y': y_profile,
            'u_numerical': u_profile,
            'u_exact': u_exact,
            'L2': L2_error,
            'Linf': Linf_error,
            'h': 2.0 / (solutions[name]['Ny'] - 1)  # Grid spacing in y
        }
        
        print(f"  {name:12s} L2 = {L2_error:.6e}  L∞ = {Linf_error:.6e}")
    
    print()
    
    # Compute convergence order
    grid_names = ['coarse', 'medium', 'fine', 'very_fine']
    available = [g for g in grid_names if g in errors]
    
    if len(available) >= 2:
        print("Convergence Order Analysis (Richardson):")
        print("-" * 70)
        
        for i in range(len(available) - 1):
            g1 = available[i]    # Coarser
            g2 = available[i+1]  # Finer
            
            h1 = errors[g1]['h']
            h2 = errors[g2]['h']
            e1 = errors[g1]['L2']
            e2 = errors[g2]['L2']
            
            r = h1 / h2  # Refinement ratio
            
            # Observed order of accuracy: p = log(e1/e2) / log(r)
            if e2 > 0:
                p_obs = np.log(e1 / e2) / np.log(r)
            else:
                p_obs = np.nan
            
            print(f"  {g1:12s} → {g2:12s}:  r = {r:.2f}    p = {p_obs:.2f}")
        
        print()
        print("Expected order for 2nd-order scheme: p ≈ 2.0")
        print()
    
    # Plot convergence
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Velocity profiles
    colors = {'coarse': 'red', 'medium': 'orange', 'fine': 'green', 'very_fine': 'blue'}
    for name in available:
        ax1.plot(errors[name]['u_numerical'], errors[name]['y'],
                 color=colors[name], linewidth=2, label=f'{name} ({solutions[name]["Ny"]} pts)')
    
    # Analytical solution (using finest grid)
    if available:
        finest = available[-1]
        ax1.plot(errors[finest]['u_exact'], errors[finest]['y'],
                 'k--', linewidth=2, label='Analytical')
    
    ax1.set_xlabel('Velocity u', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('Velocity Profiles', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Error profiles
    for name in available:
        error_profile = errors[name]['u_numerical'] - errors[name]['u_exact']
        ax2.plot(error_profile, errors[name]['y'],
                 color=colors[name], linewidth=2, label=name)
    
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Error (u_num - u_exact)', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Error Profiles', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence plot (log-log)
    h_values = [errors[g]['h'] for g in available]
    L2_values = [errors[g]['L2'] for g in available]
    Linf_values = [errors[g]['Linf'] for g in available]
    
    ax3.loglog(h_values, L2_values, 'bo-', linewidth=2, markersize=8, label='L2 error')
    ax3.loglog(h_values, Linf_values, 'rs-', linewidth=2, markersize=8, label='L∞ error')
    
    # Reference slopes
    h_ref = np.array([h_values[0], h_values[-1]])
    slope1 = L2_values[0] * (h_ref / h_values[0])**1
    slope2 = L2_values[0] * (h_ref / h_values[0])**2
    
    ax3.loglog(h_ref, slope1, 'k:', alpha=0.5, label='1st order')
    ax3.loglog(h_ref, slope2, 'k--', alpha=0.5, label='2nd order')
    
    ax3.set_xlabel('Grid Spacing h', fontsize=11)
    ax3.set_ylabel('Error', fontsize=11)
    ax3.set_title('Convergence Rate', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.invert_xaxis()
    
    # 4. Summary table
    ax4.axis('off')
    
    table_data = [['Grid', 'Nx × Ny', 'h', 'L2 Error', 'L∞ Error']]
    for name in available:
        Nx = solutions[name]['Nx']
        Ny = solutions[name]['Ny']
        h = errors[name]['h']
        L2 = errors[name]['L2']
        Linf = errors[name]['Linf']
        table_data.append([name, f'{Nx}×{Ny}', f'{h:.4f}', f'{L2:.2e}', f'{Linf:.2e}'])
    
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.2, 0.2, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header formatting
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Convergence Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / "convergence_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Try to display
    try:
        plt.show()
    except Exception:
        pass  # Ignore display errors in headless/CI environments; plot already saved
    
    print()
    print("="*70)
    print("Analysis Complete!")
    print("="*70)
    print()
    print("Interpretation:")
    print("  • Observed order p ≈ 2.0 → 2nd order accuracy (good!)")
    print("  • Observed order p < 2.0 → Check numerical scheme")
    print("  • Observed order p > 2.0 → Lucky cancellation or smooth solution")
    print()

if __name__ == '__main__':
    # Check for scipy availability
    import importlib.util
    if importlib.util.find_spec('scipy') is None:
        print("WARNING: scipy not found - install for full functionality")
        print("  pip install scipy")
        print()
    
    main()

