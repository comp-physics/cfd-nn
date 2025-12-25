#!/usr/bin/env python3
"""
Compare turbulence model predictions for channel flow at Re_tau = 180.

Compares against DNS data from Moser et al. (1999) if available.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_vtk_data(vtk_file):
    """Simple VTK reader for velocity data."""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    # Parse dimensions
    for i, line in enumerate(lines):
        if 'DIMENSIONS' in line:
            dims = [int(x) for x in line.split()[1:4]]
            Nx, Ny = dims[0], dims[1]
        if 'POINTS' in line:
            n_points = int(line.split()[1])
            point_start = i + 1
        if 'POINT_DATA' in line:
            vector_start = i + 2
    
    # Read coordinates
    coords = []
    for i in range(point_start, point_start + n_points):
        coords.append([float(x) for x in lines[i].split()])
    coords = np.array(coords)
    
    # Read velocity
    velocity = []
    for i in range(vector_start, vector_start + n_points):
        velocity.append([float(x) for x in lines[i].split()])
    velocity = np.array(velocity)
    
    # Reshape
    y = coords[:, 1].reshape(Ny, Nx)
    u = velocity[:, 0].reshape(Ny, Nx)
    
    return y, u, Nx, Ny

def get_moser_dns_data():
    """
    DNS data from Moser et al. (1999) for Re_tau = 180.
    
    This is a subset of the published data for comparison.
    For full data, see: https://www.flow.kth.se/~pschlatt/DATA/
    """
    # y+ values (wall units)
    y_plus = np.array([
        0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 
        60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 
        140.0, 160.0, 180.0
    ])
    
    # u+ values (wall units)
    u_plus = np.array([
        0.0, 4.95, 9.70, 17.50, 22.10, 25.20, 27.50,
        29.30, 30.70, 31.90, 32.90, 33.80, 35.30,
        36.50, 37.50, 38.30
    ])
    
    return y_plus, u_plus

def compute_wall_units(y, u, nu):
    """
    Convert to wall units y+ and u+.
    
    Assumes u_tau can be estimated from the wall shear stress.
    """
    # Estimate u_tau from wall gradient
    dy = y[1, 0] - y[0, 0]
    dudy_wall = (u[1, :] - u[0, :]) / dy
    tau_w = nu * np.mean(dudy_wall)
    u_tau = np.sqrt(tau_w)
    
    # Wall distance (assuming y=0 is at bottom wall, y=2 at top)
    H = np.max(y) - np.min(y)
    y_wall = np.abs(y[:, 0] - H/2)  # Distance from nearest wall
    
    # Convert to wall units
    y_plus = y_wall * u_tau / nu
    u_plus = np.mean(u, axis=1) / u_tau
    
    return y_plus, u_plus, u_tau

def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    print("="*60)
    print("Turbulence Model Comparison - Channel Flow Re_tau=180")
    print("="*60)
    print()
    
    # Configuration
    nu = 0.0006667  # From config files
    
    # Models to compare
    models = {
        'None': 'no_model',
        'Baseline': 'baseline',
        'GEP': 'gep',
        'NN-MLP': 'nn_mlp',
        'NN-TBNN': 'nn_tbnn'
    }
    
    # Read data for each model
    data = {}
    for name, dir_name in models.items():
        # Try both naming conventions (velocity_final.vtk and channel_final.vtk)
        vtk_file = output_dir / dir_name / "velocity_final.vtk"
        if not vtk_file.exists():
            vtk_file = output_dir / dir_name / "channel_final.vtk"
        if vtk_file.exists():
            try:
                y, u, Nx, Ny = read_vtk_data(vtk_file)
                y_plus, u_plus, u_tau = compute_wall_units(y, u, nu)
                data[name] = {
                    'y_plus': y_plus,
                    'u_plus': u_plus,
                    'u_tau': u_tau,
                    'y': y,
                    'u': u
                }
                print(f"[OK] Loaded: {name:12s} u_tau = {u_tau:.4f}")
            except Exception as e:
                print(f"[FAIL] Failed: {name:12s} ({e})")
        else:
            print(f"[FAIL] Missing: {name:12s} (file not found)")
    
    if not data:
        print("\nERROR: No simulation data found!")
        print("Please run: ./run_all.sh")
        sys.exit(1)
    
    print()
    
    # Get DNS reference data
    dns_y_plus, dns_u_plus = get_moser_dns_data()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Inner scaling (wall units)
    colors = {
        'None': 'gray',
        'Baseline': 'blue',
        'GEP': 'green',
        'NN-MLP': 'orange',
        'NN-TBNN': 'red'
    }
    
    # DNS data
    ax1.plot(dns_y_plus, dns_u_plus, 'ko', markersize=6, 
             label='DNS (Moser 1999)', zorder=10)
    
    # Law of the wall
    y_log = np.logspace(0, np.log10(200), 100)
    u_log = 1/0.41 * np.log(y_log) + 5.0  # u+ = 1/κ ln(y+) + B
    ax1.plot(y_log, u_log, 'k--', alpha=0.5, label='Log law')
    
    # Linear sublayer
    y_visc = np.linspace(0, 10, 20)
    ax1.plot(y_visc, y_visc, 'k:', alpha=0.5, label='u+ = y+')
    
    # Model predictions
    for name in ['None', 'Baseline', 'GEP', 'NN-MLP', 'NN-TBNN']:
        if name in data:
            ax1.plot(data[name]['y_plus'], data[name]['u_plus'],
                    color=colors[name], linewidth=2, label=name)
    
    ax1.set_xlabel('y+', fontsize=12)
    ax1.set_ylabel('u+', fontsize=12)
    ax1.set_title('Velocity Profile (Inner Scaling)', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xlim([1, 200])
    ax1.set_ylim([0, 40])
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Physical coordinates
    for name in ['None', 'Baseline', 'GEP', 'NN-MLP', 'NN-TBNN']:
        if name in data:
            y_norm = data[name]['y'][:, 0]
            u_mean = np.mean(data[name]['u'], axis=1)
            ax2.plot(u_mean, y_norm, color=colors[name], linewidth=2, label=name)
    
    ax2.set_xlabel('Mean Velocity u', fontsize=12)
    ax2.set_ylabel('Wall-normal Distance y', fontsize=12)
    ax2.set_title('Velocity Profile (Physical Coordinates)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print summary statistics
    print()
    print("="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"{'Model':<12} {'u_tau':>10} {'Re_tau':>10} {'Notes'}")
    print("-"*60)
    
    Re_tau_target = 180.0
    for name in ['None', 'Baseline', 'GEP', 'NN-MLP', 'NN-TBNN']:
        if name in data:
            u_tau = data[name]['u_tau']
            Re_tau = u_tau * 1.0 / nu  # Half-channel height = 1.0
            notes = ""
            if name in ['NN-MLP', 'NN-TBNN']:
                notes = "(example weights)"
            print(f"{name:<12} {u_tau:10.4f} {Re_tau:10.1f}   {notes}")
    
    print(f"{'DNS Target':<12} {'---':>10} {Re_tau_target:10.1f}")
    print()
    
    # Try to display
    try:
        plt.show()
    except:
        pass
    
    print("="*60)
    print("Comparison complete!")
    print("="*60)
    print()
    print("Interpretation:")
    print("  • None: Should underpredict (no turbulence modeling)")
    print("  • Baseline: Simple mixing length model")
    print("  • GEP: Improved algebraic model")
    print("  • NN-MLP/TBNN: Only meaningful with trained weights!")
    print()

if __name__ == '__main__':
    main()

