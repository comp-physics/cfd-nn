#!/usr/bin/env python3
"""
Analyze laminar channel flow results and compare with analytical Poiseuille solution.

The analytical solution for plane Poiseuille flow is:
    u(y) = (dp/dx) / (2*nu) * (y^2 - H^2/4)
    
where H is the channel height, dp/dx is the pressure gradient (negative for flow in +x).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def poiseuille_analytical(y, dp_dx, nu, H):
    """
    Analytical Poiseuille solution for velocity profile.
    
    Args:
        y: Wall-normal coordinate (y=0 at centerline)
        dp_dx: Pressure gradient (negative for flow in +x)
        nu: Kinematic viscosity
        H: Channel height
    
    Returns:
        u: Streamwise velocity
    """
    return -(dp_dx) / (2 * nu) * (H**2 / 4 - y**2)

def read_simulation_data(output_dir):
    """Read simulation results from output directory."""
    
    # Try to find the final velocity file
    vtk_files = sorted(Path(output_dir).glob("velocity_*.vtk"))
    
    if not vtk_files:
        print(f"ERROR: No VTK files found in {output_dir}")
        return None
    
    final_file = vtk_files[-1]
    print(f"Reading: {final_file}")
    
    # Simple VTK ASCII parser (assumes structured grid format from solver)
    with open(final_file, 'r') as f:
        lines = f.readlines()
    
    # Parse grid dimensions
    for i, line in enumerate(lines):
        if 'DIMENSIONS' in line:
            dims = [int(x) for x in line.split()[1:4]]
            Nx, Ny = dims[0], dims[1]
            print(f"Grid: {Nx} x {Ny}")
        
        if 'POINTS' in line:
            n_points = int(line.split()[1])
            point_data_start = i + 1
        
        if 'POINT_DATA' in line:
            vector_data_start = i + 2  # Skip VECTORS line
    
    # Read coordinates
    coords = []
    for i in range(point_data_start, point_data_start + n_points):
        coords.append([float(x) for x in lines[i].split()])
    coords = np.array(coords)
    
    # Read velocity vectors
    velocity = []
    for i in range(vector_data_start, vector_data_start + n_points):
        velocity.append([float(x) for x in lines[i].split()])
    velocity = np.array(velocity)
    
    # Reshape to 2D grid
    y_coords = coords[:, 1].reshape(Ny, Nx)
    u_velocity = velocity[:, 0].reshape(Ny, Nx)
    
    return y_coords, u_velocity, Nx, Ny

def main():
    # Configuration (should match poiseuille.cfg)
    dp_dx = -0.001
    nu = 0.01
    H = 2.0
    
    # Output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        print("Please run the simulation first: ./run.sh")
        sys.exit(1)
    
    print("="*60)
    print("Laminar Channel Flow - Poiseuille Validation")
    print("="*60)
    print()
    
    # Read simulation data
    result = read_simulation_data(output_dir)
    if result is None:
        sys.exit(1)
    
    y_coords, u_velocity, Nx, Ny = result
    
    # Extract centerline profile (average over x)
    y_profile = y_coords[:, 0] - H/2  # Shift to centerline at y=0
    u_profile = np.mean(u_velocity, axis=1)
    
    # Compute analytical solution
    u_analytical = poiseuille_analytical(y_profile, dp_dx, nu, H)
    
    # Compute error
    error = np.abs(u_profile - u_analytical)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    rel_error = max_error / np.max(u_analytical)
    
    print(f"Physical Parameters:")
    print(f"  Channel height (H): {H}")
    print(f"  Viscosity (nu): {nu}")
    print(f"  Pressure gradient (dp/dx): {dp_dx}")
    print()
    print(f"Maximum velocity:")
    print(f"  Analytical: {np.max(u_analytical):.6f}")
    print(f"  Numerical:  {np.max(u_profile):.6f}")
    print()
    print(f"Error metrics:")
    print(f"  Max error:  {max_error:.2e}")
    print(f"  RMS error:  {rms_error:.2e}")
    print(f"  Relative:   {rel_error*100:.3f}%")
    print()
    
    # Check if error is acceptable
    if rel_error < 0.01:  # 1% error
        print("✅ PASSED: Solution matches analytical Poiseuille profile!")
    elif rel_error < 0.05:  # 5% error
        print("⚠️  WARNING: Solution close but not exact (error > 1%)")
    else:
        print("❌ FAILED: Solution does not match analytical profile!")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity profile
    ax1.plot(u_analytical, y_profile, 'k-', linewidth=2, label='Analytical (Poiseuille)')
    ax1.plot(u_profile, y_profile, 'ro', markersize=4, fillstyle='none', label='Numerical')
    ax1.set_xlabel('Streamwise Velocity u', fontsize=12)
    ax1.set_ylabel('Wall-normal Coordinate y', fontsize=12)
    ax1.set_title('Velocity Profile Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.plot(error, y_profile, 'b-', linewidth=2)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Error |u_numerical - u_analytical|', fontsize=12)
    ax2.set_ylabel('Wall-normal Coordinate y', fontsize=12)
    ax2.set_title(f'Error Distribution (max: {max_error:.2e})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "poiseuille_validation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Try to display if in interactive mode
    try:
        plt.show()
    except:
        pass
    
    print()
    print("="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()

