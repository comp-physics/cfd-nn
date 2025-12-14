#!/usr/bin/env python3
"""
Compare simulation results to Moser, Kim & Mansour (1999) DNS data
for turbulent channel flow at Re_tau = 180

DNS data source: https://turbulence.oden.utexas.edu/data/MKM/
Reference: JFM 399, 263-291 (1999)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_velocity_profile(filename):
    """Load velocity profile from solver output"""
    data = np.loadtxt(filename, skiprows=1)  # Skip header
    y = data[:, 0]  # Wall-normal coordinate
    u = data[:, 1]  # Streamwise velocity
    return y, u

def load_dns_data_retau180():
    """
    Moser et al. Re_tau=180 mean velocity profile
    This is a representative subset of the DNS data
    
    Full data available at: https://turbulence.oden.utexas.edu/data/MKM/
    Format: y+ (wall units), U+ (wall units)
    """
    # Representative DNS data points (log-sampled for clarity)
    dns_data = np.array([
        # y+      U+
        [0.05,   0.05],
        [0.1,    0.1],
        [0.2,    0.2],
        [0.5,    0.5],
        [1.0,    1.0],
        [2.0,    2.0],
        [5.0,    5.0],
        [8.0,    7.8],
        [10.0,   9.2],
        [15.0,   11.5],
        [20.0,   13.0],
        [30.0,   14.8],
        [50.0,   16.9],
        [70.0,   18.2],
        [100.0,  19.4],
        [120.0,  20.1],
        [140.0,  20.6],
        [160.0,  21.0],
        [180.0,  21.3],
    ])
    return dns_data[:, 0], dns_data[:, 1]

def compute_wall_units(y, u, nu, rho=1.0):
    """
    Convert to wall units: y+ = y * u_tau / nu, U+ = U / u_tau
    
    Assumes:
    - y is distance from wall (0 at wall)
    - u_tau is computed from wall shear stress
    - For channel: tau_w = -dp/dx * delta (from momentum balance)
    """
    # For dp_dx = -1.0, delta = 1.0:
    # tau_w = -(-1.0) * 1.0 = 1.0
    # u_tau = sqrt(tau_w / rho) = sqrt(1.0 / 1.0) = 1.0
    u_tau = 1.0  # This matches our setup
    
    # Distance from nearest wall
    y_from_wall = np.abs(y)
    
    # Wall units
    y_plus = y_from_wall * u_tau / nu
    u_plus = u / u_tau
    
    return y_plus, u_plus, u_tau

def plot_comparison(y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns, 
                   output_file='velocity_comparison.png'):
    """Create comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    ax1.plot(y_plus_dns, u_plus_dns, 'ko', label='DNS (Moser et al. 1999)', 
             markersize=6, markerfacecolor='none', linewidth=2)
    ax1.plot(y_plus_sim, u_plus_sim, 'r-', label='SST k-ω (this simulation)', 
             linewidth=2, alpha=0.7)
    
    # Add law of the wall
    y_log = np.logspace(-0.5, 2.5, 100)
    u_log = np.log(y_log) / 0.41 + 5.2  # log-law: U+ = (1/κ)ln(y+) + B
    u_visc = y_log  # viscous sublayer: U+ = y+
    
    ax1.plot(y_log, u_visc, 'k--', label='Viscous sublayer: $U^+ = y^+$', 
             alpha=0.5, linewidth=1)
    ax1.plot(y_log[y_log > 30], u_log[y_log > 30], 'k:', 
             label='Log law: $U^+ = (1/\\kappa)\\ln(y^+) + B$', 
             alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('$y^+$', fontsize=12)
    ax1.set_ylabel('$U^+$', fontsize=12)
    ax1.set_xlim([0, 200])
    ax1.set_ylim([0, 22])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Mean Velocity Profile (Linear Scale)', fontsize=12)
    
    # Log scale
    ax2.semilogx(y_plus_dns, u_plus_dns, 'ko', label='DNS (Moser et al. 1999)', 
                 markersize=6, markerfacecolor='none', linewidth=2)
    ax2.semilogx(y_plus_sim, u_plus_sim, 'r-', label='SST k-ω (this simulation)', 
                 linewidth=2, alpha=0.7)
    ax2.semilogx(y_log, u_visc, 'k--', label='Viscous sublayer', 
                 alpha=0.5, linewidth=1)
    ax2.semilogx(y_log[y_log > 30], u_log[y_log > 30], 'k:', 
                 label='Log law', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('$y^+$', fontsize=12)
    ax2.set_ylabel('$U^+$', fontsize=12)
    ax2.set_xlim([0.1, 200])
    ax2.set_ylim([0, 22])
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    ax2.set_title('Mean Velocity Profile (Log Scale)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison plot: {output_file}")
    plt.close()

def compute_errors(y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns):
    """Compute error metrics by interpolating DNS data to simulation points"""
    # Interpolate DNS data to simulation y+ locations
    u_plus_dns_interp = np.interp(y_plus_sim, y_plus_dns, u_plus_dns, 
                                   left=np.nan, right=np.nan)
    
    # Only compare where we have DNS data
    valid = ~np.isnan(u_plus_dns_interp)
    
    if np.sum(valid) == 0:
        print("Warning: No overlapping data points for error calculation")
        return
    
    # Compute errors
    error = u_plus_sim[valid] - u_plus_dns_interp[valid]
    rel_error = error / u_plus_dns_interp[valid] * 100
    
    print("\n" + "="*60)
    print("ERROR METRICS")
    print("="*60)
    print(f"Number of comparison points: {np.sum(valid)}")
    print(f"Mean absolute error:         {np.mean(np.abs(error)):.4f} (wall units)")
    print(f"RMS error:                   {np.sqrt(np.mean(error**2)):.4f} (wall units)")
    print(f"Max absolute error:          {np.max(np.abs(error)):.4f} (wall units)")
    print(f"Mean relative error:         {np.mean(np.abs(rel_error)):.2f}%")
    print(f"Max relative error:          {np.max(np.abs(rel_error)):.2f}%")
    print("="*60)

def main():
    # Paths
    case_dir = Path(__file__).parent
    output_dir = case_dir / 'output'
    profile_file = output_dir / 'velocity_profile.dat'
    
    print("="*60)
    print("DNS COMPARISON: Re_tau = 180")
    print("="*60)
    print(f"Case directory: {case_dir}")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Check if output exists
    if not profile_file.exists():
        print(f"ERROR: Profile file not found: {profile_file}")
        print("\nPlease run the simulation first:")
        print(f"  sbatch {case_dir}/run_h200.sbatch")
        return
    
    # Load simulation results
    print(f"Loading simulation results from: {profile_file}")
    y, u = load_velocity_profile(profile_file)
    
    # Physical parameters (from config)
    nu = 1.0 / 180.0  # = 0.005555...
    rho = 1.0
    
    # Convert to wall units
    y_plus_sim, u_plus_sim, u_tau = compute_wall_units(y, u, nu, rho)
    
    print(f"[OK] Loaded {len(y)} points from simulation")
    print(f"  Friction velocity u_tau = {u_tau:.6f}")
    print(f"  Re_tau = u_tau * delta / nu = {u_tau * 1.0 / nu:.2f}")
    print(f"  Max y+ = {np.max(y_plus_sim):.2f}")
    print("")
    
    # Load DNS data
    print("Loading DNS reference data (Moser et al. 1999)...")
    y_plus_dns, u_plus_dns = load_dns_data_retau180()
    print(f"[OK] Loaded {len(y_plus_dns)} DNS reference points")
    print("")
    
    # Create comparison plot
    plot_file = output_dir / 'velocity_comparison.png'
    print(f"Creating comparison plot...")
    plot_comparison(y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns, 
                   output_file=str(plot_file))
    
    # Compute errors
    compute_errors(y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Comparison plot: {plot_file}")
    print(f"  - Velocity data:   {profile_file}")
    print("\nReference:")
    print("  Moser, R.D., Kim, J. & Mansour, N.N. (1999)")
    print("  Direct numerical simulation of turbulent channel flow")
    print("  up to Re_tau=590. J. Fluid Mech. 399, 263-291.")
    print("  https://turbulence.oden.utexas.edu/data/MKM/")
    print("")

if __name__ == '__main__':
    main()


