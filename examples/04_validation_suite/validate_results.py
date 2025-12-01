#!/usr/bin/env python3
"""
Validation analysis - Compare simulation results against DNS/analytical benchmarks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_vtk_solution(vtk_file):
    """Read velocity from VTK file."""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'DIMENSIONS' in line:
            dims = [int(x) for x in line.split()[1:4]]
            Nx, Ny = dims[0], dims[1]
        if 'POINTS' in line:
            n_points = int(line.split()[1])
            point_start = i + 1
        if 'POINT_DATA' in line:
            vector_start = i + 2
    
    coords = []
    for i in range(point_start, point_start + n_points):
        coords.append([float(x) for x in lines[i].split()])
    coords = np.array(coords)
    
    velocity = []
    for i in range(vector_start, vector_start + n_points):
        velocity.append([float(x) for x in lines[i].split()])
    velocity = np.array(velocity)
    
    y = coords[:, 1].reshape(Ny, Nx)
    u = velocity[:, 0].reshape(Ny, Nx)
    
    return y, u

def poiseuille_solution(y, dp_dx, nu, H):
    """Analytical Poiseuille solution."""
    return -(dp_dx) / (2 * nu) * (H**2 / 4 - y**2)

def get_dns_data(Re_tau):
    """
    DNS data from Moser et al. (1999).
    Returns subset of published data for comparison.
    """
    if Re_tau == 180:
        y_plus = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180])
        u_plus = np.array([0, 4.95, 9.70, 17.50, 22.10, 25.20, 27.50, 29.30, 30.70, 31.90, 32.90, 33.80, 35.30, 36.50, 37.50, 38.30])
    elif Re_tau == 395:
        y_plus = np.array([0, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 395])
        u_plus = np.array([0, 9.7, 17.5, 22.1, 27.5, 30.7, 33.8, 37.8, 40.5, 42.5, 44.2, 45.6, 46.7])
    else:
        return None, None
    
    return y_plus, u_plus

def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    print("="*70)
    print("Validation Suite - Results Analysis")
    print("="*70)
    print()
    
    # Validation cases
    cases = {
        'Poiseuille Re=100': {
            'dir': 'poiseuille_re100',
            'type': 'analytical',
            'params': {'dp_dx': -0.001, 'nu': 0.01, 'H': 2.0, 'Re': 100},
            'tolerance': 0.01  # 1% error
        },
        'Poiseuille Re=1000': {
            'dir': 'poiseuille_re1000',
            'type': 'analytical',
            'params': {'dp_dx': -0.00001, 'nu': 0.001, 'H': 2.0, 'Re': 1000},
            'tolerance': 0.01
        },
        'Channel Re_tau=180': {
            'dir': 'channel_re180',
            'type': 'dns',
            'params': {'Re_tau': 180, 'nu': 0.0006667},
            'tolerance': 0.15  # 15% error (RANS model)
        },
        'Channel Re_tau=395': {
            'dir': 'channel_re395',
            'type': 'dns',
            'params': {'Re_tau': 395, 'nu': 0.0003077},
            'tolerance': 0.20  # 20% error (higher Re)
        }
    }
    
    # Results storage
    results = {}
    
    # Analyze each case
    print("Loading and analyzing results...")
    print("-" * 70)
    
    for case_name, case_info in cases.items():
        vtk_file = output_dir / case_info['dir'] / "velocity_final.vtk"
        
        if not vtk_file.exists():
            print(f"✗ {case_name:25s} - File not found")
            continue
        
        try:
            y, u = read_vtk_solution(vtk_file)
            y_profile = y[:, 0]
            u_profile = np.mean(u, axis=1)
            
            # Compare against benchmark
            if case_info['type'] == 'analytical':
                # Poiseuille solution
                H = case_info['params']['H']
                y_centered = y_profile - H/2
                u_exact = poiseuille_solution(y_centered, 
                                             case_info['params']['dp_dx'],
                                             case_info['params']['nu'],
                                             H)
                
                error = np.abs(u_profile - u_exact)
                rel_error = np.max(error) / np.max(u_exact)
                
                results[case_name] = {
                    'y': y_profile,
                    'u_numerical': u_profile,
                    'u_reference': u_exact,
                    'error': rel_error,
                    'type': 'analytical',
                    'passed': rel_error < case_info['tolerance']
                }
                
                status = "✓ PASS" if rel_error < case_info['tolerance'] else "✗ FAIL"
                print(f"{status} {case_name:25s} Error: {rel_error*100:.2f}%")
                
            elif case_info['type'] == 'dns':
                # DNS comparison
                dns_y, dns_u = get_dns_data(case_info['params']['Re_tau'])
                
                if dns_y is None:
                    print(f"⚠  {case_name:25s} - No DNS data available")
                    continue
                
                # Compute u_tau from simulation
                dy = y_profile[1] - y_profile[0]
                dudy_wall = (u_profile[1] - u_profile[0]) / dy
                tau_w = case_info['params']['nu'] * dudy_wall
                u_tau = np.sqrt(tau_w)
                
                # Convert to wall units
                H = 1.0  # Half channel height
                y_wall = y_profile - y_profile[0]  # Distance from wall
                y_plus = y_wall * u_tau / case_info['params']['nu']
                u_plus = u_profile / u_tau
                
                # Interpolate to compare at DNS points
                from scipy.interpolate import interp1d
                f = interp1d(y_plus, u_plus, kind='linear', fill_value='extrapolate')
                u_plus_interp = f(dns_y)
                
                error = np.sqrt(np.mean((u_plus_interp - dns_u)**2)) / np.mean(dns_u)
                
                results[case_name] = {
                    'y_plus': y_plus,
                    'u_plus': u_plus,
                    'dns_y': dns_y,
                    'dns_u': dns_u,
                    'u_tau': u_tau,
                    'error': error,
                    'type': 'dns',
                    'passed': error < case_info['tolerance']
                }
                
                status = "✓ PASS" if error < case_info['tolerance'] else "✗ FAIL"
                Re_tau = u_tau * 1.0 / case_info['params']['nu']
                print(f"{status} {case_name:25s} Error: {error*100:.1f}%  Re_tau: {Re_tau:.0f}")
                
        except Exception as e:
            print(f"✗ {case_name:25s} - Error: {e}")
    
    print()
    
    if not results:
        print("ERROR: No results to analyze!")
        print("Run: ./run_validation.sh")
        sys.exit(1)
    
    # Create validation report figure
    n_cases = len(results)
    fig = plt.figure(figsize=(14, 4*((n_cases+1)//2)))
    
    plot_idx = 1
    for case_name, data in results.items():
        ax = plt.subplot((n_cases+1)//2, 2, plot_idx)
        
        if data['type'] == 'analytical':
            # Plot physical coordinates
            ax.plot(data['u_reference'], data['y'], 'k-', linewidth=2, label='Analytical')
            ax.plot(data['u_numerical'], data['y'], 'ro', markersize=4, fillstyle='none', label='Numerical')
            ax.set_xlabel('Velocity u')
            ax.set_ylabel('y')
            
        elif data['type'] == 'dns':
            # Plot in wall units
            ax.plot(data['dns_y'], data['dns_u'], 'ko', markersize=6, label='DNS (Moser 1999)')
            ax.plot(data['y_plus'], data['u_plus'], 'r-', linewidth=2, label='Simulation')
            
            # Add law of the wall
            y_log = np.logspace(0, np.log10(200), 100)
            u_log = 1/0.41 * np.log(y_log) + 5.0
            ax.plot(y_log, u_log, 'k--', alpha=0.4, label='Log law')
            
            ax.set_xscale('log')
            ax.set_xlabel('y+')
            ax.set_ylabel('u+')
            ax.set_xlim([1, 500])
        
        # Title with pass/fail status
        status_str = "✓ PASS" if data['passed'] else "✗ FAIL"
        ax.set_title(f"{case_name} - {status_str} (Error: {data['error']*100:.1f}%)", 
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / "validation_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Validation report saved to: {output_file}")
    
    # Summary
    print()
    print("="*70)
    print("Validation Summary")
    print("="*70)
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    print(f"  Success rate: {100*passed/total:.0f}%")
    print()
    
    if passed == total:
        print("🎉 All validation cases PASSED!")
    elif passed > 0:
        print("⚠️  Some validation cases failed - check tolerances/models")
    else:
        print("❌ All validation cases FAILED - check solver!")
    
    print("="*70)
    
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    try:
        import scipy
    except ImportError:
        print("WARNING: scipy not installed - install for full functionality")
        print("  pip install scipy")
        print()
    
    main()

