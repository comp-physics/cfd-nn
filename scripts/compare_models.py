#!/usr/bin/env python3
"""
Compare turbulence models on channel flow.

Usage:
    python scripts/compare_models.py [--output_dir DIR]

This script runs the channel solver with different turbulence models
and generates comparison plots.
"""

import subprocess
import os
import sys
import argparse
import numpy as np

def run_solver(model, output_dir, nx=32, ny=64, nu=0.01, max_iter=20000, 
               adaptive_dt=True, extra_args=None):
    """Run the channel solver with specified model."""
    
    cmd = [
        "./channel",
        "--Nx", str(nx),
        "--Ny", str(ny),
        "--nu", str(nu),
        "--max_iter", str(max_iter),
        "--model", model,
        "--output", output_dir,
    ]
    
    if adaptive_dt:
        cmd.append("--adaptive_dt")
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def load_velocity_profile(output_dir):
    """Load velocity profile from output file."""
    filename = os.path.join(output_dir, "velocity_profile.dat")
    if not os.path.exists(filename):
        # Try alternative name
        filename = os.path.join(output_dir, "channel_velocity.dat")
    
    if not os.path.exists(filename):
        print(f"Warning: Cannot find velocity file in {output_dir}")
        return None, None
    
    data = np.loadtxt(filename)
    if data.ndim == 1:
        return None, None
    
    # Assume columns: y, u (or x, y, u, v)
    if data.shape[1] >= 4:
        # Full 2D field - extract centerline profile
        # Find unique y values
        y_vals = np.unique(data[:, 1])
        u_vals = []
        for y in y_vals:
            mask = data[:, 1] == y
            u_vals.append(np.mean(data[mask, 2]))
        return y_vals, np.array(u_vals)
    elif data.shape[1] >= 2:
        return data[:, 0], data[:, 1]
    
    return None, None


def poiseuille_solution(y, nu, dp_dx=-1.0, H=1.0):
    """Analytical Poiseuille solution."""
    return -dp_dx / (2 * nu) * (H**2 - y**2)


def main():
    parser = argparse.ArgumentParser(description="Compare turbulence models")
    parser.add_argument("--output_dir", default="output/comparison", 
                       help="Output directory for results")
    parser.add_argument("--nx", type=int, default=32, help="Grid cells in x")
    parser.add_argument("--ny", type=int, default=64, help="Grid cells in y")
    parser.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity")
    parser.add_argument("--max_iter", type=int, default=20000, help="Max iterations")
    parser.add_argument("--models", nargs="+", 
                       default=["none", "baseline", "gep"],
                       help="Models to compare")
    args = parser.parse_args()
    
    # Check if we're in build directory
    if not os.path.exists("./channel"):
        print("Error: Run this script from the build directory")
        print("  cd build && python ../scripts/compare_models.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Run each model
    for model in args.models:
        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        success = run_solver(
            model=model,
            output_dir=model_dir,
            nx=args.nx,
            ny=args.ny,
            nu=args.nu,
            max_iter=args.max_iter,
            adaptive_dt=True
        )
        
        if success:
            y, u = load_velocity_profile(model_dir)
            results[model] = {"y": y, "u": u}
        else:
            print(f"Warning: Model {model} failed")
            results[model] = {"y": None, "u": None}
    
    # Generate comparison report
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nGrid: {args.nx} x {args.ny}")
    print(f"Viscosity: {args.nu}")
    print(f"Max iterations: {args.max_iter}")
    
    print("\nModels tested:")
    for model in args.models:
        status = "OK" if results[model]["y"] is not None else "FAILED"
        print(f"  - {model}: {status}")
    
    # Write summary file
    summary_file = os.path.join(args.output_dir, "comparison_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Grid: {args.nx} x {args.ny}\n")
        f.write(f"Viscosity: {args.nu}\n")
        f.write(f"Max iterations: {args.max_iter}\n\n")
        f.write("Models:\n")
        for model in args.models:
            status = "OK" if results[model]["y"] is not None else "FAILED"
            f.write(f"  {model}: {status}\n")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Summary: {summary_file}")
    
    # Try to generate plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot analytical solution
        y_anal = np.linspace(-1, 1, 100)
        u_anal = poiseuille_solution(y_anal, args.nu)
        ax.plot(u_anal, y_anal, 'k--', label='Poiseuille (analytical)', linewidth=2)
        
        # Plot each model
        colors = ['b', 'r', 'g', 'm', 'c']
        for i, model in enumerate(args.models):
            if results[model]["y"] is not None:
                y = results[model]["y"]
                u = results[model]["u"]
                color = colors[i % len(colors)]
                ax.plot(u, y, 'o-', color=color, label=model, markersize=4)
        
        ax.set_xlabel('u (streamwise velocity)')
        ax.set_ylabel('y (wall-normal)')
        ax.set_title(f'Channel Flow Velocity Profile (nu={args.nu})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_file = os.path.join(args.output_dir, "velocity_comparison.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
    except ImportError:
        print("\nNote: matplotlib not available, skipping plot generation")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

