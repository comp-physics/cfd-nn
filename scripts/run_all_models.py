#!/usr/bin/env python3
"""
Run all turbulence models and generate comprehensive comparison.

This script:
1. Runs the CFD solver with each turbulence model
2. Collects results (velocity profiles, nu_t fields, timing)
3. Generates comparison plots
4. Produces a summary report

Usage:
    python scripts/run_all_models.py --case channel
    python scripts/run_all_models.py --case periodic_hills --quick
"""

import subprocess
import sys
import argparse
import numpy as np
import json
from pathlib import Path
import time


class ModelRunner:
    """Run CFD solver with different turbulence models."""
    
    def __init__(self, case='channel', build_dir='build', quick=False):
        self.case = case
        self.build_dir = Path(build_dir)
        self.quick = quick
        self.results = {}
        
        # Grid resolution
        if quick:
            self.nx = 32 if case == 'channel' else 32
            self.ny = 64 if case == 'channel' else 48
            self.max_iter = 5000
        else:
            self.nx = 64 if case == 'channel' else 64
            self.ny = 128 if case == 'channel' else 96
            self.max_iter = 20000
        
        # Model configurations
        self.models = {
            'baseline': {
                'args': ['--model', 'baseline'],
                'description': 'Mixing length with van Driest damping'
            },
            'gep': {
                'args': ['--model', 'gep'],
                'description': 'Gene Expression Programming (algebraic)'
            }
        }
        
        # Check for available NN models
        models_dir = Path('data/models')
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / 'metadata.json').exists():
                    with open(model_dir / 'metadata.json') as f:
                        metadata = json.load(f)
                        model_type = metadata.get('type', 'unknown')
                        
                        if model_type == 'nn_mlp':
                            self.models[model_dir.name] = {
                                'args': ['--model', 'nn_mlp', '--nn_preset', model_dir.name],
                                'description': metadata.get('description', 'MLP model')
                            }
                        elif model_type == 'nn_tbnn':
                            self.models[model_dir.name] = {
                                'args': ['--model', 'nn_tbnn', '--nn_preset', model_dir.name],
                                'description': metadata.get('description', 'TBNN model')
                            }
    
    def run_model(self, model_name, model_config):
        """Run solver with specified model."""
        
        print(f"\n{'='*70}")
        print(f"Running: {model_name}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*70}\n")
        
        # Prepare output directory
        output_dir = self.build_dir / f"output_{model_name}"
        output_dir.mkdir(exist_ok=True)
        
        # Build command
        executable = self.build_dir / self.case
        if not executable.exists():
            print(f"ERROR: Executable not found: {executable}")
            print("Please build the project first: cd build && make")
            return None
        
        cmd = [
            str(executable),
            '--Nx', str(self.nx),
            '--Ny', str(self.ny),
            '--max_iter', str(self.max_iter),
            '--adaptive_dt',
            '--output', str(output_dir),
            '--quiet'
        ] + model_config['args']
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run solver
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"ERROR: Solver failed with code {result.returncode}")
                print(f"STDERR:\n{result.stderr}")
                return None
            
            # Parse output for timing info
            timing_info = self._parse_timing(result.stdout)
            
            print(f"[OK] Completed in {elapsed:.2f} seconds")
            
            return {
                'model': model_name,
                'description': model_config['description'],
                'output_dir': str(output_dir),
                'wall_time': elapsed,
                'timing': timing_info,
                'stdout': result.stdout
            }
            
        except subprocess.TimeoutExpired:
            print("ERROR: Solver timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    def _parse_timing(self, stdout):
        """Extract timing information from solver output."""
        
        timing = {}
        
        for line in stdout.split('\n'):
            # Look for timing output
            if 'Time/Iter' in line or 'ms' in line:
                # Try to parse timing data
                pass
            
            # Look for convergence info
            if 'Converged' in line or 'iterations' in line:
                # Extract iteration count
                pass
        
        return timing
    
    def run_all(self):
        """Run all available models."""
        
        print(f"\n{'#'*70}")
        print(f"# Running All Models - Case: {self.case}")
        print(f"# Grid: {self.nx} x {self.ny}")
        print(f"# Max iterations: {self.max_iter}")
        print(f"# Models to run: {len(self.models)}")
        print(f"{'#'*70}\n")
        
        results = []
        
        for model_name, model_config in self.models.items():
            result = self.run_model(model_name, model_config)
            if result:
                results.append(result)
        
        self.results = results
        return results
    
    def generate_report(self, output_file='model_comparison.txt'):
        """Generate comparison report."""
        
        if not self.results:
            print("No results to report")
            return
        
        print(f"\n{'='*70}")
        print("COMPARISON REPORT")
        print(f"{'='*70}\n")
        
        report_lines = []
        report_lines.append(f"Turbulence Model Comparison - {self.case.upper()}")
        report_lines.append(f"Grid: {self.nx} x {self.ny}, Max iterations: {self.max_iter}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Sort by wall time
        sorted_results = sorted(self.results, key=lambda x: x['wall_time'])
        
        # Table header
        report_lines.append(f"{'Model':<20} {'Description':<30} {'Time (s)':>10} {'Speedup':>10}")
        report_lines.append("-" * 70)
        
        baseline_time = sorted_results[0]['wall_time']
        
        for result in sorted_results:
            speedup = baseline_time / result['wall_time']
            report_lines.append(
                f"{result['model']:<20} "
                f"{result['description'][:30]:<30} "
                f"{result['wall_time']:>10.2f} "
                f"{speedup:>10.2f}x"
            )
        
        report_lines.append("")
        report_lines.append("Notes:")
        report_lines.append("- Speedup is relative to fastest model")
        report_lines.append("- Results saved to: build/output_<model_name>/")
        report_lines.append("")
        
        report = "\n".join(report_lines)
        print(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_file}\n")


def plot_comparison(results, case='channel'):
    """Generate comparison plots (requires matplotlib)."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    
    print("Generating comparison plots...")
    
    # Load velocity profiles from each model
    profiles = {}
    
    for result in results:
        output_dir = Path(result['output_dir'])
        vel_file = output_dir / f"{case}_velocity.dat"
        
        if not vel_file.exists():
            continue
        
        try:
            data = np.loadtxt(vel_file)
            if data.ndim == 2 and data.shape[1] >= 2:
                profiles[result['model']] = data
        except Exception:
            pass
    
    if not profiles:
        print("No velocity data found for plotting")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot velocity profiles
    ax = axes[0]
    for model_name, data in profiles.items():
        if data.shape[1] >= 4:
            # 2D field - extract centerline
            y = data[:, 1]
            u = data[:, 2]
        else:
            y = data[:, 0]
            u = data[:, 1]
        
        ax.plot(y, u, label=model_name, marker='o', markersize=3)
    
    ax.set_xlabel('y')
    ax.set_ylabel('u')
    ax.set_title('Velocity Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot timing comparison
    ax = axes[1]
    models = [r['model'] for r in results]
    times = [r['wall_time'] for r in results]
    
    ax.barh(models, times)
    ax.set_xlabel('Wall Time (s)')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("Plot saved to: model_comparison.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Run and compare all turbulence models'
    )
    parser.add_argument('--case', type=str, default='channel',
                        choices=['channel', 'periodic_hills'],
                        help='Test case to run')
    parser.add_argument('--build_dir', type=str, default='build',
                        help='Build directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with coarse grid')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--output', type=str, default='model_comparison.txt',
                        help='Output report file')
    
    args = parser.parse_args()
    
    # Check if build directory exists
    if not Path(args.build_dir).exists():
        print(f"ERROR: Build directory not found: {args.build_dir}")
        print("Please build the project first:")
        print("  mkdir build && cd build")
        print("  cmake .. && make")
        sys.exit(1)
    
    # Run models
    runner = ModelRunner(case=args.case, build_dir=args.build_dir, quick=args.quick)
    results = runner.run_all()
    
    if not results:
        print("No models completed successfully")
        sys.exit(1)
    
    # Generate report
    runner.generate_report(args.output)
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(results, args.case)
    
    print(f"\n{'='*70}")
    print(f"All done! {len(results)} models completed successfully")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

