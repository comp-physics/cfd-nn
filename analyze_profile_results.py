#!/usr/bin/env python3
"""
Analyze GPU profiling results and generate visualizations
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def parse_timing_csv(csv_file: Path) -> Dict:
    """Parse timing comparison CSV"""
    results = {"cpu": {}, "gpu": {}, "speedup": {}}
    
    if not csv_file.exists():
        return results
    
    with open(csv_file) as f:
        lines = f.readlines()[1:]  # Skip header
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
                
            test, mesh, device, time, iters, error = parts[:6]
            
            if device == "CPU":
                results["cpu"][mesh] = {
                    "time": float(time),
                    "iters": int(iters) if iters != "N/A" else 0,
                    "error": float(error) if error != "N/A" else 0.0
                }
            elif device == "GPU":
                results["gpu"][mesh] = {
                    "time": float(time),
                    "iters": int(iters) if iters != "N/A" else 0,
                    "error": float(error) if error != "N/A" else 0.0
                }
    
    # Calculate speedups
    for mesh in results["cpu"].keys():
        if mesh in results["gpu"]:
            cpu_time = results["cpu"][mesh]["time"]
            gpu_time = results["gpu"][mesh]["time"]
            if gpu_time > 0:
                results["speedup"][mesh] = cpu_time / gpu_time
    
    return results

def parse_kernel_launches(log_file: Path) -> Dict:
    """Parse kernel launch log"""
    results = {
        "total_kernels": 0,
        "kernels": {},
        "uploads": 0,
        "downloads": 0
    }
    
    if not log_file.exists():
        return results
    
    with open(log_file) as f:
        for line in f:
            if "launch CUDA kernel" in line:
                results["total_kernels"] += 1
                
                # Extract function name
                match = re.search(r'function=([^\s]+)', line)
                if match:
                    func = match.group(1)
                    results["kernels"][func] = results["kernels"].get(func, 0) + 1
            
            if "upload" in line.lower():
                results["uploads"] += 1
            if "download" in line.lower():
                results["downloads"] += 1
    
    return results

def generate_report(profile_dir: Path) -> str:
    """Generate comprehensive analysis report"""
    
    timing_file = profile_dir / "timing_comparison.csv"
    kernel_file = profile_dir / "kernel_launches.log"
    
    # Parse data
    timing = parse_timing_csv(timing_file)
    kernels = parse_kernel_launches(kernel_file)
    
    report = []
    report.append("=" * 80)
    report.append("GPU PROFILING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Kernel Analysis
    report.append("## KERNEL LAUNCH ANALYSIS")
    report.append(f"Total kernel launches: {kernels['total_kernels']}")
    report.append(f"Upload operations: {kernels['uploads']}")
    report.append(f"Download operations: {kernels['downloads']}")
    report.append("")
    
    if kernels['total_kernels'] > 0:
        report.append("Top 10 most called kernels:")
        sorted_kernels = sorted(kernels['kernels'].items(), key=lambda x: x[1], reverse=True)[:10]
        for func, count in sorted_kernels:
            report.append(f"  {count:6d}x  {func}")
        report.append("")
    
    # Performance Analysis
    if timing["cpu"] and timing["gpu"]:
        report.append("## PERFORMANCE ANALYSIS")
        report.append("")
        report.append(f"{'Mesh':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12} {'CPU Err%':<12} {'GPU Err%':<12}")
        report.append("-" * 72)
        
        for mesh in sorted(timing["cpu"].keys()):
            if mesh in timing["gpu"] and mesh in timing["speedup"]:
                cpu_data = timing["cpu"][mesh]
                gpu_data = timing["gpu"][mesh]
                speedup = timing["speedup"][mesh]
                
                report.append(
                    f"{mesh:<12} "
                    f"{cpu_data['time']:<12.3f} "
                    f"{gpu_data['time']:<12.3f} "
                    f"{speedup:<12.2f} "
                    f"{cpu_data['error']:<12.3f} "
                    f"{gpu_data['error']:<12.3f}"
                )
        
        report.append("")
        
        # Average speedup
        if timing["speedup"]:
            avg_speedup = sum(timing["speedup"].values()) / len(timing["speedup"])
            report.append(f"Average speedup: {avg_speedup:.2f}x")
            report.append("")
    
    # Memory Transfer Analysis
    report.append("## MEMORY TRANSFER ANALYSIS")
    total_transfers = kernels['uploads'] + kernels['downloads']
    report.append(f"Total transfers: {total_transfers}")
    
    if kernels['total_kernels'] > 0:
        transfers_per_kernel = total_transfers / kernels['total_kernels']
        report.append(f"Transfers per kernel: {transfers_per_kernel:.2f}")
        
        if transfers_per_kernel < 0.1:
            report.append("✅ EXCELLENT: Very few data transfers (persistent mapping working)")
        elif transfers_per_kernel < 0.5:
            report.append("✅ GOOD: Reasonable transfer rate")
        else:
            report.append("⚠️  WARNING: High transfer rate - may impact performance")
    report.append("")
    
    # Overall Assessment
    report.append("## OVERALL ASSESSMENT")
    report.append("")
    
    issues = []
    successes = []
    
    if kernels['total_kernels'] < 50:
        issues.append("Low kernel count - GPU may not be executing properly")
    else:
        successes.append(f"GPU kernels launching correctly ({kernels['total_kernels']} launches)")
    
    if timing["speedup"]:
        avg_speedup = sum(timing["speedup"].values()) / len(timing["speedup"])
        if avg_speedup > 1.5:
            successes.append(f"GPU provides speedup ({avg_speedup:.2f}x on average)")
        elif avg_speedup > 0.8:
            issues.append(f"GPU speedup marginal ({avg_speedup:.2f}x) - may have overhead issues")
        else:
            issues.append(f"GPU slower than CPU ({avg_speedup:.2f}x) - needs investigation")
    
    # Check correctness
    if timing["cpu"] and timing["gpu"]:
        for mesh in timing["cpu"].keys():
            if mesh in timing["gpu"]:
                cpu_err = timing["cpu"][mesh]["error"]
                gpu_err = timing["gpu"][mesh]["error"]
                err_diff = abs(cpu_err - gpu_err)
                
                if err_diff < 0.01:
                    successes.append(f"CPU-GPU match excellent for {mesh} (Δ={err_diff:.4f}%)")
                elif err_diff < 0.1:
                    issues.append(f"CPU-GPU small difference for {mesh} (Δ={err_diff:.4f}%)")
                else:
                    issues.append(f"CPU-GPU MISMATCH for {mesh} (Δ={err_diff:.4f}%)")
    
    if successes:
        report.append("### ✅ Successes:")
        for s in successes:
            report.append(f"  • {s}")
        report.append("")
    
    if issues:
        report.append("### ⚠️  Issues / Concerns:")
        for i in issues:
            report.append(f"  • {i}")
        report.append("")
    
    if not issues:
        report.append("🎉 No issues detected! GPU implementation looks excellent.")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    if len(sys.argv) > 1:
        profile_dir = Path(sys.argv[1])
    else:
        # Find most recent profile directory
        cwd = Path.cwd()
        profile_dirs = sorted(cwd.glob("gpu_profile_*"))
        if not profile_dirs:
            print("No profile directories found!")
            sys.exit(1)
        profile_dir = profile_dirs[-1]
    
    print(f"Analyzing profile directory: {profile_dir}")
    print()
    
    report = generate_report(profile_dir)
    print(report)
    
    # Save report
    report_file = profile_dir / "ANALYSIS_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()

