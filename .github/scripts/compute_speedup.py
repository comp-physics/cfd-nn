#!/usr/bin/env python3
"""
Extract timing data from solver logs and compute CPU vs GPU speedup.
Parses the "Timing Summary" section to extract solver_step timings.
"""

import sys
import re
from pathlib import Path


def extract_solver_step_timing(log_file):
    """
    Extract solver_step total time and average from log file.
    Returns (total_seconds, avg_milliseconds) or (None, None) if not found.
    """
    try:
        content = Path(log_file).read_text()
    except FileNotFoundError:
        return None, None
    
    # Look for: solver_step                            20.207      2000         10.104
    # Format:   solver_step <whitespace> Total(s) <whitespace> Calls <whitespace> Avg(ms)
    pattern = r'^\s*solver_step\s+(\d+\.\d+)\s+\d+\s+(\d+\.\d+)\s*$'
    
    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            total_s = float(match.group(1))
            avg_ms = float(match.group(2))
            return total_s, avg_ms
    
    return None, None


def compute_speedup(cpu_total, gpu_total):
    """Compute speedup ratio, handling edge cases."""
    if gpu_total is None or gpu_total <= 0:
        return None
    if cpu_total is None or cpu_total <= 0:
        return None
    return cpu_total / gpu_total


def format_value(val, decimals=2):
    """Format numeric value or return placeholder."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_speedup(speedup):
    """Format speedup with 'x' suffix."""
    if speedup is None:
        return "N/A"
    return f"{speedup:.2f}x"


def main():
    if len(sys.argv) != 4:
        print("Usage: compute_speedup.py <case_name> <cpu_log> <gpu_log>", file=sys.stderr)
        sys.exit(1)
    
    case_name = sys.argv[1]
    cpu_log = sys.argv[2]
    gpu_log = sys.argv[3]
    
    # Extract timings
    cpu_total, cpu_avg = extract_solver_step_timing(cpu_log)
    gpu_total, gpu_avg = extract_solver_step_timing(gpu_log)
    
    # Compute speedups
    speedup_total = compute_speedup(cpu_total, gpu_total)
    speedup_avg = compute_speedup(cpu_avg, gpu_avg)
    
    # Print summary
    print(f"CPU Total: {format_value(cpu_total)}s, Per-step: {format_value(cpu_avg)}ms")
    print(f"GPU Total: {format_value(gpu_total)}s, Per-step: {format_value(gpu_avg)}ms")
    print(f"Speedup: {format_speedup(speedup_total)} (total), {format_speedup(speedup_avg)} (per-step)")
    
    # Output pipe-delimited record for table
    print(f"{case_name}|{format_value(cpu_total)}|{format_value(cpu_avg)}|{format_value(gpu_total)}|{format_value(gpu_avg)}|{format_speedup(speedup_total)}|{format_speedup(speedup_avg)}")


if __name__ == "__main__":
    main()
