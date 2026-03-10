#!/usr/bin/env python3
"""Analyze RANS validation campaign results from SLURM log files.

Parses output/rans_campaign/slurm-*.out for convergence, forces, timing,
and GPU profiling data. Prints summary tables and sanity-check results.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns for log parsing
# ---------------------------------------------------------------------------

RE_STEP = re.compile(r"Step\s+(\d+).*?res=\s*([\d.eE+-]+)")
RE_CD = re.compile(r"Cd=\s*([\d.eE+-]+)")
RE_CL = re.compile(r"Cl=\s*([\d.eE+-]+)")
RE_FX = re.compile(r"Fx=\s*([\d.eE+-]+)")
RE_FY = re.compile(r"Fy=\s*([\d.eE+-]+)")
RE_UB = re.compile(r"U_b=\s*([\d.eE+-]+)")
RE_WALL_TIME = re.compile(r"Wall time:\s*(\d+)ms")
RE_WALL_SEC = re.compile(r"Wall time\s*:\s*\d+ms\s*\((\d+)s\)")
RE_GPU_UTIL = re.compile(r"GPU utilization:\s*([\d.]+)\s*%")
RE_EXIT_CODE = re.compile(r"Exit code\s*:\s*(\d+)")
RE_CONFIG = re.compile(r"Config\s*:\s*(\S+)")
RE_TIMING_SECTION = re.compile(r"={3,}\s*Timing Summary\s*={3,}")
RE_TIMING_ROW = re.compile(r"^\s*(\S[\w\s/]+\S)\s+:\s+([\d.]+)\s*ms")

# Convergence indicators
RE_CONVERGED = re.compile(r"[Cc]onverged")
RE_DIVERGED = re.compile(r"[Dd]iverged|NaN|nan|inf\b")


# ---------------------------------------------------------------------------
# Sanity-check benchmarks
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "cylinder_re100": {"Cd": (1.0, 1.8)},
    "airfoil_re1000": {"Cd": (0.02, 0.20), "Cl_abs_max": 0.05},
    "step_re5000": {"not_diverged": True},
    "hills_re10595": {"not_diverged": True},
}


def identify_case(config_name):
    """Return flow-case key from config basename."""
    for case_key in BENCHMARKS:
        if config_name.startswith(case_key):
            return case_key
    return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(filepath):
    """Parse a single SLURM .out log file. Returns dict of extracted quantities."""
    result = {
        "file": str(filepath),
        "config": None,
        "converged": None,
        "diverged": False,
        "last_step": None,
        "last_residual": None,
        "Cd": None,
        "Cl": None,
        "Fx": None,
        "Fy": None,
        "Ub": None,
        "wall_ms": None,
        "wall_sec": None,
        "gpu_pct": None,
        "exit_code": None,
        "timing": {},
    }

    in_timing_section = False

    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                # Config name
                m = RE_CONFIG.match(line)
                if m:
                    result["config"] = m.group(1)

                # Step / residual (keep last match)
                m = RE_STEP.search(line)
                if m:
                    result["last_step"] = int(m.group(1))
                    try:
                        result["last_residual"] = float(m.group(2))
                    except ValueError:
                        pass

                # Forces (keep last match)
                m = RE_CD.search(line)
                if m:
                    try:
                        result["Cd"] = float(m.group(1))
                    except ValueError:
                        pass

                m = RE_CL.search(line)
                if m:
                    try:
                        result["Cl"] = float(m.group(1))
                    except ValueError:
                        pass

                m = RE_FX.search(line)
                if m:
                    try:
                        result["Fx"] = float(m.group(1))
                    except ValueError:
                        pass

                m = RE_FY.search(line)
                if m:
                    try:
                        result["Fy"] = float(m.group(1))
                    except ValueError:
                        pass

                m = RE_UB.search(line)
                if m:
                    try:
                        result["Ub"] = float(m.group(1))
                    except ValueError:
                        pass

                # Wall time
                m = RE_WALL_TIME.search(line)
                if m:
                    result["wall_ms"] = int(m.group(1))

                m = RE_WALL_SEC.search(line)
                if m:
                    result["wall_sec"] = int(m.group(1))

                # GPU utilization
                m = RE_GPU_UTIL.search(line)
                if m:
                    try:
                        result["gpu_pct"] = float(m.group(1))
                    except ValueError:
                        pass

                # Exit code
                m = RE_EXIT_CODE.search(line)
                if m:
                    result["exit_code"] = int(m.group(1))

                # Convergence / divergence
                if RE_CONVERGED.search(line):
                    result["converged"] = True
                if RE_DIVERGED.search(line):
                    result["diverged"] = True

                # Timing summary section
                if RE_TIMING_SECTION.search(line):
                    in_timing_section = True
                    continue
                if in_timing_section:
                    m = RE_TIMING_ROW.match(line)
                    if m:
                        result["timing"][m.group(1).strip()] = float(m.group(2))
                    elif line.strip() == "" or line.startswith("==="):
                        in_timing_section = False

    except OSError as e:
        print(f"WARNING: cannot read {filepath}: {e}", file=sys.stderr)

    # Derive config basename from the config path if present
    if result["config"]:
        result["config_basename"] = Path(result["config"]).stem
    else:
        # Try to extract from filename pattern slurm-JOBID_TASKID.out
        result["config_basename"] = Path(filepath).stem

    # Derive wall_sec from wall_ms if not directly parsed
    if result["wall_sec"] is None and result["wall_ms"] is not None:
        result["wall_sec"] = result["wall_ms"] / 1000.0

    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_result(result):
    """Run sanity checks against benchmarks. Returns (status_str, detail_str)."""
    config = result.get("config_basename", "")
    case = identify_case(config)

    if result["diverged"]:
        return ("FAIL", "diverged/NaN detected")

    if case is None:
        return ("SKIP", "unknown case")

    bench = BENCHMARKS[case]

    # Check not-diverged requirement
    if bench.get("not_diverged") and result["diverged"]:
        return ("FAIL", "diverged")

    # Check Cd range
    if "Cd" in bench and result["Cd"] is not None:
        lo, hi = bench["Cd"]
        if lo <= result["Cd"] <= hi:
            detail = f"Cd={result['Cd']:.4f} in [{lo}, {hi}]"
        else:
            return ("WARN", f"Cd={result['Cd']:.4f} outside [{lo}, {hi}]")
    else:
        detail = ""

    # Check |Cl| for airfoil
    if "Cl_abs_max" in bench and result["Cl"] is not None:
        if abs(result["Cl"]) < bench["Cl_abs_max"]:
            cl_detail = f"|Cl|={abs(result['Cl']):.4f} < {bench['Cl_abs_max']}"
            detail = f"{detail}; {cl_detail}" if detail else cl_detail
        else:
            return ("WARN", f"|Cl|={abs(result['Cl']):.4f} >= {bench['Cl_abs_max']}")

    if not detail:
        detail = "no data to check"

    return ("PASS", detail)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_float(val, fmt=".2e"):
    """Format a float or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:{fmt}}"


def print_summary_table(results):
    """Print the main results table."""
    # Header
    hdr = (
        f"{'Config':<46s} {'Status':<10s} {'Steps':>7s} {'Residual':>12s} "
        f"{'Wall(s)':>9s} {'Cd':>8s} {'GPU%':>6s} {'Check'}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        config = r.get("config_basename", "?")[:45]

        # Status
        if r["diverged"]:
            status = "DIVERG"
        elif r["converged"]:
            status = "CONV"
        elif r["exit_code"] is not None and r["exit_code"] == 0:
            status = "DONE"
        elif r["exit_code"] is not None:
            status = f"ERR({r['exit_code']})"
        else:
            status = "???"

        steps = str(r["last_step"]) if r["last_step"] is not None else "-"
        residual = format_float(r["last_residual"])
        wall = f"{r['wall_sec']:.1f}" if r["wall_sec"] is not None else "-"
        cd = f"{r['Cd']:.4f}" if r["Cd"] is not None else "-"
        gpu = f"{r['gpu_pct']:.0f}" if r["gpu_pct"] is not None else "-"

        check_status, check_detail = check_result(r)
        check_str = f"{check_status}: {check_detail}"

        print(
            f"{config:<46s} {status:<10s} {steps:>7s} {residual:>12s} "
            f"{wall:>9s} {cd:>8s} {gpu:>6s}  {check_str}"
        )


def print_profiling_summary(results):
    """Print average timing per category across all runs."""
    # Aggregate timing categories
    category_times = defaultdict(list)
    for r in results:
        n_steps = r.get("last_step")
        if not n_steps or n_steps <= 0:
            continue
        for cat, total_ms in r["timing"].items():
            category_times[cat].append(total_ms / n_steps)

    if not category_times:
        print("\nNo timing data found in logs.")
        return

    print(f"\n{'=== Profiling Summary (avg ms/step) ==='}")
    print(f"{'Category':<35s} {'Mean':>10s} {'Min':>10s} {'Max':>10s} {'N':>5s}")
    print("-" * 65)

    for cat in sorted(category_times.keys()):
        vals = category_times[cat]
        mean_v = sum(vals) / len(vals)
        min_v = min(vals)
        max_v = max(vals)
        print(f"{cat:<35s} {mean_v:>10.3f} {min_v:>10.3f} {max_v:>10.3f} {len(vals):>5d}")


def print_failure_summary(results):
    """Print failures and warnings at bottom."""
    failures = []
    warnings = []
    for r in results:
        status, detail = check_result(r)
        config = r.get("config_basename", "?")
        if status == "FAIL":
            failures.append(f"  FAIL: {config} -- {detail}")
        elif status == "WARN":
            warnings.append(f"  WARN: {config} -- {detail}")

    # Also flag non-zero exit codes
    for r in results:
        if r["exit_code"] is not None and r["exit_code"] != 0:
            config = r.get("config_basename", "?")
            failures.append(f"  FAIL: {config} -- exit code {r['exit_code']}")

    if failures or warnings:
        print(f"\n{'=== Issues ==='}")
        for line in failures:
            print(line)
        for line in warnings:
            print(line)
        print(f"\nTotal: {len(failures)} failure(s), {len(warnings)} warning(s)")
    else:
        print("\nAll checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    output_dir = os.path.join(project_root, "output", "rans_campaign")

    # Find all slurm .out files
    log_files = sorted(Path(output_dir).glob("slurm-*.out")) if os.path.isdir(output_dir) else []

    if not log_files:
        print(f"No SLURM log files found in {output_dir}")
        print("Run the campaign first: bash scripts/rans_campaign/submit.sh")
        return 1

    print(f"Found {len(log_files)} log file(s) in {output_dir}\n")

    # Parse all logs
    results = []
    for lf in log_files:
        results.append(parse_log(str(lf)))

    # Sort by config basename
    results.sort(key=lambda r: r.get("config_basename", ""))

    # Print tables
    print_summary_table(results)
    print_profiling_summary(results)
    print_failure_summary(results)

    # Return non-zero if any failures
    for r in results:
        status, _ = check_result(r)
        if status == "FAIL":
            return 1
        if r["exit_code"] is not None and r["exit_code"] != 0:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
