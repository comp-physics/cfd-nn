#!/usr/bin/env python3
"""Analyze RANS validation results for Re_tau=180 channel flow.

Parses solver log files, loads MKM DNS reference data, generates matplotlib
plots, computes L2 errors, and writes a machine-readable summary JSON plus
a markdown validation report.

Usage:
    python3 scripts/rans_validation/analyze.py \
        [--data-dir output/rans_validation] \
        [--mkm-dir data/reference/mkm_retau180] \
        [--nu 0.005556]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Model definitions
# ============================================================================

MODELS_ACCURACY = ["none", "baseline", "gep", "earsm_wj", "sst", "komega"]
MODELS_STABILITY = ["nn_mlp", "nn_tbnn"]
MODELS_ALL = MODELS_ACCURACY + MODELS_STABILITY

MODEL_LABELS = {
    "none": "Laminar",
    "baseline": "Mixing Length",
    "gep": "GEP",
    "earsm_wj": "EARSM (WJ)",
    "sst": "SST k-omega",
    "komega": "k-omega",
    "nn_mlp": "NN-MLP (untrained)",
    "nn_tbnn": "NN-TBNN (untrained)",
}

# Grid convergence labels for SST
GRID_LABELS = {
    "sst_grid_32x64": "SST 32x64",
    "sst": "SST 64x128",
    "sst_grid_128x256": "SST 128x256",
}

GRID_SIZES = {
    "sst_grid_32x64": (32, 64),
    "sst": (64, 128),
    "sst_grid_128x256": (128, 256),
}


# ============================================================================
# Log file parser
# ============================================================================

def parse_log(path):
    """Parse a solver log file and extract residual history and final values.

    Returns a dict with keys:
        steps, residuals  -- lists of (step, residual) pairs
        final_residual, converged, total_steps
        bulk_velocity, wall_shear, u_tau, re_tau
        health  -- list of HEALTH dicts
    """
    result = {
        "steps": [],
        "residuals": [],
        "final_residual": None,
        "converged": None,
        "total_steps": None,
        "bulk_velocity": None,
        "wall_shear": None,
        "u_tau": None,
        "re_tau": None,
        "health": [],
        "diverged": False,
    }

    if not os.path.isfile(path):
        return None

    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip()

                # Step/Iter output:
                #   Iter    500 / 5000  ( 10%)  residual = 4.268e-03
                m = re.search(
                    r"(?:Iter|Step)\s+(\d+)\s*/\s*\d+\s*\(.*?\)\s*(?:t\*=\S+\s+)?residual\s*=\s*(\S+)",
                    line,
                )
                if m:
                    step = int(m.group(1))
                    try:
                        res = float(m.group(2))
                    except ValueError:
                        res = float("nan")
                    result["steps"].append(step)
                    result["residuals"].append(res)
                    continue

                # Verbose table rows (8-col step, 15-col residual, ...)
                m = re.match(r"\s+(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", line)
                if m and len(line.split()) <= 5:
                    step = int(m.group(1))
                    try:
                        res = float(m.group(2))
                    except ValueError:
                        continue
                    # Avoid duplicating steps already captured
                    if not result["steps"] or result["steps"][-1] != step:
                        result["steps"].append(step)
                        result["residuals"].append(res)

                # Final residual
                m = re.search(r"Final residual:\s*(\S+)", line)
                if m:
                    try:
                        result["final_residual"] = float(m.group(1))
                    except ValueError:
                        pass

                # Converged
                m = re.search(r"Converged:\s*(YES|NO)", line)
                if m:
                    result["converged"] = m.group(1) == "YES"

                m = re.search(r"Converged at iteration\s+(\d+)", line)
                if m:
                    result["converged"] = True
                    result["total_steps"] = int(m.group(1))

                # Diverged
                if re.search(r"(?:Solver diverged|SAFETY-VEL|BLOW-UP|nan|NaN)", line):
                    result["diverged"] = True

                # Total steps
                m = re.search(r"Iterations/Steps:\s*(\d+)", line)
                if m:
                    result["total_steps"] = int(m.group(1))

                # Bulk velocity
                m = re.search(r"Bulk velocity:\s*(\S+)", line)
                if m:
                    try:
                        result["bulk_velocity"] = float(m.group(1))
                    except ValueError:
                        pass

                # Wall shear stress
                m = re.search(r"Wall shear stress:\s*(\S+)", line)
                if m:
                    try:
                        result["wall_shear"] = float(m.group(1))
                    except ValueError:
                        pass

                # Friction velocity
                m = re.search(r"Friction velocity u_tau:\s*(\S+)", line)
                if m:
                    try:
                        result["u_tau"] = float(m.group(1))
                    except ValueError:
                        pass

                # Re_tau
                m = re.search(r"^Re_tau:\s*(\S+)", line)
                if m:
                    try:
                        result["re_tau"] = float(m.group(1))
                    except ValueError:
                        pass

                # HEALTH lines
                m = re.search(
                    r"\[HEALTH\]\s+step=(\d+)\s+Re_tau=([\d.]+)\s+U_b=([\d.]+)"
                    r"\s+v_max=(\S+)\s+w/v=([\d.]+)\s+v/u=([\d.]+)"
                    r"\s+dt=(\S+)\s+ramp=([\d.]+)\s+(\S+)",
                    line,
                )
                if m:
                    result["health"].append({
                        "step": int(m.group(1)),
                        "re_tau": float(m.group(2)),
                        "u_b": float(m.group(3)),
                        "v_max": float(m.group(4)),
                        "w_v": float(m.group(5)),
                        "v_u": float(m.group(6)),
                        "dt": float(m.group(7)),
                        "ramp": float(m.group(8)),
                        "state": m.group(9),
                    })

    except Exception as e:
        print(f"  [WARN] Error parsing {path}: {e}", file=sys.stderr)
        return None

    # Infer total_steps from last recorded step if not explicitly set
    if result["total_steps"] is None and result["steps"]:
        result["total_steps"] = result["steps"][-1]

    # Infer converged from final_residual if not explicitly set
    if result["converged"] is None and result["final_residual"] is not None:
        result["converged"] = result["final_residual"] < 1e-8

    return result


# ============================================================================
# Data loaders
# ============================================================================

def load_dat(path):
    """Load a whitespace-delimited .dat file, skipping comment lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
    return np.array(rows) if rows else np.empty((0, 0))


def load_mkm_means(mkm_dir):
    """Load MKM mean velocity profile.

    File: chan180/profiles/chan180.means
    Columns: y, y+, Umean, dUmean/dy, Wmean, dWmean/dy, Pmean
    Returns: dict with y, y_plus, u_plus arrays.
    """
    path = os.path.join(mkm_dir, "chan180", "profiles", "chan180.means")
    if not os.path.isfile(path):
        # Try flat layout
        path = os.path.join(mkm_dir, "chan180_means.dat")
    if not os.path.isfile(path):
        return None

    data = load_dat(path)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
        return None

    return {
        "y": data[:, 0],
        "y_plus": data[:, 1],
        "u_plus": data[:, 2],
    }


def load_mkm_reystress(mkm_dir):
    """Load MKM Reynolds stress data.

    File: chan180/profiles/chan180.reystress
    Returns: dict or None.
    """
    path = os.path.join(mkm_dir, "chan180", "profiles", "chan180.reystress")
    if not os.path.isfile(path):
        return None

    data = load_dat(path)
    if data.size == 0 or data.ndim != 2:
        return None

    return {"data": data}


def load_velocity_profile(path, u_tau, nu):
    """Load velocity_profile.dat and convert to wall units.

    File format: y  u_numerical  u_analytical
    Channel walls at y = -1, y = +1.

    Returns dict with y_plus, u_plus arrays (bottom half only, y < 0).
    Returns None if file missing or u_tau is zero/None.
    """
    if not os.path.isfile(path):
        return None
    if u_tau is None or u_tau <= 0:
        return None

    data = load_dat(path)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 2:
        return None

    y = data[:, 0]
    u = data[:, 1]

    # Bottom half only (y < 0)
    mask = y < 0.0
    if not np.any(mask):
        return None

    y_bot = y[mask]
    u_bot = u[mask]

    # Distance from bottom wall (wall at y = -1)
    y_wall = y_bot - (-1.0)  # = y_bot + 1

    # Wall units
    y_plus = y_wall * u_tau / nu
    u_plus = u_bot / u_tau

    # Sort by y+
    order = np.argsort(y_plus)
    y_plus = y_plus[order]
    u_plus = u_plus[order]

    return {"y_plus": y_plus, "u_plus": u_plus}


# ============================================================================
# Error metrics
# ============================================================================

def compute_l2_error(sim_yp, sim_up, ref_yp, ref_up, yp_min=0.5):
    """Interpolate simulation u+(y+) onto MKM y+ points and compute L2 error.

    L2 = sqrt(mean((u+_sim - u+_ref)^2))
    Only uses points where y+ > yp_min and both datasets overlap.
    """
    # Restrict to overlap region
    yp_lo = max(sim_yp.min(), ref_yp.min(), yp_min)
    yp_hi = min(sim_yp.max(), ref_yp.max())

    if yp_lo >= yp_hi:
        return float("nan")

    mask = (ref_yp >= yp_lo) & (ref_yp <= yp_hi)
    ref_yp_sub = ref_yp[mask]
    ref_up_sub = ref_up[mask]

    if len(ref_yp_sub) == 0:
        return float("nan")

    sim_interp = np.interp(ref_yp_sub, sim_yp, sim_up)

    return float(np.sqrt(np.mean((sim_interp - ref_up_sub) ** 2)))


# ============================================================================
# Plotting
# ============================================================================

def make_color_map(models):
    """Return a dict model -> color using tab10 palette."""
    cmap = plt.cm.tab10
    colors = {}
    for i, model in enumerate(models):
        colors[model] = cmap(i / max(len(models) - 1, 1))
    return colors


def plot_u_plus_all_models(profiles, mkm, output_path, colors):
    """Plot u+(y+) for all accuracy models + MKM DNS reference.

    Includes viscous sublayer (u+ = y+) and log law reference lines.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Viscous sublayer: u+ = y+
    yp_visc = np.linspace(0.1, 10, 50)
    ax.plot(yp_visc, yp_visc, "k:", lw=0.8, label=r"$u^+ = y^+$")

    # Log law: u+ = (1/0.41) * ln(y+) + 5.2
    yp_log = np.logspace(np.log10(30), np.log10(300), 50)
    ax.plot(yp_log, (1.0 / 0.41) * np.log(yp_log) + 5.2,
            "k--", lw=0.8, label=r"Log law ($\kappa=0.41$, $B=5.2$)")

    # MKM DNS
    if mkm is not None:
        ax.plot(mkm["y_plus"], mkm["u_plus"], "ko", ms=3, lw=1.5,
                label="MKM DNS", zorder=10, markerfacecolor="none")

    # Model profiles
    for model in MODELS_ACCURACY:
        if model not in profiles:
            continue
        prof = profiles[model]
        label = MODEL_LABELS.get(model, model)
        ax.plot(prof["y_plus"], prof["u_plus"], "-",
                color=colors.get(model, "gray"), lw=1.5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$", fontsize=12)
    ax.set_ylabel(r"$u^+$", fontsize=12)
    ax.set_title(r"Mean Velocity Profiles vs MKM DNS ($Re_\tau = 178$)", fontsize=13)
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_residual_history(logs, output_path, colors):
    """Plot residual vs step for all models on semilogy axis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS_ALL:
        if model not in logs or logs[model] is None:
            continue
        log = logs[model]
        if not log["steps"]:
            continue
        label = MODEL_LABELS.get(model, model)
        ax.semilogy(log["steps"], log["residuals"], "-",
                    color=colors.get(model, "gray"), lw=1.2, label=label)

    # Also plot grid convergence runs if present
    for key in ["sst_grid_32x64", "sst_grid_128x256"]:
        if key in logs and logs[key] is not None and logs[key]["steps"]:
            label = GRID_LABELS.get(key, key)
            ax.semilogy(logs[key]["steps"], logs[key]["residuals"],
                        "--", lw=1.0, label=label, alpha=0.7)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title("Residual Convergence History", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_grid_convergence_sst(grid_profiles, mkm, output_path):
    """Plot u+(y+) for SST on 3 grids + MKM reference."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Viscous sublayer
    yp_visc = np.linspace(0.1, 10, 50)
    ax.plot(yp_visc, yp_visc, "k:", lw=0.8, label=r"$u^+ = y^+$")

    # Log law
    yp_log = np.logspace(np.log10(30), np.log10(300), 50)
    ax.plot(yp_log, (1.0 / 0.41) * np.log(yp_log) + 5.2,
            "k--", lw=0.8, label=r"Log law")

    # MKM
    if mkm is not None:
        ax.plot(mkm["y_plus"], mkm["u_plus"], "ko", ms=3,
                label="MKM DNS", zorder=10, markerfacecolor="none")

    # Grid levels
    grid_colors = ["tab:blue", "tab:orange", "tab:green"]
    grid_order = ["sst_grid_32x64", "sst", "sst_grid_128x256"]
    for i, key in enumerate(grid_order):
        if key not in grid_profiles:
            continue
        prof = grid_profiles[key]
        label = GRID_LABELS.get(key, key)
        ax.plot(prof["y_plus"], prof["u_plus"], "-",
                color=grid_colors[i], lw=1.5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$", fontsize=12)
    ax.set_ylabel(r"$u^+$", fontsize=12)
    ax.set_title(r"SST $k$-$\omega$ Grid Convergence vs MKM DNS", fontsize=13)
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ============================================================================
# Summary JSON
# ============================================================================

def build_summary(logs, profiles, mkm, nu):
    """Build machine-readable summary dict."""
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flow": "Channel flow, Re_tau=180",
        "reference": "MKM DNS (Moser, Kim & Mansour 1999), Re_tau=178.12",
        "nu": nu,
        "models": {},
    }

    for model in MODELS_ALL:
        entry = {
            "label": MODEL_LABELS.get(model, model),
            "type": "stability" if model in MODELS_STABILITY else "accuracy",
        }

        log = logs.get(model)
        if log is not None:
            entry["re_tau"] = log["re_tau"]
            entry["u_tau"] = log["u_tau"]
            entry["bulk_velocity"] = log["bulk_velocity"]
            entry["final_residual"] = log["final_residual"]
            entry["converged"] = log["converged"]
            entry["total_steps"] = log["total_steps"]
            entry["diverged"] = log["diverged"]
        else:
            entry["re_tau"] = None
            entry["converged"] = None
            entry["total_steps"] = None
            entry["diverged"] = None

        # L2 error
        prof = profiles.get(model)
        if prof is not None and mkm is not None:
            entry["l2_error"] = compute_l2_error(
                prof["y_plus"], prof["u_plus"],
                mkm["y_plus"], mkm["u_plus"],
            )
        else:
            entry["l2_error"] = None

        summary["models"][model] = entry

    # Grid convergence
    grid_entries = {}
    for key in ["sst_grid_32x64", "sst", "sst_grid_128x256"]:
        ge = {"label": GRID_LABELS.get(key, key), "grid": GRID_SIZES.get(key)}
        log = logs.get(key)
        if log is not None:
            ge["re_tau"] = log["re_tau"]
            ge["u_tau"] = log["u_tau"]
            ge["converged"] = log["converged"]
            ge["total_steps"] = log["total_steps"]

        prof = profiles.get(key)
        if prof is not None and mkm is not None:
            ge["l2_error"] = compute_l2_error(
                prof["y_plus"], prof["u_plus"],
                mkm["y_plus"], mkm["u_plus"],
            )
        else:
            ge["l2_error"] = None

        grid_entries[key] = ge

    summary["grid_convergence"] = grid_entries

    return summary


# ============================================================================
# Markdown report
# ============================================================================

def write_report(summary, logs, output_path):
    """Write the RANS validation markdown report."""
    nu = summary["nu"]
    date_str = summary["date"]

    lines = []
    w = lines.append  # shorthand

    w("# RANS Validation Report: Channel Flow at Re_tau = 180")
    w("")
    w(f"**Generated**: {date_str}")
    w(f"**Reference**: {summary['reference']}")
    w(f"**Flow**: Fully developed turbulent channel flow, walls at y = +/-1")
    w(f"**Physics**: nu = {nu}, dp/dx = -1.0, Re_b = {1.0 / nu:.1f}")
    w("")

    # Executive summary table
    w("## Executive Summary")
    w("")
    w("| Model | Type | Re_tau | L2(u+) | Converged | Steps | Status |")
    w("|-------|------|--------|--------|-----------|-------|--------|")
    for model in MODELS_ALL:
        m = summary["models"].get(model, {})
        label = m.get("label", model)
        mtype = m.get("type", "")
        re_tau = f"{m['re_tau']:.1f}" if m.get("re_tau") is not None else "--"
        l2 = f"{m['l2_error']:.3f}" if m.get("l2_error") is not None else "--"
        conv = "Yes" if m.get("converged") else ("No" if m.get("converged") is False else "--")
        steps = str(m["total_steps"]) if m.get("total_steps") is not None else "--"
        if m.get("diverged"):
            status = "DIVERGED"
        elif m.get("converged"):
            status = "PASS"
        elif m.get("total_steps") is not None:
            status = "COMPLETED"
        else:
            status = "NO DATA"
        w(f"| {label} | {mtype} | {re_tau} | {l2} | {conv} | {steps} | {status} |")
    w("")

    # Setup
    w("## Setup")
    w("")
    w("### Grid")
    w("- Base grid: 64 x 128 (Nx x Ny), 2D (Nz = 1)")
    w("- Domain: Lx = 2pi, Ly = 2.0")
    w("- y-stretching: tanh with beta = 2.0")
    w("- Periodic in x, no-slip walls at y = +/-1")
    w("")
    w("### Solver Settings")
    w("- Convective scheme: 1st-order upwind")
    w("- Time stepping: Euler, adaptive dt (CFL_max = 0.5)")
    w("- Poisson solver: auto-selected (multigrid for stretched grids)")
    w("- Convergence tolerance: 1e-8")
    w("- Max steps: 50000")
    w("")

    # Convergence results
    w("## Convergence Results")
    w("")
    w("![Residual History](../output/rans_validation/plots/residual_history.png)")
    w("")
    w("| Model | Final Residual | Steps | Converged |")
    w("|-------|---------------|-------|-----------|")
    for model in MODELS_ALL:
        m = summary["models"].get(model, {})
        label = m.get("label", model)
        fr = f"{m['final_residual']:.3e}" if m.get("final_residual") is not None else "--"
        steps = str(m["total_steps"]) if m.get("total_steps") is not None else "--"
        conv = "Yes" if m.get("converged") else ("No" if m.get("converged") is False else "--")
        w(f"| {label} | {fr} | {steps} | {conv} |")
    w("")

    # Accuracy vs MKM DNS
    w("## Accuracy vs MKM DNS")
    w("")
    w("![u+ profiles](../output/rans_validation/plots/u_plus_all_models.png)")
    w("")
    w("### L2 Error in u+(y+)")
    w("")
    w("L2 error is computed by interpolating the numerical u+(y+) profile onto the")
    w("MKM DNS y+ points and taking sqrt(mean((u+_num - u+_DNS)^2)). Only the bottom")
    w("half of the channel (y < 0) is used, with y+ > 0.5 to exclude the wall point.")
    w("")
    w("| Model | Re_tau | u_tau | L2(u+) |")
    w("|-------|--------|-------|--------|")
    for model in MODELS_ACCURACY:
        m = summary["models"].get(model, {})
        label = m.get("label", model)
        re_tau = f"{m['re_tau']:.1f}" if m.get("re_tau") is not None else "--"
        u_tau = f"{m['u_tau']:.6f}" if m.get("u_tau") is not None else "--"
        l2 = f"{m['l2_error']:.3f}" if m.get("l2_error") is not None else "--"
        w(f"| {label} | {re_tau} | {u_tau} | {l2} |")
    w("")

    # Grid convergence
    w("## Grid Convergence (SST k-omega)")
    w("")
    w("![Grid convergence](../output/rans_validation/plots/grid_convergence_sst.png)")
    w("")
    w("| Grid | Re_tau | L2(u+) | Steps | Converged |")
    w("|------|--------|--------|-------|-----------|")
    for key in ["sst_grid_32x64", "sst", "sst_grid_128x256"]:
        ge = summary.get("grid_convergence", {}).get(key, {})
        label = ge.get("label", key)
        re_tau = f"{ge['re_tau']:.1f}" if ge.get("re_tau") is not None else "--"
        l2 = f"{ge['l2_error']:.3f}" if ge.get("l2_error") is not None else "--"
        steps = str(ge["total_steps"]) if ge.get("total_steps") is not None else "--"
        conv = "Yes" if ge.get("converged") else ("No" if ge.get("converged") is False else "--")
        w(f"| {label} | {re_tau} | {l2} | {steps} | {conv} |")
    w("")

    # NN stability
    w("## Neural Network Model Stability")
    w("")
    w("The NN-MLP and NN-TBNN models are tested with untrained (random) weights")
    w("to verify that the solver remains stable when coupled with neural network")
    w("turbulence closures. These are not expected to produce physically accurate")
    w("results.")
    w("")
    for model in MODELS_STABILITY:
        m = summary["models"].get(model, {})
        label = m.get("label", model)
        if m.get("diverged"):
            status = "DIVERGED"
        elif m.get("total_steps") is not None:
            status = f"Completed {m['total_steps']} steps"
        else:
            status = "No data"
        w(f"- **{label}**: {status}")
    w("")

    # Known limitations
    w("## Known Limitations")
    w("")
    w("1. **Upwind scheme**: 1st-order upwind convection adds significant numerical")
    w("   diffusion, which affects the accuracy of all turbulence models.")
    w("2. **2D only**: All runs are 2D (Nz=1). 3D effects are not captured.")
    w("3. **Untrained NNs**: Neural network models use random weights and are tested")
    w("   for stability only, not accuracy.")
    w("4. **Single Reynolds number**: Only Re_tau ~ 180 is validated.")
    w("5. **Steady-state solver**: The solver runs in pseudo-transient mode to steady")
    w("   state; temporal accuracy is not validated here.")
    w("")

    # References
    w("## References")
    w("")
    w("1. Moser, R. D., Kim, J., & Mansour, N. N. (1999). DNS of turbulent channel")
    w("   flow up to Re_tau = 590. *Physics of Fluids*, 11, 943-945.")
    w("2. Weatheritt, J., & Sandberg, R. D. (2016). A novel evolutionary algorithm")
    w("   applied to an unsteady flow problem. *Journal of Computational Physics*,")
    w("   325, 105-120.")
    w("3. Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for")
    w("   engineering applications. *AIAA Journal*, 32(8), 1598-1605.")
    w("4. Wallin, S., & Johansson, A. V. (2000). An explicit algebraic Reynolds")
    w("   stress model for incompressible and compressible turbulent flows. *Journal")
    w("   of Fluid Mechanics*, 403, 89-132.")
    w("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze RANS validation results for Re_tau=180 channel flow",
    )
    parser.add_argument(
        "--data-dir", default="output/rans_validation",
        help="Root directory containing logs/ and data/ subdirs (default: output/rans_validation)",
    )
    parser.add_argument(
        "--mkm-dir", default="data/reference/mkm_retau180",
        help="MKM reference data directory (default: data/reference/mkm_retau180)",
    )
    parser.add_argument(
        "--nu", type=float, default=0.005556,
        help="Kinematic viscosity (default: 0.005556 for Re_tau~180)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    mkm_dir = args.mkm_dir
    nu = args.nu

    log_dir = os.path.join(data_dir, "logs")
    prof_dir = os.path.join(data_dir, "data")
    plot_dir = os.path.join(data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("RANS Validation Analysis — Re_tau=180 Channel Flow")
    print("=" * 60)
    print(f"  Data dir:  {data_dir}")
    print(f"  MKM dir:   {mkm_dir}")
    print(f"  nu:        {nu}")
    print()

    # ── Load MKM reference ───────────────────────────────────────────
    print("Loading MKM reference data...")
    mkm = load_mkm_means(mkm_dir)
    if mkm is not None:
        print(f"  Loaded {len(mkm['y_plus'])} points (y+ range: "
              f"{mkm['y_plus'].min():.2f} to {mkm['y_plus'].max():.2f})")
    else:
        print("  [WARN] MKM mean profile not found. Plots will lack DNS reference.")
    print()

    # ── Parse log files ──────────────────────────────────────────────
    print("Parsing solver logs...")
    logs = {}

    # All 8 base models
    for model in MODELS_ALL:
        path = os.path.join(log_dir, f"{model}.log")
        log = parse_log(path)
        if log is not None:
            logs[model] = log
            status = "converged" if log["converged"] else "not converged"
            if log["diverged"]:
                status = "DIVERGED"
            n_steps = log["total_steps"] if log["total_steps"] else "?"
            re_tau = f"{log['re_tau']:.1f}" if log["re_tau"] is not None else "?"
            print(f"  {model:12s}: {status}, {n_steps} steps, Re_tau={re_tau}")
        else:
            print(f"  {model:12s}: no log found ({path})")

    # Grid convergence logs
    for key in ["sst_grid_32x64", "sst_grid_128x256"]:
        path = os.path.join(log_dir, f"{key}.log")
        log = parse_log(path)
        if log is not None:
            logs[key] = log
            print(f"  {key:12s}: {log['total_steps']} steps, "
                  f"Re_tau={log['re_tau']}")
        else:
            print(f"  {key:12s}: no log found")
    print()

    # ── Load velocity profiles ───────────────────────────────────────
    print("Loading velocity profiles...")
    profiles = {}

    for model in MODELS_ALL:
        path = os.path.join(prof_dir, model, "velocity_profile.dat")
        log = logs.get(model)
        u_tau = log["u_tau"] if log else None
        prof = load_velocity_profile(path, u_tau, nu)
        if prof is not None:
            profiles[model] = prof
            print(f"  {model:12s}: {len(prof['y_plus'])} points, "
                  f"y+ range [{prof['y_plus'].min():.2f}, {prof['y_plus'].max():.2f}]")
        else:
            print(f"  {model:12s}: no profile data")

    # Grid convergence profiles
    grid_profiles = {}
    for key in ["sst_grid_32x64", "sst_grid_128x256"]:
        path = os.path.join(prof_dir, key, "velocity_profile.dat")
        log = logs.get(key)
        u_tau = log["u_tau"] if log else None
        prof = load_velocity_profile(path, u_tau, nu)
        if prof is not None:
            grid_profiles[key] = prof
            profiles[key] = prof
            print(f"  {key:12s}: {len(prof['y_plus'])} points")
        else:
            print(f"  {key:12s}: no profile data")

    # Include base SST in grid convergence set
    if "sst" in profiles:
        grid_profiles["sst"] = profiles["sst"]
    print()

    # ── Compute L2 errors ────────────────────────────────────────────
    if mkm is not None:
        print("Computing L2 errors vs MKM DNS...")
        for model in MODELS_ACCURACY:
            if model in profiles:
                l2 = compute_l2_error(
                    profiles[model]["y_plus"], profiles[model]["u_plus"],
                    mkm["y_plus"], mkm["u_plus"],
                )
                print(f"  {model:12s}: L2 = {l2:.4f}")
            else:
                print(f"  {model:12s}: --")

        print("  Grid convergence:")
        for key in ["sst_grid_32x64", "sst", "sst_grid_128x256"]:
            if key in profiles:
                l2 = compute_l2_error(
                    profiles[key]["y_plus"], profiles[key]["u_plus"],
                    mkm["y_plus"], mkm["u_plus"],
                )
                print(f"    {GRID_LABELS.get(key, key):16s}: L2 = {l2:.4f}")
        print()

    # ── Generate plots ───────────────────────────────────────────────
    print("Generating plots...")
    colors = make_color_map(MODELS_ALL)

    # 1. u+ overlay for accuracy models
    if profiles or mkm:
        plot_u_plus_all_models(
            profiles, mkm,
            os.path.join(plot_dir, "u_plus_all_models.png"),
            colors,
        )

    # 2. Residual history
    if logs:
        plot_residual_history(
            logs,
            os.path.join(plot_dir, "residual_history.png"),
            colors,
        )

    # 3. Grid convergence for SST
    if grid_profiles:
        plot_grid_convergence_sst(
            grid_profiles, mkm,
            os.path.join(plot_dir, "grid_convergence_sst.png"),
        )
    print()

    # ── Write summary JSON ───────────────────────────────────────────
    print("Writing summary...")
    summary = build_summary(logs, profiles, mkm, nu)

    json_path = os.path.join(data_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved {json_path}")

    # ── Write markdown report ────────────────────────────────────────
    report_path = os.path.join("docs", "RANS_VALIDATION_REPORT.md")
    write_report(summary, logs, report_path)
    print()

    # ── Final status ─────────────────────────────────────────────────
    n_total = len(MODELS_ALL)
    n_found = sum(1 for m in MODELS_ALL if m in logs)
    n_profiles = sum(1 for m in MODELS_ALL if m in profiles)
    print("=" * 60)
    print(f"Analysis complete: {n_found}/{n_total} logs, "
          f"{n_profiles}/{n_total} profiles")
    print(f"Plots:   {plot_dir}/")
    print(f"Summary: {json_path}")
    print(f"Report:  {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
