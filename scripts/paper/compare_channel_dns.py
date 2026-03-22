#!/usr/bin/env python3
"""
Compare a posteriori channel flow results against MKM DNS data.

Loads solver velocity profiles from each turbulence model run,
computes u+(y+), and generates comparison plots against DNS Re_tau=180.

Usage:
    python scripts/paper/compare_channel_dns.py
    python scripts/paper/compare_channel_dns.py --results_dir results/paper/aposteriori/channel
"""

import argparse
import os
import sys
import numpy as np

# Add scripts/paper to path for plot_style import
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from plot_style import (apply_style, COLORS, MODEL_LABELS,
                        single_col_fig, double_col_fig, save_fig)

import matplotlib.pyplot as plt

# ============================================================================
# Paths
# ============================================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_RESULTS = os.path.join(ROOT_DIR, 'results', 'paper', 'aposteriori', 'channel')
DNS_MEANS = os.path.join(ROOT_DIR, 'results', 'paper', 'reference', 'chan180.means')
DNS_REYSTRESS = os.path.join(ROOT_DIR, 'results', 'paper', 'reference',
                             'chan180', 'profiles', 'chan180.reystress')

# Models to compare (in plot order)
MODELS = ['none', 'sst', 'earsm_pope', 'nn_mlp', 'nn_tbnn']

# Map directory names to plot_style color/label keys
STYLE_MAP = {
    'none':       'baseline',
    'sst':        'sst',
    'earsm_pope': 'earsm',
    'nn_mlp':     'mlp',
    'nn_tbnn':    'tbnn',
}

LINE_STYLES = {
    'none':       '--',
    'sst':        '-',
    'earsm_pope': '-',
    'nn_mlp':     '-',
    'nn_tbnn':    '-',
}


# ============================================================================
# Data loading
# ============================================================================
def load_dns_means(path):
    """Load MKM DNS chan180.means file.

    Columns: y, y+, Umean, dUmean/dy, Wmean, dWmean/dy, Pmean
    Returns dict with keys: y, yplus, Umean.
    """
    data = np.loadtxt(path, comments='#')
    return {
        'y': data[:, 0],
        'yplus': data[:, 1],
        'Umean': data[:, 2],
    }


def load_dns_reystress(path):
    """Load MKM DNS chan180.reystress file.

    Columns: y, y+, R_uu, R_vv, R_ww, R_uv, R_uw, R_vw
    Returns dict with keys: y, yplus, uu, vv, ww, uv.
    """
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, comments='#')
    return {
        'y': data[:, 0],
        'yplus': data[:, 1],
        'uu': data[:, 2],
        'vv': data[:, 3],
        'ww': data[:, 4],
        'uv': data[:, 5],
    }


def load_solver_profile(model_dir):
    """Load solver velocity_profile.dat from a model output directory.

    Format: y  u_numerical  u_analytical
    Returns dict with keys: y, u.
    Returns None if file not found.
    """
    path = os.path.join(model_dir, 'velocity_profile.dat')
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping model")
        return None
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        print(f"  WARNING: {path} has only 1 row, skipping")
        return None
    return {
        'y': data[:, 0],
        'u': data[:, 1],
    }


def parse_run_log(model_dir):
    """Extract u_tau and Re_tau from run.log if available."""
    log_path = os.path.join(model_dir, 'run.log')
    info = {'u_tau': None, 're_tau': None}
    if not os.path.exists(log_path):
        return info
    with open(log_path) as f:
        for line in f:
            if 'Friction velocity u_tau:' in line:
                try:
                    info['u_tau'] = float(line.split(':')[-1].strip())
                except ValueError:
                    pass
            if 'Re_tau:' in line:
                try:
                    info['re_tau'] = float(line.split(':')[-1].strip())
                except ValueError:
                    pass
    return info


# ============================================================================
# Error metrics
# ============================================================================
def compute_l2_error(yplus_dns, uplus_dns, yplus_model, uplus_model):
    """Compute relative L2 error of model vs DNS, interpolated to DNS y+ grid."""
    # Interpolate model onto DNS grid
    uplus_interp = np.interp(yplus_dns, yplus_model, uplus_model)
    diff = uplus_interp - uplus_dns
    l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(uplus_dns**2))
    return l2


def compute_linf_error(yplus_dns, uplus_dns, yplus_model, uplus_model):
    """Compute max absolute error of model vs DNS."""
    uplus_interp = np.interp(yplus_dns, yplus_model, uplus_model)
    return np.max(np.abs(uplus_interp - uplus_dns))


# ============================================================================
# Plotting
# ============================================================================
def plot_uplus(dns, model_data, output_dir):
    """Plot u+ vs y+ for all models against DNS."""
    apply_style()
    fig, ax = single_col_fig(height_ratio=0.85)

    # DNS reference
    ax.plot(dns['yplus'], dns['Umean'], 'k-', linewidth=1.2,
            label=MODEL_LABELS.get('dns', 'DNS'), zorder=10)

    # Law of the wall references
    yp_visc = np.linspace(0, 11, 50)
    ax.plot(yp_visc, yp_visc, 'k:', linewidth=0.5, alpha=0.4)
    yp_log = np.linspace(30, 180, 50)
    ax.plot(yp_log, (1/0.41) * np.log(yp_log) + 5.2, 'k:', linewidth=0.5, alpha=0.4)

    # Model results
    for model_name, mdata in model_data.items():
        if mdata is None:
            continue
        style_key = STYLE_MAP.get(model_name, model_name)
        color = COLORS.get(style_key, (0.5, 0.5, 0.5))
        label = MODEL_LABELS.get(style_key, model_name)
        ls = LINE_STYLES.get(model_name, '-')
        ax.plot(mdata['yplus'], mdata['uplus'], color=color, linestyle=ls,
                linewidth=1.0, label=label)

    ax.set_xscale('log')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    ax.set_xlim(0.5, 200)
    ax.set_ylim(0, 22)
    ax.legend(loc='upper left', fontsize=6)

    save_fig(fig, os.path.join(output_dir, 'uplus_comparison.pdf'))


def plot_uplus_linear(dns, model_data, output_dir):
    """Plot u+ vs y/delta on linear axes."""
    apply_style()
    fig, ax = single_col_fig(height_ratio=0.85)

    ax.plot(dns['y'], dns['Umean'], 'k-', linewidth=1.2,
            label=MODEL_LABELS.get('dns', 'DNS'), zorder=10)

    for model_name, mdata in model_data.items():
        if mdata is None:
            continue
        style_key = STYLE_MAP.get(model_name, model_name)
        color = COLORS.get(style_key, (0.5, 0.5, 0.5))
        label = MODEL_LABELS.get(style_key, model_name)
        ls = LINE_STYLES.get(model_name, '-')
        ax.plot(mdata['y_norm'], mdata['uplus'], color=color, linestyle=ls,
                linewidth=1.0, label=label)

    ax.set_xlabel(r'$y/\delta$')
    ax.set_ylabel(r'$u^+$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 22)
    ax.legend(loc='lower right', fontsize=6)

    save_fig(fig, os.path.join(output_dir, 'uplus_linear.pdf'))


def plot_error_bar(errors, output_dir):
    """Bar chart of L2 errors by model."""
    apply_style()
    fig, ax = single_col_fig(height_ratio=0.7)

    names = []
    vals = []
    colors = []
    for model_name, err in errors.items():
        style_key = STYLE_MAP.get(model_name, model_name)
        names.append(MODEL_LABELS.get(style_key, model_name))
        vals.append(err)
        colors.append(COLORS.get(style_key, (0.5, 0.5, 0.5)))

    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, edgecolor='k', linewidth=0.3, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel(r'Relative $L_2$ error in $u^+$')
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)

    save_fig(fig, os.path.join(output_dir, 'uplus_l2_error.pdf'))


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compare a posteriori channel results against MKM DNS')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS,
                        help='Directory containing model subdirectories')
    parser.add_argument('--output_dir', default=None,
                        help='Where to save figures (default: results_dir)')
    parser.add_argument('--dns_means', default=DNS_MEANS,
                        help='Path to chan180.means')
    parser.add_argument('--dns_reystress', default=DNS_REYSTRESS,
                        help='Path to chan180.reystress')
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load DNS reference
    # ------------------------------------------------------------------
    print("Loading DNS reference data ...")
    if not os.path.exists(args.dns_means):
        print(f"ERROR: DNS means file not found: {args.dns_means}")
        sys.exit(1)
    dns = load_dns_means(args.dns_means)
    dns_rs = load_dns_reystress(args.dns_reystress)
    re_tau_dns = 178.12
    print(f"  Re_tau (DNS) = {re_tau_dns}")
    print(f"  {len(dns['y'])} y-stations loaded")

    # ------------------------------------------------------------------
    # Load model results
    # ------------------------------------------------------------------
    print("\nLoading model results ...")
    model_data = {}
    errors_l2 = {}
    errors_linf = {}

    for model in MODELS:
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            print(f"  {model}: directory not found, skipping")
            continue

        print(f"  {model}:")
        profile = load_solver_profile(model_dir)
        if profile is None:
            continue

        log_info = parse_run_log(model_dir)
        u_tau = log_info['u_tau']
        re_tau = log_info['re_tau']

        if u_tau is None:
            # Estimate u_tau from dp_dx: u_tau = sqrt(|dp_dx| * delta)
            # For channel half-height delta=1, dp_dx=-1
            u_tau = 1.0  # sqrt(1.0 * 1.0)
            print(f"    u_tau not found in log, using estimate u_tau={u_tau:.4f}")
        else:
            print(f"    u_tau = {u_tau:.4f}, Re_tau = {re_tau}")

        nu = 0.005556  # from config
        re_tau_model = u_tau / nu if u_tau else re_tau_dns

        # Convert to wall units
        y = profile['y']
        u = profile['u']

        # y ranges from -1 to 1; use lower half (y in [-1, 0]) mapped to y+ = (y+1)*re_tau_model
        # Or use full profile and fold
        y_shifted = y - (-1.0)  # y_shifted in [0, 2], distance from bottom wall
        yplus = y_shifted * re_tau_model
        uplus = u / u_tau

        # Use only lower half (y_shifted in [0, 1])
        mask = y_shifted <= 1.0 + 1e-10
        yplus_half = yplus[mask]
        uplus_half = uplus[mask]
        y_norm = y_shifted[mask]  # y/delta

        # Sort by yplus
        idx = np.argsort(yplus_half)
        yplus_half = yplus_half[idx]
        uplus_half = uplus_half[idx]
        y_norm = y_norm[idx]

        model_data[model] = {
            'yplus': yplus_half,
            'uplus': uplus_half,
            'y_norm': y_norm,
            'u_tau': u_tau,
            're_tau': re_tau_model,
        }

        # Compute errors against DNS
        l2 = compute_l2_error(dns['yplus'], dns['Umean'], yplus_half, uplus_half)
        linf = compute_linf_error(dns['yplus'], dns['Umean'], yplus_half, uplus_half)
        errors_l2[model] = l2
        errors_linf[model] = linf
        print(f"    L2 error = {l2:.4f}, Linf error = {linf:.4f}")

    if not model_data:
        print("\nNo model data found. Run the SLURM job first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'Model':<15s}  {'u_tau':>8s}  {'Re_tau':>8s}  {'L2 err':>8s}  {'Linf err':>8s}")
    print("-" * 65)
    for model in MODELS:
        if model not in model_data:
            continue
        md = model_data[model]
        style_key = STYLE_MAP.get(model, model)
        label = MODEL_LABELS.get(style_key, model)
        print(f"{label:<15s}  {md['u_tau']:8.4f}  {md['re_tau']:8.1f}  "
              f"{errors_l2[model]:8.4f}  {errors_linf[model]:8.4f}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_uplus(dns, model_data, output_dir)
    plot_uplus_linear(dns, model_data, output_dir)
    if errors_l2:
        plot_error_bar(errors_l2, output_dir)

    print("\nDone. Figures saved to:", output_dir)


if __name__ == '__main__':
    main()
