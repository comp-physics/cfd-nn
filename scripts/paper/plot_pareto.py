#!/usr/bin/env python3
"""
Generate Pareto cost-accuracy plot for the NN turbulence paper.

Usage:
    python3 scripts/paper/plot_pareto.py              # real data (reads QoI files)
    python3 scripts/paper/plot_pareto.py --mock        # synthetic accuracy data
    python3 scripts/paper/plot_pareto.py --case sphere # specific case (default: sphere)

Reads:
    - Timing data from results/paper/timing_fixed_dt_h200.md (parsed) or hardcoded
    - Accuracy data from results/paper/aposteriori/{case}/{model}/qoi/qoi_summary.dat

Outputs:
    - paper/figures/pareto_cost_accuracy.pdf
    - paper/figures/pareto_cost_accuracy.png
"""
import argparse
import os
import re
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Determine project root (works from any CWD)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Allow importing sibling modules
sys.path.insert(0, str(SCRIPT_DIR))
from plot_style import apply_style, save_fig, COLORS, DOUBLE_COL, GOLDEN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Hardcoded H200 timing data (turb ms/step) — from timing_fixed_dt_h200.md
# Keys are per-case dictionaries; values are turb_ms.
# ============================================================================
COST_DATA = {
    'hills': {
        'baseline': 0.0, 'mixing_length': 0.054, 'komega': 0.010,
        'SST': 0.025, 'EARSM-WJ': 0.026, 'EARSM-GS': 0.025,
        'EARSM-Pope': 0.025, 'GEP': 0.025,
        'MLP': 0.454, 'MLP-med': 1.434,
        'TBNN-small': 2.074, 'TBNN': 4.372,
        'PI-TBNN-small': 2.076, 'PI-TBNN': 4.377,
        'TBRF-1t': 0.069,
    },
    'cylinder': {
        'baseline': 0.0, 'mixing_length': 0.058, 'komega': 0.010,
        'SST': 0.026, 'EARSM-WJ': 0.028, 'EARSM-GS': 0.028,
        'EARSM-Pope': 0.028, 'GEP': 0.027,
        'MLP': 0.637, 'MLP-med': 2.099,
        'TBNN-small': 3.012, 'TBNN': 6.451,
        'PI-TBNN-small': 3.008, 'PI-TBNN': 6.450,
        'TBRF-1t': 0.087,
    },
    'duct': {
        'baseline': 0.0, 'mixing_length': 0.064, 'komega': 0.010,
        'SST': 0.037, 'EARSM-WJ': 0.038, 'EARSM-GS': 0.038,
        'EARSM-Pope': 0.038, 'GEP': 0.037,
        'MLP': 4.525, 'MLP-med': 16.223,
        'TBNN-small': 13.318, 'TBNN': 40.743,
        'PI-TBNN-small': 13.286, 'PI-TBNN': 40.663,
        'TBRF-1t': 0.503,
    },
    'sphere': {
        'baseline': 0.0, 'mixing_length': 0.102, 'komega': 0.010,
        'SST': 0.074, 'EARSM-WJ': 0.077, 'EARSM-GS': 0.078,
        'EARSM-Pope': 0.077, 'GEP': 0.075,
        'MLP': 15.830, 'MLP-med': 57.367,
        'TBNN-small': 46.223, 'TBNN': 143.563,
        'PI-TBNN-small': 46.270, 'PI-TBNN': 144.313,
        'TBRF-1t': 1.588,
    },
}

# Total ms/step (for computing fraction, not used in main plot)
TOTAL_STEP_DATA = {
    'hills': {
        'baseline': 1.894, 'mixing_length': 1.976, 'komega': 1.974,
        'SST': 2.021, 'EARSM-WJ': 2.023, 'EARSM-GS': 2.015,
        'EARSM-Pope': 2.014, 'GEP': 1.965,
        'MLP': 2.405, 'MLP-med': 3.377,
        'TBNN-small': 4.070, 'TBNN': 6.333,
        'PI-TBNN-small': 4.028, 'PI-TBNN': 6.303,
        'TBRF-1t': 2.020,
    },
    'cylinder': {
        'baseline': 2.052, 'mixing_length': 2.160, 'komega': 2.149,
        'SST': 2.191, 'EARSM-WJ': 2.164, 'EARSM-GS': 2.188,
        'EARSM-Pope': 2.160, 'GEP': 2.127,
        'MLP': 2.722, 'MLP-med': 4.171,
        'TBNN-small': 5.126, 'TBNN': 8.558,
        'PI-TBNN-small': 5.134, 'PI-TBNN': 8.565,
        'TBRF-1t': 2.143,
    },
    'duct': {
        'baseline': 2.637, 'mixing_length': 2.705, 'komega': 2.734,
        'SST': 2.777, 'EARSM-WJ': 2.824, 'EARSM-GS': 2.813,
        'EARSM-Pope': 2.818, 'GEP': 2.736,
        'MLP': 7.214, 'MLP-med': 18.912,
        'TBNN-small': 16.035, 'TBNN': 43.464,
        'PI-TBNN-small': 15.994, 'PI-TBNN': 43.389,
        'TBRF-1t': 3.201,
    },
    'sphere': {
        'baseline': 6.027, 'mixing_length': 6.144, 'komega': 6.188,
        'SST': 6.240, 'EARSM-WJ': 6.321, 'EARSM-GS': 6.348,
        'EARSM-Pope': 6.325, 'GEP': 6.222,
        'MLP': 21.982, 'MLP-med': 63.513,
        'TBNN-small': 52.365, 'TBNN': 149.757,
        'PI-TBNN-small': 52.449, 'PI-TBNN': 150.534,
        'TBRF-1t': 7.716,
    },
}

# ============================================================================
# Model metadata
# ============================================================================
# Category assignment for coloring
MODEL_CATEGORY = {
    'baseline':     'classical',
    'mixing_length':'classical',
    'komega':       'classical',
    'SST':          'classical',
    'EARSM-WJ':    'classical',
    'EARSM-GS':    'classical',
    'EARSM-Pope':  'classical',
    'GEP':          'classical',
    'MLP':          'nn_scalar',
    'MLP-med':      'nn_scalar',
    'TBNN-small':   'nn_tensor',
    'TBNN':         'nn_tensor',
    'PI-TBNN-small':'nn_tensor',
    'PI-TBNN':      'nn_tensor',
    'TBRF-1t':      'nn_tree',
}

# Display names for labels
MODEL_DISPLAY = {
    'baseline':      'Baseline',
    'mixing_length': 'Mixing length',
    'komega':        r'$k$-$\omega$',
    'SST':           r'SST',
    'EARSM-WJ':     'EARSM-WJ',
    'EARSM-GS':     'EARSM-GS',
    'EARSM-Pope':   'EARSM-Pope',
    'GEP':           'GEP',
    'MLP':           'MLP',
    'MLP-med':       'MLP-med',
    'TBNN-small':    'TBNN-s',
    'TBNN':          'TBNN',
    'PI-TBNN-small': 'PI-TBNN-s',
    'PI-TBNN':       'PI-TBNN',
    'TBRF-1t':       'TBRF',
}

# Category colors and markers
CATEGORY_STYLE = {
    'classical':  dict(color='#2166ac', marker='s', zorder=5),   # blue squares
    'nn_scalar':  dict(color='#d6604d', marker='o', zorder=6),   # red circles
    'nn_tensor':  dict(color='#1b7837', marker='^', zorder=6),   # green triangles
    'nn_tree':    dict(color='#e08214', marker='D', zorder=6),   # orange diamonds
}

# Architecture families — members connected by lines
ARCH_FAMILIES = [
    ['MLP', 'MLP-med'],
    ['TBNN-small', 'TBNN'],
    ['PI-TBNN-small', 'PI-TBNN'],
]

# Reference Cd values for error computation
CD_REFERENCE = {
    'sphere':   0.80,    # Tomboulides & Orszag (2000), Re=200
    'cylinder': 1.35,    # Henderson (1995), Re=100
    'hills':    None,     # Use separation/reattachment instead
    'duct':     None,     # Use friction factor instead
}

# ============================================================================
# Data loading
# ============================================================================

def parse_timing_markdown(filepath):
    """Parse timing_fixed_dt_h200.md and extract per-case turb_ms data.

    Returns dict[case][model] = turb_ms.
    Falls back to hardcoded data if parsing fails.
    """
    if not os.path.isfile(filepath):
        warnings.warn(f"Timing file not found: {filepath}; using hardcoded data")
        return COST_DATA

    try:
        with open(filepath) as f:
            text = f.read()
    except Exception as e:
        warnings.warn(f"Cannot read {filepath}: {e}; using hardcoded data")
        return COST_DATA

    result = {}
    # Find each case section
    case_map = {
        'Hills': 'hills',
        'Cylinder': 'cylinder',
        'Duct': 'duct',
        'Sphere': 'sphere',
    }

    sections = re.split(r'###\s+', text)
    for section in sections:
        case_key = None
        for keyword, cname in case_map.items():
            if keyword in section.split('\n')[0]:
                case_key = cname
                break
        if case_key is None:
            continue

        result[case_key] = {}
        # Parse table rows: | model | ms/step | turb ms | ...
        for line in section.split('\n'):
            line = line.strip()
            if not line.startswith('|') or '---' in line or 'Model' in line:
                continue
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if len(cols) < 3:
                continue
            model_name = cols[0]
            try:
                turb_ms = float(cols[2])
            except ValueError:
                continue
            # Normalize model names
            name_map = {
                'k-omega': 'komega',
            }
            model_name = name_map.get(model_name, model_name)
            result[case_key][model_name] = turb_ms

    if not result:
        warnings.warn("Parsed no data from timing file; using hardcoded data")
        return COST_DATA
    return result


def load_qoi_accuracy(case, models, project_root):
    """Load a posteriori accuracy data from QoI summary files.

    Returns dict[model] = relative_error (|Cd - Cd_ref| / Cd_ref).
    Returns empty dict if no data found.
    """
    qoi_base = project_root / 'results' / 'paper' / 'aposteriori'
    accuracy = {}
    cd_ref = CD_REFERENCE.get(case)

    if cd_ref is None:
        return accuracy  # no reference for this case

    for model in models:
        # Try several naming conventions for model directories
        candidates = [
            qoi_base / case / model / 'qoi' / 'qoi_summary.dat',
            qoi_base / case / model.lower() / 'qoi' / 'qoi_summary.dat',
            qoi_base / case / model.replace('-', '_') / 'qoi' / 'qoi_summary.dat',
        ]
        for qoi_file in candidates:
            if qoi_file.is_file():
                try:
                    cd_mean = _parse_cd_from_summary(qoi_file)
                    if cd_mean is not None:
                        accuracy[model] = abs(cd_mean - cd_ref) / abs(cd_ref)
                except Exception as e:
                    warnings.warn(f"Error reading {qoi_file}: {e}")
                break

    return accuracy


def _parse_cd_from_summary(filepath):
    """Extract Cd_mean from a qoi_summary.dat file.

    Expected format: key = value lines, looking for Cd_mean or cd_mean.
    """
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Try key=value or key: value
            for sep in ['=', ':']:
                if sep in line:
                    key, _, val = line.partition(sep)
                    key = key.strip().lower()
                    if key in ('cd_mean', 'cd', 'drag_coefficient'):
                        try:
                            return float(val.strip())
                        except ValueError:
                            pass
    return None


def generate_mock_accuracy(models, case):
    """Generate synthetic accuracy data for layout testing.

    Creates plausible relative errors that roughly correlate with model
    sophistication (classical models have moderate error, NNs have varied).
    """
    np.random.seed(42)
    mock = {}
    base_errors = {
        'baseline':      0.35,
        'mixing_length': 0.28,
        'komega':        0.18,
        'SST':           0.12,
        'EARSM-WJ':     0.09,
        'EARSM-GS':     0.10,
        'EARSM-Pope':   0.09,
        'GEP':           0.11,
        'MLP':           0.08,
        'MLP-med':       0.06,
        'TBNN-small':    0.07,
        'TBNN':          0.04,
        'PI-TBNN-small': 0.06,
        'PI-TBNN':       0.035,
        'TBRF-1t':       0.05,
    }
    for model in models:
        if model in base_errors:
            noise = 1.0 + 0.15 * np.random.randn()
            mock[model] = max(base_errors[model] * noise, 0.005)
    return mock


# ============================================================================
# Pareto frontier
# ============================================================================

def compute_pareto_frontier(costs, errors):
    """Compute the Pareto frontier (lower-left optimal).

    Args:
        costs: array of cost values
        errors: array of error values

    Returns:
        indices of Pareto-optimal points, sorted by cost.
    """
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in both and < in at least one
            if costs[j] <= costs[i] and errors[j] <= errors[i]:
                if costs[j] < costs[i] or errors[j] < errors[i]:
                    is_pareto[i] = False
                    break
    pareto_idx = np.where(is_pareto)[0]
    # Sort by cost
    pareto_idx = pareto_idx[np.argsort(costs[pareto_idx])]
    return pareto_idx


# ============================================================================
# Plotting
# ============================================================================

def plot_pareto(cost_data, accuracy_data, case, output_dir, use_total_cost=False):
    """Generate the Pareto cost-accuracy figure.

    Args:
        cost_data: dict[model] = turb_ms (or total ms/step)
        accuracy_data: dict[model] = relative_error
        case: case name for title
        output_dir: Path for output files
        use_total_cost: if True, x-axis is total ms/step instead of turb ms
    """
    apply_style()

    # Collect models that have both cost and accuracy
    models = []
    costs = []
    errors = []
    for model in cost_data:
        if model not in accuracy_data:
            continue
        if model == 'baseline' and cost_data[model] == 0:
            # Use a small positive value for log scale
            costs.append(0.005)
        else:
            costs.append(cost_data[model])
        errors.append(accuracy_data[model])
        models.append(model)

    if len(models) < 2:
        print(f"WARNING: Only {len(models)} models with both cost and accuracy data. "
              "Use --mock for synthetic data.")
        return

    costs = np.array(costs)
    errors = np.array(errors)

    # Create figure
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    # --- Plot each model as a labeled point ---
    for i, model in enumerate(models):
        cat = MODEL_CATEGORY.get(model, 'classical')
        style = CATEGORY_STYLE[cat]
        ax.scatter(costs[i], errors[i],
                   c=[style['color']], marker=style['marker'],
                   s=40, edgecolors='k', linewidths=0.3,
                   zorder=style['zorder'])

    # --- Connect architecture families ---
    for family in ARCH_FAMILIES:
        fam_idx = [j for j, m in enumerate(models) if m in family]
        if len(fam_idx) >= 2:
            fam_idx_sorted = sorted(fam_idx, key=lambda j: costs[j])
            cat = MODEL_CATEGORY.get(models[fam_idx_sorted[0]], 'classical')
            col = CATEGORY_STYLE[cat]['color']
            ax.plot(costs[fam_idx_sorted], errors[fam_idx_sorted],
                    '-', color=col, linewidth=0.6, alpha=0.5, zorder=2)

    # --- Pareto frontier ---
    pareto_idx = compute_pareto_frontier(costs, errors)
    if len(pareto_idx) >= 2:
        # Extend to plot edges for visual clarity
        pc = costs[pareto_idx]
        pe = errors[pareto_idx]
        ax.step(pc, pe, where='post',
                linestyle='--', color='k', linewidth=0.8, alpha=0.4,
                zorder=1, label='Pareto frontier')
        # Extend downward from last point and leftward from first point
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # --- Labels ---
    # Smart label placement to reduce overlap
    for i, model in enumerate(models):
        label = MODEL_DISPLAY.get(model, model)
        x, y = costs[i], errors[i]

        # Default offset
        ha, va = 'left', 'bottom'
        dx, dy = 0.08, 0.04  # in axes fraction — we will use offset points instead

        # Manual adjustments for known overlapping models
        if model in ('EARSM-GS', 'EARSM-Pope', 'GEP', 'SST'):
            ha, va = 'left', 'top'
        if model in ('PI-TBNN-small',):
            ha, va = 'right', 'bottom'
        if model in ('PI-TBNN',):
            ha, va = 'right', 'top'
        if model in ('baseline',):
            ha, va = 'right', 'bottom'
        if model in ('komega',):
            ha, va = 'left', 'top'

        offset_x = 5 if ha == 'left' else -5
        offset_y = 3 if va == 'bottom' else -3

        ax.annotate(label, (x, y),
                    textcoords='offset points',
                    xytext=(offset_x, offset_y),
                    fontsize=5.5, ha=ha, va=va, color='0.2')

    # --- Axes ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel = 'Total step cost (ms)' if use_total_cost else 'Turbulence model cost (ms/step)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Relative $C_d$ error $\;|C_d - C_{d,\mathrm{ref}}|/C_{d,\mathrm{ref}}$')

    case_labels = {
        'sphere': r'Sphere $Re = 200$',
        'cylinder': r'Cylinder $Re = 100$',
        'hills': r'Periodic hills $Re_H = 10{,}595$',
        'duct': r'Square duct $Re_b = 3{,}500$',
    }
    ax.set_title(case_labels.get(case, case), fontsize=9, pad=8)

    # Minor ticks for log scale
    ax.minorticks_on()

    # --- Reference regions ---
    # Shade classical regime vs NN regime
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Classical threshold: ~0.15 ms
    classical_threshold = 0.15
    ax.axvline(x=classical_threshold, color='0.7', linewidth=0.5,
               linestyle=':', zorder=0)
    # Label the regions at the top
    ax.text(classical_threshold * 0.4, ylims[1] * 0.7,
            'Classical\nRANS', fontsize=5, color='0.5',
            ha='center', va='top', style='italic')
    ax.text(classical_threshold * 8, ylims[1] * 0.7,
            'Neural\nnetwork', fontsize=5, color='0.5',
            ha='center', va='top', style='italic')

    # --- Legend by category ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2166ac',
               markeredgecolor='k', markeredgewidth=0.3, markersize=5,
               label='Classical RANS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d6604d',
               markeredgecolor='k', markeredgewidth=0.3, markersize=5,
               label='NN scalar'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#1b7837',
               markeredgecolor='k', markeredgewidth=0.3, markersize=5,
               label='NN tensor'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e08214',
               markeredgecolor='k', markeredgewidth=0.3, markersize=5,
               label='NN tree'),
        Line2D([0], [0], linestyle='--', color='k', linewidth=0.8, alpha=0.4,
               label='Pareto frontier'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=6,
              frameon=True, fancybox=False, edgecolor='0.8',
              framealpha=0.9)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    fig.tight_layout()

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / 'pareto_cost_accuracy.pdf'
    png_path = output_dir / 'pareto_cost_accuracy.png'

    save_fig(fig, str(pdf_path), close=False)
    fig.savefig(str(png_path), format='png', dpi=300)
    print(f"  Saved: {png_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Pareto cost-accuracy plot for NN turbulence paper')
    parser.add_argument('--mock', action='store_true',
                        help='Use synthetic accuracy data for layout testing')
    parser.add_argument('--case', default='sphere',
                        choices=['sphere', 'cylinder', 'hills', 'duct'],
                        help='Flow case to plot (default: sphere)')
    parser.add_argument('--total-cost', action='store_true',
                        help='Use total ms/step instead of turbulence-only cost')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    case = args.case

    # --- Load cost data ---
    timing_file = PROJECT_ROOT / 'results' / 'paper' / 'timing_fixed_dt_h200.md'
    all_cost = parse_timing_markdown(str(timing_file))

    if case not in all_cost:
        print(f"ERROR: No cost data for case '{case}'")
        sys.exit(1)

    if args.total_cost:
        cost = TOTAL_STEP_DATA.get(case, all_cost[case])
    else:
        cost = all_cost[case]

    models = list(cost.keys())

    # --- Load accuracy data ---
    if args.mock:
        print(f"Using mock accuracy data for case '{case}'")
        accuracy = generate_mock_accuracy(models, case)
    else:
        accuracy = load_qoi_accuracy(case, models, PROJECT_ROOT)
        if not accuracy:
            print(f"WARNING: No a posteriori QoI data found for case '{case}'.")
            print("         Falling back to --mock mode for layout testing.")
            accuracy = generate_mock_accuracy(models, case)

    # Report
    n_both = sum(1 for m in models if m in accuracy)
    n_cost_only = sum(1 for m in models if m not in accuracy)
    print(f"Case: {case}")
    print(f"  Models with cost + accuracy: {n_both}")
    if n_cost_only > 0:
        missing = [m for m in models if m not in accuracy]
        print(f"  Models with cost only (skipped): {n_cost_only} — {missing}")

    # --- Plot ---
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'paper' / 'figures'
    plot_pareto(cost, accuracy, case, output_dir, use_total_cost=args.total_cost)


if __name__ == '__main__':
    main()
