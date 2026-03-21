# RANS Validation Campaign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a complete validation report proving each RANS turbulence model correctly simulates channel flow at Re_tau ~ 180, with convergence histories, accuracy vs MKM DNS, and grid independence evidence.

**Architecture:** Fix the baseline GPU u_tau bug, create per-model config files and a SLURM submission script, run all 8 models + 3 grid-convergence cases on GPU, then use a Python script to parse output, generate matplotlib plots, and write a markdown report.

**Tech Stack:** C++17 (solver), Python 3 (matplotlib, numpy for analysis), SLURM (GPU jobs), Bash (orchestration)

---

### Task 1: Fix Baseline Mixing-Length GPU u_tau Bug

**Files:**
- Modify: `src/turbulence_baseline.cpp:207-304` (GPU path)

The bug: lines 224-225 read `velocity.u(i, j)` from HOST memory, which is stale during GPU execution. The fix: move u_tau computation AFTER the gradient computation (line 246-264) and compute it from the GPU-resident `dudy` field via a GPU reduction.

**Step 1: Implement GPU u_tau computation**

Replace lines 212-233 (the u_tau block that reads HOST velocity) with a GPU reduction that runs AFTER the gradient computation at line 264. The new code computes u_tau from the wall-adjacent `dudy` values already on GPU:

```cpp
// After gradient computation (line 264), compute u_tau from GPU-resident dudy
double u_tau = 0.0;
{
    const double* dudy_d = device_view->dudy;
    const int Ng_ut = Ng;
    const int Nx_ut = Nx;
    const int stride_ut = device_view->cell_stride;
    const int j_wall = Ng_ut;  // First interior row (adjacent to bottom wall)
    const double nu_ut = nu_;
    double dudy_sum = 0.0;

    #pragma omp target teams distribute parallel for \
        map(present: dudy_d[0:cell_total_size]) reduction(+:dudy_sum)
    for (int i = 0; i < Nx_ut; ++i) {
        int idx = j_wall * stride_ut + (i + Ng_ut);
        double val = dudy_d[idx];
        if (val < 0.0) val = -val;
        dudy_sum += val;
    }
    dudy_sum /= Nx_ut;
    double tau_w = nu_ut * dudy_sum;
    u_tau = std::sqrt(tau_w);
}
u_tau = std::max(u_tau, 1e-10);
```

This replaces the old lines 212-233. The gradient computation (lines 246-264) stays where it is but must come BEFORE this new u_tau block. So the restructured GPU path is:

1. Compute gradients on GPU (existing code, lines 246-264)
2. Compute u_tau from GPU dudy via reduction (NEW)
3. Run mixing-length kernel (existing code, lines 282-300)

**Step 2: Verify CPU fast tests still pass**

Run: `cd build && ctest -L fast --output-on-failure`
Expected: All non-GPU tests pass (33/35, 2 GPU tests fail on CPU node)

**Step 3: Commit**

```
git add src/turbulence_baseline.cpp
git commit -m "Fix baseline mixing-length GPU u_tau: use GPU-resident dudy instead of stale HOST velocity"
```

---

### Task 2: Create Validation Config Files

**Files:**
- Create: `scripts/rans_validation/configs/rans_retau180_base.cfg`
- Create: `scripts/rans_validation/configs/` — 8 model configs + 2 grid variants (10 total)

**Step 1: Create base config for Re_tau=180 RANS**

All models share the same physics. Differences are only `turb_model` and `output_dir`.

`scripts/rans_validation/configs/rans_retau180_base.cfg`:
```ini
# RANS Channel Flow Validation - Re_tau = 180
# Base config: override turb_model and output_dir per model

# Grid
Nx = 64
Ny = 128

# Domain
x_min = 0.0
x_max = 6.283185307179586
y_min = -1.0
y_max = 1.0

# Physics: Re_tau = 180 => u_tau = 1.0, nu = 1/180 = 0.005556
nu = 0.005556
dp_dx = -1.0

# Solver
tol = 1e-8
max_steps = 50000
dt = 0.001
adaptive_dt = true
CFL_max = 0.5

# Numerics
convective_scheme = upwind

# Stretched y-grid
stretch_y = true
stretch_beta = 2.0

# Output
output_freq = 500
verbose = true
postprocess = true
write_fields = false

# Poisson
poisson_tol = 1e-6
poisson_max_vcycles = 16
```

**Step 2: Create per-model wrapper configs**

These are minimal — they include the base config via CLI override. Actually, the solver doesn't support config `include` directives, so we use a single base config and override `turb_model` and `output_dir` via CLI args `--model` and `--output_dir`.

But we do need separate configs for the grid convergence study (different Nx, Ny). Create:

`scripts/rans_validation/configs/grid_32x64.cfg` — same as base but `Nx=32`, `Ny=64`
`scripts/rans_validation/configs/grid_128x256.cfg` — same as base but `Nx=128`, `Ny=256`

**Step 3: Commit**

```
git add scripts/rans_validation/configs/
git commit -m "Add RANS validation config files for Re_tau=180 channel"
```

---

### Task 3: Create SLURM Submission Script

**Files:**
- Create: `scripts/rans_validation/run_rans_validation.sbatch`

**Step 1: Write the SLURM script**

This script:
1. Builds the GPU binary (if needed)
2. Runs all 8 models on the 64x128 base grid
3. Runs SST on 32x64 and 128x256 for grid convergence
4. Saves all output to `output/rans_validation/logs/`

```bash
#!/bin/bash
#SBATCH -J rans_validation
#SBATCH -A gts-sbryngelson3
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH -p gpu-h200
#SBATCH -G 1
#SBATCH --qos=embers
#SBATCH -t 04:00:00
#SBATCH -o output/rans_validation/slurm-%j.out
#SBATCH -e output/rans_validation/slurm-%j.err

set -euo pipefail

module reset
module load nvhpc
export OMP_TARGET_OFFLOAD=MANDATORY

PROJ_DIR=/storage/scratch1/6/sbryngelson3/cfd-nn
BUILD_DIR=${PROJ_DIR}/build_rans_validation
CFG_DIR=${PROJ_DIR}/scripts/rans_validation/configs
OUT_DIR=${PROJ_DIR}/output/rans_validation
LOG_DIR=${OUT_DIR}/logs

mkdir -p "${LOG_DIR}" "${OUT_DIR}/data"

# Build GPU binary
if [ ! -f "${BUILD_DIR}/channel" ]; then
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake "${PROJ_DIR}" -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON \
          -DCMAKE_BUILD_TYPE=Release -DGPU_CC=90
    make -j$(nproc) channel
fi

BINARY="${BUILD_DIR}/channel"
BASE_CFG="${CFG_DIR}/rans_retau180_base.cfg"

# ============================================================
# Part 1: All 8 models on 64x128 base grid
# ============================================================
MODELS="none baseline gep earsm sst komega nn_mlp nn_tbnn"

for model in ${MODELS}; do
    echo "========================================"
    echo "=== RANS model: ${model}"
    echo "========================================"

    model_out="${OUT_DIR}/data/${model}"
    mkdir -p "${model_out}"

    ${BINARY} --config "${BASE_CFG}" \
        --model "${model}" \
        --output_dir "${model_out}/" \
        --max_steps 50000 \
        2>&1 | tee "${LOG_DIR}/${model}.log"

    echo "=== ${model} exit code: $? ==="
    echo ""
done

# ============================================================
# Part 2: Grid convergence (SST on 3 grids)
# ============================================================
for grid_cfg in grid_32x64.cfg grid_128x256.cfg; do
    grid_name="${grid_cfg%.cfg}"
    echo "========================================"
    echo "=== Grid convergence: SST on ${grid_name}"
    echo "========================================"

    grid_out="${OUT_DIR}/data/sst_${grid_name}"
    mkdir -p "${grid_out}"

    ${BINARY} --config "${CFG_DIR}/${grid_cfg}" \
        --model sst \
        --output_dir "${grid_out}/" \
        --max_steps 50000 \
        2>&1 | tee "${LOG_DIR}/sst_${grid_name}.log"

    echo "=== sst_${grid_name} exit code: $? ==="
    echo ""
done

echo "========================================"
echo "=== All validation runs complete ==="
echo "========================================"
echo "Logs in: ${LOG_DIR}"
echo "Data in: ${OUT_DIR}/data"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/rans_validation/run_rans_validation.sbatch
git add scripts/rans_validation/run_rans_validation.sbatch
git commit -m "Add SLURM script for RANS validation campaign on GPU"
```

---

### Task 4: Create Python Analysis Script

**Files:**
- Create: `scripts/rans_validation/analyze.py`

This is the most complex task. The script:
1. Parses solver log files to extract residual histories, Re_tau, u_tau, velocity profiles
2. Loads MKM DNS reference data
3. Generates matplotlib plots
4. Writes a machine-readable summary JSON
5. Writes the markdown report

**Step 1: Write the parser and plot generator**

`scripts/rans_validation/analyze.py`:

```python
#!/usr/bin/env python3
"""
RANS Validation Analysis Script

Parses solver output logs, loads MKM DNS reference data, generates
comparison plots, and writes a markdown validation report.

Usage:
    python3 analyze.py [--data-dir output/rans_validation]
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Constants
# ============================================================
MKM_RETAU = 178.12
MODELS_ACCURACY = ['none', 'baseline', 'gep', 'earsm', 'sst', 'komega']
MODELS_STABILITY = ['nn_mlp', 'nn_tbnn']
MODELS_ALL = MODELS_ACCURACY + MODELS_STABILITY
MODEL_LABELS = {
    'none': 'Laminar',
    'baseline': 'Mixing Length',
    'gep': 'GEP',
    'earsm': 'EARSM (WJ)',
    'sst': 'SST k-omega',
    'komega': 'k-omega',
    'nn_mlp': 'NN-MLP (untrained)',
    'nn_tbnn': 'NN-TBNN (untrained)',
}

# ============================================================
# Parsing Functions
# ============================================================

def parse_log(logfile):
    """Parse a solver log file. Returns dict with residual history,
    final Re_tau, u_tau, bulk velocity, and velocity profile."""
    result = {
        'steps': [],
        'residuals': [],
        'retau_history': [],
        'final_retau': None,
        'final_utau': None,
        'final_bulk_vel': None,
        'converged': False,
        'exit_code': None,
        'profile_y': [],
        'profile_u': [],
    }

    if not os.path.exists(logfile):
        return result

    with open(logfile) as f:
        for line in f:
            # Step output: "Step  123 / 50000 (  0%)  t*=0.00  residual = 1.23e-04"
            m = re.search(r'Step\s+(\d+)\s*/.*residual\s*=\s*([\d.eE+-]+)', line)
            if m:
                result['steps'].append(int(m.group(1)))
                result['residuals'].append(float(m.group(2)))

            # HEALTH output: "[HEALTH] step=1200 Re_tau=222.1 ..."
            m = re.search(r'\[HEALTH\]\s+step=(\d+)\s+Re_tau=([\d.]+)', line)
            if m:
                result['retau_history'].append((int(m.group(1)), float(m.group(2))))

            # Final output: "Re_tau: 278.432"
            m = re.search(r'^Re_tau:\s+([\d.]+)', line)
            if m:
                result['final_retau'] = float(m.group(1))

            m = re.search(r'^Friction velocity u_tau:\s+([\d.eE+-]+)', line)
            if m:
                result['final_utau'] = float(m.group(1))

            m = re.search(r'^Bulk velocity:\s+([\d.eE+-]+)', line)
            if m:
                result['final_bulk_vel'] = float(m.group(1))

            # Convergence
            if 'Converged' in line or 'converged' in line:
                result['converged'] = True

            # Exit code from our wrapper
            m = re.search(r'exit code:\s*(\d+)', line)
            if m:
                result['exit_code'] = int(m.group(1))

    return result


def parse_velocity_profile(profile_file):
    """Parse velocity_profile.dat: columns y, u_numerical, u_analytical."""
    y, u = [], []
    if not os.path.exists(profile_file):
        return np.array(y), np.array(u)
    with open(profile_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                y.append(float(parts[0]))
                u.append(float(parts[1]))
    return np.array(y), np.array(u)


def load_mkm_means(mkm_dir):
    """Load MKM chan180.means: columns y, y+, Umean, ..."""
    fname = os.path.join(mkm_dir, 'chan180', 'profiles', 'chan180.means')
    data = np.loadtxt(fname, comments='#')
    return {
        'y': data[:, 0],
        'yplus': data[:, 1],
        'Umean': data[:, 2],
    }


def load_mkm_reystress(mkm_dir):
    """Load MKM chan180.reystress: columns y, y+, R_uu, R_vv, R_ww, R_uv."""
    fname = os.path.join(mkm_dir, 'chan180', 'profiles', 'chan180.reystress')
    data = np.loadtxt(fname, comments='#')
    return {
        'y': data[:, 0],
        'yplus': data[:, 1],
        'R_uu': data[:, 2],
        'R_vv': data[:, 3],
        'R_ww': data[:, 4],
        'R_uv': data[:, 5],
    }


# ============================================================
# Plotting Functions
# ============================================================

def to_wall_units(y, u, nu, utau):
    """Convert physical y, u to wall units y+, u+.
    y is measured from wall (y=-1 is bottom wall for channel [-1,1])."""
    yplus = np.abs(y + 1.0) * utau / nu  # distance from bottom wall
    uplus = u / utau
    return yplus, uplus


def plot_uplus_overlay(results, mkm, nu, plot_dir):
    """Plot u+(y+) for all accuracy models vs MKM DNS."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # MKM reference (bottom half: y+ from 0 to ~180)
    mask = mkm['yplus'] <= MKM_RETAU
    ax.plot(mkm['yplus'][mask], mkm['Umean'][mask], 'k-', linewidth=2,
            label='MKM DNS', zorder=10)

    # Law of the wall reference lines
    yp_visc = np.linspace(0.1, 11, 50)
    ax.plot(yp_visc, yp_visc, 'k--', linewidth=0.8, alpha=0.5, label='u+ = y+')
    yp_log = np.linspace(30, 200, 50)
    ax.plot(yp_log, (1/0.41) * np.log(yp_log) + 5.2, 'k:', linewidth=0.8,
            alpha=0.5, label='Log law')

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS_ACCURACY)))

    for i, model in enumerate(MODELS_ACCURACY):
        r = results.get(model)
        if r is None or len(r['profile_y']) == 0:
            continue
        utau = r['final_utau'] or 1.0
        yp, up = to_wall_units(np.array(r['profile_y']), np.array(r['profile_u']),
                               nu, utau)
        # Only bottom half (y+ > 0, y < 0)
        mask = (np.array(r['profile_y']) < 0) & (yp > 0.1)
        if mask.sum() > 0:
            ax.plot(yp[mask], up[mask], 'o-', color=colors[i], markersize=3,
                    linewidth=1.2, label=MODEL_LABELS[model])

    ax.set_xscale('log')
    ax.set_xlabel('y+')
    ax.set_ylabel('u+')
    ax.set_title('Mean Velocity Profile: RANS Models vs MKM DNS (Re_tau = 180)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'u_plus_all_models.png'), dpi=150)
    plt.close(fig)
    print(f"  Wrote {plot_dir}/u_plus_all_models.png")


def plot_residual_history(results, plot_dir):
    """Plot residual vs step for all models."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS_ALL)))

    for i, model in enumerate(MODELS_ALL):
        r = results.get(model)
        if r is None or len(r['steps']) == 0:
            continue
        ax.semilogy(r['steps'], r['residuals'], '-', color=colors[i],
                     linewidth=1.2, label=MODEL_LABELS[model])

    ax.set_xlabel('Step')
    ax.set_ylabel('Residual (L-inf velocity change)')
    ax.set_title('Convergence History')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'residual_history.png'), dpi=150)
    plt.close(fig)
    print(f"  Wrote {plot_dir}/residual_history.png")


def plot_grid_convergence(grid_results, mkm, nu, plot_dir):
    """Plot u+(y+) for SST on 3 grids + MKM."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    mask = mkm['yplus'] <= MKM_RETAU
    ax.plot(mkm['yplus'][mask], mkm['Umean'][mask], 'k-', linewidth=2,
            label='MKM DNS', zorder=10)

    grid_labels = {
        'sst': 'SST 64x128',
        'sst_grid_32x64': 'SST 32x64',
        'sst_grid_128x256': 'SST 128x256',
    }
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, (key, label) in enumerate(grid_labels.items()):
        r = grid_results.get(key)
        if r is None or len(r['profile_y']) == 0:
            continue
        utau = r['final_utau'] or 1.0
        yp, up = to_wall_units(np.array(r['profile_y']), np.array(r['profile_u']),
                               nu, utau)
        mask_y = (np.array(r['profile_y']) < 0) & (yp > 0.1)
        if mask_y.sum() > 0:
            ax.plot(yp[mask_y], up[mask_y], 'o-', color=colors[i], markersize=3,
                    linewidth=1.2, label=label)

    ax.set_xscale('log')
    ax.set_xlabel('y+')
    ax.set_ylabel('u+')
    ax.set_title('Grid Convergence: SST k-omega')
    ax.legend(fontsize=9)
    ax.set_xlim(0.1, 300)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'grid_convergence_sst.png'), dpi=150)
    plt.close(fig)
    print(f"  Wrote {plot_dir}/grid_convergence_sst.png")


def compute_l2_error(y_comp, u_comp, mkm, nu, utau):
    """Compute L2 error of u+(y+) vs MKM, interpolated to MKM y+ points."""
    yp_comp, up_comp = to_wall_units(np.array(y_comp), np.array(u_comp), nu, utau)

    # Only use bottom half
    mask_comp = (np.array(y_comp) < 0) & (yp_comp > 0.5)
    if mask_comp.sum() < 5:
        return float('nan')

    yp_c = yp_comp[mask_comp]
    up_c = up_comp[mask_comp]

    # Sort by y+
    order = np.argsort(yp_c)
    yp_c = yp_c[order]
    up_c = up_c[order]

    # Interpolate to MKM y+ points within our range
    mkm_mask = (mkm['yplus'] >= yp_c.min()) & (mkm['yplus'] <= yp_c.max())
    if mkm_mask.sum() < 3:
        return float('nan')

    up_interp = np.interp(mkm['yplus'][mkm_mask], yp_c, up_c)
    up_mkm = mkm['Umean'][mkm_mask]

    l2 = np.sqrt(np.mean((up_interp - up_mkm)**2))
    return l2


# ============================================================
# Report Generation
# ============================================================

def write_summary_json(results, grid_results, nu, mkm, out_dir):
    """Write machine-readable summary."""
    summary = {}
    for model in MODELS_ALL:
        r = results.get(model, {})
        utau = r.get('final_utau') or 0.0
        entry = {
            'retau': r.get('final_retau'),
            'utau': utau,
            'bulk_velocity': r.get('final_bulk_vel'),
            'converged': r.get('converged', False),
            'final_residual': r['residuals'][-1] if r.get('residuals') else None,
            'total_steps': r['steps'][-1] if r.get('steps') else None,
        }
        if model in MODELS_ACCURACY and r.get('profile_y'):
            entry['l2_error_uplus'] = compute_l2_error(
                r['profile_y'], r['profile_u'], mkm, nu, utau)
        summary[model] = entry

    # Grid convergence
    gc = {}
    for key in ['sst', 'sst_grid_32x64', 'sst_grid_128x256']:
        r = grid_results.get(key, {})
        gc[key] = {
            'retau': r.get('final_retau'),
            'converged': r.get('converged', False),
        }
    summary['grid_convergence'] = gc

    fname = os.path.join(out_dir, 'summary.json')
    with open(fname, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote {fname}")
    return summary


def write_markdown_report(results, grid_results, summary, nu, plot_dir, out_dir):
    """Write the final markdown validation report."""
    report_path = os.path.join(out_dir, '..', '..', 'docs', 'RANS_VALIDATION_REPORT.md')

    # Build relative path from docs/ to plots
    plot_rel = '../output/rans_validation/plots'

    lines = []
    lines.append('# RANS Turbulence Model Validation Report')
    lines.append('')
    lines.append(f'**Date**: {__import__("datetime").date.today().isoformat()}')
    lines.append(f'**Reference**: Moser, Kim & Mansour (1999), Re_tau = {MKM_RETAU}')
    lines.append(f'**Flow**: Fully-developed 2D channel, periodic x, no-slip walls')
    lines.append(f'**Physics**: nu = {nu}, dp/dx = -1.0, half-channel height h = 1.0')
    lines.append('')

    # Executive summary table
    lines.append('## Executive Summary')
    lines.append('')
    lines.append('| Model | Type | Re_tau | L2(u+) | Converged | Steps | Status |')
    lines.append('|-------|------|--------|--------|-----------|-------|--------|')

    for model in MODELS_ALL:
        s = summary.get(model, {})
        label = MODEL_LABELS.get(model, model)
        mtype = 'Algebraic' if model in ['baseline', 'gep', 'earsm'] else \
                'Transport' if model in ['sst', 'komega'] else \
                'NN' if model.startswith('nn_') else 'Reference'
        retau = f"{s.get('retau', 0):.1f}" if s.get('retau') else 'N/A'
        l2 = s.get('l2_error_uplus')
        l2_str = f"{l2:.3f}" if l2 and not np.isnan(l2) else 'N/A'
        conv = 'Yes' if s.get('converged') else 'No'
        steps = str(s.get('total_steps', 'N/A'))

        # Status logic
        stable = s.get('final_residual') is not None
        if not stable:
            status = 'FAILED'
        elif model in MODELS_STABILITY:
            status = 'STABLE' if stable else 'FAILED'
        elif s.get('converged'):
            status = 'PASS'
        else:
            status = 'PARTIAL'

        lines.append(f'| {label} | {mtype} | {retau} | {l2_str} | {conv} | {steps} | {status} |')

    lines.append('')

    # Setup section
    lines.append('## Setup')
    lines.append('')
    lines.append('### Grid')
    lines.append('- 64 x 128 cells (Nx x Ny), 2D')
    lines.append('- Stretched y-grid (beta = 2.0) for wall resolution')
    lines.append('- Domain: [0, 2*pi] x [-1, 1]')
    lines.append('- Periodic BC in x, no-slip walls at y = +/-1')
    lines.append('')
    lines.append('### Physics')
    lines.append(f'- Kinematic viscosity: nu = {nu}')
    lines.append('- Pressure gradient: dp/dx = -1.0')
    lines.append(f'- Target friction Reynolds number: Re_tau = 180 (MKM DNS: {MKM_RETAU})')
    lines.append('- Friction velocity: u_tau = sqrt(|dp/dx| * h) = 1.0')
    lines.append('')
    lines.append('### Solver')
    lines.append('- Upwind convective scheme (1st order, stable for steady RANS)')
    lines.append('- Adaptive time stepping, CFL_max = 0.5')
    lines.append('- Convergence criterion: residual < 1e-8 or max 50000 steps')
    lines.append('- Multigrid Poisson solver (auto-selected for stretched y-grid)')
    lines.append('')

    # Convergence
    lines.append('## Convergence Results')
    lines.append('')
    lines.append(f'![Residual History]({plot_rel}/residual_history.png)')
    lines.append('')
    for model in MODELS_ALL:
        s = summary.get(model, {})
        label = MODEL_LABELS.get(model, model)
        res = s.get('final_residual')
        if res is not None:
            lines.append(f'- **{label}**: final residual = {res:.2e}, '
                        f'steps = {s.get("total_steps", "?")}')
        else:
            lines.append(f'- **{label}**: no data')
    lines.append('')

    # Accuracy
    lines.append('## Accuracy vs MKM DNS')
    lines.append('')
    lines.append(f'![u+ vs y+]({plot_rel}/u_plus_all_models.png)')
    lines.append('')
    lines.append('### L2 Error in u+(y+)')
    lines.append('')
    lines.append('| Model | Re_tau | L2(u+) | u_tau |')
    lines.append('|-------|--------|--------|-------|')
    for model in MODELS_ACCURACY:
        s = summary.get(model, {})
        label = MODEL_LABELS.get(model, model)
        retau = f"{s.get('retau', 0):.1f}" if s.get('retau') else 'N/A'
        l2 = s.get('l2_error_uplus')
        l2_str = f"{l2:.3f}" if l2 and not np.isnan(l2) else 'N/A'
        utau = f"{s.get('utau', 0):.4f}" if s.get('utau') else 'N/A'
        lines.append(f'| {label} | {retau} | {l2_str} | {utau} |')
    lines.append('')

    # Grid convergence
    lines.append('## Grid Convergence (SST k-omega)')
    lines.append('')
    lines.append(f'![Grid Convergence]({plot_rel}/grid_convergence_sst.png)')
    lines.append('')
    lines.append('| Grid | Re_tau | Converged |')
    lines.append('|------|--------|-----------|')
    for key, label in [('sst_grid_32x64', '32x64'), ('sst', '64x128'),
                        ('sst_grid_128x256', '128x256')]:
        gc = summary.get('grid_convergence', {}).get(key, {})
        retau = f"{gc.get('retau', 0):.1f}" if gc.get('retau') else 'N/A'
        conv = 'Yes' if gc.get('converged') else 'No'
        lines.append(f'| {label} | {retau} | {conv} |')
    lines.append('')

    # NN stability
    lines.append('## Neural Network Stability (Untrained Weights)')
    lines.append('')
    lines.append('These models use random (untrained) weights and are not expected to')
    lines.append('match DNS. They are included to verify the solver remains stable with')
    lines.append('arbitrary NN-predicted eddy viscosity fields.')
    lines.append('')
    for model in MODELS_STABILITY:
        s = summary.get(model, {})
        label = MODEL_LABELS.get(model, model)
        res = s.get('final_residual')
        steps = s.get('total_steps', '?')
        status = 'Stable' if res is not None else 'Failed'
        lines.append(f'- **{label}**: {status}, {steps} steps, '
                    f'final residual = {res:.2e}' if res else f'- **{label}**: no data')
    lines.append('')

    # Known limitations
    lines.append('## Known Limitations')
    lines.append('')
    lines.append('1. **Upwind scheme**: 1st-order convection introduces numerical diffusion. '
                'Higher-order schemes (central, skew) may change accuracy.')
    lines.append('2. **2D only**: This validation uses 2D RANS. 3D validation is future work.')
    lines.append('3. **NN models untrained**: MLP and TBNN use random weights; accuracy requires '
                'training on DNS/experimental data.')
    lines.append('4. **Single Re_tau**: Only Re_tau = 180 validated. Higher Re requires finer grids.')
    lines.append('')

    lines.append('## References')
    lines.append('')
    lines.append('- Moser, R.D., Kim, J. & Mansour, N.N. (1999). DNS of Turbulent Channel '
                'Flow up to Re_tau=590. *Physics of Fluids*, 11, 943-945.')
    lines.append('- Weatheritt, J. & Sandberg, R.D. (2016). A novel evolutionary algorithm '
                'applied to an EARSM. *J. Comp. Phys.*, 325, 22-37.')
    lines.append('- Menter, F.R. (1994). Two-equation eddy-viscosity turbulence models for '
                'engineering applications. *AIAA J.*, 32(8), 1598-1605.')
    lines.append('- Wallin, S. & Johansson, A.V. (2000). An explicit algebraic Reynolds stress '
                'model. *J. Fluid Mech.*, 403, 89-132.')
    lines.append('')

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote {report_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='RANS Validation Analysis')
    parser.add_argument('--data-dir', default='output/rans_validation',
                       help='Root output directory')
    parser.add_argument('--mkm-dir', default='data/reference/mkm_retau180',
                       help='MKM reference data directory')
    parser.add_argument('--nu', type=float, default=0.005556,
                       help='Kinematic viscosity')
    args = parser.parse_args()

    log_dir = os.path.join(args.data_dir, 'logs')
    plot_dir = os.path.join(args.data_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading MKM reference data...")
    mkm = load_mkm_means(args.mkm_dir)

    print("Parsing solver logs...")
    results = {}
    for model in MODELS_ALL:
        logfile = os.path.join(log_dir, f'{model}.log')
        r = parse_log(logfile)

        # Also load velocity profile from output data dir
        profile_file = os.path.join(args.data_dir, 'data', model, 'velocity_profile.dat')
        y, u = parse_velocity_profile(profile_file)
        r['profile_y'] = y.tolist()
        r['profile_u'] = u.tolist()

        results[model] = r
        print(f"  {model}: {len(r['steps'])} steps, "
              f"Re_tau={r['final_retau']}, converged={r['converged']}")

    # Grid convergence results
    grid_results = {'sst': results.get('sst', {})}
    for grid in ['grid_32x64', 'grid_128x256']:
        key = f'sst_{grid}'
        logfile = os.path.join(log_dir, f'{key}.log')
        r = parse_log(logfile)
        profile_file = os.path.join(args.data_dir, 'data', key, 'velocity_profile.dat')
        y, u = parse_velocity_profile(profile_file)
        r['profile_y'] = y.tolist()
        r['profile_u'] = u.tolist()
        grid_results[key] = r
        print(f"  {key}: Re_tau={r['final_retau']}")

    print("\nGenerating plots...")
    plot_uplus_overlay(results, mkm, args.nu, plot_dir)
    plot_residual_history(results, plot_dir)
    plot_grid_convergence(grid_results, mkm, args.nu, plot_dir)

    print("\nWriting summary...")
    summary = write_summary_json(results, grid_results, args.nu, mkm, args.data_dir)

    print("\nWriting validation report...")
    write_markdown_report(results, grid_results, summary, args.nu, plot_dir, args.data_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```
git add scripts/rans_validation/analyze.py
git commit -m "Add Python analysis script for RANS validation plots and report"
```

---

### Task 5: Run Validation on GPU Cluster

**Step 1: Create output directory and submit SLURM job**

```bash
mkdir -p output/rans_validation/logs
sbatch scripts/rans_validation/run_rans_validation.sbatch
```

**Step 2: Monitor job progress**

```bash
squeue -u $USER --format="%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R"
# When complete, check exit codes:
grep "exit code" output/rans_validation/logs/*.log
```

**Step 3: Verify all models produced output**

```bash
ls output/rans_validation/data/*/velocity_profile.dat
# Should see 10 files (8 models + 2 grid variants)
```

---

### Task 6: Generate Plots and Write Report

**Step 1: Run analysis script**

```bash
cd /storage/scratch1/6/sbryngelson3/cfd-nn
python3 scripts/rans_validation/analyze.py \
    --data-dir output/rans_validation \
    --mkm-dir data/reference/mkm_retau180
```

**Step 2: Verify plots generated**

```bash
ls output/rans_validation/plots/
# Expected: u_plus_all_models.png, residual_history.png, grid_convergence_sst.png
```

**Step 3: Review the report**

```bash
cat docs/RANS_VALIDATION_REPORT.md
```

**Step 4: Commit report and plots**

```
git add docs/RANS_VALIDATION_REPORT.md output/rans_validation/plots/ output/rans_validation/summary.json
git commit -m "Add RANS validation report with plots for Re_tau=180 channel flow"
```
