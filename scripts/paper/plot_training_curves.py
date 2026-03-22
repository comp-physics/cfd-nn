#!/usr/bin/env python3
"""Parse training logs and generate training curve plots for the paper.

Usage:
    python3 scripts/paper/plot_training_curves.py

Reads:
    results/paper/training/full_5266424.out
    results/paper/training/pi_sweep_5300440.out

Writes:
    results/paper/figures/training_curves.pdf
    results/paper/figures/pi_tbnn_sweep.pdf
    results/paper/figures/lr_schedule.pdf
    results/paper/figures/training_data.json
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
FULL_LOG = REPO / "results" / "paper" / "training" / "full_5266424.out"
SWEEP_LOG = REPO / "results" / "paper" / "training" / "pi_sweep_5300440.out"
OUT_DIR = REPO / "results" / "paper" / "figures"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Header patterns
_HEADER_MLP = re.compile(
    r"Training\s+(MLP(?:-Large)?)\s+\((.+?)\)"
)
_HEADER_TBNN = re.compile(
    r"Training\s+((?:PI-)?TBNN)\s+\((.+?)\)"
)
_HEADER_TBRF = re.compile(
    r"Training\s+(TBRF)\s+\((.+?)\)"
)
_HEADER_PI_SWEEP = re.compile(
    r"Training\s+PI-TBNN\s+\(beta=([\d.]+)\)"
)

# Epoch patterns
# MLP format: Epoch  100/1000: val_RMSE=0.121697, best=0.120677, lr=5.01e-04
_EPOCH_MLP = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+val_RMSE=([\d.]+),\s+best=([\d.]+),\s+lr=([\d.eE+-]+)"
)
# TBNN format: Epoch  100/1000: train=0.009381, val=0.008547, best=0.008364, lr=5.01e-04
# PI-TBNN:     Epoch  100/1000: train=..., val=..., best=..., lr=..., beta=...
_EPOCH_TBNN = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+train=([\d.]+),\s+val=([\d.]+),\s+best=([\d.]+),\s+lr=([\d.eE+-]+)"
    r"(?:,\s+beta=([\d.]+))?"
)

_EARLY_STOP = re.compile(r"Early stopping at epoch (\d+)")
_FINAL_RMSE = re.compile(r"Final val RMSE(?:\(b\))?:\s+([\d.]+)")


def parse_log(path):
    """Parse a training log file, returning a list of model records.

    Each record is a dict:
        name:        str   (e.g. "MLP", "PI-TBNN (beta=0.001)")
        arch:        str   (e.g. "5->32->32->1")
        epochs:      list[int]
        val_loss:    list[float]   (val_RMSE for MLP, val MSE for TBNN)
        best_loss:   list[float]
        train_loss:  list[float] or None
        lr:          list[float]
        beta:        list[float] or None
        early_stop:  int or None
        final_rmse:  float or None
    """
    records = []
    current = None

    with open(path) as f:
        for line in f:
            # Check for model headers
            m = _HEADER_PI_SWEEP.search(line)
            if m:
                beta_val = m.group(1)
                current = _new_record(f"PI-TBNN (beta={beta_val})", f"beta={beta_val}")
                records.append(current)
                continue

            m = _HEADER_TBNN.search(line)
            if m and "PI-" in m.group(1):
                current = _new_record("PI-TBNN", m.group(2))
                records.append(current)
                continue
            elif m:
                current = _new_record("TBNN", m.group(2))
                records.append(current)
                continue

            m = _HEADER_MLP.search(line)
            if m:
                current = _new_record(m.group(1), m.group(2))
                records.append(current)
                continue

            m = _HEADER_TBRF.search(line)
            if m:
                current = _new_record("TBRF", m.group(2))
                records.append(current)
                continue

            if current is None:
                continue

            # Epoch lines — try TBNN format first (superset)
            m = _EPOCH_TBNN.match(line.strip())
            if m:
                current["epochs"].append(int(m.group(1)))
                current["train_loss"].append(float(m.group(3)))
                current["val_loss"].append(float(m.group(4)))
                current["best_loss"].append(float(m.group(5)))
                current["lr"].append(float(m.group(6)))
                if m.group(7) is not None:
                    current["beta"].append(float(m.group(7)))
                continue

            m = _EPOCH_MLP.match(line.strip())
            if m:
                current["epochs"].append(int(m.group(1)))
                current["val_loss"].append(float(m.group(3)))
                current["best_loss"].append(float(m.group(4)))
                current["lr"].append(float(m.group(5)))
                continue

            m = _EARLY_STOP.search(line)
            if m:
                current["early_stop"] = int(m.group(1))
                continue

            m = _FINAL_RMSE.search(line)
            if m:
                current["final_rmse"] = float(m.group(1))
                continue

    return records


def _new_record(name, arch):
    return {
        "name": name,
        "arch": arch,
        "epochs": [],
        "val_loss": [],
        "best_loss": [],
        "train_loss": [],
        "lr": [],
        "beta": [],
        "early_stop": None,
        "final_rmse": None,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


COLORS = {
    "MLP": "#1f77b4",
    "MLP-Large": "#ff7f0e",
    "TBNN": "#2ca02c",
    "PI-TBNN": "#d62728",
    "TBRF": "#9467bd",
}

SWEEP_COLORS = {
    "0": "#2ca02c",       # TBNN (beta=0)
    "0.001": "#1f77b4",
    "0.01": "#d62728",
    "0.1": "#ff7f0e",
}


def _find_record(records, name):
    for r in records:
        if r["name"] == name:
            return r
    return None


# ---------------------------------------------------------------------------
# Figure A: All models validation loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(full_records):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    tbrf = _find_record(full_records, "TBRF")
    tbrf_rmse = tbrf["final_rmse"] if tbrf else 0.0637

    model_order = ["MLP", "MLP-Large", "TBNN", "PI-TBNN"]
    for name in model_order:
        rec = _find_record(full_records, name)
        if rec is None or len(rec["epochs"]) == 0:
            continue

        epochs = np.array(rec["epochs"])
        val = np.array(rec["val_loss"])

        # For TBNN/PI-TBNN, val_loss is MSE; convert to RMSE for comparison
        # Actually the log stores raw val metric. MLP stores val_RMSE directly.
        # TBNN stores val (MSE of coefficients). These are different metrics,
        # so we plot them all on the same y-axis as "validation loss".
        # But the user asked for val RMSE. For MLP it's RMSE already.
        # For TBNN, the "val" column is MSE of tensor coefficients.
        # Let's just plot what we have and label accordingly.

        label = f"{name} (final={rec['final_rmse']:.4f})" if rec["final_rmse"] else name
        color = COLORS.get(name, "gray")
        ax.plot(epochs, val, linewidth=1.0, color=color, label=label)

        # Mark early stopping
        if rec["early_stop"] is not None:
            es_epoch = rec["early_stop"]
            # Find closest logged epoch
            idx = np.argmin(np.abs(epochs - es_epoch))
            ax.plot(epochs[idx], val[idx], "x", color=color, markersize=7,
                    markeredgewidth=1.5, zorder=5)

    # TBRF horizontal line
    ax.axhline(tbrf_rmse, color=COLORS["TBRF"], linestyle="--", linewidth=0.9,
               label=f"TBRF (RMSE(b)={tbrf_rmse:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure B: PI-TBNN beta comparison
# ---------------------------------------------------------------------------

def plot_pi_sweep(full_records, sweep_records):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # TBNN from main log (beta=0 reference)
    tbnn = _find_record(full_records, "TBNN")
    if tbnn and len(tbnn["epochs"]) > 0:
        ax.plot(tbnn["epochs"], tbnn["val_loss"], linewidth=1.0,
                color=SWEEP_COLORS["0"],
                label=r"TBNN ($\beta$=0)")

    # Sweep models
    for rec in sweep_records:
        if len(rec["epochs"]) == 0:
            continue
        # Extract beta value from name
        m = re.search(r"beta=([\d.]+)", rec["name"])
        beta_str = m.group(1) if m else "?"
        color = SWEEP_COLORS.get(beta_str, "gray")
        label = rf"PI-TBNN ($\beta$={beta_str})"
        if rec["final_rmse"] is not None:
            label += f" (final={rec['final_rmse']:.4f})"
        ax.plot(rec["epochs"], rec["val_loss"], linewidth=1.0,
                color=color, label=label)

        # Mark early stopping
        if rec["early_stop"] is not None:
            es = rec["early_stop"]
            idx = np.argmin(np.abs(np.array(rec["epochs"]) - es))
            ax.plot(rec["epochs"][idx], rec["val_loss"][idx], "x",
                    color=color, markersize=7, markeredgewidth=1.5, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure C: Learning rate schedule
# ---------------------------------------------------------------------------

def plot_lr_schedule(full_records):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Show LR for MLP (ran full 1000 epochs, clearest schedule)
    mlp = _find_record(full_records, "MLP")
    if mlp and len(mlp["epochs"]) > 0:
        ax.plot(mlp["epochs"], mlp["lr"], linewidth=0.9,
                color=COLORS["MLP"], label="MLP (1000 epochs)")

    # Also show TBNN (stopped at 694)
    tbnn = _find_record(full_records, "TBNN")
    if tbnn and len(tbnn["epochs"]) > 0:
        ax.plot(tbnn["epochs"], tbnn["lr"], linewidth=0.9,
                color=COLORS["TBNN"], linestyle="--", label="TBNN (694 epochs)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(full_records, sweep_records, path):
    data = {
        "full_training": [],
        "pi_sweep": [],
    }
    for rec in full_records:
        data["full_training"].append(_record_to_dict(rec))
    for rec in sweep_records:
        data["pi_sweep"].append(_record_to_dict(rec))

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _record_to_dict(rec):
    d = dict(rec)
    # Convert lists to plain Python types for JSON
    for key in ("epochs", "val_loss", "best_loss", "train_loss", "lr", "beta"):
        if d[key]:
            d[key] = [float(x) if isinstance(x, (float, np.floating)) else x
                      for x in d[key]]
        else:
            d[key] = []
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _setup_style()

    if not FULL_LOG.exists():
        print(f"ERROR: {FULL_LOG} not found", file=sys.stderr)
        sys.exit(1)
    if not SWEEP_LOG.exists():
        print(f"ERROR: {SWEEP_LOG} not found", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {FULL_LOG.name} ...")
    full_records = parse_log(FULL_LOG)
    print(f"  Found {len(full_records)} model(s): "
          + ", ".join(r['name'] for r in full_records))

    print(f"Parsing {SWEEP_LOG.name} ...")
    sweep_records = parse_log(SWEEP_LOG)
    print(f"  Found {len(sweep_records)} model(s): "
          + ", ".join(r['name'] for r in sweep_records))

    # Figure A
    fig_a = plot_training_curves(full_records)
    out_a = OUT_DIR / "training_curves.pdf"
    fig_a.savefig(out_a)
    print(f"Saved {out_a}")
    plt.close(fig_a)

    # Figure B
    fig_b = plot_pi_sweep(full_records, sweep_records)
    out_b = OUT_DIR / "pi_tbnn_sweep.pdf"
    fig_b.savefig(out_b)
    print(f"Saved {out_b}")
    plt.close(fig_b)

    # Figure C
    fig_c = plot_lr_schedule(full_records)
    out_c = OUT_DIR / "lr_schedule.pdf"
    fig_c.savefig(out_c)
    print(f"Saved {out_c}")
    plt.close(fig_c)

    # JSON
    json_path = OUT_DIR / "training_data.json"
    export_json(full_records, sweep_records, json_path)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
