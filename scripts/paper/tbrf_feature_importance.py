#!/usr/bin/env python3
"""Analyze feature importance of TBRF (Tensor Basis Random Forest) model.

Loads 10 RandomForestRegressor objects (one per tensor basis coefficient g1-g10)
and visualizes which of the 5 Pope invariant features matter most.

Outputs:
  results/paper/figures/tbrf_feature_importance.pdf      -- averaged bar chart
  results/paper/figures/tbrf_feature_importance_heatmap.pdf -- per-coefficient heatmap
"""

import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

FORESTS_PKL = os.path.join(
    PROJECT_ROOT, "data", "models", "tbrf_paper", "forests.pkl"
)
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_LABELS = [
    r"$\lambda_1 = \mathrm{tr}(\hat{S}^2)$",
    r"$\lambda_2 = \mathrm{tr}(\hat{\Omega}^2)$",
    r"$\lambda_3 = \mathrm{tr}(\hat{S}^3)$",
    r"$\lambda_4 = \mathrm{tr}(\hat{\Omega}^2 \hat{S})$",
    r"$\lambda_5 = \mathrm{tr}(\hat{\Omega}^2 \hat{S}^2)$",
]

COEFF_LABELS = [f"$g_{{{i+1}}}$" for i in range(10)]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn version warn
with open(FORESTS_PKL, "rb") as fh:
    forests = pickle.load(fh)

assert len(forests) == 10, f"Expected 10 forests, got {len(forests)}"

importances = np.array([rf.feature_importances_ for rf in forests])  # (10, 5)

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print("Feature importances (rows = g_i, cols = lambda_j):\n")
header = f"{'':>6s}" + "".join(f"{'lam'+str(j+1):>10s}" for j in range(5))
print(header)
print("-" * len(header))
for i in range(10):
    row = f"{'g'+str(i+1):>6s}" + "".join(f"{importances[i,j]:10.4f}" for j in range(5))
    print(row)

avg = importances.mean(axis=0)
print("-" * len(header))
print(f"{'avg':>6s}" + "".join(f"{v:10.4f}" for v in avg))
print()

# ---------------------------------------------------------------------------
# Plot 1: Averaged bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(5)
bars = ax.bar(x, avg, color="#4C72B0", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_LABELS, fontsize=9)
ax.set_ylabel("Mean importance (across $g_1$\u2013$g_{10}$)", fontsize=10)
ax.set_title("TBRF Feature Importance (averaged)", fontsize=11)
ax.set_ylim(0, max(avg) * 1.15)

# Value labels on bars
for bar, val in zip(bars, avg):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig.tight_layout()
out1 = os.path.join(OUT_DIR, "tbrf_feature_importance.pdf")
fig.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: Heatmap (10 coefficients x 5 features)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(importances, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

ax.set_xticks(np.arange(5))
ax.set_xticklabels(FEATURE_LABELS, fontsize=9, rotation=30, ha="right")
ax.set_yticks(np.arange(10))
ax.set_yticklabels(COEFF_LABELS, fontsize=10)
ax.set_title("TBRF Feature Importance per Basis Coefficient", fontsize=11)

# Annotate cells
for i in range(10):
    for j in range(5):
        val = importances[i, j]
        color = "white" if val > 0.55 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Importance", fontsize=9)

fig.tight_layout()
out2 = os.path.join(OUT_DIR, "tbrf_feature_importance_heatmap.pdf")
fig.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close(fig)
