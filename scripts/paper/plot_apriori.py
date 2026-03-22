#!/usr/bin/env python3
"""
Generate a priori evaluation figures: scatter plots, Lumley triangle, error distributions.

Figures produced:
  - results/paper/figures/apriori_scatter.pdf    (Fig 5: predicted vs true b_ij)
  - results/paper/figures/lumley_triangle.pdf     (Fig 6: Lumley triangle)
  - results/paper/figures/error_distribution.pdf  (pointwise RMSE histograms)

Usage:
    python3 -u scripts/paper/plot_apriori.py --data_dir mcconkey_data --device cuda
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch

# Import shared plot style and data loading
_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts', 'paper'))
from plot_style import apply_style, COLORS, save_fig, SINGLE_COL, DOUBLE_COL, GOLDEN
apply_style()
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts', 'paper'))
from train_all_models import (
    load_mcconkey_csv, extract_features_and_labels,
    TEST_CASES, TBNNModel, standardize,
)
from evaluate_apriori import (
    load_nn_weights, load_tbrf,
    predict_tbnn, predict_tbrf,
    build_split_with_case_indices,
)


# ============================================================================
# Lumley triangle utilities
# ============================================================================

def bij_6_to_3x3(bij6):
    """Convert [N, 6] symmetric tensor to [N, 3, 3]."""
    n = len(bij6)
    B = np.zeros((n, 3, 3))
    B[:, 0, 0] = bij6[:, 0]
    B[:, 0, 1] = bij6[:, 1]
    B[:, 0, 2] = bij6[:, 2]
    B[:, 1, 1] = bij6[:, 3]
    B[:, 1, 2] = bij6[:, 4]
    B[:, 2, 2] = bij6[:, 5]
    # Symmetric
    B[:, 1, 0] = B[:, 0, 1]
    B[:, 2, 0] = B[:, 0, 2]
    B[:, 2, 1] = B[:, 1, 2]
    return B


def compute_lumley_invariants(bij6):
    """
    Compute Lumley triangle invariants (xi, eta) from b_ij.

    II_b = -(1/2) * tr(b^2)    (second invariant, always <= 0)
    III_b = (1/3) * tr(b^3)    (third invariant)

    eta^2 = -2 * II_b = tr(b^2) / 1
    xi^3 = (3/2) * III_b = (1/2) * tr(b^3)

    Returns (xi, eta).
    """
    B = bij_6_to_3x3(bij6)
    B2 = np.einsum('nij,njk->nik', B, B)
    B3 = np.einsum('nij,njk->nik', B2, B)

    tr_b2 = np.einsum('nii->n', B2)
    tr_b3 = np.einsum('nii->n', B3)

    # eta^2 = tr(b^2) / 3  ... actually eta^2 = (1/3)*tr(b^2)
    # No: the standard definitions are:
    #   II = b_ij b_ji / 2 = tr(b^2) / 2
    #   III = b_ij b_jk b_ki / 3 = tr(b^3) / 3
    #   eta^2 = II/3 = tr(b^2)/6   ... wait, let me use the standard.
    #
    # Standard Lumley (1978):
    #   II_a = a_ij a_ji  (where a = 2*b + (2/3)*delta ... no)
    #
    # Actually the most common convention for the Lumley triangle uses:
    #   xi^3 = (1/2) det(b) ... no.
    #
    # Let's use the clean Pope (2000) definitions:
    #   II = b_ij b_ji / 2         (>= 0, note: some refs define II with minus sign)
    #   III = b_ij b_jk b_ki / 3
    #   eta = sqrt(II / 3)
    #   xi = cbrt(III / 2)

    II = tr_b2 / 2.0   # >= 0
    III = tr_b3 / 3.0

    eta = np.sqrt(np.maximum(II / 3.0, 0.0))
    xi = np.sign(III) * np.abs(III / 2.0) ** (1.0 / 3.0)

    return xi, eta


def lumley_triangle_boundary():
    """
    Compute the boundary curves of the Lumley triangle in (xi, eta) space.

    Returns lists of (xi, eta) arrays for each boundary segment.
    """
    # The realizability limits in (xi, eta) space:
    #   1. Axisymmetric expansion (right boundary): xi = eta, for eta in [0, 1/sqrt(6)]
    #      (one large eigenvalue, two equal small ones)
    #   2. Axisymmetric contraction (left boundary): xi = -eta, for eta in [0, 1/sqrt(6)]
    #      (one small eigenvalue, two equal large ones)
    #   3. Two-component limit (top boundary): 27*xi^2 = eta^2*(eta^2 - 1/9)*... no.
    #
    # The correct 2-component limit connects (-1/6, 1/6) to (1/3, 1/3) via:
    #   6*eta^2 = 2 - 9*xi  ... no, let me derive properly.
    #
    # The anisotropy tensor eigenvalues (lambda_1, lambda_2, lambda_3) satisfy
    #   lambda_1 + lambda_2 + lambda_3 = 0   (traceless)
    #   -1/3 <= lambda_i <= 2/3
    #
    # II = (lambda_1^2 + lambda_2^2 + lambda_3^2) / 2
    # III = lambda_1 * lambda_2 * lambda_3
    # eta = sqrt(II/3), xi = (III/2)^(1/3)
    #
    # Axisymmetric: lambda_2 = lambda_3 = -lambda_1/2
    #   II = lambda_1^2 * (1 + 1/4 + 1/4) / 2 = 3*lambda_1^2/4
    #   III = lambda_1 * lambda_1^2/4 * (-1) ... let me just parametrize.
    #
    # Axisymmetric expansion: lambda_1 >= 0, lambda_2 = lambda_3 = -lambda_1/2
    #   Range: lambda_1 in [0, 2/3]
    #   II = 3*lambda_1^2/4
    #   III = -lambda_1^3/4
    #   Wait: III = lambda_1 * (-lambda_1/2)^2 = lambda_1^3/4  (positive for expansion)
    #   eta = sqrt(II/3) = sqrt(lambda_1^2/4) = lambda_1/2
    #   xi = (III/2)^(1/3) = (lambda_1^3/8)^(1/3) = lambda_1/2
    #   So xi = eta (correct!)

    # Axisymmetric contraction: lambda_3 <= 0, lambda_1 = lambda_2 = -lambda_3/2
    #   Range: lambda_3 in [-1/3, 0]
    #   II = 3*lambda_3^2/4
    #   III = (-lambda_3/2)^2 * lambda_3 = lambda_3^3/4 (negative)
    #   eta = |lambda_3|/2
    #   xi = sign(III/2) * |III/2|^(1/3) = -|lambda_3|/2
    #   So xi = -eta (correct!)

    # 2-component limit: lambda_3 = -1/3 (one eigenvalue at minimum)
    #   lambda_1 + lambda_2 = 1/3
    #   Parametrize: lambda_1 = t, lambda_2 = 1/3 - t, lambda_3 = -1/3
    #   Range: t in [-1/3, 2/3], but lambda_2 >= -1/3 => t <= 2/3,
    #          and lambda_1 >= lambda_2 => t >= 1/6
    #   Actually for full boundary: t in [-1/3, 2/3] with constraint ordering
    #   Let's just parametrize t in [-1/3, 2/3]:

    n_pts = 500

    # Right boundary: axisymmetric expansion (xi = eta)
    eta_right = np.linspace(0, 1.0 / 3.0, n_pts)
    xi_right = eta_right.copy()

    # Left boundary: axisymmetric contraction (xi = -eta)
    eta_left = np.linspace(0, 1.0 / 6.0, n_pts)
    xi_left = -eta_left.copy()

    # Top boundary: 2-component limit (lambda_3 = -1/3)
    t = np.linspace(-1.0 / 3.0, 2.0 / 3.0, n_pts)
    lam1 = t
    lam2 = 1.0 / 3.0 - t
    lam3 = -1.0 / 3.0 * np.ones_like(t)

    II_top = (lam1**2 + lam2**2 + lam3**2) / 2.0
    III_top = lam1 * lam2 * lam3

    eta_top = np.sqrt(np.maximum(II_top / 3.0, 0.0))
    xi_top = np.sign(III_top) * np.abs(III_top / 2.0) ** (1.0 / 3.0)

    return [
        (xi_right, eta_right, 'Axisymmetric expansion'),
        (xi_left, eta_left, 'Axisymmetric contraction'),
        (xi_top, eta_top, '2-component limit'),
    ]


# ============================================================================
# Plotting functions
# ============================================================================

def plot_scatter(predictions, test_bij, output_path):
    """
    Figure 5: Scatter plots of predicted vs true b_ij.

    3 rows (TBNN, PI-TBNN, TBRF) x 2 columns (b_11, b_12).
    """
    model_names = ['TBNN', 'PI-TBNN', 'TBRF']
    model_keys = ['tbnn', 'pi_tbnn', 'tbrf']
    comp_indices = [0, 1]  # b_11, b_12
    comp_labels = [r'$b_{11}$', r'$b_{12}$']

    fig, axes = plt.subplots(3, 2, figsize=(7, 9))

    for row, (mkey, mname) in enumerate(zip(model_keys, model_names)):
        if mkey not in predictions:
            continue
        b_pred = predictions[mkey]

        for col, (cidx, clabel) in enumerate(zip(comp_indices, comp_labels)):
            ax = axes[row, col]
            true_vals = test_bij[:, cidx]
            pred_vals = b_pred[:, cidx]

            rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))

            # Density scatter using 2D histogram for coloring
            h = ax.hist2d(true_vals, pred_vals, bins=80,
                          cmap='viridis', norm=LogNorm(),
                          range=[[-0.4, 0.7], [-0.4, 0.7]])

            # y = x reference line
            lims = [-0.4, 0.7]
            ax.plot(lims, lims, 'r-', linewidth=0.8, alpha=0.8)

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            ax.set_title(f'{mname} {clabel}  (RMSE={rmse:.4f})', fontsize=9)

            if col == 0:
                ax.set_ylabel('Predicted', fontsize=9)
            if row == 2:
                ax.set_xlabel('DNS (true)', fontsize=9)

            ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter plots to {output_path}")


def plot_lumley_triangle(predictions, test_bij, output_path):
    """
    Figure 6: Lumley triangle with DNS truth and model predictions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Plot boundary
    boundaries = lumley_triangle_boundary()
    for xi_b, eta_b, label in boundaries:
        ax.plot(xi_b, eta_b, 'k-', linewidth=1.5)

    # DNS truth (gray)
    xi_dns, eta_dns = compute_lumley_invariants(test_bij)
    ax.scatter(xi_dns, eta_dns, s=1, c='0.7', alpha=0.3, rasterized=True,
               label='DNS', zorder=1)

    # Model predictions
    colors = {'tbnn': 'C0', 'pi_tbnn': 'C2', 'tbrf': 'C3'}
    labels = {'tbnn': 'TBNN', 'pi_tbnn': 'PI-TBNN', 'tbrf': 'TBRF'}
    markers = {'tbnn': 'o', 'pi_tbnn': 's', 'tbrf': '^'}

    for mkey in ['tbnn', 'tbrf']:
        if mkey not in predictions:
            continue
        b_pred = predictions[mkey]
        xi_pred, eta_pred = compute_lumley_invariants(b_pred)
        ax.scatter(xi_pred, eta_pred, s=2, c=colors[mkey], alpha=0.15,
                   rasterized=True, label=labels[mkey], zorder=2)

    # Annotate special points
    ax.annotate('Isotropic', xy=(0, 0), xytext=(0.05, 0.02),
                fontsize=8, ha='left')
    ax.annotate('1-comp', xy=(1.0/3.0, 1.0/3.0), xytext=(0.22, 0.36),
                fontsize=8, ha='left')
    ax.annotate('2-comp\naxi', xy=(-1.0/6.0, 1.0/6.0), xytext=(-0.28, 0.20),
                fontsize=8, ha='left')

    ax.set_xlabel(r'$\xi$', fontsize=11)
    ax.set_ylabel(r'$\eta$', fontsize=11)
    ax.set_title('Lumley triangle (test set)', fontsize=11)

    ax.set_xlim(-0.35, 0.45)
    ax.set_ylim(-0.02, 0.42)

    ax.legend(fontsize=8, loc='upper left', markerscale=3)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Lumley triangle to {output_path}")


def plot_error_distribution(predictions, test_bij, output_path):
    """
    Histogram of pointwise RMSE for each tensor model on the test set.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = {'tbnn': 'C0', 'pi_tbnn': 'C2', 'tbrf': 'C3'}
    labels = {'tbnn': 'TBNN', 'pi_tbnn': 'PI-TBNN', 'tbrf': 'TBRF'}

    for mkey in ['tbnn', 'pi_tbnn', 'tbrf']:
        if mkey not in predictions:
            continue
        b_pred = predictions[mkey]
        # Pointwise RMSE across 6 components
        pointwise_rmse = np.sqrt(np.mean((b_pred - test_bij) ** 2, axis=1))

        ax.hist(pointwise_rmse, bins=80, range=(0, 0.3), alpha=0.5,
                color=colors[mkey], label=labels[mkey], density=True)

        median_err = np.median(pointwise_rmse)
        ax.axvline(median_err, color=colors[mkey], linestyle='--',
                   linewidth=1.0, alpha=0.8)

    ax.set_xlabel('Pointwise RMSE', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Error distribution (test set)', fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved error distribution to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate a priori evaluation figures')
    parser.add_argument('--data_dir', default='mcconkey_data',
                        help='Path to McConkey dataset')
    parser.add_argument('--model_dir', default='data/models',
                        help='Directory containing trained model weights')
    parser.add_argument('--output_dir', default='results/paper/figures',
                        help='Output directory for figures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--rans_model', default='komegasst',
                        help='RANS baseline model in dataset')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("  A Priori Figures: Scatter, Lumley Triangle, Error Distribution")
    print("=" * 70)
    print(f"Data:       {args.data_dir}")
    print(f"Models dir: {args.model_dir}")
    print(f"Device:     {args.device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset and build test split
    # ------------------------------------------------------------------
    by_case = load_mcconkey_csv(args.data_dir, args.rans_model)

    print("\n--- Building test split ---")
    test_data = build_split_with_case_indices(
        by_case, TEST_CASES, args.rans_model, args.device)
    inv_test, bij_test, basis_test, k_test, case_indices = test_data
    print(f"Test set: {len(inv_test):,} points")

    # ------------------------------------------------------------------
    # Load models and compute predictions
    # ------------------------------------------------------------------
    predictions = {}

    # TBNN
    tbnn_path = Path(args.model_dir) / 'tbnn_paper'
    if tbnn_path.exists():
        print("\nLoading TBNN...")
        model, inv_mean, inv_std = load_nn_weights(
            tbnn_path, TBNNModel, n_in=5, hidden=[64, 64, 64], n_basis=10)
        predictions['tbnn'] = predict_tbnn(
            model, inv_test, basis_test, inv_mean, inv_std, args.device)
        rmse = np.sqrt(np.mean((predictions['tbnn'] - bij_test) ** 2))
        print(f"  TBNN test RMSE: {rmse:.6f}")
    else:
        print(f"  SKIP: {tbnn_path} not found")

    # PI-TBNN
    pi_path = Path(args.model_dir) / 'pi_tbnn_paper'
    if pi_path.exists():
        print("Loading PI-TBNN...")
        model, inv_mean, inv_std = load_nn_weights(
            pi_path, TBNNModel, n_in=5, hidden=[64, 64, 64], n_basis=10)
        predictions['pi_tbnn'] = predict_tbnn(
            model, inv_test, basis_test, inv_mean, inv_std, args.device)
        rmse = np.sqrt(np.mean((predictions['pi_tbnn'] - bij_test) ** 2))
        print(f"  PI-TBNN test RMSE: {rmse:.6f}")
    else:
        print(f"  SKIP: {pi_path} not found")

    # TBRF
    tbrf_path = Path(args.model_dir) / 'tbrf_paper'
    if tbrf_path.exists():
        print("Loading TBRF...")
        forests, inv_mean, inv_std = load_tbrf(tbrf_path)
        predictions['tbrf'] = predict_tbrf(
            forests, inv_test, basis_test, inv_mean, inv_std)
        rmse = np.sqrt(np.mean((predictions['tbrf'] - bij_test) ** 2))
        print(f"  TBRF test RMSE: {rmse:.6f}")
    else:
        print(f"  SKIP: {tbrf_path} not found")

    if not predictions:
        print("\nERROR: No models found. Cannot generate figures.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print(f"\n--- Generating figures ---")

    # Figure 5: Scatter plots
    plot_scatter(predictions, bij_test,
                 output_dir / 'apriori_scatter.pdf')

    # Figure 6: Lumley triangle
    plot_lumley_triangle(predictions, bij_test,
                         output_dir / 'lumley_triangle.pdf')

    # Error distribution
    plot_error_distribution(predictions, bij_test,
                            output_dir / 'error_distribution.pdf')

    print("\nDone.")


if __name__ == '__main__':
    main()
