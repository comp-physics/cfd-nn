#!/usr/bin/env python3
"""
A priori evaluation of all trained NN turbulence models on validation and test sets.

Loads trained models from data/models/{mlp,mlp_large,tbnn,pi_tbnn,tbrf}_paper/
and evaluates on the McConkey et al. (2021) dataset using the TBKAN 2025
case-holdout protocol.

Metrics computed:
  Tensor models (TBNN, PI-TBNN, TBRF):
    - Overall RMSE(b) across all 6 components
    - Component-wise RMSE: b_11, b_12, b_13, b_22, b_23, b_33
    - Realizability violation rate (% points with b_ii < -1/3 or b_ii > 2/3)
  Scalar models (MLP, MLP-Large):
    - RMSE of anisotropy magnitude |b|
  Test set also gets per-case RMSE (case_1p2 vs cbfs13700).

Usage:
    python3 -u scripts/paper/evaluate_apriori.py --data_dir mcconkey_data --device cuda
"""

import os
import sys
import argparse
import json
import pickle
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# Import shared data loading and model definitions from training script
# Use os.path.realpath to resolve symlinks, and also try CWD-relative path
_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts', 'paper'))
from train_all_models import (
    load_mcconkey_csv, build_split, extract_features_and_labels,
    TRAIN_CASES, VAL_CASES, TEST_CASES,
    MLPModel, TBNNModel, standardize,
)


# ============================================================================
# Constants
# ============================================================================

COMP_NAMES = ['b_11', 'b_12', 'b_13', 'b_22', 'b_23', 'b_33']
DIAG_INDICES = [0, 3, 5]  # b_11, b_22, b_33 in 6-component layout


# ============================================================================
# Model loading
# ============================================================================

def load_nn_weights(model_dir, model_class, **kwargs):
    """Load PyTorch model weights from exported text files (layer0_W.txt etc)."""
    model_dir = Path(model_dir)
    model = model_class(**kwargs)
    state = {}
    layer = 0
    while os.path.exists(model_dir / f'layer{layer}_W.txt'):
        W = np.loadtxt(model_dir / f'layer{layer}_W.txt')
        b = np.loadtxt(model_dir / f'layer{layer}_b.txt')
        # Handle 1D arrays (single output neuron)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        if b.ndim == 0:
            b = b.reshape(1)
        state[f'net.{layer * 2}.weight'] = torch.FloatTensor(W)
        state[f'net.{layer * 2}.bias'] = torch.FloatTensor(b)
        layer += 1
    model.load_state_dict(state)
    model.eval()
    inv_mean = np.loadtxt(model_dir / 'input_means.txt')
    inv_std = np.loadtxt(model_dir / 'input_stds.txt')
    return model, inv_mean, inv_std


def load_tbrf(model_dir):
    """Load TBRF model from pickle + normalization."""
    model_dir = Path(model_dir)
    with open(model_dir / 'forests.pkl', 'rb') as f:
        forests = pickle.load(f)
    inv_mean = np.loadtxt(model_dir / 'input_means.txt')
    inv_std = np.loadtxt(model_dir / 'input_stds.txt')
    return forests, inv_mean, inv_std


# ============================================================================
# Prediction
# ============================================================================

def predict_tbnn(model, invariants, basis, inv_mean, inv_std, device='cpu'):
    """TBNN/PI-TBNN: invariants -> g_n coefficients -> b_ij = sum g_n T^(n)_ij."""
    inv_norm, _, _ = standardize(invariants, inv_mean, inv_std)
    inv_t = torch.FloatTensor(inv_norm).to(device)
    basis_t = torch.FloatTensor(basis).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        g = model(inv_t)  # [N, n_basis]
        b_pred = (g.unsqueeze(2) * basis_t).sum(dim=1)  # [N, 6]
    return b_pred.cpu().numpy()


def predict_mlp(model, invariants, inv_mean, inv_std, device='cpu'):
    """MLP: invariants -> scalar |b| magnitude."""
    inv_norm, _, _ = standardize(invariants, inv_mean, inv_std)
    inv_t = torch.FloatTensor(inv_norm).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(inv_t)
    return pred.cpu().numpy()


def predict_tbrf(forests, invariants, basis, inv_mean, inv_std):
    """TBRF: RF predicts g_n coefficients, then b_ij = sum g_n T^(n)_ij."""
    inv_norm, _, _ = standardize(invariants, inv_mean, inv_std)
    n_basis = len(forests)
    g_pred = np.zeros((len(invariants), n_basis))
    for n in range(n_basis):
        g_pred[:, n] = forests[n].predict(inv_norm)
    b_pred = np.einsum('nb,nbc->nc', g_pred, basis)
    return b_pred


# ============================================================================
# Metrics
# ============================================================================

def compute_tensor_metrics(b_pred, b_true):
    """
    Compute all metrics for tensor-valued (b_ij) predictions.
    Returns dict with overall RMSE, component-wise RMSE, realizability violations.
    """
    metrics = OrderedDict()
    metrics['n_points'] = int(len(b_pred))

    # Overall RMSE
    metrics['rmse'] = float(np.sqrt(np.mean((b_pred - b_true) ** 2)))

    # Component-wise RMSE
    comp_rmse = OrderedDict()
    for i, name in enumerate(COMP_NAMES):
        comp_rmse[name] = float(np.sqrt(np.mean((b_pred[:, i] - b_true[:, i]) ** 2)))
    metrics['component_rmse'] = comp_rmse

    # Realizability violations: b_ii < -1/3 or b_ii > 2/3
    n_points = len(b_pred)
    violated = np.zeros(n_points, dtype=bool)
    for idx in DIAG_INDICES:
        violated |= (b_pred[:, idx] < -1.0 / 3.0)
        violated |= (b_pred[:, idx] > 2.0 / 3.0)
    metrics['realizability_violation_rate'] = float(np.sum(violated) / n_points)
    metrics['realizability_violations_count'] = int(np.sum(violated))

    return metrics


def compute_scalar_metrics(pred, true):
    """Compute metrics for scalar (|b| magnitude) predictions."""
    return OrderedDict([
        ('n_points', int(len(pred))),
        ('rmse', float(np.sqrt(np.mean((pred - true) ** 2)))),
    ])


# ============================================================================
# Per-case data building
# ============================================================================

def build_split_with_case_indices(by_case, cases, rans_model='komegasst', device='cpu'):
    """
    Build feature/label arrays with per-case index tracking.
    Returns (inv, bij, basis, k, case_indices) where
    case_indices maps case_name -> (start_idx, end_idx).
    """
    all_rows = []
    case_indices = OrderedDict()
    offset = 0

    for case in cases:
        if case in by_case:
            rows = by_case[case]
            case_indices[case] = (offset, offset + len(rows))
            all_rows.extend(rows)
            offset += len(rows)
        else:
            print(f"  WARNING: Case '{case}' not found in dataset")

    if not all_rows:
        return None, None, None, None, None

    print(f"  Extracting features from {len(all_rows):,} points ({len(cases)} cases)...")
    inv, bij, basis, k = extract_features_and_labels(all_rows, rans_model, device=device)
    return inv, bij, basis, k, case_indices


# ============================================================================
# Evaluation drivers
# ============================================================================

def eval_tensor_model_on_splits(name, predict_fn, val_data, test_data):
    """
    Evaluate a tensor model on val and test splits.
    predict_fn(invariants, basis) -> b_pred [N, 6]
    """
    results = OrderedDict()

    # Validation
    inv, bij, basis, k, _ = val_data
    if inv is not None:
        b_pred = predict_fn(inv, basis)
        results['val'] = compute_tensor_metrics(b_pred, bij)

    # Test (overall + per-case)
    inv, bij, basis, k, case_indices = test_data
    if inv is not None:
        b_pred = predict_fn(inv, basis)
        results['test'] = compute_tensor_metrics(b_pred, bij)

        # Per-case breakdown
        if case_indices:
            per_case = OrderedDict()
            for case_name, (start, end) in case_indices.items():
                per_case[case_name] = compute_tensor_metrics(
                    b_pred[start:end], bij[start:end])
            results['test']['per_case'] = per_case

    return results


def eval_scalar_model_on_splits(name, predict_fn, val_data, test_data):
    """
    Evaluate a scalar model on val and test splits.
    predict_fn(invariants) -> pred [N, 1]
    Target is |b| = sqrt(sum b_ij^2).
    """
    results = OrderedDict()

    # Validation
    inv, bij, basis, k, _ = val_data
    if inv is not None:
        target = np.sqrt((bij ** 2).sum(axis=1, keepdims=True))
        pred = predict_fn(inv)
        results['val'] = compute_scalar_metrics(pred, target)

    # Test (overall + per-case)
    inv, bij, basis, k, case_indices = test_data
    if inv is not None:
        target = np.sqrt((bij ** 2).sum(axis=1, keepdims=True))
        pred = predict_fn(inv)
        results['test'] = compute_scalar_metrics(pred, target)

        if case_indices:
            per_case = OrderedDict()
            for case_name, (start, end) in case_indices.items():
                per_case[case_name] = compute_scalar_metrics(
                    pred[start:end], target[start:end])
            results['test']['per_case'] = per_case

    return results


# ============================================================================
# Summary printing
# ============================================================================

def print_summary(all_results):
    """Print formatted summary tables."""

    # ------ Tensor models ------
    tensor_names = [n for n in ['tbnn', 'pi_tbnn', 'tbrf'] if n in all_results]
    if tensor_names:
        print()
        print("=" * 100)
        print("  TENSOR MODELS: b_ij prediction (TBNN, PI-TBNN, TBRF)")
        print("=" * 100)

        hdr = (f"  {'Model':<12} {'Split':<7} {'RMSE(b)':<10} "
               f"{'b_11':<9} {'b_12':<9} {'b_13':<9} "
               f"{'b_22':<9} {'b_23':<9} {'b_33':<9} {'Viol%':<7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for model_name in tensor_names:
            res = all_results[model_name]
            for split in ['val', 'test']:
                if split not in res:
                    continue
                m = res[split]
                cr = m['component_rmse']
                viol = m['realizability_violation_rate'] * 100
                print(f"  {model_name:<12} {split:<7} {m['rmse']:<10.6f} "
                      f"{cr['b_11']:<9.6f} {cr['b_12']:<9.6f} {cr['b_13']:<9.6f} "
                      f"{cr['b_22']:<9.6f} {cr['b_23']:<9.6f} {cr['b_33']:<9.6f} "
                      f"{viol:<7.2f}")

        # Per-case test breakdown
        print()
        print(f"  {'Model':<12} {'Test Case':<16} {'RMSE(b)':<10} {'Viol%':<7} {'N':>8}")
        print("  " + "-" * 55)
        for model_name in tensor_names:
            res = all_results[model_name]
            if 'test' in res and 'per_case' in res['test']:
                for case_name, cm in res['test']['per_case'].items():
                    viol = cm['realizability_violation_rate'] * 100
                    print(f"  {model_name:<12} {case_name:<16} {cm['rmse']:<10.6f} "
                          f"{viol:<7.2f} {cm['n_points']:>8}")

    # ------ Scalar models ------
    scalar_names = [n for n in ['mlp', 'mlp_large'] if n in all_results]
    if scalar_names:
        print()
        print("=" * 60)
        print("  SCALAR MODELS: |b| magnitude prediction (MLP, MLP-Large)")
        print("=" * 60)
        print(f"  {'Model':<12} {'Split':<7} {'RMSE(|b|)':<12} {'N':>8}")
        print("  " + "-" * 42)

        for model_name in scalar_names:
            res = all_results[model_name]
            for split in ['val', 'test']:
                if split not in res:
                    continue
                m = res[split]
                print(f"  {model_name:<12} {split:<7} {m['rmse']:<12.6f} {m['n_points']:>8}")

        # Per-case test breakdown
        print()
        print(f"  {'Model':<12} {'Test Case':<16} {'RMSE(|b|)':<12} {'N':>8}")
        print("  " + "-" * 50)
        for model_name in scalar_names:
            res = all_results[model_name]
            if 'test' in res and 'per_case' in res['test']:
                for case_name, cm in res['test']['per_case'].items():
                    print(f"  {model_name:<12} {case_name:<16} {cm['rmse']:<12.6f} "
                          f"{cm['n_points']:>8}")

    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='A priori evaluation of NN turbulence models on val/test sets')
    parser.add_argument('--data_dir', default='mcconkey_data',
                        help='Path to McConkey dataset')
    parser.add_argument('--model_dir', default='data/models',
                        help='Directory containing trained model weights')
    parser.add_argument('--output_dir', default='results/paper/apriori',
                        help='Output directory for metrics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--rans_model', default='komegasst',
                        help='RANS baseline model in dataset')
    parser.add_argument('--models', nargs='*',
                        default=['mlp', 'mlp_med', 'mlp_large', 'tbnn', 'tbnn_small', 'tbnn_large', 'pi_tbnn', 'pi_tbnn_small', 'pi_tbnn_large', 'tbrf'],
                        help='Models to evaluate')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("  A Priori Evaluation of NN Turbulence Models")
    print("=" * 70)
    print(f"Data:       {args.data_dir}")
    print(f"Models dir: {args.model_dir}")
    print(f"Device:     {args.device}")
    print(f"Evaluating: {', '.join(args.models)}")
    print()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    by_case = load_mcconkey_csv(args.data_dir, args.rans_model)

    # ------------------------------------------------------------------
    # Build val and test splits with per-case index tracking
    # ------------------------------------------------------------------
    print("\n--- Building validation split ---")
    val_data = build_split_with_case_indices(
        by_case, VAL_CASES, args.rans_model, args.device)

    print("\n--- Building test split ---")
    test_data = build_split_with_case_indices(
        by_case, TEST_CASES, args.rans_model, args.device)

    inv_val = val_data[0]
    inv_test = test_data[0]
    print(f"\nSplit sizes: val={len(inv_val):,}, test={len(inv_test):,}")
    if test_data[4]:
        for case_name, (start, end) in test_data[4].items():
            print(f"  test/{case_name}: {end - start:,} points")

    all_results = OrderedDict()

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------

    # ---- TBNN ----
    if 'tbnn' in args.models:
        model_path = Path(args.model_dir) / 'tbnn_paper'
        if model_path.exists():
            print(f"\n--- Evaluating TBNN (5->64->64->64->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[64, 64, 64], n_basis=10)
            # Use default args to capture current values (avoid late-binding bug)
            def _pred_tbnn(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['tbnn'] = eval_tensor_model_on_splits(
                'tbnn', _pred_tbnn, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['tbnn']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['tbnn']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- TBNN-Small ----
    if 'tbnn_small' in args.models:
        model_path = Path(args.model_dir) / 'tbnn_small_paper'
        if model_path.exists():
            print(f"\n--- Evaluating TBNN-Small (5->32->32->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[32, 32], n_basis=10)
            def _pred_tbnn_sm(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['tbnn_small'] = eval_tensor_model_on_splits(
                'tbnn_small', _pred_tbnn_sm, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['tbnn_small']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['tbnn_small']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- TBNN-Large ----
    if 'tbnn_large' in args.models:
        model_path = Path(args.model_dir) / 'tbnn_large_paper'
        if model_path.exists():
            print(f"\n--- Evaluating TBNN-Large (5->128->128->128->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[128, 128, 128], n_basis=10)
            def _pred_tbnn_lg(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['tbnn_large'] = eval_tensor_model_on_splits(
                'tbnn_large', _pred_tbnn_lg, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['tbnn_large']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['tbnn_large']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- PI-TBNN ----
    if 'pi_tbnn' in args.models:
        model_path = Path(args.model_dir) / 'pi_tbnn_paper'
        if model_path.exists():
            print(f"\n--- Evaluating PI-TBNN (5->64->64->64->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[64, 64, 64], n_basis=10)
            def _pred_pi(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['pi_tbnn'] = eval_tensor_model_on_splits(
                'pi_tbnn', _pred_pi, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['pi_tbnn']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['pi_tbnn']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- PI-TBNN-Small ----
    if 'pi_tbnn_small' in args.models:
        model_path = Path(args.model_dir) / 'pi_tbnn_small_paper'
        if model_path.exists():
            print(f"\n--- Evaluating PI-TBNN-Small (5->32->32->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[32, 32], n_basis=10)
            def _pred_pi_sm(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['pi_tbnn_small'] = eval_tensor_model_on_splits(
                'pi_tbnn_small', _pred_pi_sm, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['pi_tbnn_small']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['pi_tbnn_small']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- PI-TBNN-Large ----
    if 'pi_tbnn_large' in args.models:
        model_path = Path(args.model_dir) / 'pi_tbnn_large_paper'
        if model_path.exists():
            print(f"\n--- Evaluating PI-TBNN-Large (5->128->128->128->10) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, TBNNModel, n_in=5, hidden=[128, 128, 128], n_basis=10)
            def _pred_pi_lg(inv, basis, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_tbnn(_m, inv, basis, _mean, _std, args.device)
            all_results['pi_tbnn_large'] = eval_tensor_model_on_splits(
                'pi_tbnn_large', _pred_pi_lg, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['pi_tbnn_large']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['pi_tbnn_large']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- TBRF ----
    if 'tbrf' in args.models:
        model_path = Path(args.model_dir) / 'tbrf_paper'
        if model_path.exists():
            print(f"\n--- Evaluating TBRF (200 trees, depth 20) ---")
            forests, inv_mean, inv_std = load_tbrf(model_path)
            def _pred_tbrf(inv, basis, _f=forests, _mean=inv_mean, _std=inv_std):
                return predict_tbrf(_f, inv, basis, _mean, _std)
            all_results['tbrf'] = eval_tensor_model_on_splits(
                'tbrf', _pred_tbrf, val_data, test_data)
            print(f"  Val  RMSE(b): {all_results['tbrf']['val']['rmse']:.6f}")
            print(f"  Test RMSE(b): {all_results['tbrf']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- MLP (small) ----
    if 'mlp' in args.models:
        model_path = Path(args.model_dir) / 'mlp_paper'
        if model_path.exists():
            print(f"\n--- Evaluating MLP (5->32->32->1) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, MLPModel, n_in=5, hidden=[32, 32], n_out=1)
            def _pred_mlp(inv, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_mlp(_m, inv, _mean, _std, args.device)
            all_results['mlp'] = eval_scalar_model_on_splits(
                'mlp', _pred_mlp, val_data, test_data)
            print(f"  Val  RMSE(|b|): {all_results['mlp']['val']['rmse']:.6f}")
            print(f"  Test RMSE(|b|): {all_results['mlp']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- MLP-Medium ----
    if 'mlp_med' in args.models:
        model_path = Path(args.model_dir) / 'mlp_med_paper'
        if model_path.exists():
            print(f"\n--- Evaluating MLP-Medium (5->64->64->1) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, MLPModel, n_in=5, hidden=[64, 64], n_out=1)
            def _pred_mlp_md(inv, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_mlp(_m, inv, _mean, _std, args.device)
            all_results['mlp_med'] = eval_scalar_model_on_splits(
                'mlp_med', _pred_mlp_md, val_data, test_data)
            print(f"  Val  RMSE(|b|): {all_results['mlp_med']['val']['rmse']:.6f}")
            print(f"  Test RMSE(|b|): {all_results['mlp_med']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ---- MLP-Large ----
    if 'mlp_large' in args.models:
        model_path = Path(args.model_dir) / 'mlp_large_paper'
        if model_path.exists():
            print(f"\n--- Evaluating MLP-Large (5->128x4->1) ---")
            model, inv_mean, inv_std = load_nn_weights(
                model_path, MLPModel, n_in=5, hidden=[128, 128, 128, 128], n_out=1)
            def _pred_mlp_lg(inv, _m=model, _mean=inv_mean, _std=inv_std):
                return predict_mlp(_m, inv, _mean, _std, args.device)
            all_results['mlp_large'] = eval_scalar_model_on_splits(
                'mlp_large', _pred_mlp_lg, val_data, test_data)
            print(f"  Val  RMSE(|b|): {all_results['mlp_large']['val']['rmse']:.6f}")
            print(f"  Test RMSE(|b|): {all_results['mlp_large']['test']['rmse']:.6f}")
        else:
            print(f"\n  SKIP: {model_path} not found")

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    print_summary(all_results)

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'metrics.json'

    output = OrderedDict()
    output['metadata'] = OrderedDict([
        ('dataset', 'McConkey et al. (2021)'),
        ('rans_model', args.rans_model),
        ('split_protocol', 'TBKAN 2025 case-holdout'),
        ('val_cases', VAL_CASES),
        ('test_cases', TEST_CASES),
        ('device', args.device),
        ('eval_time_s', round(time.time() - t_start, 1)),
    ])
    output['results'] = all_results

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Total evaluation time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
