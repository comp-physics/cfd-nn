#!/usr/bin/env python3
"""
Unified training pipeline for all NN turbulence models.

Trains 5 model architectures on the McConkey et al. (2021) dataset
with a standardized case-holdout split (matching TBKAN 2025 protocol).

Models:
  1. MLP       — scalar nu_t, 3-layer (5→32→32→1)
  2. MLP-Large — scalar nu_t, 5-layer (5→128→128→128→128→1)
  3. TBNN      — anisotropy tensor, 4-layer (5→64→64→64→4)
  4. PI-TBNN   — TBNN + realizability loss (Lumley triangle)
  5. TBRF      — Random forest + tensor basis (scikit-learn)

Train/Val/Test split (TBKAN 2025 protocol):
  Train: SD Re={1100-1600, 2205-3500} excl. 2000, PH alpha={0.5,1.0,1.5}, CDC Re=12600
  Val:   SD Re=2000, PH alpha=0.8
  Test:  PH alpha=1.2, CBFS Re=13700

Usage:
    python train_all_models.py --data_dir mcconkey_data --output_dir data/models
"""

import os
import sys
import argparse
import csv
import json
import time
import math
from pathlib import Path
from collections import defaultdict

# Defer heavy imports to allow --help without deps
def import_deps():
    global np, torch, nn, optim, DataLoader, TensorDataset, RandomForestRegressor
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.ensemble import RandomForestRegressor


# ============================================================================
# Data loading
# ============================================================================

# Case-holdout split following TBKAN (2025)
TRAIN_CASES = [
    # Square duct: all Re except 2000
    'squareDuctAve_Re_1100', 'squareDuctAve_Re_1150', 'squareDuctAve_Re_1250',
    'squareDuctAve_Re_1300', 'squareDuctAve_Re_1350', 'squareDuctAve_Re_1400',
    'squareDuctAve_Re_1500', 'squareDuctAve_Re_1600',
    'squareDuctAve_Re_2205', 'squareDuctAve_Re_2400', 'squareDuctAve_Re_2600',
    'squareDuctAve_Re_2900', 'squareDuctAve_Re_3200', 'squareDuctAve_Re_3500',
    # Periodic hills: alpha = 0.5, 1.0, 1.5
    'case_0p5', 'case_1p0', 'case_1p5',
    # Converging-diverging channel
    'convdiv12600',
]

VAL_CASES = [
    'squareDuctAve_Re_2000',  # SD interpolation
    'case_0p8',               # PH interpolation
]

TEST_CASES = [
    'case_1p2',     # PH extrapolation
    'cbfs13700',    # New geometry
]


def load_mcconkey_csv(data_dir, rans_model='komegasst'):
    """Load McConkey CSV files and split by case."""
    rans_file = Path(data_dir) / f'{rans_model}.csv'
    ref_file = Path(data_dir) / 'REF.csv'

    print(f"Loading RANS data from {rans_file}...")
    rans_data = {}
    with open(rans_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row['Case']
            if case not in rans_data:
                rans_data[case] = []
            rans_data[case].append(row)

    print(f"Loading REF data from {ref_file}...")
    ref_data = {}
    with open(ref_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # REF doesn't have Case column — rows align with RANS
            pass  # Need to load by index

    # Reload both in aligned fashion
    print("Loading aligned RANS+REF data...")
    prefix = f'{rans_model}_'
    all_rows = []
    with open(rans_file) as rf, open(ref_file) as ff:
        rans_reader = csv.DictReader(rf)
        ref_reader = csv.DictReader(ff)
        for rans_row, ref_row in zip(rans_reader, ref_reader):
            combined = {}
            combined['Case'] = rans_row['Case']
            # RANS fields
            for k, v in rans_row.items():
                if k != 'Case' and k != '':
                    combined[k] = float(v)
            # REF fields
            for k, v in ref_row.items():
                if k != '' and k != 'Case':
                    combined[k] = float(v)
            all_rows.append(combined)

    # Group by case
    by_case = defaultdict(list)
    for row in all_rows:
        by_case[row['Case']].append(row)

    print(f"Loaded {len(all_rows):,} total points across {len(by_case)} cases")
    return by_case


def extract_features_and_labels(rows, rans_model='komegasst'):
    """
    Extract invariant features and anisotropy labels from raw data.

    Features (5 scalar invariants of normalized S and Omega):
      lambda_1 = tr(S_hat^2)
      lambda_2 = tr(Omega_hat^2)
      lambda_3 = tr(S_hat^3)
      lambda_4 = tr(Omega_hat^2 S_hat)
      lambda_5 = tr(Omega_hat^2 S_hat^2)

    Labels: anisotropy tensor b_ij from DNS (REF)
    """
    prefix = f'{rans_model}_'
    n = len(rows)

    # Extract velocity gradients from RANS
    gradU = np.zeros((n, 3, 3))
    k_vals = np.zeros(n)
    eps_vals = np.zeros(n)

    for idx, row in enumerate(rows):
        for i in range(3):
            for j in range(3):
                gradU[idx, i, j] = row[f'{prefix}gradU_{i+1}{j+1}']
        k_vals[idx] = max(row[f'{prefix}k'], 1e-30)
        eps_vals[idx] = max(row[f'{prefix}epsilon'], 1e-30)

    # Compute S and Omega, normalize by k/epsilon
    tau = k_vals / eps_vals  # turbulence time scale
    S = 0.5 * (gradU + np.swapaxes(gradU, 1, 2))  # symmetric
    Omega = 0.5 * (gradU - np.swapaxes(gradU, 1, 2))  # antisymmetric

    # Normalize: S_hat = tau * S, Omega_hat = tau * Omega
    S_hat = S * tau[:, None, None]
    Omega_hat = Omega * tau[:, None, None]

    # Compute 5 invariants
    invariants = np.zeros((n, 5))
    for idx in range(n):
        Sh = S_hat[idx]
        Oh = Omega_hat[idx]
        Sh2 = Sh @ Sh
        Oh2 = Oh @ Oh
        invariants[idx, 0] = np.trace(Sh2)           # tr(S^2)
        invariants[idx, 1] = np.trace(Oh2)           # tr(Omega^2)
        invariants[idx, 2] = np.trace(Sh2 @ Sh)      # tr(S^3)
        invariants[idx, 3] = np.trace(Oh2 @ Sh)      # tr(Omega^2 S)
        invariants[idx, 4] = np.trace(Oh2 @ Sh2)     # tr(Omega^2 S^2)

    # Extract anisotropy labels from DNS (REF)
    # b_ij = a_ij / (2k) - delta_ij / 3
    # McConkey provides REF_b_11, REF_b_12, etc. directly
    anisotropy = np.zeros((n, 6))  # b_11, b_12, b_13, b_22, b_23, b_33
    for idx, row in enumerate(rows):
        anisotropy[idx, 0] = row['REF_b_11']
        anisotropy[idx, 1] = row['REF_b_12']
        anisotropy[idx, 2] = row['REF_b_13']
        anisotropy[idx, 3] = row['REF_b_22']
        anisotropy[idx, 4] = row['REF_b_23']
        anisotropy[idx, 5] = row['REF_b_33']

    # Compute tensor basis (Pope 1975, 10 tensors)
    basis = compute_tensor_basis(S_hat, Omega_hat)

    return invariants, anisotropy, basis, k_vals


def compute_tensor_basis(S_hat, Omega_hat):
    """
    Compute Pope (1975) 10-tensor integrity basis.

    T1 = S, T2 = SR - RS, T3 = S^2 - tr(S^2)/3 I, ...
    Returns: [N, 10, 6] (10 basis tensors, 6 symmetric components)
    """
    n = len(S_hat)
    I3 = np.eye(3)
    basis = np.zeros((n, 10, 6))

    for idx in range(n):
        S = S_hat[idx]
        R = Omega_hat[idx]
        S2 = S @ S
        R2 = R @ R
        SR = S @ R
        RS = R @ S

        T = [None] * 10
        T[0] = S
        T[1] = SR - RS
        T[2] = S2 - np.trace(S2) / 3.0 * I3
        T[3] = R2 - np.trace(R2) / 3.0 * I3
        T[4] = R @ S2 - S2 @ R
        T[5] = R2 @ S + S @ R2 - 2.0/3.0 * np.trace(S @ R2) * I3
        T[6] = R @ S @ R2 - R2 @ S @ R
        T[7] = S @ R @ S2 - S2 @ R @ S
        T[8] = R2 @ S2 + S2 @ R2 - 2.0/3.0 * np.trace(S2 @ R2) * I3
        T[9] = R @ S2 @ R2 - R2 @ S2 @ R

        for t in range(10):
            # Store as 6 symmetric components: 11, 12, 13, 22, 23, 33
            basis[idx, t, 0] = T[t][0, 0]
            basis[idx, t, 1] = T[t][0, 1]
            basis[idx, t, 2] = T[t][0, 2]
            basis[idx, t, 3] = T[t][1, 1]
            basis[idx, t, 4] = T[t][1, 2]
            basis[idx, t, 5] = T[t][2, 2]

    return basis


def build_split(by_case, cases, rans_model='komegasst'):
    """Build feature/label arrays for a set of cases."""
    all_rows = []
    for case in cases:
        if case in by_case:
            all_rows.extend(by_case[case])
        else:
            print(f"  WARNING: Case '{case}' not found in dataset")

    if not all_rows:
        return None, None, None, None

    print(f"  Extracting features from {len(all_rows):,} points ({len(cases)} cases)...")
    return extract_features_and_labels(all_rows, rans_model)


# ============================================================================
# Model architectures
# ============================================================================

class MLPModel(nn.Module):
    """Simple MLP for scalar eddy viscosity prediction."""
    def __init__(self, n_in=5, hidden=[32, 32], n_out=1, activation='tanh'):
        super().__init__()
        layers = []
        prev = n_in
        act_fn = nn.Tanh if activation == 'tanh' else nn.LeakyReLU
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TBNNModel(nn.Module):
    """TBNN: invariants → coefficients g_n, then b_ij = sum g_n T^(n)_ij."""
    def __init__(self, n_in=5, hidden=[64, 64, 64], n_basis=10):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, n_basis))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PITBNNLoss(nn.Module):
    """Physics-informed loss with realizability constraints (Lumley triangle)."""
    def __init__(self, alpha=0.01, beta=10.0):
        super().__init__()
        self.alpha = alpha  # L2 regularization
        self.beta = beta    # realizability penalty weight

    def forward(self, b_pred, b_true, model):
        # MSE loss
        mse = nn.functional.mse_loss(b_pred, b_true)

        # L2 regularization
        l2 = sum(p.pow(2).sum() for p in model.parameters()) * self.alpha

        # Realizability constraints on b_ij eigenvalues
        # b_11, b_22, b_33 are diagonal: b_11 >= -1/3, b_ii <= 2/3
        # Also: 2*II + 8/9 >= 0 (Lumley triangle)
        b11 = b_pred[:, 0]  # b_xx
        b22 = b_pred[:, 3]  # b_yy
        b33 = b_pred[:, 5]  # b_zz
        b12 = b_pred[:, 1]

        # Constraint: b_ii >= -1/3
        c1 = torch.relu(-b11 - 1.0/3.0)
        c2 = torch.relu(-b22 - 1.0/3.0)
        c3 = torch.relu(-b33 - 1.0/3.0)

        # Constraint: b_ii <= 2/3
        c4 = torch.relu(b11 - 2.0/3.0)
        c5 = torch.relu(b22 - 2.0/3.0)
        c6 = torch.relu(b33 - 2.0/3.0)

        penalty = (c1.pow(2) + c2.pow(2) + c3.pow(2) +
                   c4.pow(2) + c5.pow(2) + c6.pow(2)).mean()

        return mse + l2 + self.beta * penalty


# ============================================================================
# Training functions
# ============================================================================

def standardize(X, mean=None, std=None):
    """Z-score standardization. Returns (X_norm, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0  # prevent division by zero
    return (X - mean) / std, mean, std


def train_tbnn(invariants_train, anisotropy_train, basis_train,
               invariants_val, anisotropy_val, basis_val,
               hidden=[64, 64, 64], n_basis=10, lr=1e-3, epochs=200,
               batch_size=256, device='cpu', physics_informed=False,
               pi_beta=10.0):
    """Train a TBNN (or PI-TBNN) model."""
    # Standardize inputs
    inv_train, inv_mean, inv_std = standardize(invariants_train)
    inv_val, _, _ = standardize(invariants_val, inv_mean, inv_std)

    # Convert to tensors
    inv_t = torch.FloatTensor(inv_train).to(device)
    bij_t = torch.FloatTensor(anisotropy_train).to(device)
    basis_t = torch.FloatTensor(basis_train).to(device)

    inv_v = torch.FloatTensor(inv_val).to(device)
    bij_v = torch.FloatTensor(anisotropy_val).to(device)
    basis_v = torch.FloatTensor(basis_val).to(device)

    model = TBNNModel(n_in=invariants_train.shape[1], hidden=hidden,
                      n_basis=n_basis).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    if physics_informed:
        pi_loss_fn = PITBNNLoss(alpha=0.01, beta=pi_beta)

    dataset = TensorDataset(inv_t, bij_t, basis_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_inv, batch_bij, batch_basis in loader:
            g = model(batch_inv)  # [B, n_basis]

            # Reconstruct b_ij = sum g_n T^(n)_ij
            # g: [B, n_basis], basis: [B, n_basis, 6]
            b_pred = (g.unsqueeze(2) * batch_basis).sum(dim=1)  # [B, 6]

            if physics_informed:
                loss = pi_loss_fn(b_pred, batch_bij, model)
            else:
                loss = nn.functional.mse_loss(b_pred, batch_bij)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_inv)

        epoch_loss /= len(inv_t)

        # Validation
        model.eval()
        with torch.no_grad():
            g_val = model(inv_v)
            b_val_pred = (g_val.unsqueeze(2) * basis_v).sum(dim=1)
            val_loss = nn.functional.mse_loss(b_val_pred, bij_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:4d}/{epochs}: train_loss={epoch_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

        if patience_counter >= 50:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Compute RMSE on validation
    with torch.no_grad():
        g_val = model(inv_v)
        b_val_pred = (g_val.unsqueeze(2) * basis_v).sum(dim=1)
        rmse = torch.sqrt(nn.functional.mse_loss(b_val_pred, bij_v)).item()

    print(f"    Final val RMSE(b): {rmse:.6f}")

    return model, inv_mean, inv_std, rmse


def train_mlp_nut(invariants_train, anisotropy_train, k_train,
                  invariants_val, anisotropy_val, k_val,
                  hidden=[32, 32], lr=1e-3, epochs=200,
                  batch_size=256, device='cpu'):
    """Train MLP for scalar eddy viscosity (simplified: predict |b| magnitude)."""
    # Target: magnitude of anisotropy as proxy for nu_t
    target_train = np.sqrt((anisotropy_train ** 2).sum(axis=1, keepdims=True))
    target_val = np.sqrt((anisotropy_val ** 2).sum(axis=1, keepdims=True))

    # Standardize
    inv_train, inv_mean, inv_std = standardize(invariants_train)
    inv_val, _, _ = standardize(invariants_val, inv_mean, inv_std)

    inv_t = torch.FloatTensor(inv_train).to(device)
    tgt_t = torch.FloatTensor(target_train).to(device)
    inv_v = torch.FloatTensor(inv_val).to(device)
    tgt_v = torch.FloatTensor(target_val).to(device)

    model = MLPModel(n_in=invariants_train.shape[1], hidden=hidden, n_out=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    dataset = TensorDataset(inv_t, tgt_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(inv_v)
            val_loss = nn.functional.mse_loss(val_pred, tgt_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            rmse = math.sqrt(val_loss)
            print(f"    Epoch {epoch+1:4d}/{epochs}: val_RMSE={rmse:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"    Final val RMSE: {math.sqrt(best_val_loss):.6f}")

    return model, inv_mean, inv_std, math.sqrt(best_val_loss)


def train_tbrf(invariants_train, anisotropy_train, basis_train,
               invariants_val, anisotropy_val, basis_val,
               n_trees=200, max_depth=20):
    """Train Tensor Basis Random Forest (Kaandorp & Dwight 2020)."""
    # Standardize inputs
    inv_train, inv_mean, inv_std = standardize(invariants_train)
    inv_val, _, _ = standardize(invariants_val, inv_mean, inv_std)

    n_basis = basis_train.shape[1]
    n_comp = anisotropy_train.shape[1]

    # Solve for g_n coefficients: b_ij = sum g_n T^(n)_ij
    # For each point, this is a least-squares problem: T @ g = b
    print(f"    Solving for tensor basis coefficients ({len(inv_train)} points)...")
    g_train = np.zeros((len(inv_train), n_basis))
    for idx in range(len(inv_train)):
        T = basis_train[idx]  # [n_basis, n_comp]
        b = anisotropy_train[idx]  # [n_comp]
        # Least squares: g = (T^T T)^{-1} T^T b
        g_train[idx], _, _, _ = np.linalg.lstsq(T.T, b, rcond=None)

    # Train one RF per coefficient
    print(f"    Training {n_basis} random forests ({n_trees} trees, depth {max_depth})...")
    forests = []
    for n in range(n_basis):
        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth,
                                   n_jobs=-1, random_state=42)
        rf.fit(inv_train, g_train[:, n])
        forests.append(rf)

    # Validate
    g_val_pred = np.zeros((len(inv_val), n_basis))
    for n in range(n_basis):
        g_val_pred[:, n] = forests[n].predict(inv_val)

    # Reconstruct b_ij
    b_val_pred = np.zeros_like(anisotropy_val)
    for idx in range(len(inv_val)):
        for n in range(n_basis):
            b_val_pred[idx] += g_val_pred[idx, n] * basis_val[idx, n]

    rmse = np.sqrt(np.mean((b_val_pred - anisotropy_val) ** 2))
    print(f"    Final val RMSE(b): {rmse:.6f}")

    return forests, inv_mean, inv_std, rmse


# ============================================================================
# Weight export
# ============================================================================

def export_pytorch_model(model, inv_mean, inv_std, output_dir, model_type,
                         metadata_extra=None):
    """Export PyTorch model weights to text files for C++ inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.cpu()
    state = model.state_dict()

    layer_idx = 0
    for name, param in state.items():
        w = param.numpy()
        if 'weight' in name:
            np.savetxt(output_dir / f'layer{layer_idx}_W.txt', w, fmt='%.10e')
        elif 'bias' in name:
            np.savetxt(output_dir / f'layer{layer_idx}_b.txt', w, fmt='%.10e')
            layer_idx += 1

    # Save scaling
    np.savetxt(output_dir / 'input_means.txt', inv_mean, fmt='%.10e')
    np.savetxt(output_dir / 'input_stds.txt', inv_std, fmt='%.10e')

    # Metadata
    meta = {
        'name': f'{model_type}_mcconkey',
        'type': model_type,
        'description': f'{model_type} trained on McConkey et al. (2021) dataset',
        'training': {
            'dataset': 'McConkey et al. (2021)',
            'split': 'TBKAN-2025 case-holdout protocol',
            'rans_baseline': 'k-omega SST',
        },
    }
    if metadata_extra:
        meta.update(metadata_extra)

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    Exported to {output_dir}/")


def export_rf_model(forests, inv_mean, inv_std, output_dir):
    """Export random forest to a format loadable by C++."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scaling
    np.savetxt(output_dir / 'input_means.txt', inv_mean, fmt='%.10e')
    np.savetxt(output_dir / 'input_stds.txt', inv_std, fmt='%.10e')

    # Save RF as pickle (for Python) and as tree export (for C++)
    import pickle
    with open(output_dir / 'forests.pkl', 'wb') as f:
        pickle.dump(forests, f)

    meta = {
        'name': 'tbrf_mcconkey',
        'type': 'nn_tbrf',
        'description': 'TBRF (Kaandorp & Dwight 2020) trained on McConkey dataset',
        'n_basis': len(forests),
        'n_trees': forests[0].n_estimators,
        'max_depth': forests[0].max_depth,
        'training': {
            'dataset': 'McConkey et al. (2021)',
            'split': 'TBKAN-2025 case-holdout protocol',
        },
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    Exported to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train all NN turbulence models')
    parser.add_argument('--data_dir', default='mcconkey_data',
                        help='Path to McConkey dataset')
    parser.add_argument('--output_dir', default='data/models',
                        help='Output directory for trained models')
    parser.add_argument('--device', default='cuda' if 'torch' in dir() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Max training epochs for NN models')
    parser.add_argument('--rans_model', default='komegasst',
                        help='RANS baseline model in dataset')
    parser.add_argument('--models', nargs='*',
                        default=['mlp', 'mlp_large', 'tbnn', 'pi_tbnn', 'tbrf'],
                        help='Models to train')
    args = parser.parse_args()

    import_deps()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("  Unified Training Pipeline")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"RANS model: {args.rans_model}")
    print(f"Device: {args.device}")
    print(f"Models: {', '.join(args.models)}")
    print()

    # Load data
    by_case = load_mcconkey_csv(args.data_dir, args.rans_model)

    # Build train/val/test splits
    print("\n--- Building splits ---")
    print(f"Train cases: {len(TRAIN_CASES)}")
    inv_train, bij_train, basis_train, k_train = build_split(by_case, TRAIN_CASES, args.rans_model)
    print(f"Val cases: {len(VAL_CASES)}")
    inv_val, bij_val, basis_val, k_val = build_split(by_case, VAL_CASES, args.rans_model)
    print(f"Test cases: {len(TEST_CASES)}")
    inv_test, bij_test, basis_test, k_test = build_split(by_case, TEST_CASES, args.rans_model)

    if inv_train is None:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    print(f"\nSplit sizes: train={len(inv_train):,}, val={len(inv_val):,}, test={len(inv_test):,}")

    results = {}

    # ---- MLP (small) ----
    if 'mlp' in args.models:
        print("\n" + "=" * 60)
        print("  Training MLP (5→32→32→1)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_mlp_nut(
            inv_train, bij_train, k_train, inv_val, bij_val, k_val,
            hidden=[32, 32], lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/mlp_paper', 'nn_mlp',
                             {'architecture': {'layers': [5, 32, 32, 1]}})
        results['mlp'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- MLP (large) ----
    if 'mlp_large' in args.models:
        print("\n" + "=" * 60)
        print("  Training MLP-Large (5→128→128→128→128→1)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_mlp_nut(
            inv_train, bij_train, k_train, inv_val, bij_val, k_val,
            hidden=[128, 128, 128, 128], lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/mlp_large_paper', 'nn_mlp',
                             {'architecture': {'layers': [5, 128, 128, 128, 128, 1]}})
        results['mlp_large'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- TBNN ----
    if 'tbnn' in args.models:
        print("\n" + "=" * 60)
        print("  Training TBNN (5→64→64→64→10)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[64, 64, 64], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/tbnn_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 64, 64, 64, 10]}})
        results['tbnn'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- PI-TBNN ----
    if 'pi_tbnn' in args.models:
        print("\n" + "=" * 60)
        print("  Training PI-TBNN (5→64→64→64→10, realizability loss)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[64, 64, 64], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device, physics_informed=True, pi_beta=10.0)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/pi_tbnn_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 64, 64, 64, 10]},
                              'physics_informed': True,
                              'realizability_beta': 10.0})
        results['pi_tbnn'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- TBRF ----
    if 'tbrf' in args.models:
        print("\n" + "=" * 60)
        print("  Training TBRF (200 trees, depth 20)")
        print("=" * 60)
        t0 = time.time()
        forests, mean, std, rmse = train_tbrf(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            n_trees=200, max_depth=20)
        export_rf_model(forests, mean, std, f'{args.output_dir}/tbrf_paper')
        results['tbrf'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Val RMSE(b)':<15} {'Train Time':<15}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<15} {res['rmse']:<15.6f} {res['time']:<15.1f}s")
    print()

    # Save summary
    summary_path = Path(args.output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
