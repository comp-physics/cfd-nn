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
                if not k or k == 'Case' or not v:
                    continue
                combined[k] = float(v)
            # REF fields
            for k, v in ref_row.items():
                if not k or k == 'Case' or not v:
                    continue
                combined[k] = float(v)
            # Skip rows with missing REF anisotropy
            if 'REF_b_11' not in combined:
                continue
            all_rows.append(combined)

    # Group by case
    by_case = defaultdict(list)
    for row in all_rows:
        by_case[row['Case']].append(row)

    print(f"Loaded {len(all_rows):,} total points across {len(by_case)} cases")
    return by_case


def extract_features_and_labels(rows, rans_model='komegasst', device='cpu'):
    """
    Extract invariant features and anisotropy labels from raw data.
    Uses PyTorch on GPU for fast batched 3x3 matrix operations.

    Features (5 scalar invariants of normalized S and Omega):
      lambda_1..5 = tr(S^2), tr(R^2), tr(S^3), tr(R^2 S), tr(R^2 S^2)

    Labels: anisotropy tensor b_ij from DNS (REF)
    """
    prefix = f'{rans_model}_'
    n = len(rows)

    # Extract raw data into numpy arrays (CPU — dict access is the bottleneck)
    gradU = np.zeros((n, 3, 3))
    k_vals = np.zeros(n)
    eps_vals = np.zeros(n)
    anisotropy = np.zeros((n, 6))

    for idx, row in enumerate(rows):
        for i in range(3):
            for j in range(3):
                gradU[idx, i, j] = row[f'{prefix}gradU_{i+1}{j+1}']
        k_vals[idx] = max(row[f'{prefix}k'], 1e-30)
        eps_vals[idx] = max(row[f'{prefix}epsilon'], 1e-30)
        anisotropy[idx, 0] = row['REF_b_11']
        anisotropy[idx, 1] = row['REF_b_12']
        anisotropy[idx, 2] = row['REF_b_13']
        anisotropy[idx, 3] = row['REF_b_22']
        anisotropy[idx, 4] = row['REF_b_23']
        anisotropy[idx, 5] = row['REF_b_33']

    # Move to GPU for fast batched matrix math
    dev = torch.device(device)
    gU = torch.tensor(gradU, dtype=torch.float64, device=dev)
    tau = torch.tensor(k_vals / eps_vals, dtype=torch.float64, device=dev)

    S_hat = 0.5 * (gU + gU.transpose(1, 2)) * tau[:, None, None]
    O_hat = 0.5 * (gU - gU.transpose(1, 2)) * tau[:, None, None]

    # Batched matmul helper
    mm = torch.bmm
    tr = lambda A: A.diagonal(dim1=1, dim2=2).sum(dim=1)

    S2 = mm(S_hat, S_hat)
    O2 = mm(O_hat, O_hat)

    # 5 invariants
    invariants = torch.zeros(n, 5, dtype=torch.float64, device=dev)
    invariants[:, 0] = tr(S2)
    invariants[:, 1] = tr(O2)
    invariants[:, 2] = tr(mm(S2, S_hat))
    invariants[:, 3] = tr(mm(O2, S_hat))
    invariants[:, 4] = tr(mm(O2, S2))

    # 10-tensor integrity basis (Pope 1975)
    basis = compute_tensor_basis_gpu(S_hat, O_hat, S2, O2, dev)

    # Back to numpy
    return (invariants.cpu().numpy().astype(np.float64),
            anisotropy,
            basis.cpu().numpy().astype(np.float64),
            k_vals)


def compute_tensor_basis_gpu(S, R, S2, R2, dev):
    """
    Compute Pope (1975) 10-tensor integrity basis on GPU.
    All inputs are [N, 3, 3] torch tensors on device.
    Returns: [N, 10, 6] tensor (6 symmetric components per basis tensor).
    """
    n = S.shape[0]
    mm = torch.bmm
    tr = lambda A: A.diagonal(dim1=1, dim2=2).sum(dim=1)
    I3 = torch.eye(3, dtype=S.dtype, device=dev).unsqueeze(0)  # [1,3,3]

    SR = mm(S, R)
    RS = mm(R, S)

    T = [None] * 10
    T[0] = S
    T[1] = SR - RS
    T[2] = S2 - tr(S2)[:, None, None] / 3.0 * I3
    T[3] = R2 - tr(R2)[:, None, None] / 3.0 * I3
    T[4] = mm(R, S2) - mm(S2, R)
    T[5] = mm(R2, S) + mm(S, R2) - 2.0/3.0 * tr(mm(S, R2))[:, None, None] * I3
    T[6] = mm(mm(R, S), R2) - mm(mm(R2, S), R)
    T[7] = mm(mm(S, R), S2) - mm(mm(S2, R), S)
    T[8] = mm(R2, S2) + mm(S2, R2) - 2.0/3.0 * tr(mm(S2, R2))[:, None, None] * I3
    T[9] = mm(mm(R, S2), R2) - mm(mm(R2, S2), R)

    # Extract 6 symmetric components: 11, 12, 13, 22, 23, 33
    basis = torch.zeros(n, 10, 6, dtype=S.dtype, device=dev)
    for t in range(10):
        basis[:, t, 0] = T[t][:, 0, 0]
        basis[:, t, 1] = T[t][:, 0, 1]
        basis[:, t, 2] = T[t][:, 0, 2]
        basis[:, t, 3] = T[t][:, 1, 1]
        basis[:, t, 4] = T[t][:, 1, 2]
        basis[:, t, 5] = T[t][:, 2, 2]

    return basis


def build_split(by_case, cases, rans_model='komegasst', device='cpu'):
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
    return extract_features_and_labels(all_rows, rans_model, device=device)


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
               hidden=[64, 64, 64], n_basis=10, lr=1e-3, epochs=1000,
               batch_size=256, device='cpu', physics_informed=False,
               pi_beta=0.1, pi_warmup=100):
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6)

    if physics_informed:
        pi_loss_fn = PITBNNLoss(alpha=1e-6, beta=pi_beta)

    dataset = TensorDataset(inv_t, bij_t, basis_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # PI-TBNN: linear warmup of realizability penalty
        if physics_informed:
            warmup_scale = min(1.0, epoch / pi_warmup)
            pi_loss_fn.beta = pi_beta * warmup_scale

        for batch_inv, batch_bij, batch_basis in loader:
            g = model(batch_inv)  # [B, n_basis]

            # Reconstruct b_ij = sum g_n T^(n)_ij
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
        scheduler.step()

        # Validation (always use pure MSE for fair comparison)
        model.eval()
        with torch.no_grad():
            g_val = model(inv_v)
            b_val_pred = (g_val.unsqueeze(2) * basis_v).sum(dim=1)
            val_loss = nn.functional.mse_loss(b_val_pred, bij_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            extra = f", beta={pi_loss_fn.beta:.3f}" if physics_informed else ""
            print(f"    Epoch {epoch+1:4d}/{epochs}: train={epoch_loss:.6f}, "
                  f"val={val_loss:.6f}, best={best_val_loss:.6f}, "
                  f"lr={lr_now:.2e}{extra}")

        if patience_counter >= 150:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Compute RMSE on validation
    with torch.no_grad():
        g_val = model(inv_v)
        b_val_pred = (g_val.unsqueeze(2) * basis_v).sum(dim=1)
        rmse = torch.sqrt(nn.functional.mse_loss(b_val_pred, bij_v)).item()

    print(f"    Final val RMSE(b): {rmse:.6f} (stopped at epoch {min(epoch+1, epochs)})")

    return model, inv_mean, inv_std, rmse


def train_mlp_nut(invariants_train, anisotropy_train, k_train,
                  invariants_val, anisotropy_val, k_val,
                  hidden=[32, 32], lr=1e-3, epochs=1000,
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6)

    dataset = TensorDataset(inv_t, tgt_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(inv_v)
            val_loss = nn.functional.mse_loss(val_pred, tgt_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            rmse = math.sqrt(val_loss)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:4d}/{epochs}: val_RMSE={rmse:.6f}, "
                  f"best={math.sqrt(best_val_loss):.6f}, lr={lr_now:.2e}")

        if patience_counter >= 150:
            print(f"    Early stopping at epoch {epoch+1}")
            break

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
    # System per point: A @ g = b, where A = T.T is [6, 10], underdetermined
    # Min-norm solution: g = A.T @ (A @ A.T)^{-1} @ b
    # A = basis.transpose [N,6,10], A.T = basis [N,10,6]
    print(f"    Solving for tensor basis coefficients ({len(inv_train)} points, vectorized)...")
    A = basis_train.transpose(0, 2, 1)  # [N, 6, 10]
    AT = basis_train  # [N, 10, 6]
    AAT = np.einsum('nij,nkj->nik', A, A)  # [N, 6, 6] = A @ A.T
    # Regularize for numerical stability
    AAT += 1e-10 * np.eye(n_comp)[None, :, :]
    AAT_inv = np.linalg.inv(AAT)  # [N, 6, 6]
    # g = A.T @ AAT_inv @ b
    AAT_inv_b = np.einsum('nij,nj->ni', AAT_inv, anisotropy_train)  # [N, 6]
    g_train = np.einsum('nij,nj->ni', AT, AAT_inv_b)  # [N, 10]

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

    # Reconstruct b_ij: b_pred[i] = sum_n g[i,n] * basis[i,n,:]
    b_val_pred = np.einsum('nb,nbc->nc', g_val_pred, basis_val)

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
    """Export random forest as pickle (full) and compact flat files (for C++)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scaling
    np.savetxt(output_dir / 'input_means.txt', inv_mean, fmt='%.10e')
    np.savetxt(output_dir / 'input_stds.txt', inv_std, fmt='%.10e')

    # Save full pickle (for Python reuse)
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


def export_rf_compact(forests, inv_mean, inv_std, output_dir, n_trees):
    """Export a compact TBRF with n_trees per basis coefficient.

    Writes flat binary files for C++ inference:
      - trees.bin: packed tree nodes (children_left, children_right,
                   feature, threshold, value) as int32/float32
      - tree_offsets.txt: start index of each tree in the flat array
      - input_means.txt, input_stds.txt: normalization
      - metadata.json: model info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / 'input_means.txt', inv_mean, fmt='%.10e')
    np.savetxt(output_dir / 'input_stds.txt', inv_std, fmt='%.10e')

    n_basis = len(forests)

    # Flatten all trees into packed arrays
    all_children_left = []
    all_children_right = []
    all_feature = []
    all_threshold = []
    all_value = []
    tree_offsets = []  # (basis_idx, tree_idx, start_node, n_nodes)
    node_offset = 0

    for basis_idx in range(n_basis):
        for tree_idx in range(n_trees):
            tree = forests[basis_idx].estimators_[tree_idx].tree_
            n = tree.node_count

            # Remap children indices to global offset
            cl = tree.children_left.copy()
            cr = tree.children_right.copy()
            # -1 (leaf sentinel) stays -1; others get offset
            mask = cl >= 0
            cl[mask] += node_offset
            mask = cr >= 0
            cr[mask] += node_offset

            all_children_left.append(cl.astype(np.int32))
            all_children_right.append(cr.astype(np.int32))
            all_feature.append(tree.feature.astype(np.int32))
            all_threshold.append(tree.threshold.astype(np.float32))
            all_value.append(tree.value[:, 0, 0].astype(np.float32))

            tree_offsets.append((basis_idx, tree_idx, node_offset, n))
            node_offset += n

    # Concatenate and write binary
    cl_flat = np.concatenate(all_children_left)
    cr_flat = np.concatenate(all_children_right)
    feat_flat = np.concatenate(all_feature)
    thresh_flat = np.concatenate(all_threshold)
    val_flat = np.concatenate(all_value)

    # Pack as single binary: [cl, cr, feat, thresh, value] each node_offset elements
    # Header: total_nodes (int32), n_basis (int32), n_trees (int32)
    total_nodes = node_offset
    header = np.array([total_nodes, n_basis, n_trees], dtype=np.int32)

    with open(output_dir / 'trees.bin', 'wb') as f:
        header.tofile(f)
        cl_flat.tofile(f)
        cr_flat.tofile(f)
        feat_flat.tofile(f)
        thresh_flat.tofile(f)
        val_flat.tofile(f)

    # Write offsets as text (human-readable)
    with open(output_dir / 'tree_offsets.txt', 'w') as f:
        f.write("# basis_idx tree_idx start_node n_nodes\n")
        for b, t, s, n in tree_offsets:
            f.write(f"{b} {t} {s} {n}\n")

    size_mb = os.path.getsize(output_dir / 'trees.bin') / 1e6
    meta = {
        'name': f'tbrf_{n_trees}t_mcconkey',
        'type': 'nn_tbrf',
        'description': f'Compact TBRF ({n_trees} trees) for C++ inference',
        'n_basis': n_basis,
        'n_trees': n_trees,
        'max_depth': forests[0].max_depth,
        'total_nodes': total_nodes,
        'binary_size_mb': round(size_mb, 1),
        'format': 'trees.bin: header(3xi32) + children_left(i32) + children_right(i32) + feature(i32) + threshold(f32) + value(f32)',
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    Exported {n_trees}-tree TBRF: {total_nodes:,} nodes, {size_mb:.1f} MB -> {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train all NN turbulence models')
    parser.add_argument('--data_dir', default='mcconkey_data',
                        help='Path to McConkey dataset')
    parser.add_argument('--output_dir', default='data/models',
                        help='Output directory for trained models')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Max training epochs for NN models')
    parser.add_argument('--rans_model', default='komegasst',
                        help='RANS baseline model in dataset')
    parser.add_argument('--models', nargs='*',
                        default=['mlp', 'mlp_med', 'mlp_large', 'tbnn', 'tbnn_small', 'tbnn_large', 'pi_tbnn', 'pi_tbnn_small', 'pi_tbnn_large', 'tbrf'],
                        help='Models to train')
    args = parser.parse_args()

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
    inv_train, bij_train, basis_train, k_train = build_split(by_case, TRAIN_CASES, args.rans_model, args.device)
    print(f"Val cases: {len(VAL_CASES)}")
    inv_val, bij_val, basis_val, k_val = build_split(by_case, VAL_CASES, args.rans_model, args.device)
    print(f"Test cases: {len(TEST_CASES)}")
    inv_test, bij_test, basis_test, k_test = build_split(by_case, TEST_CASES, args.rans_model, args.device)

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

    # ---- MLP (medium) ----
    if 'mlp_med' in args.models:
        print("\n" + "=" * 60)
        print("  Training MLP-Medium (5→64→64→1)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_mlp_nut(
            inv_train, bij_train, k_train, inv_val, bij_val, k_val,
            hidden=[64, 64], lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/mlp_med_paper', 'nn_mlp',
                             {'architecture': {'layers': [5, 64, 64, 1]}})
        results['mlp_med'] = {'rmse': rmse, 'time': time.time() - t0}

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

    # ---- TBNN (small) ----
    if 'tbnn_small' in args.models:
        print("\n" + "=" * 60)
        print("  Training TBNN-Small (5→32→32→10)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[32, 32], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/tbnn_small_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 32, 32, 10]}})
        results['tbnn_small'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- TBNN (large) ----
    if 'tbnn_large' in args.models:
        print("\n" + "=" * 60)
        print("  Training TBNN-Large (5→128→128→128→10)")
        print("=" * 60)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[128, 128, 128], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/tbnn_large_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 128, 128, 128, 10]}})
        results['tbnn_large'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- PI-TBNN sweep ----
    if 'pi_tbnn' in args.models:
        pi_betas = [0.001, 0.01, 0.1, 1.0]
        best_pi_rmse = float('inf')
        best_pi_beta = None
        pi_sweep_results = {}

        for beta in pi_betas:
            print("\n" + "=" * 60)
            print(f"  Training PI-TBNN (beta={beta})")
            print("=" * 60)
            t0 = time.time()
            model, mean, std, rmse = train_tbnn(
                inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
                hidden=[64, 64, 64], n_basis=10, lr=1e-3, epochs=args.epochs,
                device=args.device, physics_informed=True,
                pi_beta=beta, pi_warmup=100)
            elapsed = time.time() - t0
            pi_sweep_results[beta] = {'rmse': rmse, 'time': elapsed}

            if rmse < best_pi_rmse:
                best_pi_rmse = rmse
                best_pi_beta = beta
                best_pi_model = model
                best_pi_mean = mean
                best_pi_std = std

        # Export best
        export_pytorch_model(best_pi_model, best_pi_mean, best_pi_std,
                             f'{args.output_dir}/pi_tbnn_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 64, 64, 64, 10]},
                              'physics_informed': True,
                              'realizability_beta': best_pi_beta})
        results['pi_tbnn'] = {'rmse': best_pi_rmse, 'time': sum(r['time'] for r in pi_sweep_results.values()),
                              'best_beta': best_pi_beta, 'sweep': {str(k): v for k, v in pi_sweep_results.items()}}

    # ---- PI-TBNN (small) ----
    if 'pi_tbnn_small' in args.models:
        print("\n" + "=" * 60)
        print("  Training PI-TBNN-Small (5→32→32→10)")
        print("=" * 60)
        # Use best beta from medium sweep if available, else documented best (0.001)
        pi_beta_use = results.get('pi_tbnn', {}).get('best_beta', 0.001)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[32, 32], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device, physics_informed=True,
            pi_beta=pi_beta_use, pi_warmup=100)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/pi_tbnn_small_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 32, 32, 10]},
                              'physics_informed': True,
                              'realizability_beta': pi_beta_use})
        results['pi_tbnn_small'] = {'rmse': rmse, 'time': time.time() - t0}

    # ---- PI-TBNN (large) ----
    if 'pi_tbnn_large' in args.models:
        print("\n" + "=" * 60)
        print("  Training PI-TBNN-Large (5→128→128→128→10)")
        print("=" * 60)
        pi_beta_use = results.get('pi_tbnn', {}).get('best_beta', 0.001)
        t0 = time.time()
        model, mean, std, rmse = train_tbnn(
            inv_train, bij_train, basis_train, inv_val, bij_val, basis_val,
            hidden=[128, 128, 128], n_basis=10, lr=1e-3, epochs=args.epochs,
            device=args.device, physics_informed=True,
            pi_beta=pi_beta_use, pi_warmup=100)
        export_pytorch_model(model, mean, std,
                             f'{args.output_dir}/pi_tbnn_large_paper', 'nn_tbnn',
                             {'architecture': {'layers': [5, 128, 128, 128, 10]},
                              'physics_informed': True,
                              'realizability_beta': pi_beta_use})
        results['pi_tbnn_large'] = {'rmse': rmse, 'time': time.time() - t0}

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

        # Export compact variants for C++ solver experiments
        for nt in [1, 5, 10]:
            export_rf_compact(forests, mean, std,
                              f'{args.output_dir}/tbrf_{nt}t_paper', n_trees=nt)

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
