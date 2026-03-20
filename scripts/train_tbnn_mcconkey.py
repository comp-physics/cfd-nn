#!/usr/bin/env python3
"""
Train TBNN model on McConkey et al. (2021) dataset.

Reference:
    McConkey, R. et al. "A curated dataset for data-driven turbulence modelling."
    Scientific Data 8, 255 (2021).
    Dataset: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset

This script trains a Tensor Basis Neural Network (TBNN) following Ling et al. (2016)
for Reynolds stress anisotropy prediction.

Usage:
    # Download dataset first from Kaggle
    python train_tbnn_mcconkey.py --data_dir /path/to/mcconkey/data \
        --case periodic_hills --output data/models/tbnn_mcconkey_hills

Requirements:
    pip install numpy torch pandas scikit-learn matplotlib
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path


class TBNNModel(nn.Module):
    """
    Tensor Basis Neural Network for anisotropy prediction.
    
    Architecture based on Ling et al. (2016):
        Input: 5 invariants (lambda_i)
        Hidden: 3 layers of 64 neurons with tanh activation
        Output: 4 coefficients for 2D tensor basis (or 10 for 3D)
    """
    
    def __init__(self, n_inputs=5, n_basis=4, hidden_layers=[64, 64, 64]):
        super(TBNNModel, self).__init__()
        
        layers = []
        prev_size = n_inputs
        
        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.Tanh())
            prev_size = h_size
        
        # Output layer (no activation - raw coefficients)
        layers.append(nn.Linear(prev_size, n_basis))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class RANSDataset(Dataset):
    """
    Dataset for RANS turbulence modeling.
    
    Expected data format from McConkey dataset:
        - invariants: [N, 5] array of scalar invariants
        - anisotropy: [N, 3] array of b_ij (xx, xy, yy for 2D)
        - basis: [N, 4, 3] array of tensor basis functions
    """
    
    def __init__(self, invariants, anisotropy, basis):
        self.invariants = torch.FloatTensor(invariants)
        self.anisotropy = torch.FloatTensor(anisotropy)
        self.basis = torch.FloatTensor(basis)
        
    def __len__(self):
        return len(self.invariants)
    
    def __getitem__(self, idx):
        return self.invariants[idx], self.anisotropy[idx], self.basis[idx]


def load_mcconkey_data(data_dir, case='periodic_hills', split='train'):
    """
    Load data from McConkey dataset.
    
    Args:
        data_dir: Path to dataset directory
        case: Flow case ('periodic_hills', 'square_duct', etc.)
        split: 'train', 'val', or 'test'
    
    Returns:
        invariants: [N, 5] scalar invariants
        anisotropy: [N, 3] anisotropy tensor (b_xx, b_xy, b_yy)
        basis: [N, 4, 3] tensor basis
    
    Note: This is a template. Actual loading depends on dataset structure.
    Users should adapt based on the specific format of the downloaded data.
    """
    
    print(f"Loading McConkey data from {data_dir}")
    print(f"Case: {case}, Split: {split}")
    
    # Example file structure (adapt to actual dataset)
    case_dir = Path(data_dir) / case / split
    
    if not case_dir.exists():
        print(f"\nWARNING: Dataset directory not found: {case_dir}")
        print("Please download the McConkey dataset from:")
        print("https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset")
        print("\nGenerating synthetic data for demonstration...")
        return generate_synthetic_data(n_samples=10000)
    
    try:
        # Try to load NPZ format
        data = np.load(case_dir / 'data.npz')
        
        invariants = data['invariants']  # [N, 5]
        anisotropy = data['anisotropy']  # [N, 6] or [N, 3] for 2D
        
        # Compute tensor basis if not provided
        if 'basis' in data:
            basis = data['basis']
        else:
            print("Computing tensor basis from velocity gradients...")
            basis = compute_tensor_basis_from_data(data)
        
        # Convert 3D to 2D if needed (take xx, xy, yy components)
        if anisotropy.shape[1] == 6:
            anisotropy = anisotropy[:, [0, 1, 3]]  # b_xx, b_xy, b_yy
        
        if basis.shape[1] == 10:
            # Convert 3D basis to 2D: take first 4 tensors, extract
            # xx(0), xy(1), yy(3) from the 6-component layout
            basis = basis[:, :4, :][:, :, [0, 1, 3]]
        
        print(f"Loaded {len(invariants)} samples")
        print(f"  Invariants shape: {invariants.shape}")
        print(f"  Anisotropy shape: {anisotropy.shape}")
        print(f"  Basis shape: {basis.shape}")
        
        return invariants, anisotropy, basis
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data(n_samples=10000)


def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic data for testing when real dataset is not available.
    
    This creates random but physically plausible features and anisotropy.
    NOT suitable for real applications - only for testing the pipeline.
    """
    
    print(f"\nGenerating {n_samples} synthetic samples...")
    print("NOTE: This is DUMMY DATA for testing - not real DNS data!")
    
    # Generate random invariants (scaled reasonably)
    invariants = np.random.randn(n_samples, 5)
    invariants[:, 0] = np.abs(invariants[:, 0]) * 0.5  # Lambda_1 > 0
    invariants[:, 1] = np.abs(invariants[:, 1]) * 0.3  # Lambda_2 > 0
    invariants[:, 4] = np.random.rand(n_samples) * 0.5  # Wall distance
    
    # Generate synthetic anisotropy (trace-free)
    b_xy = np.random.randn(n_samples) * 0.1
    b_xx = np.random.randn(n_samples) * 0.1
    b_yy = -b_xx  # Trace-free for incompressible 2D
    
    anisotropy = np.stack([b_xx, b_xy, b_yy], axis=1)
    
    # Generate tensor basis (simplified)
    basis = np.random.randn(n_samples, 4, 3) * 0.2
    # Make trace-free
    for i in range(4):
        basis[:, i, 2] = -basis[:, i, 0]
    
    return invariants, anisotropy, basis


def compute_tensor_basis_from_data(data):
    """
    Compute Pope (1975) / McConkey et al. integrity tensor basis from
    velocity gradients.

    Supports both 2D (4 basis tensors) and 3D (10 basis tensors).

    2D basis (S, R are 2x2):
        T1 = S
        T2 = S*R - R*S
        T3 = S^2 - (1/2)*tr(S^2)*I
        T4 = R^2 - (1/2)*tr(R^2)*I

    3D basis (S, R are 3x3):
        T1  = S
        T2  = S*R - R*S
        T3  = S^2 - (1/3)*tr(S^2)*I
        T4  = R^2 - (1/3)*tr(R^2)*I
        T5  = R*S^2 - S^2*R
        T6  = R^2*S + S*R^2 - (2/3)*tr(S*R^2)*I
        T7  = R*S*R^2 - R^2*S*R
        T8  = S*R*S^2 - S^2*R*S
        T9  = R^2*S^2 + S^2*R^2 - (2/3)*tr(S^2*R^2)*I
        T10 = R*S^2*R^2 - R^2*S^2*R

    Args:
        data: Dictionary containing 'S' (strain rate) and 'Omega'
              (rotation rate) tensors, each [N, d, d] with d = 2 or 3.

    Returns:
        basis: [N, 4, 3] for 2D input (xx, xy, yy components)
               [N, 10, 6] for 3D input (xx, xy, xz, yy, yz, zz components)
    """

    S = data['S']        # [N, d, d] strain tensor
    R = data['Omega']    # [N, d, d] rotation tensor

    N = S.shape[0]
    d = S.shape[1]

    if d == 2:
        return _compute_tensor_basis_2d(S, R, N)
    elif d == 3:
        return _compute_tensor_basis_3d(S, R, N)
    else:
        raise ValueError(f"Unexpected tensor dimension d={d}; expected 2 or 3")


def _sym_components_2d(T):
    """Extract (xx, xy, yy) from [N, 2, 2] tensor batch."""
    return T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]


def _sym_components_3d(T):
    """Extract (xx, xy, xz, yy, yz, zz) from [N, 3, 3] tensor batch."""
    return (T[:, 0, 0], T[:, 0, 1], T[:, 0, 2],
            T[:, 1, 1], T[:, 1, 2], T[:, 2, 2])


def _deviatoric_2d(A, N):
    """Return deviatoric part of [N, 2, 2] tensor: A - (1/2)*tr(A)*I."""
    trA = A[:, 0, 0] + A[:, 1, 1]
    I2 = np.zeros_like(A)
    I2[:, 0, 0] = 1.0
    I2[:, 1, 1] = 1.0
    return A - 0.5 * trA[:, None, None] * I2


def _deviatoric_3d(A, N):
    """Return deviatoric part of [N, 3, 3] tensor: A - (1/3)*tr(A)*I."""
    trA = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    I3 = np.zeros_like(A)
    I3[:, 0, 0] = 1.0
    I3[:, 1, 1] = 1.0
    I3[:, 2, 2] = 1.0
    return A - (1.0 / 3.0) * trA[:, None, None] * I3


def _trace_3d(A):
    """Compute trace of [N, 3, 3] tensor batch."""
    return A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]


def _compute_tensor_basis_2d(S, R, N):
    """Compute the 4 basis tensors for the 2D case."""
    basis = np.zeros((N, 4, 3))

    # T1 = S
    basis[:, 0, 0], basis[:, 0, 1], basis[:, 0, 2] = _sym_components_2d(S)

    # T2 = S*R - R*S
    T2 = np.matmul(S, R) - np.matmul(R, S)
    basis[:, 1, 0], basis[:, 1, 1], basis[:, 1, 2] = _sym_components_2d(T2)

    # T3 = S^2 - (1/2)*tr(S^2)*I
    S2 = np.matmul(S, S)
    T3 = _deviatoric_2d(S2, N)
    basis[:, 2, 0], basis[:, 2, 1], basis[:, 2, 2] = _sym_components_2d(T3)

    # T4 = R^2 - (1/2)*tr(R^2)*I
    R2 = np.matmul(R, R)
    T4 = _deviatoric_2d(R2, N)
    basis[:, 3, 0], basis[:, 3, 1], basis[:, 3, 2] = _sym_components_2d(T4)

    return basis


def _compute_tensor_basis_3d(S, R, N):
    """
    Compute the 10 Pope (1975) integrity basis tensors for the 3D case.

    All products are standard matrix products.  S is symmetric, R is
    antisymmetric.  The output stores the 6 independent symmetric
    components (xx, xy, xz, yy, yz, zz) for each basis tensor.
    """
    basis = np.zeros((N, 10, 6))

    # Pre-compute powers and products used in multiple basis tensors
    S2 = np.matmul(S, S)      # S^2
    R2 = np.matmul(R, R)      # R^2
    SR = np.matmul(S, R)      # S*R
    RS = np.matmul(R, S)      # R*S
    SR2 = np.matmul(S, R2)    # S*R^2
    R2S = np.matmul(R2, S)    # R^2*S
    S2R = np.matmul(S2, R)    # S^2*R
    RS2 = np.matmul(R, S2)    # R*S^2
    R2S2 = np.matmul(R2, S2)  # R^2*S^2
    S2R2 = np.matmul(S2, R2)  # S^2*R^2

    def store(idx, T):
        """Store 6 symmetric components of [N, 3, 3] tensor into basis."""
        (basis[:, idx, 0], basis[:, idx, 1], basis[:, idx, 2],
         basis[:, idx, 3], basis[:, idx, 4], basis[:, idx, 5]) = \
            _sym_components_3d(T)

    I3 = np.zeros((N, 3, 3))
    I3[:, 0, 0] = 1.0
    I3[:, 1, 1] = 1.0
    I3[:, 2, 2] = 1.0

    # T1 = S
    store(0, S)

    # T2 = S*R - R*S
    store(1, SR - RS)

    # T3 = S^2 - (1/3)*tr(S^2)*I
    store(2, S2 - (1.0 / 3.0) * _trace_3d(S2)[:, None, None] * I3)

    # T4 = R^2 - (1/3)*tr(R^2)*I
    store(3, R2 - (1.0 / 3.0) * _trace_3d(R2)[:, None, None] * I3)

    # T5 = R*S^2 - S^2*R
    store(4, RS2 - S2R)

    # T6 = R^2*S + S*R^2 - (2/3)*tr(S*R^2)*I
    store(5, R2S + SR2 - (2.0 / 3.0) * _trace_3d(SR2)[:, None, None] * I3)

    # T7 = R*S*R^2 - R^2*S*R
    RSR2 = np.matmul(RS, R2)   # (R*S)*R^2
    R2SR = np.matmul(R2S, R)   # (R^2*S)*R
    store(6, RSR2 - R2SR)

    # T8 = S*R*S^2 - S^2*R*S
    SRS2 = np.matmul(SR, S2)   # (S*R)*S^2
    S2RS = np.matmul(S2R, S)   # (S^2*R)*S
    store(7, SRS2 - S2RS)

    # T9 = R^2*S^2 + S^2*R^2 - (2/3)*tr(S^2*R^2)*I
    store(8, R2S2 + S2R2 - (2.0 / 3.0) * _trace_3d(S2R2)[:, None, None] * I3)

    # T10 = R*S^2*R^2 - R^2*S^2*R
    RS2R2 = np.matmul(RS2, R2)   # (R*S^2)*R^2
    R2S2R = np.matmul(R2S2, R)   # (R^2*S^2)*R
    store(9, RS2R2 - R2S2R)

    return basis


def tbnn_loss(G_pred, b_true, basis):
    """
    TBNN loss function.
    
    Reconstructs anisotropy from predicted coefficients and compares to true values:
        b_pred = sum_n G_n * T_n
    
    Args:
        G_pred: [batch, 4] predicted coefficients
        b_true: [batch, 3] true anisotropy (b_xx, b_xy, b_yy)
        basis: [batch, 4, 3] tensor basis
    
    Returns:
        MSE loss between predicted and true anisotropy
    """
    
    # Reconstruct anisotropy: b = sum(G_n * T_n)
    # basis: [batch, 4, 3], G_pred: [batch, 4]
    # Want: [batch, 3]
    
    b_pred = torch.einsum('bn,bnc->bc', G_pred, basis)
    
    # MSE loss
    loss = torch.mean((b_pred - b_true) ** 2)
    
    return loss


def train_tbnn(model, train_loader, val_loader, device, n_epochs=100, lr=1e-3):
    """Train TBNN model."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nTraining on {device}")
    print(f"Epochs: {n_epochs}, Learning rate: {lr}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for invariants, anisotropy, basis in train_loader:
            invariants = invariants.to(device)
            anisotropy = anisotropy.to(device)
            basis = basis.to(device)
            
            optimizer.zero_grad()
            G_pred = model(invariants)
            loss = tbnn_loss(G_pred, anisotropy, basis)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for invariants, anisotropy, basis in val_loader:
                invariants = invariants.to(device)
                anisotropy = anisotropy.to(device)
                basis = basis.to(device)
                
                G_pred = model(invariants)
                loss = tbnn_loss(G_pred, anisotropy, basis)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    return model


def export_to_cpp(model, feature_means, feature_stds, output_dir):
    """Export trained model to C++ format."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting model to {output_dir}")
    
    # Extract weights from sequential model
    layer_idx = 0
    for module in model.network:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            
            W_file = os.path.join(output_dir, f'layer{layer_idx}_W.txt')
            b_file = os.path.join(output_dir, f'layer{layer_idx}_b.txt')
            
            np.savetxt(W_file, W, fmt='%.16e')
            np.savetxt(b_file, b, fmt='%.16e')
            
            print(f"  Layer {layer_idx}: W {W.shape}, b {b.shape}")
            layer_idx += 1
    
    # Export feature scaling
    means_file = os.path.join(output_dir, 'input_means.txt')
    stds_file = os.path.join(output_dir, 'input_stds.txt')
    
    np.savetxt(means_file, feature_means, fmt='%.16e')
    np.savetxt(stds_file, feature_stds, fmt='%.16e')
    
    print("  Saved feature scaling")
    
    # Create metadata
    metadata = {
        "name": "tbnn_mcconkey",
        "type": "nn_tbnn",
        "description": "TBNN trained on McConkey et al. (2021) dataset",
        "architecture": {
            "layers": [5, 64, 64, 64, 4],
            "activations": ["tanh", "tanh", "tanh", "linear"],
            "total_parameters": sum(p.numel() for p in model.parameters())
        },
        "features": {
            "type": "tbnn_ling2016_2d",
            "inputs": ["lambda_1", "lambda_2", "eta_1", "eta_2", "y_norm"],
            "normalization": "z-score",
            "description": "5 invariants from normalized strain and rotation tensors"
        },
        "reference": {
            "title": "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance",
            "authors": "Ling, J., Kurzawski, A., & Templeton, J.",
            "journal": "Journal of Fluid Mechanics",
            "year": 2016,
            "doi": "10.1017/jfm.2016.615"
        },
        "dataset": {
            "source": "McConkey et al. (2021) - Scientific Data",
            "doi": "10.1038/s41597-021-01034-2",
            "url": "https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset"
        },
        "training": {
            "framework": "PyTorch",
            "optimizer": "Adam",
            "learning_rate": "1e-3 with ReduceLROnPlateau"
        },
        "usage": {
            "command": "./channel --model nn_tbnn --nn_preset tbnn_mcconkey"
        }
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata: {metadata_file}")
    print("\nExport complete!")


def main():
    parser = argparse.ArgumentParser(description='Train TBNN on McConkey dataset')
    parser.add_argument('--data_dir', type=str, default='./mcconkey_data',
                        help='Path to McConkey dataset')
    parser.add_argument('--case', type=str, default='periodic_hills',
                        choices=['periodic_hills', 'square_duct', 'channel'],
                        help='Flow case to train on')
    parser.add_argument('--output', type=str, default='../data/models/tbnn_mcconkey',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[64, 64, 64],
                        help='Hidden layer sizes')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    inv_train, anis_train, basis_train = load_mcconkey_data(
        args.data_dir, args.case, 'train'
    )
    inv_val, anis_val, basis_val = load_mcconkey_data(
        args.data_dir, args.case, 'val'
    )
    
    # Compute feature statistics from training data
    feature_means = np.mean(inv_train, axis=0)
    feature_stds = np.std(inv_train, axis=0) + 1e-8
    
    # Normalize features
    inv_train = (inv_train - feature_means) / feature_stds
    inv_val = (inv_val - feature_means) / feature_stds
    
    # Create datasets
    train_dataset = RANSDataset(inv_train, anis_train, basis_train)
    val_dataset = RANSDataset(inv_val, anis_val, basis_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = TBNNModel(n_inputs=5, n_basis=4, hidden_layers=args.hidden)
    model = model.to(device)
    
    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    model = train_tbnn(model, train_loader, val_loader, device, 
                       n_epochs=args.epochs, lr=args.lr)
    
    # Export to C++
    export_to_cpp(model, feature_means, feature_stds, args.output)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved to: {args.output}")
    print("\nTo use in solver:")
    print(f"  ./channel --model nn_tbnn --nn_preset {Path(args.output).name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


