#!/usr/bin/env python3
"""
Train simple MLP model for scalar eddy viscosity prediction on McConkey dataset.

This is a simpler alternative to TBNN - directly predicts nu_t from flow features.

Usage:
    python train_mlp_mcconkey.py --data_dir /path/to/mcconkey/data \
        --case periodic_hills --output data/models/mlp_mcconkey_hills
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


class MLPModel(nn.Module):
    """
    Simple MLP for scalar eddy viscosity prediction.
    
    Architecture:
        Input: 6 features (S, Omega, y/delta, etc.)
        Hidden: 2-3 layers of 32-64 neurons
        Output: 1 (nu_t)
    """
    
    def __init__(self, n_inputs=6, hidden_layers=[32, 32]):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = n_inputs
        
        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.Tanh())
            prev_size = h_size
        
        # Output layer with ReLU to ensure positive nu_t
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.ReLU())  # Ensure nu_t >= 0
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


class NuTDataset(Dataset):
    """Dataset for scalar eddy viscosity prediction."""
    
    def __init__(self, features, nu_t):
        self.features = torch.FloatTensor(features)
        self.nu_t = torch.FloatTensor(nu_t)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.nu_t[idx]


def load_mcconkey_data_mlp(data_dir, case='periodic_hills', split='train'):
    """
    Load data for MLP training.
    
    Returns:
        features: [N, 6] flow features
        nu_t: [N] eddy viscosity values
    """
    
    print(f"Loading MLP data from {data_dir}")
    print(f"Case: {case}, Split: {split}")
    
    case_dir = Path(data_dir) / case / split
    
    if not case_dir.exists():
        print(f"\nWARNING: Dataset directory not found: {case_dir}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data_mlp(n_samples=10000)
    
    try:
        data = np.load(case_dir / 'data.npz')
        
        # Extract features
        # Standard features: S_mag, Omega_mag, y_norm, k, omega, Re_local
        features = []
        
        if 'S_mag' in data:
            features.append(data['S_mag'][:, None])
        if 'Omega_mag' in data:
            features.append(data['Omega_mag'][:, None])
        if 'wall_distance' in data:
            features.append(data['wall_distance'][:, None])
        if 'k' in data:
            features.append(data['k'][:, None])
        if 'omega' in data:
            features.append(data['omega'][:, None])
        if 'velocity_mag' in data:
            features.append(data['velocity_mag'][:, None])
        
        features = np.concatenate(features, axis=1)
        
        # Target: nu_t (compute from Reynolds stresses if not directly available)
        if 'nu_t' in data:
            nu_t = data['nu_t']
        elif 'tau_xy' in data and 'S_xy' in data:
            # nu_t ~ -tau_xy / (2 * S_xy) for shear flow
            nu_t = np.abs(-data['tau_xy'] / (2 * data['S_xy'] + 1e-10))
            nu_t = np.clip(nu_t, 0, 1.0)  # Reasonable bounds
        else:
            raise ValueError("Cannot find nu_t or compute from Reynolds stresses")
        
        print(f"Loaded {len(features)} samples")
        print(f"  Features shape: {features.shape}")
        print(f"  nu_t range: [{nu_t.min():.6f}, {nu_t.max():.6f}]")
        
        return features, nu_t
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data_mlp(n_samples=10000)


def generate_synthetic_data_mlp(n_samples=10000):
    """Generate synthetic data for testing."""
    
    print(f"\nGenerating {n_samples} synthetic samples for MLP...")
    print("NOTE: This is DUMMY DATA for testing - not real DNS data!")
    
    # Generate random features
    features = np.random.randn(n_samples, 6)
    features[:, 0] = np.abs(features[:, 0]) * 0.5  # S_mag > 0
    features[:, 1] = np.abs(features[:, 1]) * 0.3  # Omega_mag > 0
    features[:, 2] = np.random.rand(n_samples) * 0.5  # y_norm
    features[:, 3] = np.abs(features[:, 3]) * 0.1  # k
    features[:, 4] = np.abs(features[:, 4]) * 0.1  # omega
    features[:, 5] = np.abs(features[:, 5]) * 0.5  # u_mag
    
    # Synthetic nu_t (random but physically reasonable)
    # Simple formula: nu_t ~ kappa * y * |S| for mixing length
    kappa = 0.41
    y = features[:, 2]
    S = features[:, 0]
    nu_t = kappa * y * S * (1 - np.exp(-y * 50))  # Van Driest damping
    nu_t = np.clip(nu_t, 0, 0.5)
    
    return features, nu_t


def train_mlp(model, train_loader, val_loader, device, n_epochs=100, lr=1e-3):
    """Train MLP model."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
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
        
        for features, nu_t in train_loader:
            features = features.to(device)
            nu_t = nu_t.to(device)
            
            optimizer.zero_grad()
            nu_t_pred = model(features)
            loss = criterion(nu_t_pred, nu_t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, nu_t in val_loader:
                features = features.to(device)
                nu_t = nu_t.to(device)
                
                nu_t_pred = model(features)
                loss = criterion(nu_t_pred, nu_t)
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
    
    # Extract weights
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
    np.savetxt(os.path.join(output_dir, 'input_means.txt'), feature_means, fmt='%.16e')
    np.savetxt(os.path.join(output_dir, 'input_stds.txt'), feature_stds, fmt='%.16e')
    
    # Create metadata
    metadata = {
        "name": "mlp_mcconkey",
        "type": "nn_mlp",
        "description": "MLP for scalar eddy viscosity trained on McConkey dataset",
        "architecture": {
            "layers": [6, 32, 32, 1],
            "activations": ["tanh", "tanh", "relu"],
            "total_parameters": sum(p.numel() for p in model.parameters())
        },
        "features": {
            "type": "scalar_nut_v1",
            "inputs": ["S_mag", "Omega_mag", "y_norm", "k", "omega", "u_mag"],
            "normalization": "z-score"
        },
        "dataset": {
            "source": "McConkey et al. (2021)",
            "doi": "10.1038/s41597-021-01034-2"
        },
        "usage": {
            "command": "./channel --model nn_mlp --nn_preset mlp_mcconkey"
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Export complete!")


def main():
    parser = argparse.ArgumentParser(description='Train MLP on McConkey dataset')
    parser.add_argument('--data_dir', type=str, default='./mcconkey_data')
    parser.add_argument('--case', type=str, default='periodic_hills')
    parser.add_argument('--output', type=str, default='../data/models/mlp_mcconkey')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--device', type=str, default='auto')
    
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
    feat_train, nut_train = load_mcconkey_data_mlp(args.data_dir, args.case, 'train')
    feat_val, nut_val = load_mcconkey_data_mlp(args.data_dir, args.case, 'val')
    
    # Normalize features
    feature_means = np.mean(feat_train, axis=0)
    feature_stds = np.std(feat_train, axis=0) + 1e-8
    
    feat_train = (feat_train - feature_means) / feature_stds
    feat_val = (feat_val - feature_means) / feature_stds
    
    # Create datasets
    train_dataset = NuTDataset(feat_train, nut_train)
    val_dataset = NuTDataset(feat_val, nut_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = MLPModel(n_inputs=feat_train.shape[1], hidden_layers=args.hidden)
    model = model.to(device)
    
    print(f"\nModel: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    model = train_mlp(model, train_loader, val_loader, device, args.epochs, args.lr)
    
    # Export
    export_to_cpp(model, feature_means, feature_stds, args.output)
    
    print(f"\n{'='*60}")
    print(f"Model saved to: {args.output}")
    print(f"To use: ./channel --model nn_mlp --nn_preset {Path(args.output).name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


