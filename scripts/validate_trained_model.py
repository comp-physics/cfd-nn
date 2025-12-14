#!/usr/bin/env python3
"""
Validate a trained turbulence model against test data.

This performs "a priori" testing - comparing NN predictions against
DNS/LES ground truth WITHOUT running the full CFD solver.

Usage:
    python validate_trained_model.py \
        --model data/models/tbnn_hills \
        --test_data mcconkey_data/periodic_hills/test/data.npz
"""

import argparse
import numpy as np
import torch
import json
from pathlib import Path


def load_model(model_dir):
    """Load trained model from C++ format files."""
    
    model_dir = Path(model_dir)
    
    # Load metadata
    with open(model_dir / 'metadata.json') as f:
        metadata = json.load(f)
    
    model_type = metadata['type']
    arch = metadata['architecture']['layers']
    
    print(f"Loading model: {metadata['name']}")
    print(f"Type: {model_type}")
    print(f"Architecture: {arch}")
    
    # Build PyTorch model
    layers = []
    layer_idx = 0
    
    while (model_dir / f'layer{layer_idx}_W.txt').exists():
        W = np.loadtxt(model_dir / f'layer{layer_idx}_W.txt')
        b = np.loadtxt(model_dir / f'layer{layer_idx}_b.txt')
        
        in_features = W.shape[1]
        out_features = W.shape[0]
        
        linear = torch.nn.Linear(in_features, out_features)
        linear.weight.data = torch.FloatTensor(W)
        linear.bias.data = torch.FloatTensor(b)
        
        layers.append(linear)
        
        # Add activation (assume tanh for hidden layers, linear for output)
        if layer_idx < len(arch) - 2:  # Not the last layer
            layers.append(torch.nn.Tanh())
        
        layer_idx += 1
    
    model = torch.nn.Sequential(*layers)
    
    # Load feature normalization
    means = np.loadtxt(model_dir / 'input_means.txt')
    stds = np.loadtxt(model_dir / 'input_stds.txt')
    
    return model, means, stds, metadata


def validate_tbnn(model, means, stds, test_data):
    """Validate TBNN model."""
    
    invariants = test_data['invariants']
    anisotropy_true = test_data['anisotropy']
    
    # Handle 3D -> 2D conversion
    if anisotropy_true.shape[1] == 6:
        anisotropy_true = anisotropy_true[:, [0, 1, 3]]  # b_xx, b_xy, b_yy
    
    # Get tensor basis
    if 'basis' in test_data:
        basis = test_data['basis']
        if basis.shape[1] == 10:
            basis = basis[:, :4, :]  # Use first 4 basis functions for 2D
    else:
        print("Warning: No basis in test data, using identity")
        basis = np.zeros((len(invariants), 4, 3))
        basis[:, 0, 0] = 1.0  # T1_xx = 1
        basis[:, 0, 2] = 1.0  # T1_yy = 1
    
    # Normalize features
    invariants_norm = (invariants - means) / stds
    
    # Predict coefficients
    model.eval()
    with torch.no_grad():
        G_pred = model(torch.FloatTensor(invariants_norm)).numpy()
    
    # Reconstruct anisotropy: b = sum(G_n * T_n)
    anisotropy_pred = np.einsum('bn,bnc->bc', G_pred, basis)
    
    # Compute errors
    mse = np.mean((anisotropy_pred - anisotropy_true) ** 2)
    mae = np.mean(np.abs(anisotropy_pred - anisotropy_true))
    
    # Component-wise errors
    component_errors = np.mean((anisotropy_pred - anisotropy_true) ** 2, axis=0)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'component_rmse': {
            'b_xx': np.sqrt(component_errors[0]),
            'b_xy': np.sqrt(component_errors[1]),
            'b_yy': np.sqrt(component_errors[2])
        },
        'predictions': anisotropy_pred,
        'ground_truth': anisotropy_true
    }


def validate_mlp(model, means, stds, test_data):
    """Validate MLP model for scalar nu_t prediction."""
    
    # Extract features
    features = []
    if 'S_mag' in test_data:
        features.append(test_data['S_mag'][:, None])
    if 'Omega_mag' in test_data:
        features.append(test_data['Omega_mag'][:, None])
    if 'wall_distance' in test_data:
        features.append(test_data['wall_distance'][:, None])
    
    # Add more features as available
    if len(features) < 6:
        print("Warning: Missing some features, using available ones")
    
    if not features:
        print("Error: No features found in test data")
        return None
    
    features = np.concatenate(features, axis=1)
    
    # Get ground truth nu_t
    if 'nu_t' in test_data:
        nu_t_true = test_data['nu_t']
    else:
        print("Warning: No nu_t in test data, cannot validate")
        return None
    
    # Normalize and predict
    features_norm = (features - means[:features.shape[1]]) / stds[:features.shape[1]]
    
    model.eval()
    with torch.no_grad():
        nu_t_pred = model(torch.FloatTensor(features_norm)).numpy()
    
    # Compute errors
    mse = np.mean((nu_t_pred - nu_t_true) ** 2)
    mae = np.mean(np.abs(nu_t_pred - nu_t_true))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'predictions': nu_t_pred,
        'ground_truth': nu_t_true
    }


def print_results(results, model_type):
    """Print validation results."""
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Test Samples: {len(results['ground_truth'])}")
    print()
    
    print(f"Mean Squared Error (MSE):  {results['mse']:.6e}")
    print(f"Root Mean Squared Error:   {results['rmse']:.6e}")
    print(f"Mean Absolute Error (MAE): {results['mae']:.6e}")
    
    if 'component_rmse' in results:
        print("\nComponent-wise RMSE:")
        for comp, error in results['component_rmse'].items():
            print(f"  {comp}: {error:.6e}")
    
    # Normalized error
    std_true = np.std(results['ground_truth'])
    normalized_rmse = results['rmse'] / (std_true + 1e-10)
    print(f"\nNormalized RMSE: {normalized_rmse:.3f}")
    
    if normalized_rmse < 0.1:
        print("[EXCELLENT] Excellent agreement with DNS!")
    elif normalized_rmse < 0.3:
        print("[GOOD] Good agreement with DNS")
    elif normalized_rmse < 0.5:
        print("[WARNING] Moderate agreement - consider more training")
    else:
        print("[WARNING] Poor agreement - model may need retraining")
    
    print("\n" + "="*70 + "\n")


def plot_results(results, model_type, output_file='validation_plot.png'):
    """Generate validation plots."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plots")
        return
    
    pred = results['predictions']
    true = results['ground_truth']
    
    if model_type == 'nn_tbnn':
        # Plot each component
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        components = ['b_xx', 'b_xy', 'b_yy']
        
        for idx, (ax, comp) in enumerate(zip(axes, components)):
            ax.scatter(true[:, idx], pred[:, idx], alpha=0.3, s=1)
            
            # Add perfect prediction line
            lim = max(abs(true[:, idx]).max(), abs(pred[:, idx]).max())
            ax.plot([-lim, lim], [-lim, lim], 'r--', label='Perfect prediction')
            
            ax.set_xlabel(f'{comp} (DNS)')
            ax.set_ylabel(f'{comp} (Predicted)')
            ax.set_title(f'{comp} - RMSE: {results["component_rmse"][comp]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    else:  # MLP
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        
        ax.scatter(true, pred, alpha=0.3, s=1)
        
        lim_min = min(true.min(), pred.min())
        lim_max = max(true.max(), pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='Perfect')
        
        ax.set_xlabel('nu_t (DNS)')
        ax.set_ylabel('nu_t (Predicted)')
        ax.set_title(f'Eddy Viscosity Prediction - RMSE: {results["rmse"]:.4e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Validation plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate trained turbulence model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (.npz file)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate validation plots')
    parser.add_argument('--output', type=str, default='validation_plot.png',
                        help='Output plot filename')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"Error: Model directory not found: {args.model}")
        return
    
    if not Path(args.test_data).exists():
        print(f"Error: Test data not found: {args.test_data}")
        print("\nNote: If you don't have real test data, you can still")
        print("run the model in the CFD solver to test integration.")
        return
    
    # Load model
    model, means, stds, metadata = load_model(args.model)
    model_type = metadata['type']
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    test_data = np.load(args.test_data)
    
    # Validate
    print("\nRunning validation...")
    
    if model_type == 'nn_tbnn':
        results = validate_tbnn(model, means, stds, test_data)
    elif model_type == 'nn_mlp':
        results = validate_mlp(model, means, stds, test_data)
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    if results is None:
        print("Validation failed")
        return
    
    # Print results
    print_results(results, model_type)
    
    # Plot if requested
    if args.plot:
        plot_results(results, model_type, args.output)


if __name__ == '__main__':
    main()

