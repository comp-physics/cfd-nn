#!/usr/bin/env python3
"""
Export PyTorch model weights to C++ compatible format.

Usage:
    python export_pytorch.py model.pth --output ../data
"""

import torch
import numpy as np
import argparse
import os


def export_pytorch_model(model_path, output_dir, feature_stats=None):
    """
    Export a PyTorch model to C++ compatible text files.
    
    Args:
        model_path: Path to saved PyTorch model (.pth or .pt)
        output_dir: Directory to save exported weights
        feature_stats: Optional dict with 'means' and 'stds' for input scaling
    """
    # Load model
    print(f"Loading PyTorch model from: {model_path}")
    
    # Try to load as state dict first, then as full model
    try:
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'state_dict' in model:
            state_dict = model['state_dict']
        elif isinstance(model, dict):
            state_dict = model
        else:
            # It's a model object
            state_dict = model.state_dict()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load as a full model...")
        model = torch.load(model_path, map_location='cpu')
        state_dict = model.state_dict()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract linear/dense layers
    layer_idx = 0
    layer_names = sorted([k for k in state_dict.keys() if 'weight' in k])
    
    print(f"\nFound {len(layer_names)} layers")
    
    for weight_key in layer_names:
        # Get weight and bias
        W = state_dict[weight_key].detach().cpu().numpy()
        
        # Find corresponding bias
        bias_key = weight_key.replace('weight', 'bias')
        if bias_key in state_dict:
            b = state_dict[bias_key].detach().cpu().numpy()
        else:
            b = np.zeros(W.shape[0])
            print(f"Warning: No bias found for {weight_key}, using zeros")
        
        # PyTorch stores as (out_features, in_features), which matches our C++ convention
        print(f"Layer {layer_idx}: {W.shape}")
        
        # Save files
        W_file = os.path.join(output_dir, f'layer{layer_idx}_W.txt')
        b_file = os.path.join(output_dir, f'layer{layer_idx}_b.txt')
        
        np.savetxt(W_file, W, fmt='%.16e')
        np.savetxt(b_file, b, fmt='%.16e')
        
        print(f"  Saved: {W_file}")
        print(f"  Saved: {b_file}")
        
        layer_idx += 1
    
    # Export feature scaling if provided
    if feature_stats is not None:
        means = feature_stats.get('means', None)
        stds = feature_stats.get('stds', None)
        
        if means is not None:
            means_file = os.path.join(output_dir, 'input_means.txt')
            if isinstance(means, torch.Tensor):
                means = means.detach().cpu().numpy()
            np.savetxt(means_file, means, fmt='%.16e')
            print(f"\nSaved feature means: {means_file}")
        
        if stds is not None:
            stds_file = os.path.join(output_dir, 'input_stds.txt')
            if isinstance(stds, torch.Tensor):
                stds = stds.detach().cpu().numpy()
            np.savetxt(stds_file, stds, fmt='%.16e')
            print(f"Saved feature stds: {stds_file}")
    
    print(f"\nExport complete! {layer_idx} layers saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to C++ format')
    parser.add_argument('model', type=str, help='Path to PyTorch model (.pth or .pt)')
    parser.add_argument('--output', type=str, default='../data',
                        help='Output directory for weights')
    parser.add_argument('--means', type=str, default=None,
                        help='Optional: path to input feature means (.npy or .txt)')
    parser.add_argument('--stds', type=str, default=None,
                        help='Optional: path to input feature stds (.npy or .txt)')
    
    args = parser.parse_args()
    
    # Load feature statistics if provided
    feature_stats = {}
    if args.means:
        if args.means.endswith('.npy'):
            feature_stats['means'] = np.load(args.means)
        else:
            feature_stats['means'] = np.loadtxt(args.means)
        print(f"Loaded feature means from: {args.means}")
    
    if args.stds:
        if args.stds.endswith('.npy'):
            feature_stats['stds'] = np.load(args.stds)
        else:
            feature_stats['stds'] = np.loadtxt(args.stds)
        print(f"Loaded feature stds from: {args.stds}")
    
    export_pytorch_model(args.model, args.output, feature_stats or None)


if __name__ == '__main__':
    main()

