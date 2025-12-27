#!/usr/bin/env python3
"""
Generate dummy neural network weights for testing the C++ inference pipeline.

This creates random weights for a simple MLP that can be used to verify
the NN turbulence models are loading and running correctly.
"""

import numpy as np
import os
import argparse


def generate_mlp_weights(layer_dims, output_dir, seed=42):
    """
    Generate random weights for an MLP.
    
    Args:
        layer_dims: List of layer dimensions, e.g., [6, 32, 32, 1]
        output_dir: Directory to save weight files
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating MLP weights with architecture: {layer_dims}")
    
    for i in range(len(layer_dims) - 1):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        
        # Xavier/Glorot initialization with smaller scale for stability
        limit = np.sqrt(6.0 / (in_dim + out_dim)) * 0.1
        W = np.random.uniform(-limit, limit, size=(out_dim, in_dim))
        b = np.zeros(out_dim)
        
        # For the output layer, make sure it produces reasonable nu_t values
        if i == len(layer_dims) - 2:  # Last layer
            # Scale down output to produce nu_t ~ 0.01 - 0.1
            W = W * 0.01
            b = b * 0.01
        
        # Save in space-separated format
        W_file = os.path.join(output_dir, f'layer{i}_W.txt')
        b_file = os.path.join(output_dir, f'layer{i}_b.txt')
        
        np.savetxt(W_file, W, fmt='%.16e')
        np.savetxt(b_file, b, fmt='%.16e')
        
        print(f"  Layer {i}: [{in_dim} -> {out_dim}]")
        print(f"    Saved: {W_file}")
        print(f"    Saved: {b_file}")


def generate_scaling_params(num_features, output_dir, seed=42):
    """
    Generate dummy feature scaling parameters.
    
    Args:
        num_features: Number of input features
        output_dir: Directory to save scaling files
        seed: Random seed
    """
    np.random.seed(seed + 1)
    
    # Means around 0, stds around 1
    means = np.random.randn(num_features) * 0.5
    stds = np.random.uniform(0.5, 2.0, num_features)
    
    means_file = os.path.join(output_dir, 'input_means.txt')
    stds_file = os.path.join(output_dir, 'input_stds.txt')
    
    np.savetxt(means_file, means, fmt='%.16e')
    np.savetxt(stds_file, stds, fmt='%.16e')
    
    print("\nScaling parameters:")
    print(f"  Saved: {means_file}")
    print(f"  Saved: {stds_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate dummy NN weights for testing')
    parser.add_argument('--model', type=str, default='mlp', 
                        choices=['mlp', 'tbnn'],
                        help='Model type: mlp (scalar nu_t) or tbnn (anisotropy)')
    parser.add_argument('--output', type=str, default='../data',
                        help='Output directory for weights')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.model == 'mlp':
        # Scalar eddy viscosity model
        # Input: 6 features (see features.cpp:compute_features_scalar_nut)
        # Output: 1 value (nu_t)
        # Architecture: 6 -> 32 -> 32 -> 1
        layer_dims = [6, 32, 32, 1]
        num_features = 6
        
    elif args.model == 'tbnn':
        # TBNN model
        # Input: 5 features (invariants)
        # Output: 4 coefficients (for 4 tensor basis functions in 2D)
        # Architecture: 5 -> 64 -> 64 -> 64 -> 4
        layer_dims = [5, 64, 64, 64, 4]
        num_features = 5
    
    print("="*60)
    print(f"Generating dummy {args.model.upper()} weights")
    print("="*60)
    
    generate_mlp_weights(layer_dims, args.output, args.seed)
    generate_scaling_params(num_features, args.output, args.seed)
    
    print("\n" + "="*60)
    print("Done! Weight files generated successfully.")
    print("="*60)
    print("\nTest with:")
    print(f"  ./channel --model nn_{args.model} --weights {args.output}")


if __name__ == '__main__':
    main()

