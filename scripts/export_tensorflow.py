#!/usr/bin/env python3
"""
Export TensorFlow/Keras model weights to C++ compatible format.

Usage:
    python export_tensorflow.py model.h5 --output ../data
"""

import numpy as np
import argparse
import os


def export_tensorflow_model(model_path, output_dir, feature_stats=None):
    """
    Export a TensorFlow/Keras model to C++ compatible text files.
    
    Args:
        model_path: Path to saved model (.h5, SavedModel directory, etc.)
        output_dir: Directory to save exported weights
        feature_stats: Optional dict with 'means' and 'stds' for input scaling
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed. Install with: pip install tensorflow")
        return
    
    print(f"Loading TensorFlow model from: {model_path}")
    
    # Load model
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:
        # Try loading as SavedModel
        model = tf.keras.models.load_model(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract Dense layers
    layer_idx = 0
    print(f"\nModel architecture:")
    model.summary()
    
    print(f"\nExporting weights...")
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            
            if len(weights) == 2:
                W, b = weights
            elif len(weights) == 1:
                W = weights[0]
                b = np.zeros(W.shape[1])
                print(f"Warning: Layer {layer.name} has no bias, using zeros")
            else:
                print(f"Warning: Unexpected number of weights in {layer.name}")
                continue
            
            # TensorFlow stores as (in_features, out_features)
            # Transpose to match C++ convention (out_features, in_features)
            W = W.T
            
            print(f"Layer {layer_idx} ({layer.name}): {W.shape}")
            
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
            np.savetxt(means_file, means, fmt='%.16e')
            print(f"\nSaved feature means: {means_file}")
        
        if stds is not None:
            stds_file = os.path.join(output_dir, 'input_stds.txt')
            np.savetxt(stds_file, stds, fmt='%.16e')
            print(f"Saved feature stds: {stds_file}")
    
    print(f"\nExport complete! {layer_idx} layers saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Export TensorFlow model to C++ format')
    parser.add_argument('model', type=str, 
                        help='Path to TensorFlow model (.h5 or SavedModel directory)')
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
    
    export_tensorflow_model(args.model, args.output, feature_stats or None)


if __name__ == '__main__':
    main()

