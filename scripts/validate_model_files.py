#!/usr/bin/env python3
"""
Validate all neural network model files for correctness.

This script checks:
1. Weights and biases are finite (no NaN/Inf)
2. Normalization statistics are valid
3. File structure is complete
4. Metadata is consistent

Usage:
    python scripts/validate_model_files.py [--model MODEL_DIR]
    
    If --model is not specified, validates all models in data/models/
"""

import argparse
import numpy as np
import json
from pathlib import Path
import sys


def check_finite(arr, name, filename):
    """Check if array contains only finite values."""
    issues = []
    
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        
        # Find first problematic index
        bad_indices = np.where(~np.isfinite(arr))[0]
        first_bad = bad_indices[0] if len(bad_indices) > 0 else -1
        
        issues.append(
            f"{name} in {filename}: contains {nan_count} NaN and {inf_count} Inf values "
            f"(first at index {first_bad})"
        )
    
    return issues


def validate_model_directory(model_dir, verbose=True):
    """
    Validate a single model directory.
    
    Returns:
        (is_valid, issues) tuple
    """
    model_dir = Path(model_dir)
    issues = []
    
    if verbose:
        print(f"\nValidating: {model_dir.name}")
        print("-" * 70)
    
    # Check metadata
    metadata_file = model_dir / 'metadata.json'
    if not metadata_file.exists():
        issues.append("Missing metadata.json")
    else:
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            if verbose:
                print(f"  Model type: {metadata.get('type', 'unknown')}")
                print(f"  Architecture: {metadata.get('architecture', {}).get('layers', 'unknown')}")
        except Exception as e:
            issues.append(f"Invalid metadata.json: {e}")
    
    # Check normalization files
    means_file = model_dir / 'input_means.txt'
    stds_file = model_dir / 'input_stds.txt'
    
    if means_file.exists() and stds_file.exists():
        try:
            means = np.loadtxt(means_file)
            stds = np.loadtxt(stds_file)
            
            # Check finite
            issues.extend(check_finite(means, "input_means", means_file.name))
            issues.extend(check_finite(stds, "input_stds", stds_file.name))
            
            # Check positive stds
            if np.any(stds <= 0):
                bad_indices = np.where(stds <= 0)[0]
                issues.append(
                    f"input_stds contains non-positive values at indices: {bad_indices.tolist()}"
                )
            
            # Check extreme values
            if np.any(np.abs(means) > 1e10):
                issues.append(f"input_means contains extreme values (max: {np.max(np.abs(means)):.2e})")
            if np.any(stds > 1e10):
                issues.append(f"input_stds contains extreme values (max: {np.max(stds):.2e})")
            
            if verbose and not issues:
                print(f"  ✓ Normalization stats valid ({len(means)} features)")
        except Exception as e:
            issues.append(f"Error loading normalization files: {e}")
    else:
        if verbose:
            print(f"  ⚠ No normalization files (optional)")
    
    # Check weight/bias files
    layer_idx = 0
    total_params = 0
    
    while True:
        W_file = model_dir / f'layer{layer_idx}_W.txt'
        b_file = model_dir / f'layer{layer_idx}_b.txt'
        
        if not W_file.exists() or not b_file.exists():
            break
        
        try:
            W = np.loadtxt(W_file)
            b = np.loadtxt(b_file)
            
            # Check finite
            issues.extend(check_finite(W, f"layer{layer_idx}_W", W_file.name))
            issues.extend(check_finite(b, f"layer{layer_idx}_b", b_file.name))
            
            # Count parameters
            W_params = W.size
            b_params = b.size
            total_params += W_params + b_params
            
            if verbose and not any(f"layer{layer_idx}" in issue for issue in issues):
                print(f"  ✓ Layer {layer_idx}: W{W.shape} + b{b.shape} = {W_params + b_params} params")
        
        except Exception as e:
            issues.append(f"Error loading layer {layer_idx}: {e}")
        
        layer_idx += 1
    
    if layer_idx == 0:
        issues.append("No weight files found (layer0_W.txt, layer0_b.txt)")
    elif verbose:
        print(f"  ✓ Total: {layer_idx} layers, {total_params:,} parameters")
    
    # Summary
    if verbose:
        if issues:
            print(f"\n  ✗ FAILED: {len(issues)} issue(s) found")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  ✓ PASSED: All checks OK")
    
    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(
        description='Validate neural network model files'
    )
    parser.add_argument('--model', type=str,
                        help='Path to specific model directory (default: validate all)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print errors')
    
    args = parser.parse_args()
    
    if args.model:
        # Validate single model
        model_dirs = [Path(args.model)]
    else:
        # Validate all models in data/models/
        repo_root = Path(__file__).parent.parent
        models_dir = repo_root / 'data' / 'models'
        
        if not models_dir.exists():
            print(f"ERROR: Models directory not found: {models_dir}")
            return 1
        
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not model_dirs:
            print(f"No model directories found in {models_dir}")
            return 1
    
    print("="*70)
    print("NEURAL NETWORK MODEL VALIDATION")
    print("="*70)
    
    all_valid = True
    results = {}
    
    for model_dir in sorted(model_dirs):
        if not model_dir.exists():
            print(f"\nERROR: Model directory not found: {model_dir}")
            all_valid = False
            continue
        
        is_valid, issues = validate_model_directory(model_dir, verbose=not args.quiet)
        results[model_dir.name] = (is_valid, issues)
        
        if not is_valid:
            all_valid = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for model_name, (is_valid, issues) in sorted(results.items()):
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status}: {model_name}")
        if not is_valid and args.quiet:
            for issue in issues:
                print(f"    - {issue}")
    
    print("="*70)
    
    if all_valid:
        print("✓ All models validated successfully!")
        return 0
    else:
        print("✗ Some models have issues - see above for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())

