#!/usr/bin/env python3
"""
Fix invalid normalization statistics in trained models.

This script regenerates input_means.txt and input_stds.txt for models
that have corrupted or invalid normalization values (inf, NaN, or extreme values).

For TBNN models, reasonable defaults based on typical turbulent flow features:
- lambda_1, lambda_2: ~O(0.1-1)
- eta_1, eta_2: ~O(0.01-0.1)  
- y_norm: ~O(0-1)

Usage:
    python scripts/fix_normalization_stats.py --model data/models/tbnn_channel_caseholdout
"""

import argparse
import numpy as np
from pathlib import Path


def check_stats_validity(means, stds):
    """Check if normalization stats are valid."""
    issues = []
    
    # Check for inf/nan
    if np.any(~np.isfinite(means)):
        issues.append("means contain inf/NaN")
    if np.any(~np.isfinite(stds)):
        issues.append("stds contain inf/NaN")
    
    # Check for extreme values (likely errors)
    if np.any(np.abs(means) > 1e10):
        issues.append(f"means contain extreme values (max: {np.max(np.abs(means)):.2e})")
    if np.any(np.abs(stds) > 1e10):
        issues.append(f"stds contain extreme values (max: {np.max(stds):.2e})")
    
    # Check for zero/negative stds
    if np.any(stds <= 0):
        issues.append("stds contain zero or negative values")
    
    return issues


def generate_reasonable_tbnn_stats():
    """
    Generate reasonable normalization statistics for TBNN features.
    
    Based on typical values from turbulent channel flow:
    - lambda_1 = S_norm^2 ~ O(0.1-1)
    - lambda_2 = Omega_norm^2 ~ O(0.01-0.5)
    - eta_1 = tr(S^2) ~ O(0.1-1)
    - eta_2 = tr(Omega^2) ~ O(0.01-0.5)
    - y_norm = y/delta ~ O(0-1)
    
    Using conservative estimates that won't over-normalize:
    """
    means = np.array([
        0.15,   # lambda_1 (S^2)
        0.05,   # lambda_2 (Omega^2)
        0.15,   # eta_1 (tr(S^2))
        0.05,   # eta_2 (tr(Omega^2))
        0.5     # y_norm (wall distance)
    ])
    
    stds = np.array([
        0.2,    # lambda_1 std
        0.1,    # lambda_2 std
        0.2,    # eta_1 std
        0.1,    # eta_2 std
        0.3     # y_norm std
    ])
    
    return means, stds


def disable_normalization(n_features):
    """
    Generate identity normalization (no scaling).
    
    This sets mean=0, std=1 so that (x - mean) / std = x.
    """
    means = np.zeros(n_features)
    stds = np.ones(n_features)
    return means, stds


def fix_model_normalization(model_dir, mode='reasonable', backup=True):
    """
    Fix normalization statistics for a model.
    
    Args:
        model_dir: Path to model directory
        mode: 'reasonable' (use typical values) or 'disable' (identity transform)
        backup: Whether to backup original files
    """
    model_dir = Path(model_dir)
    
    means_file = model_dir / 'input_means.txt'
    stds_file = model_dir / 'input_stds.txt'
    
    if not means_file.exists() or not stds_file.exists():
        print(f"ERROR: Normalization files not found in {model_dir}")
        return False
    
    # Load current stats
    print(f"\nChecking model: {model_dir.name}")
    means = np.loadtxt(means_file)
    stds = np.loadtxt(stds_file)
    
    print(f"Current means: {means}")
    print(f"Current stds:  {stds}")
    
    # Check validity
    issues = check_stats_validity(means, stds)
    
    if not issues:
        print("✓ Normalization stats are valid - no fix needed")
        return True
    
    print("\n⚠ Issues detected:")
    for issue in issues:
        print(f"  - {issue}")
    
    # Backup original files
    if backup:
        backup_dir = model_dir / 'backup_original_stats'
        backup_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(means_file, backup_dir / 'input_means.txt')
        shutil.copy(stds_file, backup_dir / 'input_stds.txt')
        print(f"\n✓ Backed up original files to: {backup_dir}")
    
    # Generate new stats
    n_features = len(means)
    
    if mode == 'reasonable':
        if n_features == 5:
            new_means, new_stds = generate_reasonable_tbnn_stats()
            print(f"\n✓ Generated reasonable TBNN normalization stats")
        else:
            print(f"\n⚠ Unknown feature count ({n_features}), using identity normalization")
            new_means, new_stds = disable_normalization(n_features)
    else:  # disable
        new_means, new_stds = disable_normalization(n_features)
        print(f"\n✓ Generated identity normalization (disabled)")
    
    print(f"New means: {new_means}")
    print(f"New stds:  {new_stds}")
    
    # Save new stats
    np.savetxt(means_file, new_means, fmt='%.16e')
    np.savetxt(stds_file, new_stds, fmt='%.16e')
    
    print(f"\n✓ Updated normalization files")
    
    # Verify
    verify_means = np.loadtxt(means_file)
    verify_stds = np.loadtxt(stds_file)
    verify_issues = check_stats_validity(verify_means, verify_stds)
    
    if verify_issues:
        print(f"\n✗ ERROR: New stats still have issues: {verify_issues}")
        return False
    else:
        print(f"✓ Verified: New stats are valid")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix invalid normalization statistics in trained models'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory')
    parser.add_argument('--mode', type=str, default='reasonable',
                        choices=['reasonable', 'disable'],
                        help='Fix mode: reasonable (typical values) or disable (identity)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not backup original files')
    
    args = parser.parse_args()
    
    success = fix_model_normalization(
        args.model,
        mode=args.mode,
        backup=not args.no_backup
    )
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS: Model normalization fixed!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Test the model: ./channel --model nn_tbnn --weights", args.model)
        print("  2. If results look wrong, try --mode disable to turn off normalization")
        print("  3. Consider retraining the model with correct normalization")
    else:
        print("\n" + "="*70)
        print("FAILED: Could not fix normalization")
        print("="*70)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

