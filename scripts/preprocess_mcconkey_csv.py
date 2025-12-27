#!/usr/bin/env python3
"""
Convert McConkey CSV dataset to NPZ format for training scripts.

This processes the CSV files and computes:
- 5 scalar invariants for TBNN
- Tensor basis functions
- Anisotropy tensor from DNS

Usage:
    python preprocess_mcconkey_csv.py --output mcconkey_data_processed
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def compute_tensor_basis_2d(S, Omega, k, epsilon):
    """Compute 2D tensor basis (4 basis tensors for 2D flows)."""
    
    N = len(k)
    
    # Time scale
    T_t = np.divide(k, epsilon + 1e-20)
    
    # Normalized tensors
    Shat = T_t.reshape(-1, 1, 1) * S
    Rhat = T_t.reshape(-1, 1, 1) * Omega
    
    # Initialize basis (4 tensors for 2D, each with xx, xy, yy components)
    basis = np.zeros((N, 4, 3))
    
    # T1 = S (normalized)
    basis[:, 0, 0] = Shat[:, 0, 0]  # xx
    basis[:, 0, 1] = Shat[:, 0, 1]  # xy
    basis[:, 0, 2] = Shat[:, 1, 1]  # yy
    
    # T2 = S*Omega - Omega*S
    SOmega = np.matmul(Shat, Rhat)
    OmegaS = np.matmul(Rhat, Shat)
    T2 = SOmega - OmegaS
    basis[:, 1, 0] = T2[:, 0, 0]
    basis[:, 1, 1] = T2[:, 0, 1]
    basis[:, 1, 2] = T2[:, 1, 1]
    
    # T3 = S^2 - (1/2)*tr(S^2)*I
    S2 = np.matmul(Shat, Shat)
    trS2 = S2[:, 0, 0] + S2[:, 1, 1]
    basis[:, 2, 0] = S2[:, 0, 0] - 0.5 * trS2
    basis[:, 2, 1] = S2[:, 0, 1]
    basis[:, 2, 2] = S2[:, 1, 1] - 0.5 * trS2
    
    # T4 = Omega^2 - (1/2)*tr(Omega^2)*I
    Omega2 = np.matmul(Rhat, Rhat)
    trOmega2 = Omega2[:, 0, 0] + Omega2[:, 1, 1]
    basis[:, 3, 0] = Omega2[:, 0, 0] - 0.5 * trOmega2
    basis[:, 3, 1] = Omega2[:, 0, 1]
    basis[:, 3, 2] = Omega2[:, 1, 1] - 0.5 * trOmega2
    
    return basis


def compute_invariants(S, Omega):
    """Compute 5 scalar invariants."""
    
    N = S.shape[0]
    invariants = np.zeros((N, 5))
    
    # Lambda_1 = tr(S^2)
    S2 = np.matmul(S, S)
    invariants[:, 0] = S2[:, 0, 0] + S2[:, 1, 1] + S2[:, 2, 2]
    
    # Lambda_2 = tr(Omega^2)
    Omega2 = np.matmul(Omega, Omega)
    invariants[:, 1] = Omega2[:, 0, 0] + Omega2[:, 1, 1] + Omega2[:, 2, 2]
    
    # Lambda_3 = tr(S^3)
    S3 = np.matmul(S2, S)
    invariants[:, 2] = S3[:, 0, 0] + S3[:, 1, 1] + S3[:, 2, 2]
    
    # Lambda_4 = tr(S * Omega^2)
    SOmega2 = np.matmul(S, Omega2)
    invariants[:, 3] = SOmega2[:, 0, 0] + SOmega2[:, 1, 1] + SOmega2[:, 2, 2]
    
    # Lambda_5 = tr(S^2 * Omega^2)
    S2Omega2 = np.matmul(S2, Omega2)
    invariants[:, 4] = S2Omega2[:, 0, 0] + S2Omega2[:, 1, 1] + S2Omega2[:, 2, 2]
    
    return invariants


def assemble_tensor(df, tensor_name):
    """Assemble 3x3 tensor from dataframe columns."""
    
    N = len(df)
    tensor = np.zeros((N, 3, 3))
    
    # Upper diagonal
    tensor[:, 0, 0] = df[f'{tensor_name}_11']
    tensor[:, 0, 1] = df[f'{tensor_name}_12']
    tensor[:, 0, 2] = df[f'{tensor_name}_13']
    tensor[:, 1, 1] = df[f'{tensor_name}_22']
    tensor[:, 1, 2] = df[f'{tensor_name}_23']
    tensor[:, 2, 2] = df[f'{tensor_name}_33']
    
    # Check if symmetric
    if f'{tensor_name}_21' in df.columns:
        tensor[:, 1, 0] = df[f'{tensor_name}_21']
        tensor[:, 2, 0] = df[f'{tensor_name}_31']
        tensor[:, 2, 1] = df[f'{tensor_name}_32']
    else:
        tensor[:, 1, 0] = tensor[:, 0, 1]
        tensor[:, 2, 0] = tensor[:, 0, 2]
        tensor[:, 2, 1] = tensor[:, 1, 2]
    
    return tensor


def process_case(df_komega, df_ref, case_pattern, output_file):
    """Process a single case and save to NPZ."""
    
    # Filter for this case
    df = df_komega[df_komega['Case'].str.contains(case_pattern, na=False)].copy()
    df_ref_case = df_ref[df_ref['Case'].str.contains(case_pattern, na=False)].copy()
    
    if len(df) == 0:
        print(f"  No data found for pattern: {case_pattern}")
        return False
    
    print(f"  Found {len(df)} samples")
    
    # Get tensors from RANS
    S = assemble_tensor(df, 'komega_S')
    R = assemble_tensor(df, 'komega_R')
    
    # Get scalars
    k = df['komega_k'].values
    epsilon = df['komega_epsilon'].values
    
    # Compute invariants
    invariants = compute_invariants(S, R)
    
    # Compute tensor basis
    basis = compute_tensor_basis_2d(S, R, k, epsilon)
    
    # Get anisotropy from DNS
    anisotropy = np.zeros((len(df), 3))
    if len(df_ref_case) == len(df):
        anisotropy[:, 0] = df_ref_case['REF_b_11'].values
        anisotropy[:, 1] = df_ref_case['REF_b_12'].values
        anisotropy[:, 2] = df_ref_case['REF_b_22'].values
    else:
        print("  WARNING: DNS data size mismatch, using RANS as placeholder")
        # Use RANS anisotropy if DNS not available
        tau = assemble_tensor(df, 'komega_tau')
        k_full = k.reshape(-1, 1, 1)
        b = tau / (2 * k_full + 1e-20) - np.eye(3).reshape(1, 3, 3) / 3
        anisotropy[:, 0] = b[:, 0, 0]
        anisotropy[:, 1] = b[:, 0, 1]
        anisotropy[:, 2] = b[:, 1, 1]
    
    # Save to NPZ
    np.savez(output_file,
             invariants=invariants.astype(np.float32),
             anisotropy=anisotropy.astype(np.float32),
             basis=basis.astype(np.float32),
             k=k.astype(np.float32),
             epsilon=epsilon.astype(np.float32))
    
    print(f"  Saved to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess McConkey CSV to NPZ')
    parser.add_argument('--data_dir', type=str, default='mcconkey_data',
                        help='Directory containing CSV files')
    parser.add_argument('--output', type=str, default='mcconkey_data_processed',
                        help='Output directory for NPZ files')
    
    args = parser.parse_args()
    
    print("Loading CSV files...")
    df_komega = pd.read_csv(Path(args.data_dir) / 'komega.csv')
    df_ref = pd.read_csv(Path(args.data_dir) / 'REF.csv')
    
    print(f"Total samples: {len(df_komega)}")
    print(f"Cases: {df_komega['Case'].unique()}")
    
    # Create output directories
    output_dir = Path(args.output)
    (output_dir / 'channel' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'periodic_hills' / 'train').mkdir(parents=True, exist_ok=True)
    
    # Process flat plate (use as "channel-like" flow)
    print("\nProcessing flat plate cases...")
    fp_cases = [c for c in df_komega['Case'].unique() if c.startswith('fp_')]
    if fp_cases:
        # Use all flat plate data as training
        process_case(df_komega, df_ref, 'fp_',
                    output_dir / 'channel' / 'train' / 'data.npz')
    
    # Process periodic hills
    print("\nProcessing periodic hills cases...")
    ph_cases = [c for c in df_komega['Case'].unique() if c.startswith('case_')]
    if ph_cases:
        process_case(df_komega, df_ref, 'case_',
                    output_dir / 'periodic_hills' / 'train' / 'data.npz')
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

