#!/usr/bin/env python3
"""
Spectral Analysis for 3D Taylor-Green Vortex
=============================================

Computes the energy spectrum E(k) from VTK velocity field data to verify
turbulence development. Uses GPU acceleration via cupy when available.

For fully developed turbulence, we expect:
  - E(k) ~ k^(-5/3) in the inertial subrange (Kolmogorov scaling)
  - Energy injection at low k, dissipation at high k

Usage:
    python spectral_analysis.py <vtk_file_or_dir> [options]

Options:
    --output, -o DIR     Output directory for plots (default: current dir)
    --cpu                Force CPU computation (no cupy)
    --Re FLOAT           Reynolds number for computing scales (default: 1000)
    --batch              Process all VTK files in directory

Examples:
    python spectral_analysis.py output/tg3d_final.vtk
    python spectral_analysis.py output/ --batch --Re 1000
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Try to use cupy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import fft as cufft
    HAS_CUPY = True
    print("[INFO] Using cupy for GPU-accelerated FFT")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("[INFO] cupy not available, using numpy (CPU)")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def read_vtk_legacy(filename, downsample=1):
    """
    Read VTK legacy structured points file (ASCII or BINARY).
    Returns: mesh_dims (Nx, Ny, Nz), spacing (dx, dy, dz), velocity (u, v, w arrays)

    Args:
        filename: Path to VTK file
        downsample: Factor to downsample by (e.g., 2 means 128³ -> 64³)
    """
    import struct

    # First, read header to determine format (ASCII or BINARY)
    with open(filename, 'rb') as f:
        header_lines = []
        is_binary = False
        dims = None
        spacing = None
        n_points = 0

        # Read header lines (ASCII portion)
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)

            if line == 'BINARY':
                is_binary = True
            elif line == 'ASCII':
                is_binary = False
            elif line.startswith('DIMENSIONS'):
                dims = tuple(int(x) for x in line.split()[1:4])
            elif line.startswith('SPACING') or line.startswith('ASPECT_RATIO'):
                spacing = tuple(float(x) for x in line.split()[1:4])
            elif line.startswith('POINT_DATA'):
                n_points = int(line.split()[1])
            elif line.startswith('VECTORS'):
                # Data starts after this line
                break

        if dims is None:
            raise ValueError(f"Could not parse DIMENSIONS from {filename}")

        Nx, Ny, Nz = dims
        expected_values = n_points * 3

        if is_binary:
            # Binary format: big-endian doubles
            print(f"  Reading binary VTK ({Nx}x{Ny}x{Nz}, {expected_values*8/1e6:.1f} MB)")

            # Read all velocity data as big-endian doubles
            raw_data = f.read(expected_values * 8)
            if len(raw_data) < expected_values * 8:
                raise ValueError(f"Expected {expected_values*8} bytes, got {len(raw_data)}")

            # Convert from big-endian to native format
            velocity_data = np.frombuffer(raw_data, dtype='>f8')  # big-endian float64
            velocity_data = velocity_data.astype(np.float32)  # Convert to float32 for memory
        else:
            # ASCII format: read remaining file as text
            print(f"  Reading ASCII VTK ({Nx}x{Ny}x{Nz})")
            remaining = f.read().decode('ascii', errors='ignore')

            # Parse all numbers
            velocity_data = []
            for part in remaining.split():
                try:
                    velocity_data.append(float(part))
                except ValueError:
                    # Skip non-numeric parts (like "SCALARS", "LOOKUP_TABLE", etc.)
                    if len(velocity_data) >= expected_values:
                        break

            velocity_data = np.array(velocity_data[:expected_values], dtype=np.float32)

        if len(velocity_data) < expected_values:
            raise ValueError(f"Expected {expected_values} values, got {len(velocity_data)}")

    # Reshape: VTK uses column-major ordering (Fortran style)
    # Points are ordered x-fastest, then y, then z
    velocity = velocity_data.reshape((Nz, Ny, Nx, 3))

    # Extract components and transpose to row-major (Nx, Ny, Nz)
    u = np.ascontiguousarray(velocity[:, :, :, 0].transpose(2, 1, 0), dtype=np.float32)
    v = np.ascontiguousarray(velocity[:, :, :, 1].transpose(2, 1, 0), dtype=np.float32)
    w = np.ascontiguousarray(velocity[:, :, :, 2].transpose(2, 1, 0), dtype=np.float32)

    # Free the original array to save memory
    del velocity, velocity_data

    return dims, spacing, (u, v, w)


def compute_energy_spectrum(u, v, w, L=2*np.pi, use_gpu=True):
    """
    Compute spherically-averaged energy spectrum E(k).

    Args:
        u, v, w: 3D velocity arrays (Nx, Ny, Nz)
        L: Domain size (assumed cubic)
        use_gpu: Use cupy for GPU acceleration

    Returns:
        k_bins: Wavenumber bin centers
        E_k: Energy spectrum E(k)
    """
    xp = cp if (use_gpu and HAS_CUPY) else np

    Nx, Ny, Nz = u.shape
    assert Nx == Ny == Nz, "Only cubic grids supported"
    N = Nx

    # Memory-efficient: process one component at a time
    # Initialize energy accumulator
    energy = None

    for vel_comp in [u, v, w]:
        if use_gpu and HAS_CUPY:
            vel_gpu = cp.asarray(vel_comp.astype(np.float32))
        else:
            vel_gpu = vel_comp

        # 3D FFT of this component
        vel_hat = xp.fft.fftn(vel_gpu) / N**3

        # Accumulate energy: 0.5 * |vel_hat|^2
        comp_energy = 0.5 * xp.abs(vel_hat)**2

        if energy is None:
            energy = comp_energy
        else:
            energy += comp_energy

        # Free memory
        del vel_hat, vel_gpu
        if use_gpu and HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()

    # Wavenumber grid
    k_max = N // 2
    kx = xp.fft.fftfreq(N, d=L/(2*np.pi*N))
    ky = xp.fft.fftfreq(N, d=L/(2*np.pi*N))
    kz = xp.fft.fftfreq(N, d=L/(2*np.pi*N))

    KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = xp.sqrt(KX**2 + KY**2 + KZ**2)

    # Spherical averaging into shells
    dk = 1.0  # Shell width
    k_bins = xp.arange(0.5, k_max + 0.5, dk)
    E_k = xp.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        shell_mask = (K_mag >= k - dk/2) & (K_mag < k + dk/2)
        E_k[i] = xp.sum(energy[shell_mask])

    # Transfer back to CPU
    if use_gpu and HAS_CUPY:
        k_bins = cp.asnumpy(k_bins)
        E_k = cp.asnumpy(E_k)

    # Normalize: E(k) such that integral E(k) dk = 0.5 * <u^2>
    # The sum of E_k should equal total kinetic energy

    return k_bins, E_k


def compute_turbulence_diagnostics(u, v, w, nu, L=2*np.pi):
    """
    Compute turbulence diagnostics from velocity field.

    Returns dict with:
        - TKE: turbulent kinetic energy
        - dissipation: energy dissipation rate (estimated)
        - Re_lambda: Taylor microscale Reynolds number
        - eta: Kolmogorov length scale
        - lambda_: Taylor microscale
        - k_kolmogorov: Kolmogorov wavenumber
    """
    Nx, Ny, Nz = u.shape
    dx = L / Nx

    # Turbulent kinetic energy
    TKE = 0.5 * np.mean(u**2 + v**2 + w**2)

    # RMS velocity
    u_rms = np.sqrt(2.0/3.0 * TKE)

    # Estimate dissipation from velocity gradients
    # epsilon = 2 * nu * <S_ij S_ij>
    # For isotropic turbulence: epsilon ≈ 15 * nu * <(du/dx)^2>

    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    dwdz = np.gradient(w, dx, axis=2)

    # Simplified estimate using longitudinal derivative
    epsilon = 15.0 * nu * np.mean(dudx**2)

    # Taylor microscale: lambda = sqrt(15 * nu * u'^2 / epsilon)
    if epsilon > 0:
        lambda_taylor = np.sqrt(15.0 * nu * u_rms**2 / epsilon)

        # Taylor Reynolds number: Re_lambda = u' * lambda / nu
        Re_lambda = u_rms * lambda_taylor / nu

        # Kolmogorov length scale: eta = (nu^3 / epsilon)^(1/4)
        eta = (nu**3 / epsilon)**0.25

        # Kolmogorov wavenumber
        k_kolmogorov = 1.0 / eta
    else:
        lambda_taylor = np.nan
        Re_lambda = np.nan
        eta = np.nan
        k_kolmogorov = np.nan

    return {
        'TKE': TKE,
        'u_rms': u_rms,
        'dissipation': epsilon,
        'Re_lambda': Re_lambda,
        'eta': eta,
        'lambda_taylor': lambda_taylor,
        'k_kolmogorov': k_kolmogorov,
    }


def plot_spectrum(k, E_k, diagnostics, output_file, title="Energy Spectrum"):
    """
    Plot energy spectrum with k^(-5/3) reference line.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter out zero/negative values for log plot
    valid = (k > 0) & (E_k > 0)
    k_valid = k[valid]
    E_valid = E_k[valid]

    # Plot spectrum
    ax.loglog(k_valid, E_valid, 'b-', linewidth=2, label='E(k)')

    # k^(-5/3) reference line (Kolmogorov scaling)
    if len(k_valid) > 5:
        # Find a good reference point in the middle of the spectrum
        mid_idx = len(k_valid) // 3
        k_ref = k_valid[mid_idx]
        E_ref = E_valid[mid_idx]

        k_line = np.logspace(np.log10(k_valid[1]), np.log10(k_valid[-1]), 50)
        E_line = E_ref * (k_line / k_ref)**(-5.0/3.0)

        ax.loglog(k_line, E_line, 'r--', linewidth=1.5,
                  label=r'$k^{-5/3}$ (Kolmogorov)')

    # Mark Kolmogorov wavenumber if available
    if not np.isnan(diagnostics['k_kolmogorov']):
        k_eta = diagnostics['k_kolmogorov']
        if k_eta < k_valid[-1]:
            ax.axvline(k_eta, color='g', linestyle=':', linewidth=1.5,
                      label=f'$k_\\eta$ = {k_eta:.1f}')

    ax.set_xlabel('Wavenumber k', fontsize=14)
    ax.set_ylabel('E(k)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', alpha=0.3)

    # Add diagnostics text box
    textstr = '\n'.join([
        f"TKE = {diagnostics['TKE']:.4e}",
        f"$u'$ = {diagnostics['u_rms']:.4e}",
        f"$\\varepsilon$ = {diagnostics['dissipation']:.4e}",
        f"$Re_\\lambda$ = {diagnostics['Re_lambda']:.1f}",
        f"$\\eta$ = {diagnostics['eta']:.4e}",
        f"$\\lambda$ = {diagnostics['lambda_taylor']:.4e}",
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"[OK] Saved spectrum plot to {output_file}")


def analyze_vtk_file(vtk_file, output_dir, Re=1000, use_gpu=True):
    """
    Analyze a single VTK file and generate spectrum plot.
    """
    print(f"\n=== Analyzing {vtk_file} ===")

    # Read VTK
    dims, spacing, (u, v, w) = read_vtk_legacy(vtk_file)
    print(f"  Grid: {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"  Spacing: {spacing}")

    # Assume periodic domain [0, 2π]³ for Taylor-Green
    L = 2 * np.pi
    nu = 1.0 / Re  # Assuming V0 * L = 1

    # Compute spectrum
    print("  Computing energy spectrum...")
    k_bins, E_k = compute_energy_spectrum(u, v, w, L=L, use_gpu=use_gpu)

    # Compute diagnostics
    print("  Computing turbulence diagnostics...")
    diagnostics = compute_turbulence_diagnostics(u, v, w, nu, L=L)

    print(f"  TKE = {diagnostics['TKE']:.6e}")
    print(f"  Re_lambda = {diagnostics['Re_lambda']:.1f}")
    print(f"  Dissipation = {diagnostics['dissipation']:.6e}")

    # Generate plot
    basename = Path(vtk_file).stem
    output_file = Path(output_dir) / f"{basename}_spectrum.png"

    plot_spectrum(k_bins, E_k, diagnostics, output_file,
                  title=f"Energy Spectrum: {basename}")

    # Save spectrum data
    data_file = Path(output_dir) / f"{basename}_spectrum.csv"
    np.savetxt(data_file, np.column_stack([k_bins, E_k]),
               header="k,E_k", delimiter=',', comments='')
    print(f"[OK] Saved spectrum data to {data_file}")

    return k_bins, E_k, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Spectral analysis for 3D Taylor-Green Vortex")
    parser.add_argument('input', help='VTK file or directory')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU computation')
    parser.add_argument('--Re', type=float, default=1000, help='Reynolds number')
    parser.add_argument('--batch', action='store_true',
                        help='Process all VTK files in directory')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    use_gpu = not args.cpu and HAS_CUPY

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        analyze_vtk_file(str(input_path), args.output, Re=args.Re, use_gpu=use_gpu)

    elif input_path.is_dir():
        # Find all VTK files
        vtk_files = sorted(input_path.glob('*.vtk'))

        if not vtk_files:
            print(f"[ERROR] No VTK files found in {input_path}")
            return 1

        if args.batch:
            # Process all files
            for vtk_file in vtk_files:
                try:
                    analyze_vtk_file(str(vtk_file), args.output,
                                   Re=args.Re, use_gpu=use_gpu)
                except Exception as e:
                    print(f"[ERROR] Failed to process {vtk_file}: {e}")
        else:
            # Process only the last (presumably final) file
            print(f"[INFO] Found {len(vtk_files)} VTK files, processing last one")
            print(f"       Use --batch to process all files")
            analyze_vtk_file(str(vtk_files[-1]), args.output,
                           Re=args.Re, use_gpu=use_gpu)
    else:
        print(f"[ERROR] {args.input} not found")
        return 1

    print("\n[DONE] Spectral analysis complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
