#!/usr/bin/env python3
"""
Flow field visualization for the NN turbulence closure paper.

Reads VTK files (ASCII or binary, 2D or 3D) produced by the solver and
generates publication-quality contour plots of vorticity, velocity, pressure,
and eddy viscosity. Supports IBM body overlays for cylinder, sphere, and hills.

Usage:
    # Single VTK file — all fields
    python plot_flow_fields.py path/to/flow_final.vtk

    # Specific field
    python plot_flow_fields.py flow_final.vtk --field vorticity

    # Side-by-side model comparison (2-6 VTK files)
    python plot_flow_fields.py baseline.vtk sst.vtk mlp.vtk --field vorticity

    # 3D slice at z=0
    python plot_flow_fields.py duct_final.vtk --slice z=0

    # Specify geometry for body overlay
    python plot_flow_fields.py cyl.vtk --body cylinder --field vorticity

    # Output directory
    python plot_flow_fields.py flow.vtk -o figures/

Dependencies: numpy, matplotlib (no pyvista/vtk needed).
"""
import sys
import os
import struct
import argparse
import numpy as np

# Add scripts/paper to path for plot_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (apply_style, COLORS, SINGLE_COL, DOUBLE_COL, GOLDEN,
                        save_fig, double_col_fig)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.patheffects as pe


# ============================================================================
# VTK Parser (pure numpy, handles ASCII and binary legacy structured points)
# ============================================================================

def read_vtk(filepath):
    """Parse a VTK legacy STRUCTURED_POINTS file (ASCII or binary).

    Returns:
        dict with keys:
            'dims': (Nx, Ny, Nz)
            'origin': (x0, y0, z0)
            'spacing': (dx, dy, dz)
            'fields': {name: {'data': ndarray, 'ncomp': int, 'type': str}}
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Parse header (always ASCII)
    lines = raw.split(b'\n')
    header_end = 0
    dims = origin = spacing = None
    x_coords = y_coords = z_coords = None
    n_points = 0
    is_binary = False
    is_rectilinear = False

    i = 0
    while i < len(lines):
        text = lines[i].decode('ascii', errors='replace').strip()
        if text.startswith('BINARY'):
            is_binary = True
        elif text.startswith('ASCII'):
            is_binary = False
        elif text.startswith('DATASET RECTILINEAR_GRID'):
            is_rectilinear = True
        elif text.startswith('DIMENSIONS'):
            dims = tuple(int(x) for x in text.split()[1:4])
        elif text.startswith('ORIGIN'):
            origin = tuple(float(x) for x in text.split()[1:4])
        elif text.startswith('SPACING'):
            spacing = tuple(float(x) for x in text.split()[1:4])
        elif text.startswith('X_COORDINATES'):
            n_x = int(text.split()[1])
            vals = []
            i += 1
            while len(vals) < n_x and i < len(lines):
                t2 = lines[i].decode('ascii', errors='replace').strip()
                if t2 and not t2[0].isalpha():
                    vals.extend(float(v) for v in t2.split())
                    i += 1
                else:
                    break
            x_coords = np.array(vals[:n_x])
            continue
        elif text.startswith('Y_COORDINATES'):
            n_y = int(text.split()[1])
            vals = []
            i += 1
            while len(vals) < n_y and i < len(lines):
                t2 = lines[i].decode('ascii', errors='replace').strip()
                if t2 and not t2[0].isalpha():
                    vals.extend(float(v) for v in t2.split())
                    i += 1
                else:
                    break
            y_coords = np.array(vals[:n_y])
            continue
        elif text.startswith('Z_COORDINATES'):
            n_z = int(text.split()[1])
            vals = []
            i += 1
            while len(vals) < n_z and i < len(lines):
                t2 = lines[i].decode('ascii', errors='replace').strip()
                if t2 and not t2[0].isalpha():
                    vals.extend(float(v) for v in t2.split())
                    i += 1
                else:
                    break
            z_coords = np.array(vals[:n_z])
            continue
        elif text.startswith('POINT_DATA'):
            n_points = int(text.split()[1])
            header_end = i + 1
            break
        i += 1

    if dims is None:
        raise ValueError(f"Could not parse DIMENSIONS from {filepath}")

    result = {
        'dims': dims,
        'origin': origin,
        'spacing': spacing,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'z_coords': z_coords,
        'fields': {},
    }

    if is_binary:
        _parse_binary_fields(raw, lines, header_end, n_points, result)
    else:
        _parse_ascii_fields(lines, header_end, n_points, result)

    return result


def _parse_ascii_fields(lines, start, n_points, result):
    """Parse ASCII field data from VTK file."""
    i = start
    while i < len(lines):
        text = lines[i].decode('ascii', errors='replace').strip()
        if text.startswith('SCALARS'):
            parts = text.split()
            name = parts[1]
            ncomp = int(parts[3]) if len(parts) > 3 else 1
            # Skip LOOKUP_TABLE line
            i += 1
            next_text = lines[i].decode('ascii', errors='replace').strip()
            if next_text.startswith('LOOKUP_TABLE'):
                i += 1
            # Read data
            values = []
            while len(values) < n_points * ncomp and i < len(lines):
                text2 = lines[i].decode('ascii', errors='replace').strip()
                if text2 and not text2.startswith(('SCALARS', 'VECTORS', 'LOOKUP')):
                    values.extend(float(x) for x in text2.split())
                elif text2.startswith(('SCALARS', 'VECTORS')):
                    break
                i += 1
            arr = np.array(values[:n_points * ncomp])
            if ncomp > 1:
                arr = arr.reshape(n_points, ncomp)
            result['fields'][name] = {'data': arr, 'ncomp': ncomp, 'type': 'scalar'}
        elif text.startswith('VECTORS'):
            parts = text.split()
            name = parts[1]
            ncomp = 3
            i += 1
            values = []
            while len(values) < n_points * ncomp and i < len(lines):
                text2 = lines[i].decode('ascii', errors='replace').strip()
                if text2 and not text2.startswith(('SCALARS', 'VECTORS', 'LOOKUP')):
                    values.extend(float(x) for x in text2.split())
                elif text2.startswith(('SCALARS', 'VECTORS')):
                    break
                i += 1
            arr = np.array(values[:n_points * ncomp]).reshape(n_points, ncomp)
            result['fields'][name] = {'data': arr, 'ncomp': ncomp, 'type': 'vector'}
        else:
            i += 1


def _parse_binary_fields(raw, lines, start, n_points, result):
    """Parse binary field data from VTK file."""
    # Find byte offset after POINT_DATA line
    offset = 0
    for i in range(start):
        offset = raw.index(b'\n', offset) + 1

    while offset < len(raw):
        # Read next header line
        nl = raw.find(b'\n', offset)
        if nl == -1:
            break
        header = raw[offset:nl].decode('ascii', errors='replace').strip()
        offset = nl + 1

        if header.startswith('SCALARS'):
            parts = header.split()
            name = parts[1]
            ncomp = int(parts[3]) if len(parts) > 3 else 1
            # Skip LOOKUP_TABLE
            nl2 = raw.find(b'\n', offset)
            lookup = raw[offset:nl2].decode('ascii', errors='replace').strip()
            if lookup.startswith('LOOKUP_TABLE'):
                offset = nl2 + 1
            nbytes = n_points * ncomp * 8
            if offset + nbytes > len(raw):
                break
            arr = np.frombuffer(raw[offset:offset + nbytes], dtype='>f8')
            if ncomp > 1:
                arr = arr.reshape(n_points, ncomp)
            result['fields'][name] = {'data': arr.copy(), 'ncomp': ncomp, 'type': 'scalar'}
            offset += nbytes

        elif header.startswith('VECTORS'):
            parts = header.split()
            name = parts[1]
            ncomp = 3
            nbytes = n_points * ncomp * 8
            if offset + nbytes > len(raw):
                break
            arr = np.frombuffer(raw[offset:offset + nbytes], dtype='>f8').reshape(n_points, ncomp)
            result['fields'][name] = {'data': arr.copy(), 'ncomp': ncomp, 'type': 'vector'}
            offset += nbytes

        elif not header:
            continue
        else:
            continue


def vtk_to_grid(vtk_data):
    """Convert VTK data to coordinate arrays and reshaped fields.

    Returns:
        x, y, z: 1D coordinate arrays
        fields: {name: ndarray shaped (Nz, Ny, Nx) or (Nz, Ny, Nx, ncomp)}
    """
    Nx, Ny, Nz = vtk_data['dims']

    # Use explicit coordinates if available (RECTILINEAR_GRID), else uniform
    if vtk_data.get('x_coords') is not None:
        x = vtk_data['x_coords']
    else:
        ox, oy, oz = vtk_data['origin']
        dx, dy, dz = vtk_data['spacing']
        x = ox + np.arange(Nx) * dx

    if vtk_data.get('y_coords') is not None:
        y = vtk_data['y_coords']
    else:
        if vtk_data['origin'] is not None:
            oy = vtk_data['origin'][1]
            dy = vtk_data['spacing'][1]
        else:
            oy, dy = 0.0, 1.0
        y = oy + np.arange(Ny) * dy

    if vtk_data.get('z_coords') is not None:
        z = vtk_data['z_coords']
    else:
        if vtk_data['origin'] is not None:
            oz = vtk_data['origin'][2]
            dz = vtk_data['spacing'][2]
        else:
            oz, dz = 0.0, 1.0
        z = oz + np.arange(Nz) * dz

    # Compute spacing for derived fields (use median for stretched grids)
    dx = np.median(np.diff(x)) if len(x) > 1 else 1.0
    dy = np.median(np.diff(y)) if len(y) > 1 else 1.0
    dz = np.median(np.diff(z)) if len(z) > 1 else 1.0

    fields = {}
    for name, fdata in vtk_data['fields'].items():
        arr = fdata['data']
        ncomp = fdata['ncomp']
        if ncomp == 1:
            fields[name] = arr.reshape(Nz, Ny, Nx)
        else:
            fields[name] = arr.reshape(Nz, Ny, Nx, ncomp)

    # Derive commonly needed fields if missing
    if 'velocity' in fields and 'velocity_magnitude' not in fields:
        vel = fields['velocity']
        fields['velocity_magnitude'] = np.linalg.norm(vel, axis=-1)
    if 'velocity' in fields and 'u' not in fields:
        fields['u'] = fields['velocity'][..., 0]
    if 'velocity' in fields and 'v' not in fields:
        fields['v'] = fields['velocity'][..., 1]
    if 'velocity' in fields and fields['velocity'].shape[-1] >= 3 and 'w' not in fields:
        fields['w'] = fields['velocity'][..., 2]
    if 'velocity' in fields and 'vorticity_magnitude' not in fields:
        vel = fields['velocity']
        # Compute vorticity from finite differences
        # For 2D (Nz=1): omega_z = dv/dx - du/dy
        # For 3D: |omega| = sqrt(omega_x^2 + omega_y^2 + omega_z^2)
        if Nz == 1:
            dvdx = np.gradient(vel[0, :, :, 1], dx, axis=1)
            dudy = np.gradient(vel[0, :, :, 0], dy, axis=0)
            fields['vorticity'] = (dvdx - dudy)[np.newaxis, :, :]
            fields['vorticity_magnitude'] = np.abs(fields['vorticity'])
        else:
            u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
            dwdy = np.gradient(w, dy, axis=1)
            dvdz = np.gradient(v, dz, axis=0)
            dudz = np.gradient(u, dz, axis=0)
            dwdx = np.gradient(w, dx, axis=2)
            dvdx = np.gradient(v, dx, axis=2)
            dudy = np.gradient(u, dy, axis=1)
            ox = dwdy - dvdz
            oy = dudz - dwdx
            oz = dvdx - dudy
            fields['vorticity_magnitude'] = np.sqrt(ox**2 + oy**2 + oz**2)

    return x, y, z, fields


# ============================================================================
# IBM Body Geometry
# ============================================================================

def hills_profile(x_arr):
    """Periodic hills geometry: y_wall(x) for any x (uses modular arithmetic).

    The hill has height H=1 and half-width 0.5*H, period L=9.
    Works with any coordinate system (shifted or unshifted).
    """
    H = 1.0
    L = 9.0
    y = np.zeros_like(x_arr)
    for i, x in enumerate(x_arr):
        xp = x % L
        if xp < 0:
            xp += L
        if xp <= 0.5 * H:
            y[i] = H * (1.0 + np.cos(np.pi * xp / (0.5 * H))) / 2.0
        elif xp >= L - 0.5 * H:
            y[i] = H * (1.0 + np.cos(np.pi * (L - xp) / (0.5 * H))) / 2.0
        else:
            y[i] = 0.0
    return y


def body_mask(x, y, body_type, domain):
    """Return a boolean mask (True = inside body) on the (Ny, Nx) grid.

    Includes a small buffer above the analytical surface to hide the IBM
    transition band (penalization/ghost-cell cells with partial velocity).
    """
    xmin, xmax, ymin, ymax = domain
    X, Y = np.meshgrid(x, y)
    mask = np.zeros_like(X, dtype=bool)
    # Buffer: ~1.5 grid cells above the body to hide IBM transition band
    dy_grid = np.median(np.diff(y)) if len(y) > 1 else 0.0
    buf = 1.5 * dy_grid
    if body_type == 'cylinder' or body_type == 'sphere':
        cx = xmin + (xmax - xmin) / 3.0
        cy = (ymin + ymax) / 2.0
        r = 0.5
        mask = (X - cx)**2 + (Y - cy)**2 < (r + buf)**2
    elif body_type in ('hills', 'periodic_hills'):
        y_wall = hills_profile(x)
        for j in range(len(y)):
            mask[j, :] = y[j] < (y_wall + buf)
    return mask


def draw_body(ax, body_type, domain):
    """Draw IBM body on axes. domain = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = domain
    if body_type == 'cylinder':
        cx = xmin + (xmax - xmin) / 3.0
        cy = (ymin + ymax) / 2.0
        r = 0.5
        circle = Circle((cx, cy), r, fc='0.85', ec='k', lw=0.5, zorder=5)
        ax.add_patch(circle)
    elif body_type == 'sphere':
        cx = xmin + (xmax - xmin) / 3.0
        cy = (ymin + ymax) / 2.0
        r = 0.5
        circle = Circle((cx, cy), r, fc='0.85', ec='k', lw=0.5, zorder=5)
        ax.add_patch(circle)
    elif body_type in ('hills', 'periodic_hills'):
        xh = np.linspace(xmin, xmax, 500)
        yh = hills_profile(xh)
        ax.fill_between(xh, ymin, yh, color='0.85', zorder=5)
        ax.plot(xh, yh, 'k-', lw=0.5, zorder=6)


def guess_body(filepath, domain):
    """Guess body type from filename or domain."""
    name = os.path.basename(filepath).lower()
    if 'cyl' in name:
        return 'cylinder'
    elif 'sph' in name:
        return 'sphere'
    elif 'hill' in name:
        return 'hills'
    # Guess from domain
    xmin, xmax, ymin, ymax = domain
    if abs(xmin - (-3.0)) < 0.1 and abs(xmax - 13.0) < 0.1:
        return 'cylinder'
    if abs(xmin) < 0.1 and abs(xmax - 9.0) < 0.5:
        return 'hills'
    return None


# ============================================================================
# Colormaps and field configuration
# ============================================================================

FIELD_CONFIG = {
    'vorticity': {
        'cmap': 'RdBu_r',
        'symmetric': True,
        'label': r'$\omega_z$',
        'title': 'Vorticity',
    },
    'vorticity_magnitude': {
        'cmap': 'inferno',
        'symmetric': False,
        'label': r'$|\omega|$',
        'title': 'Vorticity magnitude',
    },
    'velocity_magnitude': {
        'cmap': 'viridis',
        'symmetric': False,
        'label': r'$|u|$',
        'title': 'Velocity magnitude',
    },
    'u': {
        'cmap': 'RdBu_r',
        'symmetric': True,
        'label': r'$u$',
        'title': 'Streamwise velocity',
    },
    'v': {
        'cmap': 'RdBu_r',
        'symmetric': True,
        'label': r'$v$',
        'title': 'Wall-normal velocity',
    },
    'pressure': {
        'cmap': 'coolwarm',
        'symmetric': True,
        'label': r'$p$',
        'title': 'Pressure',
    },
    'nu_t': {
        'cmap': 'magma',
        'symmetric': False,
        'label': r'$\nu_t$',
        'title': 'Eddy viscosity',
    },
}

# Fields to plot when --field is not specified
DEFAULT_FIELDS_2D = ['vorticity', 'velocity_magnitude', 'pressure']
DEFAULT_FIELDS_3D = ['velocity_magnitude', 'vorticity_magnitude']


# ============================================================================
# Plotting functions
# ============================================================================

def periodic_shift_hills(x, field_2d):
    """Shift periodic hills data by half a period so the hill is centered.

    The domain is [0, L] with period L. The hill peaks are at x=0 and x=L
    (wrapping). Shifting by L/2 puts the hill in the middle of the plot.
    """
    L = x[-1] - x[0] + (x[1] - x[0])  # full period including last cell
    shift = len(x) // 2
    x_shifted = x - x[0] - L / 2  # center at x=0, range [-L/2, L/2]
    x_shifted = np.roll(x_shifted, shift)
    # Fix the wrap: after roll, ensure monotonic
    x_shifted = np.sort(x_shifted)
    field_shifted = np.roll(field_2d, shift, axis=1)
    return x_shifted, field_shifted


def plot_2d_field(ax, x, y, field_2d, field_name, body_type=None,
                  vmin=None, vmax=None, n_levels=64):
    """Plot a single 2D contour field on given axes."""
    cfg = FIELD_CONFIG.get(field_name, {
        'cmap': 'viridis', 'symmetric': False,
        'label': field_name, 'title': field_name,
    })

    # Shift periodic hills so the hump is centered
    if body_type in ('hills', 'periodic_hills'):
        x, field_2d = periodic_shift_hills(x, field_2d)

    # Mask body interior so contours don't render there
    domain = (x[0], x[-1], y[0], y[-1])
    plot_data = field_2d.copy()
    if body_type:
        mask = body_mask(x, y, body_type, domain)
        plot_data[mask] = np.nan

    finite = plot_data[np.isfinite(plot_data)]
    if len(finite) == 0:
        return None, cfg

    if vmin is None:
        if cfg['symmetric']:
            vlim = np.percentile(np.abs(finite), 98)
            vmin, vmax = -vlim, vlim
        else:
            vmin = np.percentile(finite, 2)
            vmax = np.percentile(finite, 98)

    if cfg['symmetric'] and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Set axes background to body color so NaN gaps blend with body fill
    if body_type:
        ax.set_facecolor('0.85')

    # Use pcolormesh for cleaner body masking (no contour line artifacts)
    cf = ax.pcolormesh(x, y, plot_data, cmap=cfg['cmap'],
                       norm=norm, shading='gouraud', rasterized=True)

    # Body overlay (drawn on top)
    if body_type:
        draw_body(ax, body_type, domain)

    ax.set_aspect('equal')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    return cf, cfg


def plot_single_vtk(vtk_path, field_names=None, body_type=None,
                    slice_spec=None, outdir='.'):
    """Plot fields from a single VTK file."""
    apply_style()

    vtk_data = read_vtk(vtk_path)
    x, y, z, fields = vtk_to_grid(vtk_data)
    Nz = vtk_data['dims'][2]
    is_2d = (Nz == 1)

    if body_type is None:
        domain = (x[0], x[-1], y[0], y[-1])
        body_type = guess_body(vtk_path, domain)

    available = list(fields.keys())
    if field_names is None:
        field_names = DEFAULT_FIELDS_2D if is_2d else DEFAULT_FIELDS_3D
    field_names = [f for f in field_names if f in available]

    if not field_names:
        print(f"No matching fields. Available: {available}")
        return

    basename = os.path.splitext(os.path.basename(vtk_path))[0]

    for fname in field_names:
        arr = fields[fname]

        if is_2d:
            # Shape: (1, Ny, Nx) or (1, Ny, Nx, ncomp)
            if arr.ndim == 3:
                field_2d = arr[0]
            else:
                field_2d = arr[0, :, :, 0]  # take first component
        else:
            # 3D: take a slice
            kslice = Nz // 2
            if slice_spec:
                kslice = _parse_slice(slice_spec, x, y, z)
            if arr.ndim == 3:
                field_2d = arr[kslice]
            else:
                field_2d = np.sqrt(np.sum(arr[kslice] ** 2, axis=-1))

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
        cf, cfg = plot_2d_field(ax, x, y, field_2d, fname, body_type)

        cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(cfg['label'], fontsize=8)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        outpath = os.path.join(outdir, f"{basename}_{fname}.pdf")
        save_fig(fig, outpath)


def plot_comparison(vtk_paths, field_name, body_type=None, labels=None,
                    slice_spec=None, outdir='.'):
    """Side-by-side comparison of the same field across multiple VTK files."""
    apply_style()
    n = len(vtk_paths)

    fig, axes = plt.subplots(n, 1, figsize=(DOUBLE_COL, 1.8 * n + 0.3),
                             squeeze=False)

    # First pass: compute global colorbar range
    all_data = []
    parsed = []
    for vpath in vtk_paths:
        vtk_data = read_vtk(vpath)
        x, y, z, fields = vtk_to_grid(vtk_data)
        Nz = vtk_data['dims'][2]
        is_2d = (Nz == 1)

        arr = fields.get(field_name)
        if arr is None:
            print(f"Warning: {field_name} not found in {vpath}")
            parsed.append(None)
            continue

        if is_2d:
            field_2d = arr[0] if arr.ndim == 3 else arr[0, :, :, 0]
        else:
            kslice = Nz // 2
            if slice_spec:
                kslice = _parse_slice(slice_spec, x, y, z)
            field_2d = arr[kslice] if arr.ndim == 3 else np.linalg.norm(arr[kslice], axis=-1)

        all_data.append(field_2d)
        parsed.append((x, y, field_2d))

    if not all_data:
        print("No data to plot.")
        return

    # Compute shared limits
    cfg = FIELD_CONFIG.get(field_name, {'symmetric': False})
    all_vals = np.concatenate([d[np.isfinite(d)].ravel() for d in all_data])
    if cfg.get('symmetric', False):
        vlim = np.percentile(np.abs(all_vals), 98)
        vmin, vmax = -vlim, vlim
    else:
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)

    if body_type is None and parsed[0] is not None:
        x0 = parsed[0][0]
        y0 = parsed[0][1]
        body_type = guess_body(vtk_paths[0], (x0[0], x0[-1], y0[0], y0[-1]))

    # Second pass: plot
    for i, (vpath, pdata) in enumerate(zip(vtk_paths, parsed)):
        ax = axes[i, 0]
        if pdata is None:
            ax.text(0.5, 0.5, f'Missing: {field_name}', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        x, y, field_2d = pdata
        cf, _ = plot_2d_field(ax, x, y, field_2d, field_name, body_type,
                              vmin=vmin, vmax=vmax)

        # Label
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = os.path.splitext(os.path.basename(vpath))[0]
        ax.text(0.02, 0.92, label, transform=ax.transAxes, fontsize=7,
                fontweight='bold', va='top',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

    # Shared colorbar
    cfg_full = FIELD_CONFIG.get(field_name, {'label': field_name})
    fig.subplots_adjust(right=0.88, hspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label(cfg_full.get('label', field_name), fontsize=8)
    cb.ax.tick_params(labelsize=6)

    safe_name = field_name.replace(' ', '_')
    outpath = os.path.join(outdir, f"comparison_{safe_name}.pdf")
    save_fig(fig, outpath, close=True)


def plot_3d_slices(vtk_path, field_name='velocity_magnitude', body_type=None,
                   outdir='.'):
    """Plot three orthogonal slices of a 3D field."""
    apply_style()

    vtk_data = read_vtk(vtk_path)
    x, y, z, fields = vtk_to_grid(vtk_data)
    Nx, Ny, Nz = vtk_data['dims']

    arr = fields.get(field_name)
    if arr is None:
        print(f"Field {field_name} not found.")
        return

    # For multi-component, take magnitude
    if arr.ndim == 4:
        arr = np.linalg.norm(arr, axis=-1)

    cfg = FIELD_CONFIG.get(field_name, {
        'cmap': 'viridis', 'symmetric': False, 'label': field_name,
    })
    vmin = np.percentile(arr[np.isfinite(arr)], 2)
    vmax = np.percentile(arr[np.isfinite(arr)], 98)
    if cfg.get('symmetric', False):
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim

    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax) \
        if cfg.get('symmetric', False) and vmin < 0 < vmax \
        else Normalize(vmin=vmin, vmax=vmax)

    # Use gridspec for better layout: cross-section gets more space
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1],
                  hspace=0.3, wspace=0.35)

    # x-y slice at z midplane (top-left)
    kz = Nz // 2
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.pcolormesh(x, y, arr[kz, :, :], cmap=cfg['cmap'],
                   norm=norm, shading='gouraud', rasterized=True)
    ax0.set_xlabel(r'$x$')
    ax0.set_ylabel(r'$y$')
    ax0.set_title(f'$z = {z[kz]:.2f}$', fontsize=7)
    ax0.set_aspect('equal')
    if body_type:
        draw_body(ax0, body_type, (x[0], x[-1], y[0], y[-1]))

    # x-z slice at y midplane (bottom-left)
    jy = Ny // 2
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.pcolormesh(x, z, arr[:, jy, :], cmap=cfg['cmap'],
                   norm=norm, shading='gouraud', rasterized=True)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$z$')
    ax1.set_title(f'$y = {y[jy]:.2f}$', fontsize=7)
    ax1.set_aspect('equal')

    # y-z cross-section at x=domain/3 (right, spans both rows)
    ix = Nx // 3
    ax2 = fig.add_subplot(gs[:, 1])
    cf = ax2.pcolormesh(y, z, arr[:, :, ix], cmap=cfg['cmap'],
                        norm=norm, shading='gouraud', rasterized=True)
    ax2.set_xlabel(r'$y$')
    ax2.set_ylabel(r'$z$')
    ax2.set_title(f'$x = {x[ix]:.2f}$', fontsize=7)
    ax2.set_aspect('equal')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label(cfg.get('label', field_name), fontsize=8)
    cb.ax.tick_params(labelsize=6)

    basename = os.path.splitext(os.path.basename(vtk_path))[0]
    outpath = os.path.join(outdir, f"{basename}_{field_name}_3d_slices.pdf")
    save_fig(fig, outpath)


# ============================================================================
# Helpers
# ============================================================================

def _parse_slice(spec, x, y, z):
    """Parse slice specification like 'z=0' and return index."""
    axis, val = spec.split('=')
    val = float(val)
    if axis == 'x':
        return np.argmin(np.abs(x - val))
    elif axis == 'y':
        return np.argmin(np.abs(y - val))
    elif axis == 'z':
        return np.argmin(np.abs(z - val))
    raise ValueError(f"Unknown slice axis: {axis}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flow field visualization for NN turbulence paper')
    parser.add_argument('vtk_files', nargs='+', help='VTK file(s) to plot')
    parser.add_argument('--field', '-f', default=None,
                        help='Field name (vorticity, velocity_magnitude, pressure, u, v, nu_t)')
    parser.add_argument('--body', '-b', default=None,
                        choices=['cylinder', 'sphere', 'hills', 'none'],
                        help='IBM body type for overlay')
    parser.add_argument('--slice', '-s', default=None,
                        help='3D slice spec: z=0, y=0.5, x=3.0')
    parser.add_argument('--labels', '-l', nargs='*', default=None,
                        help='Labels for comparison mode')
    parser.add_argument('-o', '--outdir', default='figures/paper',
                        help='Output directory')
    parser.add_argument('--list-fields', action='store_true',
                        help='List available fields and exit')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    body = args.body if args.body != 'none' else None

    if args.list_fields:
        vtk = read_vtk(args.vtk_files[0])
        _, _, _, derived = vtk_to_grid(vtk)
        print("Available fields:")
        for name in derived:
            arr = derived[name]
            nc = arr.shape[-1] if arr.ndim == 4 else 1
            print(f"  {name} ({nc} component{'s' if nc > 1 else ''})")
        return

    if len(args.vtk_files) == 1:
        # Single file mode
        vtk = read_vtk(args.vtk_files[0])
        is_3d = vtk['dims'][2] > 1
        _, _, _, derived = vtk_to_grid(vtk)

        if args.field:
            field_names = [args.field]
        else:
            field_names = None

        if is_3d and not args.slice:
            # 3D: show orthogonal slices
            for fname in (field_names or DEFAULT_FIELDS_3D):
                if fname in derived:
                    plot_3d_slices(args.vtk_files[0], fname, body, args.outdir)
        else:
            plot_single_vtk(args.vtk_files[0], field_names, body,
                           args.slice, args.outdir)
    else:
        # Comparison mode
        if args.field is None:
            args.field = 'vorticity'
            print(f"No --field specified for comparison; defaulting to {args.field}")
        plot_comparison(args.vtk_files, args.field, body, args.labels,
                       args.slice, args.outdir)


if __name__ == '__main__':
    main()
