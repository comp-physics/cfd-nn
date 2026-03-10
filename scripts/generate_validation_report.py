#!/usr/bin/env python3
"""Generate turbulence validation report from simulation output files.

Reads .dat profile outputs from long-run RANS/DNS/TGV/Poiseuille simulations
and compares against reference data (MKM DNS, Brachet TGV, analytical Poiseuille).

Usage:
    python3 scripts/generate_validation_report.py [--output-dir DIR] [--data-dir DIR]

Output: PNG plots + error_summary.txt in output/validation_report/
"""

import argparse
import os
import sys
import glob
import numpy as np

# Optional: matplotlib for plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping plot generation")


def load_dat(path, skip_comments=True):
    """Load a whitespace-delimited .dat file, skipping comment lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or (skip_comments and line.startswith("#")):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows) if rows else np.empty((0, 0))


# ============================================================================
# MKM reference data loaders
# ============================================================================

def load_mkm_means(data_dir):
    """Load MKM mean velocity profile (y+, U+)."""
    path = os.path.join(data_dir, "mkm_retau180", "chan180_means.dat")
    if not os.path.exists(path):
        return None
    data = load_dat(path)
    if data.size == 0:
        return None
    # MKM means file: columns are y/delta, y+, U_mean, dU/dy, ...
    # First column: y/delta, second: y+, third: U_mean
    return {"y_plus": data[:, 1], "u_plus": data[:, 2]}


def load_mkm_stress(data_dir, component):
    """Load MKM Reynolds stress profile."""
    fname = f"chan180_{component}.dat"
    path = os.path.join(data_dir, "mkm_retau180", fname)
    if not os.path.exists(path):
        return None
    data = load_dat(path)
    if data.size == 0:
        return None
    return {"y_plus": data[:, 0], "value": data[:, 1]}


def load_brachet_tgv(data_dir):
    """Load Brachet TGV dissipation curve."""
    path = os.path.join(data_dir, "brachet_tgv", "dissipation_re1600.dat")
    if not os.path.exists(path):
        return None
    data = load_dat(path)
    if data.size == 0:
        return None
    return {"t_star": data[:, 0], "eps_star": data[:, 1]}


# ============================================================================
# Simulation output loaders
# ============================================================================

def find_profile_files(output_dir, pattern):
    """Find simulation output files matching a glob pattern."""
    return sorted(glob.glob(os.path.join(output_dir, pattern)))


def load_channel_profile(path):
    """Load a channel U+(y+) profile from simulation output.

    Expected format: y+  U+  (two columns, comment header)
    """
    data = load_dat(path)
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    return {"y_plus": data[:, 0], "u_plus": data[:, 1]}


def load_tgv_energy(path):
    """Load TGV kinetic energy history.

    Expected format: t  KE  (two columns)
    """
    data = load_dat(path)
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    return {"t": data[:, 0], "ke": data[:, 1]}


# ============================================================================
# Error metrics
# ============================================================================

def compute_l2_error(sim_yp, sim_up, ref_yp, ref_up):
    """Interpolate simulation onto reference y+ points and compute L2 error."""
    sim_interp = np.interp(ref_yp, sim_yp, sim_up)
    err = sim_interp - ref_up
    return np.sqrt(np.sum(err**2) / np.sum(ref_up**2))


# ============================================================================
# Plot generators
# ============================================================================

def plot_u_plus_profiles(mkm, sim_profiles, output_dir):
    """Plot U+(y+) for all RANS models vs MKM reference."""
    if not HAS_MPL or mkm is None:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # MKM reference
    ax.plot(mkm["y_plus"], mkm["u_plus"], "ko-", ms=3, lw=1.5, label="MKM DNS", zorder=10)

    # Log law
    yp = np.logspace(np.log10(30), np.log10(180), 50)
    ax.plot(yp, 2.44 * np.log(yp) + 5.2, "k--", lw=0.8, label="log law")

    # Viscous sublayer
    yp_visc = np.linspace(0.1, 5, 20)
    ax.plot(yp_visc, yp_visc, "k:", lw=0.8, label="U+=y+")

    # Simulation profiles
    colors = plt.cm.tab10(np.linspace(0, 1, len(sim_profiles)))
    for (name, prof), color in zip(sim_profiles.items(), colors):
        ax.plot(prof["y_plus"], prof["u_plus"], "-", color=color, lw=1.2, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("y+")
    ax.set_ylabel("U+")
    ax.set_title("Mean Velocity Profiles vs MKM DNS (Re_tau=180)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0.1, 200)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "u_plus_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_reynolds_stresses(mkm_stresses, sim_stresses, output_dir):
    """Plot Reynolds stress profiles vs MKM reference."""
    if not HAS_MPL:
        return

    components = ["uu", "vv", "ww", "uv"]
    labels = ["<u'u'>+", "<v'v'>+", "<w'w'>+", "-<u'v'>+"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, comp, label in zip(axes.flat, components, labels):
        ref = mkm_stresses.get(comp)
        if ref is not None:
            ax.plot(ref["y_plus"], ref["value"], "ko-", ms=3, lw=1.5, label="MKM DNS")

        for name, stresses in sim_stresses.items():
            s = stresses.get(comp)
            if s is not None:
                ax.plot(s["y_plus"], s["value"], "-", lw=1.2, label=name)

        ax.set_xlabel("y+")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reynolds Stresses vs MKM DNS (Re_tau=180)")
    path = os.path.join(output_dir, "reynolds_stresses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tgv_dissipation(brachet, sim_tgv, output_dir):
    """Plot TGV dissipation rate vs Brachet reference."""
    if not HAS_MPL or brachet is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(brachet["t_star"], brachet["eps_star"], "ko-", ms=4, lw=1.5, label="Brachet (1983)")

    if sim_tgv is not None:
        # Compute dissipation rate as -dKE/dt
        t, ke = sim_tgv["t"], sim_tgv["ke"]
        if len(t) > 2:
            eps = -np.gradient(ke, t)
            ax.plot(t, eps, "b-", lw=1.2, label="CFD-NN DNS")

    ax.set_xlabel("t*")
    ax.set_ylabel("-dK/dt / (U0^3/L)")
    ax.set_title("TGV Dissipation Rate (Re=1600)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "tgv_dissipation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_poiseuille_convergence(sim_profiles, output_dir):
    """Plot Poiseuille error convergence with grid refinement."""
    if not HAS_MPL or not sim_profiles:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ny_vals = sorted(sim_profiles.keys())
    errors = [sim_profiles[ny] for ny in ny_vals]
    h_vals = [1.0 / ny for ny in ny_vals]

    ax.loglog(h_vals, errors, "bo-", ms=6, lw=1.5, label="L2 error")

    # 2nd order reference line
    h_ref = np.array([h_vals[0], h_vals[-1]])
    e_ref = errors[0] * (h_ref / h_vals[0]) ** 2
    ax.loglog(h_ref, e_ref, "k--", lw=0.8, label="2nd order")

    ax.set_xlabel("h = 1/Ny")
    ax.set_ylabel("L2 relative error")
    ax.set_title("Poiseuille Grid Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "poiseuille_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Error summary table
# ============================================================================

def write_error_summary(mkm, sim_profiles, output_dir):
    """Write error metrics table to file."""
    path = os.path.join(output_dir, "error_summary.txt")
    with open(path, "w") as f:
        f.write("Turbulence Validation Error Summary\n")
        f.write("=" * 60 + "\n\n")

        if mkm is not None:
            f.write(f"{'Model':<20} {'L2 error':>10} {'Linf error':>12}\n")
            f.write("-" * 42 + "\n")

            for name, prof in sim_profiles.items():
                l2 = compute_l2_error(prof["y_plus"], prof["u_plus"],
                                       mkm["y_plus"], mkm["u_plus"])
                sim_interp = np.interp(mkm["y_plus"], prof["y_plus"], prof["u_plus"])
                linf = np.max(np.abs(sim_interp - mkm["u_plus"]) / (mkm["u_plus"] + 1e-30))
                f.write(f"{name:<20} {l2:>10.4f} {linf:>12.4f}\n")
        else:
            f.write("No MKM reference data found. Run scripts/download_reference_data.sh first.\n")

        f.write("\n")

    print(f"  Saved {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate turbulence validation report")
    parser.add_argument("--output-dir", default="output/validation_report",
                        help="Output directory for plots and summary")
    parser.add_argument("--data-dir", default="data/reference",
                        help="Reference data directory")
    parser.add_argument("--sim-dir", default="output/validation",
                        help="Simulation output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Turbulence Validation Report Generator ===\n")

    # Load reference data
    print("Loading reference data...")
    mkm = load_mkm_means(args.data_dir)
    if mkm is not None:
        print(f"  MKM: {len(mkm['y_plus'])} points loaded")
    else:
        print("  [WARN] MKM data not found. Run: scripts/download_reference_data.sh")

    mkm_stresses = {}
    for comp in ["uu", "vv", "ww", "uv"]:
        s = load_mkm_stress(args.data_dir, comp)
        if s is not None:
            mkm_stresses[comp] = s

    brachet = load_brachet_tgv(args.data_dir)
    if brachet is not None:
        print(f"  Brachet TGV: {len(brachet['t_star'])} points loaded")

    # Load simulation profiles
    print("\nLoading simulation output...")
    sim_profiles = {}
    for path in find_profile_files(args.sim_dir, "*_u_plus.dat"):
        name = os.path.basename(path).replace("_u_plus.dat", "")
        prof = load_channel_profile(path)
        if prof is not None:
            sim_profiles[name] = prof
            print(f"  {name}: {len(prof['y_plus'])} points")

    sim_tgv = None
    tgv_files = find_profile_files(args.sim_dir, "tgv_*_energy.dat")
    if tgv_files:
        sim_tgv = load_tgv_energy(tgv_files[0])
        if sim_tgv is not None:
            print(f"  TGV energy: {len(sim_tgv['t'])} time points")

    # Generate plots
    print("\nGenerating plots...")
    if sim_profiles:
        plot_u_plus_profiles(mkm, sim_profiles, args.output_dir)
    if mkm_stresses:
        plot_reynolds_stresses(mkm_stresses, {}, args.output_dir)
    if brachet is not None:
        plot_tgv_dissipation(brachet, sim_tgv, args.output_dir)

    # Error summary
    print("\nComputing error metrics...")
    if sim_profiles:
        write_error_summary(mkm, sim_profiles, args.output_dir)
    else:
        print("  No simulation profiles found. Run SLURM validation jobs first.")
        print(f"  Expected location: {args.sim_dir}/*_u_plus.dat")

    print(f"\nReport generated in {args.output_dir}/")
    if not sim_profiles:
        print("\nTo generate full report:")
        print("  1. Run: scripts/download_reference_data.sh")
        print("  2. Run: scripts/run_validation.sh")
        print("  3. Re-run: python3 scripts/generate_validation_report.py")


if __name__ == "__main__":
    main()
