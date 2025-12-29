#!/usr/bin/env python3
"""
Plot kinetic energy decay for 3D Taylor-Green vortex.
Compares simulation results against theoretical exponential decay.
"""

import numpy as np
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "output", "tg3d_history.dat")

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Run the simulation first: ./run.sh")
        return

    # Load data
    data = np.loadtxt(data_file, comments='#')
    time = data[:, 0]
    ke = data[:, 1]
    ke_ratio = data[:, 2]
    enstrophy = data[:, 3]

    # Compute theoretical decay (assuming Re=100, nu=0.01)
    nu = 0.01  # Default viscosity
    decay_rate = 2 * nu
    ke_theory = np.exp(-decay_rate * time)

    print("=" * 50)
    print("Taylor-Green Vortex Energy Analysis")
    print("=" * 50)
    print(f"Initial KE:     {ke[0]:.6f}")
    print(f"Final KE:       {ke[-1]:.6f}")
    print(f"KE ratio:       {ke_ratio[-1]:.6f}")
    print(f"Theory:         {ke_theory[-1]:.6f}")
    print(f"Error:          {abs(ke_ratio[-1] - ke_theory[-1]):.6f}")
    print()
    print(f"Peak enstrophy: {max(enstrophy):.6f}")
    print("=" * 50)

    # Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # KE decay
        ax1.semilogy(time, ke_ratio, 'b-', linewidth=2, label='Simulation')
        ax1.semilogy(time, ke_theory, 'r--', linewidth=2, label='Theory: exp(-2νt)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('KE / KE₀')
        ax1.set_title('Kinetic Energy Decay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Enstrophy
        ax2.plot(time, enstrophy, 'g-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Enstrophy')
        ax2.set_title('Enstrophy Evolution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_plot = os.path.join(script_dir, "output", "energy_decay.png")
        plt.savefig(output_plot, dpi=150)
        print(f"\nPlot saved to: {output_plot}")

        plt.show()

    except ImportError:
        print("\nMatplotlib not available. Install it for plotting:")
        print("  pip install matplotlib")

if __name__ == "__main__":
    main()
