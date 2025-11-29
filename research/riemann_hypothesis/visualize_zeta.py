"""
TNFR Riemann Visualization
==========================

Generates visualizations of the Structural Potential Field Φ_s over the complex plane.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_zeta_field(csv_path, output_path, title="TNFR Structural Potential Φ_s"):
    """Plot the structural potential field from CSV data."""
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Reshape data for contour plot
    # Assuming a regular grid, we can infer shape
    n_points = len(df)
    res = int(np.sqrt(n_points))

    if res * res != n_points:
        print("Data is not a square grid, attempting to pivot...")
        pivot = df.pivot(index='t', columns='sigma', values='phi_s')
        S = pivot.columns.values
        T = pivot.index.values
        Pot = pivot.values
        S, T = np.meshgrid(S, T)
    else:
        S = df['sigma'].values.reshape(res, res)
        T = df['t'].values.reshape(res, res)
        Pot = df['phi_s'].values.reshape(res, res)

    plt.figure(figsize=(12, 8))

    # Contour plot of Potential
    # We use a custom colormap where low values (zeros) are dark/distinct
    levels = np.linspace(np.nanmin(Pot), np.nanmax(Pot), 50)
    cp = plt.contourf(S, T, Pot, levels=levels, cmap='viridis_r')
    plt.colorbar(cp, label='Structural Potential Φ_s (log|ζ(s)|)')

    # Mark the critical line
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Critical Line Re(s)=0.5')

    plt.title(title)
    plt.xlabel('Re(s) (σ)')
    plt.ylabel('Im(s) (t)')
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Visualize wide scan
    plot_zeta_field(
        "research/riemann_hypothesis/data/zeta_wide_scan.csv",
        "research/riemann_hypothesis/images/zeta_wide_scan.png",
        "TNFR Structural Potential: Wide Scan"
    )

    # Visualize critical scan
    plot_zeta_field(
        "research/riemann_hypothesis/data/zeta_critical_scan.csv",
        "research/riemann_hypothesis/images/zeta_critical_scan.png",
        "TNFR Structural Potential: Critical Line Detail"
    )
