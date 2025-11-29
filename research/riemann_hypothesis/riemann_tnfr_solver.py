"""
TNFR Riemann Solver
===================

This script analyzes the Riemann Zeta function through the lens of TNFR physics.
It maps the complex plane s = σ + it to a structural potential field Φ_s.

Hypothesis:
The zeros of ζ(s) correspond to minima in the Structural Potential Φ_s,
representing points of maximum resonance/coherence in the number theoretic network.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from tnfr.mathematics.zeta import structural_potential, mp

# Configure precision
mp.dps = 15


def compute_tnfr_zeta_potential(sigma_range, t_range, resolution=100):
    """
    Compute the TNFR Structural Potential Φ_s over a grid in the complex plane.

    Φ_s is modeled here as log|ζ(s)|, which corresponds to the
    "structural pressure" of the number system at that frequency.
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], resolution)
    ts = np.linspace(t_range[0], t_range[1], resolution)

    S, T = np.meshgrid(sigmas, ts)
    Potential = np.zeros_like(S)

    print(f"Computing field over {resolution}x{resolution} grid...")

    for i in range(resolution):
        for j in range(resolution):
            s = complex(S[i, j], T[i, j])
            try:
                # Use optimized function
                Potential[i, j] = structural_potential(s)

            except Exception:
                Potential[i, j] = np.nan

    return S, T, Potential


def save_results(S, T, Potential, filename="zeta_field.csv"):
    """Save the computed field to CSV."""
    os.makedirs("research/riemann_hypothesis/data", exist_ok=True)

    # Flatten and save
    df = pd.DataFrame({
        'sigma': S.flatten(),
        't': T.flatten(),
        'phi_s': Potential.flatten()
    })

    path = f"research/riemann_hypothesis/data/{filename}"
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    # Scan the critical strip around the first few zeros
    # First zero is at t ~ 14.1347
    print("Starting TNFR Riemann Analysis...")

    # Wide scan
    S, T, Pot = compute_tnfr_zeta_potential(
        sigma_range=(-2, 4),
        t_range=(0, 30),
        resolution=200
    )
    save_results(S, T, Pot, "zeta_wide_scan.csv")

    # Focused scan on critical line
    S_crit, T_crit, Pot_crit = compute_tnfr_zeta_potential(
        sigma_range=(0.4, 0.6),
        t_range=(10, 20),
        resolution=200
    )
    save_results(S_crit, T_crit, Pot_crit, "zeta_critical_scan.csv")

    print("Analysis complete.")
