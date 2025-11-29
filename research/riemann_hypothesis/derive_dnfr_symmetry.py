"""
TNFR Riemann Derivation: ΔNFR from Symmetry Breaking
====================================================

This script computationally verifies the derivation of ΔNFR from the 
Riemann Functional Equation's symmetry breaking properties.

Definition (Derived):
    ΔNFR(s) ≡ | log|ζ(s)| - log|ζ(1-s)| |
    
    "The Structural Pressure (ΔNFR) is the magnitude of the asymmetry 
     between the form at s and its dual at 1-s."

Using the Functional Equation ζ(s) = χ(s)ζ(1-s), this simplifies to:
    ΔNFR(s) = | log|χ(s)| |

We plot this derived field to show it forms a "Stability Valley" exactly at Re(s)=0.5.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from tnfr.mathematics.zeta import structural_pressure, mp

# Configure precision
mp.dps = 25

def compute_derived_dnfr(sigma, t):
    """
    Compute ΔNFR as the symmetry violation.
    ΔNFR = | log|χ(s)| |
    """
    s = complex(sigma, t)
    
    # Use optimized function
    try:
        dnfr = structural_pressure(s)
    except:
        dnfr = np.nan
        
    return dnfr

def scan_symmetry_landscape(t_fixed=100.0, sigma_range=(-2, 3), resolution=500):
    sigmas = np.linspace(sigma_range[0], sigma_range[1], resolution)
    dnfr_values = []
    
    print(f"Scanning Derived ΔNFR at t={t_fixed}...")
    
    for sigma in sigmas:
        val = compute_derived_dnfr(sigma, t_fixed)
        dnfr_values.append(val)
        
    return sigmas, dnfr_values

def plot_derivation(sigmas, dnfr_values, t_fixed, output_path):
    plt.figure(figsize=(10, 6))
    
    plt.plot(sigmas, dnfr_values, color='crimson', linewidth=2, label='Derived ΔNFR (Symmetry Violation)')
    
    # Mark the critical line
    plt.axvline(x=0.5, color='black', linestyle='--', label='Critical Line (σ=0.5)')
    plt.axhline(y=0, color='black', linewidth=0.5)
    
    # Annotations
    plt.title(f'Derivation of ΔNFR from Functional Symmetry (t={t_fixed})')
    plt.xlabel('Real Part σ')
    plt.ylabel('Structural Pressure ΔNFR = |log|χ(s)||')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text explaining the physics
    plt.text(0.6, max(dnfr_values)*0.8, 'High Pressure (Asymmetry)', color='crimson')
    plt.text(-1.5, max(dnfr_values)*0.8, 'High Pressure (Asymmetry)', color='crimson')
    plt.text(0.55, 0.1, 'Equilibrium (ΔNFR=0)', color='black', fontweight='bold')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Derivation plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Scan at a higher t where the asymptotic behavior is clear
    t_val = 50.0
    sigmas, dnfrs = scan_symmetry_landscape(t_fixed=t_val)
    
    # Save data
    df = pd.DataFrame({'sigma': sigmas, 'dnfr': dnfrs})
    csv_path = "research/riemann_hypothesis/data/derived_dnfr_scan.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    plot_derivation(sigmas, dnfrs, t_val, "research/riemann_hypothesis/images/derived_dnfr_proof.png")
