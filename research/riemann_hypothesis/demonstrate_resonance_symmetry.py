"""
TNFR Riemann Resonance & Symmetry Demonstration
===============================================

This script demonstrates the TNFR interpretation of the Riemann Hypothesis:
"Coherent Resonance (U3) is only possible where the Structural Symmetry Factor is Unitary."

We analyze the Riemann Functional Equation factor χ(s):
    ζ(s) = χ(s) ζ(1-s)

In TNFR terms:
- ζ(s) is the Structural Form (EPI).
- χ(s) is the Coupling Operator (UM) connecting s to its dual 1-s.
- For "Resonant Coupling" (U3), the operator must be Unitary (|χ(s)| = 1).
- If |χ(s)| ≠ 1, the coupling introduces gain/loss (Dissonance/Instability).

We demonstrate that |χ(s)| = 1 if and only if Re(s) = 0.5.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from tnfr.mathematics.zeta import zeta_function, chi_factor, structural_potential, mp

# Configure precision
mp.dps = 25

def analyze_symmetry_landscape(t_fixed=14.134725, sigma_range=(-1.0, 2.0), resolution=500):
    """Scan across the real axis (sigma) at a fixed height t (near a zero)."""
    sigmas = np.linspace(sigma_range[0], sigma_range[1], resolution)
    
    data = {
        'sigma': [],
        'chi_mag': [],
        'log_chi_mag': [],
        'phi_s': [],
        'unitary_error': []
    }
    
    print(f"Scanning Transverse Symmetry at t={t_fixed}...")
    
    for sigma in sigmas:
        s = complex(sigma, t_fixed)
        try:
            # 1. Compute Symmetry Factor χ(s)
            # Use optimized function
            chi = chi_factor(s)
            chi_mag = float(mp.fabs(chi))
            
            # 2. Compute Structural Potential Φ_s
            # Use optimized function
            phi_s = structural_potential(s)
            
            # 3. Unitary Error (Dissonance Metric)
            # How far is the coupling from being unitary?
            # D = | |χ(s)| - 1 |
            unitary_error = abs(chi_mag - 1.0)
            
            data['sigma'].append(sigma)
            data['chi_mag'].append(chi_mag)
            data['log_chi_mag'].append(np.log(chi_mag + 1e-20)) # Log scale for visualization
            data['phi_s'].append(phi_s)
            data['unitary_error'].append(unitary_error)
            
        except Exception as e:
            print(f"Error at s={s}: {e}")
            
    return pd.DataFrame(data)

def plot_demonstration(df, t_fixed, output_path):
    """Visualize the Unitary Resonance condition."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: The Unitary Condition (The "Why")
    # We plot log(|χ(s)|). It should be 0 only at sigma=0.5
    ax1.plot(df['sigma'], df['log_chi_mag'], color='purple', linewidth=2, label='Log Magnitude log|χ(s)|')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Critical Line (Re(s)=0.5)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Unitary Threshold (0)')
    
    ax1.set_title(f'TNFR U3 Condition: Unitary Coupling at t={t_fixed}')
    ax1.set_ylabel('Coupling Gain (Log Magnitude)')
    ax1.text(0.6, 2, 'Amplification (Dissonance)', color='purple', alpha=0.7)
    ax1.text(0.6, -2, 'Attenuation (Dissonance)', color='purple', alpha=0.7)
    ax1.text(0.51, 0.2, 'Unitary Resonance', color='red', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The Structural Potential (The "Where")
    # Where do the zeros actually live?
    ax2.plot(df['sigma'], df['phi_s'], color='blue', label='Structural Potential Φ_s')
    ax2.axvline(x=0.5, color='red', linestyle='--')
    
    # Highlight the minimum
    min_idx = df['phi_s'].idxmin()
    min_sigma = df['sigma'].iloc[min_idx]
    min_val = df['phi_s'].iloc[min_idx]
    
    ax2.scatter([min_sigma], [min_val], color='red', s=50, zorder=5, label=f'Resonance Node (Zero) at σ={min_sigma:.4f}')
    
    ax2.set_title('Resulting Structural Potential Φ_s')
    ax2.set_xlabel('Real Part σ (Re(s))')
    ax2.set_ylabel('Potential Φ_s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Demonstration plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Use the first Riemann zero height
    t_zero = 14.13472514173469
    
    df = analyze_symmetry_landscape(t_fixed=t_zero, sigma_range=(0.0, 1.0), resolution=1000)
    
    # Save data
    csv_path = "research/riemann_hypothesis/data/unitary_symmetry_scan.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Plot
    plot_demonstration(df, t_zero, "research/riemann_hypothesis/images/unitary_resonance_proof.png")
