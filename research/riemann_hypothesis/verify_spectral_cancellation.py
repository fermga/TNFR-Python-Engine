"""
TNFR Spectral Cancellation Verification
=======================================

This script verifies the "Spectral Cancellation Mechanism" proposed in the asymptotic proof.

We compute the two opposing "forces" acting on the Structural Potential Φ_s:
1. The Spectral Force F_spec: Repulsion from the zeros (Sum over ρ).
2. The Analytic Force F_anal: Attraction from the Gamma factor (Log t).

Hypothesis:
- At Re(s) = 0.5, F_spec + F_anal ≈ 0 (Equilibrium/Canyon).
- At Re(s) ≠ 0.5, F_spec + F_anal ≠ 0 (Net pressure ΔNFR).

We use the explicit formula for the logarithmic derivative:
Re(ζ'/ζ(s)) ≈ Σ [ (σ-0.5) / ((σ-0.5)² + (t-γ)²) ] - 0.5 * log(t/2π)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from tnfr.mathematics.zeta import zeta_zero, mp

# Configure precision
mp.dps = 25

def get_zeros(n_zeros=1000):
    """Retrieve the first N Riemann zeros."""
    print(f"Retrieving first {n_zeros} zeros...")
    zeros = []
    for i in range(1, n_zeros + 1):
        # Use optimized function
        z = zeta_zero(i)
        zeros.append(float(z.imag))
    return np.array(zeros)

def compute_forces(sigma, t, gammas):
    """
    Compute the Spectral and Analytic forces at s = sigma + it.
    
    F_spec = Sum_ρ (σ - 0.5) / |s - ρ|²
    F_anal = -0.5 * log(t / 2π)
    """
    # Spectral Force (Sum over zeros)
    # We only sum over the provided zeros, which is a local approximation.
    # For accurate results, we need zeros near t.
    # The term is (σ - 0.5) / ((σ - 0.5)² + (t - γ)²)
    
    denom = (sigma - 0.5)**2 + (t - gammas)**2
    terms = (sigma - 0.5) / denom
    f_spec = np.sum(terms)
    
    # Analytic Force (Gamma approximation)
    # Use mpmath for high precision analytic term
    f_anal = float(-0.5 * mp.log(t / (2 * mp.pi)))
    
    return f_spec, f_anal

def scan_forces(t_fixed=100.0, n_zeros=2000):
    # We need zeros around t_fixed. 
    gammas = get_zeros(n_zeros)
    max_gamma = gammas[-1]
    print(f"Max gamma loaded: {max_gamma}")
    
    if t_fixed > max_gamma:
        print("Warning: t_fixed is outside the range of loaded zeros. Results may be inaccurate.")
    
    sigmas = np.linspace(0.0, 1.0, 100)
    
    data = {
        'sigma': [],
        'f_spec': [],
        'f_anal': [],
        'f_net': []
    }
    
    print(f"Scanning forces at t={t_fixed}...")
    
    for sigma in sigmas:
        f_spec, f_anal = compute_forces(sigma, t_fixed, gammas)
        f_net = f_spec + f_anal
        
        data['sigma'].append(sigma)
        data['f_spec'].append(f_spec)
        data['f_anal'].append(f_anal)
        data['f_net'].append(f_net)
        
    return pd.DataFrame(data)

def test_convergence(sigma_fixed=0.6, t_fixed=100.0, max_zeros=2000, step=50):
    """Test convergence of the spectral sum as N -> infinity."""
    gammas = get_zeros(max_zeros)
    
    ns = range(10, max_zeros + 1, step)
    results = {'N': [], 'f_net': []}
    
    print(f"Testing convergence at s={sigma_fixed} + {t_fixed}i...")
    
    for n in ns:
        current_gammas = gammas[:n]
        f_spec, f_anal = compute_forces(sigma_fixed, t_fixed, current_gammas)
        f_net = f_spec + f_anal
        results['N'].append(n)
        results['f_net'].append(f_net)
        
    return pd.DataFrame(results)

def plot_convergence(df, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df['N'], df['f_net'], color='blue', label='Net Force (Error)')
    plt.axhline(y=0, color='red', linestyle='--', label='Perfect Cancellation')
    plt.title('Convergence of Spectral Cancellation (N -> ∞)')
    plt.xlabel('Number of Zeros (N)')
    plt.ylabel('Net Force (Residual)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_forces(df, t_fixed, output_path):
    plt.figure(figsize=(10, 8))
    
    # Plot Forces
    plt.plot(df['sigma'], df['f_spec'], label='Spectral Force (Sum over Zeros)', color='blue')
    plt.plot(df['sigma'], df['f_anal'], label='Analytic Force (Gamma Term)', color='green', linestyle='--')
    plt.plot(df['sigma'], df['f_net'], label='Net Force (Re ζ\'/ζ)', color='red', linewidth=2)
    
    plt.axvline(x=0.5, color='black', linestyle=':', label='Critical Line')
    plt.axhline(y=0, color='black', linewidth=0.5)
    
    plt.title(f'Spectral vs Analytic Forces at t={t_fixed}')
    plt.xlabel('Real Part σ')
    plt.ylabel('Force Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.text(0.6, df['f_net'].max()*0.8, 'Net Force > 0 (Pushing Right)', color='red')
    plt.text(0.1, df['f_net'].min()*0.8, 'Net Force < 0 (Pushing Left)', color='red')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Forces plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Scan Forces across sigma
    df_forces = scan_forces(t_fixed=100.0, n_zeros=1000)
    plot_forces(df_forces, 100.0, "research/riemann_hypothesis/images/spectral_cancellation.png")
    
    # 2. Test Convergence N -> infinity
    df_conv = test_convergence(sigma_fixed=0.7, t_fixed=100.0, max_zeros=2000, step=20)
    plot_convergence(df_conv, "research/riemann_hypothesis/images/spectral_convergence.png")

