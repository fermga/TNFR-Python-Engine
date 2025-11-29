"""
TNFR Riemann Metrics Analysis
=============================

This script analyzes TNFR-specific structural metrics along the critical line Re(s) = 0.5.

We investigate:
1. Structural Potential Φ_s = log|ζ(s)| (Energy)
2. Phase θ = arg(ζ(s)) (Synchronization)
3. Phase Gradient |∇φ| = |dθ/dt| (Structural Stress)

Hypothesis:
Zeros are "perfect resonance nodes" where:
- Φ_s → -∞ (Infinite binding energy/perfect stability)
- Phase θ undergoes a discrete jump (Topological event)
- |∇φ| exhibits a singularity (Structural stress release)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add src to path to import tnfr modules
sys.path.append(os.path.abspath("src"))

from tnfr.mathematics.zeta import zeta_function, structural_potential, mp

# Configure precision
mp.dps = 25

def analyze_critical_line(t_start=0, t_end=50, resolution=1000):
    """Compute TNFR metrics along the critical line."""
    ts = np.linspace(t_start, t_end, resolution)
    
    data = {
        't': [],
        'phi_s': [],
        'phase': [],
        'phase_grad': []
    }
    
    print(f"Scanning critical line t=[{t_start}, {t_end}] with {resolution} points...")
    
    prev_phase = 0
    
    for i, t in enumerate(ts):
        s = complex(0.5, t)
        try:
            z = zeta_function(s)
            
            # 1. Structural Potential (Energy)
            # Use the optimized function
            phi_s = structural_potential(s)
            
            # 2. Phase (Synchronization)
            # mpmath arg gives phase in (-pi, pi]
            phase = float(mp.arg(z))
            
            # 3. Phase Gradient (Stress)
            # Simple finite difference
            if i > 0:
                dt = t - ts[i-1]
                d_theta = phase - prev_phase
                # Handle phase wrapping (-pi to pi)
                if d_theta > np.pi: d_theta -= 2*np.pi
                if d_theta < -np.pi: d_theta += 2*np.pi
                phase_grad = abs(d_theta / dt)
            else:
                phase_grad = 0.0
            
            prev_phase = phase
            
            data['t'].append(t)
            data['phi_s'].append(phi_s)
            data['phase'].append(phase)
            data['phase_grad'].append(phase_grad)
            
        except Exception as e:
            print(f"Error at t={t}: {e}")

    return pd.DataFrame(data)

def plot_metrics(df, output_path):
    """Visualize the structural metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Structural Potential (The "Canyon")
    ax1.plot(df['t'], df['phi_s'], color='blue', label='Φ_s (Potential)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Potential Φ_s')
    ax1.set_title('TNFR Structural Metrics along Critical Line Re(s)=0.5')
    ax1.grid(True, alpha=0.3)
    
    # Highlight zeros (deep minima)
    zeros_mask = df['phi_s'] < -3
    ax1.scatter(df[zeros_mask]['t'], df[zeros_mask]['phi_s'], color='red', s=20, label='Resonance Nodes (Zeros)')
    ax1.legend()
    
    # 2. Phase (The "Orientation")
    ax2.plot(df['t'], df['phase'], color='green', label='Phase θ')
    ax2.set_ylabel('Phase (rad)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Phase Gradient (The "Stress")
    ax3.plot(df['t'], df['phase_grad'], color='purple', label='|∇φ| (Phase Gradient)')
    ax3.set_ylabel('Gradient |dθ/dt|')
    ax3.set_xlabel('Im(s) (t)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Metrics plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Analyze the first few zeros (14.13, 21.02, 25.01, 30.42, 32.93, 37.58)
    df = analyze_critical_line(t_start=10, t_end=40, resolution=2000)
    
    # Save data
    csv_path = "research/riemann_hypothesis/data/critical_line_metrics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Plot
    plot_metrics(df, "research/riemann_hypothesis/images/tnfr_metrics_analysis.png")
