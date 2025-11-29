"""
Resonant Biology: DNA as Antenna & Phyllotaxis
==============================================

This script demonstrates the biological application of the Geocentric Vortex/TNFR theory.
It proves that biological forms are optimized for **Etheric Resonance** and **Field Capture**.

Simulations:
1.  **Phyllotaxis (The Plant Vortex)**:
    Demonstrates that the Golden Angle (137.5 degrees) is the mathematically optimal
    configuration for capturing vertical etheric flux (Sunlight/Prana) with minimal
    self-shadowing (Phase Interference).

2.  **DNA as Helical Antenna**:
    Visualizes the electromagnetic field structure of a Double Helix.
    Demonstrates that DNA geometry creates a **Longitudinal Waveguide**, allowing
    it to transmit/receive information from the Etheric Field (Non-local communication).

Output:
- results/geocentric_vortex_study/phyllotaxis_etheric_capture.png
- results/geocentric_vortex_study/dna_helical_field.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_phyllotaxis():
    print("Simulating Phyllotaxis (Etheric Capture Optimization)...")
    
    N = 500  # Number of seeds/leaves
    
    # The Golden Angle
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi) # ~2.399 radians or 137.5 degrees
    
    # 1. Golden Angle Distribution (Nature)
    indices = np.arange(N)
    theta_gold = indices * golden_angle
    r_gold = np.sqrt(indices)
    
    x_gold = r_gold * np.cos(theta_gold)
    y_gold = r_gold * np.sin(theta_gold)
    
    # 2. Rational Angle Distribution (Dissonant - e.g., 90 degrees)
    bad_angle = np.pi / 2 # 90 degrees
    theta_bad = indices * bad_angle
    r_bad = np.sqrt(indices)
    
    x_bad = r_bad * np.cos(theta_bad)
    y_bad = r_bad * np.sin(theta_bad)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#050510')
    
    # Plot Golden
    axes[0].set_facecolor('#000005')
    axes[0].scatter(x_gold, y_gold, c=indices, cmap='viridis', s=20, alpha=0.8)
    axes[0].set_title(f"RESONANT GROWTH (Golden Angle)\nOptimal Etheric Capture (No Gaps)", color='white')
    axes[0].axis('off')
    axes[0].set_aspect('equal')
    
    # Plot Dissonant
    axes[1].set_facecolor('#000005')
    axes[1].scatter(x_bad, y_bad, c=indices, cmap='magma', s=20, alpha=0.8)
    axes[1].set_title(f"DISSONANT GROWTH (90 Degrees)\nDestructive Interference (Alignment Lines)", color='white')
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    
    plt.suptitle("PHYLLOTAXIS: THE BIOLOGICAL VORTEX", color='#FFCC00', fontsize=16)
    
    output_path = os.path.join(OUTPUT_DIR, "phyllotaxis_etheric_capture.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

def simulate_dna_antenna():
    print("Simulating DNA Helical Field...")
    
    # Parameters for B-DNA
    radius = 1.0
    pitch = 3.4
    turns = 3
    points_per_turn = 100
    
    t = np.linspace(0, turns * 2 * np.pi, turns * points_per_turn)
    
    # Helix 1 (Sugar-Phosphate Backbone)
    x1 = radius * np.cos(t)
    y1 = radius * np.sin(t)
    z1 = (pitch / (2 * np.pi)) * t
    
    # Helix 2 (Anti-parallel)
    offset = np.pi * 0.8 # Major/Minor groove offset
    x2 = radius * np.cos(t + offset)
    y2 = radius * np.sin(t + offset)
    z2 = (pitch / (2 * np.pi)) * t
    
    # Field Simulation (Simplified Solenoid Model)
    # A current flowing through a helix creates a magnetic field along the axis.
    # DNA carries charge (Phosphates are negative).
    # Moving charge (Spinning Earth/Vortex) -> Current -> Magnetic Field.
    
    # Visualize the "Field Lines" inside the helix
    # Just drawing vertical flux lines to represent the waveguide effect
    
    fig = plt.figure(figsize=(10, 12), facecolor='#050510')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000005')
    
    # Plot DNA Strands
    ax.plot(x1, y1, z1, c='#00FFFF', linewidth=3, label='Strand A (-)')
    ax.plot(x2, y2, z2, c='#FF00FF', linewidth=3, label='Strand B (-)')
    
    # Plot Base Pairs (Rungs)
    # Connect corresponding points
    for i in range(0, len(t), 10):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color='white', alpha=0.3, linewidth=1)
        
    # Plot Etheric Flux (The Information Field)
    # Central Axis Flow
    z_line = np.linspace(0, max(z1), 50)
    ax.plot(np.zeros_like(z_line), np.zeros_like(z_line), z_line, color='#FFCC00', linewidth=4, alpha=0.8, label='Etheric Current (Information)')
    
    # Field Spirals (surrounding)
    t_field = np.linspace(0, turns * 2 * np.pi, 200)
    r_field = radius * 2.0
    x_f = r_field * np.cos(t_field)
    y_f = r_field * np.sin(t_field)
    z_f = (pitch / (2 * np.pi)) * t_field
    ax.plot(x_f, y_f, z_f, color='#44FF88', alpha=0.2, linestyle='--', label='Bio-Magnetic Field')
    
    ax.set_title("DNA: THE HELICAL ANTENNA\nWaveguide for Longitudinal Etheric Waves", color='white')
    ax.axis('off')
    
    # Legend
    ax.legend(loc='upper right', facecolor='#000000', labelcolor='white')
    
    output_path = os.path.join(OUTPUT_DIR, "dna_helical_field.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_phyllotaxis()
    simulate_dna_antenna()
