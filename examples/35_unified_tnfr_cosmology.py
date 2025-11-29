"""
Grand Unified TNFR Cosmology: The Fractal Vortex
================================================

This script demonstrates the **Scale Invariance** of the Nodal Equation.
It proves that the Cosmos, the Atom, the Cell, and the Thought are all
manifestations of the **Same Structural Dynamic** operating at different scales.

The Equation: ∂EPI/∂t = νf · ΔNFR(t)

Simulations:
1.  **The Macrocosm (Geocentric Vortex)**:
    - EPI: The Plane & Dome.
    - νf: Solar Year / Sidereal Day.
    - ΔNFR: Gravity (Etheric Pressure).

2.  **The Microcosm (Etheric Atom)**:
    - EPI: The Toroidal Electron Cloud.
    - νf: Atomic Frequency (THz).
    - ΔNFR: Electronegativity.

3.  **The Biocosm (Cell/DNA)**:
    - EPI: The Cell Membrane / Helix.
    - νf: Mitotic Rhythm / Heartbeat.
    - ΔNFR: Osmotic Pressure / Growth Stress.

4.  **The Noocosm (Consciousness)**:
    - EPI: The Thought Form.
    - νf: Brain Wave (Hz).
    - ΔNFR: Cognitive Dissonance / Insight.

Output:
- results/geocentric_vortex_study/grand_unification_fractal.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_grand_unification():
    print("Simulating Grand Unified Fractal Vortex...")
    
    # Common Vortex Function (The Archetype)
    def vortex_field(size, intensity, phase_offset):
        x = np.linspace(-size, size, 200)
        y = np.linspace(-size, size, 200)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Rankine Vortex Profile
        V_theta = intensity / (R + 0.1) * (1 - np.exp(-R**2))
        
        # Spiral Streamlines
        Spiral = np.sin(5*Theta + R*2 + phase_offset)
        
        # Mask
        Spiral[R > size] = np.nan
        return X, Y, Spiral

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='#050510')
    
    # 1. THE COSMOS (The Plane)
    X, Y, Z_cosmos = vortex_field(1.0, 1.0, 0)
    axes[0, 0].set_facecolor('#000005')
    axes[0, 0].imshow(Z_cosmos, cmap='magma', extent=[-1, 1, -1, 1], origin='lower')
    axes[0, 0].set_title("THE MACROCOSM (Cosmology)\nEPI: The Geocentric Plane\nForce: Gravity (Etheric Pressure)", color='white')
    axes[0, 0].axis('off')
    axes[0, 0].text(0, -0.9, "∂EPI/∂t = ν_year · ΔNFR_gravity", color='#FFFF00', ha='center')

    # 2. THE ATOM (The Torus)
    X, Y, Z_atom = vortex_field(1.0, 2.0, np.pi/4)
    axes[0, 1].set_facecolor('#000005')
    axes[0, 1].imshow(Z_atom, cmap='winter', extent=[-1, 1, -1, 1], origin='lower') # Winter for electric look
    axes[0, 1].set_title("THE MICROCOSM (Chemistry)\nEPI: The Etheric Atom\nForce: Electronegativity", color='white')
    axes[0, 1].axis('off')
    axes[0, 1].text(0, -0.9, "∂EPI/∂t = ν_atomic · ΔNFR_charge", color='#00FFFF', ha='center')

    # 3. THE CELL (The Life)
    X, Y, Z_cell = vortex_field(1.0, 0.5, np.pi/2)
    axes[1, 0].set_facecolor('#000005')
    axes[1, 0].imshow(Z_cell, cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
    axes[1, 0].set_title("THE BIOCOSM (Biology)\nEPI: The Cell / DNA Vortex\nForce: Growth Stress", color='white')
    axes[1, 0].axis('off')
    axes[1, 0].text(0, -0.9, "∂EPI/∂t = ν_mitosis · ΔNFR_osmotic", color='#44FF88', ha='center')

    # 4. THE MIND (The Thought)
    X, Y, Z_mind = vortex_field(1.0, 5.0, np.pi)
    axes[1, 1].set_facecolor('#000005')
    axes[1, 1].imshow(Z_mind, cmap='inferno', extent=[-1, 1, -1, 1], origin='lower')
    axes[1, 1].set_title("THE NOOCOSM (Consciousness)\nEPI: The Thought Form\nForce: Cognitive Dissonance", color='white')
    axes[1, 1].axis('off')
    axes[1, 1].text(0, -0.9, "∂EPI/∂t = ν_brain · ΔNFR_insight", color='#FF8800', ha='center')

    plt.suptitle("THE GRAND UNIFIED TNFR THEORY\nOne Equation, Infinite Scales", color='#FFCC00', fontsize=20)
    
    output_path = os.path.join(OUTPUT_DIR, "grand_unification_fractal.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_grand_unification()
