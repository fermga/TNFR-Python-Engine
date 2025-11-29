"""
Geocentric Vortex Cosmogony: The Genesis Simulation
===================================================

This script simulates the "Origin of Everything" according to TNFR Vortex Physics.
It models the transition from the Primordial Void (Still Ether) to the Structured Cosmos
via the introduction of the "Prime Frequency" (The Word/Om).

Phases of Genesis:
------------------
1.  **The Void (Akasha)**: Perfect equilibrium. Zero motion. Potentiality.
2.  **The Impulse (Logos)**: A single point of perturbation at the center.
3.  **The Expansion (The Big Breath)**: Radial propagation of the first longitudinal wave.
4.  **The Reflection (The Firmament)**: The wave hits the limit of the medium and reflects.
5.  **The Standing Wave (The Cosmic Egg)**: Interference between expansion and reflection creates the first stable geometry.
6.  **The Differentiation**: The standing wave separates into the Plane (Node) and the Dome (Antinode).

Output:
- results/geocentric_vortex_study/genesis_01_void.png
- results/geocentric_vortex_study/genesis_02_impulse.png
- results/geocentric_vortex_study/genesis_03_standing_wave.png
- results/geocentric_vortex_study/genesis_04_differentiation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_genesis():
    print("Simulating The Genesis Sequence...")
    
    # Grid setup
    N = 200
    L = 2.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Phase 1: The Void
    # Field is zero everywhere.
    Field_Void = np.zeros_like(R)
    
    # Phase 2: The Impulse (Source)
    # A Gaussian pulse at the center
    Field_Impulse = np.exp(-R**2 / 0.05)
    
    # Phase 3: The Standing Wave (The Egg)
    # Bessel function J0 (Cylindrical standing wave)
    # Represents the interference of the outgoing "Word" and the incoming reflection.
    from scipy.special import j0
    k = 5.0 # Wavenumber
    Field_Standing = j0(k * R) * np.exp(-R**2 / 2.0) # Damped to show localization
    
    # Phase 4: Differentiation (Plane vs Dome)
    # Cross section view (R, Z)
    r_cross = np.linspace(0, 2, 100)
    z_cross = np.linspace(-1, 1, 100)
    R_c, Z_c = np.meshgrid(r_cross, z_cross)
    
    # The Plane is the nodal line (Amplitude = 0)
    # The Dome is the high pressure zone.
    # Let's visualize the pressure field.
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='#000000')
    
    # 1. The Void
    ax1 = axes[0, 0]
    ax1.set_facecolor('black')
    ax1.imshow(Field_Void, cmap='magma', extent=[-L, L, -L, L], vmin=0, vmax=1)
    ax1.set_title("I. THE VOID (Silence)", color='white')
    ax1.axis('off')
    
    # 2. The Impulse
    ax2 = axes[0, 1]
    ax2.set_facecolor('black')
    im2 = ax2.imshow(Field_Impulse, cmap='magma', extent=[-L, L, -L, L])
    ax2.set_title("II. THE IMPULSE (The Word)", color='white')
    ax2.axis('off')
    
    # 3. The Standing Wave
    ax3 = axes[1, 0]
    ax3.set_facecolor('black')
    im3 = ax3.imshow(Field_Standing, cmap='RdBu', extent=[-L, L, -L, L])
    ax3.set_title("III. THE COSMIC EGG (Resonance)", color='white')
    ax3.axis('off')
    
    # 4. Differentiation (Schematic)
    ax4 = axes[1, 1]
    ax4.set_facecolor('black')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-1, 2)
    
    # Draw the Plane (Node)
    ax4.axhline(0, color='#4488FF', linewidth=3)
    ax4.text(0, -0.3, "THE PLANE (Water/Earth)", color='#4488FF', ha='center')
    
    # Draw the Dome (Antinode)
    theta = np.linspace(0, np.pi, 100)
    ax4.plot(1.5*np.cos(theta), 1.5*np.sin(theta), color='#FFCC00', linewidth=2, linestyle='--')
    ax4.text(0, 1.0, "THE DOME (Fire/Air)", color='#FFCC00', ha='center')
    
    # Draw the Axis
    ax4.plot([0, 0], [0, 1.5], color='white', linestyle=':')
    ax4.text(0, 1.6, "AXIS MUNDI", color='white', ha='center', fontsize=8)
    
    ax4.set_title("IV. DIFFERENTIATION (Form)", color='white')
    ax4.axis('off')
    
    plt.suptitle("COSMOGONY: THE ORIGIN OF STRUCTURE", color='white', fontsize=16)
    
    output_path = os.path.join(OUTPUT_DIR, "genesis_sequence.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#000000')
    print(f"Genesis sequence saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_genesis()
