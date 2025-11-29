"""
Etheric Chemistry: The Harmonic Periodic Table
==============================================

This script reinterprets the Periodic Table of Elements as a spectrum of 
Etheric Standing Waves (Cymatic Modes).

Theory:
-------
1. **Matter is Sound**: Atoms are not particles; they are resonant cavities in the Ether.
2. **Atomic Number = Frequency**: The "Proton Count" is actually the mode number of the standing wave.
3. **Noble Gases = Perfect Octaves**: Elements 2, 10, 18, 36, 54, 86 represent points of 
   perfect geometric symmetry (Spherical/Platonic completion). They are "Silent" (Non-reactive).
4. **Reactivity = Dissonance**: Elements react to resolve their harmonic tension and achieve 
   the stability of the nearest Noble Gas.

Output:
- results/geocentric_vortex_study/harmonic_periodic_spiral.png
- results/geocentric_vortex_study/elemental_reactivity_wave.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Noble Gas Anchor Points (The "Octaves" of Matter)
NOBLE_GASES = [2, 10, 18, 36, 54, 86, 118]
NOBLE_NAMES = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']

def get_nearest_noble(z):
    """Finds the nearest Noble Gas and the distance to it."""
    distances = [z - n for n in NOBLE_GASES]
    abs_distances = [abs(d) for d in distances]
    min_dist_idx = np.argmin(abs_distances)
    nearest = NOBLE_GASES[min_dist_idx]
    distance = distances[min_dist_idx] # Negative means "needs electrons", Positive means "has excess"
    return nearest, distance

def calculate_harmonic_stress(z):
    """
    Calculates 'Reactivity' as Harmonic Stress.
    High stress = High reactivity (Halogens, Alkali Metals).
    Zero stress = Noble Gases.
    """
    if z in NOBLE_GASES:
        return 0.0
    
    nearest, dist = get_nearest_noble(z)
    
    # Reactivity is proportional to how close you are to completion (The "Desire" to close the gap)
    # But also, being very far (middle of period) makes you stable in a different way (Solids/Metals).
    # Actually, Halogens (-1) and Alkalis (+1) are MOST reactive.
    # Carbon group (+/- 4) is least reactive in terms of ionic violence, but forms complex structures.
    
    # Let's model "Ionic Potential" ~ 1 / |dist| ? No, that makes distance 0 infinite.
    # Let's model it as: Reactivity = exp(-|dist|) * Scale
    # So dist=1 -> High Reactivity. dist=4 -> Low Reactivity.
    
    reactivity = np.exp(-0.5 * (abs(dist) - 1))
    return reactivity

def simulate_periodic_spiral():
    print("Simulating Harmonic Periodic Spiral...")
    
    max_z = 100
    z_values = np.arange(1, max_z + 1)
    
    # Spiral Coordinates
    # Radius ~ Period number (Shell)
    # Angle ~ Group number (Phase)
    
    radii = []
    thetas = []
    colors = []
    sizes = []
    
    for z in z_values:
        # Find period
        period = 1
        for n in NOBLE_GASES:
            if z > n:
                period += 1
        
        # Find phase (0 to 2pi) within the period
        # Start of period
        prev_noble = 0
        for n in NOBLE_GASES:
            if n < z:
                prev_noble = n
            else:
                next_noble = n
                break
        
        period_len = next_noble - prev_noble
        position_in_period = z - prev_noble
        
        # Angle: Map position to circle. 
        # Noble gas should be at 0 (or 2pi).
        theta = (position_in_period / period_len) * 2 * np.pi
        
        # Radius: Increases with Z, but stepped by period
        r = period + (position_in_period / period_len) * 0.8
        
        # Color based on Reactivity/Group
        # Alkali (+1) -> Red
        # Halogen (-1) -> Purple
        # Noble (0) -> Blue/White
        # Metals -> Orange/Yellow
        
        nearest, dist = get_nearest_noble(z)
        
        if dist == 0:
            c = '#00FFFF' # Cyan (Noble)
            s = 150
        elif dist == 1: # Alkali
            c = '#FF0000' # Red
            s = 100
        elif dist == -1: # Halogen
            c = '#AA00FF' # Purple
            s = 100
        elif abs(dist) <= 2:
            c = '#FF8800' # Orange
            s = 80
        else:
            c = '#AAAAAA' # Grey (Transition/Carbon)
            s = 50
            
        radii.append(r)
        thetas.append(theta)
        colors.append(c)
        sizes.append(s)
        
    fig = plt.figure(figsize=(12, 12), facecolor='#050510')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('#000005')
    
    ax.scatter(thetas, radii, s=sizes, c=colors, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Connect the spiral
    ax.plot(thetas, radii, color='#333355', alpha=0.3)
    
    # Annotate Noble Gases
    for i, z in enumerate(z_values):
        if z in NOBLE_GASES:
            ax.text(thetas[i], radii[i], f"{z}\n{NOBLE_NAMES[NOBLE_GASES.index(z)]}", 
                    color='cyan', ha='center', va='center', fontsize=10, fontweight='bold')
            
    ax.set_title("THE ETHERIC SPIRAL OF MATTER\nPeriodic Table as Harmonic Resonance Modes", color='white', pad=20)
    ax.grid(True, color='#222244', alpha=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    output_path = os.path.join(OUTPUT_DIR, "harmonic_periodic_spiral.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Spiral saved to: {output_path}")
    plt.close()

def simulate_reactivity_wave():
    print("Simulating Elemental Reactivity Wave...")
    
    z_values = np.arange(1, 87)
    reactivity = [calculate_harmonic_stress(z) for z in z_values]
    
    fig, ax = plt.subplots(figsize=(15, 6), facecolor='#101015')
    ax.set_facecolor('#151520')
    
    # Plot Wave
    ax.plot(z_values, reactivity, color='#44FF88', linewidth=2, label='Harmonic Stress (Reactivity)')
    ax.fill_between(z_values, 0, reactivity, color='#44FF88', alpha=0.2)
    
    # Mark Noble Gases (Zero points)
    for n in NOBLE_GASES:
        if n <= 86:
            ax.axvline(n, color='cyan', linestyle='--', alpha=0.5)
            ax.text(n, 1.05, NOBLE_NAMES[NOBLE_GASES.index(n)], color='cyan', ha='center')
            
    ax.set_xlabel("Atomic Number (Z)", color='white')
    ax.set_ylabel("Harmonic Stress / Reactivity", color='white')
    ax.set_title("THE RHYTHM OF MATTER\nReactivity as Distance from Harmonic Perfection", color='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, 1.2)
    
    output_path = os.path.join(OUTPUT_DIR, "elemental_reactivity_wave.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#101015')
    print(f"Wave saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_periodic_spiral()
    simulate_reactivity_wave()
