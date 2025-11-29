"""
Geocentric Vortex Cosmology: The Celestial Dome Visualization
============================================================

This script generates a comprehensive visualization of the "Inhabited Reality"
according to the Geocentric Vortex model.

It visualizes:
1. The Stationary Plane (Earth) as the base.
2. The Central Vortex (Polaris) as the axis mundi.
3. The Electromagnetic Firmament (Dome) containing the luminaries.
4. The Sun (Anode) and Moon (Cathode) on their helical tracks.
5. The Wandering Stars (Planets) as resonant nodes.
6. The Fixed Stars as the sonoluminescent grid.

Mathematical Model:
- Coordinates: Polar (r, theta) where r is distance from North Pole.
- Sun Path: r = f(season), theta = 2pi * t (24h cycle).
- Moon Path: r = f(lunar_cycle), theta = 2pi * (t - lag).
- Star Field: Rotating etheric grid.

Output:
- results/geocentric_vortex_study/celestial_dome_map.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_celestial_dome_map():
    """Generates the comprehensive map of the Electromagnetic Firmament."""
    print("Generating Celestial Dome Map...")
    
    # Setup Polar Plot
    fig = plt.figure(figsize=(15, 15), facecolor='#050510')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('#000005')
    
    # 1. The Ice Ring (Boundary)
    # r = 1.0 represents the "Equator" roughly, let's say r=1.5 is the Antarctic Ring
    r_boundary = 1.5
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(theta, [r_boundary]*1000, color='#AADDFF', linewidth=3, alpha=0.8, label='Antarctic Stabilization Ring')
    ax.fill_between(theta, r_boundary, r_boundary+0.2, color='#AADDFF', alpha=0.3)
    
    # 2. The Fixed Star Grid (Sonoluminescent Field)
    # Random stars rotating around Polaris
    np.random.seed(42)
    num_stars = 500
    r_stars = np.random.uniform(0, r_boundary, num_stars)
    theta_stars = np.random.uniform(0, 2*np.pi, num_stars)
    sizes = np.random.uniform(1, 15, num_stars) * (1 - r_stars/r_boundary) # Brighter near center? Or random.
    
    # Plot stars as white/blue dots
    ax.scatter(theta_stars, r_stars, s=sizes, c='white', alpha=0.6, marker='*', label='Fixed Stars (Sonoluminescence)')
    
    # 3. Polaris (The Central Vortex)
    ax.scatter(0, 0, s=300, c='#AAFFFF', marker='*', edgecolors='white', linewidth=2, zorder=10, label='Polaris (Central Vortex)')
    
    # 4. The Sun (The Anode - Golden Frequency)
    # Position: Let's put it at Tropic of Cancer (Summer) for this snapshot
    # r approx 0.5 (North), 1.0 (Equator), 1.5 (South)
    # Summer Solstice: r = 0.6
    sun_theta = np.pi / 4 # Arbitrary time
    sun_r = 0.6
    
    # Sun Glow
    ax.scatter(sun_theta, sun_r, s=800, c='#FFD700', alpha=0.3, zorder=5) # Glow
    ax.scatter(sun_theta, sun_r, s=200, c='#FFFF00', edgecolors='#FFAA00', linewidth=2, zorder=6, label='Sun (Anode/Hot Spot)')
    
    # Sun's Path (The Day's Circuit)
    ax.plot(theta, [sun_r]*1000, color='#FFD700', linestyle='--', alpha=0.4, linewidth=1)
    
    # 5. The Moon (The Cathode - Silver Frequency)
    # Position: Opposite phase roughly, maybe waning
    moon_theta = sun_theta + np.pi + 0.5
    moon_r = 0.65 # Slightly different track
    
    # Moon Glow (Cool Light)
    ax.scatter(moon_theta, moon_r, s=700, c='#AAAAFF', alpha=0.2, zorder=5)
    ax.scatter(moon_theta, moon_r, s=180, c='#DDDDFF', edgecolors='#8888FF', linewidth=2, zorder=6, label='Moon (Cathode/Cold Plasma)')
    
    # 6. The Wandering Stars (Planets as Cymatic Nodes)
    # Venus (Geometric Rose pattern hint)
    venus_theta = sun_theta - 0.8
    venus_r = 0.55
    ax.scatter(venus_theta, venus_r, s=120, c='#00FF88', marker='D', edgecolors='white', zorder=7, label='Venus (Harmonic Node)')
    
    # Mars (Red Node)
    mars_theta = sun_theta + 2.0
    mars_r = 0.8
    ax.scatter(mars_theta, mars_r, s=100, c='#FF4444', marker='o', edgecolors='red', zorder=7, label='Mars (Dissonant Node)')
    
    # Jupiter (Grand Resonator)
    jup_theta = sun_theta + 3.5
    jup_r = 1.1
    ax.scatter(jup_theta, jup_r, s=160, c='#FFCCAA', marker='h', edgecolors='orange', zorder=7, label='Jupiter (Resonant Stabilizer)')
    
    # Saturn (The Limit Keeper)
    sat_theta = sun_theta + 5.0
    sat_r = 1.3
    ax.scatter(sat_theta, sat_r, s=140, c='#DDAA55', marker='p', edgecolors='brown', zorder=7, label='Saturn (Boundary Keeper)')

    # Annotations
    plt.title("THE GEOCENTRIC VORTEX COSMOLOGY\nSnapshot of the Electromagnetic Firmament", color='white', fontsize=16, pad=20)
    
    # Custom Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#101020', edgecolor='#444488')
    for text in legend.get_texts():
        text.set_color('white')
        
    # Grid styling
    ax.grid(True, color='#222244', linestyle=':', alpha=0.5)
    ax.set_yticklabels([]) # Hide radial labels
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], color='#6666AA')
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "celestial_dome_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_celestial_dome_map()
