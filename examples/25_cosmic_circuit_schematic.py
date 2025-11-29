"""
Geocentric Vortex Cosmology: The Electromagnetic Circuit
======================================================

This script visualizes the "Universe as a Circuit" concept.
It creates a schematic diagram showing:
1. The Earth Plane (Negative Plate).
2. The Ionosphere/Dome (Positive Plate).
3. The Central Vortex (Induction Coil).
4. The Sun/Moon (Anode/Cathode).
5. The Toroidal Field Lines connecting the system.

Output:
- results/geocentric_vortex_study/cosmic_circuit_schematic.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_circuit_schematic():
    print("Generating Cosmic Circuit Schematic...")
    
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#101015')
    ax.set_facecolor('#101015')
    
    # 1. The Earth Plane (Ground/Negative)
    # Draw a flat line at y=0
    ax.axhline(y=0, color='#4488FF', linewidth=4, label='Salt Water Electrolyte (Ground)')
    ax.fill_between([-10, 10], -2, 0, color='#224488', alpha=0.5)
    ax.text(0, -1, "STATIONARY EARTH PLANE (NEGATIVE PLATE)", color='#88CCFF', ha='center', fontsize=14, fontweight='bold')
    
    # 2. The Dome (Ionosphere/Positive)
    # Draw a semi-circle arc
    theta = np.linspace(0, np.pi, 100)
    r = 8
    x_dome = r * np.cos(theta)
    y_dome = r * np.sin(theta)
    ax.plot(x_dome, y_dome, color='#FFCC00', linewidth=3, linestyle='--', label='Ionosphere (Positive Plate)')
    
    # 3. The Central Vortex (Axis Mundi)
    # Vertical line at x=0
    ax.plot([0, 0], [0, r], color='#AAFFFF', linewidth=2, linestyle='-.')
    ax.text(0.2, r/2, "AXIS MUNDI\n(VORTEX)", color='#AAFFFF', fontsize=10, rotation=90)
    
    # 4. Toroidal Field Lines
    # Curves from center out to dome
    for scale in [0.3, 0.5, 0.7, 0.9]:
        x_field = x_dome * scale
        y_field = y_dome * scale * 0.8 # Flattened torus
        ax.plot(x_field, y_field, color='#AAFFFF', alpha=0.2, linewidth=1)
        
    # 5. The Sun (Anode)
    sun_x = 3
    sun_y = 4
    ax.scatter(sun_x, sun_y, s=500, c='#FFD700', edgecolors='white', zorder=10, label='Sun (Anode)')
    # Lightning/Induction from Sun to Earth
    ax.arrow(sun_x, sun_y, 0, -3, color='#FFD700', width=0.05, head_width=0.2, alpha=0.6)
    ax.text(sun_x + 0.5, sun_y, "SUN\n(Anode)", color='#FFD700', fontsize=12)
    
    # 6. The Moon (Cathode)
    moon_x = -3
    moon_y = 3.5
    ax.scatter(moon_x, moon_y, s=450, c='#AAAAFF', edgecolors='white', zorder=10, label='Moon (Cathode)')
    ax.text(moon_x - 1.5, moon_y, "MOON\n(Cathode)", color='#AAAAFF', fontsize=12)
    
    # 7. Stars (Sonoluminescence)
    # Scattered in the upper dome area
    star_x = np.random.uniform(-7, 7, 50)
    star_y = np.random.uniform(5, 7.5, 50)
    # Filter to keep inside dome roughly
    mask = (star_x**2 + star_y**2) < r**2
    ax.scatter(star_x[mask], star_y[mask], s=20, c='white', alpha=0.8, marker='*')
    
    # 8. Circuit Symbols
    # Battery symbol at the center bottom? Or just field lines.
    # Let's draw "Capacitor" plates logic
    
    # Annotations
    ax.set_xlim(-9, 9)
    ax.set_ylim(-2, 9)
    ax.axis('off')
    
    plt.title("THE COSMIC SOLID-STATE CIRCUIT\nElectromagnetic Etheric System", color='white', fontsize=20, pad=20)
    
    # Legend
    legend = ax.legend(loc='lower right', facecolor='#202030', edgecolor='#444488')
    for text in legend.get_texts():
        text.set_color('white')
        
    output_path = os.path.join(OUTPUT_DIR, "cosmic_circuit_schematic.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#101015')
    print(f"Schematic saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_circuit_schematic()
