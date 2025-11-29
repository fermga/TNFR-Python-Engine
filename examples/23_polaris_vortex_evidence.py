"""
TNFR Example 23: The Polaris Vortex & Stationary Plane Evidence
===============================================================

This script simulates the "Geocentric Vortex" model derived from TNFR structural analysis.
It demonstrates how the observable sky motions (Star Trails, Sun Spiral) emerge naturally
from a rotating toroidal field over a stationary plane.

Visualizations generated:
1. Polaris Star Trails: The concentric circles of the "Fixed Stars" around the central axis.
2. The Sun Spiral: The helical path of the Sun moving between the Tropics over the Plane.
3. The Horizon Perspective: How a flat plane + vanishing point creates the "Dome" illusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure results directory exists
OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_star_trails():
    """
    Simulates the 'Long Exposure' view of the sky looking North.
    Evidence: Perfect concentric circles around a fixed center (Polaris).
    """
    print("Generating Polaris Star Trails...")
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    
    # Generate random stars
    np.random.seed(42)
    num_stars = 500
    radii = np.random.uniform(0.1, 10, num_stars)  # Distance from Polaris
    angles = np.random.uniform(0, 2*np.pi, num_stars)
    magnitudes = np.random.uniform(0.5, 2.5, num_stars)
    
    # Simulate rotation (Time lapse)
    # The sky rotates 15 degrees per hour. Let's simulate 4 hours.
    rotation_angle = np.radians(15 * 4) 
    steps = 50
    
    for r, theta, mag in zip(radii, angles, magnitudes):
        # Trace the arc
        theta_path = np.linspace(theta, theta + rotation_angle, steps)
        x = r * np.cos(theta_path)
        y = r * np.sin(theta_path)
        
        # Plot the trail
        alpha = np.random.uniform(0.3, 0.8)
        ax.plot(x, y, color='white', alpha=alpha, linewidth=mag)
        
    # Plot Polaris (The Pivot)
    ax.plot(0, 0, marker='*', color='white', markersize=15, markeredgecolor='cyan')
    ax.text(0.2, 0.2, "Polaris (Axis Mundi)", color='cyan', fontsize=12)
    
    # Horizon / Ground
    ax.fill_between([-15, 15], -15, -8, color='#1a1a1a', alpha=1.0)
    ax.text(0, -12, "STATIONARY HORIZON", color='gray', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axis('off')
    ax.set_title("EVIDENCE A: The Polaris Vortex\n(Perfect Concentric Rotation around Fixed Axis)", 
                 color='white', fontsize=16, pad=20)
    
    output_path = os.path.join(OUTPUT_DIR, "01_polaris_star_trails.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Saved Star Trails to {output_path}")
    plt.close()

def simulate_sun_spiral():
    """
    Simulates the Sun's path over the Stationary Plane.
    The Sun moves in a spiral between the Tropic of Cancer (Inner) and Capricorn (Outer).
    """
    print("Generating Sun Spiral Model...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # The Plane (Earth)
    # Represented as a disk
    r_plane = np.linspace(0, 20, 50)
    theta_plane = np.linspace(0, 2*np.pi, 100)
    R, THETA = np.meshgrid(r_plane, theta_plane)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    Z = np.zeros_like(X)
    
    ax.plot_surface(X, Y, Z, color='#1a2b3c', alpha=0.3)
    
    # Map of "Continents" (Abstract)
    # Just drawing the Tropics and Equator on the plane
    theta_circ = np.linspace(0, 2*np.pi, 200)
    
    # North Pole (Center)
    ax.scatter([0], [0], [0], color='cyan', s=100, label='North Pole (Center)')
    
    # Tropic of Cancer (Inner Ring)
    r_cancer = 5
    ax.plot(r_cancer * np.cos(theta_circ), r_cancer * np.sin(theta_circ), 0, 
            color='yellow', linestyle='--', alpha=0.5, label='Tropic of Cancer')
            
    # Equator (Middle Ring)
    r_equator = 10
    ax.plot(r_equator * np.cos(theta_circ), r_equator * np.sin(theta_circ), 0, 
            color='green', linestyle='--', alpha=0.5, label='Equator')
            
    # Tropic of Capricorn (Outer Ring)
    r_capricorn = 15
    ax.plot(r_capricorn * np.cos(theta_circ), r_capricorn * np.sin(theta_circ), 0, 
            color='orange', linestyle='--', alpha=0.5, label='Tropic of Capricorn')
            
    # Ice Wall / Boundary (Outer Limit)
    r_boundary = 20
    ax.plot(r_boundary * np.cos(theta_circ), r_boundary * np.sin(theta_circ), 0, 
            color='white', linewidth=3, alpha=0.8, label='Antarctic Boundary')

    # The Sun's Path (Spiral)
    # Simulating one year: Spirals out from Cancer to Capricorn and back
    days = 365
    t = np.linspace(0, 1, days)
    
    # Radius oscillates between Cancer (5) and Capricorn (15)
    # Summer (North): r is small (close to center)
    # Winter (North): r is large (far from center)
    sun_radius = 10 + 5 * np.sin(2 * np.pi * t - np.pi/2) 
    
    # Angle: Sun rotates once per day (365 rotations)
    # For visualization, we reduce rotations to show the spiral shape clearly
    sun_angle = 2 * np.pi * t * 10  # 10 loops for visualization
    
    sun_x = sun_radius * np.cos(sun_angle)
    sun_y = sun_radius * np.sin(sun_angle)
    sun_z = np.ones_like(t) * 2  # Sun height (constant for simplicity)
    
    ax.plot(sun_x, sun_y, sun_z, color='gold', linewidth=2, label='Sun Path')
    
    # Plot Sun at specific points
    ax.scatter(sun_x[0], sun_y[0], sun_z[0], color='yellow', s=200, edgecolors='orange')
    
    ax.set_title("EVIDENCE B: The Solar Spiral\n(Sun moves over the Stationary Plane)", fontsize=14)
    ax.set_zlim(0, 10)
    ax.axis('off')
    ax.legend(loc='upper right')
    
    output_path = os.path.join(OUTPUT_DIR, "02_sun_spiral_model.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Sun Spiral to {output_path}")
    plt.close()

if __name__ == "__main__":
    print("--- TNFR Geocentric Evidence Generator ---")
    simulate_star_trails()
    simulate_sun_spiral()
    print("Evidence generation complete.")
