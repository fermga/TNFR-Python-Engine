import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuration ---
DT = 0.01
RESULTS_DIR = Path("results/geocentric_vortex_study")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class GeocentricPlanet:
    def __init__(self, name, period_helio, radius_helio, color, type='outer'):
        self.name = name
        self.color = color
        self.type = type
        
        # Orbital Parameters (Simplified circular model)
        self.T_p = period_helio
        self.R_p = radius_helio
        
        # Earth/Sun Parameters
        self.T_e = 1.0  # 1 Year
        self.R_e = 1.0  # 1 AU
        
        # Angular Velocities
        self.w_p = 2 * np.pi / self.T_p
        self.w_e = 2 * np.pi / self.T_e
        
    def get_position(self, t):
        # Heliocentric Positions
        # Earth
        xe = self.R_e * np.cos(self.w_e * t)
        ye = self.R_e * np.sin(self.w_e * t)
        
        # Planet
        xp = self.R_p * np.cos(self.w_p * t)
        yp = self.R_p * np.sin(self.w_p * t)
        
        # Geocentric Position (Vector Subtraction: P - E)
        # This is equivalent to the Deferent/Epicycle sum in Nodal Dynamics
        x_geo = xp - xe
        y_geo = yp - ye
        
        return np.array([x_geo, y_geo])

def run_mandalas_demo() -> None:
    print("--- TNFR Planetary Mandalas: The Geometry of Resonance ---")
    ensure_dir(RESULTS_DIR)
    
    # Define Planets with Real Data (approx)
    planets = [
        # Name, Period (Yr), Radius (AU), Color, Type
        GeocentricPlanet("Venus", 0.615, 0.723, 'magenta', 'inner'),
        GeocentricPlanet("Mars", 1.881, 1.524, 'red', 'outer'),
        GeocentricPlanet("Jupiter", 11.86, 5.203, 'orange', 'outer'),
        GeocentricPlanet("Mercury", 0.241, 0.387, 'gray', 'inner')
    ]
    
    # Simulation Durations (to close the loops)
    # Venus: 8 Years (13 orbits) -> The Rose
    # Mars: 15 Years
    # Jupiter: 12 Years
    durations = {
        "Venus": 8.0,
        "Mars": 16.0,
        "Jupiter": 12.0,
        "Mercury": 1.0
    }
    
    plt.figure(figsize=(20, 5))
    
    for i, planet in enumerate(planets):
        print(f"Simulating {planet.name}...")
        
        duration = durations[planet.name]
        steps = int(duration / DT * 100) # High resolution
        
        path_list = []
        for s in range(steps):
            t = s * DT / 100  # Slower time for smoothness
            pos = planet.get_position(t)
            path_list.append(pos)
            
        path = np.array(path_list)
        
        # Plot
        ax = plt.subplot(1, 4, i+1)
        ax.plot(0, 0, 'b+', markersize=10, label='Earth')
        ax.plot(path[:, 0], path[:, 1], '-', color=planet.color, linewidth=0.8, alpha=0.8)
        
        ax.set_title(f"{planet.name} ({duration} Years)")
        ax.axis('equal')
        ax.axis('off') # Make it look like art
        
    plt.suptitle("The Flowers of the Sky: Geocentric Resonant Patterns")
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / "01_planetary_flowers.png")
    print(f"Saved mandalas to {RESULTS_DIR / '01_planetary_flowers.png'}")
    
    # Detail: The Venus Rose
    print("Generating High-Res Venus Rose...")
    venus = planets[0]
    duration = 8.0
    steps = int(duration / DT * 500)
    
    path_list = []
    for s in range(steps):
        t = s * DT / 500
        pos = venus.get_position(t)
        path_list.append(pos)
    path = np.array(path_list)
    
    plt.figure(figsize=(10, 10))
    plt.plot(path[:, 0], path[:, 1], 'm-', linewidth=1.5)
    plt.plot(0, 0, 'b+', markersize=20, markeredgewidth=3)
    plt.title("The Rose of Venus (8 Earth Years / 13 Venus Years)")
    plt.axis('equal')
    plt.axis('off')
    
    plt.savefig(RESULTS_DIR / "02_venus_rose.png")
    print(f"Saved Venus Rose to {RESULTS_DIR / '02_venus_rose.png'}")

if __name__ == "__main__":
    run_mandalas_demo()
