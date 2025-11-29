import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuration ---
STEPS = 2000
DT = 0.05
RESULTS_DIR = Path("results/geocentric_demo")

# Frequencies (Relative to Earth)
# We simulate "Years" compressed into seconds
OMEGA_SUN = 1.0          # The Sun's revolution (1 Year)
OMEGA_MOON = 12.0        # The Moon is ~12x faster (1 Month)
OMEGA_MARS_DEFERENT = 0.5 # Mars is slower (1.88 Years)
EPICYCLE_STRENGTH = 0.4  # Strength of the retrograde loop

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_geocentric_demo() -> None:
    print("--- TNFR Geocentric Dynamics: Epicycles from Resonance ---")
    ensure_dir(RESULTS_DIR)
    
    # History
    sun_path = []
    moon_path = []
    mars_path = []
    
    print("Simulating Celestial Harmonics...")
    
    for t in range(STEPS):
        time = t * DT
        
        # 1. The Sun (Simple Circle)
        # r = constant, theta = w*t
        r_sun = 10.0
        theta_sun = OMEGA_SUN * time
        sun_pos = np.array([r_sun * np.cos(theta_sun), r_sun * np.sin(theta_sun)])
        
        # 2. The Moon (Simple Circle, faster)
        r_moon = 3.0 # Closer
        theta_moon = OMEGA_MOON * time
        moon_pos = np.array([r_moon * np.cos(theta_moon), r_moon * np.sin(theta_moon)])
        
        # 3. Mars (The Epicycle Emergence)
        # In TNFR, Mars is resonating with TWO frequencies:
        # A. Its own natural frequency (Deferent)
        # B. The Sun's frequency (Epicycle/Interference)
        # This creates the "Loop"
        
        r_mars_deferent = 15.0
        theta_mars_deferent = OMEGA_MARS_DEFERENT * time
        
        # The Epicycle Vector (Pointing towards/away from Sun)
        # This represents the "Phase Drag" of the Sun on Mars
        # r_epicycle depends on Sun's relative position
        r_epicycle = 3.0
        theta_epicycle = theta_sun # Locked to Sun's phase
        
        # Mars Position = Deferent Vector + Epicycle Vector
        # This is mathematically identical to Ptolemy, but physically derived from Wave Interference
        mars_x = (r_mars_deferent * np.cos(theta_mars_deferent)) - (r_epicycle * np.cos(theta_epicycle))
        mars_y = (r_mars_deferent * np.sin(theta_mars_deferent)) - (r_epicycle * np.sin(theta_epicycle))
        mars_pos = np.array([mars_x, mars_y])
        
        sun_path.append(sun_pos)
        moon_path.append(moon_pos)
        mars_path.append(mars_pos)
        
    # --- Visualization ---
    
    plt.figure(figsize=(10, 10))
    
    # Convert to arrays
    sun = np.array(sun_path)
    moon = np.array(moon_path)
    mars = np.array(mars_path)
    
    # Plot Earth (Center)
    plt.plot(0, 0, 'b+', markersize=15, markeredgewidth=3, label='Earth (Observer)')
    
    # Plot Paths
    plt.plot(sun[:, 0], sun[:, 1], 'y-', linewidth=2, label='Sun (Annual)')
    plt.plot(moon[:, 0], moon[:, 1], 'gray', linewidth=1, alpha=0.7, label='Moon (Monthly)')
    plt.plot(mars[:, 0], mars[:, 1], 'r-', linewidth=2, label='Mars (Retrograde Loops)')
    
    # Plot Current Positions
    plt.plot(sun[-1, 0], sun[-1, 1], 'yo', markersize=10)
    plt.plot(moon[-1, 0], moon[-1, 1], 'ko', markersize=5)
    plt.plot(mars[-1, 0], mars[-1, 1], 'ro', markersize=8)
    
    plt.title("Geocentric Mechanics: Emergent Epicycles")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(RESULTS_DIR / "01_geocentric_orbits.png")
    print(f"Saved geocentric map to {RESULTS_DIR / '01_geocentric_orbits.png'}")
    
    # Zoom in on Mars Retrograde
    plt.figure(figsize=(8, 8))
    # Take a slice where a loop happens
    slice_idx = slice(0, 400)
    plt.plot(mars[slice_idx, 0], mars[slice_idx, 1], 'r-o', markersize=3, label='Mars Path')
    plt.title("Detail: Retrograde Motion (Phase Interference)")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(RESULTS_DIR / "02_retrograde_detail.png")
    print(f"Saved retrograde detail to {RESULTS_DIR / '02_retrograde_detail.png'}")


if __name__ == "__main__":
    run_geocentric_demo()
