import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuration ---
STEPS = 1000
DT = 0.05
RESULTS_DIR = Path("results/celestial_demo")

# Physics Parameters
SUN_FREQ = 1.0         # Rotation speed of the phase field
PHASE_DRAG = 0.1       # Strength of the "Phase Wind"
RADIAL_TENSION = 0.05  # Tendency to stay at a specific resonance radius

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_celestial_demo() -> None:
    print("--- TNFR Celestial Mechanics: Orbits as Phase Cycles ---")
    ensure_dir(RESULTS_DIR)
    
    # 1. Initialize Bodies
    # Polaris is fixed at (0,0) - The Pivot of the Sky
    # Sun/Star starts at (10, 0)
    planet_pos = np.array([10.0, 0.0])
    planet_vel = np.array([0.0, 5.0])  # Initial tangential velocity
    
    history_pos = []
    
    print("Simulating Sky Vortex (Polaris-Centric)...")
    
    for t in range(STEPS):
        time = t * DT
        
        # A. Calculate Local Phase Field (The "Ether/Firmament")
        # The Sky rotates as a rigid body around Polaris
        r = np.linalg.norm(planet_pos)
        theta = np.arctan2(planet_pos[1], planet_pos[0])
        
        # Tangential direction vector
        tangent = np.array([-np.sin(theta), np.cos(theta)])
        radial = np.array([np.cos(theta), np.sin(theta)])
        
        # The Phase Field Flow Velocity at radius r
        # Rigid rotation: v = omega * r
        # This creates the "Daily Motion" of stars
        v_field_mag = 0.5 * r
        v_field = v_field_mag * tangent
        
        # B. Apply Nodal Dynamics (Motion)
        # The body is "dragged" by the rotating firmament
        # F_drag = k * (v_field - v_body)
        f_drag = PHASE_DRAG * (v_field - planet_vel)
        
        # Radial Tension (Keeping it in the sky at a specific declination/radius)
        # Without gravity, this is just "Structural Confinement" to a shell
        f_attract = - (20.0 / (r + 0.1)) * radial
        
        # Total Force (Acceleration)
        acc = f_drag + f_attract
        
        # Update Kinematics (Verlet/Euler)
        planet_vel += acc * DT
        planet_pos += planet_vel * DT
        
        history_pos.append(planet_pos.copy())
        
    # --- Visualization ---
    
    # 1. Orbit Plot
    plt.figure(figsize=(8, 8))
    hist = np.array(history_pos)
    
    # Plot Polaris
    plt.plot(0, 0, 'w*', markersize=25, markeredgecolor='b', label='Polaris (Sky Pivot)')
    
    # Plot Orbit
    plt.plot(hist[:, 0], hist[:, 1], 'y-', linewidth=2, label='Sun/Star Path')
    plt.plot(hist[0, 0], hist[0, 1], 'go', label='Start')
    plt.plot(hist[-1, 0], hist[-1, 1], 'ro', label='End')
    
    # Plot Field Vectors (Background)
    x, y = np.meshgrid(np.linspace(-15, 15, 10), np.linspace(-15, 15, 10))
    # Rigid rotation field (u = -wy, v = wx)
    omega = 0.5
    u = -omega * y
    v = omega * x
    plt.quiver(x, y, u, v, alpha=0.2, color='blue', label='Firmament Rotation')
    
    plt.title("Emergence of Orbits from Phase Vortex Drag")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    
    plt.savefig(RESULTS_DIR / "01_orbit_emergence.png")
    print(f"Saved orbit plot to {RESULTS_DIR / '01_orbit_emergence.png'}")
    
    # 2. Velocity Profile Check (Kepler's Law?)
    # v vs r
    rs = np.linalg.norm(hist, axis=1)
    vs = np.linalg.norm(np.diff(hist, axis=0, prepend=hist[0:1]), axis=1) / DT
    
    plt.figure(figsize=(10, 6))
    plt.scatter(rs, vs, c=np.arange(STEPS), cmap='viridis', s=5, alpha=0.5)
    plt.title("Orbital Velocity vs Radius")
    plt.xlabel("Radius (r)")
    plt.ylabel("Velocity (v)")
    plt.grid(True)
    plt.colorbar(label="Time Step")
    
    plt.savefig(RESULTS_DIR / "02_kepler_check.png")
    print(f"Saved velocity check to {RESULTS_DIR / '02_kepler_check.png'}")


if __name__ == "__main__":
    run_celestial_demo()
