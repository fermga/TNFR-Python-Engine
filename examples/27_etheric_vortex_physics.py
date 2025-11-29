"""
Geocentric Vortex Cosmology: Etheric Field Dynamics
===================================================

This script simulates the fluid dynamics of the Etheric Vortex that drives the cosmos.
It models the "Sky" not as empty space, but as a Superfluid Medium in toroidal rotation.

Physics Model:
--------------
1. **Rankine Vortex Structure**:
   - **Core (Polaris)**: Solid-body rotation (V ~ r).
   - **Field (Firmament)**: Irrotational vortex (V ~ 1/r).
   
2. **Pressure/Density Gradient**:
   - Pressure increases with distance from center (Centrifugal effect).
   - This creates the "Dome" shape (Isobars).

3. **Toroidal Flow (The Gravity Mechanism)**:
   - The Ether flows DOWN towards the plane, creating the "Vertical Pressure" we call Gravity.
   - It flows OUT along the plane (Radial breeze).
   - It flows UP at the Antarctic boundary.
   - It flows IN at the top of the Dome.

Output:
- results/geocentric_vortex_study/etheric_velocity_field.png (Top Down)
- results/geocentric_vortex_study/toroidal_gravity_flow.png (Cross Section)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_top_down_vortex():
    print("Simulating Top-Down Etheric Vortex Velocity Field...")
    
    # Grid setup
    N = 100
    L = 2.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    # Rankine Vortex Model
    # Core radius
    R_core = 0.2
    Gamma = 1.0 # Circulation strength
    
    # Velocity magnitude
    # Inside core: V = (Gamma * r) / (2 * pi * R_core^2)
    # Outside core: V = Gamma / (2 * pi * r)
    
    V_mag = np.zeros_like(R)
    mask_core = R < R_core
    mask_field = R >= R_core
    
    # Avoid division by zero at exact center
    R_safe = np.maximum(R, 1e-6)
    
    V_mag[mask_core] = (Gamma * R[mask_core]) / (2 * np.pi * R_core**2)
    V_mag[mask_field] = Gamma / (2 * np.pi * R_safe[mask_field])
    
    # Vector components (Tangential flow)
    # Vx = -V * sin(theta)
    # Vy = V * cos(theta)
    U = -V_mag * np.sin(Theta)
    V = V_mag * np.cos(Theta)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Streamlines
    strm = ax.streamplot(X, Y, U, V, color=V_mag, cmap='plasma', density=1.5, linewidth=1, arrowsize=1)
    
    # Luminaries overlay
    # Sun at r=0.8
    ax.scatter(0.8, 0, s=200, c='#FFD700', edgecolors='white', zorder=10, label='Sun (Carried by Ether)')
    # Moon at r=0.9
    ax.scatter(-0.9, 0, s=150, c='#AAAAFF', edgecolors='white', zorder=10, label='Moon')
    
    # Annotations
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_title("THE ETHERIC WHIRLPOOL\nTop-Down Velocity Field of the Firmament", color='white', pad=20)
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Etheric Velocity (c)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    output_path = os.path.join(OUTPUT_DIR, "etheric_velocity_field.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Velocity field saved to: {output_path}")
    plt.close()

def simulate_toroidal_cross_section():
    print("Simulating Toroidal Gravity Flow (Cross Section)...")
    
    # Grid setup (R, Z)
    # R from 0 to 2
    # Z from 0 to 1 (Dome height)
    N = 50
    r = np.linspace(0, 2.0, N)
    z = np.linspace(0, 1.5, N)
    R, Z = np.meshgrid(r, z)
    
    # Toroidal Flow Model (Poloidal component)
    # Flow goes DOWN at center (Gravity source?) 
    # Actually, in the "Dielectric Acceleration" model, the potential gradient is uniform down.
    # But let's model a dynamic torus.
    # Center of torus tube at (R=1, Z=0.5) approx?
    
    # Simple model:
    # Vr = (Z - Zc)
    # Vz = -(R - Rc)
    # This creates circular flow around (Rc, Zc)
    
    Rc = 1.0
    Zc = 0.75
    
    # We want flow:
    # DOWN at R < Rc (Inner system) -> Vz < 0
    # OUT at Z near 0 (Ground) -> Vr > 0
    # UP at R > Rc (Outer boundary) -> Vz > 0
    # IN at Z near Top (Dome) -> Vr < 0
    
    Vr = (Z - Zc)
    Vz = -(R - Rc)
    
    # Mask out the "Ground" (Z<0) and "Dome" (R>2)
    
    # Magnitude for color
    Mag = np.sqrt(Vr**2 + Vz**2)
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Quiver plot
    q = ax.quiver(R, Z, Vr, Vz, Mag, cmap='coolwarm', pivot='mid', scale=20, width=0.003)
    
    # Draw the Plane
    ax.axhline(0, color='#4488FF', linewidth=3)
    ax.text(0.2, -0.1, "STATIONARY EARTH PLANE", color='#4488FF', fontweight='bold')
    
    # Draw the Dome Boundary
    theta = np.linspace(0, np.pi/2, 100)
    dome_r = 1.8 * np.sin(theta) # Just an arc
    dome_z = 1.4 * np.cos(theta) # Inverted? No.
    # Ellipse quadrant
    x_dome = 1.8 * np.cos(np.linspace(0, np.pi/2, 100)) # 1.8 to 0? No.
    # Let's just draw a box or arc
    ax.plot(np.linspace(0, 2, 100), np.sqrt(4 - np.linspace(0, 2, 100)**2)*0.7, color='#FFCC00', linestyle='--', alpha=0.5)
    
    # Annotations
    ax.text(0.5, 0.5, "DOWNWARD ETHERIC PRESSURE\n(GRAVITY)", color='cyan', ha='center', fontsize=10, alpha=0.8)
    ax.text(1.5, 0.5, "UPWARD RETURN FLOW\n(ANTARCTIC WALL)", color='red', ha='center', fontsize=10, alpha=0.8)
    
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.2, 1.6)
    ax.set_title("THE TOROIDAL BREATH\nCross-Section of Etheric Circulation", color='white', pad=20)
    ax.axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "toroidal_gravity_flow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Toroidal flow saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_top_down_vortex()
    simulate_toroidal_cross_section()
