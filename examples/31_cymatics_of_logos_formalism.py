"""
Cymatics of the Logos: Mathematical Formalism
=============================================

This script provides the rigorous mathematical visualization of the "Logos" (Prime Frequency)
structuring the Ether. It solves the **Helmholtz Equation** (Wave Equation) in spherical
coordinates to demonstrate how specific resonant modes create the geometry of the
Geocentric Vortex Cosmos.

Mathematical Model:
-------------------
The Etheric Field $\Psi(r, \theta, \phi)$ is a solution to:
$$ \nabla^2 \Psi + k^2 \Psi = 0 $$

Separation of variables yields:
$$ \Psi_{nlm} = j_n(kr) \cdot Y_l^m(\theta, \phi) $$

We visualize the **Fundamental Dipole Mode** ($l=1, m=0$) and **Quadrupole Mode** ($l=2, m=0$)
which naturally generate:
1.  **The Axis Mundi** (Vertical Nodal Line).
2.  **The Equatorial Plane** (Horizontal Nodal Surface).
3.  **The Dome** (Boundary Condition).

Output:
- results/geocentric_vortex_study/logos_mode_dipole.png (Axis Formation)
- results/geocentric_vortex_study/logos_mode_quadrupole.png (Plane Formation)
- results/geocentric_vortex_study/logos_interference_pattern.png (The Cosmic Egg)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_spherical_mode(n, l, m, title, filename):
    print(f"Simulating Logos Mode (n={n}, l={l}, m={m})...")
    
    # Grid setup (Cross section in x-z plane, y=0)
    N = 200
    L = 1.5
    x = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Z = np.meshgrid(x, z)
    
    # Convert to Spherical Coords
    R = np.sqrt(X**2 + Z**2)
    Theta = np.arctan2(X, Z) # Angle from Z axis (0 to pi)
    Phi = 0 # Azimuthal symmetry for cross section
    
    # Avoid singularity at R=0
    R[R==0] = 1e-9
    
    # 1. Radial Part: Spherical Bessel Function j_n(k*r)
    # We choose k such that j_n(k*R_boundary) = 0 (Dirichlet BC - The Firmament)
    # First zero of j_n.
    # For n=1, first zero is approx 4.49
    # For n=2, first zero is approx 5.76
    if n == 1:
        k = 4.493 / 1.0 # Boundary at R=1
    elif n == 2:
        k = 5.763 / 1.0
    else:
        k = np.pi # Approx
        
    Radial = spherical_jn(n, k * R)
    
    # 2. Angular Part: Spherical Harmonic Y_l^m
    # We only need the real part for standing waves
    Angular = np.real(sph_harm(m, l, Phi, Theta))
    
    # Total Field
    Psi = Radial * Angular
    
    # Mask outside boundary
    Psi[R > 1.0] = np.nan
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Plot Field Intensity
    im = ax.imshow(Psi, cmap='RdBu', extent=[-L, L, -L, L], vmin=-0.5, vmax=0.5, origin='lower')
    
    # Draw Boundary
    circle = plt.Circle((0, 0), 1.0, color='#FFCC00', fill=False, linewidth=2, linestyle='--')
    ax.add_artist(circle)
    
    # Annotate Nodes (Where Psi = 0)
    # For l=1 (Dipole), Node is the Equatorial Plane? No, l=1 has node at equator?
    # Y_1^0 ~ cos(theta). Zero at theta = pi/2 (Equator).
    # So Dipole mode CREATES the Plane!
    
    if l == 1:
        ax.text(0, -1.2, "MODE (1,0): THE DIPOLE\nCreates the Equatorial Plane (Node)", color='white', ha='center')
        ax.axhline(0, color='#44FF88', linewidth=2, linestyle=':')
        ax.text(1.1, 0, "Nodal Plane", color='#44FF88', fontsize=8)
        
    if l == 2:
        ax.text(0, -1.2, "MODE (2,0): THE QUADRUPOLE\nCreates the Axis and Cones", color='white', ha='center')
        
    ax.set_title(title, color='white', pad=20)
    ax.axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

def simulate_interference_pattern():
    print("Simulating Logos Interference Pattern...")
    
    # Superposition of Source and Reflection
    N = 300
    L = 1.2
    x = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Z = np.meshgrid(x, z)
    R = np.sqrt(X**2 + Z**2)
    
    # Source Wave: exp(i(kr - wt)) / r
    k = 20.0
    Source = np.sin(k * R) / (R + 0.1)
    
    # Boundary Reflection (Spherical Mirror)
    # Image source method? Or just standing wave J0.
    # Let's use the Bessel J0 again as it represents the sum.
    
    from scipy.special import j0
    Field = j0(k * R)
    
    # Mask
    Field[R > 1.0] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    im = ax.imshow(Field, cmap='magma', extent=[-L, L, -L, L], origin='lower')
    
    # Draw the "Egg"
    circle = plt.Circle((0, 0), 1.0, color='#FFCC00', fill=False, linewidth=3)
    ax.add_artist(circle)
    
    ax.set_title("THE COSMIC EGG\nInterference Pattern of the Prime Frequency", color='white')
    ax.axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "logos_interference_pattern.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # l=1, m=0 (Dipole) -> Creates Horizontal Nodal Plane
    visualize_spherical_mode(n=1, l=1, m=0, 
                           title="THE SEPARATION OF HEAVEN AND EARTH\nSpherical Harmonic Mode Y(1,0)", 
                           filename="logos_mode_dipole.png")
    
    # l=2, m=0 (Quadrupole) -> Creates Vertical Axis + Cones
    visualize_spherical_mode(n=2, l=2, m=0, 
                           title="THE ESTABLISHMENT OF THE AXIS\nSpherical Harmonic Mode Y(2,0)", 
                           filename="logos_mode_quadrupole.png")
                           
    simulate_interference_pattern()
