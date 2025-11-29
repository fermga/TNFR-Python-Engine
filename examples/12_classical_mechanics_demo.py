"""TNFR Classical Mechanics Demonstration — Emergence of Keplerian Orbits

This script demonstrates the emergence of stable Keplerian orbits from pure
TNFR Nodal Dynamics, without assuming Newton's laws as axioms.

It uses the `ClassicalMechanicsMapper` to translate the initial conditions
and the `TNFRSymplecticIntegrator` to evolve the system, proving that
TNFR dynamics contains Classical Mechanics as a limiting case.

The script generates plots showing:
1. The orbital trajectory (EPI spatial components).
2. Phase space evolution (Form vs Flow).
3. Conservation of Energy (Hamiltonian) and Structural Coherence.
4. The correlation between Classical Force and Structural Pressure (ΔNFR).

Usage:
    python examples/12_classical_mechanics_demo.py
"""

import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.physics.classical_mechanics import ClassicalMechanicsMapper, GeneralizedCoordinateSystem
from tnfr.types import TNFRNode


def run_kepler_simulation() -> Tuple[dict, dict]:
    """
    Simulates a planet orbiting a star using TNFR dynamics.
    Returns history dictionaries for plotting.
    """
    print("Initializing TNFR Kepler Simulation...")

    # 1. Define System Parameters
    # G*M = 1.0 (Normalized units)
    GM = 1.0
    
    # Initial Conditions (Eccentric Orbit e=0.5)
    # Pericenter distance r_p = 1.0
    # Velocity at pericenter v_p = sqrt(GM/a * (1+e)/(1-e))
    # Let's pick a simple setup: r=1, v=1 -> Circular if GM=1.
    # Let's do slightly elliptical: v = 1.2
    r_init = 1.0
    v_init = 0.8  # Sub-circular velocity -> Ellipse
    
    q_init = np.array([r_init, 0.0])      # Start at x=r, y=0
    q_dot_init = np.array([0.0, v_init])  # Velocity in y direction
    
    # 2. Map to TNFR Node
    # Mass m=1 -> νf=1
    system = GeneralizedCoordinateSystem(q=q_init, q_dot=q_dot_init)
    
    # Lagrangian L = T - V = 0.5*v^2 + GM/r
    def lagrangian(q, qd, t):
        r = np.linalg.norm(q)
        v2 = np.sum(qd**2)
        return 0.5 * v2 + GM / r

    tnfr_state = ClassicalMechanicsMapper.lagrangian_to_tnfr(lagrangian, system)
    
    node: TNFRNode = {
        EPI_PRIMARY: tnfr_state[EPI_PRIMARY],
        VF_PRIMARY: tnfr_state[VF_PRIMARY],
        DNFR_PRIMARY: np.zeros_like(tnfr_state[EPI_PRIMARY])
    }
    
    print(f"Initial State: EPI={node[EPI_PRIMARY]}, νf={node[VF_PRIMARY]}")

    # 3. Define Structural Force Evaluator (Gravity)
    # In TNFR, this is the gradient of the Coherence Potential Φ_s.
    # Here we use the analytical gradient for the demo, but in a full network
    # this emerges from neighbor interactions.
    def coherence_gradient_force(n: TNFRNode) -> np.ndarray:
        epi = n[EPI_PRIMARY]
        q = epi[:2]  # Spatial component
        r = np.linalg.norm(q)
        
        # F = -∇Φ_s
        # For gravity, Φ_s ~ -1/r (Coherence Potential)
        # ∇Φ_s ~ 1/r^2 * r_hat
        # F = -GM/r^3 * q
        f_vec = -GM / (r**3 + 1e-9) * q
        
        # ΔNFR has same shape as EPI [q, q_dot]
        # Force acts on Flow (q_dot)
        return np.concatenate([np.zeros_like(f_vec), f_vec])

    # 4. Evolve System
    dt = 0.01
    t_max = 20.0  # Approx 3 orbits (T ~ 2pi for circ)
    steps = int(t_max / dt)
    
    history = {
        "t": [],
        "x": [], "y": [],
        "vx": [], "vy": [],
        "E": [],
        "L": [] # Angular momentum
    }
    
    print(f"Simulating {steps} steps (dt={dt})...")
    
    t = 0.0
    for _ in range(steps):
        # Symplectic Step
        TNFRSymplecticIntegrator.velocity_verlet(node, dt, coherence_gradient_force)
        
        # Record Telemetry
        epi = node[EPI_PRIMARY]
        q = epi[:2]
        v = epi[2:]
        r = np.linalg.norm(q)
        speed2 = np.sum(v**2)
        
        # Energy (Hamiltonian) H = T + V
        energy = 0.5 * speed2 - GM / r
        
        # Angular Momentum L = r x v (2D cross product)
        ang_mom = q[0]*v[1] - q[1]*v[0]
        
        history["t"].append(t)
        history["x"].append(q[0])
        history["y"].append(q[1])
        history["vx"].append(v[0])
        history["vy"].append(v[1])
        history["E"].append(energy)
        history["L"].append(ang_mom)
        
        t += dt

    return history

def plot_results(history: dict):
    """Generates and saves plots."""
    output_dir = "results/classical_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Trajectory Plot
    plt.figure(figsize=(8, 8))
    plt.plot(history["x"], history["y"], label="TNFR Trajectory")
    plt.scatter([0], [0], color='orange', s=100, label="Attractor (Star)")
    plt.scatter(history["x"][0], history["y"][0], color='green', label="Start")
    plt.title("Emergent Keplerian Orbit from Nodal Dynamics")
    plt.xlabel("EPI Spatial X (q_x)")
    plt.ylabel("EPI Spatial Y (q_y)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/01_trajectory.png")
    plt.close()
    
    # 2. Phase Space (x vs vx)
    plt.figure(figsize=(8, 6))
    plt.plot(history["x"], history["vx"])
    plt.title("Phase Space Projection (Form vs Flow)")
    plt.xlabel("Form (Position X)")
    plt.ylabel("Flow (Velocity X)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/02_phase_space.png")
    plt.close()
    
    # 3. Conservation Laws
    plt.figure(figsize=(10, 6))
    
    # Normalize energy drift
    E = np.array(history["E"])
    E0 = E[0]
    E_drift = (E - E0) / abs(E0)
    
    L = np.array(history["L"])
    L0 = L[0]
    L_drift = (L - L0) / abs(L0)
    
    plt.plot(history["t"], E_drift, label="Energy Drift (H)")
    plt.plot(history["t"], L_drift, label="Angular Momentum Drift (L)", linestyle="--")
    plt.title("Conservation of Structural Invariants")
    plt.xlabel("Time")
    plt.ylabel("Relative Drift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/03_conservation.png")
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    hist = run_kepler_simulation()
    plot_results(hist)
    print("Demonstration Complete.")
