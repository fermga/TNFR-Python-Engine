"""Demonstrate an explicitly supplied Newtonian central-force adapter.

The script supplies the Newtonian potential ``U=-GM/r`` and its force as model
inputs. ``ClassicalMechanicsMapper`` packages the initial classical state in a
TNFR-shaped payload, and ``TNFRSymplecticIntegrator`` applies velocity Verlet.
The resulting orbit and classical invariant drift validate that declared
adapter at the selected step size.

No graph pressure, tetrad field or canonical C(t) is computed here. The
trajectory therefore does not derive gravity from the TNFR nodal equation.

Usage:
    python examples/02_physics_regimes/12_classical_mechanics_demo.py
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.physics.classical_mechanics import (
    ClassicalMechanicsMapper,
    GeneralizedCoordinateSystem,
)
from tnfr.types import TNFRNode


def run_kepler_simulation() -> dict[str, list[float]]:
    """Simulate one declared inverse-square central-force trajectory."""

    print("Initializing the Newtonian central-force adapter...")

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

    q_init = np.array([r_init, 0.0])  # Start at x=r, y=0
    q_dot_init = np.array([0.0, v_init])  # Velocity in y direction

    # 2. Pack state in the classical adapter representation. The assignment
    # nu_f=1/m is local to this adapter.
    system = GeneralizedCoordinateSystem(q=q_init, q_dot=q_dot_init)

    # Lagrangian L = T - V = 0.5*v^2 + GM/r
    def lagrangian(q, qd, _t):
        r = np.linalg.norm(q)
        v2 = np.sum(qd**2)
        return 0.5 * v2 + GM / r

    tnfr_state = ClassicalMechanicsMapper.lagrangian_to_tnfr(lagrangian, system)

    node: TNFRNode = {
        EPI_PRIMARY: tnfr_state[EPI_PRIMARY],
        VF_PRIMARY: tnfr_state[VF_PRIMARY],
        DNFR_PRIMARY: np.zeros_like(tnfr_state[EPI_PRIMARY]),
    }

    print(f"Initial State: EPI={node[EPI_PRIMARY]}, νf={node[VF_PRIMARY]}")

    # 3. Supply the external Newtonian force. The integrator stores it in its
    # full adapter force slot; this is not canonical scalar graph pressure.
    def newtonian_central_force(n: TNFRNode) -> np.ndarray:
        epi = n[EPI_PRIMARY]
        q = epi[:2]  # Spatial component
        r = np.linalg.norm(q)
        if r == 0.0:
            raise ValueError("the unsoftened Newtonian force is singular at r=0")

        # F = -GM/r^3 * q
        f_vec = -GM / r**3 * q

        # The adapter force acts on the velocity half of [q, q_dot].
        return np.concatenate([np.zeros_like(f_vec), f_vec])

    # Velocity Verlet expects the initial force already materialized.
    node[DNFR_PRIMARY] = newtonian_central_force(node)

    # 4. Evolve System
    dt = 0.01
    t_max = 20.0  # Approx 3 orbits (T ~ 2pi for circ)
    steps = int(t_max / dt)

    history = {
        "t": [],
        "x": [],
        "y": [],
        "vx": [],
        "vy": [],
        "E": [],
        "L": [],  # Angular momentum
    }

    print(f"Simulating {steps} steps (dt={dt})...")

    t = 0.0
    for _ in range(steps):
        TNFRSymplecticIntegrator.velocity_verlet(
            node, dt, newtonian_central_force
        )
        t += dt

        # Record Telemetry
        epi = node[EPI_PRIMARY]
        q = epi[:2]
        v = epi[2:]
        r = np.linalg.norm(q)
        speed2 = np.sum(v**2)

        # Energy (Hamiltonian) H = T + V
        energy = 0.5 * speed2 - GM / r

        # Angular Momentum L = r x v (2D cross product)
        ang_mom = q[0] * v[1] - q[1] * v[0]

        history["t"].append(t)
        history["x"].append(q[0])
        history["y"].append(q[1])
        history["vx"].append(v[0])
        history["vy"].append(v[1])
        history["E"].append(energy)
        history["L"].append(ang_mom)

    return history


def plot_results(history: dict[str, list[float]]) -> None:
    """Generate and save adapter trajectory plots."""
    output_dir = "results/classical_demo"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Trajectory Plot
    plt.figure(figsize=(8, 8))
    plt.plot(history["x"], history["y"], label="Adapter trajectory")
    plt.scatter([0], [0], color="orange", s=100, label="Fixed force center")
    plt.scatter(history["x"][0], history["y"][0], color="green", label="Start")
    plt.title("Declared Newtonian Central-Force Trajectory")
    plt.xlabel("Adapter position q_x")
    plt.ylabel("Adapter position q_y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/01_trajectory.png")
    plt.close()

    # 2. Phase Space (x vs vx)
    plt.figure(figsize=(8, 6))
    plt.plot(history["x"], history["vx"])
    plt.title("Classical Phase-Space Projection")
    plt.xlabel("Position q_x")
    plt.ylabel("Velocity dq_x/dt")
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
    plt.title("Classical Invariant Drift")
    plt.xlabel("Adapter time")
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
