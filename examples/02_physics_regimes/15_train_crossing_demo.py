"""Compare a prescribed constant-velocity adapter with an analytic crossing.

The classical adapter stores ``[q, q_dot]`` and a separately declared
second-order step advances position at zero external force. Canonical TNFR
zero pressure would instead freeze the EPI chart. This example computes no
canonical pressure, tetrad field or C(t).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator
from tnfr.physics.classical_mechanics import (
    ClassicalMechanicsMapper,
    GeneralizedCoordinateSystem,
)
from tnfr.types import TNFRNode


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zero_adapter_force(node: TNFRNode) -> np.ndarray:
    """Return the declared zero force in the mechanical adapter slot."""

    return np.zeros_like(node[EPI_PRIMARY])


def run_train_crossing_demo() -> None:
    print("--- Constant-Velocity Adapter: The Two Trains Problem ---")

    # Problem Parameters
    # Train A: Madrid -> Barcelona
    # Train B: Barcelona -> Madrid
    DISTANCE_KM = 600.0
    SPEED_A_KMH = 300.0  # AVE
    SPEED_B_KMH = 250.0  # Alvia

    # Convert to SI units (meters, seconds) for the physics engine
    # Though the engine is unit-agnostic, consistency helps
    dist_m = DISTANCE_KM * 1000.0
    v_a_ms = SPEED_A_KMH / 3.6
    v_b_ms = -SPEED_B_KMH / 3.6  # Moving left

    print(f"Distance: {dist_m/1000:.1f} km")
    print(f"Train A Speed: {v_a_ms:.2f} m/s")
    print(f"Train B Speed: {v_b_ms:.2f} m/s")

    # Analytical Solution
    # t = d / (v1 + v2)
    # x = v1 * t
    t_analytical = dist_m / (v_a_ms + abs(v_b_ms))
    x_analytical = v_a_ms * t_analytical

    print("\nAnalytical prediction:")
    print(f"Time to cross: {t_analytical:.2f} s ({t_analytical/60:.2f} min)")
    print(f"Crossing point: {x_analytical/1000:.2f} km from Madrid")

    # Pack both prescribed classical states. nu_f=1 follows only from the
    # adapter's unit-mass convention.
    def make_node(position: float, velocity: float) -> TNFRNode:
        state = GeneralizedCoordinateSystem(
            q=np.array([position, 0.0]),
            q_dot=np.array([velocity, 0.0]),
            masses=np.ones(2),
        )
        payload = ClassicalMechanicsMapper.lagrangian_to_tnfr(
            lambda _q, q_dot, _t: float(0.5 * np.sum(q_dot**2)), state
        )
        return {
            EPI_PRIMARY: payload[EPI_PRIMARY],
            VF_PRIMARY: payload[VF_PRIMARY],
            DNFR_PRIMARY: payload[DNFR_PRIMARY],
        }

    node_a = make_node(0.0, v_a_ms)
    node_b = make_node(dist_m, v_b_ms)

    dt = 1.0  # 1 second steps
    time = 0.0

    history_a = []
    history_b = []
    times = []

    print(f"\nRunning the external kinematic adapter (dt={dt}s)...")

    crossing_detected = False
    crossing_time = 0.0
    crossing_pos = 0.0

    # Run loop
    # We run a bit past the expected time to show the crossing
    max_steps = int(t_analytical * 1.2)

    for step in range(max_steps):
        # Store history
        q_a = node_a[EPI_PRIMARY][:2]
        q_b = node_b[EPI_PRIMARY][:2]
        history_a.append(q_a[0])
        history_b.append(q_b[0])
        times.append(time)

        # Check crossing
        # If A was behind B and now is ahead (or equal)
        if not crossing_detected and q_a[0] >= q_b[0]:
            crossing_detected = True
            # Linear interpolation for precise time
            # xA_prev, xB_prev at t-dt
            # xA_curr, xB_curr at t
            # Find tau where xA(tau) = xB(tau)

            xA_prev = history_a[-2]
            xB_prev = history_b[-2]
            xA_curr = q_a[0]
            xB_curr = q_b[0]

            # Relative distance D(t) = xB - xA
            # D_prev = xB_prev - xA_prev (> 0)
            # D_curr = xB_curr - xA_curr (<= 0)
            # Fraction f = D_prev / (D_prev - D_curr)

            D_prev = xB_prev - xA_prev
            D_curr = xB_curr - xA_curr
            fraction = D_prev / (D_prev - D_curr)

            crossing_time = time - dt + fraction * dt
            crossing_pos = xA_prev + v_a_ms * (fraction * dt)

            print(f"-> Crossing Detected at Step {step}!")

        # Advance the declared second-order adapter with zero external force.
        TNFRSymplecticIntegrator.velocity_verlet(
            node_a, dt, zero_adapter_force
        )
        TNFRSymplecticIntegrator.velocity_verlet(
            node_b, dt, zero_adapter_force
        )

        time += dt

    # --- Results ---
    print("\nAdapter results:")
    print(f"Time to cross: {crossing_time:.2f} s")
    print(f"Crossing point: {crossing_pos/1000:.2f} km")

    error_t = abs(crossing_time - t_analytical)
    error_x = abs(crossing_pos - x_analytical)

    print(f"\nAccuracy:")
    print(f"Time Error: {error_t:.6f} s")
    print(f"Position Error: {error_x:.6f} m")

    if error_t < 1e-3:
        print("PASS: the adapter matches the analytic constant-velocity result.")
    else:
        print("WARNING: Discrepancy detected.")

    # --- Visualization ---
    results_dir = Path("results/kinematics_demo")
    ensure_dir(results_dir)

    plt.figure(figsize=(10, 6))

    # Convert to km and min for plotting
    times_min = np.array(times) / 60.0
    pos_a_km = np.array(history_a) / 1000.0
    pos_b_km = np.array(history_b) / 1000.0

    plt.plot(
        times_min, pos_a_km, label="Train A (Madrid -> BCN)", color="blue", linewidth=2
    )
    plt.plot(
        times_min, pos_b_km, label="Train B (BCN -> Madrid)", color="red", linewidth=2
    )

    # Mark crossing
    cross_t_min = crossing_time / 60.0
    cross_x_km = crossing_pos / 1000.0

    plt.plot(cross_t_min, cross_x_km, "ko", markersize=10, label="Crossing Point")
    plt.annotate(
        f"  t={cross_t_min:.1f} min\n  x={cross_x_km:.1f} km",
        (cross_t_min, cross_x_km),
        xytext=(10, -20),
        textcoords="offset points",
    )

    plt.title("Constant-Velocity Adapter: Two Trains Problem")
    plt.xlabel("Adapter time (minutes)")
    plt.ylabel("Position (km)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(results_dir / "01_train_crossing.png")
    print(f"Saved plot to {results_dir / '01_train_crossing.png'}")


if __name__ == "__main__":
    run_train_crossing_demo()
