"""Finite comparison of two explicitly different N-body models.

The Newtonian reference assumes inverse-square gravity.  The TNFR-variable
adapter assumes the phase-coupled pair law documented in
``tnfr.dynamics.nbody_tnfr``.  Sharing initial conditions does not make either
law a limit of the other. This example measures their finite trajectory
difference and makes no convergence claim.

The adapter mapping ``nu_f = 1 / mass`` is a declared convention.  Its
position-dependent pair potential is kept separate from the auxiliary
``InternalHamiltonian`` ground-state read-out.  The historical diagonal
projector commutator is also reported; it is algebraically zero and supplies no
phase evolution, so phases remain fixed in this model.

Usage
-----
python examples/02_physics_regimes/11_classical_limit_comparison.py
    --t-final 2 --dt 0.002
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from tnfr.dynamics.nbody import NBodySystem
from tnfr.dynamics.nbody_tnfr import (
    TNFRNBodySystem,
    compute_nbody_pair_forces,
    compute_tnfr_coherence_potential,
    compute_tnfr_delta_nfr,
)


@dataclass(frozen=True)
class SimulationResult:
    """Finite diagnostics from one declared N-body model."""

    label: str
    history: dict[str, Any]
    energy_drift_pct: float
    momentum_drift: float
    angular_momentum_drift: float
    final_separation: float


def _prepare_initial_conditions(
    distance: float = 1.0,
    gravitational_constant: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one shared two-body state based on a Newtonian orbit estimate."""
    masses = np.array([1.0, 0.2], dtype=float)
    total_mass = float(np.sum(masses))

    positions = np.array(
        [
            [-masses[1] / total_mass * distance, 0.0, 0.0],
            [masses[0] / total_mass * distance, 0.0, 0.0],
        ],
        dtype=float,
    )
    orbital_speed = np.sqrt(gravitational_constant * total_mass / distance)
    velocities = np.array(
        [
            [0.0, -orbital_speed * masses[1] / total_mass, 0.0],
            [0.0, orbital_speed * masses[0] / total_mass, 0.0],
        ],
        dtype=float,
    )
    return masses, positions, velocities


def _relative_energy_drift(history: dict[str, Any]) -> float:
    energy = np.asarray(history["energy"], dtype=float)
    scale = max(abs(float(energy[0])), float(np.finfo(float).eps))
    return float(abs(energy[-1] - energy[0]) / scale * 100.0)


def _vector_drift(history: dict[str, Any], key: str) -> float:
    values = np.asarray(history[key], dtype=float)
    return float(np.linalg.norm(values[-1] - values[0]))


def _final_separation(history: dict[str, Any]) -> float:
    positions = np.asarray(history["positions"], dtype=float)
    return float(np.linalg.norm(positions[-1, 1] - positions[-1, 0]))


def _result(label: str, history: dict[str, Any]) -> SimulationResult:
    return SimulationResult(
        label=label,
        history=history,
        energy_drift_pct=_relative_energy_drift(history),
        momentum_drift=_vector_drift(history, "momentum"),
        angular_momentum_drift=_vector_drift(history, "angular_momentum"),
        final_separation=_final_separation(history),
    )


def run_classical_system(t_final: float, dt: float) -> SimulationResult:
    """Run the Newtonian reference with its declared inverse-square law."""
    masses, positions, velocities = _prepare_initial_conditions()
    system = NBodySystem(len(masses), masses=masses, G=1.0)
    system.set_state(positions, velocities)
    history = system.evolve(t_final=t_final, dt=dt, store_interval=10)
    return _result("Newtonian inverse-square reference", history)


def run_adapter_system(
    t_final: float,
    dt: float,
) -> tuple[SimulationResult, TNFRNBodySystem, dict[str, float]]:
    """Run the declared phase-coupled adapter and capture scope diagnostics."""
    masses, positions, velocities = _prepare_initial_conditions()
    system = TNFRNBodySystem(
        len(masses),
        masses=masses,
        positions=positions,
        velocities=velocities,
        phases=np.zeros(len(masses), dtype=float),
        coupling_strength=0.6,
        coherence_strength=-2.0,
    )

    initial_positions = system.positions.copy()
    initial_ground_state = compute_tnfr_coherence_potential(
        system.graph, initial_positions, system.hbar_str
    )
    translated_positions = initial_positions + np.array([4.0, -2.0, 1.0])
    translated_ground_state = compute_tnfr_coherence_potential(
        system.graph, translated_positions, system.hbar_str
    )
    node_ids = [f"body_{index}" for index in range(system.n_bodies)]
    projector_rate_norm = float(
        np.linalg.norm(
            compute_tnfr_delta_nfr(system.graph, node_ids, system.hbar_str)
        )
    )
    initial_force_norm = float(
        np.linalg.norm(compute_nbody_pair_forces(system.graph, initial_positions))
    )

    history = system.evolve(t_final=t_final, dt=dt, store_interval=10)
    diagnostics = {
        "ground_state_translation_delta": abs(
            translated_ground_state - initial_ground_state
        ),
        "projector_rate_norm": projector_rate_norm,
        "initial_force_norm": initial_force_norm,
        "phase_drift": float(
            np.max(np.abs(history["phases"][-1] - history["phases"][0]))
        ),
    }
    return _result("Declared phase-coupled adapter", history), system, diagnostics


def _trajectory_rmse(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Return position RMSE over their common stored finite horizon."""
    reference_positions = np.asarray(reference["positions"], dtype=float)
    candidate_positions = np.asarray(candidate["positions"], dtype=float)
    count = min(len(reference_positions), len(candidate_positions))
    difference = candidate_positions[:count] - reference_positions[:count]
    return float(np.sqrt(np.mean(difference**2)))


def format_result(result: SimulationResult) -> str:
    """Format one model's directly measured finite diagnostics."""
    return (
        f"{result.label}\n"
        f"  relative energy drift : {result.energy_drift_pct:.6e}%\n"
        f"  momentum drift norm   : {result.momentum_drift:.6e}\n"
        f"  angular-momentum drift: {result.angular_momentum_drift:.6e}\n"
        f"  final separation      : {result.final_separation:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-final", type=float, default=2.0, help="Finite comparison horizon"
    )
    parser.add_argument("--dt", type=float, default=0.002, help="Integration step")
    args = parser.parse_args()

    classical = run_classical_system(args.t_final, args.dt)
    adapter, system, diagnostics = run_adapter_system(args.t_final, args.dt)

    print("Finite N-body model comparison\n")
    print(format_result(classical))
    print()
    print(format_result(adapter))
    print()
    print(
        "Position trajectory RMSE : "
        f"{_trajectory_rmse(classical.history, adapter.history):.6e}"
    )
    print(f"Adapter initial force norm: {diagnostics['initial_force_norm']:.6e}")
    print(
        "Hamiltonian ground-state translation delta: "
        f"{diagnostics['ground_state_translation_delta']:.6e}"
    )
    print(
        "Localized-projector commutator norm: "
        f"{diagnostics['projector_rate_norm']:.6e}"
    )
    print(f"Adapter phase drift       : {diagnostics['phase_drift']:.6e}")
    print(f"Adapter scope             : {system.graph.graph['MODEL_SCOPE']}")
    print(
        "\nInterpretation: these are finite diagnostics for two declared laws. "
        "A classical-limit claim would require a parameterized convergence "
        "theorem or a controlled scaling study; this run supplies neither."
    )


if __name__ == "__main__":
    main()
