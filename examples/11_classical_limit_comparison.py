"""TNFR vs Classical N-body comparison in the low-dissonance regime.

This experiment mirrors the documentation in `docs/TNFR_CLASSICAL_MAPPING.md` by
running the classical N-body solver (which assumes Newtonian gravity) and the
pure TNFR N-body solver (which derives forces from coherence alone) on identical
initial conditions. When phases remain synchronized and coherence length is
large, the TNFR dynamics collapses onto the classical prediction.

The script now also reports per-body physical quantities by extracting them from
TNFR telemetry: the structural frequency νf gives inertial mass (m = 1/νf), the
EPI velocity component maps to classical velocity, ΔNFR tracks force density,
and the combination of Φ_s with ξ_C produces an effective coherence size that we
use to estimate density. This makes the “velocity/density/mass/size” mapping
explicit for each body.

Usage
-----
python examples/11_classical_limit_comparison.py --t-final 15.0 --dt 0.01
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.constants.canonical import PHI
from tnfr.dynamics.nbody import NBodySystem, compute_gravitational_dnfr
from tnfr.dynamics.nbody_tnfr import TNFRNBodySystem, compute_tnfr_delta_nfr
from tnfr.physics.fields import compute_structural_telemetry
from tnfr.types import TNFRGraph


@dataclass
class BodyAttributes:
    node_id: str
    mass: float
    speed: float
    velocity_vector: np.ndarray
    density: float
    size: float
    phi_s: float
    dnfr: float


@dataclass
class SimulationResult:
    label: str
    history: Dict[str, Any]
    graph: TNFRGraph
    energy_drift_pct: float
    potential_energy: float
    telemetry_stats: Dict[str, float]
    telemetry_fields: Dict[str, Any]


def _prepare_initial_conditions(distance: float = 1.0, g_const: float = 1.0):
    """Return positions and velocities for a near-circular two-body orbit."""
    masses = np.array([1.0, 0.2], dtype=float)
    total_mass = float(np.sum(masses))

    body0_offset = -masses[1] / total_mass * distance
    body1_offset = masses[0] / total_mass * distance
    positions = np.array(
        [
            [body0_offset, 0.0, 0.0],
            [body1_offset, 0.0, 0.0],
        ],
        dtype=float,
    )

    orbital_speed = np.sqrt(g_const * total_mass / distance)
    velocities = np.array(
        [
            [0.0, -orbital_speed * masses[1] / total_mass, 0.0],
            [0.0, orbital_speed * masses[0] / total_mass, 0.0],
        ],
        dtype=float,
    )

    return masses, positions, velocities


def _compute_energy_drift(history: Dict[str, Any]) -> float:
    energy = np.asarray(history.get("energy", []), dtype=float)
    if energy.size < 2 or np.isclose(energy[0], 0.0):
        return 0.0
    return float(abs(energy[-1] - energy[0]) / max(abs(energy[0]), 1e-12) * 100.0)


def _annotate_delta_nfr(graph: TNFRGraph, node_ids: List[str], values: np.ndarray) -> None:
    """Store scalar ΔNFR telemetry on each node for later field analysis."""
    flat_values = np.asarray(values, dtype=float).reshape(len(node_ids), -1)
    magnitudes = np.linalg.norm(flat_values, axis=1)
    for node_id, magnitude in zip(node_ids, magnitudes):
        graph.nodes[node_id][DNFR_PRIMARY] = float(magnitude)


def _capture_telemetry(graph: TNFRGraph) -> tuple[Dict[str, Any], Dict[str, float]]:
    telemetry = compute_structural_telemetry(graph)
    phi_vals = np.array(list(telemetry.get("phi_s", {}).values()) or [0.0], dtype=float)
    grad_vals = np.array(list(telemetry.get("grad_phi", {}).values()) or [0.0], dtype=float)
    curv_vals = np.array(list(telemetry.get("curv_phi", {}).values()) or [0.0], dtype=float)

    summary = {
        "phi_s_mean": float(np.mean(phi_vals)),
        "phi_s_std": float(np.std(phi_vals)),
        "grad_phi_mean": float(np.mean(np.abs(grad_vals))),
        "curv_phi_mean": float(np.mean(np.abs(curv_vals))),
        "xi_c": float(telemetry.get("xi_c", float("nan"))),
    }
    return telemetry, summary


def run_classical_system(t_final: float, dt: float) -> SimulationResult:
    masses, positions, velocities = _prepare_initial_conditions()
    classical = NBodySystem(len(masses), masses=masses, G=1.0)
    classical.set_state(positions, velocities)

    history = classical.evolve(t_final=t_final, dt=dt, store_interval=10)

    accelerations = compute_gravitational_dnfr(
        classical.positions,
        classical.masses,
        classical.G,
        classical.softening,
    )
    node_ids = [f"body_{i}" for i in range(len(masses))]
    _annotate_delta_nfr(classical.graph, node_ids, accelerations)
    telemetry_fields, telemetry_stats = _capture_telemetry(classical.graph)

    return SimulationResult(
        label="Classical (assumed gravity)",
        history=history,
        graph=classical.graph,
        energy_drift_pct=_compute_energy_drift(history),
        potential_energy=float(history["potential"][-1]),
        telemetry_stats=telemetry_stats,
        telemetry_fields=telemetry_fields,
    )


def run_tnfr_system(t_final: float, dt: float) -> SimulationResult:
    masses, positions, velocities = _prepare_initial_conditions()
    phases = np.zeros(len(masses), dtype=float)

    tnfr = TNFRNBodySystem(
        len(masses),
        masses=masses,
        positions=positions,
        velocities=velocities,
        phases=phases,
        coupling_strength=0.6,
        coherence_strength=-2.0,
    )

    history = tnfr.evolve(t_final=t_final, dt=dt, store_interval=10)

    node_ids = [f"body_{i}" for i in range(len(masses))]
    dnfr_scalars = compute_tnfr_delta_nfr(tnfr.graph, node_ids, tnfr.hbar_str)
    _annotate_delta_nfr(tnfr.graph, node_ids, dnfr_scalars)
    telemetry_fields, telemetry_stats = _capture_telemetry(tnfr.graph)

    return SimulationResult(
        label="Pure TNFR (coherence-driven)",
        history=history,
        graph=tnfr.graph,
        energy_drift_pct=_compute_energy_drift(history),
        potential_energy=float(history["potential"][-1]),
        telemetry_stats=telemetry_stats,
        telemetry_fields=telemetry_fields,
    )


def _sanitize_coherence_length(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    return float(value)


def _estimate_body_size(phi_value: float, xi_c: float) -> float:
    """Translate Φ_s confinement into a coherence radius."""
    base_radius = _sanitize_coherence_length(xi_c)
    confinement = 1.0 / (1.0 + abs(phi_value) / max(PHI, 1e-9))
    return float(max(1e-6, base_radius * confinement))


def _estimate_density(mass: float, radius: float) -> float:
    volume = (4.0 / 3.0) * math.pi * max(radius, 1e-6) ** 3
    return float(mass / volume)


def _derive_body_attributes(graph: TNFRGraph, telemetry: Dict[str, Any]) -> List[BodyAttributes]:
    phi_map = telemetry.get("phi_s", {})
    xi_c = _sanitize_coherence_length(float(telemetry.get("xi_c", 1.0)))
    attributes: List[BodyAttributes] = []

    for node_id in sorted(graph.nodes()):
        node_data = graph.nodes[node_id]
        vf = float(node_data.get(VF_PRIMARY, 0.0))
        mass = float(math.inf if np.isclose(vf, 0.0) else 1.0 / vf)
        epi_state = node_data.get(EPI_PRIMARY, {})
        if isinstance(epi_state, dict):
            velocity_source = epi_state.get("velocity", np.zeros(3))
        else:
            velocity_source = np.zeros(3)
        velocity_vector = np.array(velocity_source, dtype=float)
        speed = float(np.linalg.norm(velocity_vector))
        phi_s_value = float(phi_map.get(node_id, 0.0))
        size = _estimate_body_size(phi_s_value, xi_c)
        density = _estimate_density(mass, size)
        dnfr_value = float(node_data.get(DNFR_PRIMARY, 0.0))

        attributes.append(
            BodyAttributes(
                node_id=node_id,
                mass=mass,
                speed=speed,
                velocity_vector=velocity_vector,
                density=density,
                size=size,
                phi_s=phi_s_value,
                dnfr=dnfr_value,
            )
        )

    return attributes


def format_result(result: SimulationResult) -> str:
    stats = result.telemetry_stats
    return (
        f"{result.label}\n"
        f"  Energy drift      : {result.energy_drift_pct:.3e}%\n"
        f"  Potential energy  : {result.potential_energy:.6f}\n"
        f"  ⟨Φ_s⟩, σ(Φ_s)     : {stats['phi_s_mean']:.6f}, {stats['phi_s_std']:.6f}\n"
        f"  ⟨|∇φ|⟩            : {stats['grad_phi_mean']:.6f}\n"
        f"  ⟨|K_φ|⟩           : {stats['curv_phi_mean']:.6f}\n"
        f"  ξ_C               : {stats['xi_c']:.6f}\n"
    )


def _format_body_block(title: str, attributes: List[BodyAttributes]) -> str:
    lines = [f"{title} body mappings:"]
    for attr in attributes:
        lines.append(
            "    "
            f"{attr.node_id:<7} mass={attr.mass:7.4f}  |v|={attr.speed:7.4f}  ρ={attr.density:9.4f}"
            f"  r={attr.size:7.4f}  Φ_s={attr.phi_s:7.4f}  ΔNFR={attr.dnfr:7.4f}"
        )
        vx, vy, vz = attr.velocity_vector
        lines.append(f"            v=({vx:7.4f}, {vy:7.4f}, {vz:7.4f})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-final", type=float, default=15.0, help="Total structural time to simulate")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration time step")
    args = parser.parse_args()

    classical_result = run_classical_system(args.t_final, args.dt)
    tnfr_result = run_tnfr_system(args.t_final, args.dt)

    print("TNFR ↔ Classical Mechanics Comparison (low-dissonance limit)\n")
    print(format_result(classical_result))
    print(format_result(tnfr_result))

    classical_attrs = _derive_body_attributes(classical_result.graph, classical_result.telemetry_fields)
    tnfr_attrs = _derive_body_attributes(tnfr_result.graph, tnfr_result.telemetry_fields)

    print(_format_body_block("Classical", classical_attrs))
    print()
    print(_format_body_block("Pure TNFR", tnfr_attrs))
    print()
    print(
        "Telemetry alignment: compare potential energies against ⟨Φ_s⟩ statistics "
        "and verify that low phase gradients correspond to long coherence length (ξ_C)."
    )


if __name__ == "__main__":
    main()
