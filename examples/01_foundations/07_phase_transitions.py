r"""07 - Controlled TNFR phase-diagnostic and finite-size sweep.

This example samples a declared family of graph states.  It does not pretend
that the disorder coordinate is time, apply hand-written substitutes for
canonical operators, or infer a universal transition from one graph family.

The workflow is:

1. build reproducible graph states on one shared disorder grid;
2. read the canonical symmetry, chirality, susceptibility and coherence-length
   diagnostics;
3. locate sampled susceptibility maxima at each node count; and
4. fit descriptive slopes against node count ``N``.

Converting those slopes to thermodynamic exponent ratios requires a separate
linear-size model and replication across graph families.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from tnfr.physics.phase_scaling import (
    PhaseScalingDiagnostic,
    analyze_phase_finite_size_scaling,
)
from tnfr.physics.phase_transition import PhaseSnapshot, capture_phase_snapshot


@dataclass(frozen=True)
class SweepData:
    """Balanced arrays produced by the declared graph-state protocol."""

    sizes: tuple[int, ...]
    disorder: tuple[float, ...]
    order_parameter: np.ndarray
    susceptibility: np.ndarray
    coherence_length: np.ndarray
    snapshots: tuple[tuple[tuple[PhaseSnapshot, ...], ...], ...]


def build_phase_sweep(
    *,
    sizes: tuple[int, ...] = (16, 32, 64),
    disorder: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    replicates: int = 4,
    seed: int = 42,
) -> SweepData:
    """Construct a reproducible Watts-Strogatz state ensemble.

    ``disorder`` scales both phase spread and heterogeneous structural pressure.
    It is a chosen experimental coordinate, not a canonical TNFR constant.
    Node attributes initialize observations; no evolution or operator action is
    claimed.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")

    shape = (len(sizes), len(disorder), replicates)
    order = np.empty(shape, dtype=float)
    susceptibility = np.empty(shape, dtype=float)
    coherence_length = np.empty(shape, dtype=float)
    all_snapshots: list[list[list[PhaseSnapshot]]] = []

    for size_index, size in enumerate(sizes):
        size_snapshots: list[list[PhaseSnapshot]] = []
        for control_index, amplitude in enumerate(disorder):
            control_snapshots: list[PhaseSnapshot] = []
            for replicate in range(replicates):
                local_seed = (
                    seed + 10_000 * size_index + 100 * control_index + replicate
                )
                rng = np.random.default_rng(local_seed)
                graph = nx.watts_strogatz_graph(
                    size, 4, 0.25, seed=local_seed
                )
                inject_defaults(graph)
                for node in graph:
                    theta = float(rng.uniform(-np.pi * amplitude, np.pi * amplitude))
                    pressure = float(rng.normal(0.0, 0.5 * amplitude))
                    capacity = float(get_attr(graph.nodes[node], ALIAS_VF, 0.0))
                    graph.nodes[node][ALIAS_THETA[0]] = theta
                    graph.nodes[node][ALIAS_DNFR[0]] = pressure
                    graph.nodes[node][ALIAS_DEPI[0]] = capacity * pressure

                snapshot = capture_phase_snapshot(graph)
                order[size_index, control_index, replicate] = (
                    snapshot.order_parameter
                )
                susceptibility[size_index, control_index, replicate] = (
                    snapshot.susceptibility
                )
                coherence_length[size_index, control_index, replicate] = (
                    snapshot.coherence_length
                )
                control_snapshots.append(snapshot)
            size_snapshots.append(control_snapshots)
        all_snapshots.append(size_snapshots)

    return SweepData(
        sizes=sizes,
        disorder=disorder,
        order_parameter=order,
        susceptibility=susceptibility,
        coherence_length=coherence_length,
        snapshots=tuple(
            tuple(tuple(cell) for cell in size_cells)
            for size_cells in all_snapshots
        ),
    )


def analyze_sweep(data: SweepData) -> PhaseScalingDiagnostic:
    """Apply the balanced finite-size protocol to one sweep."""
    return analyze_phase_finite_size_scaling(
        data.sizes,
        data.disorder,
        data.order_parameter,
        data.susceptibility,
        data.coherence_length,
    )


def main() -> None:
    data = build_phase_sweep()
    result = analyze_sweep(data)

    print("TNFR finite-size phase diagnostic")
    print(f"status: {result.status}")
    print(f"replicates per cell: {result.replicate_count}")
    print("sampled pseudocritical disorder by node count:")
    for size, control, peak, sem in zip(
        result.system_sizes,
        result.pseudocritical_control,
        result.peak_susceptibility,
        result.peak_susceptibility_sem,
    ):
        sem_text = "unavailable" if sem is None else f"{sem:.6g}"
        control_text = "unavailable" if control is None else f"{control:.3f}"
        print(
            f"  N={size:3d}: g*={control_text}, "
            f"chi_peak={peak:.6g}, SEM={sem_text}"
        )

    fits = (
        ("peak susceptibility", result.susceptibility_size_fit),
        ("order magnitude at peak", result.order_size_fit),
        ("coherence length at peak", result.coherence_length_size_fit),
    )
    print("descriptive log-log slopes against N:")
    for name, fit in fits:
        if fit is None:
            print(f"  {name}: unavailable")
        else:
            print(f"  {name}: slope={fit.slope:.6g}, R^2={fit.r_squared:.6g}")

    print("scope:")
    for limitation in result.limitations:
        print(f"  - {limitation}")


if __name__ == "__main__":
    main()
