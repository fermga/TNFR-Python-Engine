"""Auxiliary phase-threshold clique controls, not canonical TNFR evolution.

The preparation explicitly appends phase samples, rebuilds a threshold graph,
and applies a selected phasor-averaging map. These operations are comparison
constructions: they execute no Emission, Coupling or Resonance operator and
carry no nodal EPI/capacity evolution or grammar/causal certificate.

A clique with m vertices has simplex dimension m-1. Its size is not its
simplex dimension, and neither is a selected physical or spectral dimension.
The threshold pi/6 and averaging factor 0.4 are declared auxiliary settings.
Run: python benchmarks/emergent_dimension_dynamics.py
"""

from __future__ import annotations

import networkx as nx
import numpy as np

PHASE_THRESHOLD = np.pi / 6
AVERAGING_FACTOR = 0.4
AVERAGING_STEPS = 60


def _circ(a: float, b: float) -> float:
    """Circular separation for this declared threshold construction."""
    return float(abs(np.angle(np.exp(1j * (a - b)))))


def _resonant_graph(phases: list[float]) -> nx.Graph:
    """Construct a graph using the selected threshold, without operator calls."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(phases)))
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            if _circ(phases[i], phases[j]) <= PHASE_THRESHOLD:
                graph.add_edge(i, j)
    return graph


def _max_clique_size(graph: nx.Graph) -> int:
    """Return the largest clique's vertex count; an empty graph has size zero."""
    return max((len(clique) for clique in nx.find_cliques(graph)), default=0)


def _simplex_readout(phases: list[float]) -> dict:
    size = _max_clique_size(_resonant_graph(phases))
    return {
        "phases": tuple(float(value) for value in phases),
        "node_count": len(phases),
        "max_clique_size": size,
        "simplex_dimension": size - 1 if size else None,
    }


def run_dimension_controls() -> dict:
    """Retain the existing three preparations with explicit auxiliary scope."""
    phases = [0.0]
    accretion = [_simplex_readout(phases)]
    for k in range(1, 5):
        phases.append(0.01 * ((-1) ** k))
        accretion.append(_simplex_readout(phases))

    tetra = [0.0, 0.01, -0.01, 0.005]
    incompatible = {
        "before": _simplex_readout(tetra),
        "after": _simplex_readout(tetra + [np.pi]),
    }

    spread = [0.0, 0.15, 0.30, 0.45, 0.60]
    phi = np.array(spread)
    for _ in range(AVERAGING_STEPS):
        graph = _resonant_graph(list(phi))
        following = phi.copy()
        for i in range(len(phi)):
            neighbors = list(graph.neighbors(i))
            if neighbors:
                mean = np.angle(np.mean(np.exp(1j * phi[neighbors])))
                following[i] = phi[i] + AVERAGING_FACTOR * np.angle(
                    np.exp(1j * (mean - phi[i]))
                )
        phi = following
    return {
        "status": "AUXILIARY_COMPARISON",
        "canonical_operators_executed": False,
        "physical_dimension_selection_demonstrated": False,
        "phase_threshold": float(PHASE_THRESHOLD),
        "averaging_factor": AVERAGING_FACTOR,
        "averaging_steps": AVERAGING_STEPS,
        "accretion": tuple(accretion),
        "incompatible_append": incompatible,
        "averaging": {
            "before": _simplex_readout(spread),
            "after": _simplex_readout(list(phi)),
        },
    }


def main() -> None:
    result = run_dimension_controls()
    print("AUXILIARY PHASE-THRESHOLD CLIQUE CONTROLS")
    print("Explicit phase samples and a selected averaging map; no glyph execution.")
    print("Selected threshold:", result["phase_threshold"])
    sizes = [item["max_clique_size"] for item in result["accretion"]]
    dimensions = [item["simplex_dimension"] for item in result["accretion"]]
    print("Compatible samples appended: clique sizes", sizes)
    print("Corresponding simplex dimensions:", dimensions)
    for label, key in (
        ("Incompatible append", "incompatible_append"),
        ("Auxiliary averaging", "averaging"),
    ):
        pair = result[key]
        print(
            label,
            "clique size:",
            pair["before"]["max_clique_size"],
            "->",
            pair["after"]["max_clique_size"],
        )
        print(
            label,
            "simplex dimension:",
            pair["before"]["simplex_dimension"],
            "->",
            pair["after"]["simplex_dimension"],
        )
    print(
        "These finite constructions do not demonstrate autonomous or physical dimension."
    )


if __name__ == "__main__":
    main()
