#!/usr/bin/env python3
"""Controlled dissonance scenario demonstrating bifurcation and re-coherence."""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Coupling,
    Coherence,
    Dissonance,
    Emission,
    Mutation,
    Reception,
    Resonance,
    Silence,
    Transition,
)
from tnfr.trace import register_trace
from tnfr.glyph_history import ensure_history


def run_example() -> None:
    """Execute the three-node ring with a controlled dissonance pulse."""
    G, _ = create_nfr("A", epi=0.24, vf=1.0, theta=0.0)
    create_nfr("B", epi=0.18, vf=0.9, theta=0.45, graph=G)
    create_nfr("C", epi=0.27, vf=1.1, theta=-0.35, graph=G)
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    G.graph.update(
        {
            "UM_FUNCTIONAL_LINKS": True,
            "UM_CANDIDATE_COUNT": 2,
            "UM_CANDIDATE_MODE": "proximity",
        }
    )

    run_sequence(
        G,
        "A",
        [Emission(), Reception(), Coherence(), Coupling(), Resonance(), Coherence(), Silence()],
    )
    run_sequence(
        G,
        "B",
        [Emission(), Reception(), Coherence(), Resonance(), Coherence(), Silence()],
    )
    run_sequence(
        G,
        "C",
        [
            Emission(),
            Reception(),
            Coherence(),
            Transition(),
            Dissonance(),
            Mutation(),
            Coherence(),
            Silence(),
        ],
    )

    register_metrics_callbacks(G)
    register_trace(G)
    run(G, steps=8, dt=0.1)

    C, dnfr_mean, depi_mean = compute_coherence(G, return_means=True)
    Si = compute_Si(G)
    history = ensure_history(G)

    print(f"C(t)={C:.3f}, ΔNFR̄={dnfr_mean:.3f}, dEPI/dt̄={depi_mean:.3f}")
    print({node: round(val, 3) for node, val in Si.items()})
    print(history["W_stats"][-1])
    print(list(history["nodal_diag"][-1].items())[:2])


if __name__ == "__main__":
    run_example()
