#!/usr/bin/env python3
"""Optical cavity feedback loop aligning laser, mirror stage, and detector."""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Acoplamiento as Coupling,
    Autoorganizacion as SelfOrganization,
    Coherencia as Coherence,
    Emision as Emission,
    Expansion as Expansion,
    Mutacion as Mutation,
    Recepcion as Reception,
    Resonancia as Resonance,
    Silencio as Silence,
)
from tnfr.trace import register_trace


def run_example() -> None:
    """Align an optical cavity after a thermal drift using TNFR operators."""
    G, _ = create_nfr("LaserHead", epi=0.23, vf=1.0, theta=0.0)
    create_nfr("MirrorStage", epi=0.28, vf=1.08, theta=0.15, graph=G)
    create_nfr("DetectorArray", epi=0.19, vf=0.92, theta=-0.22, graph=G)

    for node, component in {
        "LaserHead": "pump + modulation source",
        "MirrorStage": "piezo alignment stage",
        "DetectorArray": "photodiode feedback",
    }.items():
        G.nodes[node]["component"] = component

    workflow = {
        "LaserHead": [
            Emission(),
            Reception(),
            Coherence(),
            Coupling(),
            Resonance(),
            Silence(),
        ],
        "MirrorStage": [
            Emission(),
            Reception(),
            Coherence(),
            SelfOrganization(),
            Mutation(),
            Coherence(),
            Resonance(),
            Silence(),
        ],
        "DetectorArray": [
            Emission(),
            Reception(),
            Coherence(),
            Expansion(),
            Resonance(),
            Silence(),
        ],
    }

    for node, ops in workflow.items():
        run_sequence(G, node, ops)

    register_metrics_callbacks(G)
    register_trace(G)
    run(G, steps=6, dt=0.1)

    C, mean_delta_nfr, _ = compute_coherence(G, return_means=True)
    si = compute_Si(G)

    print(f"C(t)={C:.3f}, ΔNFR̄={mean_delta_nfr:.3f}")
    print({device: round(value, 3) for device, value in si.items()})


if __name__ == "__main__":
    run_example()
