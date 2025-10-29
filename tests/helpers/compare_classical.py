"""Helpers comparing classical runtime traces with the math integration layer."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.node import NodeNX, add_edge
from tnfr.operators.definitions import Coherence, Emission, Reception, Resonance, Transition
from tnfr.structural import create_nfr, run_sequence

from .mathematics import build_node_with_operators


def classical_operator_snapshot(
    ops: Iterable[Any],
    *,
    epi: float = 0.8,
    nu_f: float = 1.2,
    theta: float = 0.1,
    partner_epi: float = 0.5,
    partner_nu_f: float = 0.9,
    partner_theta: float = 0.0,
    coupling: float = 1.0,
) -> Mapping[str, Mapping[str, float]]:
    """Return a deterministic classical snapshot for ``ops``."""

    G, primary = create_nfr("classic-seed", epi=epi, vf=nu_f, theta=theta)
    _, partner = create_nfr(
        "classic-partner",
        epi=partner_epi,
        vf=partner_nu_f,
        theta=partner_theta,
        graph=G,
    )
    add_edge(G, primary, partner, coupling)
    run_sequence(G, primary, list(ops))

    def _payload(node: str) -> Mapping[str, float]:
        nd = G.nodes[node]
        return {
            "EPI": float(nd[EPI_PRIMARY]),
            "vf": float(nd[VF_PRIMARY]),
            "theta": float(nd[THETA_PRIMARY]),
            "dnfr": float(nd[DNFR_PRIMARY]),
        }

    return {"classic-seed": _payload(primary), "classic-partner": _payload(partner)}


def math_sequence_summary(
    ops: Iterable[Any],
    *,
    epi: float = 0.8,
    nu_f: float = 1.2,
    theta: float = 0.1,
    partner_epi: float = 0.5,
    partner_nu_f: float = 0.9,
    partner_theta: float = 0.0,
    coupling: float = 1.0,
    rng_seed: int | None = None,
) -> tuple[dict[str, Any], NodeNX]:
    """Run ``NodeNX.run_sequence_with_validation`` mirroring the classical layout."""

    node, _, _ = build_node_with_operators(epi=epi, nu_f=nu_f, theta=theta)
    create_nfr(
        "math-partner",
        epi=partner_epi,
        vf=partner_nu_f,
        theta=partner_theta,
        graph=node.G,
    )
    add_edge(node.G, node.n, "math-partner", coupling)
    if rng_seed is None:
        rng = None
    else:
        rng = _SeedProxy(rng_seed)
    summary = node.run_sequence_with_validation(list(ops), enable_validation=True, rng=rng)
    return summary, node


DEFAULT_ACCEPTANCE_OPS = (
    Emission(),
    Reception(),
    Coherence(),
    Resonance(),
    Transition(),
)


class _SeedProxy:
    """Minimal adapter exposing a ``bit_generator.state`` integer."""

    def __init__(self, seed: int) -> None:
        self.bit_generator = type("_SeedCarrier", (), {"state": seed})()

