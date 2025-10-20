"""Structural analysis."""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from .constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY
from .dynamics import (
    set_delta_nfr_hook,
    dnfr_epi_vf_mixed,
)
from .types import DeltaNFRHook, NodeId, TNFRGraph
from .operators.definitions import (
    Operator,
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
)
from .operators.registry import OPERATORS
from .validation import validate_sequence


# ---------------------------------------------------------------------------
# 1) NFR factory
# ---------------------------------------------------------------------------


def create_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = dnfr_epi_vf_mixed,
) -> tuple[TNFRGraph, str]:
    """Create a graph with an initialised NFR node.

    Returns the tuple ``(G, name)`` for convenience.
    """
    G = graph if graph is not None else nx.Graph()
    G.add_node(
        name,
        **{
            EPI_PRIMARY: float(epi),
            VF_PRIMARY: float(vf),
            THETA_PRIMARY: float(theta),
        },
    )
    set_delta_nfr_hook(G, dnfr_hook)
    return G, name


__all__ = (
    "create_nfr",
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
    "OPERATORS",
    "validate_sequence",
    "run_sequence",
)


def run_sequence(G: TNFRGraph, node: NodeId, ops: Iterable[Operator]) -> None:
    """Execute a sequence of operators on ``node`` after validation."""

    compute = G.graph.get("compute_delta_nfr")
    ops_list = list(ops)
    names = [op.name for op in ops_list]

    ok, msg = validate_sequence(names)
    if not ok:
        raise ValueError(f"Invalid sequence: {msg}")

    for op in ops_list:
        op(G, node)
        if callable(compute):
            compute(G)
        # ``update_epi_via_nodal_equation`` was previously invoked here to
        # recalculate the EPI value after each operator. The responsibility for
        # updating EPI now lies with the dynamics hook configured in
        # ``compute_delta_nfr`` or with external callers.
