"""Structural analysis."""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from .constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from .dynamics import (
    dnfr_epi_vf_mixed,
    set_delta_nfr_hook,
)
from .operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from .operators.registry import OPERATORS
from .types import DeltaNFRHook, NodeId, TNFRGraph
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

    Parameters
    ----------
    name : str
        Identifier for the new node. The identifier is stored as the node key
        and must remain hashable by :mod:`networkx`.
    epi : float, optional
        Initial Primary Information Structure (EPI) assigned to the node. The
        value provides the baseline form that subsequent ΔNFR hooks reorganise
        through the nodal equation.
    vf : float, optional
        Structural frequency (νf, expressed in Hz_str) used as the starting
        reorganisation rate for the node.
    theta : float, optional
        Initial phase of the node in radians, used to keep phase alignment with
        neighbouring coherence structures.
    graph : TNFRGraph, optional
        Existing graph where the node will be registered. When omitted a new
        :class:`networkx.Graph` instance is created.
    dnfr_hook : DeltaNFRHook, optional
        Callable responsible for computing ΔNFR and updating EPI/νf after each
        operator application. By default the canonical ``dnfr_epi_vf_mixed``
        hook is installed, which keeps the nodal equation coherent with TNFR
        invariants.

    Returns
    -------
    tuple[TNFRGraph, str]
        The graph that stores the node together with the node identifier. The
        tuple form allows immediate reuse with :func:`run_sequence`.

    Raises
    ------
    None
        The factory does not introduce additional TNFR-specific errors. Any
        exceptions raised by :mod:`networkx` when adding nodes propagate
        unchanged.

    Examples
    --------
    Create a node whose ΔNFR hook keeps ``EPI`` and ``νf`` aligned with the
    canonical nodal equation.

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("seed", epi=1.0, vf=2.0)
    >>> def stabilise_delta(graph):
    ...     delta = graph.nodes[node][VF_PRIMARY] * 0.1
    ...     graph.nodes[node][DNFR_PRIMARY] = delta
    ...     graph.nodes[node][EPI_PRIMARY] += delta
    ...     graph.nodes[node][VF_PRIMARY] += delta * 0.05
    >>> set_delta_nfr_hook(G, stabilise_delta)

    The hook keeps the evolution of EPI and νf tied to ΔNFR, ready for
    :func:`run_sequence` or other operator pipelines.
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
    """Execute a sequence of operators on ``node`` after validation.

    Parameters
    ----------
    G : TNFRGraph
        Graph that stores the node and its ΔNFR orchestration hook. The hook is
        read from ``G.graph['compute_delta_nfr']`` and is responsible for
        keeping the nodal equation up to date after each operator.
    node : NodeId
        Identifier of the node that will receive the operators. The node must
        already contain the canonical attributes ``EPI``, ``νf`` and ``θ``.
    ops : Iterable[Operator]
        Iterable of canonical structural operators to apply. Their
        concatenation must respect the validated TNFR grammar.

    Returns
    -------
    None
        The function mutates ``G`` in-place by updating the node attributes.

    Raises
    ------
    ValueError
        Raised when the provided operator names do not satisfy the canonical
        sequence validation rules.

    Examples
    --------
    Apply three canonical operators while a ΔNFR hook updates EPI and νf after
    each step.

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import (
    ...     Emission,
    ...     Coherence,
    ...     Resonance,
    ...     create_nfr,
    ...     run_sequence,
    ... )
    >>> G, node = create_nfr("seed", epi=1.0, vf=2.0)
    >>> def stabilise_delta(graph):
    ...     delta = graph.nodes[node][VF_PRIMARY] * 0.1
    ...     graph.nodes[node][DNFR_PRIMARY] = delta
    ...     graph.nodes[node][EPI_PRIMARY] += delta
    ...     graph.nodes[node][VF_PRIMARY] += delta * 0.05
    >>> set_delta_nfr_hook(G, stabilise_delta)
    >>> run_sequence(G, node, [Emission(), Resonance(), Coherence()])
    After each operator the hook recomputes ΔNFR and adjusts EPI/νf, keeping the
    canonical nodal equation stable.
    """

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
