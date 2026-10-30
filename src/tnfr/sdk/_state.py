"""Detached SDK graph data; runtime caches are rebuilt by their owners."""

import math
from copy import deepcopy

import networkx as nx

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import compute_circular_mean
from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS


def stored_phase(data) -> float:
    """Read the authoritative phase without repairing malformed stored values."""
    raw = get_attr(data, ALIAS_THETA, 0.0, strict=True, conv=lambda value: value)
    try:
        return finite_represented_real(raw, "phase")[0]
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(str(exc)) from exc


def observed_mean_phase(graph: nx.Graph) -> float | None:
    """Use the shared circular-mean tolerance for both public SDK interfaces."""
    phases = [stored_phase(data) for _, data in graph.nodes(data=True)]
    if not phases:
        return None
    try:
        mean = compute_circular_mean(phases)
    except TNFRValueError:
        # Input has already been validated. The remaining rejection is the
        # shared owner's undefined finite resultant, not missing/invalid data.
        return None
    period = 2.0 * math.pi
    normalized = mean % period
    return 0.0 if normalized == period else normalized


def copy_graph_state(graph: nx.Graph) -> nx.Graph:
    """Copy graph kind and mutable data while retaining node/key identities.

    Rebuildable TNFR caches are excluded. Registered ``CallbackSpec`` carriers,
    functions and the exact external resources recognized by the transaction
    boundary (standard locks and loggers, plus validated immutable mapping
    proxies) retain identity. Callable objects stored as ordinary data and all
    other values follow Python deepcopy semantics. Mutable state and alias
    topology crossing an external resource remain outside the copy-independence
    boundary. Explicit references from graph data back to the graph container
    itself are likewise outside this data-copy contract.
    """
    snapshot = graph.copy()
    for key in GRAPH_RUNTIME_CACHE_KEYS:
        snapshot.graph.pop(key, None)
    memo = {id(node): node for node in graph}
    if graph.is_multigraph():
        memo.update({id(key): key for _, _, key in graph.edges(keys=True)})
    # Keep SDK copies aligned with the transaction layer's centralized resource
    # classification without introducing an import-time SDK/operator cycle.
    from ..operators.network_stage import _seed_runtime_resource_memo

    # Structural node and edge-key identities are opaque even when they are
    # tuple-like. Seed them as already visited while the shared transaction
    # walker discovers resources throughout graph data without virtual views.
    _seed_runtime_resource_memo(
        snapshot,
        memo,
        set(memo),
        preserve_bound_methods=False,
    )
    return deepcopy(snapshot, memo)
