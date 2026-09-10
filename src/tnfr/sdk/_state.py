"""Detached SDK graph data; runtime caches are rebuilt by their owners."""

from copy import deepcopy

import networkx as nx

from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS


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
