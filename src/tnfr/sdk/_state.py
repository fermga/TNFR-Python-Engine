"""Detached SDK graph data; runtime caches are rebuilt by their owners."""

from copy import deepcopy

import networkx as nx

from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS


def copy_graph_state(graph: nx.Graph) -> nx.Graph:
    """Copy graph kind and mutable data while retaining node/key identities.

    Rebuildable TNFR caches are excluded. Standard Python deepcopy rules apply
    to other data: functions remain shared, and unsupported runtime objects
    raise rather than silently becoming shared mutable state. This is not an
    arbitrary-runtime checkpoint or an independent copy of external callbacks.
    """
    snapshot = graph.copy()
    for key in GRAPH_RUNTIME_CACHE_KEYS:
        snapshot.graph.pop(key, None)
    memo = {id(node): node for node in graph}
    if graph.is_multigraph():
        memo.update({id(key): key for _, _, key in graph.edges(keys=True)})
    return deepcopy(snapshot, memo)
