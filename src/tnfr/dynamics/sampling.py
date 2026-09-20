"""Sampling helpers used by runtime selectors and glyph application."""

from __future__ import annotations

from typing import cast

from ..rng import _rng_for_step, base_seed
from ..types import NodeId, TNFRGraph
from ..utils import cached_node_list

__all__ = ("update_node_sample",)


def update_node_sample(G: TNFRGraph, *, step: int) -> None:
    """Refresh ``G.graph['_node_sample']`` with a random subset of nodes.

    The sample is limited by ``UM_CANDIDATE_COUNT`` and refreshed every
    simulation step. When the network is small (``< 50`` nodes) or the limit
    is non‑positive, the full node set is used and sampling is effectively
    disabled. A snapshot of nodes is cached via the NodeCache helper from
    ``tnfr.utils`` stored in
    ``G.graph['_node_list_cache']`` and reused across steps, with node identity
    and ordering checked by that owner. Sampling operates directly on the cached
    tuple of nodes. ``step`` is the runtime's admitted invocation ordinal,
    independent of telemetry retention and of the physical time increment.
    """
    graph = G.graph
    seed = base_seed(G)
    limit = int(graph.get("UM_CANDIDATE_COUNT", 0))
    nodes = cast(tuple[NodeId, ...], cached_node_list(G))
    current_n = len(nodes)
    if limit <= 0 or current_n < 50 or limit >= current_n:
        graph["_node_sample"] = nodes
        return

    rng = _rng_for_step(seed, step)
    graph["_node_sample"] = rng.sample(nodes, limit)
