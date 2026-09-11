"""Sparse representations for memory-efficient TNFR networks.

This module provides memory-optimized graph representations whose tracked
component estimate can remain below 1 KB per node for genuinely sparse
workloads. It preserves explicit nodal-update and deterministic-storage
semantics.

Implemented Contracts
---------------------
1. Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
2. Sparse storage: only non-default values stored
3. Cache coherence: bounded write-age caching with dependency invalidation
4. Deterministic computation: same inputs yield same outputs

Canonical operator grammar, operator-only mutation, and U3 admission are
higher-level engine contracts; this storage representation does not enforce
them.

Examples
--------
Create a sparse graph with about ten expected neighbours per node:

>>> from tnfr.sparse import SparseTNFRGraph
>>> graph = SparseTNFRGraph(
...     node_count=1000, expected_density=0.01, seed=42
... )
>>> footprint = graph.memory_footprint()
>>> footprint.per_node_kb < 1.0
True

``expected_density`` controls seeded random initialization; it is neither a
preallocation setting nor a guaranteed storage bound. The realized topology,
non-default attributes, and cache occupancy determine the estimate. Python and
SciPy object headers, allocator fragmentation, and imported modules are outside
this component report.
"""

from __future__ import annotations

from .representations import (
    CompactAttributeStore,
    MemoryReport,
    SparseCache,
    SparseTNFRGraph,
)

__all__ = [
    "SparseTNFRGraph",
    "CompactAttributeStore",
    "MemoryReport",
    "SparseCache",
]
