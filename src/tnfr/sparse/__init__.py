"""Sparse representations for memory-efficient TNFR networks.

This module provides memory-optimized graph representations that reduce
per-node memory footprint from ~8.5KB to <1KB while preserving all TNFR
canonical invariants.

Canonical Invariants Preserved
------------------------------
1. Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
2. Sparse storage: only non-default values stored
3. Cache coherence: intelligent caching for repeated computations
4. Deterministic computation: same inputs yield same outputs

Examples
--------
Create a sparse graph with 10,000 nodes:

>>> from tnfr.sparse import SparseTNFRGraph
>>> graph = SparseTNFRGraph(node_count=10000, expected_density=0.1)
>>> footprint = graph.memory_footprint()
>>> print(f"Memory per node: {footprint.per_node_kb:.2f} KB")  # doctest: +SKIP
Memory per node: 0.85 KB
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
