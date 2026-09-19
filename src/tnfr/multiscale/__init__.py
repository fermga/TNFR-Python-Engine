"""Multi-scale hierarchical TNFR network support.

This module manages distinct TNFR graphs, composes directed cross-scale
pressure contributions from simultaneous source snapshots, and advances EPI
through the nodal equation. It supplies an operational multi-scale model and
reproducible storage boundary. U3 admission, U5 hierarchy normalization, and
canonical operator words require their dedicated engine interfaces.

Examples
--------
Create a reproducible hierarchy:

>>> from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition
>>> scales = [
...     ScaleDefinition("micro", node_count=4, coupling_strength=0.8),
...     ScaleDefinition("macro", node_count=3, coupling_strength=0.4),
... ]
>>> network = HierarchicalTNFRNetwork(scales, seed=42)
>>> set(network.networks_by_scale) == {"micro", "macro"}
True
"""

from __future__ import annotations

from .hierarchical import HierarchicalTNFRNetwork, ScaleDefinition

__all__ = ["HierarchicalTNFRNetwork", "ScaleDefinition"]
