"""Multi-scale hierarchical TNFR network support.

This module implements operational fractality (§3.7) by enabling TNFR networks
to operate recursively at multiple scales simultaneously, preserving structural
coherence across scale transitions.

Canonical Invariants Preserved
------------------------------
1. EPI operational fractality - nested EPIs maintain functional identity
2. Cross-scale ΔNFR - coherence propagates between scales
3. Phase synchrony - maintained within and across scales
4. Deterministic evolution - reproducible multi-scale dynamics

Examples
--------
Create a hierarchical network spanning quantum to organism scales:

>>> from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition
>>> scales = [
...     ScaleDefinition("quantum", node_count=1000, coupling_strength=0.8),
...     ScaleDefinition("molecular", node_count=500, coupling_strength=0.6),
...     ScaleDefinition("cellular", node_count=100, coupling_strength=0.4),
... ]
>>> network = HierarchicalTNFRNetwork(scales)
>>> # Network supports simultaneous evolution at all scales
"""

from __future__ import annotations

from .hierarchical import HierarchicalTNFRNetwork, ScaleDefinition

__all__ = ["HierarchicalTNFRNetwork", "ScaleDefinition"]
