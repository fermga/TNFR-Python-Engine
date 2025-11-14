"""Core utilities for TNFR operator metrics.

This module provides shared utilities and helper functions used across
all operator-specific metric collectors.

Terminology (TNFR semantics):
- "node" == resonant locus (coherent structural anchor); retained for NetworkX compatibility.
- Not related to the Node.js runtime; purely graph-theoretic locus.
- Future migration may introduce `locus` aliases without breaking public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph
else:
    NodeId = Any  # runtime fallback
    TNFRGraph = Any  # runtime fallback

from ..alias import get_attr, get_attr_str
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)

# Emission timestamp alias - defensive runtime check
_HAS_EMISSION_TIMESTAMP_ALIAS = False
_ALIAS_EMISSION_TIMESTAMP_TUPLE: tuple[str, ...] = ()
try:
    from ..constants.aliases import ALIAS_EMISSION_TIMESTAMP as _ALIAS_TS  # type: ignore

    _ALIAS_EMISSION_TIMESTAMP_TUPLE = _ALIAS_TS
    _HAS_EMISSION_TIMESTAMP_ALIAS = True
except Exception:
    pass

__all__ = [
    "get_node_attr",
    "HAS_EMISSION_TIMESTAMP_ALIAS",
    "EMISSION_TIMESTAMP_TUPLE",
    "ALIAS_D2EPI",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_THETA",
    "ALIAS_VF",
]

# Export emission timestamp helpers for use in other metric modules
HAS_EMISSION_TIMESTAMP_ALIAS = _HAS_EMISSION_TIMESTAMP_ALIAS
EMISSION_TIMESTAMP_TUPLE = _ALIAS_EMISSION_TIMESTAMP_TUPLE


def get_node_attr(G: TNFRGraph, node: NodeId, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Get node attribute using alias fallback.

    Wrapper around alias.get_attr that ensures float return type and handles
    exceptions gracefully.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to get attribute from
    aliases : tuple[str, ...]
        Tuple of possible attribute names (aliases) to try
    default : float, optional
        Default value if attribute not found, by default 0.0

    Returns
    -------
    float
        Attribute value as float, or default if not found/invalid

    Notes
    -----
    This function is the single point of attribute access for all metric collectors,
    ensuring consistent handling of attribute aliases and type conversion.
    """
    value = get_attr(G.nodes[node], aliases, default)
    try:
        return float(cast(float, value))
    except Exception:
        return float(default)
