"""Compatibility wrapper for graph helpers moved to :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.graph import (
    get_graph,
    get_graph_mapping,
    mark_dnfr_prep_dirty,
    supports_add_edge,
)

__all__ = (
    "get_graph",
    "get_graph_mapping",
    "mark_dnfr_prep_dirty",
    "supports_add_edge",
)

warnings.warn(
    "'tnfr.graph_utils' is deprecated; import from 'tnfr.utils.graph' instead",
    DeprecationWarning,
    stacklevel=2,
)
