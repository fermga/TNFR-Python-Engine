"""Compatibility layer for cache helpers.

This module re-exports the public APIs from :mod:`node_cache` and
:mod:`edge_cache` so legacy imports such as ``tnfr.helpers.cache`` remain
functional. New code should import from :mod:`tnfr.helpers.node_cache` or
:mod:`tnfr.helpers.edge_cache` directly.

# TODO: drop this shim in a future major release.
"""

from .node_cache import *  # noqa: F401,F403
from .edge_cache import *  # noqa: F401,F403

from .node_cache import __all__ as _node_all
from .edge_cache import __all__ as _edge_all

__all__ = (*_node_all, *_edge_all)

