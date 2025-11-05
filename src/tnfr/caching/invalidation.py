"""Graph change tracking for intelligent cache invalidation.

.. deprecated:: 
   This module is deprecated. Import from ``tnfr.cache`` or ``tnfr.utils.cache`` instead.

This module provides hooks to track structural changes in TNFR graphs and
trigger selective cache invalidation based on which properties changed.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from tnfr.caching.invalidation is deprecated. "
    "Use 'from tnfr.cache import GraphChangeTracker, track_node_property_update' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from consolidated location for backward compatibility
from ..utils.cache import (
    GraphChangeTracker,
    track_node_property_update,
)

__all__ = ["GraphChangeTracker", "track_node_property_update"]
