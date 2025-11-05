"""Persistent cache with disk-backed storage for expensive TNFR computations.

.. deprecated:: 
   This module is deprecated. Import from ``tnfr.cache`` or ``tnfr.utils.cache`` instead.

This module provides optional persistence for cache entries, allowing
expensive computations to survive between sessions.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from tnfr.caching.persistence is deprecated. "
    "Use 'from tnfr.cache import PersistentTNFRCache' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from consolidated location for backward compatibility
from ..utils.cache import (
    PersistentTNFRCache,
)

__all__ = ["PersistentTNFRCache"]
