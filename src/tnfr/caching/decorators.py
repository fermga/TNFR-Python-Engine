"""Decorators for transparent caching of TNFR computations.

.. deprecated::
   This module is deprecated. Import from ``tnfr.cache`` or ``tnfr.utils.cache`` instead.

This module provides decorator-based caching that integrates seamlessly with
existing TNFR functions, automatically managing cache keys, dependencies,
and invalidation.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from tnfr.caching.decorators is deprecated. "
    "Use 'from tnfr.cache import cache_tnfr_computation, invalidate_function_cache' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from consolidated location for backward compatibility
from ..utils.cache import (
    CacheLevel,
    TNFRHierarchicalCache,
    cache_tnfr_computation,
    invalidate_function_cache,
    get_global_cache,
    set_global_cache,
    reset_global_cache,
)

__all__ = [
    "cache_tnfr_computation",
    "get_global_cache",
    "set_global_cache",
    "invalidate_function_cache",
    "reset_global_cache",
]
