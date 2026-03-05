"""Unified buffer cache for TNFR metrics hot paths.

This module consolidates buffer management across hot path computations
(Sense index, coherence, ΔNFR) to eliminate duplication and ensure consistent
cache key patterns and invalidation strategies.
"""

from __future__ import annotations

from typing import Any

from ..types import GraphLike

__all__ = ("ensure_numpy_buffers",)

def ensure_numpy_buffers(
    G: GraphLike,
    *,
    key_prefix: str,
    count: int,
    buffer_count: int,
    dtype: Any = None,
    max_cache_entries: int | None = 128,
) -> tuple[Any, ...]: ...
