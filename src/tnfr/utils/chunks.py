"""Chunk sizing heuristics for batching structural computations.

The helpers in this module determine how large each processing block should be
when splitting work across workers or vectorised loops.  They take into account
the number of items involved, approximate memory pressure, and available CPU
parallelism so the caller can balance throughput with deterministic behaviour.
"""

from __future__ import annotations

import math
import os
from typing import Final

DEFAULT_APPROX_BYTES_PER_ITEM: Final[int] = 64
DEFAULT_CHUNK_CLAMP: Final[int] | None = 131_072


def _estimate_available_memory() -> int | None:
    """Best-effort estimation of free memory available to the process."""

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (
        AttributeError,
        ValueError,
        OSError,
    ):  # pragma: no cover - platform specific
        return None
    if page_size <= 0 or avail_pages <= 0:
        return None
    return int(page_size) * int(avail_pages)


def auto_chunk_size(
    total_items: int,
    *,
    minimum: int = 1,
    approx_bytes_per_item: int = DEFAULT_APPROX_BYTES_PER_ITEM,
    clamp_to: int | None = DEFAULT_CHUNK_CLAMP,
) -> int:
    """Infer a safe chunk length when the caller does not specify one."""

    if total_items <= 0:
        return 0

    minimum = max(1, minimum)
    approx_bytes_per_item = max(1, approx_bytes_per_item)

    available_memory = _estimate_available_memory()
    if available_memory is not None and available_memory > 0:
        safe_bytes = max(approx_bytes_per_item * minimum, available_memory // 8)
        mem_bound = max(minimum, min(total_items, safe_bytes // approx_bytes_per_item))
    else:
        mem_bound = total_items

    if clamp_to is not None:
        mem_bound = min(mem_bound, clamp_to)

    cpu_count = os.cpu_count() or 1
    target_chunks = max(1, cpu_count * 4)
    cpu_chunk = max(minimum, math.ceil(total_items / target_chunks))
    baseline = max(minimum, min(total_items, 1024))
    target = max(cpu_chunk, baseline)

    chunk = min(mem_bound, target)
    chunk = max(minimum, min(total_items, chunk))
    return chunk


def resolve_chunk_size(
    chunk_size: int | None,
    total_items: int,
    *,
    minimum: int = 1,
    approx_bytes_per_item: int = DEFAULT_APPROX_BYTES_PER_ITEM,
    clamp_to: int | None = DEFAULT_CHUNK_CLAMP,
) -> int:
    """Return a sanitised chunk size honouring automatic fallbacks."""

    if total_items <= 0:
        return 0

    resolved: int | None
    if chunk_size is None:
        resolved = None
    else:
        try:
            resolved = int(chunk_size)
        except (TypeError, ValueError):
            resolved = None
        else:
            if resolved <= 0:
                resolved = None

    if resolved is None:
        resolved = auto_chunk_size(
            total_items,
            minimum=minimum,
            approx_bytes_per_item=approx_bytes_per_item,
            clamp_to=clamp_to,
        )

    return max(minimum, min(total_items, resolved))


__all__ = ["auto_chunk_size", "resolve_chunk_size"]
