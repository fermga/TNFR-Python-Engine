from __future__ import annotations

from typing import Final

DEFAULT_APPROX_BYTES_PER_ITEM: Final[int]
DEFAULT_CHUNK_CLAMP: Final[int | None]

def auto_chunk_size(
    total_items: int,
    *,
    minimum: int = ...,
    approx_bytes_per_item: int = ...,
    clamp_to: int | None = ...,
) -> int: ...
def resolve_chunk_size(
    chunk_size: int | None,
    total_items: int,
    *,
    minimum: int = ...,
    approx_bytes_per_item: int = ...,
    clamp_to: int | None = ...,
) -> int: ...
