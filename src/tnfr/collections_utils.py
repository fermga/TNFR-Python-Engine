"""Utilities for working with collections and weights."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Collection, Sequence
from typing import Any, TypeVar, cast
import logging
import math
from itertools import islice
from .logging_utils import get_logger

from .value_utils import _convert_value

T = TypeVar("T")

logger = get_logger(__name__)

NEGATIVE_WEIGHTS_MSG = "Negative weights detected: %s"

__all__ = [
    "MAX_MATERIALIZE_DEFAULT",
    "ensure_collection",
    "normalize_weights",
    "normalize_counter",
    "mix_groups",
]

MAX_MATERIALIZE_DEFAULT: int = 1000  # default materialization limit in ensure_collection


def ensure_collection(
    it: Iterable[T], *, max_materialize: int | None = MAX_MATERIALIZE_DEFAULT
) -> Collection[T]:
    """Return ``it`` as a ``Collection`` materializing if necessary.

    Strings and bytes are treated as single elements. ``max_materialize``
    controls the maximum number of items to materialize when ``it`` is not
    already a collection; ``None`` means no limit. A :class:`ValueError`` is
    raised for negative ``max_materialize`` or when the iterable yields more
    items than allowed. ``TypeError`` is raised when ``it`` is not iterable.
    The input is consumed at most once and no extra items beyond the limit
    are stored in memory.
    """

    if isinstance(it, Collection) and not isinstance(it, (str, bytes, bytearray)):
        # Already a collection; no materialization needed
        return it
    if isinstance(it, (str, bytes, bytearray)):
        # Treat raw bytes/strings as single elements
        return cast(Collection[T], (it,))
    if max_materialize is not None and max_materialize < 0:
        raise ValueError("'max_materialize' must be non-negative")

    try:
        if max_materialize is None:
            # No limit: consume iterable fully
            return tuple(it)
        limit = max_materialize  # materialization cap
        if limit == 0:
            # Explicitly allow empty result without consumption
            return ()
        materialized = list(islice(it, limit + 1))
        if len(materialized) > limit:
            raise ValueError(
                f"Iterable produced {len(materialized)} items, exceeds limit {limit}"
            )
        return tuple(materialized)
    except TypeError as exc:
        raise TypeError(f"{it!r} is not iterable") from exc


def normalize_weights(
    dict_like: dict[str, Any],
    keys: Iterable[str] | Sequence[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
) -> dict[str, float]:
    """Normalize ``keys`` in ``dict_like`` so their sum is 1.

    ``keys`` may be any iterable of strings. Sequences and other collections
    are used directly while non-collection iterables are materialized.
    """
    if not isinstance(keys, Collection):
        keys = list(keys)
    default_float = float(default)
    if not keys:
        return {}
    weights: dict[str, float] = {}
    negatives: dict[str, float] = {}
    for k in keys:
        val = dict_like.get(k, default_float)
        ok, converted = _convert_value(
            val,
            float,
            strict=error_on_negative,
            key=k,
            log_level=logging.WARNING,
        )
        w = converted if ok and converted is not None else default_float
        if w < 0:
            negatives[k] = w
            w = 0.0
        weights[k] = w
    if negatives:
        if error_on_negative:
            raise ValueError(NEGATIVE_WEIGHTS_MSG % negatives)
        logger.warning(NEGATIVE_WEIGHTS_MSG, negatives)
    total = math.fsum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}
    return {k: w / total for k, w in weights.items()}


def normalize_counter(
    counts: Mapping[str, int],
) -> tuple[dict[str, float], int]:
    """Normalize a ``Counter`` returning proportions and total."""
    total = sum(counts.values())
    if total <= 0:
        return {}, 0
    dist = {k: v / total for k, v in counts.items() if v}
    return dist, total


def mix_groups(
    dist: Mapping[str, float],
    groups: Mapping[str, Iterable[str]],
    *,
    prefix: str = "_",
) -> dict[str, float]:
    """Aggregate values of ``dist`` according to ``groups``."""
    out: dict[str, float] = dict(dist)
    out.update(
        {
            f"{prefix}{label}": sum(dist.get(k, 0.0) for k in keys)
            for label, keys in groups.items()
        }
    )
    return out
