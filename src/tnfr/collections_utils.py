"""Utilities for working with collections and weights."""

from __future__ import annotations

from typing import (
    Iterable,
    Any,
    TypeVar,
    Mapping,
    Collection,
    cast,
)
import logging
import math

from .value_utils import _convert_value
from itertools import islice

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_MATERIALIZE_DEFAULT",
    "ensure_collection",
    "normalize_weights",
    "normalize_counter",
    "mix_groups",
]

MAX_MATERIALIZE_DEFAULT = (
    1000  # default materialization limit in ensure_collection
)


def ensure_collection(
    it: Iterable[T], *, max_materialize: int | None = MAX_MATERIALIZE_DEFAULT
) -> Collection[T]:
    """Return ``it`` as a ``Collection`` materializing if necessary.

    Strings and bytes are treated as single elements. ``max_materialize``
    controls the maximum number of items to materialize when ``it`` is not
    already a collection; ``None`` means no limit. A :class:`ValueError`` is
    raised for negative ``max_materialize`` or when the iterable yields more
    items than allowed. ``TypeError`` is raised when ``it`` is not iterable.
    """

    if isinstance(it, Collection) and not isinstance(it, (str, bytes, bytearray)):
        return it
    if isinstance(it, (str, bytes, bytearray)):
        return cast(Collection[T], (it,))
    if max_materialize is not None and max_materialize < 0:
        raise ValueError("'max_materialize' must be non-negative")

    def _materialise(iterable: Iterable[T]) -> Collection[T]:
        if max_materialize is None:
            return tuple(iterable)
        limit = max_materialize
        if limit == 0:
            return ()
        items = list(islice(iterable, limit + 1))
        if len(items) > limit:
            raise ValueError(
                f"Iterable produced {len(items)} items, exceeds limit {limit}"
            )
        return tuple(items)

    try:
        return _materialise(it)
    except TypeError as exc:
        raise TypeError(f"{it!r} is not iterable") from exc


def normalize_weights(
    dict_like: dict[str, Any],
    keys: Iterable[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
) -> dict[str, float]:
    """Normalize ``keys`` in ``dict_like`` so their sum is 1."""
    keys = list(keys)
    default_float = float(default)
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
            raise ValueError(f"Pesos negativos detectados: {negatives}")
    if negatives and not error_on_negative:
        logger.warning("Pesos negativos detectados: %s", negatives)
    total = math.fsum(weights.values())
    n = len(keys)
    if total <= 0:
        if n == 0:
            return {}
        uniform = 1.0 / n
        return {k: uniform for k in keys}
    return {k: weights[k] / total for k in keys}


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
