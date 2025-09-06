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
import itertools

from .value_utils import _convert_value

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
    """Return ``it`` if it is a ``Collection``; otherwise materialize into a tuple.

    Strings and bytes are treated as single elements rather than iterables. When
    ``max_materialize`` is ``None``, the entire iterable is materialized without a
    limit. A :class:`ValueError` is raised if ``max_materialize`` is negative or if
    the iterable yields more than ``max_materialize`` items. A :class:`TypeError`
    is raised when ``it`` is not iterable.
    """
    if isinstance(it, Collection) and not isinstance(
        it, (str, bytes, bytearray)
    ):
        return it
    if isinstance(it, (str, bytes, bytearray)):
        return cast(Collection[T], (it,))
    if max_materialize is not None and max_materialize < 0:
        raise ValueError("'max_materialize' must be non-negative")
    try:
        if max_materialize is None:
            return tuple(it)
        limit = max_materialize
        out = tuple(itertools.islice(it, limit + 1))
        if len(out) > limit:
            raise ValueError(
                f"Iterable produced {len(out)} items, exceeds limit {limit}"
            )
        return out[:limit]
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
        weights[k] = w
        if w < 0:
            negatives[k] = w
    if negatives:
        if error_on_negative:
            raise ValueError(f"Pesos negativos detectados: {negatives}")
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
    for label, keys in groups.items():
        out[f"{prefix}{label}"] = sum(dist.get(k, 0.0) for k in keys)
    return out
