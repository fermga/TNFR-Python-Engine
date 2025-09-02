"""Utilities for working with collections and weights."""
from __future__ import annotations

from typing import (
    Iterable,
    Sequence,
    Dict,
    Any,
    Callable,
    TypeVar,
    Mapping,
    Optional,
    Collection,
    cast,
)
import logging
import math
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

MAX_MATERIALIZE_DEFAULT = 1000  # default materialization limit in ensure_collection


def ensure_collection(
    it: Iterable[T], *, max_materialize: int | None = MAX_MATERIALIZE_DEFAULT
) -> Collection[T]:
    """Return ``it`` if it's a ``Collection`` else materialize into ``tuple``.

    Strings and bytes are treated as single elements rather than iterables. If
    ``it`` is not iterable, :class:`TypeError` is raised.
    """
    if isinstance(it, Collection) and not isinstance(it, (str, bytes, bytearray)):
        return it
    if isinstance(it, (str, bytes, bytearray)):
        return cast(Collection[T], (it,))
    if max_materialize is not None and max_materialize < 0:
        raise ValueError("'max_materialize' must be non-negative")
    try:
        limit = MAX_MATERIALIZE_DEFAULT if max_materialize is None else max_materialize
        it = iter(it)
        data = tuple(islice(it, limit))
        extra = next(it, None)
        if extra is not None:
            raise ValueError(f"Iterable materialization exceeded {limit} items")
        return data
    except TypeError as exc:
        raise TypeError(f"{it!r} is not iterable") from exc


def _convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    key: str | None = None,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Try to convert ``value`` using ``conv`` handling errors."""
    try:
        return True, conv(value)
    except (ValueError, TypeError) as exc:
        level = log_level if log_level is not None else (
            logging.ERROR if strict else logging.DEBUG
        )
        if key is not None:
            logger.log(level, "No se pudo convertir el valor para %r: %s", key, exc)
        else:
            logger.log(level, "No se pudo convertir el valor: %s", exc)
        if strict:
            raise
        return False, None


def normalize_weights(
    dict_like: Dict[str, Any],
    keys: Iterable[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
) -> Dict[str, float]:
    """Normalize ``keys`` in ``dict_like`` so their sum is 1."""
    keys = list(keys)
    default_float = float(default)

    weights = _convert_weights(dict_like, keys, default_float, error_on_negative)
    _validate_weights(weights, error_on_negative)
    return _normalize_distribution(weights, keys)


def _convert_weights(
    dict_like: Mapping[str, Any],
    keys: Iterable[str],
    default_float: float,
    error_on_negative: bool,
) -> Dict[str, float]:
    def _to_float(k: str) -> float:
        val = dict_like.get(k, default_float)
        ok, converted = _convert_value(
            val,
            float,
            strict=error_on_negative,
            key=k,
            log_level=logging.WARNING,
        )
        return converted if ok and converted is not None else default_float

    return {k: _to_float(k) for k in keys}


def _validate_weights(weights: Mapping[str, float], error_on_negative: bool) -> None:
    negatives = {k: v for k, v in weights.items() if v < 0}
    if not negatives:
        return
    if error_on_negative:
        raise ValueError(f"Pesos negativos detectados: {negatives}")
    logger.warning("Pesos negativos detectados: %s", negatives)


def _normalize_distribution(
    weights: Mapping[str, float], keys: Sequence[str]
) -> Dict[str, float]:
    total = math.fsum(weights.values())
    n = len(keys)
    if total <= 0:
        if n == 0:
            return {}
        uniform = 1.0 / n
        return {k: uniform for k in keys}
    return {k: v / total for k, v in weights.items()}


def normalize_counter(counts: Mapping[str, int]) -> tuple[Dict[str, float], int]:
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
) -> Dict[str, float]:
    """Aggregate values of ``dist`` according to ``groups``."""
    out: Dict[str, float] = dict(dist)
    for label, keys in groups.items():
        out[f"{prefix}{label}"] = sum(dist.get(k, 0.0) for k in keys)
    return out

