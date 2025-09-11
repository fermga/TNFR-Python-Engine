"""Utilities for working with generic collections and weight mappings."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import islice
from typing import Any, Callable, Iterator, TypeVar, cast

from .logging_utils import get_logger, warn_once


from .helpers.numeric import kahan_sum

T = TypeVar("T")

logger = get_logger(__name__)

STRING_TYPES = (str, bytes, bytearray)

NEGATIVE_WEIGHTS_MSG = "Negative weights detected: %s"


_WARNED_NEGATIVE_KEYS_LIMIT = 1024
_log_negative_keys_once = warn_once(
    logger, NEGATIVE_WEIGHTS_MSG, maxsize=_WARNED_NEGATIVE_KEYS_LIMIT
)


def clear_warned_negative_keys() -> None:
    """Clear the cache of warned negative weight keys."""
    _log_negative_keys_once.clear()


def is_non_string_sequence(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is an ``Iterable`` but not string-like or a mapping."""
    return isinstance(obj, Iterable) and not isinstance(obj, (*STRING_TYPES, Mapping))


def flatten_structure(
    obj: Any,
    *,
    expand: Callable[[Any], Iterable[Any] | None] | None = None,
) -> Iterator[Any]:
    """Yield leaf items from ``obj``.

    The order of yielded items follows the order of the input iterable when it
    is defined. For unordered iterables like :class:`set` the resulting order is
    arbitrary. Mappings are treated as atomic items and not expanded.

    Parameters
    ----------
    obj:
        Object that may contain nested iterables.
    expand:
        Optional callable returning a replacement iterable for ``item``. When
        it returns ``None`` the ``item`` is processed normally.
    """

    stack = deque([obj])
    while stack:
        item = stack.pop()
        if expand is not None:
            replacement = expand(item)
            if replacement is not None:
                stack.extendleft(replacement)
                continue
        if is_non_string_sequence(item):
            stack.extendleft(item)
        else:
            yield item


__all__ = (
    "MAX_MATERIALIZE_DEFAULT",
    "is_non_string_sequence",
    "flatten_structure",
    "ensure_collection",
    "normalize_weights",
    "normalize_counter",
    "mix_groups",
    "clear_warned_negative_keys",
)

MAX_MATERIALIZE_DEFAULT: int = 1000
"""Default materialization limit used by :func:`ensure_collection`.

This guard prevents accidentally consuming huge or infinite iterables when a
limit is not explicitly provided. Pass ``max_materialize=None`` to disable the
limit.
"""


def _validate_limit(max_materialize: int | None) -> int | None:
    """Normalize and validate ``max_materialize`` returning a usable limit."""
    if max_materialize is None:
        return None
    limit = int(max_materialize)
    if limit < 0:
        raise ValueError("'max_materialize' must be non-negative")
    return limit


def ensure_collection(
    it: Iterable[T],
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
    error_msg: str | None = None,
    treat_strings_as_iterables: bool = False,
) -> Collection[T]:
    """Return ``it`` as a ``Collection`` materializing if necessary.

    Step 1 detects collections and handles string-like inputs (``str``,
    ``bytes`` and ``bytearray``) specially. By default these are wrapped as a
    single item tuple so they are not iterated character by character. Pass
    ``treat_strings_as_iterables=True`` to materialize them like any other
    iterable. ``max_materialize`` limits materialization for non-collection
    iterables; ``None`` disables the limit. ``error_msg`` customizes the
    :class:`ValueError` raised when the iterable yields more items than allowed.
    ``TypeError`` is raised when ``it`` is not iterable. The input is consumed
    at most once and no extra items beyond the limit are stored in memory.
    """

    # Step 1: detect collections and raw strings/bytes early
    if isinstance(it, Collection):
        if isinstance(it, STRING_TYPES):
            if not treat_strings_as_iterables:
                return (cast(T, it),)
            # Fall through to materialization when treating as iterable
        else:
            return it

    # Step 2: validate limit
    limit = _validate_limit(max_materialize)

    # Step 3: materialize up to ``limit`` items using ``islice`` only once
    try:
        if limit is None:
            return tuple(it)
        if limit == 0:
            return ()
        materialized = tuple(islice(it, limit + 1))
        if len(materialized) > limit:
            examples = ", ".join(repr(x) for x in materialized[:3])
            msg = error_msg or (
                f"Iterable produced {len(materialized)} items, "
                f"exceeds limit {limit}; first items: [{examples}]"
            )
            raise ValueError(msg)
        return materialized
    except TypeError as exc:
        raise TypeError(f"{it!r} is not iterable") from exc


def _process_negative_weights(
    weights: dict[str, float],
    negatives: dict[str, float],
    warn_once: bool,
    total: float,
) -> float:
    """Handle negative weights by logging, clamping and adjusting total."""
    if warn_once:
        # ``warn_once`` returns a callable that logs at most once per key.
        # ``_log_negative_keys_once`` expects a mapping of keys to values so
        # pass the entire ``negatives`` dict at once.  This ensures each
        # negative weight is only warned about on its first occurrence across
        # all calls to :func:`normalize_weights`.
        _log_negative_keys_once(negatives)
    else:
        logger.warning(NEGATIVE_WEIGHTS_MSG, negatives)
    for k, w in negatives.items():
        weights[k] = 0.0
        total -= w
    return total


def normalize_weights(
    dict_like: dict[str, Any],
    keys: Iterable[str] | Sequence[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
    warn_once: bool = True,
    error_on_conversion: bool = False,
) -> dict[str, float]:
    """Normalize ``keys`` in ``dict_like`` so their sum is 1.

    ``keys`` may be any iterable of strings and is materialized once while
    collapsing repeated entries preserving their first occurrence.

    Negative weights are handled according to ``error_on_negative``. When
    ``True`` a :class:`ValueError` is raised. Otherwise negatives are logged,
    replaced with ``0`` and the remaining weights are renormalized. If all
    weights are non-positive a uniform distribution is returned.

    Conversion errors are controlled separately by ``error_on_conversion``. When
    ``True`` any :class:`TypeError` or :class:`ValueError` while converting a
    value to ``float`` is propagated. Otherwise the error is logged and the
    ``default`` value is used.

    When ``warn_once`` is ``True`` warnings for a given key are emitted only on
    their first occurrence across calls.
    """
    keys = list(dict.fromkeys(keys))
    default_float = float(default)
    if not keys:
        return {}

    def _get_float(key: str) -> float:
        val = dict_like.get(key, default_float)
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            if error_on_conversion:
                raise
            logger.warning("Could not convert value for %r: %s", key, exc)
            return default_float

    weights = {k: _get_float(k) for k in keys}
    negatives = {k: w for k, w in weights.items() if w < 0}
    total = kahan_sum(weights.values())
    if negatives:
        if error_on_negative:
            raise ValueError(NEGATIVE_WEIGHTS_MSG % negatives)
        total = _process_negative_weights(weights, negatives, warn_once, total)
    if total <= 0:
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}
    return {k: w / total for k, w in weights.items()}


def normalize_counter(
    counts: Mapping[str, float | int],
) -> tuple[dict[str, float], float]:
    """Normalize a ``Counter`` returning proportions and total."""
    total = kahan_sum(counts.values())
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
            f"{prefix}{label}": kahan_sum(dist.get(k, 0.0) for k in keys)
            for label, keys in groups.items()
        }
    )
    return out
