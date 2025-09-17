"""Utilities for working with generic collections and weight mappings."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import islice
from typing import Any, Callable, Iterator, TypeVar, cast

from .logging_utils import get_logger
from .logging_utils import warn_once as _warn_once_factory
from .value_utils import convert_value
from .helpers.numeric import kahan_sum_nd

T = TypeVar("T")

logger = get_logger(__name__)

STRING_TYPES = (str, bytes, bytearray)

NEGATIVE_WEIGHTS_MSG = "Negative weights detected: %s"

_NEGATIVE_WARN_ONCE_MAXSIZE = 1024


def negative_weights_warn_once(
    *, maxsize: int = _NEGATIVE_WARN_ONCE_MAXSIZE
) -> Callable[[Mapping[str, float]], None]:
    """Return a ``WarnOnce`` callable for negative weight warnings.

    The returned callable may be reused across multiple
    :func:`normalize_weights` invocations to suppress duplicate warnings for
    the same keys.
    """

    return _warn_once_factory(logger, NEGATIVE_WEIGHTS_MSG, maxsize=maxsize)


def _log_negative_weights(negatives: Mapping[str, float]) -> None:
    """Log negative weight warnings without deduplicating keys."""

    logger.warning(NEGATIVE_WEIGHTS_MSG, negatives)


def _resolve_negative_warn_handler(
    warn_once: bool | Callable[[Mapping[str, float]], None]
) -> Callable[[Mapping[str, float]], None]:
    """Return a callable that logs negative weight warnings."""

    if callable(warn_once):
        return warn_once
    if warn_once:
        return negative_weights_warn_once()
    return _log_negative_weights


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
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        if expand is not None:
            replacement = expand(item)
            if replacement is not None:
                seen.add(item_id)
                stack.extendleft(replacement)
                continue
        if is_non_string_sequence(item):
            seen.add(item_id)
            stack.extendleft(item)
        else:
            yield item


__all__ = (
    "MAX_MATERIALIZE_DEFAULT",
    "normalize_materialize_limit",
    "is_non_string_sequence",
    "flatten_structure",
    "STRING_TYPES",
    "ensure_collection",
    "normalize_weights",
    "negative_weights_warn_once",
    "normalize_counter",
    "mix_groups",
)

MAX_MATERIALIZE_DEFAULT: int = 1000
"""Default materialization limit used by :func:`ensure_collection`.

This guard prevents accidentally consuming huge or infinite iterables when a
limit is not explicitly provided. Pass ``max_materialize=None`` to disable the
limit.
"""


def normalize_materialize_limit(max_materialize: int | None) -> int | None:
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
    """Return ``it`` as a :class:`Collection`, materializing when needed.

    Checks are executed in the following order:

    1. Existing collections are returned directly. String-like inputs
       (``str``, ``bytes`` and ``bytearray``) are wrapped as a single item
       tuple unless ``treat_strings_as_iterables=True``.
    2. The object must be an :class:`Iterable`; otherwise ``TypeError`` is
       raised.
    3. Remaining iterables are materialized up to ``max_materialize`` items.
       ``None`` disables the limit. ``error_msg`` customizes the
       :class:`ValueError` raised when the iterable yields more items than
       allowed. The input is consumed at most once and no extra items beyond the
       limit are stored in memory.
    """

    # Step 1: early-return for collections and raw strings/bytes
    if isinstance(it, Collection):
        if isinstance(it, STRING_TYPES):
            if not treat_strings_as_iterables:
                return (cast(T, it),)
            # fall through for materialization when treating as iterable
        else:
            return it

    # Step 2: ensure the input is iterable
    if not isinstance(it, Iterable):
        raise TypeError(f"{it!r} is not iterable")

    # Step 3: validate limit and materialize items once
    limit = normalize_materialize_limit(max_materialize)
    if limit is None:
        return tuple(it)
    if limit == 0:
        return ()

    items = tuple(islice(it, limit + 1))
    if len(items) > limit:
        examples = ", ".join(repr(x) for x in items[:3])
        msg = error_msg or (
            f"Iterable produced {len(items)} items, exceeds limit {limit}; first items: [{examples}]"
        )
        raise ValueError(msg)
    return items


def _convert_and_validate_weights(
    dict_like: Mapping[str, Any],
    keys: Iterable[str] | Sequence[str],
    default: float,
    *,
    error_on_conversion: bool,
    error_on_negative: bool,
    warn_once: bool | Callable[[Mapping[str, float]], None],
) -> tuple[dict[str, float], list[str], float]:
    """Return converted weights, deduplicated keys and the accumulated total."""

    keys_list = list(dict.fromkeys(keys))
    default_float = float(default)

    def convert(k: str) -> float:
        ok, val = convert_value(
            dict_like.get(k, default_float),
            float,
            strict=error_on_conversion,
            key=k,
            log_level=logging.WARNING,
        )
        return cast(float, val) if ok else default_float

    weights = {k: convert(k) for k in keys_list}
    negatives = {k: w for k, w in weights.items() if w < 0}
    total = kahan_sum_nd(((w,) for w in weights.values()), dims=1)[0]

    if negatives:
        if error_on_negative:
            raise ValueError(NEGATIVE_WEIGHTS_MSG % negatives)
        warn_negative = _resolve_negative_warn_handler(warn_once)
        warn_negative(negatives)
        for key, weight in negatives.items():
            weights[key] = 0.0
            total -= weight

    return weights, keys_list, total


def normalize_weights(
    dict_like: Mapping[str, Any],
    keys: Iterable[str] | Sequence[str],
    default: float = 0.0,
    *,
    error_on_negative: bool = False,
    warn_once: bool | Callable[[Mapping[str, float]], None] = True,
    error_on_conversion: bool = False,
) -> dict[str, float]:
    """Normalize ``keys`` in mapping ``dict_like`` so their sum is 1.

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

    ``warn_once`` accepts either a boolean or a callable. ``False`` logs all
    negative weights using :func:`logging.Logger.warning`. ``True`` (the
    default) creates a fresh :class:`~tnfr.logging_utils.WarnOnce` instance for
    the call, emitting a single warning containing all negative keys. To reuse
    deduplication state across calls, pass a callable such as
    :func:`negative_weights_warn_once`.
    """
    weights, keys_list, total = _convert_and_validate_weights(
        dict_like,
        keys,
        default,
        error_on_conversion=error_on_conversion,
        error_on_negative=error_on_negative,
        warn_once=warn_once,
    )
    if not keys_list:
        return {}
    if total <= 0:
        uniform = 1.0 / len(keys_list)
        return {k: uniform for k in keys_list}
    return {k: w / total for k, w in weights.items()}


def normalize_counter(
    counts: Mapping[str, float | int],
) -> tuple[dict[str, float], float]:
    """Normalize a ``Counter`` returning proportions and total."""
    total = kahan_sum_nd(((c,) for c in counts.values()), dims=1)[0]
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
            f"{prefix}{label}": kahan_sum_nd(
                ((dist.get(k, 0.0),) for k in keys),
                dims=1,
            )[0]
            for label, keys in groups.items()
        }
    )
    return out
