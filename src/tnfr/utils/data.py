"""Utilities for manipulating collections and scalar values within TNFR."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from numbers import Real
from itertools import chain, islice
from typing import (
    Any,
    Callable,
    Iterable as TypingIterable,
    Iterator,
    Literal,
    TypeVar,
    cast,
    overload,
)

from .numeric import kahan_sum_nd
from .init import get_logger
from .init import warn_once as _warn_once_factory

T = TypeVar("T")

_collections_logger = get_logger("tnfr.utils.data.collections")
_value_logger = get_logger("tnfr.utils.data")

STRING_TYPES = (str, bytes, bytearray)

NEGATIVE_WEIGHTS_MSG = "Negative weights detected: %s"

_MAX_NEGATIVE_WARN_ONCE = 1024

__all__ = (
    "convert_value",
    "normalize_optional_int",
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


def convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    key: str | None = None,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Attempt to convert a value and report failures."""

    try:
        converted = conv(value)
    except (ValueError, TypeError) as exc:
        if strict:
            raise
        level = log_level if log_level is not None else logging.DEBUG
        if key is not None:
            _value_logger.log(level, "Could not convert value for %r: %s", key, exc)
        else:
            _value_logger.log(level, "Could not convert value: %s", exc)
        return False, None
    if isinstance(converted, float) and not math.isfinite(converted):
        if strict:
            target = f"{key!r}" if key is not None else "value"
            raise ValueError(f"Non-finite value {converted!r} for {target}")
        level = log_level if log_level is not None else logging.DEBUG
        if key is not None:
            _value_logger.log(level, "Non-finite value for %r: %s", key, converted)
        else:
            _value_logger.log(level, "Non-finite value: %s", converted)
        return False, None
    return True, converted


_DEFAULT_SENTINELS = frozenset({"auto", "none", "null"})


def normalize_optional_int(
    value: Any,
    *,
    sentinels: Collection[str] | None = _DEFAULT_SENTINELS,
    allow_non_positive: bool = True,
    strict: bool = False,
    error_message: str | None = None,
) -> int | None:
    """Normalise optional integers shared by CLI and runtime helpers.

    Parameters
    ----------
    value:
        Arbitrary object obtained from configuration, CLI options or graph
        metadata.
    sentinels:
        Collection of case-insensitive strings that should be interpreted as
        ``None``. When ``None`` or empty, no sentinel mapping is applied.
    allow_non_positive:
        When ``False`` values ``<= 0`` are rejected and converted to ``None``.
    strict:
        When ``True`` invalid inputs raise :class:`ValueError` instead of
        returning ``None``.
    error_message:
        Optional message used when ``strict`` mode raises due to invalid input
        or disallowed non-positive values.
    """

    if value is None:
        return None

    if isinstance(value, int):
        result = value
    elif isinstance(value, Real):
        result = int(value)
    else:
        text = str(value).strip()
        if not text:
            if strict:
                raise ValueError(
                    error_message or "Empty value is not allowed for configuration options."
                )
            return None
        sentinel_set: set[str] | None = None
        if sentinels:
            sentinel_set = {s.lower() for s in sentinels}
            lowered = text.lower()
            if lowered in sentinel_set:
                return None
        try:
            result = int(text)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(error_message or f"Invalid integer value: {value!r}") from exc
            return None

    if not allow_non_positive and result <= 0:
        if strict:
            raise ValueError(
                error_message or "Non-positive values are not permitted for this option."
            )
        return None

    return result


def negative_weights_warn_once(
    *, maxsize: int = _MAX_NEGATIVE_WARN_ONCE
) -> Callable[[Mapping[str, float]], None]:
    """Return a ``WarnOnce`` callable for negative weight warnings."""

    return _warn_once_factory(_collections_logger, NEGATIVE_WEIGHTS_MSG, maxsize=maxsize)


def _log_negative_weights(negatives: Mapping[str, float]) -> None:
    """Log negative weight warnings without deduplicating keys."""

    _collections_logger.warning(NEGATIVE_WEIGHTS_MSG, negatives)


def _resolve_negative_warn_handler(
    warn_once: bool | Callable[[Mapping[str, float]], None],
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
    """Yield leaf items from ``obj`` following breadth-first semantics."""

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


MAX_MATERIALIZE_DEFAULT: int = 1000
"""Default materialization limit used by :func:`ensure_collection`."""


def normalize_materialize_limit(max_materialize: int | None) -> int | None:
    """Normalize and validate ``max_materialize`` returning a usable limit."""

    if max_materialize is None:
        return None
    limit = int(max_materialize)
    if limit < 0:
        raise ValueError("'max_materialize' must be non-negative")
    return limit


@overload
def ensure_collection(
    it: Iterable[T],
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
    error_msg: str | None = None,
    return_view: Literal[False] = False,
) -> Collection[T]: ...


@overload
def ensure_collection(
    it: Iterable[T],
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
    error_msg: str | None = None,
    return_view: Literal[True],
) -> tuple[Collection[T], TypingIterable[T]]: ...


def ensure_collection(
    it: Iterable[T],
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
    error_msg: str | None = None,
    return_view: bool = False,
) -> Collection[T] | tuple[Collection[T], TypingIterable[T]]:
    """Return ``it`` as a :class:`Collection`, materializing when needed.

    When ``return_view`` is ``True`` the function returns a tuple containing the
    materialised preview and an iterable that can be used to continue streaming
    from the same source after the preview limit. The preview will contain up to
    ``max_materialize`` items (when the limit is enforced); when ``max_materialize``
    is ``None`` the preview is empty and the returned iterable is the original
    stream.
    """

    def _finalize(
        collection: Collection[T],
        view: TypingIterable[T] | None = None,
    ) -> Collection[T] | tuple[Collection[T], TypingIterable[T]]:
        if not return_view:
            return collection
        if view is None:
            return collection, collection
        return collection, view

    if isinstance(it, Collection):
        if isinstance(it, STRING_TYPES):
            wrapped = (cast(T, it),)
            return _finalize(wrapped)
        return _finalize(cast(Collection[T], it), cast(TypingIterable[T], it))

    if isinstance(it, STRING_TYPES):
        wrapped = (cast(T, it),)
        return _finalize(wrapped)

    if not isinstance(it, Iterable):
        raise TypeError(f"{it!r} is not iterable")

    limit = normalize_materialize_limit(max_materialize)

    if return_view:
        if limit is None:
            return (), cast(TypingIterable[T], it)
        if limit == 0:
            return (), ()

        iterator = iter(it)
        preview = tuple(islice(iterator, limit + 1))
        if len(preview) > limit:
            examples = ", ".join(repr(x) for x in preview[:3])
            msg = error_msg or (
                f"Iterable produced {len(preview)} items, exceeds limit {limit}; first items: [{examples}]"
            )
            raise ValueError(msg)
        if not preview:
            return (), iterator
        return preview, chain(preview, iterator)

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
    """Normalize ``keys`` in mapping ``dict_like`` so their sum is 1."""

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
