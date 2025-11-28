"""Utilities for freezing objects and checking immutability.

Handlers registered via :func:`functools.singledispatch` live in this module
and are triggered indirectly by the dispatcher when matching types are
encountered.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import lru_cache, partial, singledispatch, wraps
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, cast

from ._compat import TypeAlias

# Types considered immutable without further inspection
IMMUTABLE_SIMPLE = frozenset({int, float, complex, str, bool, bytes, type(None)})

FrozenPrimitive: TypeAlias = int | float | complex | str | bool | bytes | None
"""Primitive immutable values handled directly by :func:`_freeze`."""

FrozenCollectionItems: TypeAlias = tuple["FrozenSnapshot", ...]
"""Frozen representation for generic iterables."""

FrozenMappingItems: TypeAlias = tuple[tuple[Any, "FrozenSnapshot"], ...]
"""Frozen representation for mapping ``items()`` snapshots."""

FrozenTaggedCollection: TypeAlias = tuple[str, FrozenCollectionItems]
"""Tagged iterable snapshot identifying the original container type."""

FrozenTaggedMapping: TypeAlias = tuple[str, FrozenMappingItems]
"""Tagged mapping snapshot identifying the original mapping flavour."""

FrozenSnapshot: TypeAlias = (
    FrozenPrimitive | FrozenCollectionItems | FrozenTaggedCollection | FrozenTaggedMapping
)
"""Union describing the immutable snapshot returned by :func:`_freeze`."""


@contextmanager
def _cycle_guard(value: Any, seen: set[int] | None = None) -> Iterator[set[int]]:
    """Context manager that detects reference cycles during freezing."""
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        raise ValueError("cycle detected")
    seen.add(obj_id)
    try:
        yield seen
    finally:
        seen.remove(obj_id)


def _check_cycle(
    func: Callable[[Any, set[int] | None], FrozenSnapshot],
) -> Callable[[Any, set[int] | None], FrozenSnapshot]:
    """Apply :func:`_cycle_guard` to ``func``."""

    @wraps(func)
    def wrapper(value: Any, seen: set[int] | None = None) -> FrozenSnapshot:
        with _cycle_guard(value, seen) as guard_seen:
            return func(value, guard_seen)

    return wrapper


def _freeze_dataclass(value: Any, seen: set[int]) -> FrozenTaggedMapping:
    params = getattr(type(value), "__dataclass_params__", None)
    frozen = bool(params and params.frozen)
    data = asdict(value)
    tag = "mapping" if frozen else "dict"
    return (tag, tuple((k, _freeze(v, seen)) for k, v in data.items()))


@singledispatch
@_check_cycle
def _freeze(value: Any, seen: set[int] | None = None) -> FrozenSnapshot:
    """Recursively convert ``value`` into an immutable representation."""
    if is_dataclass(value) and not isinstance(value, type):
        assert seen is not None
        return _freeze_dataclass(value, seen)
    if type(value) in IMMUTABLE_SIMPLE:
        return value
    raise TypeError


@_freeze.register(tuple)
@_check_cycle
def _freeze_tuple(
    value: tuple[Any, ...], seen: set[int] | None = None
) -> FrozenCollectionItems:  # noqa: F401
    assert seen is not None
    return tuple(_freeze(v, seen) for v in value)


def _freeze_iterable(container: Iterable[Any], tag: str, seen: set[int]) -> FrozenTaggedCollection:
    return (tag, tuple(_freeze(v, seen) for v in container))


def _freeze_iterable_with_tag(
    value: Iterable[Any], seen: set[int] | None = None, *, tag: str
) -> FrozenTaggedCollection:
    assert seen is not None
    return _freeze_iterable(value, tag, seen)


def _register_iterable(cls: type, tag: str) -> None:
    handler = _check_cycle(partial(_freeze_iterable_with_tag, tag=tag))
    _freeze.register(cls)(cast(Callable[[Any, set[int] | None], FrozenSnapshot], handler))


for _cls, _tag in (
    (list, "list"),
    (set, "set"),
    (frozenset, "frozenset"),
    (bytearray, "bytearray"),
):
    _register_iterable(_cls, _tag)


@_freeze.register(Mapping)
@_check_cycle
def _freeze_mapping(
    value: Mapping[Any, Any], seen: set[int] | None = None
) -> FrozenTaggedMapping:  # noqa: F401
    assert seen is not None
    tag = "dict" if hasattr(value, "__setitem__") else "mapping"
    return (tag, tuple((k, _freeze(v, seen)) for k, v in value.items()))


def _all_immutable(iterable: Iterable[Any]) -> bool:
    return all(_is_immutable_inner(v) for v in iterable)


# Dispatch table kept immutable to avoid accidental mutation.
ImmutableTagHandler: TypeAlias = Callable[[tuple[Any, ...]], bool]

_IMMUTABLE_TAG_DISPATCH: Mapping[str, ImmutableTagHandler] = MappingProxyType(
    {
        "mapping": lambda v: _all_immutable(v[1]),
        "frozenset": lambda v: _all_immutable(v[1]),
        "list": lambda v: False,
        "set": lambda v: False,
        "bytearray": lambda v: False,
        "dict": lambda v: False,
    }
)


@lru_cache(maxsize=1024)
@singledispatch
def _is_immutable_inner(value: Any) -> bool:
    """Return ``True`` when ``value`` belongs to the canonical immutable set."""

    return type(value) in IMMUTABLE_SIMPLE


@_is_immutable_inner.register(tuple)
def _is_immutable_inner_tuple(value: tuple[Any, ...]) -> bool:  # noqa: F401
    if value and isinstance(value[0], str):
        handler = _IMMUTABLE_TAG_DISPATCH.get(value[0])
        if handler is not None:
            return handler(value)
    return _all_immutable(value)


@_is_immutable_inner.register(frozenset)
def _is_immutable_inner_frozenset(value: frozenset[Any]) -> bool:  # noqa: F401
    return _all_immutable(value)


_IMMUTABLE_CACHE: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
_IMMUTABLE_CACHE_LOCK = threading.Lock()


def _is_immutable(value: Any) -> bool:
    """Check recursively if ``value`` is immutable with caching."""
    with _IMMUTABLE_CACHE_LOCK:
        try:
            return _IMMUTABLE_CACHE[value]
        except (KeyError, TypeError):
            pass  # Not in cache or value is unhashable

    try:
        frozen = _freeze(value)
    except (TypeError, ValueError):
        result = False
    else:
        result = _is_immutable_inner(frozen)

    with _IMMUTABLE_CACHE_LOCK:
        try:
            _IMMUTABLE_CACHE[value] = result
        except TypeError:
            pass  # Value is unhashable, cannot cache

    return result


__all__ = (
    "_freeze",
    "_is_immutable",
    "_is_immutable_inner",
    "_IMMUTABLE_CACHE",
)
