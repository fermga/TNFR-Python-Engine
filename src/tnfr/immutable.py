"""Utilities for freezing objects and checking immutability."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import lru_cache, singledispatch, wraps
from typing import Any, Callable
from collections.abc import Mapping
from types import MappingProxyType
import threading
import weakref

# Types considered immutable without further inspection
IMMUTABLE_SIMPLE = frozenset(
    {int, float, complex, str, bool, bytes, type(None)}
)


@contextmanager
def _cycle_guard(value: Any, seen: set[int] | None = None):
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


def _check_cycle(func: Callable[[Any, set[int] | None], Any]):
    """Decorator applying :func:`_cycle_guard` to ``func``."""

    @wraps(func)
    def wrapper(value: Any, seen: set[int] | None = None):
        with _cycle_guard(value, seen) as seen:
            return func(value, seen)

    return wrapper


def _freeze_dataclass(value: Any, seen: set[int]):
    params = getattr(type(value), "__dataclass_params__", None)
    frozen = bool(params and params.frozen)
    data = asdict(value)
    tag = "mapping" if frozen else "dict"
    return (tag, tuple((k, _freeze(v, seen)) for k, v in data.items()))


@singledispatch
@_check_cycle
def _freeze(value: Any, seen: set[int] | None = None):
    """Recursively convert ``value`` into an immutable representation."""
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_dataclass(value, seen)
    if type(value) in IMMUTABLE_SIMPLE:
        return value
    raise TypeError


@_freeze.register(tuple)
@_check_cycle
def _freeze_tuple(value: tuple, seen: set[int] | None = None):
    return tuple(_freeze(v, seen) for v in value)


def _freeze_iterable(container: Any, tag: str, seen: set[int] | None) -> tuple[str, tuple]:
    return (tag, tuple(_freeze(v, seen) for v in container))


def _register_iterable(cls: type, tag: str) -> None:
    @_freeze.register(cls)
    @_check_cycle
    def _freeze_iterable_wrapper(value: Any, seen: set[int] | None = None):
        return _freeze_iterable(value, tag, seen)


for _cls, _tag in (
    (list, "list"),
    (set, "set"),
    (frozenset, "frozenset"),
    (bytearray, "bytearray"),
):
    _register_iterable(_cls, _tag)

@_freeze.register(Mapping)
@_check_cycle
def _freeze_mapping(value: Mapping, seen: set[int] | None = None):
    tag = "dict" if hasattr(value, "__setitem__") else "mapping"
    return (tag, tuple((k, _freeze(v, seen)) for k, v in value.items()))


def _all_immutable(iterable) -> bool:
    return all(_is_immutable_inner(v) for v in iterable)


# Dispatch table kept immutable to avoid accidental mutation.
_IMMUTABLE_TAG_DISPATCH: Mapping[str, Callable[[tuple], bool]] = MappingProxyType(
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
    return type(value) in IMMUTABLE_SIMPLE


@_is_immutable_inner.register(tuple)
def _is_immutable_inner_tuple(value: tuple) -> bool:
    if value and isinstance(value[0], str):
        handler = _IMMUTABLE_TAG_DISPATCH.get(value[0])
        if handler is not None:
            return handler(value)
    return _all_immutable(value)


@_is_immutable_inner.register(frozenset)
def _is_immutable_inner_frozenset(value: frozenset) -> bool:
    return _all_immutable(value)


_IMMUTABLE_CACHE: weakref.WeakKeyDictionary[Any, bool] = (
    weakref.WeakKeyDictionary()
)
_IMMUTABLE_CACHE_LOCK = threading.Lock()


def _is_immutable(value: Any) -> bool:
    """Check recursively if ``value`` is immutable with caching."""
    with _IMMUTABLE_CACHE_LOCK:
        try:
            return _IMMUTABLE_CACHE[value]
        except (KeyError, TypeError):
            pass

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
            pass

    return result


__all__ = (
    "_freeze",
    "_is_immutable",
    "_is_immutable_inner",
    "_IMMUTABLE_CACHE",
)
