"""Tests for _is_immutable helper."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tnfr.constants import _is_immutable, _is_immutable_inner
from tnfr.constants import _IMMUTABLE_CACHE
import gc


def test_is_immutable_nested_structures():
    nested = (
        1,
        frozenset({2, (3, 4)}),
        MappingProxyType({"a": (5, 6), "b": frozenset({7})}),
    )
    assert _is_immutable(nested)


def test_is_immutable_detects_mutable():
    data = MappingProxyType({"a": [1, 2]})
    assert not _is_immutable(data)


def test_is_immutable_detects_set_and_bytearray():
    assert not _is_immutable({1, 2})
    assert not _is_immutable(bytearray(b"abc"))


def test_is_immutable_lists_dicts_nested():
    data = (1, [2, {"a": (3, 4)}])
    assert not _is_immutable(data)
    # call twice to exercise cache behaviour
    assert not _is_immutable(data)


def test_is_immutable_inner_handles_mapping_tag():
    frozen = ("mapping", (("a", 1), ("b", 2)))
    assert _is_immutable_inner(frozen)


def test_is_immutable_inner_handles_dict_tag():
    frozen = ("dict", (("a", 1),))
    assert not _is_immutable_inner(frozen)


def test_is_immutable_inner_handles_set_tag():
    frozen = ("set", (1, 2))
    assert not _is_immutable_inner(frozen)


def test_is_immutable_inner_handles_bytearray_tag():
    frozen = ("bytearray", b"abc")
    assert not _is_immutable_inner(frozen)


@dataclass(frozen=True)
class FrozenDC:
    x: int
    y: int


def test_is_immutable_frozen_dataclass():
    assert _is_immutable(FrozenDC(1, 2))


@dataclass
class MutableDC:
    items: list[int]


def test_is_immutable_mutable_dataclass():
    assert not _is_immutable(MutableDC([1, 2]))


def test_is_immutable_detects_cycles():
    lst: list[Any] = []
    lst.append(lst)
    assert not _is_immutable(lst)
    d: dict[str, Any] = {}
    d["self"] = d
    assert not _is_immutable(d)


def test_is_immutable_cache_auto_cleanup():
    class Dummy:
        pass

    obj = Dummy()
    _is_immutable(obj)
    obj_id = id(obj)

    # ensure our object is present in cache
    assert obj in _IMMUTABLE_CACHE

    del obj
    gc.collect()

    # the weak cache should have removed the entry
    assert obj_id not in {id(k) for k in _IMMUTABLE_CACHE.keys()}
