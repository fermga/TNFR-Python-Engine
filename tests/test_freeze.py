"""Tests for _freeze singledispatch."""

from dataclasses import dataclass
from types import MappingProxyType

import pytest

from tnfr.immutable import _freeze


def test_freeze_tuple():
    assert _freeze((1, 2)) == (1, 2)


def test_freeze_list():
    assert _freeze([1, 2]) == ("list", (1, 2))


def test_freeze_set():
    result = _freeze({1, 2})
    assert result[0] == "set"
    assert set(result[1]) == {1, 2}


def test_freeze_frozenset():
    result = _freeze(frozenset({1, 2}))
    assert result[0] == "frozenset"
    assert set(result[1]) == {1, 2}


def test_freeze_bytearray():
    assert _freeze(bytearray(b"ab")) == ("bytearray", (97, 98))


def test_freeze_mapping_and_dict():
    assert _freeze({"a": 1}) == ("dict", (("a", 1),))
    assert _freeze(MappingProxyType({"a": 1})) == ("mapping", (("a", 1),))


@dataclass(frozen=True)
class FrozenDC:
    x: int
    y: int


@dataclass
class MutableDC:
    x: int
    items: list[int]


def test_freeze_dataclasses():
    assert _freeze(FrozenDC(1, 2)) == ("mapping", (("x", 1), ("y", 2)))
    assert _freeze(MutableDC(1, [2])) == (
        "dict",
        (("x", 1), ("items", ("list", (2,)))),
    )
