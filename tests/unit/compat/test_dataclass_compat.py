"""Test Python 3.9 compatibility for dataclass wrapper."""

import sys
from typing import Hashable, Optional

import pytest

from tnfr.compat.dataclass import dataclass


def test_dataclass_wrapper_creates_class():
    """Test that dataclass wrapper successfully creates a class."""

    @dataclass(slots=True)
    class Point:
        x: float
        y: float

    p = Point(1.0, 2.0)
    assert p.x == 1.0
    assert p.y == 2.0


def test_dataclass_with_keyword_args():
    """Test dataclass with keyword arguments."""

    @dataclass(slots=True)
    class Node:
        value: float
        label: Optional[Hashable] = None

    n1 = Node(value=3.14, label="test")
    assert n1.value == 3.14
    assert n1.label == "test"

    n2 = Node(2.71)
    assert n2.value == 2.71
    assert n2.label is None


def test_dataclass_frozen():
    """Test frozen dataclass."""

    @dataclass(frozen=True, slots=True)
    class Immutable:
        x: int

    obj = Immutable(42)
    assert obj.x == 42

    with pytest.raises(AttributeError):
        obj.x = 100  # Should fail for frozen dataclass


def test_dataclass_without_parentheses():
    """Test decorator without parentheses."""

    @dataclass
    class Simple:
        value: str

    s = Simple("hello")
    assert s.value == "hello"


def test_slots_functionality_on_py310_plus():
    """On Python 3.10+, verify slots actually work."""

    @dataclass(slots=True)
    class SlottedClass:
        a: int
        b: str

    obj = SlottedClass(1, "test")
    assert obj.a == 1
    assert obj.b == "test"

    # On Python 3.10+, slots should prevent adding arbitrary attributes
    if sys.version_info >= (3, 10):
        with pytest.raises(AttributeError):
            obj.c = 3  # Should fail due to __slots__


def test_py39_compatibility():
    """Verify the wrapper works on Python 3.9 (doesn't crash)."""

    # This test primarily ensures no TypeError on Python 3.9
    # when slots=True is passed but not supported
    @dataclass(slots=True)
    class Compat:
        x: float

    c = Compat(1.5)
    assert c.x == 1.5

    # Should work regardless of Python version
    assert hasattr(c, "x")


def test_all_dataclass_params():
    """Test that all dataclass parameters are handled."""

    @dataclass(
        init=True,
        repr=True,
        eq=True,
        order=False,
        unsafe_hash=False,
        frozen=False,
        slots=True,
    )
    class Full:
        value: int

    obj = Full(100)
    assert obj.value == 100
    assert repr(obj)  # Should have repr
    assert obj == Full(100)  # Should have eq


def test_dataclass_with_defaults():
    """Test dataclass with default values."""

    @dataclass(slots=True)
    class WithDefaults:
        required: str
        optional: int = 42

    obj1 = WithDefaults("test")
    assert obj1.required == "test"
    assert obj1.optional == 42

    obj2 = WithDefaults("test", 100)
    assert obj2.required == "test"
    assert obj2.optional == 100


def test_multiple_dataclasses():
    """Test creating multiple dataclasses with the wrapper."""

    @dataclass(slots=True)
    class First:
        a: int

    @dataclass(slots=True)
    class Second:
        b: str

    f = First(1)
    s = Second("test")

    assert f.a == 1
    assert s.b == "test"
    assert type(f) != type(s)
