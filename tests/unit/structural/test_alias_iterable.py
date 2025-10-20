"""Unit tests for alias helpers handling generic iterables."""



import pytest
from tnfr.alias import AliasAccessor


def test_get_attr_accepts_hashable_iterable():
    d = {"b": "1"}
    acc = AliasAccessor(int)
    assert acc.get(d, ("a", "b")) == 1


def test_set_attr_accepts_hashable_iterable():
    d = {}
    acc = AliasAccessor(int)
    acc.set(d, ("x", "y"), "5")
    assert d["x"] == 5


def test_rejects_str():
    with pytest.raises(TypeError):
        AliasAccessor(int).get({}, "x")
    with pytest.raises(TypeError):
        AliasAccessor(int).set({}, "x", 1)


def test_accepts_list_iterable():
    d = {"b": "1"}
    acc = AliasAccessor(int)
    assert acc.get(d, ["a", "b"]) == 1
    d2 = {}
    acc.set(d2, ["x", "y"], "5")
    assert d2["x"] == 5


def test_accepts_generator_iterable():
    d = {"b": "1"}
    aliases = (a for a in ("a", "b"))
    acc = AliasAccessor(int)
    assert acc.get(d, aliases) == 1
    d2 = {}
    acc.set(d2, (a for a in ("x", "y")), "5")
    assert d2["x"] == 5
