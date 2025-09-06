"""Pruebas de helpers de alias con secuencias genéricas."""

import pytest
from tnfr.alias import alias_get, alias_set


def test_alias_get_accepts_sequence():
    d = {"b": "1"}
    assert alias_get(d, ["a", "b"], int) == 1


def test_alias_set_accepts_sequence():
    d = {}
    alias_set(d, ["x", "y"], int, "5")
    assert d["x"] == 5


def test_alias_rejects_str():
    with pytest.raises(TypeError):
        alias_get({}, "x", int)
    with pytest.raises(TypeError):
        alias_set({}, "x", int, 1)
