"""Pruebas de defaults integrity."""

from collections import ChainMap
import pytest

from tnfr.constants import DEFAULTS, DEFAULT_SECTIONS


def test_defaults_is_union_of_parts():
    expected = dict(ChainMap(*reversed(tuple(DEFAULT_SECTIONS.values()))))
    assert DEFAULTS == expected


def test_defaults_contains_submodule_parts():
    for part in DEFAULT_SECTIONS.values():
        for k, v in part.items():
            assert DEFAULTS[k] == v


def test_defaults_is_immutable():
    with pytest.raises(TypeError):
        DEFAULTS["foo"] = 1  # type: ignore[misc]
