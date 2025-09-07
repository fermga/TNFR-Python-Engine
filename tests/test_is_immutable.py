"""Tests for _is_immutable helper."""

from types import MappingProxyType

from tnfr.constants import _is_immutable, _prune_cache


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


def test_prune_cache_allows_gc():
    class Dummy:
        pass

    d = Dummy()
    assert not _is_immutable(d)
    _prune_cache()
    assert not _is_immutable(d)
