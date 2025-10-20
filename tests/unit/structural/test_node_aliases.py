"""Ensure legacy node aliases remain available with deprecation warnings."""

from __future__ import annotations

import importlib
import warnings

import pytest


@pytest.mark.parametrize(
    ("alias", "new_name"),
    [
        ("NodoNX", "NodeNX"),
        ("NodoProtocol", "NodeProtocol"),
    ],
)
def test_node_alias_warns(alias: str, new_name: str) -> None:
    module = importlib.import_module("tnfr.node")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        value = getattr(module, alias)
    assert value is getattr(module, new_name)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_get_nodonx_warns_and_matches() -> None:
    utils = importlib.import_module("tnfr.utils")
    with pytest.warns(DeprecationWarning):
        alias_value = utils.get_nodonx()
    canonical_value = utils.get_nodenx()
    assert alias_value is canonical_value
    assert canonical_value is not None
