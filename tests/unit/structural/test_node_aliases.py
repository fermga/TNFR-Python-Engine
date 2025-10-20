"""Validate that Spanish node aliases and helpers have been removed."""

from __future__ import annotations

import importlib

import pytest


def test_spanish_node_aliases_removed() -> None:
    module = importlib.import_module("tnfr.node")
    assert "NodoNX" not in getattr(module, "__all__", ())
    assert "NodoProtocol" not in getattr(module, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(module, "NodoNX")
    with pytest.raises(AttributeError):
        getattr(module, "NodoProtocol")


def test_utils_spanish_helper_removed() -> None:
    utils = importlib.import_module("tnfr.utils")
    assert "get_nodonx" not in getattr(utils, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(utils, "get_nodonx")
    assert utils.get_nodenx() is not None
