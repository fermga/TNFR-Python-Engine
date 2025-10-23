"""Validate that Spanish node aliases and helpers have been removed."""

from __future__ import annotations

import importlib

import pytest

DEPRECATED_NODE_CLASS = "NodeNXLegacy"
DEPRECATED_NODE_PROTOCOL = "NodeProtocolLegacy"
DEPRECATED_UTILS_HELPER = "get_nodenx_legacy"


def test_spanish_node_aliases_removed() -> None:
    module = importlib.import_module("tnfr.node")
    assert DEPRECATED_NODE_CLASS not in getattr(module, "__all__", ())
    assert DEPRECATED_NODE_PROTOCOL not in getattr(module, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(module, DEPRECATED_NODE_CLASS)
    with pytest.raises(AttributeError):
        getattr(module, DEPRECATED_NODE_PROTOCOL)


def test_utils_spanish_helper_removed() -> None:
    utils = importlib.import_module("tnfr.utils")
    assert DEPRECATED_UTILS_HELPER not in getattr(utils, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(utils, DEPRECATED_UTILS_HELPER)
    assert utils.get_nodenx() is not None
