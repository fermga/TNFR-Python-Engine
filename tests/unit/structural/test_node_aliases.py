"""Validate that Spanish node aliases and helpers have been removed."""

from __future__ import annotations

import importlib

import pytest

from tests.legacy_tokens import (
    LEGACY_NODE_CLASS,
    LEGACY_NODE_PROTOCOL,
    LEGACY_UTILS_HELPER,
)


def test_spanish_node_aliases_removed() -> None:
    module = importlib.import_module("tnfr.node")
    assert LEGACY_NODE_CLASS not in getattr(module, "__all__", ())
    assert LEGACY_NODE_PROTOCOL not in getattr(module, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(module, LEGACY_NODE_CLASS)
    with pytest.raises(AttributeError):
        getattr(module, LEGACY_NODE_PROTOCOL)


def test_utils_spanish_helper_removed() -> None:
    utils = importlib.import_module("tnfr.utils")
    assert LEGACY_UTILS_HELPER not in getattr(utils, "__all__", ())
    with pytest.raises(AttributeError):
        getattr(utils, LEGACY_UTILS_HELPER)
    assert utils.get_nodenx() is not None
