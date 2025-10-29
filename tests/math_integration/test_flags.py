"""Integration tests for math feature flags."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.config.feature_flags import context_flags, get_flags
from tnfr.node import NodeNX


def test_enable_math_validation_precedence() -> None:
    base_flags = get_flags()

    graph = nx.Graph()
    graph.add_node("default")
    node_default = NodeNX(graph, "default")
    assert node_default.enable_math_validation is base_flags.enable_math_validation

    with context_flags(enable_math_validation=True):
        graph.add_node("global")
        node_global = NodeNX(graph, "global")
        assert node_global.enable_math_validation is True

        graph.add_node("explicit")
        node_explicit = NodeNX(graph, "explicit", enable_math_validation=False)
        assert node_explicit.enable_math_validation is False


def test_context_flags_restore_after_exception() -> None:
    original = get_flags()
    toggled_value = not original.enable_math_validation

    with pytest.raises(RuntimeError):
        with context_flags(enable_math_validation=toggled_value):
            assert get_flags().enable_math_validation is toggled_value
            raise RuntimeError("boom")

    assert get_flags() == original


def test_context_flags_log_performance_restore() -> None:
    original = get_flags()

    with context_flags(log_performance=True):
        assert get_flags().log_performance is True

    assert get_flags() == original
