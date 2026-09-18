"""Numerical admission of the structural/physical unit bridge."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.units import get_hz_bridge, hz_str_to_hz, hz_to_hz_str


def graph_with_bridge(value):
    graph = nx.Graph()
    graph.graph["HZ_STR_BRIDGE"] = value
    return graph


@pytest.mark.parametrize("value", [True, False, np.bool_(True), "1", None])
def test_bridge_refuses_nonreal_or_boolean_inputs(value):
    with pytest.raises(TypeError):
        get_hz_bridge(graph_with_bridge(value))


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf,
                                   10 ** 1000, Fraction(1, 10 ** 1000)])
def test_bridge_refuses_invalid_or_unrepresentable_factor(value):
    with pytest.raises(ValueError):
        get_hz_bridge(graph_with_bridge(value))


@pytest.mark.parametrize("function", [hz_str_to_hz, hz_to_hz_str])
@pytest.mark.parametrize("value", [True, np.bool_(False), np.nan, np.inf,
                                   -np.inf, Fraction(1, 10 ** 1000)])
def test_conversion_refuses_invalid_values(function, value):
    with pytest.raises((ValueError, TypeError)):
        function(value, graph_with_bridge(1))


def test_conversion_preserves_finite_sign_and_independent_graph_factor():
    graph = graph_with_bridge(2)
    assert hz_str_to_hz(3, graph) == 6
    assert hz_to_hz_str(6, graph) == 3
    assert hz_str_to_hz(-3, graph) == -6  # Conversion is not capacity admission.
    assert hz_to_hz_str(0, graph) == 0
    assert get_hz_bridge(nx.Graph()) == 1


@pytest.mark.parametrize("function,value,factor", [
    (hz_str_to_hz, 1e308, 1e308), (hz_str_to_hz, 5e-324, 0.5),
    (hz_to_hz_str, 1e308, 5e-324), (hz_to_hz_str, 5e-324, 2),
])
def test_conversion_does_not_silently_overflow_or_erase_activity(function, value, factor):
    with pytest.raises(ValueError, match="overflowed or erased"):
        function(value, graph_with_bridge(factor))
