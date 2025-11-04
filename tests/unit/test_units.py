"""Unit tests for structural unit conversions."""

import pytest

from tnfr.constants import inject_defaults, merge_overrides
from tnfr.units import hz_str_to_hz, hz_to_hz_str

@pytest.fixture
def graph_with_bridge(graph_canon):
    """Return a graph with a custom Hz_str bridge factor."""

    G = graph_canon()
    merge_overrides(G, HZ_STR_BRIDGE=1.75)
    return G

def test_hz_str_to_hz_round_trip(graph_with_bridge):
    """Conversions must remain invertible under custom bridge overrides."""

    G = graph_with_bridge
    values = [0.0, 0.5, 1.0, 2.4, -3.5]
    for value in values:
        hz = hz_str_to_hz(value, G)
        back = hz_to_hz_str(hz, G)
        assert back == pytest.approx(float(value))

def test_hz_conversion_respects_injected_defaults(graph_canon):
    """Injecting defaults should provide canonical conversion behaviour."""

    G = graph_canon()
    inject_defaults(G, override=True)
    expected = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-2.0, -2.0),
    ]
    for hz_str, hz_expected in expected:
        assert hz_str_to_hz(hz_str, G) == pytest.approx(hz_expected)
        assert hz_to_hz_str(hz_expected, G) == pytest.approx(hz_str)
