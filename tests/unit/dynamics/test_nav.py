"""Unit tests for navigation helpers managing neighbor selection and glyph choices."""

import pytest

from tnfr.constants import inject_defaults
from tnfr.operators import apply_glyph


def test_nav_converges_to_vf_without_jitter(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["ΔNFR"] = 0.2
    nd["νf"] = 1.0
    G.graph["GLYPH_FACTORS"]["NAV_jitter"] = 0.0
    apply_glyph(G, 0, "NAV")
    eta = G.graph["GLYPH_FACTORS"]["NAV_eta"]
    expected = (1 - eta) * 0.2 + eta * 1.0
    assert nd["ΔNFR"] == pytest.approx(expected)


def test_nav_strict_sets_dnfr_to_vf(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    nd = G.nodes[0]
    nd["ΔNFR"] = -0.5
    nd["νf"] = 0.8
    G.graph["GLYPH_FACTORS"]["NAV_jitter"] = 0.0
    G.graph["NAV_STRICT"] = True
    apply_glyph(G, 0, "NAV")
    assert nd["ΔNFR"] == pytest.approx(0.8)
