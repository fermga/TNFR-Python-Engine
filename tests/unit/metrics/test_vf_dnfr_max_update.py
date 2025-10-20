import pytest

from tnfr.alias import (
    AbsMaxResult,
    set_attr,
    set_attr_and_cache,
    set_attr_with_max,
    set_scalar,
)
from tnfr.constants import get_aliases
from tnfr.metrics.common import _get_vf_dnfr_max

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_get_vf_dnfr_max_updates_graph_on_none(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    set_attr(G.nodes[1], ALIAS_VF, 0.5)
    set_attr(G.nodes[2], ALIAS_VF, -1.5)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[2], ALIAS_DNFR, -0.4)
    G.graph["_vfmax"] = None
    G.graph["_dnfrmax"] = None

    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    assert vfmax == pytest.approx(1.5)
    assert dnfrmax == pytest.approx(0.4)
    assert G.graph["_vfmax"] == pytest.approx(1.5)
    assert G.graph["_dnfrmax"] == pytest.approx(0.4)


def test_set_attr_and_cache_returns_none_without_cache(graph_canon):
    G = graph_canon()
    G.add_node(1)

    result = set_attr_and_cache(G, 1, ALIAS_VF, 0.8)

    assert result is None


def test_set_attr_with_max_returns_abs_result(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])

    first = set_attr_with_max(G, 1, ALIAS_VF, -0.5, cache="_vfmax")
    second = set_attr_with_max(G, 2, ALIAS_VF, 2.0, cache="_vfmax")

    assert isinstance(first, AbsMaxResult)
    assert isinstance(second, AbsMaxResult)
    assert first.max_value == pytest.approx(0.5)
    assert first.node == 1
    assert second.max_value == pytest.approx(2.0)
    assert second.node == 2
    assert G.graph["_vfmax"] == pytest.approx(2.0)
    assert G.graph["_vfmax_node"] == 2


def test_set_scalar_handles_cache_response(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])

    none_result = set_scalar(G, 1, ALIAS_DNFR, 0.1)
    cached = set_scalar(G, 2, ALIAS_DNFR, -0.6, cache="_dnfrmax")

    assert none_result is None
    assert isinstance(cached, AbsMaxResult)
    assert cached.max_value == pytest.approx(0.6)
    assert cached.node == 2
    assert G.graph["_dnfrmax"] == pytest.approx(0.6)
    assert G.graph["_dnfrmax_node"] == 2
