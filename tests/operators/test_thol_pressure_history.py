"""THOL graph pressure must use active history rather than cached curvature."""

from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI
from tnfr.node import NodeNX
from tnfr.operators import _op_THOL, apply_glyph, apply_glyph_obj
from tnfr.operators.self_organization import SelfOrganization


def _graph():
    graph = nx.Graph(GLYPH_FACTORS={"THOL_accel": 0.25})
    graph.add_node(
        0, EPI=9.0, nu_f=1.0, theta=0.0, delta_nfr=0.5,
        epi_time_history=[(0.0, 0.0), (1.0, 1.0), (3.0, 9.0)],
        epi_history=[0.0, 100.0, 0.0],
        glyph_history=["IL", "OZ"],
    )
    graph.nodes[0][ALIAS_D2EPI[0]] = -77.0
    return graph


@pytest.mark.parametrize("route", ["graph", "node", "primitive"])
def test_graph_routes_reconstruct_physical_acceleration(route):
    graph = _graph()
    before = deepcopy(dict(graph.nodes[0]))
    if route == "graph":
        apply_glyph(graph, 0, "THOL")
    elif route == "node":
        apply_glyph_obj(NodeNX(graph, 0), "THOL")
    else:
        _op_THOL(NodeNX(graph, 0), {"THOL_accel": 0.25})
    assert graph.nodes[0]["delta_nfr"] == 1.0
    assert graph.nodes[0][ALIAS_D2EPI[0]] == 2.0
    for key in ("EPI", "nu_f", "theta", "epi_time_history", "epi_history"):
        assert graph.nodes[0][key] == before[key]
    assert tuple(graph) == (0,)
    assert "sub_nodes" not in graph.nodes[0]


@pytest.mark.parametrize("history", [[], [(3.0, 9.0)], [(1.0, 1.0), (3.0, 9.0)]])
def test_short_physical_history_never_reuses_cached_or_legacy_acceleration(history):
    graph = _graph()
    graph.nodes[0]["epi_time_history"] = history
    apply_glyph(graph, 0, "THOL")
    assert graph.nodes[0]["delta_nfr"] == 0.5
    assert graph.nodes[0][ALIAS_D2EPI[0]] == 0.0


@pytest.mark.parametrize("bad_history", [
    [(0.0, 0.0), (0.0, 1.0), (3.0, 9.0)],
    [(0.0, 0.0), (1.0, 1.0), (3.0, 8.0)],
    [(0.0, 0.0), (1.0, float("nan")), (3.0, 9.0)],
])
def test_invalid_history_rejects_before_pressure_telemetry_or_provenance_writes(bad_history):
    graph = _graph()
    graph.nodes[0]["epi_time_history"] = bad_history
    # Bind the adapter before the snapshot: cache creation is not a channel write.
    node = NodeNX(graph, 0)
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(ValueError):
        apply_glyph_obj(node, "THOL")
    assert graph.nodes[0] == before


def test_graphless_primitive_retains_explicit_signed_acceleration_contract():
    node = SimpleNamespace(dnfr=0.5, d2EPI=-2.0)
    _op_THOL(node, {"THOL_accel": 0.25})
    assert node.dnfr == 0.0
    assert node.d2EPI == -2.0


@pytest.mark.parametrize("stale", [float("nan"), float("inf"), -77.0])
def test_signed_physical_history_overrides_even_nonfinite_cached_acceleration(stale):
    graph = _graph()
    graph.nodes[0].update(
        EPI=1.0, epi_time_history=[(0.0, 1.0), (0.5, 1.25), (1.0, 1.0)]
    )
    graph.nodes[0][ALIAS_D2EPI[0]] = stale
    apply_glyph(graph, 0, "THOL")
    assert graph.nodes[0]["delta_nfr"] == 0.0
    assert graph.nodes[0][ALIAS_D2EPI[0]] == -2.0


@pytest.mark.parametrize("histories, expected", [
    ({}, 0.0),
    ({"epi_history": [0.0, 1.0, 4.0], "_epi_history": [0.0, 50.0, 0.0]}, 2.0),
    ({"epi_history": [], "_epi_history": [0.0, 1.0, 4.0]}, 2.0),
])
def test_legacy_and_absent_history_use_shared_precedence(histories, expected):
    graph = _graph()
    del graph.nodes[0]["epi_time_history"]
    del graph.nodes[0]["epi_history"]
    graph.nodes[0].update(histories)
    apply_glyph(graph, 0, "THOL")
    assert graph.nodes[0][ALIAS_D2EPI[0]] == expected
    assert graph.nodes[0]["delta_nfr"] == 0.5 + 0.25 * expected


def test_graph_dispatch_prepares_physical_samples_only_once():
    reads = []

    class CountedHistory(list):
        def __getitem__(self, index):
            reads.append(index)
            return super().__getitem__(index)

    graph = _graph()
    graph.nodes[0]["epi_time_history"] = CountedHistory(
        graph.nodes[0]["epi_time_history"]
    )
    apply_glyph(graph, 0, "THOL")
    assert reads == [-3, -2, -1]
    assert graph.nodes[0]["delta_nfr"] == 1.0


def test_graph_dispatch_rejects_stale_history_before_adapter_cache_creation():
    graph = _graph()
    graph.nodes[0]["EPI"] = 8.0
    metadata = deepcopy(graph.graph)
    node_data = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(ValueError, match="final EPI"):
        apply_glyph(graph, 0, "THOL")
    assert graph.graph == metadata
    assert graph.nodes[0] == node_data


def test_primitive_and_public_pressure_match_without_promoting_primitive_birth():
    direct, primitive = _graph(), _graph()
    # The configured existing depth limit suppresses birth, not pressure feedback.
    direct.graph["THOL_MAX_BIFURCATION_DEPTH"] = 0
    SelfOrganization()(direct, 0)
    apply_glyph(primitive, 0, "THOL")
    assert direct.nodes[0]["delta_nfr"] == primitive.nodes[0]["delta_nfr"] == 1.0
    assert direct.nodes[0][ALIAS_D2EPI[0]] == primitive.nodes[0][ALIAS_D2EPI[0]]


def test_overflow_rejects_before_refreshing_cached_telemetry():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"THOL_accel": 1e308}
    before = deepcopy(dict(graph.nodes[0]))
    metadata = deepcopy(graph.graph)
    with pytest.raises(ValueError):
        apply_glyph(graph, 0, "THOL")
    assert graph.nodes[0] == before
    assert graph.graph == metadata
