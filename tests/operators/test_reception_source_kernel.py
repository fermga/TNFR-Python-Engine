"""Explicit EN source requests change the actual read, not diagnostic ranking."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_attr_str
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_EPI_KIND
from tnfr.node import NodeNX
from tnfr.operators import _apply_prepared_reception_glyph, apply_glyph
from tnfr.operators._epi_domain import validate_affine_epi_graph_input
from tnfr.operators._reception_kernel import (
    apply_reception_read_snapshot,
    capture_reception_read_snapshot,
    reception_read_snapshot_matches_graph,
    validate_reception_source_nodes,
)
from tnfr.types import Glyph


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    for node, form, kind in (
        (0, 0.25, "target"),
        (1, 0.5, "first"),
        (2, 1.0, "second"),
    ):
        graph.add_node(node, **{ALIAS_EPI[0]: form, ALIAS_EPI_KIND[0]: kind})
    graph.add_edges_from([(1, 0), (2, 0)])
    return graph


def _capture(graph, sources=None, **kwargs):
    return capture_reception_read_snapshot(
        graph, 0, track_sources=False, source_nodes=sources, **kwargs
    )


def _state(graph):
    return deepcopy((dict(graph.nodes(data=True)), tuple(graph.edges), graph.graph))


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_explicit_subset_is_the_actual_mean_and_retains_source_association(graph_type):
    graph = _graph(graph_type)
    if graph.is_multigraph():
        graph.add_edge(1, 0)
    before = _state(graph)

    snapshot = _capture(graph, (2,))

    assert snapshot.source_nodes == snapshot.neighbors == (2,)
    assert snapshot.neighbor_epi_values == (1.0,)
    assert snapshot.neighbor_epi_kinds == ("second",)
    assert snapshot.neighbor_epi_mean == 1.0
    assert snapshot.source_selection_origin == "explicit_request"
    assert snapshot._proof_fields_are_intact()
    assert _state(graph) == before
    apply_reception_read_snapshot(NodeNX(graph, 0), 0.5, snapshot)
    assert get_attr(graph.nodes[0], ALIAS_EPI) == 0.625
    assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == "second"


def test_legacy_and_explicit_all_inputs_share_arithmetic_but_not_request_proof():
    graph = _graph()
    legacy = _capture(graph)
    selected = _capture(graph, (1, 2))

    assert legacy.neighbor_epi_mean == selected.neighbor_epi_mean == 0.75
    assert legacy.neighbor_epi_values == selected.neighbor_epi_values
    assert legacy.neighbor_epi_kinds == selected.neighbor_epi_kinds
    assert legacy.source_nodes is None
    assert legacy.source_selection_origin == "legacy_all_neighbors"
    assert legacy._proof_stamp != selected._proof_stamp
    assert validate_reception_source_nodes(None, None, None) is None


@pytest.mark.parametrize("sources, kind", [((1, 2), "first"), ((2, 1), "second")])
def test_supplied_order_resolves_equal_magnitude_kind_ties(sources, kind):
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.0
    graph.nodes[1][ALIAS_EPI[0]] = 1.0
    graph.nodes[2][ALIAS_EPI[0]] = -1.0
    snapshot = _capture(graph, sources)

    apply_reception_read_snapshot(NodeNX(graph, 0), 1.0, snapshot)

    assert get_attr(graph.nodes[0], ALIAS_EPI) == 0.0
    assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == kind


@pytest.mark.parametrize(
    "sources", [(), (1, 1), (99,), (0,), ([],), [1], {1}, "1", True]
)
def test_invalid_source_requests_reject_without_writes(sources):
    graph = _graph()
    before = _state(graph)
    with pytest.raises((ValueError, TypeError)):
        _capture(graph, sources)
    assert _state(graph) == before


def test_directed_source_must_be_a_realized_predecessor():
    graph = _graph(nx.DiGraph)
    graph.remove_edge(2, 0)
    graph.add_edge(0, 2)
    assert _capture(graph, (1,)).neighbors == (1,)
    with pytest.raises(ValueError, match="incoming"):
        _capture(graph, (2,))


def test_realized_self_loop_remains_a_valid_explicit_source():
    graph = _graph()
    graph.add_edge(0, 0)
    snapshot = _capture(graph, (0,))
    assert snapshot.neighbors == (0,)
    assert snapshot.neighbor_epi_mean == 0.25


@pytest.mark.parametrize(
    "invalid", [True, "0.5", None, math.nan, math.inf, Fraction(1, 10**400)]
)
def test_selected_authoritative_form_cannot_fall_through_to_another_alias(invalid):
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = invalid
    graph.nodes[1][ALIAS_EPI[1]] = 0.5
    before = _state(graph)
    with pytest.raises(ValueError):
        _capture(graph, (1,))
    assert _state(graph) == before


def test_secondary_alias_is_accepted_only_when_authoritative_alias_is_absent():
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[1]] = graph.nodes[1].pop(ALIAS_EPI[0])
    assert _capture(graph, (1,)).neighbor_epi_mean == 0.5


def test_explicit_source_requires_present_form_and_legacy_keeps_missing_fallback():
    graph = _graph()
    graph.nodes[1].pop(ALIAS_EPI[0])
    legacy = _capture(graph)
    assert legacy.neighbors == (1, 2)
    assert legacy.neighbor_epi_values == (0.25, 1.0)
    assert legacy.neighbor_dominant_values == (0.0, 1.0)
    assert legacy.neighbor_epi_mean == 0.625
    with pytest.raises(ValueError, match="explicit EPI"):
        _capture(graph, (1,))
    assert _capture(graph, (2,)).neighbor_epi_mean == 1.0


@pytest.mark.parametrize("isolated", [False, True])
def test_legacy_all_missing_and_isolate_retain_the_existing_empty_decision(isolated):
    graph = _graph()
    if isolated:
        graph.remove_edges_from(tuple(graph.edges))
    else:
        for source in (1, 2):
            graph.nodes[source].pop(ALIAS_EPI[0])
    snapshot = _capture(graph)
    assert snapshot.neighbors == ()
    assert snapshot.neighbor_epi_mean == 0.25
    apply_reception_read_snapshot(NodeNX(graph, 0), 0.5, snapshot)
    assert get_attr(graph.nodes[0], ALIAS_EPI) == 0.25
    assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == "EN"


def test_unselected_forms_and_unconsumed_phase_capacity_do_not_define_selection():
    graph = _graph()
    graph.nodes[2][ALIAS_EPI[0]] = "excluded"
    graph.nodes[1].update(theta=math.nan, nu_f="not an oscillation frequency")
    validate_affine_epi_graph_input(graph, 0, Glyph.EN, source_nodes=(1,))
    snapshot = _capture(graph, (1,))
    assert snapshot.neighbor_epi_mean == 0.5
    graph.nodes[1].update(theta=1.0, nu_f=-1.0)
    assert reception_read_snapshot_matches_graph(graph, snapshot)
    with pytest.raises(ValueError):
        validate_affine_epi_graph_input(graph, 0, Glyph.EN)


@pytest.mark.parametrize("glyph", [Glyph.AL, Glyph.RA, Glyph.UM, None])
def test_other_glyphs_cannot_consume_an_explicit_reception_source_request(glyph):
    with pytest.raises(ValueError, match="only for Reception"):
        validate_affine_epi_graph_input(_graph(), 0, glyph, source_nodes=(1,))


@pytest.mark.parametrize("change", ["form", "kind", "presence", "support"])
def test_live_revalidation_binds_selected_source_state_and_support(change):
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = 0.0
    snapshot = _capture(graph, (1,))
    if change == "form":
        graph.nodes[1][ALIAS_EPI[0]] = 0.5
    elif change == "kind":
        graph.nodes[1][ALIAS_EPI_KIND[0]] = "changed"
    elif change == "presence":
        graph.nodes[1].pop(ALIAS_EPI[0])
    else:
        graph.remove_edge(1, 0)
    assert not reception_read_snapshot_matches_graph(graph, snapshot)


def test_unselected_read_changes_do_not_stale_a_track_free_selected_snapshot():
    graph = _graph()
    snapshot = _capture(graph, (1,))
    graph.nodes[2][ALIAS_EPI[0]] = "excluded"
    graph.add_node(3, EPI="also excluded")
    graph.add_edge(3, 0)
    assert reception_read_snapshot_matches_graph(graph, snapshot)


def test_tampered_request_or_source_association_cannot_reuse_the_read_seal():
    graph = _graph()
    snapshot = _capture(graph, (1, 2))
    demoted = replace(snapshot, source_nodes=None)
    assert not demoted._proof_fields_are_intact()
    reordered = replace(
        snapshot,
        source_nodes=(2, 1),
        neighbors=(2, 1),
        neighbor_epi_values=(1.0, 0.5),
        neighbor_dominant_values=(1.0, 0.5),
        neighbor_epi_kinds=("second", "first"),
    )
    assert not reordered._proof_fields_are_intact()
    with pytest.raises(ValueError, match="requested source order"):
        replace(snapshot, source_nodes=(2, 1))


def test_metric_consumer_binding_recaptures_the_same_selected_request():
    graph = _graph()
    consumer = deepcopy(graph)
    consumer.nodes[2][ALIAS_EPI[0]] = "excluded"
    snapshot = _capture(graph, (1,), _metric_consumer_graph_owner=consumer)
    assert snapshot.source_nodes == (1,)
    assert snapshot._proof_fields_are_intact()
    consumer.nodes[1][ALIAS_EPI[0]] = 0.9
    assert not reception_read_snapshot_matches_graph(graph, snapshot)
    with pytest.raises(ValueError, match="consumer does not match"):
        _capture(graph, (1,), _metric_consumer_graph_owner=consumer)


@pytest.mark.parametrize("route", ["selected", "stale", "legacy"])
def test_raw_prepared_adapter_keeps_the_actual_source_domain(route):
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"EN_mix": 0.5}
    for data in graph.nodes.values():
        data.update(nu_f=1.0, theta=0.0, delta_nfr=0.1, glyph_history=["AL"])
    graph.nodes[2][ALIAS_EPI[0]] = "excluded invalid form"
    snapshot = _capture(graph, (1,))
    if route == "stale":
        graph.nodes[1][ALIAS_EPI[0]] = 0.75
    before = _state(graph)

    if route == "selected":
        _apply_prepared_reception_glyph(
            graph, 0, Glyph.EN, window=7, prepared_state=snapshot
        )
        assert get_attr(graph.nodes[0], ALIAS_EPI) == 0.375
        assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == "first"
        assert graph.nodes[0]["glyph_history"][-1] == "EN"
        assert graph.nodes[2][ALIAS_EPI[0]] == "excluded invalid form"
    else:
        if route == "stale":
            with pytest.raises(RuntimeError, match="stale"):
                _apply_prepared_reception_glyph(
                    graph, 0, Glyph.EN, window=7, prepared_state=snapshot
                )
        else:
            with pytest.raises(ValueError):
                apply_glyph(graph, 0, Glyph.EN)
        assert _state(graph) == before
        assert "_node_cache" not in graph.graph
