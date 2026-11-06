"""Independent controls for explicitly supplied causal Reception inputs."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators._reception_kernel import (
    capture_reception_read_snapshot,
    reception_read_snapshot_matches_graph,
)
from tnfr.operators.definitions import Coherence, Reception, Resonance
from tnfr.operators.definitions_base import Operator
from tnfr.operators.event_runtime import _reception_observation_matches_neighbor_step
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.metrics_basic import reception_metrics
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import require_finite_real_scalar_epi


def _graph(graph_type=nx.DiGraph):
    graph = graph_type(
        GLYPH_FACTORS={"EN_mix": 0.25},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_MIN=-2.0,
        EPI_MAX=2.0,
    )
    for node, value in enumerate((0.2, 0.8, -0.4, 0.2)):
        graph.add_node(
            node,
            EPI=value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.05,
            epi_kind=f"kind-{node}",
            glyph_history=["AL", "IL"],
            _grammar_u2_debt=0,
            _grammar_prior_coherence=True,
            _operator_step=2,
        )
    for source in (1, 2, 3):
        graph.add_edge(source, 0, weight=1.0)
    return graph


def _form(graph, node=0):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _plain(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return deepcopy(
        (
            tuple(graph.nodes(data=True)),
            edges,
            graph.graph,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        )
    )


def _apply(graph, route, source_nodes, *, targets=(0,), **kwargs):
    if route == "public":
        assert targets == (0,)
        assert (
            Reception()(
                graph, 0, source_nodes=source_nodes, track_sources=False, **kwargs
            )
            is None
        )
        return None
    executor = (
        execute_neighbor_stage if route == "stage" else execute_network_operator_stage
    )
    if route == "stage":
        kwargs["track_sources"] = False
    return executor(graph, Reception(), targets, source_nodes=source_nodes, **kwargs)


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_explicit_supported_source_controls_the_actual_form_and_occurrence(
    route, graph_type
):
    graph = _graph(graph_type)
    peers = {node: deepcopy(graph.nodes[node]) for node in (1, 2, 3)}
    result = _apply(graph, route, (1,))
    assert _form(graph) == (1.0 - 0.25) * 0.2 + 0.25 * 0.8
    assert (
        get_attr(graph.nodes[0], ALIAS_EPI_KIND, strict=True, conv=lambda value: value)
        == "kind-1"
    )
    assert {node: graph.nodes[node] for node in peers} == peers
    assert read_grammar_state(graph.nodes[0]).history == ("AL", "IL", "EN")
    assert read_grammar_state(graph.nodes[0]).operator_step == 3
    for aliases, expected in ((ALIAS_VF, 1.0), (ALIAS_DNFR, 0.05), (ALIAS_THETA, 0.0)):
        assert get_attr(graph.nodes[0], aliases) == expected
    if result is not None:
        assert result.nodes_processed == 1
        (observation,) = result.reception_observations
        assert observation.source_nodes == (1,)
        assert observation.source_selection_origin == "explicit_request"
        assert observation.neighbors == (1,)
        assert observation.neighbor_epi_values == (0.8,)
        assert observation._proof_fields_are_intact()
        assert observation.auxiliary_stability_certified is False


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
def test_omitted_and_none_keep_the_ordinary_all_incoming_map(route):
    omitted, explicit = _graph(), _graph()
    if route == "public":
        Reception()(omitted, 0, track_sources=False)
    else:
        executor = (
            execute_neighbor_stage
            if route == "stage"
            else execute_network_operator_stage
        )
        executor(
            omitted,
            Reception(),
            (0,),
            **({"track_sources": False} if route == "stage" else {}),
        )
    result = _apply(explicit, route, None)
    assert _form(omitted) == _form(explicit)
    assert _form(explicit) == pytest.approx(0.2)
    assert get_attr(
        omitted.nodes[0], ALIAS_EPI_KIND, strict=True, conv=lambda value: value
    ) == get_attr(
        explicit.nodes[0], ALIAS_EPI_KIND, strict=True, conv=lambda value: value
    )
    assert read_grammar_state(omitted.nodes[0]) == read_grammar_state(explicit.nodes[0])
    if result is not None:
        assert result.reception_observations[0].source_nodes is None
        assert (
            result.reception_observations[0].source_selection_origin
            == "legacy_all_neighbors"
        )


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
@pytest.mark.parametrize(
    "sources",
    [(), (1, 1), (99,), (0,), [1], "1"],
    ids=["empty", "duplicate", "absent", "unsupported_self", "list", "string"],
)
def test_invalid_explicit_source_request_rejects_without_owned_writes(route, sources):
    graph = _graph()
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _apply(graph, route, sources)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_directed_outgoing_arc_cannot_supply_reception(route):
    graph = _graph()
    graph.remove_edge(1, 0)
    graph.add_edge(0, 1, weight=1.0)
    before = _plain(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _apply(graph, route, (1,))
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize(
    "invalid",
    [None, True, math.nan, Fraction(1, 10**400)],
    ids=["missing", "boolean", "nan", "underflow"],
)
def test_selected_form_must_be_explicit_and_authoritatively_admitted(route, invalid):
    graph = _graph()
    if invalid is None:
        for alias in ALIAS_EPI:
            graph.nodes[1].pop(alias, None)
    else:
        graph.nodes[1][ALIAS_EPI[0]] = invalid
        graph.nodes[1][ALIAS_EPI[-1]] = 0.8
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _apply(graph, route, (1,))
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_explicit_selection_does_not_read_unselected_form_or_unsupported_phase_capacity(
    route,
):
    graph = _graph()
    graph.nodes[2][ALIAS_EPI[0]] = math.nan
    for node in graph:
        graph.nodes[node][ALIAS_THETA[0]] = math.nan
        graph.nodes[node][ALIAS_VF[0]] = -1.0
    _apply(graph, route, (1,))
    assert _form(graph) == (1.0 - 0.25) * 0.2 + 0.25 * 0.8


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
def test_caller_order_retains_kind_tie_even_when_numeric_response_is_identical(route):
    outputs = []
    for sources in ((1, 2), (2, 1)):
        graph = _graph()
        graph.graph["GLYPH_FACTORS"]["EN_mix"] = 0.5
        graph.nodes[0]["EPI"] = 0.1
        graph.nodes[1]["EPI"] = 0.6
        graph.nodes[2]["EPI"] = -0.6
        _apply(graph, route, sources)
        outputs.append(
            (
                _form(graph),
                get_attr(
                    graph.nodes[0],
                    ALIAS_EPI_KIND,
                    strict=True,
                    conv=lambda value: value,
                ),
            )
        )
    assert outputs == [(0.05, "kind-1"), (0.05, "kind-2")]


def test_common_selected_sources_are_read_from_one_jacobi_state():
    graph = _graph()
    graph.add_edge(1, 1, weight=1.0)
    graph.add_edge(2, 1, weight=1.0)
    before = {node: _form(graph, node) for node in graph}
    result = _apply(graph, "stage", (1, 2), targets=(0, 1))
    mean = (before[1] + before[2]) / 2.0
    for node in (0, 1):
        assert _form(graph, node) == (1.0 - 0.25) * before[node] + 0.25 * mean
    assert all(
        item.neighbor_epi_values == (0.8, -0.4)
        for item in result.reception_observations
    )
    assert all(item.source_nodes == (1, 2) for item in result.reception_observations)


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize("mode", ["raise", "fallback"])
def test_grammar_cannot_report_substitution_as_selected_reception(route, mode):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    graph.nodes[0]["_grammar_u2_debt"] = 3
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        _apply(graph, route, (1,))
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_late_callback_failure_restores_selected_form_history_and_metadata(route):
    graph = _graph()
    before = _plain(graph)

    def reject(subject):
        assert _form(subject) != 0.2
        assert read_grammar_state(subject.nodes[0]).history[-1] == "EN"
        subject.graph["late_selection_marker"] = True
        raise RuntimeError("synthetic late EN rejection")

    with pytest.raises(RuntimeError, match="late EN rejection"):
        _apply(graph, route, (1,), compute_delta_nfr=reject)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_canonical_execution_hook_is_required_for_new_domain(route, monkeypatch):
    graph = _graph()
    before = _plain(graph)

    def changed(self, subject, node, **kwargs):
        pytest.fail("changed execution must not be silently bypassed")

    monkeypatch.setattr(Operator, "_execute", changed)
    with pytest.raises(ValueError, match="canonical"):
        _apply(graph, route, (1,))
    assert _plain(graph) == before


def test_snapshot_selection_is_sealed_and_rechecked_without_value_only_reassociation():
    graph = _graph()
    graph.nodes[2]["EPI"] = 0.8
    first = capture_reception_read_snapshot(
        graph, 0, track_sources=False, source_nodes=(1,)
    )
    second = capture_reception_read_snapshot(
        graph, 0, track_sources=False, source_nodes=(2,)
    )
    assert first.neighbor_epi_mean == second.neighbor_epi_mean
    assert first.source_nodes != second.source_nodes
    with pytest.raises(ValueError, match="requested source order"):
        replace(first, source_nodes=(2,))
    assert not replace(
        first, source_nodes=(2,), neighbors=(2,)
    )._proof_fields_are_intact()
    assert reception_read_snapshot_matches_graph(graph, first)
    graph.remove_edge(1, 0)
    assert not reception_read_snapshot_matches_graph(graph, first)


def test_executor_metrics_keep_selection_and_standalone_metrics_do_not_infer_it():
    graph = _graph()
    _apply(graph, "public", (1,), collect_metrics=True)
    recorded = graph.graph["operator_metrics"][-1]
    assert recorded["source_nodes"] == (1,)
    assert recorded["input_neighbors"] == (1,)
    assert recorded["source_selection_origin"] == "explicit_request"
    live = reception_metrics(graph, 0, 0.2)
    assert live["source_selection_origin"] == "live_legacy_observation"
    assert live["source_nodes"] is None


@pytest.mark.parametrize("sources", [(1,), (1, 2, 3)])
def test_new_selection_abstains_from_old_map_certificate_even_when_all_are_selected(
    sources,
):
    graph = _graph()
    result = _apply(graph, "stage", sources, include_epi_jump_certificate=True)
    assert result.epi_jump_certificate is None
    assert "source" in result.epi_jump_certificate_abstention_reason
    assert result.reception_observations[0]._proof_fields_are_intact()


def test_event_certificate_association_rejects_new_request_even_when_row_matches():
    old = nx.path_graph(2)
    old.graph.update(GLYPH_FACTORS={"EN_mix": 0.5})
    for node, value in enumerate((0.2, 0.4)):
        old.nodes[node].update(
            EPI=value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.05,
            epi_kind="same",
            glyph_history=["AL", "IL"],
        )
    new = deepcopy(old)
    legacy = execute_neighbor_stage(
        old, Reception(), (0, 1), track_sources=False, include_epi_jump_certificate=True
    )
    selected = _apply(new, "stage", (1,), include_epi_jump_certificate=True)
    assert legacy.epi_jump_certificate is not None
    step = legacy.epi_jump_certificate.steps[0]
    assert _reception_observation_matches_neighbor_step(
        legacy.reception_observations[0], 0, step
    )
    observation = selected.reception_observations[0]
    assert (
        observation.target_epi_after
        == legacy.reception_observations[0].target_epi_after
    )
    assert observation.neighbors == legacy.reception_observations[0].neighbors
    assert not _reception_observation_matches_neighbor_step(observation, 0, step)


def test_dispatcher_does_not_silently_assign_selected_sources_to_another_operator():
    graph = _graph()
    before = _plain(graph)
    with pytest.raises(ValueError):
        execute_network_operator_stage(graph, Coherence(), (0,), source_nodes=(1,))
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_selected_received_form_changes_a_later_actual_resonance_response(route):
    outputs = []
    for source_nodes in ((1,), None):
        graph = _graph()
        # Incoming EN sources are 1,2,3; target 3's same arc 3->0 later
        # gives RA its outgoing input. All four phase/capacity states agree.
        for node, form in enumerate((0.25, 0.75, -0.25, 0.25)):
            graph.nodes[node]["EPI"] = form
            graph.nodes[node]["epi_kind"] = "same"
        graph.graph["GLYPH_FACTORS"]["RA_epi_diff"] = 0.5
        _apply(graph, route, source_nodes)
        expected_received = 0.375 if source_nodes is not None else 0.25
        assert _form(graph, 0) == expected_received
        Resonance()(graph, 3)
        expected_response = 0.5 * 0.25 + 0.5 * expected_received
        assert _form(graph, 3) == expected_response
        assert read_grammar_state(graph.nodes[3]).history[-1] == "RA"
        assert read_grammar_state(graph.nodes[0]).history[-1] == "EN"
        assert set(graph.edges) == {(1, 0), (2, 0), (3, 0)}
        outputs.append(_form(graph, 3))
    assert outputs == [0.3125, 0.25]


def test_ranked_telemetry_can_retain_an_excluded_donor_without_using_its_form_or_kind():
    graph = _graph()
    graph.nodes[2]["EPI"] = 1.25
    graph.nodes[2]["theta"] = math.pi
    result = execute_neighbor_stage(
        graph,
        Reception(),
        (0,),
        source_nodes=(1,),
        track_sources=True,
        collect_metrics=True,
    )
    (observation,) = result.reception_observations
    ranked_ids = tuple(item[0] for item in observation.reception_sources)
    assert 2 in ranked_ids and 2 not in observation.neighbors
    assert observation.source_nodes == observation.neighbors == (1,)
    assert observation.neighbor_epi_values == (0.8,)
    assert _form(graph) == (1.0 - 0.25) * 0.2 + 0.25 * 0.8
    assert (
        get_attr(graph.nodes[0], ALIAS_EPI_KIND, strict=True, conv=lambda value: value)
        == "kind-1"
    )
    assert tuple(item[0] for item in graph.nodes[0]["_reception_sources"]) == ranked_ids
    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["input_neighbors"] == (1,)
    assert metrics["neighbor_count"] == 1
    assert metrics["num_sources"] == len(ranked_ids) > 1
