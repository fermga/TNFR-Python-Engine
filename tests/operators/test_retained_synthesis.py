"""Finite retained NUL/EN contract controls, without producers or flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import THETA_PRIMARY
from tnfr.operators import retained_synthesis as implementation
from tnfr.operators.contraction import Contraction
from tnfr.operators.network_stage import execute_neighbor_stage, execute_pointwise_stage
from tnfr.operators.reception import Reception
from tnfr.operators.retained_organization import retained_organization_matches_graph
from tnfr.operators.retained_synthesis import (
    RetainedPairState,
    capture_retained_pair,
    execute_retained_synthesis_stage,
)
from tnfr.utils._structural_signature import structural_proof_signature


def _graph(graph_type=nx.Graph, *, core=0.75, member=0.25):
    graph = graph_type(
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        GLYPH_FACTORS={"NUL_scale": 0.5, "EN_mix": 0.25},
        DNFR_RECEPTION_MAX=0.75,
        RANDOM_SEED=17,
    )
    for node, epi, capacity, pressure, history in (
        ("core", core, 1.5, 0.125, ["AL", "IL"]),
        ("member", member, 0.5, 0.0, ["EN"]),
        ("peer", 0.125, 2.0, 0.25, ["AL"]),
    ):
        graph.add_node(
            node,
            EPI=epi,
            nu_f=capacity,
            theta=0.25,
            delta_nfr=pressure,
            epi_kind="form",
            glyph_history=history,
            private_memory={"owner": node},
        )
    graph.add_edge("member", "core", weight=0.375, label="internal")
    graph.add_edge("peer", "core", weight=0.0, label="boundary")
    if graph.is_multigraph():
        graph.add_edge("member", "core", weight=0.75, label="parallel")
    return graph


def _plain(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            edges,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _pair(graph):
    return capture_retained_pair(graph, ("core", "member"))


@pytest.mark.parametrize(
    "core, member",
    [
        (0.75, 0.25),
        (-0.75, 0.25),
        (-0.0, 0.0),
        (float.fromhex("0x1.fffffffffffffp+1023"),) * 2,
        (
            float.fromhex("0x1.fffffffffffffp+1023"),
            -float.fromhex("0x1.fffffffffffffp+1023"),
        ),
        (float.fromhex("0x0.0000000000001p-1022"), 0.0),
    ],
)
def test_exact_chart_round_trip_avoids_intermediate_float_arithmetic(core, member):
    graph = _graph(core=core, member=member)
    graph.remove_edge("member", "core")
    before = _plain(graph)
    pair = _pair(graph)
    assert _plain(graph) == before
    assert type(pair.form_mean) is Fraction
    assert type(pair.form_contrast) is Fraction
    assert pair.form_mean + pair.form_contrast == Fraction.from_float(core)
    assert pair.form_mean - pair.form_contrast == Fraction.from_float(member)
    assert pair.members == ("core", "member")
    assert pair.core == "core" and pair.member == "member"
    assert retained_organization_matches_graph(graph, pair.organization)
    if core == float.fromhex("0x0.0000000000001p-1022"):
        assert pair.form_mean > 0 and float(pair.form_mean) == 0.0


def test_chart_zero_does_not_erase_raw_signed_zero_distinction():
    graph = _graph(core=-0.0, member=0.0)
    negative = _pair(graph)
    graph.nodes["core"]["EPI"] = 0.0
    positive = _pair(graph)
    assert negative.form_mean == positive.form_mean == 0
    assert negative.form_contrast == positive.form_contrast == 0
    assert negative.organization._proof_stamp != positive.organization._proof_stamp


@pytest.mark.parametrize(
    "members",
    [(), ("core",), ("core", "member", "peer"), ["core", "member"], ("core", "core")],
)
def test_pair_cardinality_and_uniqueness_are_required_without_mutation(members):
    graph = _graph()
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError)):
        capture_retained_pair(graph, members)
    assert _plain(graph) == before


@pytest.mark.parametrize("field", ["form_mean", "form_contrast"])
def test_chart_derived_values_are_readmitted_in_constructor_and_execution(field):
    graph = _graph()
    pair = _pair(graph)
    with pytest.raises(ValueError, match="chart"):
        replace(pair, **{field: Fraction(99)})
    with pytest.raises(ValueError, match="chart"):
        replace(pair, **{field: float(getattr(pair, field))})
    object.__setattr__(pair, field, Fraction(99))
    before = _plain(graph)
    with pytest.raises(ValueError, match="chart"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_chart_rejects_unsealed_organization():
    graph = _graph()
    pair = _pair(graph)
    forged = replace(pair.organization, members=("member", "core"))
    with pytest.raises(ValueError, match="intact"):
        RetainedPairState(
            forged, pair.form_mean, pair.form_contrast, pair.reception_kinds
        )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_fragment_matches_actual_nul_and_explicit_en_and_exact_dyadic_chart(graph_type):
    graph = _graph(graph_type)
    ordinary = deepcopy(graph)
    original_ports = tuple(graph.edges)
    peer = deepcopy(graph.nodes["peer"])
    pair = _pair(graph)
    result = execute_retained_synthesis_stage(graph, pair)
    execute_pointwise_stage(
        ordinary, Contraction(), ("core", "member"), require_exact_operator=True
    )
    execute_neighbor_stage(
        ordinary, Reception(), ("core",), source_nodes=("member",), track_sources=False
    )
    assert tuple(graph) == ("core", "member", "peer")
    assert tuple(graph.edges) == original_ports
    assert graph.nodes["peer"] == peer
    assert graph.nodes["core"] == ordinary.nodes["core"]
    assert graph.nodes["member"] == ordinary.nodes["member"]
    assert result.after.organization.member_states[0].epi == 0.3125
    assert result.after.organization.member_states[1].epi == 0.125
    assert result.before.form_mean == Fraction(1, 2)
    assert result.before.form_contrast == Fraction(1, 4)
    assert result.contracted.form_mean == Fraction(1, 4)
    assert result.contracted.form_contrast == Fraction(1, 8)
    assert result.after.form_mean == Fraction(7, 32)
    assert result.after.form_contrast == Fraction(3, 32)
    assert result.contraction_result.stage_result.glyph == "NUL"
    assert result.reception_result.glyph == "EN"
    assert result.after.organization.member_states[0].grammar.history[-2:] == (
        "NUL",
        "EN",
    )
    assert result.after.organization.member_states[1].grammar.history[-1:] == ("NUL",)
    observed = result.reception_result.reception_observations[0]
    assert observed.source_nodes == ("member",)
    assert observed.neighbors == ("member",)
    assert observed.neighbor_epi_values == (0.125,)
    assert observed.target_epi_before == 0.375
    assert observed.target_epi_after == 0.3125
    assert observed.source_selection_origin == "explicit_request"
    assert not observed.source_tracking_enabled
    assert retained_organization_matches_graph(graph, result.after.organization)


def test_unread_external_form_is_not_admitted_as_a_donor():
    graph = _graph()
    graph.nodes["peer"]["EPI"] = float("nan")
    graph.nodes["peer"]["theta"] = None
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    assert result.reception_result.reception_observations[0].neighbors == ("member",)


@pytest.mark.parametrize("kind", [None, "absent"])
@pytest.mark.parametrize("node", ["core", "member"])
def test_reception_kind_view_keeps_none_distinct_from_absent(node, kind):
    graph = _graph()
    if kind is None:
        graph.nodes[node]["epi_kind"] = None
    else:
        del graph.nodes[node]["epi_kind"]
    pair = _pair(graph)
    index = pair.members.index(node)
    assert pair.organization.member_states[index].epi_kind == ""
    assert pair.reception_kinds[index] == ("None" if kind is None else "")
    result = execute_retained_synthesis_stage(graph, pair)
    observed = result.reception_result.reception_observations[0]
    assert observed.target_epi_kind_before == pair.reception_kinds[0]
    assert observed.neighbor_epi_kinds == (pair.reception_kinds[1],)


def test_reception_kind_channel_is_rechecked_against_live_authoritative_state():
    graph = _graph()
    del graph.nodes["core"]["epi_kind"]
    pair = _pair(graph)
    forged = replace(pair, reception_kinds=("None", "form"))
    before = _plain(graph)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_synthesis_stage(graph, forged)
    assert _plain(graph) == before


def test_declared_strict_pressure_policy_admits_the_actual_fragment():
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    result = execute_retained_synthesis_stage(
        graph,
        _pair(graph),
        contraction_kwargs={"validate_preconditions": True},
        reception_kwargs={"validate_preconditions": True},
    )
    assert result.after.form_mean == Fraction(7, 32)


@pytest.mark.parametrize("core, member", [(-0.75, 0.25), (0.0, 0.0), (0.5, -0.5)])
def test_signed_and_zero_forms_follow_current_kernels(core, member):
    graph = _graph(core=core, member=member)
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    expected_core = 0.5 * (0.75 * core + 0.25 * member)
    expected_member = 0.5 * member
    assert result.after.organization.member_states[0].epi == expected_core
    assert result.after.organization.member_states[1].epi == expected_member
    assert result.after.form_mean + result.after.form_contrast == Fraction.from_float(
        expected_core
    )


def test_clipping_is_observed_without_asserting_unclipped_chart_transport():
    graph = _graph()
    graph.graph["EPI_MAX"] = 0.2
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    assert result.contracted.organization.member_states[0].epi == 0.2
    after_core = 0.75 * 0.2 + 0.25 * 0.125
    assert result.after.organization.member_states[0].epi == after_core
    assert (
        result.after.form_mean == (Fraction.from_float(after_core) + Fraction(1, 8)) / 2
    )
    assert result.after.form_mean != Fraction(7, 32)


def test_disabled_optional_form_scale_keeps_actual_nul_law():
    graph = _graph()
    graph.graph["EDGE_AWARE_ENABLED"] = False
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    assert result.contracted.form_mean == result.before.form_mean
    assert result.contracted.form_contrast == result.before.form_contrast
    assert result.after.organization.member_states[0].epi == 0.625
    assert result.after.organization.member_states[0].capacity == 0.75
    assert result.after.organization.member_states[0].pressure == 0.25


def test_mix_one_can_erase_previous_core_form_without_removing_identity():
    results = []
    for core in (0.75, 0.5):
        graph = _graph(core=core)
        graph.graph["GLYPH_FACTORS"]["EN_mix"] = 1.0
        result = execute_retained_synthesis_stage(graph, _pair(graph))
        assert result.after.members == ("core", "member")
        assert result.after.form_mean == Fraction(1, 8)
        assert result.after.form_contrast == 0
        results.append(result)
    assert results[0].before.form_mean != results[1].before.form_mean


@pytest.mark.parametrize(
    "factor, bad",
    [("EN_mix", 0.0), ("EN_mix", 1.25), ("NUL_scale", 0.0), ("NUL_scale", 1.0)],
)
def test_existing_factor_domains_reject_and_restore_complete_graph(factor, bad):
    graph = _graph()
    graph.graph["GLYPH_FACTORS"][factor] = bad
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(ValueError):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_default_reception_pressure_admission_rolls_back_successful_nul():
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    del graph.graph["DNFR_RECEPTION_MAX"]
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(ValueError, match="DNFR"):
        execute_retained_synthesis_stage(
            graph, pair, reception_kwargs={"validate_preconditions": True}
        )
    assert _plain(graph) == before


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph])
def test_capture_without_internal_source_is_valid_but_action_rejects(graph_type):
    graph = _graph(graph_type)
    graph.remove_edge("member", "core")
    if graph.is_directed():
        graph.add_edge("core", "member", weight=0.5)
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(ValueError, match="incoming source"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["form", "history", "temporal", "edge", "port"])
def test_live_source_changes_reject_old_observation_without_further_mutation(change):
    graph = _graph()
    pair = _pair(graph)
    if change == "form":
        graph.nodes["member"]["EPI"] = 0.375
    elif change == "history":
        graph.nodes["member"]["glyph_history"].append("IL")
    elif change == "temporal":
        graph.nodes["member"]["epi_time_history"] = [(0, 0.25)]
    elif change == "edge":
        graph.edges["member", "core"]["weight"] = 0.5
    else:
        graph.add_edge("member", "peer")
    before = _plain(graph)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_foreign_owner_rejects_even_for_equal_primitive_state():
    graph = _graph()
    pair = _pair(_graph())
    before = _plain(graph)
    with pytest.raises(ValueError, match="foreign"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


@pytest.mark.parametrize(
    "option",
    [
        "source_nodes",
        "sequence_context",
        "compute_delta_nfr",
        "track_sources",
        "transaction_snapshot",
        "require_exact_operator",
    ],
)
@pytest.mark.parametrize("phase", ["contraction_kwargs", "reception_kwargs"])
def test_forbidden_overrides_reject_before_any_stage(option, phase):
    graph = _graph()
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(ValueError, match="unsupported"):
        execute_retained_synthesis_stage(graph, pair, **{phase: {option: None}})
    assert _plain(graph) == before


@pytest.mark.parametrize("field", ["__call__", "_execute", "name", "glyph"])
def test_canonical_reception_bindings_are_required(monkeypatch, field):
    graph = _graph()
    pair = _pair(graph)
    before = _plain(graph)
    replacement = "other" if field == "name" else (lambda *args, **kwargs: None)
    monkeypatch.setattr(Reception, field, replacement)
    with pytest.raises(TypeError, match="canonical Reception"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_second_stage_failure_restores_first_stage_and_graph_metadata(monkeypatch):
    graph = _graph()
    pair = _pair(graph)
    before = _plain(graph)
    original = implementation.execute_neighbor_stage

    def fail_after_en(*args, **kwargs):
        original(*args, **kwargs)
        graph.graph["partial_report"] = [1]
        raise RuntimeError("synthetic second-stage failure")

    monkeypatch.setattr(implementation, "execute_neighbor_stage", fail_after_en)
    with pytest.raises(RuntimeError, match="second-stage failure"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["donor", "peer", "edge", "phase"])
def test_protected_state_violation_after_real_en_rolls_back_both_stages(
    monkeypatch, change
):
    graph = _graph()
    pair = _pair(graph)
    before = _plain(graph)
    original = implementation.execute_neighbor_stage

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "donor":
            graph.nodes["member"]["private_memory"]["changed"] = True
        elif change == "peer":
            graph.nodes["peer"]["EPI"] = 0.5
        elif change == "edge":
            graph.edges["member", "core"]["weight"] = 0.5
        else:
            graph.nodes["core"][THETA_PRIMARY] = 0.5
        return result

    monkeypatch.setattr(implementation, "execute_neighbor_stage", corrupt)
    with pytest.raises(RuntimeError, match="protected"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_final_observation_failure_rolls_back_the_complete_fragment(monkeypatch):
    graph = _graph()
    pair = _pair(graph)
    before = _plain(graph)
    original = implementation.capture_retained_pair
    count = 0

    def fail_last(*args, **kwargs):
        nonlocal count
        count += 1
        captured = original(*args, **kwargs)
        if count == 3:
            assert captured.organization.member_states[0].grammar.history[-2:] == (
                "NUL",
                "EN",
            )
            assert captured.organization.member_states[1].grammar.history[-1:] == (
                "NUL",
            )
            raise RuntimeError("synthetic final observation failure")
        return captured

    monkeypatch.setattr(implementation, "capture_retained_pair", fail_last)
    with pytest.raises(RuntimeError, match="final observation failure"):
        execute_retained_synthesis_stage(graph, pair)
    assert _plain(graph) == before


def test_dispatch_rejects_unrelated_contraction_and_changed_donor_state():
    graph = _graph()
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    with pytest.raises(ValueError, match="boundaries"):
        replace(result, contracted=result.before)
    graph.nodes["member"]["private_memory"]["changed"] = True
    altered = _pair(graph)
    with pytest.raises(ValueError, match="other member"):
        replace(result, after=altered)


def test_dispatch_rejects_unrelated_en_observation_and_core_channels():
    graph = _graph()
    result = execute_retained_synthesis_stage(graph, _pair(graph))
    other = _graph(core=0.5)
    unrelated = execute_retained_synthesis_stage(other, _pair(other))
    with pytest.raises(ValueError, match="Reception read"):
        replace(result, reception_result=unrelated.reception_result)
    for field in ("nu_f", "delta_nfr"):
        old = graph.nodes["core"][field]
        graph.nodes["core"][field] = old + 0.125
        altered = _pair(graph)
        with pytest.raises(ValueError, match="capacity and pressure"):
            replace(result, after=altered)
        graph.nodes["core"][field] = old
