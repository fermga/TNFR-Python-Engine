"""Finite retained ZHIR admission and transaction controls, no producer."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_VF
from tnfr.mathematics.unified_numerical import normalize_phase
from tnfr.operators import retained_mutation as implementation
from tnfr.operators.mutation import Mutation
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.retained_mutation import execute_retained_mutation_stage
from tnfr.operators.retained_organization import (
    _protected_state,
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.utils._structural_signature import structural_proof_signature


def _graph(graph_type=nx.Graph):
    graph = graph_type(
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=17,
        _t=2.0,
    )
    for node, epi, phase in (
        ("core", 0.5, 0.0),
        ("member", 0.25, 0.0),
        ("peer", 0.75, 0.25),
    ):
        graph.add_node(
            node,
            EPI=epi,
            nu_f=1.0,
            theta=phase,
            delta_nfr=0.0625,
            epi_kind="form",
            glyph_history=["AL", "IL", "OZ"],
            epi_time_history=[(0.0, epi - 0.25), (1.0, epi - 0.125), (2.0, epi)],
            private_memory={"owner": node},
        )
    graph.add_edge("member", "core", weight=0.375, label="internal")
    graph.add_edge("peer", "core", weight=0.0, label="boundary")
    if graph.is_multigraph():
        graph.add_edge("member", "core", weight=0.75, label="parallel")
    return graph


def _record(graph):
    return capture_retained_organization(graph, ("core", "member"))


def _plain(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            (
                tuple(graph.edges(keys=True, data=True))
                if graph.is_multigraph()
                else tuple(graph.edges(data=True))
            ),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("targets", [("core",), ("member", "core")])
def test_actual_mutation_matches_shared_stage_and_retains_complete_boundary(
    graph_type, targets
):
    graph = _graph(graph_type)
    ordinary = deepcopy(graph)
    source = _record(graph)
    peer = deepcopy(graph.nodes["peer"])
    old_nodes = tuple(graph)
    old_edges = deepcopy(tuple(graph.edges(data=True)))
    result = execute_retained_mutation_stage(graph, source, targets)
    execute_pointwise_stage(ordinary, Mutation(), targets, require_exact_operator=True)
    assert tuple(graph) == old_nodes
    assert tuple(graph.edges(data=True)) == old_edges
    assert graph.nodes["peer"] == peer
    for node in graph:
        assert graph.nodes[node] == ordinary.nodes[node]
    assert graph.graph["_t"] == 2.0
    assert result.targets == targets
    assert result.stage_result.glyph == "ZHIR"
    assert result.stage_result.pointwise_epi_jump_certificate is None
    assert (
        tuple(item.node for item in result.stage_result.mutation_decision_observations)
        == targets
    )
    for index, observed in enumerate(
        result.stage_result.mutation_decision_observations
    ):
        assert observed.target_index == index and observed._proof_fields_are_intact()
        assert observed.theta_before == 0.0 and observed.theta_after == 0.25
        assert observed.operator_step == 4
        assert observed.recent_history == ("emission", "coherence", "dissonance")
        assert observed.trigger_certificate.observed_depi_dt == 0.125
        assert observed.trigger_certificate.predicted_depi_dt == 0.0625
        assert observed.trigger_certificate.observed_crossed
        assert not observed.trigger_certificate.predicted_crossed
        assert observed.trigger_certificate.current_endpoint_matches_state
    assert retained_organization_matches_graph(graph, result.after)
    assert not retained_organization_matches_graph(graph, source)


def test_private_guard_keeps_old_default_and_only_relaxes_selected_phase():
    graph = _graph()
    default = _protected_state(graph, ("core",))
    relaxed = _protected_state(graph, ("core",), preserve_member_phase=False)
    assert default == _protected_state(graph, ("core",), preserve_member_phase=True)
    graph.nodes["core"]["theta"] = 0.5
    assert default != _protected_state(graph, ("core",))
    assert relaxed == _protected_state(graph, ("core",), preserve_member_phase=False)
    graph.nodes["member"]["theta"] = 0.5
    assert relaxed != _protected_state(graph, ("core",), preserve_member_phase=False)
    with pytest.raises(TypeError, match="bool"):
        _protected_state(graph, ("core",), preserve_member_phase="false")


@pytest.mark.parametrize("phase", [-0.125, math.tau + 0.25])
def test_noncanonical_phase_chart_is_bound_to_normalized_proposal(phase):
    graph = _graph()
    graph.nodes["core"]["theta"] = phase
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    observed = result.stage_result.mutation_decision_observations[0]
    assert result.before.member_states[0].phase == phase
    assert observed.theta_before == float(normalize_phase(phase))
    assert observed.theta_after == result.after.member_states[0].phase


@pytest.mark.parametrize("pressure", [0.0, -0.0])
def test_signed_pressure_zero_retains_its_actual_direction(pressure):
    graph = _graph()
    graph.nodes["core"]["delta_nfr"] = pressure
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    observed = result.stage_result.mutation_decision_observations[0]
    assert observed.theta_shift == math.copysign(0.25, pressure)
    assert math.copysign(1.0, graph.nodes["core"]["delta_nfr"]) == math.copysign(
        1.0, pressure
    )
    assert result.after.member_states[0].pressure == 0.0


@pytest.mark.parametrize("kind", [None, "absent", ""])
def test_kind_presence_and_none_preserved_without_new_kind_policy(kind):
    graph = _graph()
    if kind == "absent":
        del graph.nodes["core"]["epi_kind"]
    else:
        graph.nodes["core"]["epi_kind"] = kind
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    observed = result.stage_result.mutation_decision_observations[0]
    assert observed.epi_kind_before == (None if kind == "absent" else kind)
    assert ("epi_kind" in graph.nodes["core"]) == (kind != "absent")
    assert result.after.member_states[0].epi_kind == ""


@pytest.mark.parametrize("window", [0, 1, 5])
def test_bounded_trace_keeps_own_monotonic_step_and_debt(window):
    graph = _graph()
    result = execute_retained_mutation_stage(
        graph, _record(graph), ("core",), execution_kwargs={"window": window}
    )
    state = result.after.member_states[0].grammar
    assert state.operator_step == 4 and state.u2_debt == 2 and state.prior_coherence
    expected = ("AL", "IL", "OZ", "ZHIR")
    assert state.history == (expected[-window:] if window else ())


def test_real_optional_postconditions_tau_and_metrics_are_forwarded():
    graph = _graph()
    result = execute_retained_mutation_stage(
        graph,
        _record(graph),
        ("core",),
        execution_kwargs={
            "tau": 0.125,
            "validate_postconditions": True,
            "collect_metrics": True,
        },
    )
    observed = result.stage_result.mutation_decision_observations[0]
    assert observed.tau == 0.125 and observed.structural_acceleration == 0.0
    assert not observed.bifurcation_potential
    assert len(graph.graph["operator_metrics"]) == 1


def test_two_samples_admit_growth_without_claiming_available_acceleration():
    graph = _graph()
    graph.nodes["core"]["epi_time_history"] = [(1.0, 0.375), (2.0, 0.5)]
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    observed = result.stage_result.mutation_decision_observations[0]
    assert observed.trigger_certificate.observed_depi_dt == 0.125
    assert observed.structural_acceleration == 0.0  # Existing unavailable fallback.


def test_legacy_step_evidence_does_not_gain_timestamp_endpoint_provenance():
    graph = _graph()
    del graph.nodes["core"]["epi_time_history"]
    graph.nodes["core"]["epi_history"] = [0.0, 0.125, 0.375]
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    certificate = result.stage_result.mutation_decision_observations[
        0
    ].trigger_certificate
    assert certificate.evidence.current_epi == 0.375
    assert certificate.current_epi == 0.5
    assert not certificate.physical_time_resolved
    assert certificate.current_endpoint_matches_state is None
    assert certificate.rate_gap is None
    assert certificate.evidence.sample_interval == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "one",
        "stale",
        "negative",
        "equal",
        "nan",
        "same_time",
        "zero_capacity",
        "no_il",
        "no_destabilizer",
    ],
)
def test_own_hard_evidence_and_grammar_rejection_is_atomic(bad):
    graph = _graph()
    data = graph.nodes["core"]
    if bad == "missing":
        del data["epi_time_history"]
    elif bad == "one":
        data["epi_time_history"] = [(2.0, 0.5)]
    elif bad == "stale":
        data["epi_time_history"][-1] = (2.0, 0.625)
    elif bad == "negative":
        data["epi_time_history"] = [(1.0, 0.625), (2.0, 0.5)]
    elif bad == "equal":
        graph.graph["ZHIR_THRESHOLD_XI"] = 0.125
    elif bad == "nan":
        data["epi_time_history"][-2] = (1.0, float("nan"))
    elif bad == "same_time":
        data["epi_time_history"][-2] = (2.0, 0.375)
    elif bad == "zero_capacity":
        data["nu_f"] = 0.0
    elif bad == "no_il":
        data["glyph_history"] = ["AL", "OZ"]
    else:
        data["glyph_history"] = ["AL", "IL"]
    source = _record(graph)
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        execute_retained_mutation_stage(
            graph, source, ("core",), execution_kwargs={"validate_preconditions": False}
        )
    assert _plain(graph) == before


@pytest.mark.parametrize(
    "targets", [(), ["core"], ("core", "core"), ("peer",), ("absent",)]
)
def test_explicit_target_subset_is_admitted_before_execution(targets):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError)):
        execute_retained_mutation_stage(graph, source, targets)
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["phase", "history", "temporal", "edge"])
def test_stale_source_rejects_without_repairing_it(change):
    graph = _graph()
    source = _record(graph)
    if change == "phase":
        graph.nodes["member"]["theta"] = 0.5
    elif change == "history":
        graph.nodes["member"]["glyph_history"].append("IL")
    elif change == "temporal":
        graph.nodes["core"]["epi_time_history"][-2] = (1.0, 0.25)
    else:
        graph.edges["member", "core"]["weight"] = 0.5
    before = _plain(graph)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


def test_foreign_and_forged_records_reject_without_mutation():
    graph = _graph()
    for source in (
        _record(_graph()),
        replace(_record(graph), members=("member", "core")),
    ):
        before = _plain(graph)
        with pytest.raises(ValueError, match="foreign"):
            execute_retained_mutation_stage(graph, source, ("core",))
        assert _plain(graph) == before


@pytest.mark.parametrize(
    "option",
    [
        "compute_delta_nfr",
        "sequence_context",
        "epi_time_history",
        "trigger_certificate",
        "transaction_snapshot",
        "require_exact_operator",
    ],
)
def test_context_history_callback_and_fallback_overrides_are_forbidden(option):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    with pytest.raises(ValueError, match="unsupported"):
        execute_retained_mutation_stage(
            graph, source, ("core",), execution_kwargs={option: None}
        )
    assert _plain(graph) == before


@pytest.mark.parametrize("field", ["__call__", "_execute", "name", "glyph"])
def test_mutation_class_bindings_must_remain_canonical(monkeypatch, field):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    monkeypatch.setattr(
        Mutation, field, "other" if field == "name" else lambda *args, **kwargs: None
    )
    with pytest.raises(TypeError, match="canonical"):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


@pytest.mark.parametrize("shift", [math.tau, 1e-20])
def test_exact_phase_effect_contract_rejects_full_turn_or_rounded_noop(shift):
    graph = _graph()
    graph.nodes["core"]["theta"] = 0.5
    graph.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift": shift}
    source = _record(graph)
    before = _plain(graph)
    with pytest.raises(ValueError):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


def test_fixed_tiny_phase_effect_has_no_arbitrary_size_cutoff():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift": 1e-20}
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    assert result.after.member_states[0].phase == 1e-20
    assert result.stage_result.mutation_decision_observations[0].fixed_mode


@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_VF, ALIAS_DNFR])
def test_raw_nonphase_alias_corruption_after_real_stage_rolls_back(
    monkeypatch, aliases
):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    original = implementation.execute_pointwise_stage

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        graph.nodes["core"][aliases[-1]] = (
            "changed" if aliases == ALIAS_EPI_KIND else 0.375
        )
        return result

    monkeypatch.setattr(implementation, "execute_pointwise_stage", corrupt)
    with pytest.raises(RuntimeError, match="protected"):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["non_target", "peer", "edge"])
def test_whole_non_target_state_and_ports_survive_or_roll_back(monkeypatch, change):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    original = implementation.execute_pointwise_stage

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "edge":
            graph.edges["member", "core"]["weight"] = 0.5
        else:
            node = "member" if change == "non_target" else "peer"
            graph.nodes[node]["private_memory"]["unexpected"] = True
        return result

    monkeypatch.setattr(implementation, "execute_pointwise_stage", corrupt)
    with pytest.raises(RuntimeError, match="protected"):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


def test_final_observation_failure_restores_real_phase_history_and_telemetry(
    monkeypatch,
):
    graph = _graph()
    source = _record(graph)
    before = _plain(graph)
    original = implementation.capture_retained_organization

    def fail_after_capture(*args, **kwargs):
        record = original(*args, **kwargs)
        assert record.member_states[0].grammar.history[-1] == "ZHIR"
        assert record.member_states[0].phase == 0.25
        raise RuntimeError("synthetic final observation failure")

    monkeypatch.setattr(
        implementation, "capture_retained_organization", fail_after_capture
    )
    with pytest.raises(RuntimeError, match="final observation failure"):
        execute_retained_mutation_stage(graph, source, ("core",))
    assert _plain(graph) == before


def test_summary_rejects_wrong_target_phase_context_and_non_target_state():
    graph = _graph()
    result = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    with pytest.raises(ValueError, match="non-target"):
        replace(result, targets=("member",))
    graph.nodes["core"]["theta"] = 0.5
    with pytest.raises(ValueError, match="decision"):
        replace(result, after=_record(graph))
    graph.nodes["core"]["theta"] = 0.25
    graph.nodes["core"]["_operator_step"] = 5
    with pytest.raises(ValueError, match="grammar"):
        replace(result, after=_record(graph))
    graph.nodes["core"]["_operator_step"] = 4
    graph.nodes["member"]["private_memory"]["changed"] = True
    with pytest.raises(ValueError, match="non-target"):
        replace(result, after=_record(graph))


def test_public_summary_is_not_authentication_of_the_sampled_history():
    graph = _graph()
    first = execute_retained_mutation_stage(graph, _record(graph), ("core",))
    other = _graph()
    other.nodes["core"]["epi_time_history"] = [(0.0, 0.25), (0.5, 0.375), (1.0, 0.5)]
    second = execute_retained_mutation_stage(other, _record(other), ("core",))
    assert first.before.member_states[0].epi == second.before.member_states[0].epi
    assert (
        first.before.member_states[0].temporal_context
        != second.before.member_states[0].temporal_context
    )
    first_certificate = first.stage_result.mutation_decision_observations[
        0
    ].trigger_certificate
    second_certificate = second.stage_result.mutation_decision_observations[
        0
    ].trigger_certificate
    assert first_certificate.observed_depi_dt == 0.125
    assert second_certificate.observed_depi_dt == 0.25
    # Public construction checks compatible fields, not opaque-history origin.
    mixed = replace(first, stage_result=second.stage_result)
    assert mixed.before is first.before and mixed.stage_result is second.stage_result
