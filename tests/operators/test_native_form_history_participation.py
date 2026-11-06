"""Finite native jump-boundary controls with supplied own observations.

The four-node source is self-contained. No flow, trajectory producer, inherited
measurements or imports of another test module prepare its temporal evidence.
"""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.runtime import _record_mutation_flow_boundary
from tnfr.dynamics.selectors import DefaultGlyphSelector, apply_native_operator_batch
from tnfr.metrics.sense_index import compute_Si
from tnfr.operators._mutation_gate import mutation_trigger_certificate
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.types import require_finite_real_scalar_epi

CORE, MEMBER, PEER, ANCHOR = "core", "member", "external", "anchor"
NODES = (CORE, MEMBER, PEER, ANCHOR)
POLICY = "consume_timestamped_interval"
LEDGER = "_native_zhir_consumed_intervals"


def _form(graph, node):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _coordinates(graph):
    return tuple(
        (
            _form(graph, node),
            get_attr(graph.nodes[node], ALIAS_VF),
            get_attr(graph.nodes[node], ALIAS_THETA),
            get_attr(graph.nodes[node], ALIAS_DNFR),
        )
        for node in graph
    )


def _prepare_core(graph):
    """Real grammar preparation, preserving existing supplied EPI samples."""
    before = _coordinates(graph)
    raw_history = graph.nodes[CORE].get("epi_time_history")
    for step in ("IL", "OZ", "IL"):
        if step == "OZ":
            execute_dissonance_stage(
                graph,
                Dissonance(),
                (CORE,),
                propagate_to_network=False,
                validate_preconditions=True,
            )
            assert get_attr(graph.nodes[CORE], ALIAS_DNFR) == 0.1
        else:
            execute_pointwise_stage(
                graph,
                Coherence(),
                (CORE,),
                phase_locking_coefficient=0.0,
                require_exact_operator=True,
                validate_preconditions=True,
            )
    assert _coordinates(graph) == before
    assert graph.nodes[CORE].get("epi_time_history") is raw_history
    context = read_grammar_state(graph.nodes[CORE])
    assert context.history[-3:] == ("IL", "OZ", "IL")
    assert context.prior_coherence and context.u2_debt == 0


def _source(graph_type=nx.DiGraph, *, policy=POLICY, aligned=False):
    graph = graph_type()
    inject_defaults(graph)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        EPI_LATENT_MAX=0.875,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        _t=2.0,
        RANDOM_SEED=19,
        glyph_selector=DefaultGlyphSelector(),
    )
    if policy is not None:
        graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = policy
    else:
        graph.graph.pop("NATIVE_ZHIR_OCCURRENCE_POLICY", None)
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.25,
        IL_dnfr_factor=0.0,
        OZ_dnfr_factor=2.0,
        EN_mix=1.0,
        RA_epi_diff=0.5,
        RA_phase_coupling=0.5,
        RA_vf_amplification=1.0,
    )
    for node, form, capacity, phase in (
        (CORE, 0.5, 0.125, 0.0),
        (MEMBER, 0.25, 1.0, 0.0),
        (PEER, 0.75, 1.0, 0.25),
        (ANCHOR, 0.5, 1.0, 0.0 if aligned else math.pi),
    ):
        graph.add_node(
            node,
            EPI=form,
            nu_f=capacity,
            theta=phase,
            delta_nfr=0.0,
            epi_kind="same-form",
            glyph_history=[],
            _grammar_u2_debt=0,
            _grammar_prior_coherence=False,
            _operator_step=0,
        )
    graph.add_edge(MEMBER, CORE, weight=0.375)
    graph.add_edge(PEER, CORE, weight=0.625)
    graph.add_edge(CORE, ANCHOR, weight=0.5)
    if graph.is_multigraph():
        graph.add_edge(MEMBER, CORE, weight=0.0)
    graph.nodes[CORE]["epi_time_history"] = [(0.0, 0.25), (1.0, 0.375), (2.0, 0.5)]
    # A positive legacy fallback makes deletion/None an observable wrong fix.
    graph.nodes[CORE]["epi_history"] = [0.25, 0.5]
    _prepare_core(graph)
    return graph


def _batch(graph, history):
    compute_Si(graph, inplace=True)
    trace = []
    apply_native_operator_batch(graph, history, decision_trace=trace)
    assert tuple(item.node for item in trace) == NODES
    assert all(item.status == "applied" for item in trace)
    assert all(item.applied_glyph == "IL" for item in trace[1:])
    return trace[0]


def _round_trip(graph):
    history = {
        "since_AL": {CORE: graph.graph["AL_MAX_LAG"]},
        "since_EN": {CORE: graph.graph["EN_MAX_LAG"]},
    }
    first = _batch(graph, history)
    assert first.requested_glyph == first.applied_glyph == "AL"
    assert first.lag_reason == "al_lag" and not first.selector_evaluated
    assert _form(graph, CORE) == 0.75
    second = _batch(graph, history)
    assert second.requested_glyph == second.applied_glyph == "EN"
    assert second.lag_reason == "en_lag" and not second.selector_evaluated
    # EN consumes the two incoming nodes once each, ignoring edge multiplicity.
    assert _form(graph, CORE) == (0.25 + 0.75) / 2 == 0.5
    assert graph.graph["_t"] == 2.0
    return history


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("policy", [None, "level", POLICY])
def test_actual_native_round_trip_cannot_reactivate_pre_jump_growth(graph_type, policy):
    graph = _source(graph_type, policy=policy)
    before = _coordinates(graph)
    support = deepcopy(tuple(graph.edges(data=True)))
    old_samples = graph.nodes[CORE]["epi_time_history"]
    history = _round_trip(graph)
    # Restore only own grammar context. No EPI/time sample is supplied here.
    _prepare_core(graph)
    third = _batch(graph, history)
    assert third.selector_evaluated and third.selection.base_candidate == "ZHIR"
    assert third.requested_glyph == "ZHIR"
    assert third.applied_glyph == "IL"
    assert third.mutation_abstention_reason == "insufficient_history"
    assert third.mutation_certificate.source == "epi_time_history"
    assert not third.mutation_certificate.evidence_valid
    assert graph.nodes[CORE]["epi_time_history"] == ()
    assert old_samples == [(0.0, 0.25), (1.0, 0.375), (2.0, 0.5)]
    assert graph.nodes[CORE]["epi_history"] == [0.25, 0.5]
    assert LEDGER not in graph.nodes[CORE]
    assert _coordinates(graph) == before
    assert tuple(graph.edges(data=True)) == support


def test_phase_only_mutation_retains_the_exact_observation_object():
    graph = _source()
    samples = graph.nodes[CORE]["epi_time_history"]
    result = _batch(graph, {})
    assert result.applied_glyph == "ZHIR"
    assert result.mutation_interval_committed
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.25
    assert graph.nodes[CORE]["epi_time_history"] is samples
    assert _form(graph, CORE) == 0.5


def test_form_preserving_resonance_retains_samples_despite_capacity_change():
    graph = _source(aligned=True)
    samples = graph.nodes[CORE]["epi_time_history"]
    result = _batch(graph, {})
    assert result.requested_glyph == result.applied_glyph == "RA"
    assert _form(graph, CORE) == 0.5
    assert get_attr(graph.nodes[CORE], ALIAS_VF) == 0.25
    assert graph.nodes[CORE]["epi_time_history"] is samples


@pytest.mark.parametrize("history_value", ["absent", None])
def test_native_form_jump_does_not_create_a_physical_clock_or_history(history_value):
    graph = _source()
    graph.nodes[CORE].pop("epi_time_history")
    if history_value is None:
        graph.nodes[CORE]["epi_time_history"] = None
    graph.graph.pop("_t")
    history = {"since_AL": {CORE: graph.graph["AL_MAX_LAG"]}}
    assert _batch(graph, history).applied_glyph == "AL"
    assert _form(graph, CORE) == 0.75
    assert "_t" not in graph.graph
    if history_value == "absent":
        assert "epi_time_history" not in graph.nodes[CORE]
    else:
        assert graph.nodes[CORE]["epi_time_history"] is None


def test_live_recorder_starts_only_the_returned_right_endpoint_without_flow():
    graph = _source()
    _round_trip(graph)
    # This records the supplied existing clock; it advances no dynamics/time.
    _record_mutation_flow_boundary(graph)
    assert tuple(graph.nodes[CORE]["epi_time_history"]) == ((2.0, 0.5),)
    assert graph.graph["_t"] == 2.0
    certificate = mutation_trigger_certificate(graph.nodes[CORE], graph.graph)
    assert not certificate.evidence_valid
    assert certificate.reason == "insufficient_history"
    _record_mutation_flow_boundary(graph)
    assert tuple(graph.nodes[CORE]["epi_time_history"]) == ((2.0, 0.5),)


def test_single_native_jump_invalidates_before_an_outer_runtime_boundary():
    graph = _source()
    old_samples = graph.nodes[CORE]["epi_time_history"]
    assert (
        _batch(graph, {"since_AL": {CORE: graph.graph["AL_MAX_LAG"]}}).applied_glyph
        == "AL"
    )
    assert _form(graph, CORE) == 0.75
    assert graph.nodes[CORE]["epi_time_history"] == ()
    assert old_samples[-1] == (2.0, 0.5)
    assert graph.graph["_t"] == 2.0
