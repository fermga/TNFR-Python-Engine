"""Bounded native interval-policy controls with supplied own observations.

The four-node fixture is defined here, not imported from another test module.
It retains the corrected actual preparation of the prior closed native study.
No source measurement is attributed to an operator jump or a simulated flow.
"""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.selectors import DefaultGlyphSelector, apply_native_operator_batch
from tnfr.metrics.sense_index import compute_Si
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.reception_selection import (
    ReceptionCompatibilityPolicy,
    execute_eligible_reception_stage,
)
from tnfr.operators.resonance import Resonance
from tnfr.types import require_finite_real_scalar_epi
from tnfr.validation import StructuralGrammarError

CORE, MEMBER, PEER, ANCHOR = "core", "member", "external", "anchor"
NODES = (CORE, MEMBER, PEER, ANCHOR)
POLICY = "consume_timestamped_interval"
LEDGER = "_native_zhir_consumed_intervals"
INTERVAL_A = ("epi_time_history", 1.0, 2.0)
INTERVAL_B = ("epi_time_history", 3.0, 4.0)


def _form(graph, node):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _phase(graph, node):
    return get_attr(graph.nodes[node], ALIAS_THETA)


def _coordinates(graph):
    return tuple(
        (
            _form(graph, node),
            get_attr(graph.nodes[node], ALIAS_VF),
            _phase(graph, node),
            get_attr(graph.nodes[node], ALIAS_DNFR),
            graph.nodes[node]["epi_kind"],
        )
        for node in graph
    )


def _support(graph):
    if graph.is_multigraph():
        return tuple(
            (a, b, key, deepcopy(data))
            for a, b, key, data in graph.edges(keys=True, data=True)
        )
    return tuple((a, b, deepcopy(data)) for a, b, data in graph.edges(data=True))


def _prepare_core(graph):
    before = _coordinates(graph)
    samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
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
    assert graph.nodes[CORE]["epi_time_history"] == samples
    context = read_grammar_state(graph.nodes[CORE])
    assert context.history[-3:] == ("IL", "OZ", "IL")
    assert context.u2_debt == 0 and context.prior_coherence


def _source(graph_type=nx.DiGraph, *, policy=POLICY, prepare=True):
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
        IL_dnfr_factor=0.0,
        OZ_dnfr_factor=2.0,
        EN_mix=0.25,
        RA_epi_diff=0.5,
        RA_phase_coupling=0.5,
        RA_vf_amplification=1.0,
    )
    for node, form, capacity, phase in (
        (CORE, 0.5, 0.125, 0.0),
        (MEMBER, 0.25, 1.0, 0.0),
        (PEER, 0.75, 1.0, 0.25),
        (ANCHOR, 0.5, 1.0, math.pi),
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
    graph.add_edge(MEMBER, CORE, weight=0.375, role="incoming-member")
    graph.add_edge(PEER, CORE, weight=0.625, role="incoming-external")
    graph.add_edge(CORE, ANCHOR, weight=0.5, role="outgoing-diagnostic")
    if graph.is_multigraph():
        graph.add_edge(MEMBER, CORE, weight=0.0, role="parallel-zero")
    graph.nodes[CORE]["epi_time_history"] = [(0.0, 0.25), (1.0, 0.375), (2.0, 0.5)]
    if prepare:
        _prepare_core(graph)
    return graph


def _batch(graph, history):
    # Actual fresh observation, using declared held zero pressure.
    compute_Si(graph, inplace=True)
    trace = []
    apply_native_operator_batch(graph, history, decision_trace=trace)
    assert tuple(item.node for item in trace) == NODES
    assert trace[0].selection.base_candidate == "ZHIR"
    assert all(item.applied_glyph == "IL" for item in trace[1:])
    assert all(item.status == "applied" for item in trace)
    return trace[0]


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("policy", [None, "level", POLICY])
def test_two_actual_native_requests_distinguish_occurrence_and_en_availability(
    graph_type, policy
):
    graph, history = _source(graph_type, policy=policy), {}
    before, edges = _coordinates(graph), _support(graph)
    samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
    first = _batch(graph, history)
    assert first.requested_glyph == first.applied_glyph == "ZHIR"
    assert first.mutation_certificate.observed_depi_dt == 0.125
    assert _phase(graph, CORE) == 0.25
    second = _batch(graph, history)
    assert second.requested_glyph == "ZHIR"
    assert second.mutation_certificate.threshold_gate_satisfied
    assert second.since_al == second.since_en == 2
    assert graph.graph["_t"] == 2.0
    assert graph.nodes[CORE]["epi_time_history"] == samples
    assert _support(graph) == edges
    after = _coordinates(graph)
    assert after[1:] == before[1:]
    assert after[0][:2] == before[0][:2] and after[0][3:] == before[0][3:]

    consumes = policy == POLICY
    assert first.mutation_interval_committed is consumes
    assert not second.mutation_interval_committed
    if consumes:
        assert first.mutation_interval == second.mutation_interval == INTERVAL_A
        assert first.mutation_interval_previously_consumed is False
        assert second.mutation_interval_previously_consumed is True
        assert second.mutation_abstention_reason == "mutation_interval_already_consumed"
        assert second.post_gate_glyph == second.applied_glyph == "IL"
        assert graph.nodes[CORE][LEDGER] == (INTERVAL_A,)
        assert _phase(graph, CORE) == 0.25
    else:
        assert (
            first.mutation_occurrence_policy
            == second.mutation_occurrence_policy
            == "level"
        )
        assert second.applied_glyph == "ZHIR"
        assert second.mutation_abstention_reason is None
        assert LEDGER not in graph.nodes[CORE]
        assert _phase(graph, CORE) == 0.5

    own_context = read_grammar_state(graph.nodes[CORE])
    received = execute_eligible_reception_stage(
        graph,
        CORE,
        ReceptionCompatibilityPolicy(0.125, 1.0),
        execution_kwargs={"validate_preconditions": True},
    )
    if consumes:
        assert received.executed
        assert received.compatibility.selected_sources == (PEER,)
        assert _form(graph, CORE) == 9 / 16
        execute_neighbor_stage(graph, Resonance(), (PEER,), validate_preconditions=True)
        assert _form(graph, PEER) == 21 / 32
        assert get_attr(graph.nodes[PEER], ALIAS_VF) == 2.0
    else:
        assert not received.executed and received.stage_result is None
        assert received.compatibility.selected_sources == ()
        assert read_grammar_state(graph.nodes[CORE]) == own_context
        assert _form(graph, CORE) == 0.5 and _form(graph, PEER) == 0.75
    assert _support(graph) == edges


@pytest.mark.parametrize("previous_form", [0.25, 0.3125])
def test_revised_values_do_not_renew_the_same_interval(previous_form):
    graph, history = _source(), {}
    _batch(graph, history)
    graph.nodes[CORE]["epi_time_history"][-2] = (1.0, previous_form)
    decision = _batch(graph, history)
    assert decision.mutation_certificate.observed_depi_dt == 0.5 - previous_form
    assert decision.mutation_certificate.threshold_gate_satisfied
    assert decision.mutation_interval == INTERVAL_A
    assert decision.mutation_interval_previously_consumed
    assert decision.applied_glyph == "IL" and not decision.mutation_interval_committed
    assert graph.nodes[CORE][LEDGER] == (INTERVAL_A,)
    assert _phase(graph, CORE) == 0.25


def test_repackaged_samples_do_not_renew_the_interval():
    graph, history = _source(), {}
    _batch(graph, history)
    graph.nodes[CORE]["epi_time_history"] = (
        (-1.0, 0.125),
        *graph.nodes[CORE]["epi_time_history"],
    )
    decision = _batch(graph, history)
    assert decision.mutation_interval == INTERVAL_A
    assert decision.mutation_interval_previously_consumed
    assert decision.applied_glyph == "IL"
    assert graph.nodes[CORE][LEDGER] == (INTERVAL_A,)


def test_distinct_supplied_interval_is_admitted_and_old_interval_cannot_be_replayed():
    graph, history = _source(), {}
    _batch(graph, history)
    original_samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
    # Explicit new measured-history fixture; no event/flow generated these samples.
    graph.nodes[CORE]["epi_time_history"].extend([(3.0, 0.375), (4.0, 0.5)])
    graph.graph["_t"] = 4.0
    renewed = _batch(graph, history)
    assert renewed.mutation_certificate.observed_depi_dt == 0.125
    assert renewed.mutation_interval == INTERVAL_B
    assert renewed.mutation_interval_previously_consumed is False
    assert renewed.mutation_interval_committed and renewed.applied_glyph == "ZHIR"
    assert graph.nodes[CORE][LEDGER] == (INTERVAL_A, INTERVAL_B)
    assert _phase(graph, CORE) == 0.5

    graph.nodes[CORE]["epi_time_history"] = original_samples
    replayed = _batch(graph, history)
    assert replayed.requested_glyph == "ZHIR"
    assert replayed.mutation_interval == INTERVAL_A
    assert replayed.mutation_interval_previously_consumed
    assert replayed.applied_glyph == "IL" and not replayed.mutation_interval_committed
    assert graph.nodes[CORE][LEDGER] == (INTERVAL_A, INTERVAL_B)
    assert _phase(graph, CORE) == 0.5


def test_grammar_failure_leaves_interval_available_for_actual_prepared_retry():
    graph, history, trace = _source(prepare=False), {}, []
    compute_Si(graph, inplace=True)
    with pytest.raises(StructuralGrammarError):
        apply_native_operator_batch(graph, history, decision_trace=trace)
    assert trace[0].requested_glyph == "ZHIR"
    assert trace[0].mutation_certificate.threshold_gate_satisfied
    assert trace[0].applied_glyph is None
    assert not trace[0].mutation_interval_committed
    assert LEDGER not in graph.nodes[CORE]
    assert _phase(graph, CORE) == 0.0
    _prepare_core(graph)
    retried = _batch(graph, history)
    assert retried.mutation_interval == INTERVAL_A
    assert retried.mutation_interval_committed and retried.applied_glyph == "ZHIR"
    assert graph.nodes[CORE][LEDGER] == (INTERVAL_A,)
    assert _phase(graph, CORE) == 0.25
