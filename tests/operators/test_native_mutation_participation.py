"""Native state-derived requests with supplied own evidence and finite EN/RA."""

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
from tnfr.operators._mutation_gate import mutation_trigger_certificate
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
    observe_reception_compatibility,
)
from tnfr.operators.resonance import Resonance
from tnfr.types import require_finite_real_scalar_epi
from tnfr.validation import StructuralGrammarError

CORE, MEMBER, PEER, ANCHOR = "core", "member", "external", "anchor"
NODES = (CORE, MEMBER, PEER, ANCHOR)


def _form(graph, node):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _phase(graph, node):
    return get_attr(graph.nodes[node], ALIAS_THETA)


def _primitive_state(graph):
    return tuple(
        (
            node,
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
            for a, b, key, data in graph.edges(data=True, keys=True)
        )
    return tuple((a, b, deepcopy(data)) for a, b, data in graph.edges(data=True))


def _source(graph_type=nx.DiGraph, *, aligned=False, rotation=0.0, prepare=True):
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
        (ANCHOR, 0.5, 1.0, 0.0 if aligned else math.pi),
    ):
        graph.add_node(
            node,
            EPI=form,
            nu_f=capacity,
            theta=(phase + rotation) % math.tau,
            delta_nfr=0.0,  # Supplied held driver; no pressure refresh is claimed.
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
    # Supplied own measured history, not manufactured by the preparation word.
    graph.nodes[CORE]["epi_time_history"] = [
        (0.0, 0.25),
        (1.0, 0.375),
        (2.0, 0.5),
    ]
    if prepare:
        before = _primitive_state(graph)
        own_samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
        execute_pointwise_stage(
            graph,
            Coherence(),
            (CORE,),
            phase_locking_coefficient=0.0,
            require_exact_operator=True,
            validate_preconditions=True,
        )
        execute_dissonance_stage(
            graph,
            Dissonance(),
            (CORE,),
            propagate_to_network=False,
            validate_preconditions=True,
        )
        # OZ's actual zero-pressure branch seeds 0.1; a declared IL factor zero
        # then closes the source at +0 without editing pressure or own samples.
        assert get_attr(graph.nodes[CORE], ALIAS_DNFR) == 0.1
        execute_pointwise_stage(
            graph,
            Coherence(),
            (CORE,),
            phase_locking_coefficient=0.0,
            require_exact_operator=True,
            validate_preconditions=True,
        )
        assert _primitive_state(graph) == before
        assert graph.nodes[CORE]["epi_time_history"] == own_samples
        context = read_grammar_state(graph.nodes[CORE])
        assert context.history == ("IL", "OZ", "IL")
        assert context.u2_debt == 0 and context.prior_coherence
    compute_Si(graph, inplace=True)
    return graph


def _receive_and_resonate(graph):
    policy = ReceptionCompatibilityPolicy(0.125, 1.0)
    received = execute_eligible_reception_stage(
        graph,
        CORE,
        policy,
        execution_kwargs={"validate_preconditions": True},
    )
    assert received.executed
    execute_neighbor_stage(graph, Resonance(), (PEER,), validate_preconditions=True)
    return (
        received.compatibility.selected_sources,
        _form(graph, CORE),
        _form(graph, PEER),
    )


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("aligned", [False, True])
def test_fresh_relational_selector_executes_native_request_then_en_ra(
    graph_type, aligned
):
    graph = _source(graph_type, aligned=aligned)
    before, edges = _primitive_state(graph), _support(graph)
    own_samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
    certificate = mutation_trigger_certificate(graph.nodes[CORE], graph.graph)
    assert certificate.observed_depi_dt == 0.125
    assert certificate.threshold_gate_satisfied
    assert (
        get_attr(graph.nodes[CORE], ALIAS_VF) * get_attr(graph.nodes[CORE], ALIAS_DNFR)
        == 0.0
    )
    alpha = math.pi / (math.pi + 1.0)
    beta = math.pi / (math.pi + 1.0) ** 2
    gamma = 1.0 / (math.pi + 1.0) ** 2
    expected_si = alpha / 8 + gamma + (beta if aligned else 0)
    assert graph.nodes[CORE]["Si"] == pytest.approx(expected_si)
    assert all(graph.nodes[node]["Si"] > 0.5 for node in NODES[1:])
    assert (0.25 < expected_si < 0.5) if aligned else (expected_si < 0.25)
    expected_mark = "RA" if aligned else "ZHIR"
    assert DefaultGlyphSelector().select(graph, CORE) == expected_mark

    trace, history = [], {}
    assert apply_native_operator_batch(graph, history, decision_trace=trace) is None
    assert tuple(item.node for item in trace) == NODES
    assert tuple(item.applied_glyph for item in trace) == (
        expected_mark,
        "IL",
        "IL",
        "IL",
    )
    assert all(item.status == "applied" for item in trace)
    core = trace[0]
    assert core.selector_evaluated and core.lag_reason is None
    assert core.selection.base_candidate == expected_mark
    assert core.requested_glyph == core.post_gate_glyph == expected_mark
    assert core.grammar_decision.outcome == "accepted"
    assert core.mutation_abstention_reason is None
    assert (core.mutation_certificate is None) is aligned
    assert graph.graph["_t"] == 2.0
    assert graph.nodes[CORE]["epi_time_history"] == own_samples
    assert _support(graph) == edges
    after = _primitive_state(graph)
    assert after[1:] == before[1:]
    assert after[0][:2] == before[0][:2]
    assert after[0][2] == (0.25 if aligned else 0.125)
    assert after[0][4:] == before[0][4:]
    assert _phase(graph, CORE) == (0.0 if aligned else 0.25)
    assert read_grammar_state(graph.nodes[CORE]).u2_debt == (0 if aligned else 1)

    sources, core_form, peer_form = _receive_and_resonate(graph)
    assert sources == ((MEMBER,) if aligned else (PEER,))
    assert core_form == (7 / 16 if aligned else 9 / 16)
    assert peer_form == (19 / 32 if aligned else 21 / 32)
    assert _support(graph) == edges
    assert tuple(get_attr(graph.nodes[n], ALIAS_VF) for n in graph) == (
        0.25 if aligned else 0.125,
        1,
        2,
        1,
    )


def test_trace_is_passive_for_the_same_native_source():
    plain, traced = _source(), _source()
    plain_history, traced_history, trace = {}, {}, []
    apply_native_operator_batch(plain, plain_history)
    apply_native_operator_batch(traced, traced_history, decision_trace=trace)
    assert _primitive_state(plain) == _primitive_state(traced)
    assert _support(plain) == _support(traced)
    assert plain_history == traced_history
    assert tuple(read_grammar_state(plain.nodes[n]) for n in plain) == tuple(
        read_grammar_state(traced.nodes[n]) for n in traced
    )
    assert _receive_and_resonate(plain) == _receive_and_resonate(traced)


@pytest.mark.parametrize(
    "own_samples", [[(2.0, 0.5)], [(0.0, 0.25), (1.0, 0.375)], [(1.0, 0.5), (2.0, 0.5)]]
)
def test_own_evidence_blocks_action_without_changing_relational_zhir_proposal(
    own_samples,
):
    graph = _source()
    graph.nodes[MEMBER]["epi_time_history"] = [(0.0, 0.0), (1.0, 0.125), (2.0, 0.25)]
    graph.nodes[CORE]["epi_time_history"] = own_samples
    trace, history = [], {}
    apply_native_operator_batch(graph, history, decision_trace=trace)
    core = trace[0]
    assert core.selection.base_candidate == core.requested_glyph == "ZHIR"
    assert core.post_gate_glyph == core.applied_glyph == "IL"
    assert core.mutation_abstention_reason
    assert not core.mutation_certificate.threshold_gate_satisfied
    assert _phase(graph, CORE) == 0.0
    assert graph.nodes[CORE]["glyph_history"][-1] == "IL"
    assert history["mutation_abstentions"][-1]["node"] == CORE
    assert _receive_and_resonate(graph) == ((MEMBER,), 7 / 16, 19 / 32)


def test_inactive_capacity_is_a_live_gate_not_a_changed_zhir_request():
    graph = _source()
    graph.nodes[CORE]["nu_f"] = 0.0
    compute_Si(graph, inplace=True)
    trace = []
    apply_native_operator_batch(graph, {}, decision_trace=trace)
    core = trace[0]
    assert core.requested_glyph == "ZHIR"
    assert core.mutation_abstention_reason == "inactive_structural_capacity"
    assert core.post_gate_glyph == core.applied_glyph == "IL"
    assert _phase(graph, CORE) == 0.0


@pytest.mark.parametrize("mode", ["raise", "fallback"])
def test_temporally_admitted_request_still_requires_own_grammar(mode):
    graph = _source(prepare=False)
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    trace, history = [], {}
    if mode == "raise":
        with pytest.raises(StructuralGrammarError):
            apply_native_operator_batch(graph, history, decision_trace=trace)
    else:
        apply_native_operator_batch(graph, history, decision_trace=trace)
    core = trace[0]
    assert core.requested_glyph == core.post_gate_glyph == "ZHIR"
    assert core.mutation_certificate.threshold_gate_satisfied
    assert core.mutation_abstention_reason is None
    assert core.grammar_decision.outcome == (
        "rejected" if mode == "raise" else "fallback"
    )
    assert core.applied_glyph == (None if mode == "raise" else "IL")
    assert _phase(graph, CORE) == 0.0
    assert history["since_AL"][CORE] == history["since_EN"][CORE] == 1
    if mode == "raise":
        assert core.status == "failed"
        assert all(item.status == "not_reached" for item in trace[1:])
        assert all(not graph.nodes[n]["glyph_history"] for n in graph)


@pytest.mark.parametrize(
    "force_al,force_en,expected",
    [(True, False, "AL"), (False, True, "EN"), (True, True, "AL")],
)
def test_forced_lag_does_not_report_an_evaluated_zhir_choice(
    force_al, force_en, expected
):
    graph = _source()
    history = {
        "since_AL": {CORE: graph.graph["AL_MAX_LAG"] if force_al else 0},
        "since_EN": {CORE: graph.graph["EN_MAX_LAG"] if force_en else 0},
    }
    trace = []
    apply_native_operator_batch(graph, history, decision_trace=trace)
    core = trace[0]
    assert core.requested_glyph == core.applied_glyph == expected
    assert not core.selector_evaluated and core.selection is None
    assert core.lag_reason is not None and core.mutation_certificate is None
    assert _phase(graph, CORE) == 0.0
    assert graph.nodes[CORE]["glyph_history"][-1] == expected


@pytest.mark.parametrize("rotation", [0.125, -0.125])
def test_common_rotation_preserves_the_finite_relational_response(rotation):
    graph = _source(rotation=rotation)
    original_support = _support(graph)
    trace = []
    apply_native_operator_batch(graph, {}, decision_trace=trace)
    assert trace[0].requested_glyph == trace[0].applied_glyph == "ZHIR"
    assert _phase(graph, CORE) == pytest.approx((rotation + 0.25) % math.tau)
    assert _receive_and_resonate(graph) == ((PEER,), 9 / 16, 21 / 32)
    assert _support(graph) == original_support


def test_two_bounded_native_calls_advance_lags_without_advancing_time():
    graph, history = _source(aligned=True), {}
    samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
    before = _primitive_state(graph)
    for expected_lag in (1, 2):
        compute_Si(graph, inplace=True)
        trace = []
        apply_native_operator_batch(graph, history, decision_trace=trace)
        assert trace[0].requested_glyph == trace[0].applied_glyph == "RA"
        assert trace[0].since_al == trace[0].since_en == expected_lag
        assert graph.graph["_t"] == 2.0
        assert graph.nodes[CORE]["epi_time_history"] == samples
        after = _primitive_state(graph)
        assert after[1:] == before[1:]
        assert after[0][:2] == before[0][:2]
        assert after[0][2] == 0.125 * 2**expected_lag
        assert after[0][3:] == before[0][3:]


def test_outgoing_anchor_is_not_an_incoming_en_donor():
    graph = _source()
    observed = observe_reception_compatibility(
        graph, CORE, ReceptionCompatibilityPolicy(math.pi, 1.0)
    )
    assert observed.selected_sources == (MEMBER, PEER)
    assert ANCHOR not in observed.selected_sources
