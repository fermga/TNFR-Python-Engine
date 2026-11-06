"""Actual native consequences of explicit no-action Mutation abstention.

Stored Si, pressure and own samples are declared inputs. Real IL/OZ preparation
acquires grammar; no source acquisition, refresh law or autonomous clock is
inferred. These finite local controls do not replay a scientific producer.
"""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics.selectors import DefaultGlyphSelector, apply_native_operator_batch
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.validation import StructuralGrammarError

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
POLICY = "NATIVE_ZHIR_ABSTENTION_POLICY"
LEDGER = "_native_zhir_consumed_intervals"
INTERVAL = ("epi_time_history", 0.0, 1.0)


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _cohere_core(graph):
    execute_pointwise_stage(
        graph,
        Coherence(),
        (CORE,),
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )


def _source(policy=None, *, positive=False, stabilize=False, consume=False):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=29,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=False,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        ZHIR_THRESHOLD_XI=0.0625,
        glyph_selector=DefaultGlyphSelector(),
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        ZHIR_theta_shift=0.25,
    )
    if policy is not None:
        graph.graph[POLICY] = policy
    if consume:
        graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = "consume_timestamped_interval"
    for node, form, pressure, sense in (
        (CORE, 0.25, 1 / 32, 0.0),
        (PEER, 0.75, 0.5, 1.0),
    ):
        graph.nodes[node].update(
            EPI=form,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            Si=sense,
            d2EPI=0.0,
            epi_kind="form",
            glyph_history=[],
        )
    # Supplied temporal observations, not attributed to the preparation word.
    graph.nodes[CORE]["epi_time_history"] = [
        (0.0, 0.125 if positive else 0.25),
        (1.0, 0.25),
    ]
    samples = deepcopy(graph.nodes[CORE]["epi_time_history"])
    _cohere_core(graph)
    assert graph.nodes[CORE]["delta_nfr"] == 1 / 64
    execute_dissonance_stage(
        graph,
        Dissonance(),
        (CORE,),
        propagate_to_network=False,
        validate_preconditions=True,
    )
    assert graph.nodes[CORE]["delta_nfr"] == 1 / 32
    assert graph.nodes[PEER]["delta_nfr"] == 0.5
    grammar = read_grammar_state(graph.nodes[CORE])
    assert grammar.history == ("IL", "OZ")
    assert grammar.u2_debt == 1 and grammar.prior_coherence
    if stabilize:
        _cohere_core(graph)
        assert read_grammar_state(graph.nodes[CORE]).history == ("IL", "OZ", "IL")
    assert graph.nodes[CORE]["epi_time_history"] == samples
    # The first two skipped requests stay below the unchanged default cut;
    # the third can cross it solely because the peer's real IL lowers Pmax.
    assert 1 / 8 < graph.graph["SELECTOR_THRESHOLDS"]["dnfr_hi"] < 1 / 4
    assert graph.graph["AL_MAX_LAG"] == 5
    assert graph.graph["EN_MAX_LAG"] == 3
    return graph


def _batch(graph, memory):
    trace = []
    apply_native_operator_batch(graph, memory, decision_trace=trace)
    assert tuple(item.node for item in trace) == NODES
    assert graph.graph["_t"] == 1.0
    return trace


@pytest.mark.parametrize("policy", [None, "coherence", "skip"])
def test_own_noncrossing_request_distinguishes_il_action_from_no_action(policy):
    graph = _source(policy)
    before = deepcopy(dict(graph.nodes[CORE]))
    prior = read_grammar_state(graph.nodes[CORE])
    memory = {}
    core, peer = _batch(graph, memory)
    skips = policy == "skip"
    assert core.selection.base_candidate == core.requested_glyph == "ZHIR"
    assert core.selection.normalized_metrics == (0.0, 1 / 16, 0.0)
    assert core.mutation_abstention_policy == (policy or "coherence")
    assert core.mutation_abstention_reason == "observed_threshold_not_crossed"
    assert core.mutation_certificate.observed_depi_dt == 0.0
    assert core.status == ("abstained" if skips else "applied")
    assert core.applied_glyph == core.post_gate_glyph == (None if skips else "IL")
    assert graph.nodes[CORE]["delta_nfr"] == (1 / 32 if skips else 1 / 64)
    assert graph.nodes[PEER]["delta_nfr"] == 0.25
    assert peer.applied_glyph == "IL"
    current = read_grammar_state(graph.nodes[CORE])
    assert current.u2_debt == (1 if skips else 0)
    assert current.operator_step == prior.operator_step + (0 if skips else 1)
    assert current.history == (prior.history if skips else (*prior.history, "IL"))
    assert current.prior_coherence
    assert graph.nodes[CORE]["epi_time_history"] == before["epi_time_history"]
    assert tuple(_form(graph, node) for node in NODES) == (0.25, 0.75)
    assert tuple(graph.nodes[node]["theta"] for node in NODES) == (0.0, 0.0)
    assert memory["since_AL"] == memory["since_EN"] == dict.fromkeys(NODES, 1)
    assert memory["mutation_abstentions"][-1]["applied_glyph"] == core.applied_glyph
    assert LEDGER not in graph.nodes[CORE]
    if skips:
        assert core.grammar_decision is None
        assert core.physical_history_invalidated is None
        assert dict(graph.nodes[CORE]) == before
    else:
        assert core.grammar_decision.outcome == "accepted"


def test_real_prior_actions_change_the_third_native_source_choice():
    legacy, quiet = _source(), _source("skip")
    legacy_memory, quiet_memory = {}, {}
    for ordinal in (1, 2):
        legacy_trace = _batch(legacy, legacy_memory)
        quiet_trace = _batch(quiet, quiet_memory)
        assert (
            legacy_trace[0].requested_glyph == quiet_trace[0].requested_glyph == "ZHIR"
        )
        assert legacy_trace[0].applied_glyph == "IL"
        assert quiet_trace[0].status == "abstained"
        assert legacy.nodes[CORE]["delta_nfr"] == 1 / (32 * 2**ordinal)
        assert quiet.nodes[CORE]["delta_nfr"] == 1 / 32
        assert legacy.nodes[PEER]["delta_nfr"] == quiet.nodes[PEER]["delta_nfr"]

    legacy_core, _ = _batch(legacy, legacy_memory)
    quiet_core, _ = _batch(quiet, quiet_memory)
    assert legacy_core.selection.normalized_metrics == (0.0, 1 / 16, 0.0)
    assert quiet_core.selection.normalized_metrics == (0.0, 1 / 4, 0.0)
    assert legacy_core.requested_glyph == "ZHIR" and legacy_core.applied_glyph == "IL"
    assert quiet_core.requested_glyph == quiet_core.applied_glyph == "OZ"
    assert quiet_core.mutation_certificate is None
    assert legacy.nodes[CORE]["delta_nfr"] == 1 / 256
    assert quiet.nodes[CORE]["delta_nfr"] == 1 / 16
    assert read_grammar_state(quiet.nodes[CORE]).history == ("IL", "OZ", "OZ")
    assert read_grammar_state(quiet.nodes[CORE]).u2_debt == 2
    assert (
        legacy_memory["since_AL"] == quiet_memory["since_AL"] == dict.fromkeys(NODES, 3)
    )
    assert len(legacy_memory["mutation_abstentions"]) == 3
    assert len(quiet_memory["mutation_abstentions"]) == 2


def test_a_skipped_attempt_still_advances_existing_al_before_en_competition():
    graph = _source("skip")
    # Supplied policy memory, not a reconstructed acquisition trajectory.
    memory = {"since_AL": {CORE: 4}, "since_EN": {CORE: 2}}
    first, _ = _batch(graph, memory)
    assert first.status == "abstained"
    assert (memory["since_AL"][CORE], memory["since_EN"][CORE]) == (5, 3)
    second, _ = _batch(graph, memory)
    assert second.lag_reason == "al_lag"
    assert second.requested_glyph == second.applied_glyph == "AL"
    assert not second.selector_evaluated and second.selection is None
    assert second.mutation_certificate is None
    assert _form(graph, CORE) == 3 / 8
    assert (memory["since_AL"][CORE], memory["since_EN"][CORE]) == (0, 3)
    third, _ = _batch(graph, memory)
    assert third.lag_reason == "en_lag"
    assert third.requested_glyph == third.applied_glyph == "EN"
    assert not third.selector_evaluated and third.selection is None
    assert third.mutation_certificate is None
    assert _form(graph, CORE) == 9 / 16
    assert (memory["since_AL"][CORE], memory["since_EN"][CORE]) == (1, 0)
    assert read_grammar_state(graph.nodes[CORE]).history == ("IL", "OZ", "AL", "EN")
    assert len(memory["mutation_abstentions"]) == 1


@pytest.mark.parametrize("policy", ["coherence", "skip"])
def test_admitted_native_mutation_then_consumed_interval_obeys_selected_policy(policy):
    graph = _source(policy, positive=True, consume=True)
    memory = {}
    first, _ = _batch(graph, memory)
    assert first.requested_glyph == first.applied_glyph == "ZHIR"
    assert first.status == "applied" and first.mutation_interval_committed
    assert first.mutation_certificate.observed_depi_dt == 0.125
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[CORE][LEDGER] == (INTERVAL,)
    prior = read_grammar_state(graph.nodes[CORE])
    second, _ = _batch(graph, memory)
    assert second.requested_glyph == "ZHIR"
    assert second.mutation_abstention_reason == "mutation_interval_already_consumed"
    assert second.applied_glyph == (None if policy == "skip" else "IL")
    assert not second.mutation_interval_committed
    assert graph.nodes[CORE][LEDGER] == (INTERVAL,)
    assert graph.nodes[CORE]["theta"] == 0.25
    current = read_grammar_state(graph.nodes[CORE])
    assert current.operator_step == prior.operator_step + (policy == "coherence")
    assert current.u2_debt == (2 if policy == "skip" else 1)
    assert memory["since_AL"][CORE] == memory["since_EN"][CORE] == 2


def test_no_action_retains_context_that_real_il_attempts_age_out():
    legacy, quiet = _source(stabilize=True), _source("skip", stabilize=True)
    legacy_memory, quiet_memory = {}, {}
    for _ in range(2):
        assert _batch(legacy, legacy_memory)[0].applied_glyph == "IL"
        assert _batch(quiet, quiet_memory)[0].status == "abstained"
    assert read_grammar_state(legacy.nodes[CORE]).history[-3:] == ("IL", "IL", "IL")
    assert read_grammar_state(quiet.nodes[CORE]).history == ("IL", "OZ", "IL")
    # A declared new observation intervention; these samples are not claimed
    # to have been generated by the previous zero-duration operator batches.
    for graph in (legacy, quiet):
        graph.nodes[CORE]["epi_time_history"] = [(0.0, 0.125), (1.0, 0.25)]
    rejected = []
    with pytest.raises(StructuralGrammarError):
        apply_native_operator_batch(legacy, legacy_memory, decision_trace=rejected)
    assert rejected[0].requested_glyph == "ZHIR"
    assert rejected[0].mutation_certificate.threshold_gate_satisfied
    assert rejected[0].grammar_decision.outcome == "rejected"
    assert rejected[0].applied_glyph is None
    accepted, _ = _batch(quiet, quiet_memory)
    assert accepted.selection.normalized_metrics == (0.0, 1 / 8, 0.0)
    assert accepted.requested_glyph == accepted.applied_glyph == "ZHIR"
    assert quiet.nodes[CORE]["theta"] == 0.25
    assert legacy.nodes[CORE]["theta"] == 0.0
