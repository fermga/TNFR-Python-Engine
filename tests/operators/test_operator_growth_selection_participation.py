"""Actual source-to-selection controls for conditional hybrid Mutation.

The source word and clock remain supplied. Target eligibility is computed from
the resulting own growth, capacity, grammar and retained consumption memory;
these finite controls do not claim autonomous source generation or persistence.
"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.operator_growth_mutation import (
    execute_eligible_operator_growth_mutation,
    execute_operator_growth_mutation,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

CORE, DONOR = "core", "donor"
NODES = (CORE, DONOR)
INITIAL = (Fraction(3, 8), Fraction(1, 8))
ZERO = (Fraction(0), Fraction(0))
WORD = ("emission", "coherence", "dissonance", "coherence", "silence")
CANCELLING_WORD = ("emission", "reception") + WORD[1:]
RECEIPT = "_operator_growth_mutation_end"


def _source(*, forms=INITIAL, capacities=(1.0, 1.0), floor=0.0, silence=0.5):
    graph = nx.Graph()
    graph.add_edge(CORE, DONOR, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        EDGE_AWARE_ENABLED=False,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        OZ_NOISE_MODE=False,
        RANDOM_SEED=31,
        ZHIR_THRESHOLD_XI=0.0625,
        ZHIR_MIN_VF=floor,
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        IL_dnfr_factor=0.0,
        SHA_vf_factor=silence,
        ZHIR_theta_shift=0.25,
    )
    for node, form, capacity in zip(NODES, forms, capacities, strict=True):
        graph.nodes[node].update(
            EPI=float(form),
            nu_f=capacity,
            theta=0.0,
            delta_nfr=0.0,
            epi_kind="form",
        )
        assert "epi_history" not in graph.nodes[node]
        assert "epi_time_history" not in graph.nodes[node]
        assert "glyph_history" not in graph.nodes[node]
    assert "compute_delta_nfr" not in graph.graph
    return graph


def _schedule(word=WORD, *, start=0.0):
    return build_operator_event_schedule(
        word,
        start_time=start,
        flow_durations=(0.5, 0.5) + (0.0,) * (len(word) - 1),
    )


def _execute(graph, *, word=WORD, start=0.0, candidates=None):
    return execute_eligible_operator_growth_mutation(
        graph,
        _schedule(word, start=start),
        candidates,
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
    )


def _forms(graph):
    return tuple(
        Fraction.from_float(
            require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))
        )
        for node in NODES
    )


def _state(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _decisions(result):
    assert tuple(decision.node for decision in result.decisions) == result.candidates
    return {decision.node: decision for decision in result.decisions}


def _assert_one_source(graph, result, word=WORD, *, prior_steps=None, prior_events=0):
    """Independent actual counters/logs exclude an extra hidden source replay."""
    prior_steps = {node: 0 for node in NODES} if prior_steps is None else prior_steps
    assert result.source_result.target_nodes == NODES
    assert result.source_result.pressure_refresh_callback_invocations == 0
    assert result.source_result.positive_flow_interval_indices == (0, 1)
    assert len(result.source_result.events) == len(word)
    events = graph.graph["hybrid_event_log"]
    assert len(events) == prior_events + len(word)
    assert tuple(event["operator_name"] for event in events[prior_events:]) == word
    selected = set(result.selected_targets)
    for node in NODES:
        expected = prior_steps[node] + len(word) + (node in selected)
        assert read_grammar_state(graph.nodes[node]).operator_step == expected
    if selected:
        assert result.executed and result.stage_result is not None
        assert result.stage_result.nodes_processed == len(selected)
        assert (
            tuple(
                decision.node
                for decision in result.stage_result.mutation_decision_observations
            )
            == result.selected_targets
        )
    else:
        assert not result.executed and result.stage_result is None
        assert getattr(graph, "_last_operator_applied", None) == word[-1]


def test_positive_actual_source_selects_both_nodes_without_supplied_targets():
    graph = _source()
    result = _execute(graph)
    assert result.candidates == result.selected_targets == NODES
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert result.growth.exact_flow_increment == ZERO
    assert _forms(graph) == (Fraction(1, 2), Fraction(1, 4))
    assert tuple(graph.edges) == ((CORE, DONOR),)
    assert graph.graph["_t"] == 1.0
    for node, decision in _decisions(result).items():
        assert decision.selected and decision.reasons == ()
        assert decision.exact_rate == Fraction(1, 8)
        assert decision.exact_xi == Fraction(1, 16)
        assert decision.previous_consumed_end is None
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.25
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.5
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert graph.nodes[node][RECEIPT] == 1
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == ("AL", "IL", "OZ", "IL", "SHA", "ZHIR")
        assert grammar.prior_coherence and grammar.u2_debt == 1
    _assert_one_source(graph, result)


def test_actual_cancellation_selects_only_the_donor_from_the_full_population():
    graph = _source()
    result = _execute(graph, word=CANCELLING_WORD)
    decisions = _decisions(result)
    assert result.candidates == NODES and result.selected_targets == (DONOR,)
    assert result.growth.exact_net_growth_rate == (Fraction(0), Fraction(1, 4))
    assert result.growth.exact_jump_variation == (Fraction(1, 4),) * 2
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert decisions[CORE].reasons == ("observed_threshold_not_crossed",)
    assert not decisions[CORE].selected and decisions[DONOR].selected
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.25
    assert RECEIPT not in graph.nodes[CORE] and graph.nodes[DONOR][RECEIPT] == 1
    core = read_grammar_state(graph.nodes[CORE])
    assert core.history == ("AL", "EN", "IL", "OZ", "IL", "SHA")
    assert core.prior_coherence and core.u2_debt == 0
    _assert_one_source(graph, result, CANCELLING_WORD)


def test_complete_grammar_zero_growth_abstains_while_preserving_the_real_source():
    graph = _source(forms=(Fraction(3, 8),) * 2)
    word = WORD[1:]
    result = _execute(graph, word=word)
    assert result.selected_targets == ()
    assert result.growth.exact_net_growth_rate == ZERO
    assert result.growth.exact_jump_increment == ZERO
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert graph.graph["_t"] == 1.0
    for node, decision in _decisions(result).items():
        assert decision.reasons == ("observed_threshold_not_crossed",)
        assert decision.grammar_violations == ()
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.5
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0
        assert RECEIPT not in graph.nodes[node]
        assert read_grammar_state(graph.nodes[node]).history == (
            "IL",
            "OZ",
            "IL",
            "SHA",
        )
    _assert_one_source(graph, result, word)


@pytest.mark.parametrize(
    "word",
    [
        ("emission", "silence"),
        ("emission", "coherence", "silence"),
        WORD + ("silence",),
    ],
)
def test_positive_growth_with_missing_or_expired_grammar_abstains_without_fallback(
    word,
):
    graph = _source()
    result = _execute(graph, word=word)
    assert result.selected_targets == ()
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert _forms(graph) == (Fraction(1, 2), Fraction(1, 4))
    for node, decision in _decisions(result).items():
        assert decision.reasons == ("grammar_rejected",)
        assert "U4b" in decision.grammar_violations
        assert not decision.selected
        grammar = read_grammar_state(graph.nodes[node])
        assert "ZHIR" not in grammar.history
        if "coherence" not in word:
            assert "IL" not in grammar.history and not grammar.prior_coherence
        assert RECEIPT not in graph.nodes[node]
    _assert_one_source(graph, result, word)


def test_actual_silence_capacity_and_configured_floor_select_only_active_enough_peer():
    graph = _source(capacities=(0.5, 1.0), floor=0.375)
    result = _execute(graph)
    decisions = _decisions(result)
    assert result.selected_targets == (DONOR,)
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert decisions[CORE].nu_f == 0.25
    assert decisions[CORE].minimum_nu_f == 0.375
    assert decisions[CORE].reasons == ("below_configured_capacity_floor",)
    assert decisions[DONOR].nu_f == 0.5 and decisions[DONOR].selected
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.25
    assert RECEIPT not in graph.nodes[CORE]
    _assert_one_source(graph, result)


def test_actual_complete_silence_abstains_despite_positive_form_growth():
    graph = _source(silence=0.0)
    result = _execute(graph)
    assert result.selected_targets == ()
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    for node, decision in _decisions(result).items():
        assert decision.nu_f == 0.0
        assert "inactive_structural_capacity" in decision.reasons
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.0
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0
        assert RECEIPT not in graph.nodes[node]
    _assert_one_source(graph, result)


def test_supplied_consumption_memory_excludes_one_node_without_vetoing_its_peer():
    graph = _source()
    # A declared memory-admission control, not a claim that this preparation
    # acquired the receipt. Source history and growth are still executed here.
    graph.nodes[CORE][RECEIPT] = Fraction(1)
    result = _execute(graph)
    decisions = _decisions(result)
    assert result.selected_targets == (DONOR,)
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert decisions[CORE].previous_consumed_end == Fraction(1)
    assert decisions[CORE].reasons == ("window_overlaps_consumed_time",)
    assert decisions[DONOR].previous_consumed_end is None
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.25
    assert graph.nodes[CORE][RECEIPT] == 1 and graph.nodes[DONOR][RECEIPT] == 1
    assert "ZHIR" not in read_grammar_state(graph.nodes[CORE]).history
    _assert_one_source(graph, result)


def test_actual_disjoint_reception_source_selects_its_positive_donor_automatically():
    graph = _source()
    first = _execute(graph)
    steps = {
        node: read_grammar_state(graph.nodes[node]).operator_step for node in NODES
    }
    word = ("reception",) + WORD[1:]
    second = _execute(graph, word=word, start=1.0)
    assert first.selected_targets == NODES and second.selected_targets == (DONOR,)
    assert first.growth.exact_end_time == second.growth.exact_start_time == 1
    assert second.growth.exact_net_growth_rate == (Fraction(-1, 8), Fraction(1, 8))
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.25
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.5
    assert graph.nodes[CORE][RECEIPT] == 1 and graph.nodes[DONOR][RECEIPT] == 2
    for decision in second.decisions:
        assert decision.previous_consumed_end == 1
    _assert_one_source(graph, second, word, prior_steps=steps, prior_events=len(WORD))


def test_explicit_candidate_subset_can_abstain_while_the_full_source_still_runs():
    graph = _source()
    result = _execute(graph, word=CANCELLING_WORD, candidates=(CORE,))
    assert result.candidates == (CORE,)
    assert result.selected_targets == ()
    assert result.growth.nodes == NODES
    assert result.growth.exact_net_growth_rate == (Fraction(0), Fraction(1, 4))
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in NODES)
    assert all(RECEIPT not in graph.nodes[node] for node in NODES)
    _assert_one_source(graph, result, CANCELLING_WORD)


def test_strict_all_target_control_still_rejects_the_same_cancelling_source():
    graph = _source()
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="exact signed operator growth"):
        execute_operator_growth_mutation(
            graph,
            _schedule(CANCELLING_WORD),
            NODES,
            context={"initial_epi_nonzero": True},
            method="euler",
            n_jobs=1,
        )
    assert _state(graph) == before
    assert _forms(graph) == INITIAL and graph.graph["_t"] == 0.0
