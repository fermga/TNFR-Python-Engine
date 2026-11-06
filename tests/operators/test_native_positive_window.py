"""Actual native jumps supply their own bounded positive-window response.

These prospective finite controls use held zero nodal flow and actual
IL/OZ/IL preparation. Clock, initial graph, selector and waiting memory are
declared inputs; neither recurring timing nor autonomous acquisition is claimed.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.selectors import DefaultGlyphSelector
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.native_positive_window import execute_native_positive_window
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
INITIAL = {CORE: Q(3, 8), PEER: Q(1, 4)}
RECEIPT = "_operator_growth_mutation_end"


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
    )


def _prepare(order=NODES, *, dissonance_targets=NODES, xi=1 / 16):
    graph = nx.Graph()
    graph.add_nodes_from(order)
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=37,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        ZHIR_THRESHOLD_XI=xi,
        glyph_selector=DefaultGlyphSelector(),
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        IL_dnfr_factor=0.0,
        OZ_dnfr_factor=2.0,
        ZHIR_theta_shift=0.25,
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=float(INITIAL[node]),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            Si=1.0,
            d2EPI=0.0,
            epi_kind="form",
        )
    execute_pointwise_stage(
        graph,
        Coherence(),
        tuple(graph),
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    if dissonance_targets:
        execute_dissonance_stage(
            graph,
            Dissonance(),
            dissonance_targets,
            propagate_to_network=False,
            validate_preconditions=True,
        )
        for node in graph:
            expected = 0.1 if node in dissonance_targets else 0.0
            assert get_attr(graph.nodes[node], ALIAS_DNFR) == expected
        execute_pointwise_stage(
            graph,
            Coherence(),
            tuple(graph),
            phase_locking_coefficient=0.0,
            require_exact_operator=True,
            validate_preconditions=True,
        )
    for node in graph:
        grammar = read_grammar_state(graph.nodes[node])
        if node in dissonance_targets:
            expected = ("IL", "OZ", "IL")
        elif dissonance_targets:
            expected = ("IL", "IL")
        else:
            expected = ("IL",)
        assert grammar.history == expected
        assert grammar.u2_debt == 0 and grammar.prior_coherence
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert _form(graph, node) == INITIAL[node]
        assert "epi_time_history" not in graph.nodes[node]
        assert "epi_history" not in graph.nodes[node]
        assert RECEIPT not in graph.nodes[node]
    assert "compute_delta_nfr" not in graph.graph
    return graph


def _make_due(graph, *, emission=CORE, reception=PEER):
    """Declare waiting inputs; the actual native priority selects the actions."""
    memory = graph.graph["history"]
    memory.setdefault("since_AL", {})[emission] = graph.graph["AL_MAX_LAG"]
    memory.setdefault("since_EN", {})[reception] = graph.graph["EN_MAX_LAG"]


def _execute(graph, *, duration=1.0):
    steps = {
        node: read_grammar_state(graph.nodes[node]).operator_step for node in graph
    }
    before = {node: _form(graph, node) for node in graph}
    start = Q.from_float(graph.graph["_t"])
    memory = graph.graph["history"]
    result = execute_native_positive_window(
        graph, duration=duration, method="euler", n_jobs=1
    )
    assert graph.graph["history"] is memory
    assert result.candidates == tuple(graph)
    assert tuple(item.node for item in result.decisions) == result.candidates
    growth = result.growth
    assert growth.available and growth.unavailability_reasons == ()
    assert growth.nodes == tuple(graph)
    assert growth.exact_start_time == start
    assert growth.exact_duration == Q.from_float(duration)
    assert growth.exact_end_time == start + Q.from_float(duration)
    assert Q.from_float(graph.graph["_t"]) == growth.exact_end_time
    assert growth.exact_epi_before == tuple(before[node] for node in graph)
    assert growth.exact_epi_after == tuple(_form(graph, node) for node in graph)
    assert growth.exact_flow_increment == (Q(0), Q(0))
    assert growth.exact_net_increment == growth.exact_jump_increment
    assert growth.exact_net_growth_rate == tuple(
        value / growth.exact_duration for value in growth.exact_net_increment
    )
    source = result.source_result
    assert source.final_time == graph.graph["_t"]
    flow = source.flow_result
    assert flow.target_nodes == tuple(graph)
    assert flow.events == ()
    assert flow.positive_flow_interval_indices == (0,)
    assert flow.pressure_refresh_callback_invocations == 0
    assert flow.all_positive_flow_intervals_binary64_held_pressure_identified
    assert flow.final_time == graph.graph["_t"]
    batch = source.native_batch
    assert batch.completed and batch.available and batch.clock_preserved
    assert batch.initial.exact_time == batch.final.exact_time == growth.exact_end_time
    assert batch.exact_outside_dispatch_variation == (Q(0), Q(0))
    assert batch.exact_outside_dispatch_increment == (Q(0), Q(0))
    assert batch.exact_net_increment == growth.exact_jump_increment
    assert len(batch.dispatches) == len(source.native_decisions) == 2
    assert all(item.returned and item.failure_type is None for item in batch.dispatches)
    selected = set(result.selected_targets)
    for node in graph:
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.operator_step == steps[node] + 1 + (node in selected)
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert get_attr(graph.nodes[node], ALIAS_VF) == 1.0
    if selected:
        assert result.executed and result.stage_result is not None
        assert result.stage_result.nodes_processed == len(selected)
        for item in result.stage_result.mutation_decision_observations:
            assert item.node in selected
            certificate = item.trigger_certificate
            assert certificate.source == growth.source
            assert certificate.exact_start_time == growth.exact_start_time
            assert certificate.exact_end_time == growth.exact_end_time
            index = growth.nodes.index(item.node)
            assert (
                certificate.exact_observed_rate == growth.exact_net_growth_rate[index]
            )
            assert certificate.predicted_depi_dt == 0.0
            assert graph.nodes[item.node][RECEIPT] == growth.exact_end_time
    else:
        assert not result.executed and result.stage_result is None
    return result


def _decisions(result):
    return {item.node: item for item in result.decisions}


@pytest.mark.parametrize("order", [NODES, (PEER, CORE)])
def test_actual_native_order_selects_response_from_own_exact_growth(order):
    graph = _prepare(order)
    _make_due(graph)
    result = _execute(graph)
    peer_increment = Q(1, 8) if order == NODES else Q(1, 16)
    expected = {CORE: Q(1, 8), PEER: peer_increment}
    assert result.growth.exact_net_growth_rate == tuple(
        expected[node] for node in order
    )
    assert _form(graph, CORE) == Q(1, 2)
    assert _form(graph, PEER) == Q(1, 4) + peer_increment
    source = result.source_result
    assert tuple(item.node for item in source.native_batch.dispatches) == order
    assert tuple(item.glyph for item in source.native_batch.dispatches) == (
        ("AL", "EN") if order == NODES else ("EN", "AL")
    )
    assert all(not item.selector_evaluated for item in source.native_decisions)
    expected_selected = NODES if order == NODES else (CORE,)
    assert result.selected_targets == expected_selected
    for node, decision in _decisions(result).items():
        assert decision.exact_rate == expected[node]
        assert decision.exact_xi == Q(1, 16)
        assert decision.previous_consumed_end is None
        assert decision.grammar_violations == ()
        if node in expected_selected:
            assert decision.selected and decision.reasons == ()
            assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.25
        else:
            assert decision.reasons == ("observed_threshold_not_crossed",)
            assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0
            assert RECEIPT not in graph.nodes[node]
        assert graph.nodes[node]["epi_time_history"] == ()
    assert graph.graph["history"]["since_AL"][CORE] == 0
    assert graph.graph["history"]["since_EN"][PEER] == 0


def test_actual_longer_window_preserves_jumps_but_strict_equality_abstains():
    graph = _prepare()
    _make_due(graph)
    result = _execute(graph, duration=2.0)
    assert result.growth.exact_jump_increment == (Q(1, 8), Q(1, 8))
    assert result.growth.exact_net_growth_rate == (Q(1, 16), Q(1, 16))
    assert result.selected_targets == ()
    assert tuple(_form(graph, node) for node in NODES) == (Q(1, 2), Q(3, 8))
    for item in result.decisions:
        assert item.exact_rate == item.exact_xi
        assert item.reasons == ("observed_threshold_not_crossed",)
        assert get_attr(graph.nodes[item.node], ALIAS_THETA) == 0.0
        assert RECEIPT not in graph.nodes[item.node]


@pytest.mark.parametrize("dissonance_targets", [(CORE,), ()])
def test_positive_source_retains_independent_own_grammar_admission(dissonance_targets):
    graph = _prepare(dissonance_targets=dissonance_targets)
    _make_due(graph)
    result = _execute(graph)
    assert result.growth.exact_net_growth_rate == (Q(1, 8), Q(1, 8))
    assert result.selected_targets == dissonance_targets
    for node, item in _decisions(result).items():
        if node in dissonance_targets:
            assert item.selected and item.reasons == ()
            assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.25
        else:
            assert item.reasons == ("grammar_rejected",)
            assert "U4b" in item.grammar_violations
            assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0
            assert RECEIPT not in graph.nodes[node]
            assert "ZHIR" not in read_grammar_state(graph.nodes[node]).history


def test_actual_zero_growth_native_actions_remain_when_response_is_empty():
    graph = _prepare()
    result = _execute(graph)
    assert result.selected_targets == ()
    assert result.growth.exact_net_growth_rate == (Q(0), Q(0))
    assert result.growth.exact_jump_variation == (Q(0), Q(0))
    assert tuple(
        item.applied_glyph for item in result.source_result.native_decisions
    ) == (
        "IL",
        "IL",
    )
    for node, item in _decisions(result).items():
        assert item.reasons == ("observed_threshold_not_crossed",)
        assert read_grammar_state(graph.nodes[node]).history == ("IL", "OZ", "IL", "IL")
        assert tuple(graph.nodes[node]["epi_time_history"]) == (
            (0.0, float(INITIAL[node])),
            (1.0, float(INITIAL[node])),
        )
        assert RECEIPT not in graph.nodes[node]
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0


def test_actual_negative_reception_is_not_positive_dispatch_variation():
    graph = _prepare(xi=1 / 32)
    _make_due(graph, emission=PEER, reception=CORE)
    result = _execute(graph)
    assert result.growth.exact_net_growth_rate == (Q(-1, 16), Q(1, 8))
    assert result.growth.exact_jump_variation == (Q(1, 16), Q(1, 8))
    assert result.selected_targets == (PEER,)
    assert tuple(_form(graph, node) for node in NODES) == (Q(5, 16), Q(3, 8))
    decisions = _decisions(result)
    assert decisions[CORE].reasons == ("observed_threshold_not_crossed",)
    assert decisions[PEER].selected and decisions[PEER].reasons == ()
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[PEER], ALIAS_THETA) == 0.25
    assert RECEIPT not in graph.nodes[CORE]


def test_actual_adjacent_windows_preserve_receipts_and_recent_mutation_context():
    graph = _prepare()
    _make_due(graph)
    first = _execute(graph)
    assert first.selected_targets == NODES
    for node in NODES:
        assert graph.nodes[node][RECEIPT] == Q(1)
        assert read_grammar_state(graph.nodes[node]).u2_debt == 1
    # This renewal is a declared waiting-memory input, not an inferred clock or
    # autonomous repeated selection. All source, grammar and receipt state stays.
    _make_due(graph)
    second = _execute(graph)
    assert second.selected_targets == NODES
    assert first.growth.exact_end_time == second.growth.exact_start_time == Q(1)
    assert second.growth.exact_end_time == Q(2)
    assert second.growth.exact_net_growth_rate == (Q(1, 8), Q(1, 8))
    assert tuple(_form(graph, node) for node in NODES) == (Q(5, 8), Q(1, 2))
    for node, item in _decisions(second).items():
        assert item.previous_consumed_end == Q(1)
        assert item.reasons == () and item.selected
        assert graph.nodes[node][RECEIPT] == Q(2)
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.5
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.u2_debt == 2
        glyph = "AL" if node == CORE else "EN"
        assert grammar.history == ("IL", "OZ", "IL", glyph, "ZHIR", glyph, "ZHIR")
