"""Prospective actual Emission episodes and native/hybrid source competition.

The duration, held pressure family and single-completion lifecycle are an
explicit constitutive lift. Ordinary AL stays instantaneous; no finite-jump
history limit or autonomous timing is inferred from these bounded controls.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.selectors import DefaultGlyphSelector, apply_native_operator_batch
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission import Emission
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.native_emission_window import execute_native_emission_window
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
PREPARED_PRESSURES = (Q(1, 16), Q(1, 2))
CAPACITIES = (8.0, 1.0)
HYBRID_RECEIPT = "_operator_growth_mutation_end"
NATIVE_RECEIPTS = "_native_zhir_consumed_intervals"


def _forms(graph):
    return tuple(
        Q.from_float(
            require_finite_real_scalar_epi(
                get_attr(
                    graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
                )
            )
        )
        for node in NODES
    )


def _cohere(graph, targets=NODES):
    execute_pointwise_stage(
        graph,
        Coherence(),
        targets,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )


def _prepared(*, dt_min=0.0, core_si=0.0):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=dt_min,
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
        ZHIR_THRESHOLD_XI=0.0625,
        NATIVE_ZHIR_OCCURRENCE_POLICY="consume_timestamped_interval",
        NATIVE_ZHIR_ABSTENTION_POLICY="skip",
        glyph_selector=DefaultGlyphSelector(),
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        ZHIR_theta_shift=0.25,
    )
    for node, capacity, pressure, si in zip(
        NODES, CAPACITIES, (0.125, 0.5), (core_si, 1.0), strict=True
    ):
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=capacity,
            theta=0.0,
            delta_nfr=pressure,
            Si=si,  # Declared stored diagnostics, not a fresh Si derivation.
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _cohere(graph)
    assert tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in NODES) == (
        1 / 16,
        1 / 4,
    )
    execute_dissonance_stage(
        graph,
        Dissonance(),
        NODES,
        propagate_to_network=False,
        validate_preconditions=True,
    )
    assert tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in NODES) == (
        1 / 8,
        1 / 2,
    )
    _cohere(graph, (CORE,))
    for node, pressure, history, debt in zip(
        NODES,
        PREPARED_PRESSURES,
        (("IL", "OZ", "IL"), ("IL", "OZ")),
        (0, 1),
        strict=True,
    ):
        assert Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR)) == pressure
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == history
        assert grammar.u2_debt == debt and grammar.prior_coherence
        assert "epi_time_history" not in graph.nodes[node]
        assert "epi_history" not in graph.nodes[node]
        assert "_structural_lineage" not in graph.nodes[node]
        assert NATIVE_RECEIPTS not in graph.nodes[node]
        assert HYBRID_RECEIPT not in graph.nodes[node]
        assert get_attr(graph.nodes[node], ALIAS_DEPI) == 0.0
        assert get_attr(graph.nodes[node], ALIAS_D2EPI) == 0.0
    assert _forms(graph) == (Q(1, 4), Q(1, 4))
    assert graph.graph["_t"] == 0.0
    assert "compute_delta_nfr" not in graph.graph
    return graph


def _episode(graph, duration):
    return execute_emission_episode(
        graph, NODES, duration=duration, method="euler", n_jobs=1
    )


def _assert_episode(result, *, start, duration, before, after, steps):
    assert result.nodes == result.targets == NODES
    assert result.start_time == float(start)
    assert result.final_time == float(start + duration)
    assert result.exact_rates == (Q(1, 2), Q(1, 2))
    assert result.effective_boosts == (float(duration / 2),) * 2
    assert result.exact_epi_before == (before, before)
    assert result.exact_epi_after == (after, after)
    assert result.flow_result.events == ()
    assert result.flow_result.target_nodes == NODES
    assert result.flow_result.positive_flow_interval_indices == (0,)
    assert result.flow_result.pressure_refresh_callback_invocations == 0
    assert (
        result.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
    )
    evidence = result.flow_result.flow_interval_evidence
    assert len(evidence) == 1 and evidence[0].resolved_substeps == steps


@pytest.mark.parametrize("dt_min,steps,acceleration", [(0.0, 1, 2.0), (0.125, 2, 0.0)])
def test_one_actual_episode_keeps_one_occurrence_across_solver_subdivision(
    dt_min, steps, acceleration
):
    graph = _prepared(dt_min=dt_min)
    result = _episode(graph, 0.25)
    _assert_episode(
        result, start=Q(0), duration=Q(1, 4), before=Q(1, 4), after=Q(3, 8), steps=steps
    )
    assert graph.graph["_t"] == 0.25 and _forms(graph) == (Q(3, 8), Q(3, 8))
    for node, pressure, capacity, history, debt in zip(
        NODES,
        PREPARED_PRESSURES,
        CAPACITIES,
        (("IL", "OZ", "IL", "AL"), ("IL", "OZ", "AL")),
        (0, 1),
        strict=True,
    ):
        data = graph.nodes[node]
        grammar = read_grammar_state(data)
        assert grammar.history == history
        assert grammar.operator_step == len(history) and grammar.u2_debt == debt
        assert data["_structural_lineage"]["activation_count"] == 1
        assert data["_emission_activated"]
        assert data["_emission_origin"] == data["_structural_lineage"]["origin"]
        assert tuple(data["epi_time_history"]) == ((0.0, 0.25), (0.25, 0.375))
        assert Q.from_float(get_attr(data, ALIAS_DNFR)) == pressure
        assert get_attr(data, ALIAS_VF) == capacity
        assert get_attr(data, ALIAS_THETA) == 0.0
        assert get_attr(data, ALIAS_DEPI) == 0.5
        # Coordinate/count invariance does not imply full diagnostic invariance.
        assert get_attr(data, ALIAS_D2EPI) == acceleration
        assert NATIVE_RECEIPTS not in data and HYBRID_RECEIPT not in data


def test_two_actual_episodes_preserve_segmentation_in_lifecycle_and_history():
    graph = _prepared()
    first = _episode(graph, 0.125)
    _assert_episode(
        first, start=Q(0), duration=Q(1, 8), before=Q(1, 4), after=Q(5, 16), steps=1
    )
    origins = {node: graph.nodes[node]["_emission_origin"] for node in NODES}
    assert all(
        graph.nodes[node]["_structural_lineage"]["activation_count"] == 1
        for node in NODES
    )
    second = _episode(graph, 0.125)
    _assert_episode(
        second, start=Q(1, 8), duration=Q(1, 8), before=Q(5, 16), after=Q(3, 8), steps=1
    )
    assert _forms(graph) == (Q(3, 8), Q(3, 8)) and graph.graph["_t"] == 0.25
    for node in NODES:
        data = graph.nodes[node]
        assert data["_structural_lineage"]["activation_count"] == 2
        assert data["_emission_origin"] == origins[node]
        assert tuple(data["epi_time_history"]) == (
            (0.0, 0.25),
            (0.125, 0.3125),
            (0.25, 0.375),
        )
        assert get_attr(data, ALIAS_DEPI) == 0.5
        assert get_attr(data, ALIAS_D2EPI) == 0.0
    assert read_grammar_state(graph.nodes[CORE]).history == (
        "IL",
        "OZ",
        "IL",
        "AL",
        "AL",
    )
    assert read_grammar_state(graph.nodes[PEER]).history == ("IL", "OZ", "AL", "AL")
    assert "OZ" not in read_grammar_state(graph.nodes[CORE]).history[-3:]
    assert "OZ" in read_grammar_state(graph.nodes[PEER]).history[-3:]


def test_ordinary_al_jump_matches_coordinates_without_episode_time_or_samples():
    graph = _prepared()
    execute_pointwise_stage(
        graph,
        Emission(),
        NODES,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    assert _forms(graph) == (Q(3, 8), Q(3, 8))
    assert graph.graph["_t"] == 0.0
    for node in NODES:
        data = graph.nodes[node]
        assert data["_structural_lineage"]["activation_count"] == 1
        assert read_grammar_state(data).history[-1] == "AL"
        assert "epi_time_history" not in data
        assert get_attr(data, ALIAS_DEPI) == 0.0
        assert get_attr(data, ALIAS_D2EPI) == 0.0


def _native_window(graph, duration=0.25):
    memory = graph.graph["history"]
    result = execute_native_emission_window(
        graph, NODES, duration=duration, method="euler", n_jobs=1
    )
    assert graph.graph["history"] is memory
    assert result.candidates == NODES
    assert result.selected_targets == (PEER,)
    assert result.executed and result.stage_result.nodes_processed == 1
    source = result.source_result
    end = Q(1, 4) + Q.from_float(duration) / 2
    _assert_episode(
        source.episode_result,
        start=Q(0),
        duration=Q.from_float(duration),
        before=Q(1, 4),
        after=end,
        steps=1,
    )
    assert source.flow_result is source.episode_result.flow_result
    assert source.final_time == graph.graph["_t"] == duration
    assert source.native_batch.completed and source.native_batch.available
    assert source.native_batch.clock_preserved
    assert (
        source.native_batch.initial.exact_time
        == source.native_batch.final.exact_time
        == Q.from_float(duration)
    )
    assert source.native_batch.exact_net_increment == (Q(0), Q(0))
    assert source.native_batch.exact_outside_dispatch_variation == (Q(0), Q(0))
    growth = result.growth
    assert growth.available and growth.unavailability_reasons == ()
    assert growth.source == "native_emission_window_net_growth"
    assert growth.nodes == NODES
    assert growth.exact_start_time == Q(0)
    assert growth.exact_end_time == growth.exact_duration == Q.from_float(duration)
    assert growth.exact_epi_before == (Q(1, 4), Q(1, 4))
    assert growth.exact_epi_after == _forms(graph) == (end, end)
    assert growth.exact_flow_increment == (Q.from_float(duration) / 2,) * 2
    assert growth.exact_jump_increment == growth.exact_jump_variation == (Q(0), Q(0))
    assert growth.exact_net_growth_rate == (Q(1, 2), Q(1, 2))
    return result


@pytest.mark.parametrize("duration", [0.125, 0.25])
def test_actual_episode_feeds_native_mutation_then_distinct_supplementary_response(
    duration,
):
    graph = _prepared()
    result = _native_window(graph, duration)
    native = result.source_result.native_decisions
    assert tuple(row.requested_glyph for row in native) == ("ZHIR", "IL")
    assert tuple(row.applied_glyph for row in native) == ("ZHIR", "IL")
    assert all(row.selector_evaluated and row.lag_reason is None for row in native)
    assert native[0].selection.normalized_metrics[1] == 1 / 8
    assert native[0].mutation_certificate.observed_depi_dt == 0.5
    assert native[0].mutation_certificate.evidence.previous_time == 0.0
    assert native[0].mutation_certificate.evidence.current_time == duration
    assert native[0].mutation_interval_committed
    decisions = {item.node: item for item in result.decisions}
    core = decisions[CORE]
    assert core.exact_rate == Q(1, 2) and core.exact_xi == Q(1, 16)
    assert core.grammar_violations == () and not core.selected
    assert core.reasons == ("native_mutation_precedence",)
    assert decisions[PEER].selected and decisions[PEER].reasons == ()
    response = result.stage_result.mutation_decision_observations
    assert len(response) == 1 and response[0].node == PEER
    certificate = response[0].trigger_certificate
    assert certificate.source == "native_emission_window_net_growth"
    assert certificate.time_basis == "native_emission_window_time"
    assert certificate.exact_observed_rate == Q(1, 2)
    assert certificate.predicted_depi_dt == 0.25
    assert tuple(get_attr(graph.nodes[node], ALIAS_THETA) for node in NODES) == (
        0.25,
        0.25,
    )
    assert tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in NODES) == (
        1 / 16,
        1 / 4,
    )
    for node, history in zip(
        NODES,
        (("IL", "OZ", "IL", "AL", "ZHIR"), ("IL", "OZ", "AL", "IL", "ZHIR")),
        strict=True,
    ):
        data = graph.nodes[node]
        grammar = read_grammar_state(data)
        assert grammar.history == history and grammar.operator_step == 5
        assert grammar.u2_debt == 1
        assert data["_structural_lineage"]["activation_count"] == 1
        assert data[NATIVE_RECEIPTS] == (("epi_time_history", 0.0, duration),)
        assert tuple(data["epi_time_history"]) == (
            (0.0, 0.25),
            (duration, float(result.growth.exact_epi_after[0])),
        )
    assert HYBRID_RECEIPT not in graph.nodes[CORE]
    assert graph.nodes[PEER][HYBRID_RECEIPT] == Q.from_float(duration)


def test_actual_physical_source_cannot_repeat_through_native_after_hybrid_consumption():
    graph = _prepared()
    _native_window(graph)
    # Declared actual intervention makes both default requests ZHIR, without
    # changing the episode's live EPI samples or inserting a new time interval.
    graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"] = 0.0
    _cohere(graph)
    before = {node: read_grammar_state(graph.nodes[node]) for node in NODES}
    samples = {node: tuple(graph.nodes[node]["epi_time_history"]) for node in NODES}
    for node in NODES:
        graph.nodes[node]["Si"] = 0.0
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
    trace = []
    apply_native_operator_batch(graph, graph.graph["history"], decision_trace=trace)
    assert tuple(row.requested_glyph for row in trace) == ("ZHIR", "ZHIR")
    assert all(row.status == "abstained" and row.applied_glyph is None for row in trace)
    assert all(
        row.mutation_abstention_reason == "mutation_interval_already_consumed"
        for row in trace
    )
    assert graph.graph["_t"] == 0.25
    assert _forms(graph) == (Q(3, 8), Q(3, 8))
    for node in NODES:
        data = graph.nodes[node]
        assert read_grammar_state(data) == before[node]
        assert tuple(data["epi_time_history"]) == samples[node]
        assert get_attr(data, ALIAS_THETA) == 0.25
        assert data["_structural_lineage"]["activation_count"] == 1
        assert data[NATIVE_RECEIPTS] == (("epi_time_history", 0.0, 0.25),)
    assert HYBRID_RECEIPT not in graph.nodes[CORE]
    assert graph.nodes[PEER][HYBRID_RECEIPT] == Q(1, 4)


def test_actual_native_coherence_can_expire_context_despite_positive_episode_growth():
    graph = _prepared(core_si=1.0)
    result = _native_window(graph)
    assert tuple(
        row.applied_glyph for row in result.source_result.native_decisions
    ) == ("IL", "IL")
    decisions = {item.node: item for item in result.decisions}
    assert decisions[CORE].exact_rate == Q(1, 2)
    assert decisions[CORE].reasons == ("grammar_rejected",)
    assert "U4b" in decisions[CORE].grammar_violations
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[PEER], ALIAS_THETA) == 0.25
    assert read_grammar_state(graph.nodes[CORE]).history == (
        "IL",
        "OZ",
        "IL",
        "AL",
        "IL",
    )
    assert NATIVE_RECEIPTS not in graph.nodes[CORE]
    assert HYBRID_RECEIPT not in graph.nodes[CORE]
    assert graph.nodes[PEER][NATIVE_RECEIPTS] == (("epi_time_history", 0.0, 0.25),)
    assert graph.nodes[PEER][HYBRID_RECEIPT] == Q(1, 4)
