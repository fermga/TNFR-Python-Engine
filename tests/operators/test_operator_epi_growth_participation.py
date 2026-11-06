"""Actual bounded schedule controls for detached typed growth observation.

All sources are local to this file. Positive held-pressure flow is an explicit
auxiliary comparator, not a claimed operator-grounded acquisition mechanism.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators._mutation_gate import mutation_trigger_certificate
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.operator_growth import observe_operator_epi_growth
from tnfr.types import require_finite_real_scalar_epi

CORE, DONOR = "core", "donor"
NODES = (CORE, DONOR)
INITIAL = (Fraction(3, 8), Fraction(1, 8))
ZERO = (Fraction(0), Fraction(0))


def _source(*, pressure=0.0):
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
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=31,
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        SHA_vf_factor=0.5,
    )
    assert "compute_delta_nfr" not in graph.graph
    for node, form in zip(NODES, INITIAL, strict=True):
        graph.nodes[node].update(
            EPI=float(form),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            epi_kind="form",
            glyph_history=[],
            epi_history=[float(form) / 2, float(form)],
        )
    return graph


def _execute(word, durations, *, pressure=0.0, capture=True):
    graph = _source(pressure=pressure)
    schedule = build_operator_event_schedule(
        word, start_time=0.0, flow_durations=durations
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
        include_stage_certificates=capture,
    )
    assert result.target_nodes == NODES
    assert result.pressure_refresh_callback_invocations == 0
    assert all(get_attr(graph.nodes[n], ALIAS_DNFR) == pressure for n in NODES)
    assert all(get_attr(graph.nodes[n], ALIAS_THETA) == 0 for n in NODES)
    assert all(get_attr(graph.nodes[n], ALIAS_VF) == 0.5 for n in NODES)
    assert all(graph.nodes[n]["glyph_history"][-1] == "SHA" for n in NODES)
    return graph, result


def _forms(graph):
    return tuple(
        Fraction.from_float(
            require_finite_real_scalar_epi(get_attr(graph.nodes[n], ALIAS_EPI))
        )
        for n in NODES
    )


def _observe_unchanged(graph, result):
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_time = graph.graph["_t"]
    before_events = deepcopy(graph.graph["hybrid_event_log"])
    histories = tuple(graph.nodes[n]["epi_time_history"] for n in NODES)
    observation = observe_operator_epi_growth(result)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph["_t"] == before_time
    assert graph.graph["hybrid_event_log"] == before_events
    assert all(
        graph.nodes[n]["epi_time_history"] is old
        for n, old in zip(NODES, histories, strict=True)
    )
    assert all("_native_zhir_consumed_intervals" not in graph.nodes[n] for n in NODES)
    assert observation.source == "operator_schedule_net_growth"
    return observation


def test_positive_actual_jump_is_observed_without_continuous_growth_or_gain_claim():
    graph, result = _execute(("emission", "silence"), (0.5, 0.0, 0.5))
    observation = _observe_unchanged(graph, result)
    assert observation.available
    assert observation.nodes == NODES
    assert observation.exact_duration == 1
    assert observation.exact_epi_before == INITIAL
    assert (
        observation.exact_epi_after
        == _forms(graph)
        == (
            Fraction(1, 2),
            Fraction(1, 4),
        )
    )
    assert observation.exact_flow_increment == ZERO
    assert observation.exact_jump_increment == (Fraction(1, 8),) * 2
    assert observation.exact_jump_variation == (Fraction(1, 8),) * 2
    assert (
        observation.exact_net_increment
        == observation.exact_net_growth_rate
        == (Fraction(1, 8),) * 2
    )
    assert observation.included_event_indices == (0, 1)
    assert observation.excluded_initial_event_indices == ()
    # No pure-EPI pressure law is claimed for this held zero-driver source.
    composition = result.represented_epi_schedule_composition
    assert not composition.represented_affine_composition_gain_certified
    assert any(operation.ineligibility_reasons for operation in composition.operations)
    certificate = mutation_trigger_certificate(graph.nodes[CORE], graph.graph)
    assert certificate.source == "epi_time_history"
    assert certificate.observed_depi_dt == 0
    assert not certificate.threshold_gate_satisfied
    assert observe_operator_epi_growth(result) == observation


def test_actual_cancelling_core_pair_preserves_signed_contributions_and_peer_gain():
    graph, result = _execute(
        ("emission", "reception", "silence"), (0.25, 0.5, 0.0, 0.25)
    )
    observation = _observe_unchanged(graph, result)
    assert observation.available
    assert observation.exact_epi_before == INITIAL
    assert observation.exact_epi_after == _forms(graph) == (Fraction(3, 8),) * 2
    assert observation.exact_flow_increment == ZERO
    assert observation.exact_jump_increment == (Fraction(0), Fraction(1, 4))
    assert observation.exact_jump_variation == (Fraction(1, 4),) * 2
    assert observation.exact_net_growth_rate == (Fraction(0), Fraction(1, 4))
    glyphs = [c for c in observation.contributions if c.operation_kind == "glyph"]
    assert tuple(c.operator_name for c in glyphs) == (
        "emission",
        "reception",
        "silence",
    )
    assert tuple(c.exact_increment[0] for c in glyphs) == (
        Fraction(1, 8),
        Fraction(-1, 8),
        Fraction(0),
    )
    assert sum(abs(c.exact_increment[0]) for c in glyphs) == Fraction(1, 4)
    held_graph, held_result = _execute(("silence",), (1.0, 0.0))
    held = _observe_unchanged(held_graph, held_result)
    assert held.available and held.exact_net_increment == ZERO
    assert held.exact_epi_before[0] == observation.exact_epi_before[0]
    assert held.exact_epi_after[0] == observation.exact_epi_after[0]
    assert len(held.contributions) != len(observation.contributions)


def test_equal_epi_endpoints_distinguish_actual_jump_and_held_driver_flow():
    jump_graph, jump_result = _execute(("emission", "silence"), (0.5, 0.0, 0.5))
    flow_graph, flow_result = _execute(("silence",), (1.0, 0.0), pressure=0.125)
    jump = _observe_unchanged(jump_graph, jump_result)
    flow = _observe_unchanged(flow_graph, flow_result)
    assert jump.available and flow.available
    assert jump.exact_epi_before == flow.exact_epi_before == INITIAL
    assert jump.exact_epi_after == flow.exact_epi_after
    assert jump.exact_net_growth_rate == flow.exact_net_growth_rate
    assert jump.exact_flow_increment == flow.exact_jump_increment == ZERO
    assert (
        jump.exact_jump_increment == flow.exact_flow_increment == (Fraction(1, 8),) * 2
    )
    # The actual continuous samples are distinct from the jump contribution.
    certificate = mutation_trigger_certificate(flow_graph.nodes[CORE], flow_graph.graph)
    assert certificate.observed_depi_dt == 0.125
    assert certificate.threshold_gate_satisfied
    assert all("ZHIR" not in flow_graph.nodes[n]["glyph_history"] for n in NODES)


def test_mixed_schedule_accounts_for_actual_silence_capacity_change():
    graph, result = _execute(("emission", "silence"), (0.5, 0.0, 0.5), pressure=0.125)
    observation = _observe_unchanged(graph, result)
    assert observation.available
    assert observation.exact_flow_increment == (Fraction(3, 32),) * 2
    assert observation.exact_jump_increment == (Fraction(1, 8),) * 2
    assert (
        observation.exact_net_increment
        == observation.exact_net_growth_rate
        == (Fraction(7, 32),) * 2
    )
    assert (
        observation.exact_epi_after
        == _forms(graph)
        == (
            Fraction(19, 32),
            Fraction(11, 32),
        )
    )
    flows = [c for c in observation.contributions if c.operation_kind == "flow"]
    assert [c.exact_increment for c in flows] == [
        (Fraction(1, 16),) * 2,
        (Fraction(1, 32),) * 2,
    ]
    assert mutation_trigger_certificate(
        graph.nodes[CORE], graph.graph
    ).observed_depi_dt == Fraction(1, 16)


@pytest.mark.parametrize(
    "durations,excluded,included,before,increment",
    [
        ((0.0, 1.0, 0.0), (0,), (1,), (Fraction(1, 2), Fraction(1, 4)), ZERO),
        ((1.0, 0.0, 0.0), (), (0, 1), INITIAL, (Fraction(1, 8),) * 2),
    ],
)
def test_right_continuous_window_distinguishes_initial_and_terminal_glyphs(
    durations, excluded, included, before, increment
):
    graph, result = _execute(("emission", "silence"), durations)
    observation = _observe_unchanged(graph, result)
    assert observation.available
    assert observation.exact_start_time == 0 and observation.exact_end_time == 1
    assert observation.excluded_initial_event_indices == excluded
    assert observation.included_event_indices == included
    assert observation.exact_epi_before == before
    assert (
        observation.exact_net_increment
        == observation.exact_net_growth_rate
        == increment
    )
    assert (
        tuple(
            c.operation_index
            for c in observation.contributions
            if c.operation_kind == "glyph"
        )
        == included
    )


def test_all_same_time_initial_glyphs_are_excluded_before_the_observed_baseline():
    graph, result = _execute(("emission", "reception", "silence"), (0.0, 0.0, 1.0, 0.0))
    observation = _observe_unchanged(graph, result)
    assert observation.available
    assert observation.excluded_initial_event_indices == (0, 1)
    assert observation.included_event_indices == (2,)
    assert (
        observation.exact_epi_before
        == observation.exact_epi_after
        == (Fraction(3, 8),) * 2
    )
    assert observation.exact_flow_increment == observation.exact_jump_increment == ZERO
    assert observation.exact_net_growth_rate == ZERO


@pytest.mark.parametrize(
    "capture,durations", [(False, (0.5, 0.0, 0.5)), (True, (0, 0, 0))]
)
def test_missing_captures_or_zero_duration_do_not_become_zero_growth(
    capture, durations
):
    graph, result = _execute(("emission", "silence"), durations, capture=capture)
    observation = _observe_unchanged(graph, result)
    assert not observation.available
    assert observation.unavailability_reasons
    assert observation.exact_net_growth_rate is None
