"""Actual episode source and declared Parametric acceleration consumption.

Interior acceleration belongs to the chosen held episode law. It is separate
from solver diagnostics and unavailable three-sample curvature. These bounded
controls establish neither autonomous timing nor arbitrary action invariance.
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
from tnfr.dynamics.selectors import ParametricGlyphSelector
from tnfr.operators import native_emission_window as native_window_owner
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_observation import (
    EmissionEpisodeObservation,
    observe_emission_episode,
)
from tnfr.operators.grammar_debt import (
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.native_emission_window import execute_native_emission_window
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
NATIVE_RECEIPTS = "_native_zhir_consumed_intervals"
HYBRID_RECEIPT = "_operator_growth_mutation_end"
INTERVAL = ("epi_time_history", 0.0, 0.25)


def _forms(graph):
    return tuple(
        Q.from_float(
            require_finite_real_scalar_epi(
                get_attr(data, ALIAS_EPI, strict=True, conv=lambda value: value)
            )
        )
        for _, data in graph.nodes(data=True)
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


def _prepared(dt_min):
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
        glyph_selector=ParametricGlyphSelector(),
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        ZHIR_theta_shift=0.25,
    )
    for node, capacity, pressure, sense in zip(
        NODES, (8.0, 1.0), (0.125, 0.5), (0.0, 1.0), strict=True
    ):
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=capacity,
            theta=0.0,
            delta_nfr=pressure,
            Si=sense,  # Explicit stored policy input, not freshly derived Si.
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _cohere(graph)
    execute_dissonance_stage(
        graph,
        Dissonance(),
        NODES,
        propagate_to_network=False,
        validate_preconditions=True,
    )
    _cohere(graph, (CORE,))
    for node, pressure, history, debt in zip(
        NODES,
        (1 / 16, 1 / 2),
        (("IL", "OZ", "IL"), ("IL", "OZ")),
        (0, 1),
        strict=True,
    ):
        data = graph.nodes[node]
        state = read_grammar_state(data)
        assert get_attr(data, ALIAS_DNFR) == pressure
        assert state.history == history and state.u2_debt == debt
        assert state.prior_coherence and state.operator_step == len(history)
        assert "epi_time_history" not in data
        assert "_structural_lineage" not in data
        assert NATIVE_RECEIPTS not in data and HYBRID_RECEIPT not in data
        assert get_attr(data, ALIAS_D2EPI) == 0.0
    assert _forms(graph) == (Q(1, 4), Q(1, 4))
    return graph


def _window(graph, mode, *, candidates=None):
    options = (
        {} if mode == "solver_diagnostic" else {"selector_acceleration_source": mode}
    )
    return execute_native_emission_window(
        graph,
        NODES,
        duration=0.25,
        candidates=candidates,
        method="euler",
        n_jobs=1,
        **options,
    )


def _assert_source(graph, result, mode, acceleration, steps):
    source = result.source_result
    observation = observe_emission_episode(source.episode_result)
    assert type(observation) is EmissionEpisodeObservation
    assert observation.nodes == observation.targets == NODES
    assert observation.exact_start_time == Q(0)
    assert observation.exact_end_time == observation.exact_duration == Q(1, 4)
    assert observation.exact_observed_rates == (Q(1, 2), Q(1, 2))
    assert observation.exact_held_rates == (Q(1, 2), Q(1, 2))
    assert observation.exact_interior_accelerations == (Q(0), Q(0))
    assert observation.solver_rates == (0.5, 0.5)
    assert observation.solver_accelerations == (acceleration, acceleration)
    assert observation.source == "emission_episode_interior"
    assert observation.acceleration_basis == "held_emission_law"
    assert observation.episode_sampled_acceleration_available is False
    assert observation.boundary_transition_available is False
    assert source.selector_acceleration_source == mode
    if mode == "emission_interior":
        assert source.selector_observation == observation
    else:
        assert source.selector_observation is None
    evidence = source.flow_result.flow_interval_evidence
    assert len(evidence) == 1 and evidence[0].resolved_substeps == steps
    assert source.native_batch.completed and source.native_batch.available
    assert source.native_batch.exact_net_increment == (Q(0), Q(0))
    assert source.native_batch.exact_outside_dispatch_variation == (Q(0), Q(0))
    assert result.growth.exact_flow_increment == (Q(1, 8), Q(1, 8))
    assert result.growth.exact_jump_increment == (Q(0), Q(0))
    assert result.growth.exact_net_growth_rate == (Q(1, 2), Q(1, 2))
    assert result.selected_targets == (PEER,)
    assert result.executed and result.stage_result.nodes_processed == 1
    assert graph.graph["_t"] == 0.25 and _forms(graph) == (Q(3, 8), Q(3, 8))
    assert tuple(graph.nodes) == NODES and tuple(graph.edges) == ((CORE, PEER),)
    for node in NODES:
        data = graph.nodes[node]
        assert data["_structural_lineage"]["activation_count"] == 1
        assert tuple(data["epi_time_history"]) == ((0.0, 0.25), (0.25, 0.375))
        sampled = observe_structural_acceleration(graph, node)
        assert sampled.source == "epi_time_history"
        assert sampled.available is False and sampled.value is None
    peer = graph.nodes[PEER]
    assert get_attr(peer, ALIAS_DNFR) == 0.25
    assert get_attr(peer, ALIAS_THETA) == 0.25
    assert get_attr(peer, ALIAS_VF) == 1.0
    assert peer[NATIVE_RECEIPTS] == (INTERVAL,)
    assert peer[HYBRID_RECEIPT] == Q(1, 4)
    state = read_grammar_state(peer)
    assert state.history == ("IL", "OZ", "AL", "IL", "ZHIR")
    assert state.operator_step == 5 and state.u2_debt == 1
    return source.native_decisions


@pytest.mark.parametrize(
    "mode,dt_min,steps,acceleration,core_glyph,consumed_acceleration",
    [
        ("solver_diagnostic", 0.0, 1, 2.0, "THOL", 1.0),
        ("solver_diagnostic", 0.125, 2, 0.0, "ZHIR", 0.0),
        ("emission_interior", 0.0, 1, 2.0, "ZHIR", 0.0),
        ("emission_interior", 0.125, 2, 0.0, "ZHIR", 0.0),
    ],
)
def test_actual_parametric_action_uses_declared_episode_acceleration_channel(
    mode, dt_min, steps, acceleration, core_glyph, consumed_acceleration, monkeypatch
):
    graph = _prepared(dt_min)
    before_response = []
    actual_stage = native_window_owner._execute_growth_stage

    def observe_before_actual_stage(subject, *args, **kwargs):
        before_response.append(
            tuple(get_attr(subject.nodes[node], ALIAS_D2EPI) for node in NODES)
        )
        return actual_stage(subject, *args, **kwargs)

    monkeypatch.setattr(
        native_window_owner, "_execute_growth_stage", observe_before_actual_stage
    )
    result = _window(graph, mode)
    assert before_response == [
        (0.0 if core_glyph == "THOL" else acceleration, acceleration)
    ]
    native = _assert_source(graph, result, mode, acceleration, steps)
    assert tuple(row.requested_glyph for row in native) == (core_glyph, "IL")
    assert tuple(row.applied_glyph for row in native) == (core_glyph, "IL")
    assert all(row.selector_evaluated and row.lag_reason is None for row in native)
    selection = native[0].selection
    assert selection.kind == "parametric"
    assert selection.normalized_metrics == (0.0, 0.125, consumed_acceleration)
    assert selection.base_candidate == selection.soft_filter_candidate == core_glyph
    assert selection.hysteresis_candidate is None
    assert selection.acceleration_source == mode
    assert selection.acceleration_window == (
        (Q(0), Q(1, 4)) if mode == "emission_interior" else None
    )
    core = graph.nodes[CORE]
    state = read_grammar_state(core)
    assert state.history == ("IL", "OZ", "IL", "AL", core_glyph)
    assert state.operator_step == 5
    assert get_attr(core, ALIAS_DNFR) == 1 / 16
    assert get_attr(core, ALIAS_VF) == 8.0
    assert HYBRID_RECEIPT not in core
    decisions = {item.node: item for item in result.decisions}
    assert decisions[PEER].selected
    if core_glyph == "THOL":
        assert state.u2_debt == 0
        assert get_attr(core, ALIAS_THETA) == 0.0
        assert get_attr(core, ALIAS_D2EPI) == 0.0
        assert NATIVE_RECEIPTS not in core
        assert decisions[CORE].reasons == ("grammar_rejected",)
        assert "U4b" in decisions[CORE].grammar_violations
    else:
        assert state.u2_debt == 1
        assert get_attr(core, ALIAS_THETA) == 0.25
        assert get_attr(core, ALIAS_D2EPI) == acceleration
        assert core[NATIVE_RECEIPTS] == (INTERVAL,)
        assert native[0].mutation_interval_committed
        assert decisions[CORE].reasons == ("native_mutation_precedence",)
    # The actual supplementary staged ZHIR commits history-derived curvature.
    # Two samples give its existing compatibility0, after the checkpoint above.
    assert get_attr(graph.nodes[PEER], ALIAS_D2EPI) == 0.0


@pytest.mark.parametrize("dt_min,steps,acceleration", [(0.0, 1, 2.0), (0.125, 2, 0.0)])
def test_actual_repeat_fallback_consumes_same_interior_channel_in_soft_force(
    dt_min, steps, acceleration
):
    graph = _prepared(dt_min)
    # Explicit unit context, not a claimed acquired ZHIR before physical time.
    graph.nodes[CORE]["glyph_history"] = ["IL", "ZHIR", "IL"]
    reset_grammar_state_from_history(graph.nodes[CORE])
    assert read_grammar_state(graph.nodes[CORE]).u2_debt == 0
    graph.graph.update(NAV_STRICT=False, NAV_RANDOM=False)
    graph.graph["GLYPH_FACTORS"].update(NAV_eta=0.5, NAV_jitter=0.0)
    result = _window(graph, "emission_interior", candidates=(PEER,))
    native = _assert_source(graph, result, "emission_interior", acceleration, steps)
    assert result.candidates == (PEER,)
    assert tuple(row.requested_glyph for row in native) == ("NAV", "IL")
    assert tuple(row.applied_glyph for row in native) == ("NAV", "IL")
    selection = native[0].selection
    assert selection.normalized_metrics == (0.0, 0.125, 0.0)
    assert selection.base_candidate == selection.override_candidate == "ZHIR"
    assert selection.hysteresis_candidate is None
    assert selection.soft_filter_candidate == "NAV"
    assert selection.acceleration_source == "emission_interior"
    assert selection.acceleration_window == (Q(0), Q(1, 4))
    core = graph.nodes[CORE]
    assert get_attr(core, ALIAS_DNFR) == 129 / 32
    assert get_attr(core, ALIAS_VF) == 8.0
    assert get_attr(core, ALIAS_THETA) == 0.0
    assert get_attr(core, ALIAS_D2EPI) == acceleration
    assert NATIVE_RECEIPTS not in core and HYBRID_RECEIPT not in core
    state = read_grammar_state(core)
    assert state.history == ("IL", "ZHIR", "IL", "AL", "NAV")
    assert state.operator_step == 5 and state.u2_debt == 0
