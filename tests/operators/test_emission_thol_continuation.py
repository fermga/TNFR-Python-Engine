"""Actual AL history, signed THOL pressure and a successor AL episode.

All clocks, factors and requests are declared finite interventions. Physical
samples and grammar come from real operators; no autonomy, scale-independent
curvature or identical lifecycle across different dispatch routes is inferred.
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
from tnfr.dynamics.selectors import apply_native_operator_batch
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
    execute_self_organization_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.self_organization import SelfOrganization
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
T = Q(1, 8)


def _forms(graph):
    return tuple(
        Q.from_float(
            require_finite_real_scalar_epi(
                get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda x: x)
            )
        )
        for node in NODES
    )


def _pressures(graph):
    return tuple(
        Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in NODES
    )


def _physical(graph, node):
    return tuple(
        (Q.from_float(time), Q.from_float(epi))
        for time, epi in graph.nodes[node]["epi_time_history"]
    )


def _cohere_then_dissonate(graph, factor):
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = float(factor)
    execute_pointwise_stage(
        graph,
        Coherence(),
        NODES,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    execute_dissonance_stage(
        graph,
        Dissonance(),
        NODES,
        propagate_to_network=False,
        validate_preconditions=True,
    )


def _episode(graph, *, duration=T):
    return execute_emission_episode(
        graph, NODES, duration=float(duration), method="euler", n_jobs=1
    )


def _prepared(*, dt_min=0.0):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=dt_min,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=61,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        EDGE_AWARE_ENABLED=True,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5, OZ_dnfr_factor=2.0, THOL_accel=0.125
    )
    for node in NODES:
        graph.nodes[node].update(
            EPI=0.25, nu_f=1.0, theta=0.0, delta_nfr=0.5, epi_kind="form"
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _cohere_then_dissonate(graph, Q(2))
    assert _forms(graph) == (Q(1, 4),) * 2
    assert _pressures(graph) == (Q(1, 2),) * 2
    for node in NODES:
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == ("IL", "OZ")
        assert grammar.u2_debt == 1 and grammar.prior_coherence
        assert "epi_time_history" not in graph.nodes[node]
        assert "_structural_lineage" not in graph.nodes[node]
    assert graph.graph["_t"] == 0.0
    assert "compute_delta_nfr" not in graph.graph
    return graph


def _source(factor, *, dt_min=0.0, split_second=False):
    graph = _prepared(dt_min=dt_min)
    first = _episode(graph)
    assert first.exact_rates == (Q(1, 2),) * 2
    assert first.exact_epi_after == (Q(5, 16),) * 2
    _cohere_then_dissonate(graph, factor)
    assert _pressures(graph) == (factor / 4,) * 2
    if split_second:
        _episode(graph, duration=T / 2)
        second = _episode(graph, duration=T / 2)
    else:
        second = _episode(graph)
    assert second.exact_rates == (factor / 4,) * 2
    assert graph.graph["_t"] == float(2 * T)
    assert _forms(graph) == (Q(5, 16) + factor / 32,) * 2
    for result in (first, second):
        assert result.flow_result.events == ()
        assert result.flow_result.pressure_refresh_callback_invocations == 0
        assert (
            result.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
        )
        expected_steps = 2 if dt_min else 1
        assert (
            result.flow_result.flow_interval_evidence[0].resolved_substeps
            == expected_steps
        )
    return graph


def _assert_source(graph, *, factor, split_second=False):
    endpoint = Q(5, 16) + factor / 32
    samples = [(Q(0), Q(1, 4)), (T, Q(5, 16))]
    if split_second:
        samples.append((3 * T / 2, Q(5, 16) + factor / 64))
    samples.append((2 * T, endpoint))
    expected_acceleration = Q(0) if split_second else 2 * factor - 4
    for node in NODES:
        data = graph.nodes[node]
        assert _physical(graph, node) == tuple(samples)
        observation = observe_structural_acceleration(graph, node)
        assert observation.available
        assert observation.source == "epi_time_history"
        assert observation.time_basis == "physical_time"
        assert observation.history_length == len(samples)
        assert observation.current_endpoint_matches_state is True
        assert observation.value == float(expected_acceleration)
        assert observation.samples == tuple(
            (float(t), float(x)) for t, x in samples[-3:]
        )
        grammar = read_grammar_state(data)
        assert grammar.history[-3:] == (
            ("OZ", "AL", "AL") if split_second else ("IL", "OZ", "AL")
        )
        assert grammar.operator_step == 6 + int(split_second)
        assert grammar.u2_debt == 1 and grammar.prior_coherence
        assert data["_structural_lineage"]["activation_count"] == 2 + int(split_second)
    return tuple(samples)


def _stage_thol(graph):
    # Fixed-support policy: abs(observed A) <= 1 does not cross strict tau=1.
    return execute_self_organization_stage(
        graph, SelfOrganization(), NODES, tau=1.0, validate_preconditions=True
    )


def _assert_pressure_only(graph, *, before, pressure, acceleration, samples):
    assert _forms(graph) == (before,) * 2
    assert _pressures(graph) == (pressure,) * 2
    assert graph.graph["_t"] == float(2 * T)
    assert tuple(graph) == NODES and tuple(graph.edges) == ((CORE, PEER),)
    assert graph.edges[CORE, PEER]["weight"] == 1.0
    for node in NODES:
        data = graph.nodes[node]
        assert _physical(graph, node) == samples
        assert get_attr(data, ALIAS_VF) == 1.0
        assert get_attr(data, ALIAS_THETA) == 0.0
        assert get_attr(data, ALIAS_D2EPI) == float(acceleration)
        assert data["epi_kind"] == "form"
        assert not data.get("sub_nodes") and not data.get("sub_epis")
        grammar = read_grammar_state(data)
        assert grammar.history[-1] == "THOL" and grammar.u2_debt == 0


def _assert_successor(
    graph, *, pressure, endpoint, samples, thol=True, split_second=False
):
    origins = tuple(graph.nodes[node]["_emission_origin"] for node in NODES)
    result = _episode(graph)
    assert result.exact_rates == (pressure,) * 2
    assert result.exact_epi_after == (endpoint,) * 2
    assert result.effective_boosts == (float(pressure * T),) * 2
    assert _forms(graph) == (endpoint,) * 2
    assert _pressures(graph) == (pressure,) * 2
    assert graph.graph["_t"] == float(3 * T)
    assert tuple(graph) == NODES and tuple(graph.edges) == ((CORE, PEER),)
    for node, origin in zip(NODES, origins, strict=True):
        data = graph.nodes[node]
        # The actual physical recorder retains four endpoints, including in
        # the split source whose fifth historical endpoint evicts the first.
        assert _physical(graph, node) == (*samples, (3 * T, endpoint))[-4:]
        assert data["_structural_lineage"]["activation_count"] == 3 + int(split_second)
        assert data["_emission_origin"] == origin
        assert get_attr(data, ALIAS_VF) == 1.0
        assert get_attr(data, ALIAS_THETA) == 0.0
        grammar = read_grammar_state(data)
        assert grammar.operator_step == 7 + int(thol) + int(split_second)
        assert grammar.u2_debt == (0 if thol else 1)
        assert grammar.prior_coherence and grammar.history[-1] == "AL"


@pytest.mark.parametrize(
    "factor,acceleration,source_epi,pressure,after,control_after",
    [
        (Q(3, 2), Q(-1), Q(23, 64), Q(1, 4), Q(25, 64), Q(26, 64)),
        (Q(2), Q(0), Q(24, 64), Q(1, 2), Q(28, 64), Q(28, 64)),
        (Q(5, 2), Q(1), Q(25, 64), Q(3, 4), Q(31, 64), Q(30, 64)),
    ],
)
def test_actual_staged_thol_signed_pressure_changes_successor_emission(
    factor, acceleration, source_epi, pressure, after, control_after
):
    graph = _source(factor)
    control = _source(factor)
    samples = _assert_source(graph, factor=factor)
    assert _assert_source(control, factor=factor) == samples
    _stage_thol(graph)
    _assert_pressure_only(
        graph,
        before=source_epi,
        pressure=pressure,
        acceleration=acceleration,
        samples=samples,
    )
    _assert_successor(graph, pressure=pressure, endpoint=after, samples=samples)
    _assert_successor(
        control,
        pressure=factor / 4,
        endpoint=control_after,
        samples=samples,
        thol=False,
    )
    assert (
        tuple(a - b for a, b in zip(_forms(graph), _forms(control), strict=True))
        == (acceleration / 64,) * 2
    )


def test_actual_native_thol_uses_episode_history_after_solver_subdivision():
    graph = _source(Q(3, 2), dt_min=1 / 16)
    samples = _assert_source(graph, factor=Q(3, 2))
    assert all(get_attr(graph.nodes[node], ALIAS_D2EPI) == 0.0 for node in NODES)
    # This is an explicit request policy; runtime still admits actual grammar.
    graph.graph["glyph_selector"] = lambda graph, node: "THOL"
    trace = []
    apply_native_operator_batch(graph, graph.graph["history"], decision_trace=trace)
    assert tuple(row.node for row in trace) == NODES
    for row in trace:
        assert row.requested_glyph == row.post_gate_glyph == row.applied_glyph == "THOL"
        assert row.status == "applied" and row.grammar_decision is not None
        assert row.physical_history_invalidated is False
        assert row.since_al == row.since_en == 1
    _assert_pressure_only(
        graph, before=Q(23, 64), pressure=Q(1, 4), acceleration=Q(-1), samples=samples
    )
    _assert_successor(graph, pressure=Q(1, 4), endpoint=Q(25, 64), samples=samples)


def test_actual_public_thol_reads_own_history_before_successor_episode():
    graph = _source(Q(3, 2))
    samples = _assert_source(graph, factor=Q(3, 2))
    for node in NODES:
        SelfOrganization()(graph, node, tau=1.0, validate_preconditions=True)
    _assert_pressure_only(
        graph, before=Q(23, 64), pressure=Q(1, 4), acceleration=Q(-1), samples=samples
    )
    _assert_successor(graph, pressure=Q(1, 4), endpoint=Q(25, 64), samples=samples)


def test_separate_episode_boundary_changes_thol_window_without_changing_source_path():
    graph = _source(Q(3, 2), split_second=True)
    unsplit = _source(Q(3, 2))
    samples = _assert_source(graph, factor=Q(3, 2), split_second=True)
    unsplit_samples = _assert_source(unsplit, factor=Q(3, 2))
    assert _forms(graph) == _forms(unsplit) == (Q(23, 64),) * 2
    assert _pressures(graph) == _pressures(unsplit) == (Q(3, 8),) * 2
    assert samples == (*unsplit_samples[:2], (Q(3, 16), Q(43, 128)), unsplit_samples[2])
    _stage_thol(graph)
    _assert_pressure_only(
        graph, before=Q(23, 64), pressure=Q(3, 8), acceleration=Q(0), samples=samples
    )
    _assert_successor(
        graph, pressure=Q(3, 8), endpoint=Q(26, 64), samples=samples, split_second=True
    )
