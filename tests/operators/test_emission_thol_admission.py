"""Acquired AL-history boundaries for actual THOL and successor Emission.

Initial coordinates, factors, durations and native requests are explicit inputs.
The physical samples and grammar contexts below come from real IL/OZ/AL actions.
Fault tests corrupt that acquired evidence explicitly; they do not claim another
real source. A supplied native request is not an autonomous occurrence law.
Plain primitive calls check pressure/domain compatibility only, without a claim
of full sequence admission; the native continuation uses actual runtime grammar.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics.selectors import apply_native_operator_batch
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
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
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.self_organization import SelfOrganization
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
T = 0.125


class _RequestThol:
    """Declared same-occasion requests, with no metric or history writes."""

    def __call__(self, graph, node):
        return "THOL"


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _forms(graph):
    return tuple(Q.from_float(_form(graph, node)) for node in NODES)


def _pressures(graph):
    return tuple(
        Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in NODES
    )


def _state(graph):
    """Actual nodal, support, clock and event/selection state, excluding adapters."""
    return deepcopy(
        (
            tuple((node, dict(data)) for node, data in graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            graph.graph["_t"],
            graph.graph["history"],
            graph.graph.get("hybrid_event_log"),
        )
    )


def _cohere_then_dissonate(graph, factor):
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = factor
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


def _episode(graph):
    return execute_emission_episode(graph, NODES, duration=T, method="euler", n_jobs=1)


def _acquired(*, episodes=2):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=59,
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
        EPI_LATENT_MAX=0.5,
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
        history={
            "since_AL": dict.fromkeys(NODES, 0),
            "since_EN": dict.fromkeys(NODES, 0),
        },
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        THOL_accel=0.125,
    )
    for node in NODES:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5,
            Si=0.5,
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _cohere_then_dissonate(graph, 2.0)
    assert _pressures(graph) == (Q(1, 2),) * 2
    assert all(
        read_grammar_state(graph.nodes[node]).history == ("IL", "OZ") for node in NODES
    )
    assert all("epi_time_history" not in graph.nodes[node] for node in NODES)
    _episode(graph)
    assert _forms(graph) == (Q(5, 16),) * 2
    if episodes == 2:
        _cohere_then_dissonate(graph, 1.5)
        assert _pressures(graph) == (Q(3, 8),) * 2
        _episode(graph)
        assert _forms(graph) == (Q(23, 64),) * 2
    for node in NODES:
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == episodes
        assert len(graph.nodes[node]["epi_time_history"]) == episodes + 1
    assert graph.graph["_t"] == T * episodes
    return graph


def _native_thol(graph):
    graph.graph["glyph_selector"] = _RequestThol()
    trace = []
    apply_native_operator_batch(graph, graph.graph["history"], decision_trace=trace)
    assert tuple(row.node for row in trace) == NODES
    for row in trace:
        assert row.requested_glyph == row.applied_glyph == "THOL"
        assert row.status == "applied"
        assert row.grammar_decision is not None
        assert row.physical_history_invalidated is False
        assert row.since_al == row.since_en == 1
    return trace


@pytest.mark.parametrize("route", ["public", "stage"])
def test_short_acquired_history_rejects_strict_thol_without_inventing_curvature(route):
    graph = _acquired(episodes=1)
    observation = observe_structural_acceleration(graph, CORE)
    assert not observation.available and observation.value is None
    assert observation.source == "epi_time_history"
    assert observation.history_length == 2
    assert observation.reason == "insufficient_history"
    assert get_attr(graph.nodes[CORE], ALIAS_D2EPI) == 4.0
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="2 samples"):
        if route == "public":
            SelfOrganization()(graph, CORE, tau=1.0, validate_preconditions=True)
        else:
            execute_self_organization_stage(
                graph, SelfOrganization(), NODES, tau=1.0, validate_preconditions=True
            )
    assert _state(graph) == before


@pytest.mark.parametrize("route", ["primitive", "native"])
def test_short_acquired_history_keeps_primitive_compatibility_zero_separate(route):
    graph = _acquired(episodes=1)
    physical = tuple(deepcopy(graph.nodes[node]["epi_time_history"]) for node in NODES)
    if route == "primitive":
        for node in NODES:
            apply_glyph(graph, node, "THOL")
    else:
        _native_thol(graph)
    assert _pressures(graph) == (Q(1, 2),) * 2
    assert _forms(graph) == (Q(5, 16),) * 2
    for node, samples in zip(NODES, physical, strict=True):
        assert graph.nodes[node]["epi_time_history"] == samples
        assert get_attr(graph.nodes[node], ALIAS_D2EPI) == 0.0
        observation = observe_structural_acceleration(graph, node)
        assert not observation.available and observation.value is None
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 1
    assert graph.graph["_t"] == T


@pytest.mark.parametrize("route", ["public", "primitive"])
@pytest.mark.parametrize("fault", ["live_endpoint", "timestamp", "sample_scalar"])
def test_acquired_history_faults_are_rejected_without_fallback_or_action(route, fault):
    graph = _acquired()
    observation = observe_structural_acceleration(graph, CORE)
    assert observation.available and observation.value == -1.0
    # Fault injection starts only after successful acquisition of all three samples.
    if fault == "live_endpoint":
        set_attr(graph.nodes[CORE], ALIAS_EPI, 0.375)
    else:
        samples = list(graph.nodes[CORE]["epi_time_history"])
        time, epi = samples[1]
        samples[1] = (samples[0][0], epi) if fault == "timestamp" else (time, True)
        graph.nodes[CORE]["epi_time_history"] = samples
    before = _state(graph)
    error = OperatorPreconditionError if route == "public" else TNFRValueError
    with pytest.raises(error):
        if route == "public":
            SelfOrganization()(graph, CORE, tau=1.0, validate_preconditions=True)
        else:
            apply_glyph(graph, CORE, "THOL")
    assert _state(graph) == before


@pytest.mark.parametrize("gain,pressure", [(0.375, Q(0)), (0.5, Q(-1, 8))])
def test_completed_native_thol_with_nonpositive_pressure_blocks_only_successor(
    gain, pressure
):
    graph = _acquired()
    graph.graph["GLYPH_FACTORS"]["THOL_accel"] = gain
    physical = tuple(deepcopy(graph.nodes[node]["epi_time_history"]) for node in NODES)
    _native_thol(graph)
    assert _pressures(graph) == (pressure,) * 2
    assert _forms(graph) == (Q(23, 64),) * 2
    for node, samples in zip(NODES, physical, strict=True):
        assert graph.nodes[node]["epi_time_history"] == samples
        assert read_grammar_state(graph.nodes[node]).history[-1] == "THOL"
        assert get_attr(graph.nodes[node], ALIAS_D2EPI) == -1.0
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    before_successor = _state(graph)
    with pytest.raises(ValueError, match="positive held form rate"):
        _episode(graph)
    assert _state(graph) == before_successor
    assert graph.graph["_t"] == 0.25


def test_explicit_native_thol_request_changes_real_successor_episode():
    graph = _acquired()
    physical = tuple(deepcopy(graph.nodes[node]["epi_time_history"]) for node in NODES)
    _native_thol(graph)
    assert _pressures(graph) == (Q(1, 4),) * 2
    assert _forms(graph) == (Q(23, 64),) * 2
    for node, samples in zip(NODES, physical, strict=True):
        assert graph.nodes[node]["epi_time_history"] == samples
        assert observe_structural_acceleration(graph, node).value == -1.0
        assert get_attr(graph.nodes[node], ALIAS_D2EPI) == -1.0
    successor = _episode(graph)
    assert successor.exact_rates == (Q(1, 4),) * 2
    assert successor.exact_epi_before == (Q(23, 64),) * 2
    assert successor.exact_epi_after == (Q(25, 64),) * 2
    assert graph.graph["_t"] == 0.375
    assert _forms(graph) == (Q(25, 64),) * 2
    for node in NODES:
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 3
        assert len(graph.nodes[node]["epi_time_history"]) == 4
        assert graph.graph["history"]["since_AL"][node] == 1
        assert graph.graph["history"]["since_EN"][node] == 1
        assert "sub_nodes" not in graph.nodes[node]
    assert tuple(graph) == NODES and tuple(graph.edges) == ((CORE, PEER),)
