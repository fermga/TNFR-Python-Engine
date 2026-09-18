"""Opt-in exact global reduction, fixed local law and real graph rollback."""

from collections import deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction as F
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_theta_attr
from tnfr.dynamics import coordination
from tnfr.metrics.trig import neighbor_phase_mean_list
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.utils import angle_diff


POLICY = "exact_components_v1"


def _graph(phases=(0.0, .25, .75, 1.25), *, order=None, adaptive=False):
    nodes = tuple(range(len(phases)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes if order is None else order)
    graph.add_edges_from((i, i+1) for i in range(len(nodes)-1))
    if len(nodes) > 2:
        graph.add_edge(0, len(nodes)-1)
    shared = {"nested": [1, {"values": [2, 3]}]}
    history = {"phase_state": deque(["stable"], maxlen=4), "phase_R": deque([.75], maxlen=4),
               "phase_disr": deque([0.0], maxlen=4), "phase_kG": [.0625], "phase_kL": [.125]}
    graph.graph.update(history=history, PHASE_K_GLOBAL=.0625, PHASE_K_LOCAL=.125,
                       PHASE_HISTORY_MAXLEN=4, PHASE_ADAPT={"enabled": adaptive}, shared=shared)
    for node in nodes:
        graph.nodes[node].update(theta=phases[node], phase=phases[node], EPI=.5+node/32,
                                 nu_f=1.0, delta_nfr=.125, glyph_history=deque(["IL"], maxlen=8), shared=shared)
    if nodes:
        graph.nodes[nodes[0]]["phase_state_alias"] = history["phase_state"]
    for u, v in graph.edges:
        graph.edges[u, v].update(weight=1.0, shared=shared)
    return graph


def _freeze(value, seen=None):
    """Value and alias projection of test-owned mutable graph state."""
    seen = {} if seen is None else seen
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    identity = id(value)
    if identity in seen:
        return ("alias", seen[identity])
    seen[identity] = len(seen)
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, Mapping):
        return ("mapping", tuple((_freeze(k, seen), _freeze(v, seen)) for k, v in value.items()))
    if isinstance(value, deque):
        return ("deque", value.maxlen, tuple(_freeze(v, seen) for v in value))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze(v, seen) for v in value))
    if is_dataclass(value):
        return (type(value).__name__, tuple((field.name, _freeze(getattr(value, field.name), seen)) for field in fields(value)))
    return ("runtime-owner", type(value).__name__, identity)


def _state(graph, cache=None):
    return _freeze((graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)),
                    tuple((node, tuple(graph.neighbors(node))) for node in graph), cache))


def _primitive_expectations(graph):
    cache = get_trig_cache(graph)
    nodes = tuple(graph)
    phases = tuple(float(get_theta_attr(graph.nodes[node])) for node in nodes)
    components = tuple((float(cache.cos[node]), float(cache.sin[node])) for node in nodes)
    sums = tuple(sum((F.from_float(pair[i]) for pair in components), F(0)) for i in (0, 1))
    scale = max(map(abs, sums))
    angle = None if scale == 0 else math.atan2(float(sums[1]/scale), float(sums[0]/scale))
    local = tuple(neighbor_phase_mean_list(tuple(graph.neighbors(node)), cache.cos, cache.sin, fallback=phase)
                  if graph.degree(node) else phase for node, phase in zip(nodes, phases, strict=True))
    return nodes, phases, components, sums, angle, local


def _assert_aliases(graph, shared, history, cache, manager):
    assert graph.graph["shared"] is shared and graph.graph["history"] is history
    assert graph.nodes[0]["phase_state_alias"] is history["phase_state"]
    assert list(history["phase_state"]) == ["stable"] and history["phase_state"].maxlen == 4
    assert all(data["shared"] is shared for _, data in graph.nodes(data=True))
    assert all(data["shared"] is shared for _, _, data in graph.edges(data=True))
    assert graph.graph["_edge_cache_manager"] is manager
    assert get_trig_cache(graph) is cache


@pytest.mark.parametrize("scalar", (False, True))
def test_explicit_gains_exact_global_sum_and_unchanged_local_law(monkeypatch, scalar):
    graph = _graph((.125, -.25, .75, 1.25))
    nodes, phases, components, sums, target, local = _primitive_expectations(graph)
    if scalar:
        monkeypatch.setattr(coordination, "np", None)
    report = coordination.coordinate_global_local_phase(
        graph, global_force=.0625, local_force=.125, global_reduction=POLICY, n_jobs=1)
    assert report.version == POLICY and report.status == "applied"
    assert report.nodes == nodes and report.primitive_phases == phases
    assert report.resultant.components == components
    assert (report.resultant.real_sum, report.resultant.imag_sum) == sums
    assert report.global_target == target and report.global_term_active
    assert report.local_targets == local
    assert report.requested_global_force == report.effective_global_force == .0625
    assert report.requested_local_force == report.effective_local_force == .125
    expected = tuple(th+.0625*angle_diff(target, th)+.125*angle_diff(th_l, th)
                     for th, th_l in zip(phases, local, strict=True))
    assert report.raw_proposals == expected
    assert report.realized_phases == tuple(get_theta_attr(graph.nodes[node]) for node in nodes)
    assert report.neighbor_order == tuple((node, tuple(graph.neighbors(node))) for node in nodes)
    for node in nodes:
        assert graph.nodes[node]["theta"] == graph.nodes[node]["phase"]
        assert (graph.nodes[node]["EPI"], graph.nodes[node]["nu_f"], graph.nodes[node]["delta_nfr"]) == (
            .5+node/32, 1.0, .125)


def test_default_and_explicit_legacy_are_identical_and_return_none():
    implicit, explicit = _graph(), _graph()
    assert coordination.coordinate_global_local_phase(implicit, .0625, .125) is None
    assert coordination.coordinate_global_local_phase(explicit, .0625, .125, global_reduction="legacy") is None
    assert tuple(get_theta_attr(implicit.nodes[n]) for n in implicit) == tuple(get_theta_attr(explicit.nodes[n]) for n in explicit)
    assert implicit.graph["history"] == explicit.graph["history"]
    assert implicit.graph["PHASE_K_GLOBAL"] == explicit.graph["PHASE_K_GLOBAL"]
    assert implicit.graph["PHASE_K_LOCAL"] == explicit.graph["PHASE_K_LOCAL"]


def test_adaptive_gain_policy_and_local_targets_match_legacy():
    legacy, exact = _graph(adaptive=True), _graph(adaptive=True)
    _, phases, _, sums, target, local = _primitive_expectations(exact)
    coordination.coordinate_global_local_phase(legacy)
    report = coordination.coordinate_global_local_phase(exact, global_reduction=POLICY)
    assert report.requested_global_force is report.requested_local_force is None
    assert exact.graph["history"] == legacy.graph["history"]
    assert report.effective_global_force == legacy.graph["PHASE_K_GLOBAL"]
    assert report.effective_local_force == legacy.graph["PHASE_K_LOCAL"]
    assert report.local_targets == local
    assert (report.resultant.real_sum, report.resultant.imag_sum) == sums
    expected = tuple(th+report.effective_global_force*angle_diff(target, th)
                     + report.effective_local_force*angle_diff(th_l, th)
                     for th, th_l in zip(phases, local, strict=True))
    assert report.raw_proposals == expected


def test_inactive_global_term_preserves_local_only_legacy_proposals():
    legacy, exact = _graph(), _graph()
    coordination.coordinate_global_local_phase(legacy, global_force=0.0, local_force=.125)
    report = coordination.coordinate_global_local_phase(
        exact, global_force=0.0, local_force=.125, global_reduction=POLICY)
    assert report.global_target is None and not report.global_term_active
    assert report.resultant.angle is not None
    assert report.realized_phases == tuple(get_theta_attr(legacy.nodes[n]) for n in legacy)
    assert exact.graph["history"] == legacy.graph["history"]


def test_portable_retained_phase_vector_is_order_invariant_for_fixed_components():
    # These primitive floats are copied from the retained first coordination
    # boundary. Topology and gains here are synthetic; this is not a replay of
    # the archived execution or a gauge/backend invariance claim.
    phases = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345,
              -3.141592653589793, -2.356194490192345, -1.5707963267948966,
              -0.7853981633974492)*2
    orders = (tuple(range(16)), tuple(reversed(range(16))), tuple(range(1, 16))+(0,))
    reports, component_maps, neighbors = [], [], []
    for order in orders:
        graph = _graph(phases, order=order)
        nodes, _, components, sums, _, _ = _primitive_expectations(graph)
        component_maps.append(dict(zip(nodes, components, strict=True)))
        neighbors.append({node: tuple(graph.neighbors(node)) for node in graph})
        report = coordination.coordinate_global_local_phase(
            graph, global_force=.0625, local_force=.125, global_reduction=POLICY)
        assert (report.resultant.real_sum, report.resultant.imag_sum) == sums
        assert not report.resultant.joint_zero
        assert max(map(abs, sums)) < F(1, 10**12)
        reports.append(report)
    assert component_maps[0] == component_maps[1] == component_maps[2]
    assert neighbors[0] == neighbors[1] == neighbors[2]
    for other in reports[1:]:
        assert other.resultant.real_sum == reports[0].resultant.real_sum
        assert other.resultant.imag_sum == reports[0].resultant.imag_sum
        assert other.global_target == reports[0].global_target
        assert dict(zip(other.nodes, other.local_targets, strict=True)) == dict(zip(reports[0].nodes, reports[0].local_targets, strict=True))
        assert dict(zip(other.nodes, other.raw_proposals, strict=True)) == dict(zip(reports[0].nodes, reports[0].raw_proposals, strict=True))


def test_exact_joint_zero_with_active_adaptive_gain_restores_owned_state_and_aliases():
    graph = _graph((0.0, 0.0, math.pi, -math.pi), adaptive=True)
    cache = get_trig_cache(graph)
    assert sum(map(F.from_float, cache.cos.values()), F(0)) == 0
    assert sum(map(F.from_float, cache.sin.values()), F(0)) == 0
    shared, history = graph.graph["shared"], graph.graph["history"]
    manager = graph.graph["_edge_cache_manager"]
    before = _state(graph, cache)
    with pytest.raises(coordination.UndefinedGlobalPhaseError):
        coordination.coordinate_global_local_phase(graph, global_reduction=POLICY)
    assert _state(graph, cache) == before
    _assert_aliases(graph, shared, history, cache, manager)


@pytest.mark.parametrize("scalar", (False, True))
def test_exact_joint_zero_with_inactive_global_gain_uses_no_direction(monkeypatch, scalar):
    graph = _graph((0.0, 0.0, math.pi, -math.pi))
    _, phases, _, sums, _, local = _primitive_expectations(graph)
    assert sums == (F(0), F(0))
    if scalar:
        monkeypatch.setattr(coordination, "np", None)
    report = coordination.coordinate_global_local_phase(
        graph, global_force=0.0, local_force=.125, global_reduction=POLICY, n_jobs=1)
    assert report.resultant.joint_zero and report.resultant.angle is None
    assert report.global_target is None and not report.global_term_active
    assert report.raw_proposals == tuple(th+.125*angle_diff(local_target, th)
                                         for th, local_target in zip(phases, local, strict=True))


def test_empty_graph_has_distinct_evidence_not_a_zero_resultant_error():
    report = coordination.coordinate_global_local_phase(
        _graph(()), global_force=.0625, local_force=.125, global_reduction=POLICY)
    assert report.status == "empty_graph" and report.resultant is None
    assert report.nodes == report.primitive_phases == report.local_targets == report.raw_proposals == report.realized_phases == ()
    assert report.global_target is None


@pytest.mark.parametrize("policy", ("exact", "", None, True))
def test_unknown_policy_fails_before_any_graph_write(policy):
    graph = _graph(adaptive=True)
    before = _state(graph)
    with pytest.raises(ValueError, match="global_reduction"):
        coordination.coordinate_global_local_phase(graph, global_reduction=policy)
    assert _state(graph) == before


def test_real_transaction_precedes_caller_gain_conversion():
    graph = _graph()
    before = _state(graph)
    consumed = []

    class MutatingGain(float):
        def __float__(self):
            consumed.append("conversion")
            graph.graph["shared"]["nested"].append("gain conversion")
            graph.graph["PHASE_K_GLOBAL"] = 999
            graph.add_node("leaked")
            raise ValueError("deliberate conversion failure")

    with pytest.raises(ValueError):
        coordination.coordinate_global_local_phase(graph, global_force=MutatingGain(.0625), global_reduction=POLICY)
    assert consumed == ["conversion"]
    assert _state(graph) == before


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_late_commit_failure_restores_real_caches_metadata_and_topology(monkeypatch, failure_type):
    graph = _graph(adaptive=True)
    cache = get_trig_cache(graph)
    shared, history = graph.graph["shared"], graph.graph["history"]
    manager = graph.graph["_edge_cache_manager"]
    before = _state(graph, cache)
    original = coordination.set_theta
    calls = []

    def fail_after_second_commit(subject, node, value):
        original(subject, node, value)
        calls.append(node)
        if len(calls) == 2:
            shared["nested"][1]["values"].append(999)
            history["phase_R"].append(999)
            cache.cos[0] = 999.0
            cache.theta_values[:] = 999.0
            subject.remove_edge(0, 1)
            subject.add_node("leaked", theta=99.0)
            raise failure_type("deliberate late phase failure")

    monkeypatch.setattr(coordination, "set_theta", fail_after_second_commit)
    with pytest.raises(failure_type, match="late phase failure"):
        coordination.coordinate_global_local_phase(graph, global_reduction=POLICY)
    assert calls == [0, 1]
    assert _state(graph, cache) == before
    _assert_aliases(graph, shared, history, cache, manager)


def test_scalar_process_worker_uses_same_exact_global_and_local_proposals(monkeypatch):
    sequential, parallel = _graph(), _graph()
    _primitive_expectations(sequential)
    _primitive_expectations(parallel)
    monkeypatch.setattr(coordination, "np", None)
    first = coordination.coordinate_global_local_phase(sequential, .0625, .125, n_jobs=1, global_reduction=POLICY)
    second = coordination.coordinate_global_local_phase(parallel, .0625, .125, n_jobs=2, global_reduction=POLICY)
    assert second.resultant == first.resultant
    assert second.local_targets == first.local_targets
    assert second.raw_proposals == first.raw_proposals
    assert second.realized_phases == first.realized_phases


@pytest.mark.parametrize("source,value", (("initial", math.inf), ("up", math.inf), ("kG_max", math.nan)))
def test_nonfinite_gain_cannot_be_hidden_by_adaptive_clamping(source, value):
    graph = _graph(adaptive=True)
    if source == "initial":
        graph.graph["PHASE_K_GLOBAL"] = value
    else:
        graph.graph["PHASE_ADAPT"][source] = value
    before = _state(graph)
    with pytest.raises(ValueError):
        coordination.coordinate_global_local_phase(graph, global_reduction=POLICY)
    assert _state(graph) == before


def test_failed_worker_count_conversion_restores_previous_gain_and_history_writes():
    graph = _graph(adaptive=True)
    before = _state(graph)
    consumed = []

    class MutatingJobs:
        def __int__(self):
            consumed.append(True)
            graph.graph["shared"]["nested"].append("jobs conversion")
            graph.nodes[0]["theta"] = 999
            raise ValueError("deliberate jobs conversion failure")

    with pytest.raises(ValueError, match="jobs conversion"):
        coordination.coordinate_global_local_phase(graph, n_jobs=MutatingJobs(), global_reduction=POLICY)
    assert consumed == [True]
    assert _state(graph) == before


def test_final_evidence_failure_rolls_back_all_successful_phase_commits(monkeypatch):
    graph = _graph(adaptive=True)
    cache = get_trig_cache(graph)
    before = _state(graph, cache)
    phases = tuple(get_theta_attr(graph.nodes[node]) for node in graph)
    seen = []

    def fail_evidence(**values):
        seen.append(values)
        assert tuple(get_theta_attr(graph.nodes[node]) for node in graph) == values["realized_phases"]
        assert values["realized_phases"] != phases
        graph.graph["shared"]["nested"].append("evidence failure")
        raise RuntimeError("deliberate final evidence failure")

    monkeypatch.setattr(coordination, "GlobalPhaseCoordinationEvidence", fail_evidence)
    with pytest.raises(RuntimeError, match="final evidence"):
        coordination.coordinate_global_local_phase(graph, global_reduction=POLICY)
    assert len(seen) == 1
    assert _state(graph, cache) == before


def test_unknown_policy_does_not_invoke_arbitrary_equality():
    graph = _graph()
    before = _state(graph)

    class MutatingPolicy:
        def __eq__(self, other):
            graph.graph["unexpected_policy_equality"] = True
            raise AssertionError("unsupported policy equality must not execute")

    with pytest.raises(ValueError, match="global_reduction"):
        coordination.coordinate_global_local_phase(graph, global_reduction=MutatingPolicy())
    assert _state(graph) == before
