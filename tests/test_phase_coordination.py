"""NumPy and sequential coordination share the canonical circular displacement."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_theta_attr
from tnfr.constants import DEFAULTS, STATE_DISSONANT, STATE_TRANSITION
from tnfr.dynamics import coordination
from tnfr.utils import angle_diff


def _run(monkeypatch, phases, *, global_force, local_force, vectorized, connected=True):
    graph = nx.path_graph(len(phases)) if connected else nx.empty_graph(len(phases))
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(
            theta=phase, phase=phase, EPI=0.75, nu_f=1.0, delta_nfr=0.2
        )
    old_edges = tuple(graph.edges)
    with monkeypatch.context() as context:
        context.setattr(coordination, "np", np if vectorized else None)
        coordination.coordinate_global_local_phase(
            graph,
            global_force=global_force,
            local_force=local_force,
            n_jobs=1,
        )
    assert tuple(graph.edges) == old_edges
    for node in graph:
        assert tuple(
            graph.nodes[node][name] for name in ("EPI", "nu_f", "delta_nfr")
        ) == (
            0.75,
            1.0,
            0.2,
        )
    return tuple(get_theta_attr(graph.nodes[node]) for node in graph)


@pytest.mark.parametrize(
    "global_force,local_force", [(0.1, 0.0), (0.0, 0.1), (0.07, 0.13)]
)
@pytest.mark.parametrize("upper_representative", [True, False])
def test_seam_coordination_uses_short_arcs_in_each_channel(
    monkeypatch,
    global_force,
    local_force,
    upper_representative,
):
    epsilon = 0.1
    phases = (epsilon, math.tau - epsilon if upper_representative else -epsilon)
    kwargs = dict(global_force=global_force, local_force=local_force)
    vector = _run(monkeypatch, phases, vectorized=True, **kwargs)
    scalar = _run(monkeypatch, phases, vectorized=False, **kwargs)
    # Global mean is direction zero; each singleton local neighbor is 2*epsilon
    # away. Both updates must approach that mean along the short arc.
    displacement = epsilon * (global_force + 2 * local_force)
    for before, actual, fallback, expected in zip(
        phases,
        vector,
        scalar,
        (-displacement, displacement),
        strict=True,
    ):
        assert angle_diff(actual, before) == pytest.approx(expected, abs=2e-15)
        assert angle_diff(actual, fallback) == pytest.approx(0.0, abs=2e-15)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("channel", ["global", "local"])
def test_antipodal_tie_matches_shared_signed_pi_convention(monkeypatch, sign, channel):
    # A majority at the antipode makes the global direction well-defined.
    phases = (
        (0.0, sign * math.pi, sign * math.pi)
        if channel == "global"
        else (
            0.0,
            sign * math.pi,
        )
    )
    kwargs = dict(
        global_force=0.25 if channel == "global" else 0.0,
        local_force=0.25 if channel == "local" else 0.0,
        connected=channel == "local",
    )
    vector = _run(monkeypatch, phases, vectorized=True, **kwargs)
    scalar = _run(monkeypatch, phases, vectorized=False, **kwargs)
    assert angle_diff(sign * math.pi, 0.0) == sign * math.pi
    assert angle_diff(vector[0], phases[0]) == pytest.approx(
        sign * math.pi / 4, abs=2e-15
    )
    for actual, fallback in zip(vector, scalar, strict=True):
        assert angle_diff(actual, fallback) == pytest.approx(0.0, abs=2e-15)


@pytest.mark.parametrize("phases", [(), (math.tau - 0.1,)])
def test_empty_and_singleton_global_targets_keep_the_vector_shape(monkeypatch, phases):
    kwargs = dict(global_force=0.1, local_force=0.1)
    vector = _run(monkeypatch, phases, vectorized=True, **kwargs)
    scalar = _run(monkeypatch, phases, vectorized=False, **kwargs)
    assert len(vector) == len(phases)
    for actual, fallback, before in zip(vector, scalar, phases, strict=True):
        assert angle_diff(actual, fallback) == pytest.approx(0.0, abs=2e-15)
        assert angle_diff(actual, before) == pytest.approx(0.0, abs=2e-15)


@pytest.mark.parametrize(
    "policy,value",
    [
        ("legacy", "0.5"),
        ("legacy", math.inf),
        ("exact_components_v1", True),
        ("exact_components_v1", Fraction(1, 2**2000)),
    ],
)
def test_raw_phase_admission_precedes_cache_and_history_work(policy, value):
    graph = nx.path_graph(2)
    graph.nodes[0]["theta"] = 0.0
    graph.nodes[1].update(theta=value, phase=0.5)
    before_nodes, before_graph = deepcopy(dict(graph.nodes(data=True))), deepcopy(
        graph.graph
    )
    with pytest.raises((TypeError, ValueError)):
        coordination.coordinate_global_local_phase(
            graph, 0.0, 0.1, global_reduction=policy
        )
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


@pytest.mark.parametrize("vectorized", [False, True])
def test_legacy_late_proposal_overflow_cannot_write_any_phase(monkeypatch, vectorized):
    graph = nx.path_graph(3)
    for node, phase in enumerate((0.0, 0.0, 3.0)):
        graph.nodes[node].update(theta=phase, phase=phase)
    before = deepcopy(dict(graph.nodes(data=True)))
    monkeypatch.setattr(coordination, "np", np if vectorized else None)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="raw phase proposal"):
            coordination.coordinate_global_local_phase(graph, 0.0, 1e308, n_jobs=1)
    assert dict(graph.nodes(data=True)) == before


def test_local_phase_uses_unique_outgoing_support_even_at_zero_conductance():
    graph = nx.MultiDiGraph()
    for node, phase in enumerate((0.0, 0.0, math.pi / 2, math.pi)):
        graph.add_node(node, theta=phase)
    graph.add_edge(0, 1, weight=1000.0)
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(0, 2, weight=0.0)
    graph.add_edge(3, 0, weight=1.0)
    report = coordination.coordinate_global_local_phase(
        graph, 0.0, 0.2, global_reduction="exact_components_v1"
    )
    assert report.neighbor_order == ((0, (1, 2)), (1, ()), (2, ()), (3, (0,)))
    assert graph.nodes[0]["theta"] == pytest.approx(math.pi / 20)
    assert graph.nodes[1]["theta"] == 0.0
    assert graph.nodes[2]["theta"] == math.pi / 2


def test_adaptive_midpoint_remains_finite_when_the_endpoint_sum_overflows():
    config = {**DEFAULTS["PHASE_ADAPT"], "up": 1.0, "down": 1.0}
    config.update(kG_min=1e308, kG_max=1.6e308, kL_min=1e308, kL_max=1.6e308)
    first, second = coordination._smooth_adjust_k(
        1e308, 1e308, STATE_TRANSITION, config
    )
    assert first == second == 1.3e308


def test_adaptive_convex_step_survives_overflow_of_the_endpoint_difference():
    config = {**DEFAULTS["PHASE_ADAPT"], "up": 0.9, "kG_min": 0.0, "kG_max": 1e308}
    first, _ = coordination._smooth_adjust_k(-1e308, 0.1, STATE_DISSONANT, config)
    assert first == pytest.approx(8e307)
