"""IL and pressure share certified geometry without changing channel support."""

from fractions import Fraction
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import dnfr, fused_dnfr
from tnfr.mathematics._phase_midpoint import certified_two_neighbor_phase
from tnfr.operators._coherence_stage_kernel import propose_coherence_phase


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    graph.add_edge(0, 1, weight=0.0)
    graph.add_edge(0, 2, weight=3.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=7.0)
    phases = (3.1414074614528316, 4716575516971799 / 2**51, 2358183504581023 / 2**49)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, phases[node])
        set_attr(graph.nodes[node], ALIAS_EPI, .5 + node / 8)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0 + node / 4)
        set_attr(graph.nodes[node], ALIAS_DNFR, .125)
    graph.graph["_dnfr_weights"] = {key: float(key == "phase") for key in ("phase", "epi", "vf", "topo")}
    return graph, phases


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
@pytest.mark.parametrize("vectorized,n_jobs", ((True, None), (False, None), (False, 2)))
def test_all_pressure_dispatches_share_il_delta_for_two_support_neighbors(graph_type, vectorized, n_jobs):
    graph, phases = _graph(graph_type)
    certificate = certified_two_neighbor_phase(*phases)
    assert certificate is not None
    proposal = propose_coherence_phase(graph, 0, .3)
    assert proposal.delta_theta == certificate.delta
    assert proposal.theta_network == certificate.mean
    assert proposal.method == "exact_two_neighbor_midpoint"
    graph.graph["vectorized_dnfr"] = vectorized
    before = tuple(dict(graph.nodes[node]) for node in graph)
    dnfr.default_compute_delta_nfr(graph, n_jobs=n_jobs)
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == certificate.delta / math.pi
    for node, previous in zip(graph, before, strict=True):
        for aliases in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF):
            assert get_attr(graph.nodes[node], aliases) == get_attr(previous, aliases)


@pytest.mark.parametrize("n_jobs", (None, 2))
def test_scalar_without_numpy_uses_the_same_certified_phase(n_jobs, monkeypatch):
    import tnfr.mathematics.unified_numerical as numerical

    graph, phases = _graph(nx.DiGraph)
    certificate = certified_two_neighbor_phase(*phases)
    graph.graph["vectorized_dnfr"] = False
    monkeypatch.setattr(dnfr, "np", None)
    monkeypatch.setattr(numerical, "np", None)
    monkeypatch.setattr(numerical, "NUMPY_AVAILABLE", False)
    dnfr.default_compute_delta_nfr(graph, n_jobs=n_jobs)
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == certificate.delta / math.pi


def test_delta_is_not_reconstructed_from_independently_rounded_mean():
    graph, phases = _graph()
    proposal = propose_coherence_phase(graph, 0, .3)
    exact = (Fraction(phases[1]) + Fraction(phases[2])) / 2 - Fraction(phases[0])
    assert proposal.delta_theta == float(exact)
    assert proposal.delta_theta != proposal.theta_network - phases[0]
    assert proposal.theta_after == (phases[0] + .3 * float(exact)) % math.tau


@pytest.mark.parametrize("phases", ((0.0, 2.0, 4.0), (1.0, math.tau, .75)))
def test_ineligible_neighbors_keep_legacy_il_path(phases):
    from tnfr.metrics.trig import neighbor_phase_mean_list
    from tnfr.utils import angle_diff

    graph, _ = _graph()
    for node, phase in enumerate(phases):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    cosine = {i: math.cos(p) for i, p in enumerate(phases)}
    sine = {i: math.sin(p) for i, p in enumerate(phases)}
    expected_mean = neighbor_phase_mean_list((1, 2), cosine, sine, fallback=phases[0]) % math.tau
    proposal = propose_coherence_phase(graph, 0, .3)
    assert proposal.method == "phasor"
    assert proposal.theta_network == expected_mean
    assert proposal.delta_theta == angle_diff(expected_mean, phases[0])


def test_three_neighbor_il_keeps_phasor_semantics():
    graph, _ = _graph()
    graph.add_node(3, theta=3.0)
    graph.add_edge(0, 3)
    assert propose_coherence_phase(graph, 0, .3).method == "phasor"


@pytest.mark.parametrize("reverse", (False, True))
def test_fused_actual_contribution_count_and_zero_weight_phase_support(reverse):
    phases = np.array([1.0, 1.0 - 2**-25, 1.0 + 2**-24])
    src, dst = np.array([0, 0]), np.array([1, 2])
    if reverse:
        src, dst = dst, src
    actual = fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=src, edge_dst=dst, phase=phases, epi=np.ones(3), vf=np.ones(3),
        edge_weight=np.array([0.0, 123.0]), weights={"w_phase": 1.0},
        accumulate_both_directions=reverse, use_jit=False,
    )
    expected = certified_two_neighbor_phase(*(float(p) for p in phases))
    assert actual[0] == expected.delta / math.pi


def test_large_dispatch_preserves_shared_phase_even_if_jit_has_different_rounding(monkeypatch):
    size = 102
    src, dst = np.arange(size), (np.arange(size) + 1) % size
    phase = np.array([1.0 + (i % 3) * 2**-24 for i in range(size)])
    calls = []

    def coarse_jit(*args):
        calls.append(True)
        args[-1][:] = 42.0

    monkeypatch.setattr(fused_dnfr, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(fused_dnfr, "_compute_canonical_gradients_jit", coarse_jit)
    inputs = dict(
        edge_src=src, edge_dst=dst, phase=phase, epi=np.full(size, .5),
        vf=np.ones(size), weights={"w_phase": 1.0}, accumulate_both_directions=True,
    )
    expected = fused_dnfr.compute_fused_gradients_symmetric(**inputs, use_jit=False)
    actual = fused_dnfr.compute_fused_gradients_symmetric(**inputs, use_jit=True)
    assert calls == [True]
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("amplitude", (0.0, 2**-12, -2**-12))
def test_c6_exact_edge_cancellation_gives_local_rounding_bound(amplitude):
    phases = tuple(i * math.pi / 3 + amplitude * (1 if i % 2 == 0 else -1)
                   for i in range(6))
    phases = tuple(value % math.tau for value in phases)
    certificates = tuple(certified_two_neighbor_phase(phases[i], phases[(i - 1) % 6], phases[(i + 1) % 6])
                         for i in range(6))
    assert all(certificate is not None for certificate in certificates)
    # The exact affine numerators cancel; no pressure projection is performed.
    assert sum(c.delta_rational for c in certificates) == 0
    assert sum(c.delta_pi_coefficient for c in certificates) == 0
    pi = Fraction(math.pi)
    gradient = tuple(c.delta / math.pi for c in certificates)
    bound = sum((Fraction(math.ulp(c.delta)) / (2 * pi) + Fraction(math.ulp(g)) / 2
                 for c, g in zip(certificates, gradient, strict=True)), Fraction(0)) / 6
    assert abs(sum(map(Fraction, gradient)) / 6) <= bound
