"""Execution paths must preserve the same nodal-pressure channels."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import fused_dnfr
from tnfr.dynamics.dnfr import default_compute_delta_nfr


def _pressure(graph, *, vectorized, n_jobs=None):
    copy = graph.copy()
    copy.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(copy, n_jobs=n_jobs)
    return np.array([get_attr(copy.nodes[node], ALIAS_DNFR) for node in copy])


def _use_pure_python(monkeypatch):
    import tnfr.dynamics.dnfr as dnfr_module
    import tnfr.mathematics.unified_numerical as numerical_module

    monkeypatch.setattr(dnfr_module, "np", None)
    monkeypatch.setattr(numerical_module, "np", None)
    monkeypatch.setattr(numerical_module, "NUMPY_AVAILABLE", False)


@pytest.mark.parametrize("channel", ["phase", "epi", "vf", "topo"])
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_fallback_matches_fused_channels(graph_type, channel):
    """Weighted successors, loops, parallel arcs and sinks retain semantics."""
    graph = graph_type()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from(
        [(0, 0, 1.0), (0, 1, 0.5), (0, 2, 1.5), (0, 1, 0.25), (1, 3, 0.0)]
    )
    graph.graph["_dnfr_weights"] = {
        key: float(key == channel) for key in ("phase", "epi", "vf", "topo")
    }
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, node * 0.25)
        set_attr(graph.nodes[node], ALIAS_THETA, node * 0.2)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0 + node * 0.1)
    expected = _pressure(graph, vectorized=True)
    assert _pressure(graph, vectorized=False) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n_jobs", [None, 2])
@pytest.mark.parametrize("pure_python", [False, True])
def test_fallback_weighted_epi_matches_random_walk_laplacian(
    n_jobs, pure_python, monkeypatch
):
    """EPI pressure is D^-1 W EPI - EPI; outgoing-isolated nodes have zero."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 0.5), (0, 2, 1.5)])
    graph.graph["_dnfr_weights"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, float(node))
        set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0)
    if pure_python:
        _use_pure_python(monkeypatch)
    assert _pressure(graph, vectorized=False, n_jobs=n_jobs) == pytest.approx(
        [1.75, 0.0, 0.0]
    )


@pytest.mark.parametrize("phase", (math.pi, -math.pi, 1e-16))
@pytest.mark.parametrize("path", ("fused", "fallback", "python"))
def test_single_neighbor_phase_retains_signed_branch_and_small_displacement(
    phase, path, monkeypatch
):
    """A single neighbor has a unique target; wrapping must not erase its sign."""
    graph = nx.path_graph(2)
    graph.graph["DNFR_WEIGHTS"] = dict(phase=1.0, epi=0.0, vf=0.0, topo=0.0)
    for node, angle in enumerate((0.0, phase)):
        graph.nodes[node].update(theta=angle, EPI=0.5, nu_f=1.0)
    if path == "python":
        _use_pure_python(monkeypatch)
    target_gap = math.atan2(math.sin(phase), math.cos(phase))
    expected = np.array((target_gap, -target_gap)) / math.pi
    # Relative tolerance retains the nonzero sub-ULP-of-pi case. An absolute
    # tolerance would silently accept its complete disappearance.
    np.testing.assert_allclose(
        _pressure(graph, vectorized=path == "fused"),
        expected,
        rtol=4e-16,
        atol=0.0,
    )


@pytest.mark.parametrize("perturbation", (0.0, 1e-13, -1e-13))
@pytest.mark.parametrize("path", ("fused", "fallback", "python"))
def test_small_nonzero_resultant_is_not_silently_thresholded(
    perturbation, path, monkeypatch
):
    """Zero of the accumulated pair and a small nonzero direction are distinct."""
    phases = (0.7, perturbation, 0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi)
    graph = nx.DiGraph((0, neighbor) for neighbor in range(1, len(phases)))
    graph.graph["DNFR_WEIGHTS"] = dict(phase=1.0, epi=0.0, vf=0.0, topo=0.0)
    for node, phase in enumerate(phases):
        graph.nodes[node].update(theta=phase, EPI=0.5, nu_f=1.0)
    if path == "python":
        _use_pure_python(monkeypatch)
    # Reproduce only the declared ordered floating component sums. Their zero
    # is not the exact mathematical resultant of the represented phases.
    real = imaginary = 0.0
    for phase in phases[1:]:
        real += float(np.cos(phase))
        imaginary += float(np.sin(phase))
    assert real == 0.0
    if perturbation == 0.0:
        assert imaginary == 0.0
        expected = 0.0  # Explicit center fallback, not an inferred direction.
    else:
        assert 0 < abs(imaginary) / 6 < 1e-12
        target = math.copysign(math.pi / 2, perturbation)
        raw = target - phases[0]
        expected = math.atan2(math.sin(raw), math.cos(raw)) / math.pi
    np.testing.assert_allclose(
        _pressure(graph, vectorized=path == "fused"),
        [expected] + [0.0] * 6,
        rtol=4e-16,
        atol=0.0,
    )


@pytest.mark.parametrize("phase", (math.pi, -math.pi, 1e-16))
@pytest.mark.parametrize("execution", ("numpy", "kernel", "numba"))
def test_large_fused_dispatch_keeps_signed_single_neighbor_phase(
    phase, execution, monkeypatch
):
    """Exercise the >100-edge dispatch without claiming compilation if absent."""
    source = np.arange(0, 208, 2, dtype=np.intp)
    target = source + 1
    phases = np.zeros(208)
    phases[target] = phase
    if execution == "numba" and not fused_dnfr._NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed; the kernel path is tested separately")
    if execution == "kernel":
        monkeypatch.setattr(fused_dnfr, "_NUMBA_AVAILABLE", True)
        monkeypatch.setattr(
            fused_dnfr,
            "_compute_canonical_gradients_jit",
            fused_dnfr._compute_canonical_gradients_jit_kernel,
        )
    expected = np.zeros(208)
    expected[source] = math.atan2(math.sin(phase), math.cos(phase)) / math.pi
    result = fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=source,
        edge_dst=target,
        phase=phases,
        epi=np.full(208, 0.5),
        vf=np.ones(208),
        weights={"w_phase": 1.0},
        accumulate_both_directions=False,
        use_jit=execution != "numpy",
    )
    np.testing.assert_allclose(result, expected, rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("phase", (1e308, -1e308))
@pytest.mark.parametrize("path", ("fused", "fallback", "python"))
def test_uniform_large_raw_phase_has_no_single_neighbor_pressure(
    phase, path, monkeypatch
):
    graph = nx.DiGraph([(0, 1), (1, 0)])
    graph.add_node(2)  # An isolated huge center must not gain phase pressure.
    graph.add_edges_from((3, neighbor) for neighbor in range(4, 10))
    graph.graph["DNFR_WEIGHTS"] = dict(phase=1.0, epi=0.0, vf=0.0, topo=0.0)
    for node in graph:
        graph.nodes[node].update(theta=phase, EPI=0.5, nu_f=1.0)
    for node, angle in zip(
        range(4, 10), (0.0, 0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi)
    ):
        graph.nodes[node]["theta"] = angle
    if path == "python":
        _use_pure_python(monkeypatch)
    # Identical represented input phases give identical neighbor and center
    # phasors. No reduction by a binary64 substitute for 2*pi is needed.
    # Row 3 also retains zero under the joint-zero center-fallback extension.
    np.testing.assert_array_equal(_pressure(graph, vectorized=path == "fused"), 0.0)


@pytest.mark.parametrize("phase", (1e308, -1e308))
@pytest.mark.parametrize("execution", ("numpy", "kernel", "numba"))
def test_large_raw_fused_dispatch_uses_the_same_center_phasor(
    phase, execution, monkeypatch
):
    size = 104
    source = np.arange(size, dtype=np.intp)
    target = (source + 1) % size
    if execution == "numba" and not fused_dnfr._NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed; the kernel path is tested separately")
    if execution == "kernel":
        monkeypatch.setattr(fused_dnfr, "_NUMBA_AVAILABLE", True)
        monkeypatch.setattr(
            fused_dnfr,
            "_compute_canonical_gradients_jit",
            fused_dnfr._compute_canonical_gradients_jit_kernel,
        )
    result = fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=source,
        edge_dst=target,
        phase=np.full(size, phase),
        epi=np.full(size, 0.5),
        vf=np.ones(size),
        weights={"w_phase": 1.0},
        accumulate_both_directions=False,
        use_jit=execution != "numpy",
    )
    np.testing.assert_array_equal(result, 0.0)
