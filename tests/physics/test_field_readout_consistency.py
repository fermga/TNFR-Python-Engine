"""Read-only fields must implement the same neighborhood means on every graph."""

from __future__ import annotations

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

import tnfr.physics.extended as extended
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.extended import compute_dnfr_flux, compute_phase_current
from tnfr.physics.fields import measure_phase_symmetry
from tnfr.physics.telemetry import compute_structural_telemetry
from tnfr.physics.vectorized_ops import compute_dnfr_flux_vectorized
from tnfr.utils.cache import reset_global_cache


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_fields_match_unique_neighbor_means(graph_type):
    """Successors, loops and parallel edges use the documented G.neighbors set."""
    graph = graph_type()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 0), (0, 1), (0, 1), (1, 2)])
    phases = {0: 0.1, 1: 2 * math.pi - 0.2, 2: 0.8, 3: 1.5}
    pressure = {0: 0.2, 1: 0.5, 2: 1.0, 3: 0.3}
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, phases[node])
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure[node])

    expected = {key: {} for key in ("grad_phi", "curv_phi", "j_phi", "j_dnfr")}
    for node in graph:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            for field in expected.values():
                field[node] = 0.0
            continue
        differences = np.array([phases[other] - phases[node] for other in neighbors])
        wrapped = (differences + math.pi) % (2 * math.pi) - math.pi
        mean_phase = np.angle(
            np.mean(np.exp(1j * np.array([phases[n] for n in neighbors])))
        )
        expected["grad_phi"][node] = float(np.mean(np.abs(wrapped)))
        expected["curv_phi"][node] = (phases[node] - mean_phase + math.pi) % (
            2 * math.pi
        ) - math.pi
        expected["j_phi"][node] = float(np.mean(np.sin(wrapped)))
        expected["j_dnfr"][node] = float(
            np.mean([pressure[n] - pressure[node] for n in neighbors])
        )

    standalone = {
        "grad_phi": compute_phase_gradient(graph),
        "curv_phi": compute_phase_curvature(graph),
        "j_phi": compute_phase_current(graph),
        "j_dnfr": compute_dnfr_flux(graph),
    }
    telemetry = compute_structural_telemetry(graph)
    for key in expected:
        assert standalone[key] == pytest.approx(expected[key], abs=1e-12), key
        assert telemetry[key] == pytest.approx(expected[key], abs=1e-12), key


def test_uniform_pressure_has_zero_directed_flux():
    """An equilibrium pressure field cannot acquire a sink from normalization."""
    graph = nx.DiGraph([(0, 1), (1, 2)])
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.5)
    assert compute_dnfr_flux(graph) == pytest.approx({node: 0.0 for node in graph})


def test_disconnected_potential_has_no_cross_component_source():
    """Unreachable sources contribute exactly zero, without a finite sentinel."""
    graph = nx.empty_graph(2)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.5)
    assert compute_structural_potential(graph) == {0: 0.0, 1: 0.0}


def test_coherence_length_provenance_matches_scalar_readout():
    graph = nx.path_graph(8)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.1 * (node + 1))
    estimate = estimate_coherence_length_with_provenance(graph)
    assert estimate.method in {"autocorrelation_fit", "spectral_gap", "unavailable"}
    assert estimate.distance_weighting != "unspecified"
    assert estimate.sample_selection != "unspecified"
    assert estimate.fit_quality != "unspecified"
    assert estimate.positive_mode_selection != "unspecified"
    assert estimate.graph_regime == "undirected; connected"
    if estimate.method != "unavailable":
        assert estimate.value == pytest.approx(estimate_coherence_length(graph))


def test_coherence_length_gap_fallback_does_not_cache_mode_shapes():
    graph = nx.path_graph(3)
    estimate = estimate_coherence_length_with_provenance(graph)
    assert estimate.method == "spectral_gap"
    assert "vecs" not in graph.graph["_tnfr_spectrum_cache"]


def test_coherence_length_directed_flat_field_refuses_symmetric_fallback():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    estimate = estimate_coherence_length_with_provenance(graph)
    assert estimate.method == "unavailable"
    assert estimate.graph_regime == "directed; connected"


def test_landmark_potential_is_zero_at_equilibrium():
    """The landmark correction excludes self-interaction even for 0/0."""
    graph = nx.path_graph(12)
    assert compute_structural_potential(graph, landmark_ratio=0.5) == {
        node: 0.0 for node in graph
    }


@pytest.mark.parametrize("phase", [1e-16, -1e-16, math.pi, -math.pi, 1e308])
@pytest.mark.parametrize("vectorized", [True, False])
def test_phase_current_preserves_represented_sine_and_antisymmetry(
    phase, vectorized, monkeypatch
):
    graph = nx.Graph([(0, 1)])
    set_attr(graph.nodes[0], ALIAS_THETA, 0.0)
    set_attr(graph.nodes[1], ALIAS_THETA, phase)
    if not vectorized:

        def unavailable(*args, **kwargs):
            raise RuntimeError("exercise loop fallback")

        monkeypatch.setattr(extended, "compute_phase_current_vectorized", unavailable)
    reset_global_cache()
    current = compute_phase_current(graph)
    assert current[0] == pytest.approx(math.sin(phase), rel=2e-15, abs=0.0)
    assert current[1] == -current[0]


@pytest.mark.parametrize("vectorized", [True, False])
def test_phase_current_rejects_overflow_instead_of_caching_nan(vectorized, monkeypatch):
    graph = nx.Graph([(0, 1)])
    for node, phase in enumerate((1e308, -1e308)):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    if not vectorized:

        def unavailable(*args, **kwargs):
            raise RuntimeError("exercise loop fallback")

        monkeypatch.setattr(extended, "compute_phase_current_vectorized", unavailable)
    reset_global_cache()
    with pytest.raises(ValueError, match="differences must be finite"):
        compute_phase_current(graph)


@pytest.mark.parametrize(
    "pressures",
    [
        (1e308, 1e308, 1e308),
        (0.0, 1e308, 1.0, -1e308),
        (-1e308, 1e308, -1e308),
    ],
    ids=("uniform-large-offset", "signed-cancellation", "overflowing-pair-difference"),
)
@pytest.mark.parametrize("vectorized", [True, False], ids=("vector", "fallback"))
def test_pressure_flux_retains_representable_neighbor_contrast(
    pressures, vectorized, monkeypatch
):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(len(pressures)))
    graph.add_edges_from(
        (0, neighbor, {"weight": 0.0, "length": 1e150})
        for neighbor in range(1, len(pressures))
    )
    # Parallel conductance and an incoming-only arc cannot reweight successors.
    graph.add_edge(0, 1, weight=7.0, length=1e150)
    graph.add_node("incoming", **{ALIAS_DNFR[0]: pressures[0]})
    graph.add_edge("incoming", 0, weight=5.0, length=1e150)
    for node, pressure in enumerate(pressures):
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure)
    if not vectorized:

        def unavailable(*args, **kwargs):
            raise RuntimeError("exercise loop fallback")

        monkeypatch.setattr(extended, "compute_dnfr_flux_vectorized", unavailable)
    reset_global_cache()
    center = Fraction.from_float(pressures[0])
    expected = float(
        sum(
            (Fraction.from_float(value) - center for value in pressures[1:]), Fraction()
        )
        / (len(pressures) - 1)
    )
    flux = compute_dnfr_flux(graph)
    assert flux[0] == expected
    assert all(flux[node] == 0.0 for node in graph if node != 0)
    # The telemetry path imports the same array owner independently of the
    # standalone fallback switch, so this checks both actual consumers.
    assert compute_structural_telemetry(graph)["j_dnfr"] == flux


@pytest.mark.parametrize("vectorized", [True, False], ids=("vector", "fallback"))
def test_unrepresentable_final_pressure_flux_is_not_cached(vectorized, monkeypatch):
    graph = nx.DiGraph([(0, 1)])
    for node, pressure in enumerate((-1e308, 1e308)):
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure)
    if not vectorized:

        def unavailable(*args, **kwargs):
            raise RuntimeError("exercise loop fallback")

        monkeypatch.setattr(extended, "compute_dnfr_flux_vectorized", unavailable)
    reset_global_cache()
    with pytest.raises(ValueError, match="finite floating-point range"):
        compute_dnfr_flux(graph)
    set_attr(graph.nodes[1], ALIAS_DNFR, -1e308)
    assert compute_dnfr_flux(graph) == {0: 0.0, 1: 0.0}


def test_pressure_flux_array_cannot_silently_ignore_wrong_neighbor_counts():
    with pytest.raises(ValueError, match="neighbor counts"):
        compute_dnfr_flux_vectorized(
            np.array([0.0, 2.0]), np.array([1]), np.array([0]), np.array([2, 0])
        )


def test_pressure_flux_rejects_output_dtype_overflow():
    arguments = (np.array([0.0, 1e100]), np.array([1]), np.array([0]), np.array([1, 0]))
    assert compute_dnfr_flux_vectorized(*arguments).tolist() == [1e100, 0.0]
    with pytest.raises(ValueError, match="output dtype range"):
        compute_dnfr_flux_vectorized(*arguments, dtype=np.float32)


@pytest.mark.parametrize("bad", [False, "0.0", math.nan, Fraction(1, 2**2000)])
def test_phase_symmetry_rejects_invalid_authoritative_data(bad):
    graph = nx.empty_graph(1)
    set_attr(graph.nodes[0], ALIAS_THETA, 0.0)
    graph.nodes[0][ALIAS_THETA[0]] = bad
    with pytest.raises((TypeError, ValueError)):
        measure_phase_symmetry(graph)


def test_phase_symmetry_distinguishes_empty_sample_and_undefined_mean():
    graph = nx.empty_graph(2)
    assert measure_phase_symmetry(graph) == 0.0  # Explicit legacy convention.
    for node, phase in enumerate((0.0, math.pi)):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    with pytest.raises(ValueError, match="vanishing resultant"):
        measure_phase_symmetry(graph)


def test_phase_symmetry_is_axial_and_uses_primitive_phasors():
    graph = nx.empty_graph(3)
    for node, phase in enumerate((0.0, 0.0, math.pi)):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    assert measure_phase_symmetry(graph) == pytest.approx(1.0)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, 1e308)
    assert measure_phase_symmetry(graph) == pytest.approx(1.0)
