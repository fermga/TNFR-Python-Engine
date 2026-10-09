"""Boundary controls for the preserved auxiliary pulse experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.graph_fixtures import sierpinski_simplex


def _load(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FRACTAL = _load("emergent_fractal_pulse")


@pytest.mark.parametrize("m, levels", [(2, 2), (3, 0), (3, 1), (3, 2), (4, 2)])
def test_supplied_gasket_preserves_its_outer_boundary(m, levels):
    graph, corners = sierpinski_simplex(m, levels)
    # Each of m copies shares one corner with every other copy. Boundary
    # vertices retain one copy's degree; each glued vertex joins two copies.
    assert len(graph) == (m ** (levels + 1) + m) // 2
    assert graph.number_of_edges() == m**levels * m * (m - 1) // 2
    assert len(set(corners)) == m
    assert nx.is_connected(graph)
    assert nx.number_of_selfloops(graph) == 0
    for node in graph:
        assert graph.degree(node) == (m - 1 if node in corners else 2 * (m - 1))
    for i, left in enumerate(corners):
        for right in corners[i + 1 :]:
            assert nx.shortest_path_length(graph, left, right) == 2**levels


@pytest.mark.parametrize(
    "name",
    [
        "emergent_nfr_geometry",
        "emergent_fractal_simplex_dimension",
        "emergent_resonant_pattern_tower",
        "emergent_rhythm",
    ],
)
def test_auxiliary_geometry_consumers_share_the_same_preparation(name):
    pytest.importorskip("scipy")
    assert _load(name).sierpinski_simplex is sierpinski_simplex


def test_supplied_gasket_is_fresh_and_keeps_ordered_recursive_corner_labels():
    graph, corners = sierpinski_simplex(3, 1)
    assert list(graph) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    assert corners == [(0, 0), (1, 1), (2, 2)]
    graph.remove_node(corners[0])
    corners.clear()
    fresh, boundary = sierpinski_simplex(3, 1)
    assert len(fresh) == 6
    assert boundary == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize("m, levels", [(1, 0), (3, -1), (3.0, 1), (3, True)])
def test_supplied_gasket_rejects_undefined_construction_parameters(m, levels):
    with pytest.raises((TypeError, ValueError)):
        sierpinski_simplex(m, levels)


@pytest.mark.parametrize("reader", ["fractal", "rhythm"])
def test_auxiliary_wave_geometry_keeps_isolates_stationary(reader):
    graph = nx.Graph()
    graph.add_nodes_from(["isolate", "left", "right"])
    graph.add_edge("left", "right", weight=0.25)
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    with np.errstate(all="raise"):
        if reader == "fractal":
            laplacian = FRACTAL.lsym(graph, list(graph))
        else:
            pytest.importorskip("scipy")
            eigenvalues, vectors, laplacian = _load("emergent_rhythm").lsym_eigh(graph)
            np.testing.assert_allclose(eigenvalues, [0.0, 0.0, 2.0], atol=1e-15)
            np.testing.assert_allclose(laplacian @ vectors, vectors * eigenvalues)
    np.testing.assert_array_equal(laplacian, expected)
    # A disconnected node has neither restoring acceleration nor diffusion.
    np.testing.assert_array_equal(laplacian @ [1.0, 0.0, 0.0], np.zeros(3))


def test_normalized_auxiliary_phase_step_is_invariant_to_conductance_scale():
    theta = np.array([0.0, np.pi / 6.0, 0.7])
    adjacency = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected = theta + np.array([0.025, -0.025, 0.0])
    with np.errstate(all="raise"):
        ordinary = FRACTAL.lrw_phase_step(
            theta, adjacency, adjacency.sum(axis=1), 0.5, 0.1
        )
        small = adjacency / 8.0
        rescaled = FRACTAL.lrw_phase_step(theta, small, small.sum(axis=1), 0.5, 0.1)
    np.testing.assert_allclose(ordinary, expected, atol=1e-16, rtol=0)
    np.testing.assert_array_equal(rescaled, ordinary)
    assert rescaled[2] == theta[2]


def test_nested_order_hierarchy_already_holds_without_phase_evolution():
    # Equal-size averaging plus the triangle inequality supplies this ordering.
    # The strict hierarchy at the initial snapshot is not a dynamic cascade.
    theta = np.array(
        [0.0, 0.0, np.pi / 2, np.pi / 2, np.pi, np.pi, -np.pi / 2, -np.pi / 2]
    )
    leaves = [[0, 1], [2, 3], [4, 5], [6, 7]]
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    leaf_order = FRACTAL.order_param(theta, leaves)
    group_order = FRACTAL.order_param(theta, groups)
    whole_order = FRACTAL.order_param(theta, [list(range(8))])
    assert leaf_order == pytest.approx(1.0)
    assert group_order == pytest.approx(np.sqrt(0.5))
    assert whole_order == pytest.approx(0.0, abs=1e-15)
    assert leaf_order > group_order > whole_order


def test_winding_comparison_preserves_tiny_phase_response():
    catalog = _load("emergent_particle_catalog")
    phases = np.array([0.0, 1e-16, 0.0])
    assert catalog.ring_phase_gradient_mean(phases) == pytest.approx(2e-16 / 3, abs=0.0)
    updated, report = catalog.relax_phase_ring(phases, dt=0.1, steps=1)
    np.testing.assert_allclose(updated, [5e-18, 9e-17, 5e-18], atol=0.0, rtol=1e-15)
    assert report["path_windings"] == (0, 0)
    assert report["minimum_branch_margin"] > 0.0


def test_winding_comparison_keeps_half_open_antipodal_boundary():
    catalog = _load("emergent_particle_catalog")
    np.testing.assert_array_equal(
        catalog._wrap_pi_array(np.array([np.pi, -np.pi])), [-np.pi, -np.pi]
    )
    with pytest.raises(ValueError, match="wrap branch boundary"):
        catalog.relax_phase_ring(np.array([0.0, np.pi, 0.0]), steps=0)


def test_fixed_diffusion_comparison_uses_shared_step_with_declared_time(monkeypatch):
    comparison = _load("emergent_structural_cosmology")
    actual_euler = comparison.euler_update
    observed_steps = []

    def record_euler(epi, dt, rate):
        observed_steps.append(dt)
        return actual_euler(epi, dt, rate)

    monkeypatch.setattr(comparison, "euler_update", record_euler)
    graph = nx.path_graph(2)
    nodes, lrw = comparison.structural_diffusion_operator(graph)
    initial = np.array([1.0, -1.0])
    updated = comparison.diffusion_step(initial, lrw, 1.0, 0.25)
    np.testing.assert_array_equal(updated, [0.5, -0.5])
    assert comparison.dirichlet_energy(graph, nodes, initial) == 2.0
    assert comparison.dirichlet_energy(graph, nodes, updated) == 0.5
    # Shared arithmetic does not make an excessive explicit step stable.
    overshot = comparison.diffusion_step(initial, lrw, 1.0, 1.25)
    np.testing.assert_array_equal(overshot, [-1.5, 1.5])
    assert comparison.dirichlet_energy(graph, nodes, overshot) == 4.5
    assert observed_steps == [0.25, 1.25]
