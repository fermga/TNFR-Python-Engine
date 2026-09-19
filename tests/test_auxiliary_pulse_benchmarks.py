"""Boundary controls for the preserved auxiliary pulse experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FRACTAL = _load("emergent_fractal_pulse")


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
