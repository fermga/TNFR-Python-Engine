"""Solver selection follows admitted matrix symmetry, not graph labels."""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg

from tnfr.errors import TNFRValueError
from tnfr.mathematics import spectral
from tnfr.physics.structural_diffusion import (
    structural_diffusion_operator,
    symmetric_normalized_laplacian,
)
from tnfr.utils.cache import invalidate_function_cache


@pytest.mark.parametrize("operator", ["symmetric", "random_walk", "combinatorial"])
def test_reciprocal_triangle_preserves_parseval_in_repeated_eigenspace(operator):
    graph = nx.complete_graph(3).to_directed()
    eigenvalues, basis = spectral.get_laplacian_spectrum(graph, operator=operator)
    scale = 1 if operator == "combinatorial" else 0.5
    laplacian = scale * (3 * np.eye(3) - np.ones((3, 3)))
    np.testing.assert_allclose(eigenvalues, [0, 3 * scale, 3 * scale], atol=2e-14)
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(3), atol=2e-14)
    np.testing.assert_allclose(laplacian @ basis, basis * eigenvalues, atol=2e-14)
    signal = np.array([-1.0, 0.0, 1.0])
    coefficients = spectral.gft(signal, basis)
    assert np.vdot(coefficients, coefficients).real == pytest.approx(2.0, abs=2e-14)
    np.testing.assert_allclose(spectral.igft(coefficients, basis), signal, atol=2e-14)


def test_weighted_reciprocal_normalization_is_exactly_symmetric():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0.1), (1, 2, 3.7), (1, 3, 0.0001), (2, 3, 2.3)]
    )
    _, laplacian = symmetric_normalized_laplacian(graph)
    _, reciprocal_laplacian = symmetric_normalized_laplacian(graph.to_directed())
    np.testing.assert_array_equal(laplacian, laplacian.T)
    np.testing.assert_array_equal(reciprocal_laplacian, laplacian)
    undirected_values, _ = spectral.get_laplacian_spectrum(graph)
    values, basis = spectral.get_laplacian_spectrum(graph.to_directed())
    np.testing.assert_array_equal(values, undirected_values)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=2e-14)
    np.testing.assert_allclose(laplacian @ basis, basis * values, atol=2e-14)


@pytest.mark.parametrize("operator", ["symmetric", "random_walk", "combinatorial"])
def test_symmetric_partial_operator_uses_valid_orthonormal_modes(operator, monkeypatch):
    graph = nx.complete_graph(4).to_directed()
    invalidate_function_cache(spectral._get_laplacian_spectrum_cached)

    def forbidden_general_solver(*args, **kwargs):
        raise AssertionError("A symmetric admitted matrix must use a Hermitian solver")

    monkeypatch.setattr(scipy.linalg, "eig", forbidden_general_solver)
    eigenvalues, basis = spectral.get_laplacian_spectrum(graph, operator=operator, k=3)
    scale = 1 if operator == "combinatorial" else 1 / 3
    laplacian = scale * (4 * np.eye(4) - np.ones((4, 4)))
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-14)
    np.testing.assert_allclose(laplacian @ basis, basis * eigenvalues, atol=2e-14)
    signal = np.array([1.0, -1.0, 2.0, 3.0])
    projected = spectral.igft(spectral.gft(signal, basis), basis)
    np.testing.assert_allclose(projected, basis @ basis.T @ signal, atol=2e-14)


def test_equal_operators_share_cache_across_graph_direction_labels(monkeypatch):
    invalidate_function_cache(spectral._get_laplacian_spectrum_cached)
    original = scipy.linalg.eigh
    matrices = []

    def counted(matrix, *args, **kwargs):
        matrices.append(matrix.copy())
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "eigh", counted)
    graph = nx.complete_graph(3)
    first_values, first_basis = spectral.get_laplacian_spectrum(graph.to_directed())
    second_values, second_basis = spectral.get_laplacian_spectrum(graph)
    assert len(matrices) == 1
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(first_basis, second_basis)
    first_basis[:] = 0
    assert np.linalg.norm(second_basis) == pytest.approx(np.sqrt(3))


def test_directed_asymmetric_walk_keeps_general_right_basis(monkeypatch):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [(0, 1, 1.0), (0, 2, 2.0), (1, 0, 3.0), (1, 2, 0.5), (2, 0, 0.2), (2, 1, 4.0)]
    )
    _, laplacian = structural_diffusion_operator(graph)
    assert not np.array_equal(laplacian, laplacian.T)
    invalidate_function_cache(spectral._get_laplacian_spectrum_cached)

    def forbidden_hermitian_solver(*args, **kwargs):
        raise AssertionError(
            "An asymmetric admitted matrix cannot use a Hermitian solver"
        )

    monkeypatch.setattr(scipy.linalg, "eigh", forbidden_hermitian_solver)
    monkeypatch.setattr(scipy.sparse.linalg, "eigsh", forbidden_hermitian_solver)
    values, basis = spectral.get_laplacian_spectrum(graph, operator="random_walk")
    np.testing.assert_allclose(laplacian @ basis, basis * values, atol=2e-14)
    assert not np.allclose(basis.conj().T @ basis, np.eye(3))
    signal = np.array([1.0, -2.0, 4.0])
    np.testing.assert_allclose(
        spectral.igft(spectral.gft(signal, basis), basis), signal, atol=2e-14
    )
    np.testing.assert_allclose(
        spectral.heat_diffusion(signal, basis, values, 0.2),
        scipy.linalg.expm(-0.2 * laplacian) @ signal,
        atol=2e-14,
    )
    _, partial_basis = spectral.get_laplacian_spectrum(
        graph, operator="random_walk", k=2
    )
    with pytest.raises(TNFRValueError, match="partial graph spectral basis"):
        spectral.gft(signal, partial_basis)


def test_tiny_represented_asymmetry_is_not_erased_by_solver_tolerance(monkeypatch):
    graph = nx.complete_graph(3).to_directed()
    graph.edges[0, 1]["weight"] = np.nextafter(1.0, 2.0)
    _, laplacian = structural_diffusion_operator(graph)
    assert np.allclose(laplacian, laplacian.T)
    assert not np.array_equal(laplacian, laplacian.T)
    original = spectral._get_laplacian_spectrum_cached
    solver_modes = []

    def observe(matrix_bytes, size, k, use_general_eig):
        solver_modes.append(use_general_eig)
        return original(matrix_bytes, size, k, use_general_eig)

    monkeypatch.setattr(spectral, "_get_laplacian_spectrum_cached", observe)
    values, basis = spectral.get_laplacian_spectrum(graph, operator="random_walk")
    assert solver_modes == [True]
    np.testing.assert_allclose(laplacian @ basis, basis * values, atol=2e-14)
