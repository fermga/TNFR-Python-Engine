"""Independent outgoing-walk checks for validation, scaling, and stationarity."""

import numpy as np
import pytest

from tnfr.physics.directed_diffusion import (
    directed_rw_laplacian,
    stationary_distribution,
)


@pytest.mark.parametrize("reader", [directed_rw_laplacian, stationary_distribution])
@pytest.mark.parametrize("adjacency", [
    1.0,
    [1.0, 2.0],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, -1.0], [1.0, 0.0]],
    [[0.0, float("nan")], [1.0, 0.0]],
    [[0.0, float("inf")], [1.0, 0.0]],
    [[0.0, 1.0j], [1.0, 0.0]],
])
def test_matrix_readers_reject_invalid_conductance(reader, adjacency):
    with pytest.raises(ValueError, match="square|finite nonnegative"):
        reader(adjacency)


@pytest.mark.parametrize("scale", [np.nextafter(0.0, 1.0), 1e-310, 1.0, 1e307])
def test_positive_row_scaling_preserves_walk_and_stationary_measure(scale):
    adjacency = scale * np.array([[0.0, 2.0, 1.0],
                                  [1.0, 0.0, 1.0],
                                  [1.0, 3.0, 0.0]])
    expected_transition = np.array([[0.0, 2 / 3, 1 / 3],
                                    [1 / 2, 0.0, 1 / 2],
                                    [1 / 4, 3 / 4, 0.0]])
    before = adjacency.copy()
    with np.errstate(all="raise"):
        laplacian = directed_rw_laplacian(adjacency)
        stationary = stationary_distribution(adjacency)
    np.testing.assert_allclose(np.eye(3) - laplacian, expected_transition, atol=1e-15)
    # Solve the three rational balance equations independently: weights 15:22:16.
    np.testing.assert_allclose(stationary, np.array([15, 22, 16]) / 53, atol=1e-15)
    np.testing.assert_allclose(stationary @ expected_transition, stationary, atol=1e-15)
    np.testing.assert_array_equal(adjacency, before)


def test_finite_conductance_with_overflowing_row_sum_remains_normalizable():
    adjacency = np.array([[1e308, 1e308], [1e308, 0.0]])
    with np.errstate(all="raise"):
        laplacian = directed_rw_laplacian(adjacency)
        stationary = stationary_distribution(adjacency)
    np.testing.assert_array_equal(np.eye(2) - laplacian, [[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(stationary, [2 / 3, 1 / 3], atol=1e-15)


def test_zero_strength_singleton_is_an_absorbing_stationary_walk():
    adjacency = np.zeros((1, 1))
    laplacian = directed_rw_laplacian(adjacency)
    stationary = stationary_distribution(adjacency)
    transition = np.eye(1) - laplacian
    np.testing.assert_array_equal(laplacian, [[0.0]])
    np.testing.assert_array_equal(transition, [[1.0]])
    np.testing.assert_array_equal(stationary @ transition, stationary)
    np.testing.assert_array_equal(stationary, [1.0])


def test_sink_state_is_fixed_but_drives_its_incoming_neighbor():
    laplacian = directed_rw_laplacian([[0.0, 1.0], [0.0, 0.0]])
    state = np.array([0.0, 1.0])
    capacity = np.array([2.0, 7.0])
    np.testing.assert_array_equal(-capacity * (laplacian @ state), [2.0, 0.0])
    np.testing.assert_array_equal(np.eye(2) - laplacian, [[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        stationary_distribution([[0.0, 1.0], [0.0, 0.0]])


def test_empty_generator_has_no_normalized_stationary_distribution():
    assert directed_rw_laplacian(np.zeros((0, 0))).shape == (0, 0)
    with pytest.raises(ValueError, match="at least one node"):
        stationary_distribution(np.zeros((0, 0)))


@pytest.mark.parametrize("tol", [-1.0, float("nan"), float("inf")])
def test_stationary_tolerance_cannot_disable_validation(tol):
    with pytest.raises(ValueError, match="tolerance"):
        stationary_distribution([[0.0]], tol=tol)


def test_undirected_random_walk_uses_degree_similarity_not_raw_symmetry():
    adjacency = np.array([[0.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0]])
    laplacian = directed_rw_laplacian(adjacency)
    root_degree = np.sqrt(adjacency.sum(axis=1))
    symmetric_representation = root_degree[:, None] * laplacian / root_degree[None, :]
    assert not np.allclose(laplacian, laplacian.T)
    np.testing.assert_allclose(symmetric_representation, symmetric_representation.T)
