"""Finite arithmetic spectral diagnostics, without a zero-location claim."""

import numpy as np
import pytest

from tnfr.riemann.hilbert_polya import (
    hp_resolvent_schatten_norms,
    verify_hp_self_adjoint,
    wasserstein_1_distance,
)


def test_self_adjointness_admits_complex_hermitian_matrices():
    matrix = np.array([[1, 1j], [-1j, 2]])
    assert np.array_equal(matrix, matrix.conj().T)
    report = verify_hp_self_adjoint(matrix)
    assert report["self_adjoint"]
    assert report["asymmetry_frobenius"] == 0.0
    assert report["imaginary_frobenius"] == pytest.approx(np.sqrt(2.0))
    assert not verify_hp_self_adjoint(np.array([[1, 1j], [1j, 2]]))["self_adjoint"]


def test_resolvent_norms_describe_the_same_operator():
    # gamma=(0, sqrt(3)), s=1 gives R=diag(1, 1/2), not R squared.
    report = hp_resolvent_schatten_norms(np.array([0.0, np.sqrt(3.0)]))
    assert report["schatten_1_norm"] == pytest.approx(1.5)
    assert report["schatten_2_norm"] == pytest.approx(np.sqrt(1.25))
    assert report["operator_norm_inverse"] == 1.0
    assert report["trace_class"]


def test_resolvent_norms_match_dense_singular_values():
    gammas = np.array([-3.0, 0.0, 2.0, 7.0])
    shift = 2.0
    # Build the positive square root of the inverse shifted-square operator.
    inverse_squared = np.linalg.inv(np.diag(gammas**2) + shift**2 * np.eye(4))
    eigenvalues, eigenvectors = np.linalg.eigh(inverse_squared)
    resolvent = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    singular_values = np.linalg.svd(resolvent, compute_uv=False)
    report = hp_resolvent_schatten_norms(gammas, shift=shift)
    assert report["schatten_1_norm"] == pytest.approx(singular_values.sum())
    assert report["schatten_2_norm"] == pytest.approx(np.linalg.norm(resolvent, "fro"))
    assert report["operator_norm_inverse"] == pytest.approx(singular_values[0])


def test_resolvent_norms_avoid_intermediate_square_overflow():
    scale = np.finfo(float).max
    report = hp_resolvent_schatten_norms([scale], shift=scale)
    expected = (1.0 / np.sqrt(2.0)) / scale
    for name in ("schatten_1_norm", "schatten_2_norm", "operator_norm_inverse"):
        assert report[name] > 0.0
        assert report[name] / expected == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [0.0, -1.0, np.nan, np.inf, True])
def test_resolvent_rejects_invalid_shift(shift):
    with pytest.raises(ValueError, match="shift"):
        hp_resolvent_schatten_norms([1.0], shift=shift)


@pytest.mark.parametrize("spectrum", [[], [np.nan], [np.inf], [[1.0]], [1.0j]])
def test_resolvent_rejects_invalid_spectrum(spectrum):
    with pytest.raises(ValueError):
        hp_resolvent_schatten_norms(spectrum)


def test_resolvent_rejects_unrepresentable_norm():
    with pytest.raises(ValueError, match="finite floating-point range"):
        hp_resolvent_schatten_norms([0.0], shift=np.nextafter(0.0, 1.0))


def test_empirical_distance_preserves_unequal_sample_masses():
    # CDF differences are 1/6 on both intervals (0, 5) and (5, 10).
    # Interpolating the two-point quantiles incorrectly made this distance 0.
    left = np.array([10.0, 0.0])
    right = np.array([5.0, 10.0, 0.0])
    assert wasserstein_1_distance(left, right) == pytest.approx(5.0 / 3.0)
    assert wasserstein_1_distance(right, left) == pytest.approx(5.0 / 3.0)
    # Repeating every atom preserves each empirical probability measure.
    assert wasserstein_1_distance(
        np.repeat(left, 3), np.repeat(right, 2)
    ) == pytest.approx(5.0 / 3.0)


def test_equal_sample_distance_is_mean_sorted_absolute_difference():
    assert wasserstein_1_distance([0.0], [2.0]) == 2.0
    assert wasserstein_1_distance([9.0, 1.0], [0.0, 4.0]) == 3.0
    assert wasserstein_1_distance([], []) == 0.0
    assert wasserstein_1_distance([0.0, 0.0], [0.0]) == 0.0


def test_empirical_distance_recovers_finite_extreme_result():
    # The naive reduction overflows even though its mean is representable.
    assert wasserstein_1_distance([1e308, 1e308], [0.0, 0.0]) == 1e308
    # CDF support differences can overflow before their probability weighting.
    result = wasserstein_1_distance([-1e308, 1e308], [0.0, 0.0, 0.0])
    assert result / 1e308 == pytest.approx(1.0)


def test_empirical_distance_rejects_unrepresentable_result():
    with pytest.raises(ValueError, match="finite floating-point range"):
        wasserstein_1_distance([-1e308], [1e308])
    tiny = np.nextafter(0.0, 1.0)
    for zeros in ([0.0], [0.0, 0.0]):
        with pytest.raises(ValueError, match="underflows"):
            wasserstein_1_distance([0.0, tiny], zeros)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1.0]], [1.0j], 1.0])
def test_empirical_distance_rejects_unavailable_measure(bad):
    with pytest.raises(ValueError):
        wasserstein_1_distance([0.0], bad)
    with pytest.raises(ValueError):
        wasserstein_1_distance(bad, [0.0])
