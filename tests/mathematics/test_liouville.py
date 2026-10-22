"""Admitted spectral sign checks and detached Liouvillian observations."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import liouville
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.generators import build_lindblad_delta_nfr


@pytest.fixture(autouse=True)
def numpy_solver(monkeypatch):
    monkeypatch.setattr(liouville, "get_backend", lambda: get_backend("numpy"))


def test_amplitude_damping_spectrum_and_slow_decay():
    generator = build_lindblad_delta_nfr(
        collapse_operators=[np.array([[0, 1], [0, 0]])]
    )
    spectrum = liouville.compute_liouvillian_spectrum(generator)
    np.testing.assert_allclose(spectrum, [-1, -0.5, -0.5, 0], atol=1e-14)
    assert liouville.get_slow_relaxation_mode(spectrum) == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "invalid", [True, "1", float("nan"), float("inf"), -1, Fraction(1, 10**400)]
)
def test_invalid_thresholds_cannot_bypass_spectral_checks(invalid):
    with pytest.raises((TypeError, ValueError), match="atol"):
        liouville.compute_liouvillian_spectrum(np.diag([0, 0, 0, 1]), atol=invalid)
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        liouville.get_slow_relaxation_mode([0, -1], tolerance=invalid)


def test_positive_spectral_growth_is_rejected_unless_check_is_disabled():
    matrix = np.diag([0, 0, 0, 1])
    with pytest.raises(ValueError, match="contractivity"):
        liouville.compute_liouvillian_spectrum(matrix)
    np.testing.assert_array_equal(
        liouville.compute_liouvillian_spectrum(matrix, validate_contractivity=False),
        [0, 0, 0, 1],
    )


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        "-1",
        complex(-1, float("nan")),
        complex(-1, float("inf")),
        -np.inf,
        Fraction(1, 10**400),
    ],
)
def test_invalid_primitive_spectra_reject_before_storage_or_selection(invalid):
    graph = nx.Graph()
    liouville.store_liouvillian_spectrum(graph, [0, -1])
    retained = graph.graph["LIOUVILLIAN_EIGS"].copy()
    with pytest.raises((TypeError, ValueError)):
        liouville.store_liouvillian_spectrum(graph, [0, invalid])
    assert graph.graph["LIOUVILLIAN_EIGS"] == retained
    with pytest.raises((TypeError, ValueError)):
        liouville.get_slow_relaxation_mode([0, invalid])
    graph.graph["LIOUVILLIAN_EIGS"] = [invalid]
    with pytest.raises((TypeError, ValueError)):
        liouville.get_liouvillian_spectrum(graph)


@pytest.mark.parametrize(
    "matrix", [np.eye(2, dtype=bool), [["0"]], [[np.nan]], [[Fraction(1, 10**400)]]]
)
def test_liouvillian_inputs_reject_before_solver_conversion(matrix):
    with pytest.raises((TypeError, ValueError)):
        liouville.compute_liouvillian_spectrum(matrix)


def test_spectrum_storage_and_reads_are_detached():
    graph = nx.Graph()
    original = np.array([0, -2 + 1j])
    liouville.store_liouvillian_spectrum(graph, original)
    original[:] = 42
    observed = liouville.get_liouvillian_spectrum(graph)
    np.testing.assert_array_equal(observed, [0, -2 + 1j])
    observed[:] = 99
    np.testing.assert_array_equal(
        liouville.get_liouvillian_spectrum(graph), [0, -2 + 1j]
    )
    assert liouville.get_slow_relaxation_mode([]) is None
    assert liouville.get_slow_relaxation_mode([0, 1, 2j]) is None


@pytest.mark.parametrize(
    "dtype", [np.int64, np.uint64, np.float32, np.float64, np.complex64, np.complex128]
)
def test_numeric_array_admission_preserves_values_shape_and_detached_c_order(dtype):
    source = np.arange(24, dtype=np.float64).reshape(4, 6).astype(dtype).T[:, ::2]
    expected = source.astype(np.complex128)
    admitted = liouville._finite_complex_array(source, "matrix")
    np.testing.assert_array_equal(admitted, expected)
    assert admitted.shape == source.shape
    assert admitted.dtype == np.complex128
    assert admitted.flags.c_contiguous
    assert not np.shares_memory(admitted, source)
    admitted[:] = 99
    np.testing.assert_array_equal(source, expected)


def test_numeric_array_admission_retains_canonical_positive_zero_channels():
    source = np.array([complex(-0.0, -0.0), complex(-1, -0.0), complex(-0.0, 2)])
    admitted = liouville._finite_complex_array(source, "spectrum")
    np.testing.assert_array_equal(admitted, source)
    assert not np.any(np.signbit(admitted.real[admitted.real == 0]))
    assert not np.any(np.signbit(admitted.imag[admitted.imag == 0]))
    assert np.signbit(source[0].real) and np.signbit(source[0].imag)


@pytest.mark.parametrize(
    "values, error, message",
    [
        (np.array([0.0, np.inf, np.nan]), ValueError, r"values\[1\] must be finite"),
        (
            np.array([0j, complex(-1, np.nan)]),
            ValueError,
            r"values\[1\].imag must be finite",
        ),
        (np.array([True]), TypeError, r"values\[0\].*not a boolean"),
        (np.array(["1"]), TypeError, r"values\[0\].*real or complex scalar"),
        (
            np.array([0, Fraction(1, 10**400)], dtype=object),
            ValueError,
            r"values\[1\].*underflows",
        ),
    ],
)
def test_array_admission_retains_scalar_error_class_and_first_bad_index(
    values, error, message
):
    with pytest.raises(error, match=message) as caught:
        liouville._finite_complex_array(values, "values")
    assert type(caught.value) is error


def test_array_admission_retains_empty_shapes():
    admitted = liouville._finite_complex_array(np.empty((2, 0, 3)), "values")
    assert admitted.shape == (2, 0, 3)
    assert admitted.dtype == np.complex128


@pytest.mark.parametrize("name", ["numpy", "torch", "jax"])
def test_spectrum_still_uses_selected_backend_and_preserves_unsorted_order(
    monkeypatch, name
):
    if name != "numpy":
        pytest.importorskip(name)
    backend = get_backend(name)
    monkeypatch.setattr(liouville, "get_backend", lambda: backend)
    original_eig = type(backend).eig
    observed = []

    def record_eig(self, matrix):
        result = original_eig(self, matrix)
        observed.append(np.array(self.to_numpy(result[0]), copy=True))
        return result

    monkeypatch.setattr(type(backend), "eig", record_eig)
    result = liouville.compute_liouvillian_spectrum(np.diag([-1, 0, -2]), sort=False)
    assert len(observed) == 1
    np.testing.assert_array_equal(result, observed[0])
    assert not np.shares_memory(result, observed[0])


@pytest.mark.parametrize(
    "values, tolerance, expected",
    [
        ([-0.5, 0.5, -1.0, 0], 0.5, -1.0),
        ([-0.25 + 2j, -0.25 - 7j, -1], 0, -0.25 + 2j),
        ([-0.25 - 7j, -0.25 + 2j, -1], 0, -0.25 - 7j),
        ([-0.0, 0.0, 1j, -1j, 2], 0, None),
        (
            [-np.nextafter(0.0, 1.0), -2 * np.nextafter(0.0, 1.0)],
            np.nextafter(0.0, 1.0),
            -2 * np.nextafter(0.0, 1.0),
        ),
    ],
)
def test_slow_mode_combined_filter_preserves_strict_cut_and_first_tie(
    values, tolerance, expected
):
    assert (
        liouville.get_slow_relaxation_mode(np.array(values), tolerance=tolerance)
        == expected
    )
