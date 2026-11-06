"""Per-invocation spectral reuse without changing vector arithmetic or hooks."""

from types import SimpleNamespace

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import spectral


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("kind", ["real", "complex", "partial", "right"])
def test_grouped_transform_matches_individual_arithmetic_bitwise(layout, kind):
    rng = np.random.default_rng(391)
    matrix = rng.normal(size=(8, 8))
    if kind == "complex":
        matrix = matrix + 1j * rng.normal(size=(8, 8))
    basis, _ = np.linalg.qr(matrix)
    if kind == "partial":
        basis = basis[:, :4]
    elif kind == "right":
        basis = basis @ np.diag(np.arange(1, 9))
    if layout == "strided":
        storage = np.empty((basis.shape[0] * 2, basis.shape[1] * 2), dtype=basis.dtype)
        storage[::2, ::2] = basis
        basis = storage[::2, ::2]
    else:
        basis = np.array(basis, order=layout)
    signals = [rng.normal(size=8) + 1j * rng.normal(size=8) for _ in range(4)]
    expected = tuple(spectral.gft(signal, basis) for signal in signals)
    actual = spectral._gft_many(signals, basis)
    for reference, result in zip(expected, actual):
        np.testing.assert_array_equal(result, reference)


def test_general_right_basis_uses_solve_and_validates_once(monkeypatch):
    basis = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])
    coefficients = [np.array([1.0, -2.0, 3.0]), np.array([2.0, 0.0, -1.0])]
    signals = [basis @ vector for vector in coefficients]
    validator = spectral._validate_transform_basis
    validations = []

    def counted(matrix):
        validations.append(matrix)
        assert not matrix.flags.writeable
        assert not np.shares_memory(matrix, basis)
        return validator(matrix)

    monkeypatch.setattr(spectral, "_validate_transform_basis", counted)
    results = spectral._gft_many(signals, basis)
    assert len(validations) == 1
    for actual, expected in zip(results, coefficients):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert not np.allclose(basis.T @ signals[0], coefficients[0])


@pytest.mark.parametrize(
    "signals,basis",
    [
        ([np.ones(3)], np.array([[1.0], [1.0], [0.0]])),
        ([np.ones(2)], np.ones((2, 2))),
        ([np.ones(2)], np.eye(3)),
        ([np.ones((2, 1))], np.eye(2)),
        ([np.ones(2)], np.ones(2)),
        ([np.ones(2), np.ones(3)], np.eye(2)),
    ],
)
def test_grouped_validation_matches_individual_failure(signals, basis):
    with pytest.raises(TNFRValueError) as individual:
        tuple(spectral.gft(signal, basis) for signal in signals)
    with pytest.raises(TNFRValueError) as grouped:
        spectral._gft_many(signals, basis)
    assert str(grouped.value) == str(individual.value)


def test_each_call_reads_current_basis_and_outputs_are_independent():
    basis = np.eye(2)
    signals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    first = spectral._gft_many(signals, basis)
    basis *= 2
    second = spectral._gft_many(signals, basis)
    for original, current, signal in zip(first, second, signals):
        np.testing.assert_array_equal(original, signal)
        np.testing.assert_array_equal(current, signal / 2)
        assert not np.shares_memory(current, signal)
    first[0][:] = -99
    np.testing.assert_array_equal(second[0], signals[0] / 2)


@pytest.mark.parametrize("container", ["tuple", "generator"])
def test_custom_signal_conversion_keeps_live_basis_and_call_order(container):
    basis = np.eye(2)
    first = np.array([2.0, 4.0])

    class ChangingSignal:
        def __array__(self, dtype=None, copy=None):
            basis[:] *= 2
            return np.array([6.0, 8.0], dtype=dtype)

    signals = (first, ChangingSignal())
    if container == "generator":
        signals = (signal for signal in signals)
    actual = spectral._gft_many(signals, basis)
    np.testing.assert_array_equal(actual[0], [2.0, 4.0])
    np.testing.assert_array_equal(actual[1], [3.0, 4.0])


def test_array_subclasses_keep_individual_transform_dispatch(monkeypatch):
    class Signal(np.ndarray):
        pass

    signal = np.array([1.0, 2.0]).view(Signal)
    validator = spectral._validate_transform_basis
    calls = []

    def counted(basis):
        calls.append(basis)
        return validator(basis)

    monkeypatch.setattr(spectral, "_validate_transform_basis", counted)
    actual = spectral._gft_many([signal, signal], np.eye(2))
    assert len(calls) == 2
    np.testing.assert_array_equal(actual[0], signal)


def test_replaced_public_transform_is_called_for_each_signal(monkeypatch):
    signals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    basis = np.eye(2)
    calls = []

    def replacement(signal, supplied_basis):
        assert supplied_basis is basis
        calls.append(signal)
        return signal + len(calls)

    monkeypatch.setattr(spectral, "gft", replacement)
    result = spectral._gft_many(signals, basis)
    assert calls[0] is signals[0] and calls[1] is signals[1]
    np.testing.assert_array_equal(result[0], [2, 3])
    np.testing.assert_array_equal(result[1], [5, 6])


@pytest.mark.parametrize("failing", [False, True])
def test_accelerator_routing_and_cpu_fallback_remain_per_vector(monkeypatch, failing):
    basis = np.eye(101)
    signals = [np.arange(101.0), -np.arange(101.0)]
    calls = []

    def as_array(value):
        calls.append(("array", value.shape))
        assert value.flags.writeable
        return value

    def matmul(left, right):
        calls.append(("matmul", right[1]))
        if failing:
            raise RuntimeError("accelerator unavailable")
        return left @ right

    backend = SimpleNamespace(
        supports_autodiff=True,
        as_array=as_array,
        conjugate_transpose=lambda value: value.conj().T,
        matmul=matmul,
        to_numpy=np.asarray,
    )

    def get_backend():
        calls.append(("backend",))
        return backend

    monkeypatch.setattr(spectral, "get_backend", get_backend)
    actual = spectral._gft_many(signals, basis)
    assert calls == [
        ("backend",),
        ("array", (101, 101)),
        ("array", (101,)),
        ("matmul", 1.0),
        ("backend",),
        ("array", (101, 101)),
        ("array", (101,)),
        ("matmul", -1.0),
    ]
    for result, signal in zip(actual, signals):
        np.testing.assert_array_equal(result, signal)


def test_later_signal_failure_follows_first_backend_dispatch(monkeypatch):
    seen = []

    def get_backend():
        seen.append("backend")
        return SimpleNamespace(supports_autodiff=False)

    monkeypatch.setattr(spectral, "get_backend", get_backend)
    with pytest.raises(TNFRValueError, match="dimensions are incompatible"):
        spectral._gft_many([np.ones(101), np.ones(100)], np.eye(101))
    assert seen == ["backend"]


def test_backend_cannot_mutate_retained_basis_before_cpu_fallback(monkeypatch):
    basis = np.eye(101)

    def as_array(value):
        value[:] = 0
        raise RuntimeError("failed after receiving its own buffer")

    monkeypatch.setattr(
        spectral,
        "get_backend",
        lambda: SimpleNamespace(supports_autodiff=True, as_array=as_array),
    )
    signals = [np.ones(101), np.full(101, 2.0)]
    for result, signal in zip(spectral._gft_many(signals, basis), signals):
        np.testing.assert_array_equal(result, signal)
    np.testing.assert_array_equal(basis, np.eye(101))


def test_public_transform_replacement_during_dispatch_affects_later_vectors(
    monkeypatch,
):
    replacement_calls = []

    def replacement(signal, basis):
        replacement_calls.append(signal)
        return signal + 1

    def get_backend():
        monkeypatch.setattr(spectral, "gft", replacement)
        return SimpleNamespace(supports_autodiff=False)

    monkeypatch.setattr(spectral, "get_backend", get_backend)
    signals = [np.ones(101), np.full(101, 2.0)]
    result = spectral._gft_many(signals, np.eye(101))
    np.testing.assert_array_equal(result[0], signals[0])
    np.testing.assert_array_equal(result[1], signals[1] + 1)
    assert replacement_calls[0] is signals[1]


def test_filter_still_readmits_basis_after_user_callback():
    basis = np.eye(2)

    def filter_response(eigenvalues):
        basis[:, 1] = basis[:, 0]
        return np.ones_like(eigenvalues)

    with pytest.raises(TNFRValueError, match="basis is singular"):
        spectral.spectral_filter(
            np.ones(2), basis, np.array([0.0, 1.0]), filter_response
        )


def test_empty_signal_collection_performs_no_transform():
    assert spectral._gft_many([], np.array(0.0)) == ()
