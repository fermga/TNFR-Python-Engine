"""Numerical and native-backend boundaries of spectral runtime observations."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import runtime
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.operators import FrequencyOperator, SpectralExpectationOperator
from tnfr.mathematics.spaces import HilbertSpace


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request, monkeypatch):
    resolved = get_backend(request.param)
    if resolved.name != request.param:
        pytest.skip(f"{request.param} is unavailable")
    monkeypatch.setattr(runtime, "get_backend", lambda: resolved)
    return resolved


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200, 1e307])
def test_norm_observation_preserves_representable_scale(backend, scale):
    state = backend.as_array(np.array([3 * scale, 4j * scale]), dtype=np.complex128)
    passed, norm = runtime.normalized(state, HilbertSpace(2))
    assert passed is False
    assert norm == pytest.approx(5 * scale, rel=2e-15, abs=0.0)


def test_norm_observation_distinguishes_zero_and_smallest_subnormal(backend):
    assert runtime.normalized([0, 0], HilbertSpace(2)) == (False, 0.0)
    tiny = np.nextafter(0.0, 1.0)
    # JAX's CPU arithmetic flushes subnormals, but admission preserves the
    # concrete array and this detached observation does no backend arithmetic.
    assert runtime.normalized([tiny, 0], HilbertSpace(2)) == (False, tiny)


@pytest.mark.parametrize("scale", [1e-200, 1e200, 1e308])
def test_unitary_uses_shared_scaled_normalization(backend, scale):
    operator = SpectralExpectationOperator([[0, 1], [1, 0]], backend=backend)
    state = backend.as_array(np.array([scale, 1j * scale]), dtype=np.complex128)
    passed, norm = runtime.stable_unitary(state, operator, HilbertSpace(2), atol=0.0)
    assert passed is True
    assert norm == pytest.approx(1.0, rel=2e-14)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_unitary_without_normalization_retains_supplied_amplitude(backend, scale):
    operator = SpectralExpectationOperator([0, 0], backend=backend)
    passed, norm = runtime.stable_unitary(
        [3 * scale, 4j * scale], operator, HilbertSpace(2), normalise=False
    )
    assert passed is False
    assert norm == pytest.approx(5 * scale, rel=2e-15, abs=0.0)


@pytest.mark.parametrize(
    "invalid",
    [True, "1", float("inf"), float("nan"), Fraction(1, 10**400), 10**400],
)
def test_runtime_admits_source_scalars_before_backend_conversion(backend, invalid):
    operator = SpectralExpectationOperator([0, 0], backend=backend)
    state = [invalid, 0]
    with pytest.raises(TNFRValueError):
        runtime.normalized(state, HilbertSpace(2))
    with pytest.raises(TNFRValueError):
        runtime.stable_unitary(state, operator, HilbertSpace(2))


@pytest.mark.parametrize("dtype", [bool, str, object])
def test_runtime_array_dtypes_cannot_hide_invalid_original_scalars(backend, dtype):
    state = np.array([True, False], dtype=dtype)
    with pytest.raises(TNFRValueError):
        runtime.normalized(state, HilbertSpace(2))


def test_unrepresentable_norm_observation_rejects_but_unit_direction_survives(backend):
    operator = SpectralExpectationOperator([0, 0], backend=backend)
    state = [complex(1.7e308, 1.7e308), 0]
    with pytest.raises(TNFRValueError, match="norm.*finite float"):
        runtime.normalized(state, HilbertSpace(2))
    with pytest.raises(TNFRValueError, match="norm.*finite float"):
        runtime.stable_unitary(state, operator, HilbertSpace(2), normalise=False)
    assert runtime.stable_unitary(state, operator, HilbertSpace(2))[0] is True


def test_unitary_null_cutoff_and_dimension_errors_remain_explicit(backend):
    operator = SpectralExpectationOperator([0, 0], backend=backend)
    space = HilbertSpace(2)
    for state in ([0, 0], [1e-9, 0]):
        with pytest.raises(TNFRValueError, match="null"):
            runtime.stable_unitary(state, operator, space)
    above = np.nextafter(1e-9, np.inf)
    assert runtime.stable_unitary([above, 0], operator, space)[0] is True
    with pytest.raises(TNFRValueError, match="dimension mismatch"):
        runtime.normalized([1, 0, 0], space)
    with pytest.raises(TNFRValueError, match="dimension mismatch"):
        runtime.stable_unitary([1, 0, 0], operator, space)


@pytest.mark.parametrize(
    "invalid", [True, "1", -1.0, np.inf, np.nan, Fraction(1, 10**400)]
)
def test_invalid_runtime_tolerances_cannot_produce_passing_observations(invalid):
    backend = get_backend("numpy")
    operator = SpectralExpectationOperator([1, 2], backend=backend)
    frequency = FrequencyOperator([1, 2], backend=backend)
    calls = [
        lambda: runtime.normalized([1, 0], HilbertSpace(2), atol=invalid),
        lambda: runtime.stable_unitary([1, 0], operator, HilbertSpace(2), atol=invalid),
        lambda: runtime.meets_spectral_expectation_threshold(
            [1, 0], operator, 1.0, atol=invalid
        ),
        lambda: runtime.frequency_positive([1, 0], frequency, atol=invalid),
    ]
    for call in calls:
        with pytest.raises(TNFRValueError, match="atol"):
            call()


@pytest.mark.parametrize(
    "invalid", [True, "1", np.inf, np.nan, Fraction(1, 10**400), 10**400]
)
def test_runtime_threshold_uses_shared_represented_real_admission(invalid):
    operator = SpectralExpectationOperator([1, 2], backend=get_backend("numpy"))
    with pytest.raises(TNFRValueError, match="threshold"):
        runtime.meets_spectral_expectation_threshold([1, 0], operator, invalid)


def test_compatibility_aliases_and_frequency_summary_preserve_observable_scope():
    backend = get_backend("numpy")
    operator = SpectralExpectationOperator([2, 6], backend=backend)
    state = [1, 1j]
    assert runtime.coherence_expectation(state, operator) == pytest.approx(4.0)
    assert runtime.coherence(state, operator, Fraction(7, 2)) == pytest.approx(
        (True, 4.0)
    )
    summary = runtime.frequency_positive(
        state, FrequencyOperator([2, 6], backend=backend)
    )
    assert summary == {
        "passed": True,
        "value": pytest.approx(4.0),
        "enforce": True,
        "spectrum_psd": True,
        "spectrum_min": 2.0,
        "projection_passed": True,
    }


@pytest.mark.parametrize("conjugate", [False, True])
def test_unitary_observation_preserves_native_torch_state_and_operator_gradients(
    monkeypatch, conjugate
):
    backend = get_backend("torch")
    if backend.name != "torch":
        pytest.skip("torch is unavailable")
    torch = pytest.importorskip("torch")
    source = torch.tensor(
        [0.6 + 0.2j, -0.3 + 0.4j], dtype=torch.complex128, requires_grad=True
    )
    phases = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    operator = SpectralExpectationOperator(torch.diag(phases), backend=backend)
    captured = []
    original = type(backend).matmul

    def observe(instance, left, right):
        result = original(instance, left, right)
        captured.append(result)
        return result

    monkeypatch.setattr(type(backend), "matmul", observe)
    state = source.conj() if conjugate else source
    passed, norm = runtime.stable_unitary(state, operator, HilbertSpace(2))
    assert passed is True
    assert norm == pytest.approx(1.0)
    assert len(captured) == 1
    evolved = captured[0].reshape(2)
    expected = (torch.exp(-1j * phases) * state / torch.linalg.vector_norm(state)).to(
        evolved.device
    )
    torch.testing.assert_close(evolved, expected)
    actual_gradients = torch.autograd.grad(
        evolved.real.sum(), (source, phases), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.real.sum(), (source, phases))
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference)
