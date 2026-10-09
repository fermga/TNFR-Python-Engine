"""Factory admission, native gradients and shared spectral scalar contracts."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr._spectral_expectation import spectral_expectation_payload
from tnfr.errors import TNFRValueError
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.operators import SpectralExpectationOperator
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
    make_spectral_expectation_operator,
)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "1", 1.0, 0, -1])
def test_factory_dimensions_use_the_shared_strict_integer_boundary(bad):
    with pytest.raises(TNFRValueError, match="positive integer"):
        make_spectral_expectation_operator(bad)


@pytest.mark.parametrize(
    "bad", [True, "2", float("nan"), float("inf"), Fraction(1, 10**400)]
)
@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("kind", ["spectrum", "frequency"])
def test_factories_admit_original_components_before_backend_conversion(
    monkeypatch, bad, backend_name, kind
):
    backend = get_backend(backend_name)
    if backend.name != backend_name:
        pytest.skip(f"{backend_name} is unavailable")
    monkeypatch.setenv("TNFR_MATH_BACKEND", backend_name)
    with pytest.raises(TNFRValueError):
        if kind == "spectrum":
            make_spectral_expectation_operator(2, spectrum=[1, bad])
        else:
            make_frequency_operator([[1, 0], [0, bad]])


@pytest.mark.parametrize(
    "bad", [True, "0.5", float("nan"), float("inf"), Fraction(1, 10**400)]
)
@pytest.mark.parametrize("route", ["factory", "legacy_factory", "operator", "payload"])
def test_comparison_floors_and_thresholds_share_represented_real_admission(bad, route):
    with pytest.raises(TNFRValueError, match="finite representable real"):
        if route == "factory":
            make_spectral_expectation_operator(
                2, spectrum=[1, 2], expectation_floor=bad
            )
        elif route == "legacy_factory":
            make_coherence_operator(2, spectrum=[1, 2], c_min=bad)
        elif route == "operator":
            SpectralExpectationOperator([1, 2], expectation_floor=bad)
        else:
            spectral_expectation_payload(
                value=1, threshold=bad, passed=False, provenance="test"
            )


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_factory_retains_real_projection_and_detached_ownership(
    monkeypatch, backend_name
):
    backend = get_backend(backend_name)
    if backend.name != backend_name:
        pytest.skip(f"{backend_name} is unavailable")
    monkeypatch.setenv("TNFR_MATH_BACKEND", backend_name)
    spectrum = np.array([2 + 1e-10j, 4 - 1e-10j])
    operator = make_spectral_expectation_operator(np.int64(2), spectrum=spectrum)
    spectrum[:] = -10

    np.testing.assert_array_equal(operator.matrix, np.diag([2, 4]))
    assert operator.expectation([1, 1j]) == pytest.approx(3)
    with pytest.raises(TNFRValueError, match="real-valued"):
        make_spectral_expectation_operator(2, spectrum=[2 + 1e-8j, 4])


def test_factory_real_spectrum_projection_preserves_torch_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TNFR_MATH_BACKEND", "torch")
    spectrum = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)

    operator = make_spectral_expectation_operator(2, spectrum=spectrum)
    state = torch.tensor(
        [1, 2j], dtype=torch.complex128, device=operator._matrix_backend.device
    )
    expectation = torch.vdot(state, operator._matrix_backend @ state).real
    expectation.backward()

    assert expectation.item() == pytest.approx(18)
    # d(s1*|1|^2 + s2*|2j|^2)/d(s1,s2) = (1,4).
    np.testing.assert_array_equal(spectrum.grad.numpy(), [1, 4])


def test_frequency_factory_keeps_its_matrix_only_domain():
    with pytest.raises(TNFRValueError, match="square"):
        make_frequency_operator([1, 2])


def test_explicit_comparison_floor_can_be_negative():
    operator = make_spectral_expectation_operator(
        2, spectrum=[2, 4], expectation_floor=Fraction(-1, 2)
    )
    assert operator.expectation_floor == -0.5
    assert operator.is_positive_semidefinite()
