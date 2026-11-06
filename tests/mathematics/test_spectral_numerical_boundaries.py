"""Independent range, ownership and primitive-admission spectral controls."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import (
    BanachSpaceEPI,
    BEPIElement,
    ContractiveDynamicsEngine,
    FrequencyOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
    SpectralExpectationOperator,
)
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.metrics import spectral_weighted_angle


def _backend(name):
    backend = get_backend(name)
    if backend.name != name:
        pytest.skip(f"{name} is unavailable")
    return backend


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("magnitude", [1e200, 1e308])
def test_large_state_normalization_preserves_spectral_expectation(
    backend_name, magnitude
):
    backend = _backend(backend_name)
    operator = SpectralExpectationOperator([2.0, 3.0], backend=backend)
    # The equally weighted components have expectation (2+3)/2 regardless of
    # their common nonzero amplitude or global phase.
    state = np.array([magnitude + magnitude * 1j, magnitude - magnitude * 1j])
    assert operator.expectation(state) == pytest.approx(2.5, rel=2e-14)


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_zero_generator_preserves_normalized_large_state(backend_name):
    backend = _backend(backend_name)
    engine = MathematicalDynamicsEngine(
        np.zeros((2, 2)), HilbertSpace(2), backend=backend
    )
    result = backend.to_numpy(engine.step([1e200, 1e200j], dt=0.0))
    expected = np.array([1, 1j]) / math.sqrt(2)
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0.0)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_weighted_angle_of_large_finite_rays_matches_their_unit_directions():
    operator = SpectralExpectationOperator([2.0, 3.0], backend=_backend("numpy"))
    actual = spectral_weighted_angle([1e200, 0], [0, 1e200], operator)
    assert actual == math.pi / 2
    assert spectral_weighted_angle([1e200, 1e200j], [1e200, 1e200j], operator) == 0


@pytest.mark.parametrize(
    "operator_type", [SpectralExpectationOperator, FrequencyOperator]
)
@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_operator_owns_input_and_exposes_detached_consistent_spectrum(
    operator_type, backend_name
):
    backend = _backend(backend_name)
    source = np.eye(2, dtype=np.complex128)
    operator = operator_type(source, backend=backend)
    source[0, 0] = -1
    operator.matrix[0, 0] = -3
    operator.eigenvalues[:] = -4
    operator.spectrum()[:] = -5

    assert operator.is_positive_semidefinite()
    assert operator.expectation([1, 0]) == 1
    np.testing.assert_array_equal(operator.matrix, np.eye(2))
    np.testing.assert_array_equal(operator.eigenvalues, [1, 1])
    for name in ("matrix", "eigenvalues"):
        with pytest.raises(AttributeError):
            setattr(operator, name, np.array([-1, -1]))


def test_internal_operator_reads_avoid_public_snapshot_copies(monkeypatch):
    operator = SpectralExpectationOperator([2, 3], backend=_backend("numpy"))
    matrix = operator.matrix
    eigenvalues = operator.eigenvalues
    matrix[0, 0] = -9
    eigenvalues[:] = -8
    calls = {"matrix": 0, "eigenvalues": 0}

    def counted_property(name):
        original = getattr(SpectralExpectationOperator, name)

        def observe(instance):
            calls[name] += 1
            return original.fget(instance)

        return property(observe)

    for name in calls:
        monkeypatch.setattr(SpectralExpectationOperator, name, counted_property(name))

    assert operator.expectation([1, 0]) == 2
    assert operator.is_positive_semidefinite()
    assert calls == {"matrix": 0, "eigenvalues": 0}
    np.testing.assert_array_equal(operator.matrix, np.diag([2, 3]))
    np.testing.assert_array_equal(operator.eigenvalues, [2, 3])
    assert calls == {"matrix": 1, "eigenvalues": 1}


@pytest.mark.parametrize("native_input", [False, True])
def test_jax_operator_ownership_survives_repeated_host_buffer_mutation(native_input):
    backend = _backend("jax")
    operators = []
    # Repeat transfers with distinct buffers and mutate immediately after each
    # construction; a deferred device transfer cannot retain caller storage.
    for index in range(8):
        diagonal = np.array([index + 1.0, index + 2.0])
        source = np.diag(diagonal).astype(np.complex128)
        supplied = (
            backend.as_array(source, dtype=np.complex128) if native_input else source
        )
        operator = SpectralExpectationOperator(supplied, backend=backend)
        source[:] = -np.eye(2)
        operators.append((operator, diagonal))

    for operator, diagonal in operators:
        np.testing.assert_array_equal(operator.matrix, np.diag(diagonal))
        np.testing.assert_array_equal(operator.spectrum(), diagonal)
        assert operator.is_positive_semidefinite()
        assert operator.expectation([1, 0]) == diagonal[0]


@pytest.mark.parametrize(
    "engine_type", [MathematicalDynamicsEngine, ContractiveDynamicsEngine]
)
def test_generator_input_and_export_cannot_change_the_admitted_law(engine_type):
    size = 2 if engine_type is MathematicalDynamicsEngine else 4
    source = np.zeros((size, size), dtype=np.complex128)
    engine = engine_type(source, HilbertSpace(2), backend=_backend("numpy"))
    source[0, 1] = 9
    engine.generator[1, 0] = 8
    np.testing.assert_array_equal(engine.generator, np.zeros((size, size)))
    with pytest.raises(AttributeError):
        engine.generator = source
    if engine_type is MathematicalDynamicsEngine:
        np.testing.assert_array_equal(engine.step([1, 0]), [1, 0])
    else:
        np.testing.assert_array_equal(engine.step(np.eye(2) / 2), np.eye(2) / 2)


@pytest.mark.parametrize(
    "bad", [True, "1", Fraction(1, 10**400), Fraction(10**400), np.inf]
)
@pytest.mark.parametrize(
    "container", [list, lambda values: np.array(values, dtype=object)]
)
def test_spectral_inputs_are_admitted_before_coercion(bad, container):
    backend = _backend("numpy")
    state = container([1, bad])
    operator = SpectralExpectationOperator([1, 2], backend=backend)
    engine = MathematicalDynamicsEngine(
        np.zeros((2, 2)), HilbertSpace(2), backend=backend
    )
    density_engine = ContractiveDynamicsEngine(
        np.zeros((4, 4)), HilbertSpace(2), backend=backend
    )
    with pytest.raises(TNFRValueError):
        operator.expectation(state)
    with pytest.raises(TNFRValueError):
        spectral_weighted_angle(state, [1, 0], operator)
    with pytest.raises(TNFRValueError):
        SpectralExpectationOperator(state, backend=backend)
    with pytest.raises(TNFRValueError):
        MathematicalDynamicsEngine(
            container([[1, 0], [0, bad]]), HilbertSpace(2), backend=backend
        )
    with pytest.raises(TNFRValueError):
        engine.step(state)
    with pytest.raises(TNFRValueError):
        density_engine.step(container([[1, 0], [0, bad]]))


@pytest.mark.parametrize("bad", [True, "0", 0j, Fraction(1, 10**400), -np.inf])
@pytest.mark.parametrize("producer", ["element", "zero", "basis"])
def test_bepi_grid_producers_preserve_original_scalar_admission(bad, producer):
    grid = [bad, 1]
    space = BanachSpaceEPI()
    with pytest.raises(TNFRValueError, match="x_grid"):
        if producer == "element":
            BEPIElement([0, 1], [0], grid)
        elif producer == "zero":
            space.zero_element(continuous_size=2, discrete_size=1, x_grid=grid)
        else:
            space.canonical_basis(continuous_size=2, discrete_size=1, x_grid=grid)


def test_bepi_grid_rejects_complex_array_without_discarding_imaginary_channel():
    with pytest.raises(TNFRValueError, match="x_grid"):
        BEPIElement([0, 1], [0], np.array([2j, 1 + 3j]))


def test_bepi_grid_preserves_exact_reals_and_owns_the_materialized_coordinates():
    grid = np.array([0.0, 0.5, 1.0])
    element = BEPIElement([0, 1, 0], [0], grid)
    grid[0] = -1
    np.testing.assert_array_equal(element.x_grid, [0, 0.5, 1])
    exact = BEPIElement([0, 1, 0], [0], [0, Fraction(1, 2), 1])
    np.testing.assert_array_equal(exact.x_grid, [0, 0.5, 1])


@pytest.mark.parametrize("bad", [True, -1, np.nan, np.inf, Fraction(1, 10**400)])
def test_invalid_tolerance_cannot_certify_spectral_properties(bad):
    operator = SpectralExpectationOperator([-1, 1], backend=_backend("numpy"))
    with pytest.raises(TNFRValueError):
        operator.is_positive_semidefinite(atol=bad)


def test_unrepresentable_unnormalized_expectation_rejects_instead_of_returning_infinity():
    operator = SpectralExpectationOperator([2, 3], backend=_backend("numpy"))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(TNFRValueError):
        operator.expectation([1e200, 0], normalise=False)


def test_torch_owned_generator_and_scale_normalization_retain_gradients():
    backend = _backend("torch")
    torch = pytest.importorskip("torch")
    generator = torch.tensor(
        [[0.3, 0.0], [0.0, -0.2]], dtype=torch.complex128, requires_grad=True
    )
    engine = MathematicalDynamicsEngine(generator, HilbertSpace(2), backend=backend)
    state = torch.tensor([2.0, 1.0], dtype=torch.complex128, requires_grad=True)
    value = engine.step(state, dt=0.2).real[0]
    generator_grad, state_grad = torch.autograd.grad(value, (generator, state))
    assert torch.isfinite(generator_grad).all()
    assert torch.isfinite(state_grad).all()
    expected = 1 / (5 * math.sqrt(5)) * math.cos(0.06)
    assert state_grad[0].real.item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("view_kind", ["conjugate", "negative"])
@pytest.mark.parametrize("normalize", [False, True])
def test_torch_lazy_views_keep_values_and_gradients_through_admission(
    view_kind, normalize
):
    backend = _backend("torch")
    torch = pytest.importorskip("torch")
    source = torch.tensor([1 + 1j, 2 - 1j], dtype=torch.complex128, requires_grad=True)
    # The imaginary part of a conjugate supplies a public negative-bit view.
    view = source.conj() if view_kind == "conjugate" else source.conj().imag
    assert view.is_conj() if view_kind == "conjugate" else view.is_neg()
    before = (view.is_conj(), view.is_neg())
    engine = MathematicalDynamicsEngine(
        np.zeros((2, 2)), HilbertSpace(2), backend=backend
    )

    result = engine.step(view, dt=0.0, normalize=normalize)

    expected = (
        np.array([1 - 1j, 2 + 1j])
        if view_kind == "conjugate"
        else np.array([-1.0, 1.0])
    )
    if normalize:
        expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(backend.to_numpy(result), expected, rtol=2e-15, atol=0)
    assert (view.is_conj(), view.is_neg()) == before

    # Independent native formula, starting from a separate leaf, must retain
    # the same gradient through the conjugation/imaginary projection.
    reference = torch.tensor(
        [1 + 1j, 2 - 1j], dtype=torch.complex128, requires_grad=True
    )
    reference_view = (
        reference.conj() if view_kind == "conjugate" else reference.conj().imag
    ).to(dtype=torch.complex128)
    if normalize:
        reference_view = reference_view / torch.linalg.vector_norm(reference_view)
    value = result[0].real + 2 * result[1].imag
    expected_value = reference_view[0].real + 2 * reference_view[1].imag
    (actual_gradient,) = torch.autograd.grad(value, source)
    (expected_gradient,) = torch.autograd.grad(expected_value, reference)
    np.testing.assert_allclose(
        backend.to_numpy(actual_gradient),
        backend.to_numpy(expected_gradient),
        rtol=2e-14,
        atol=1e-16,
    )
