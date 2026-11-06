"""Evolution scalar admission and preservation of native exponential routes."""

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import dynamics
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.dynamics import (
    ContractiveDynamicsEngine,
    MathematicalDynamicsEngine,
)
from tnfr.mathematics.spaces import HilbertSpace


def _engine_and_state(kind, *, backend=None, use_scipy=None):
    backend = backend or get_backend("numpy")
    if kind == "unitary":
        return (
            MathematicalDynamicsEngine(
                np.diag([1.0, 0.0]),
                HilbertSpace(2),
                backend=backend,
                use_scipy=use_scipy,
            ),
            np.array([1.0, 0.0]),
        )
    return (
        ContractiveDynamicsEngine(
            -np.eye(4), HilbertSpace(2), backend=backend, use_scipy=use_scipy
        ),
        np.diag([1.0, 0.0]),
    )


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(False), "0.2", 1j, np.nan, np.inf, Fraction(1, 10**400), [1]],
)
@pytest.mark.parametrize("zero_steps", [False, True])
def test_time_is_admitted_even_without_evolution(kind, bad, zero_steps):
    engine, state = _engine_and_state(kind)
    with pytest.raises(TNFRValueError, match="dt must be"):
        if zero_steps:
            engine.evolve(state, steps=0, dt=bad)
        else:
            engine.step(state, dt=bad)


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, "1", -1])
def test_trajectory_count_is_a_nonnegative_integer(kind, bad):
    engine, state = _engine_and_state(kind)
    with pytest.raises(TNFRValueError, match="non-negative integer"):
        engine.evolve(state, steps=bad)


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_signed_time_and_numpy_integer_counts_remain_available(kind):
    engine, state = _engine_and_state(kind)
    kwargs = {"normalize": False} if kind == "unitary" else {"normalize_trace": False}
    trajectory = engine.evolve(state, steps=np.int64(2), dt=Fraction(-1, 4), **kwargs)
    multiplier = np.exp(0.5j) if kind == "unitary" else math.exp(0.5)
    np.testing.assert_allclose(trajectory[-1], multiplier * state, atol=1e-14)


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_nonfinite_exponential_argument_rejects_before_backend_call(kind, monkeypatch):
    if kind == "unitary":
        engine = MathematicalDynamicsEngine(
            np.diag([1e308, 0]), HilbertSpace(2), use_scipy=False
        )
        state = [1, 0]
    else:
        engine = ContractiveDynamicsEngine(
            -1e308 * np.eye(4), HilbertSpace(2), use_scipy=False
        )
        state = np.eye(2) / 2

    def unexpected_call(_argument):
        pytest.fail("An unrepresentable argument must not reach matrix_exp")

    monkeypatch.setattr(
        type(engine.backend), "matrix_exp", lambda self, a: unexpected_call(a)
    )
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(TNFRValueError):
        engine.step(state, dt=2)


def test_concrete_nonfinite_evolution_rejects_when_normalization_is_disabled():
    engine = ContractiveDynamicsEngine(
        -np.eye(4), HilbertSpace(2), backend=get_backend("numpy")
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(TNFRValueError, match="finite"),
    ):
        engine.step(
            np.eye(2) / 2,
            dt=-1000,
            normalize_trace=False,
            enforce_contractivity=False,
            symmetrize=False,
        )


@pytest.mark.parametrize("symmetrize", [False, True])
def test_nonfinite_trace_cannot_normalize_finite_density_to_zero(symmetrize):
    engine = ContractiveDynamicsEngine(
        np.zeros((4, 4)), HilbertSpace(2), backend=get_backend("numpy")
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(TNFRValueError, match="Trace must be finite"),
    ):
        engine.step(
            np.diag([1e308, 1e308]),
            dt=0,
            symmetrize=symmetrize,
            enforce_contractivity=False,
        )


class _MissingExponentialBackend:
    """A functioning numerical backend with an unavailable exponential."""

    def __init__(self):
        self.delegate = get_backend("numpy")
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def matrix_exp(self, _matrix):
        self.calls += 1
        raise NotImplementedError("native exponential unavailable")


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("requested", [None, True, False])
def test_explicit_and_automatic_fallback_routes_preserve_their_semantics(
    kind, requested
):
    pytest.importorskip("scipy.linalg")
    backend = _MissingExponentialBackend()
    engine, state = _engine_and_state(kind, backend=backend, use_scipy=requested)
    assert backend.calls == (1 if requested is None else 0)
    if requested is False:
        with pytest.raises(NotImplementedError, match="native exponential unavailable"):
            engine.step(state, dt=0.25)
    else:
        result = engine.step(state, dt=0.25)
        expected = state * np.exp(-0.25j) if kind == "unitary" else state
        np.testing.assert_allclose(result, expected, atol=1e-14)
        assert backend.calls == (1 if requested is None else 0)


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("requested", [None, True])
def test_unavailable_fallback_stays_explicit(kind, requested, monkeypatch):
    monkeypatch.setattr(dynamics, "_scipy_expm", None)
    with pytest.raises(RuntimeError, match="SciPy"):
        _engine_and_state(
            kind, backend=_MissingExponentialBackend(), use_scipy=requested
        )


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_false_string_does_not_request_scipy(kind, monkeypatch):
    monkeypatch.setattr(dynamics, "_scipy_expm", None)
    engine, state = _engine_and_state(kind, use_scipy="false")
    expected = state * np.exp(-0.25j) if kind == "unitary" else state
    np.testing.assert_allclose(engine.step(state, dt=0.25), expected, atol=1e-14)


@pytest.mark.parametrize("backend_name", ["torch", "jax"])
@pytest.mark.parametrize("bad", [True, 1j, np.nan, np.inf, [0.2]])
def test_native_times_retain_scalar_real_finite_admission(backend_name, bad):
    backend = get_backend(backend_name)
    if backend.name != backend_name:
        pytest.skip(f"{backend_name} backend unavailable")
    engine, state = _engine_and_state("unitary", backend=backend)
    with pytest.raises(TNFRValueError, match="dt must be"):
        engine.step(state, dt=backend.as_array(bad))


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_torch_time_gradient_survives_admission(kind):
    torch = pytest.importorskip("torch")
    backend = get_backend("torch")
    if backend.name != "torch":
        pytest.skip("Torch backend unavailable")
    engine, state = _engine_and_state(kind, backend=backend)
    time = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    if kind == "unitary":
        value = engine.step(state, dt=time, normalize=False)[0].real
        expected = -math.sin(0.4)
    else:
        value = engine.step(
            state, dt=time, normalize_trace=False, enforce_contractivity=False
        )[0, 0].real
        expected = -math.exp(-0.4)
    value.backward()
    assert time.grad.item() == pytest.approx(expected, rel=1e-12)


def test_torch_integer_time_is_not_prematurely_narrowed_to_complex64():
    torch = pytest.importorskip("torch")
    backend = get_backend("torch")
    if backend.name != "torch":
        pytest.skip("Torch backend unavailable")
    engine, state = _engine_and_state("unitary", backend=backend)
    time = 2**24 + 1
    result = engine.step(
        state, dt=torch.tensor(time, dtype=torch.int64), normalize=False
    )
    np.testing.assert_allclose(
        backend.to_numpy(result), [np.exp(-1j * time), 0], atol=1e-14
    )


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_jax_traced_time_keeps_its_gradient(kind):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = get_backend("jax")
    if backend.name != "jax":
        pytest.skip("JAX backend unavailable")
    engine, state = _engine_and_state(kind, backend=backend)

    def value(time):
        if kind == "unitary":
            return engine.step(state, dt=time, normalize=False)[0].real
        return engine.step(
            state, dt=time, normalize_trace=False, enforce_contractivity=False
        )[0, 0].real

    expected = -math.sin(0.4) if kind == "unitary" else -math.exp(-0.4)
    assert float(jax.grad(value)(jnp.asarray(0.4))) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("amplitude", [1e-200, 1e200])
@pytest.mark.parametrize("center", [False, True])
def test_density_frobenius_norm_observes_finite_large_and_tiny_amplitudes(
    amplitude, center
):
    engine, _ = _engine_and_state("density")
    density = np.diag([amplitude, 0.0])
    expected = amplitude / math.sqrt(2) if center else amplitude
    assert engine.frobenius_norm(density, center=center) == pytest.approx(
        expected, rel=3e-15, abs=0
    )


def test_large_finite_growth_cannot_bypass_the_contractivity_monitor():
    engine = ContractiveDynamicsEngine(
        np.eye(4),
        HilbertSpace(2),
        ensure_contractive=False,
        backend=get_backend("numpy"),
    )
    density = np.diag([1e200, 0.0])
    with pytest.raises(TNFRValueError, match="Frobenius norm increased"):
        engine.step(density, dt=0.1, normalize_trace=False, raise_on_violation=True)
    assert engine.last_contractivity_gap == pytest.approx(
        1e200 * (1 - math.exp(0.1)) / math.sqrt(2), rel=5e-15
    )


def test_unrepresentable_concrete_density_norm_rejects_instead_of_becoming_unavailable():
    engine, _ = _engine_and_state("density")
    density = np.array([[0.0, 1.5e308], [1.5e308, 0.0]])
    with pytest.raises(TNFRValueError, match="norm is not representable"):
        engine.frobenius_norm(density)
    with pytest.raises(TNFRValueError, match="norm is not representable"):
        engine.step(density, dt=0, normalize_trace=False, symmetrize=False)


@pytest.mark.parametrize("backend_name", ["torch", "jax"])
def test_monitored_density_evolution_retains_native_time_gradient(backend_name):
    library = pytest.importorskip(backend_name)
    backend = get_backend(backend_name)
    if backend.name != backend_name:
        pytest.skip(f"{backend_name} backend unavailable")
    engine, state = _engine_and_state("density", backend=backend)

    def value(time):
        return engine.step(
            state, dt=time, normalize_trace=False, raise_on_violation=True
        )[0, 0].real

    if backend_name == "torch":
        time = library.tensor(0.4, dtype=library.float64, requires_grad=True)
        value(time).backward()
        derivative = time.grad.item()
        assert engine.last_contractivity_gap == pytest.approx(
            (1 - math.exp(-0.4)) / math.sqrt(2), rel=1e-12
        )
    else:
        derivative = float(library.grad(value)(library.numpy.asarray(0.4)))
        assert math.isnan(engine.last_contractivity_gap)
    assert derivative == pytest.approx(-math.exp(-0.4), rel=1e-12)
