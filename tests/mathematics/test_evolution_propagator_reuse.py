"""Invocation-local propagators preserve state controls and native derivatives."""

from __future__ import annotations

import math

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


class _CountingBackend:
    def __init__(self):
        self.delegate = get_backend("numpy")
        self.exponentials = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def matrix_exp(self, matrix):
        self.exponentials += 1
        return self.delegate.matrix_exp(matrix)


def _case(kind, *, backend=None, use_scipy=False, engine_type=None):
    if kind == "unitary":
        cls = engine_type or MathematicalDynamicsEngine
        generator = np.array([[0.2, 0.3], [0.3, -0.4]])
        state = np.array([0.6, 0.8j])
    else:
        cls = engine_type or ContractiveDynamicsEngine
        # Pure dephasing preserves populations and decays both coherences.
        generator = np.diag([0, -0.7, -0.7, 0])
        state = np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]])
    engine = cls(
        generator,
        HilbertSpace(2),
        backend=backend or get_backend("numpy"),
        use_scipy=use_scipy,
    )
    return engine, state, generator


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("use_scipy", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_evolve_computes_one_exponential_per_nonempty_call(
    kind, use_scipy, steps, monkeypatch
):
    backend = _CountingBackend()
    scipy_calls = []
    original = dynamics._scipy_expm

    def counted(matrix):
        scipy_calls.append(matrix.copy())
        return original(matrix)

    monkeypatch.setattr(dynamics, "_scipy_expm", counted)
    engine, state, _ = _case(kind, backend=backend, use_scipy=use_scipy)
    result = engine.evolve(state, steps=steps, dt=0.1)
    assert result.shape[0] == steps + 1
    expected = int(steps > 0)
    assert backend.exponentials == (0 if use_scipy else expected)
    assert len(scipy_calls) == (expected if use_scipy else 0)
    if kind == "density" and not steps:
        assert math.isnan(engine.last_contractivity_gap)


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_reused_trajectory_and_latest_monitor_match_individual_steps(kind):
    optimized, state, _ = _case(kind)
    reference, _, _ = _case(kind)
    expected = [state]
    for _ in range(5):
        expected.append(reference.step(expected[-1], dt=0.13))
    actual = optimized.evolve(state, steps=5, dt=0.13)
    np.testing.assert_array_equal(actual, expected)
    if kind == "density":
        assert optimized.last_contractivity_gap == reference.last_contractivity_gap
        expected_final = state.copy()
        expected_final[0, 1] *= np.exp(-0.7 * 5 * 0.13)
        expected_final[1, 0] *= np.exp(-0.7 * 5 * 0.13)
        np.testing.assert_allclose(actual[-1], expected_final, atol=2e-15)


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_later_calls_read_current_time_and_state_without_reusing_old_propagator(kind):
    backend = _CountingBackend()
    engine, state, generator_source = _case(kind, backend=backend)
    reference, _, _ = _case(kind)
    time = np.array(0.1)
    engine.evolve(state, steps=4, dt=time)
    time[...] = -0.2
    state[...] = state[::-1]
    generator_source[...] = 0  # The engine's generator remains its owned snapshot.
    actual = engine.evolve(state, steps=4, dt=time)
    expected = [state.copy()]
    for _ in range(4):
        expected.append(reference.step(expected[-1], dt=-0.2))
    assert backend.exponentials == 2
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("override", ["subclass", "class_method"])
def test_custom_public_step_keeps_per_step_dispatch(kind, override, monkeypatch):
    base = (
        MathematicalDynamicsEngine if kind == "unitary" else ContractiveDynamicsEngine
    )
    original_step = base.step
    calls = []

    def replacement(self, state, **kwargs):
        calls.append(kwargs["dt"])
        return original_step(self, state, **kwargs)

    if override == "subclass":
        cls = type("CustomEvolution", (base,), {"step": replacement})
    else:
        cls = base
        monkeypatch.setattr(base, "step", replacement)
    backend = _CountingBackend()
    engine, state, _ = _case(kind, backend=backend, engine_type=cls)
    engine.evolve(state, steps=3, dt=0.2)
    assert calls == [0.2, 0.2, 0.2]
    assert backend.exponentials == 3


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_each_step_still_checks_concrete_outputs(kind):
    backend = _CountingBackend()
    engine, state, _ = _case(kind, backend=backend)
    calls = 0

    def matmul(left, right):
        nonlocal calls
        calls += 1
        result = backend.delegate.matmul(left, right)
        return np.full_like(result, np.nan) if calls == 3 else result

    backend.matmul = matmul
    with pytest.raises(TNFRValueError):
        engine.evolve(state, steps=5, dt=0.1)
    assert calls == 3
    assert backend.exponentials == 1


@pytest.mark.parametrize("kind", ["unitary", "density"])
@pytest.mark.parametrize("invalid", ["time", "state", "shape"])
def test_admission_failures_precede_the_first_exponential(kind, invalid):
    backend = _CountingBackend()
    engine, state, _ = _case(kind, backend=backend)
    time = True if invalid == "time" else 0.1
    if invalid == "state":
        state = [True, 0] if kind == "unitary" else [[True, 0], [0, 0]]
    elif invalid == "shape":
        state = [1, 0, 0] if kind == "unitary" else np.eye(3)
    with pytest.raises(TNFRValueError):
        engine.evolve(state, steps=4, dt=time)
    assert backend.exponentials == 0


def test_reused_propagator_retains_contractivity_violation_and_measured_gap():
    backend = _CountingBackend()
    engine, state, _ = _case("density", backend=backend)
    reference, _, _ = _case("density")
    with pytest.raises(TNFRValueError, match="Contractivity violated"):
        reference.step(state, dt=-0.2, raise_on_violation=True)
    with pytest.raises(TNFRValueError, match="Contractivity violated"):
        engine.evolve(state, steps=4, dt=-0.2, raise_on_violation=True)
    assert engine.last_contractivity_gap == reference.last_contractivity_gap
    assert engine.last_contractivity_gap < 0
    assert backend.exponentials == 1


@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_torch_reuse_preserves_time_and_generator_gradients(kind):
    backend = get_backend("torch")
    if backend.name != "torch":
        pytest.skip("Torch backend unavailable")
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    time = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    steps = 4
    if kind == "unitary":
        generator = rate * torch.eye(2, dtype=torch.complex128)
        engine = MathematicalDynamicsEngine(generator, HilbertSpace(2), backend=backend)
        value = engine.evolve([1, 0], steps=steps, dt=time, normalize=False)[-1, 0].real
        derivative = -math.sin(steps * 0.7 * 0.2)
    else:
        generator = -rate * torch.eye(4, dtype=torch.complex128)
        engine = ContractiveDynamicsEngine(generator, HilbertSpace(2), backend=backend)
        value = engine.evolve(
            [[1, 0], [0, 0]],
            steps=steps,
            dt=time,
            normalize_trace=False,
            enforce_contractivity=False,
        )[-1, 0, 0].real
        derivative = -math.exp(-steps * 0.7 * 0.2)
    time_gradient, rate_gradient = torch.autograd.grad(value, (time, rate))
    assert time_gradient.item() == pytest.approx(steps * 0.7 * derivative, rel=2e-13)
    assert rate_gradient.item() == pytest.approx(steps * 0.2 * derivative, rel=2e-13)


@pytest.mark.parametrize("backend_name", ["torch", "jax"])
@pytest.mark.parametrize("kind", ["unitary", "density"])
def test_repeated_evolve_calls_build_fresh_native_time_gradient_graphs(
    backend_name, kind
):
    backend = get_backend(backend_name)
    if backend.name != backend_name:
        pytest.skip(f"{backend_name} backend unavailable")
    cls = MathematicalDynamicsEngine if kind == "unitary" else ContractiveDynamicsEngine
    generator = np.eye(2) if kind == "unitary" else -np.eye(4)
    engine = cls(generator, HilbertSpace(2), backend=backend)

    def objective(time):
        if kind == "unitary":
            return engine.evolve([1, 0], steps=3, dt=time, normalize=False)[-1, 0].real
        return engine.evolve(
            [[1, 0], [0, 0]],
            steps=3,
            dt=time,
            normalize_trace=False,
            enforce_contractivity=False,
        )[-1, 0, 0].real

    for point in [0.2, 0.4]:
        if backend_name == "torch":
            torch = pytest.importorskip("torch")
            time = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            (gradient,) = torch.autograd.grad(objective(time), (time,))
            actual = gradient.item()
        else:
            jax = pytest.importorskip("jax")
            actual = float(jax.grad(objective)(point))
        expected = -3 * (
            math.sin(3 * point) if kind == "unitary" else math.exp(-3 * point)
        )
        assert actual == pytest.approx(expected, rel=2e-13)
