"""Automatic differentiation checks for optional backends."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.dynamics import ContractiveDynamicsEngine, MathematicalDynamicsEngine
from tnfr.mathematics.spaces import HilbertSpace


def _require_backend(name: str) -> object:
    backend = get_backend(name)
    if backend.name != name:
        pytest.skip(f"Backend '{name}' is unavailable; installed: {backend.name!r}.")
    return backend


def test_jax_norm_gradient_matches_analytic() -> None:
    backend = _require_backend("jax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    vector = jnp.array([0.4, -1.2, 0.8], dtype=jnp.float64)

    def norm_squared(val: object) -> object:
        arr = backend.as_array(val, dtype=jnp.float64)
        norm = backend.norm(arr)
        return jnp.real(norm ** 2)

    grad = jax.grad(norm_squared)(vector)
    expected = 2.0 * vector
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_jax_expectation_gradient_matches_closed_form() -> None:
    backend = _require_backend("jax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    matrix_np = np.array([[1.5, 0.2], [0.2, -0.3]], dtype=np.float64)
    matrix = backend.as_array(matrix_np, dtype=jnp.float64)
    vector_np = np.array([0.7, -0.45], dtype=np.float64)
    vector = jnp.array(vector_np)

    def expectation(val: object) -> object:
        arr = backend.as_array(val, dtype=jnp.float64)
        hv = backend.matmul(matrix, arr)
        numerator = jnp.vdot(arr, hv)
        denom = jnp.vdot(arr, arr)
        return jnp.real(numerator / denom)

    grad = jax.grad(expectation)(vector)

    denom_np = float(np.dot(vector_np, vector_np))
    energy_np = float(vector_np @ (matrix_np @ vector_np) / denom_np)
    expected = 2.0 * (matrix_np @ vector_np - energy_np * vector_np) / denom_np
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-6, atol=1e-6)


def test_torch_norm_gradient_matches_analytic() -> None:
    backend = _require_backend("torch")
    torch = pytest.importorskip("torch")

    vector = torch.tensor([0.5, -0.9, 0.25], dtype=torch.double, requires_grad=True)

    arr = backend.as_array(vector)
    norm = backend.norm(arr)
    value = (norm ** 2).real
    grad, = torch.autograd.grad(value, vector)

    expected = 2.0 * vector.detach().cpu().numpy()
    np.testing.assert_allclose(grad.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_torch_expectation_gradient_matches_closed_form() -> None:
    backend = _require_backend("torch")
    torch = pytest.importorskip("torch")

    matrix_np = np.array([[1.2, -0.15], [-0.15, 0.6]], dtype=np.float64)
    matrix = backend.as_array(torch.tensor(matrix_np, dtype=torch.double))
    vector = torch.tensor([0.9, -0.4], dtype=torch.double, requires_grad=True)

    arr = backend.as_array(vector)
    hv = backend.matmul(matrix, arr)
    numerator = torch.dot(arr, hv)
    denom = torch.dot(arr, arr)
    value = (numerator / denom).real
    grad, = torch.autograd.grad(value, vector)

    vector_np = vector.detach().cpu().numpy()
    denom_np = float(np.dot(vector_np, vector_np))
    energy_np = float(vector_np @ (matrix_np @ vector_np) / denom_np)
    expected = 2.0 * (matrix_np @ vector_np - energy_np * vector_np) / denom_np
    np.testing.assert_allclose(grad.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_torch_dynamics_step_supports_autodiff() -> None:
    backend = _require_backend("torch")
    torch = pytest.importorskip("torch")

    hilbert = HilbertSpace(dimension=2)
    generator = torch.tensor(
        [[0.3 + 0.0j, 0.2 - 0.1j], [0.2 + 0.1j, -0.4 + 0.0j]],
        dtype=torch.complex128,
    )
    engine = MathematicalDynamicsEngine(generator, hilbert, backend=backend)

    state = torch.tensor([0.8 + 0.0j, 0.1 + 0.3j], dtype=torch.complex128, requires_grad=True)
    result = engine.step(state, dt=0.25, normalize=False)
    value = result.abs().pow(2).sum()

    grad, = torch.autograd.grad(value, state)
    grad_np = grad.detach().cpu().numpy()
    assert grad_np.shape == (2,)
    assert np.all(np.isfinite(grad_np))


def test_torch_dynamics_step_autodiff_with_normalization() -> None:
    backend = _require_backend("torch")
    torch = pytest.importorskip("torch")

    hilbert = HilbertSpace(dimension=2)
    generator = torch.tensor(
        [[0.1 + 0.0j, 0.3 - 0.2j], [0.3 + 0.2j, -0.25 + 0.0j]],
        dtype=torch.complex128,
    )
    engine = MathematicalDynamicsEngine(generator, hilbert, backend=backend)

    state = torch.tensor([0.4 + 0.1j, 0.2 - 0.3j], dtype=torch.complex128, requires_grad=True)
    result = engine.step(state, dt=0.4, normalize=True)
    value = result.abs().pow(2).sum()

    grad, = torch.autograd.grad(value, state)
    grad_np = grad.detach().cpu().numpy()
    assert grad_np.shape == (2,)
    assert np.all(np.isfinite(grad_np))


def test_jax_contractive_step_supports_autodiff() -> None:
    backend = _require_backend("jax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    hilbert = HilbertSpace(dimension=2)
    generator_np = np.diag([-0.4, -0.2, -0.3, -0.1]).astype(np.complex128)
    generator = backend.as_array(generator_np, dtype=jnp.complex128)
    engine = ContractiveDynamicsEngine(
        generator,
        hilbert,
        ensure_contractive=False,
        backend=backend,
    )

    density = jnp.array(
        [[0.7 + 0.0j, 0.1 - 0.05j], [0.1 + 0.05j, 0.3 + 0.0j]],
        dtype=jnp.complex128,
    )

    def frobenius_energy(density_matrix: object) -> object:
        evolved = engine.step(
            density_matrix,
            dt=0.15,
            normalize_trace=False,
            enforce_contractivity=False,
            symmetrize=False,
        )
        return jnp.real(jnp.sum(jnp.abs(evolved) ** 2))

    grad = jax.grad(frobenius_energy)(density)
    grad_np = np.asarray(grad)
    assert grad_np.shape == density.shape
    assert np.all(np.isfinite(grad_np))


def test_jax_contractive_step_autodiff_with_controls() -> None:
    backend = _require_backend("jax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    hilbert = HilbertSpace(dimension=2)
    generator_np = np.diag([-0.3, -0.2, -0.25, -0.15]).astype(np.complex128)
    generator = backend.as_array(generator_np, dtype=jnp.complex128)
    engine = ContractiveDynamicsEngine(
        generator,
        hilbert,
        ensure_contractive=False,
        backend=backend,
    )

    density = jnp.array(
        [[0.6 + 0.0j, 0.05 - 0.08j], [0.05 + 0.08j, 0.4 + 0.0j]],
        dtype=jnp.complex128,
    )

    def monitored_energy(density_matrix: object) -> object:
        evolved = engine.step(
            density_matrix,
            dt=0.2,
            normalize_trace=True,
            enforce_contractivity=True,
            raise_on_violation=False,
            symmetrize=True,
        )
        return jnp.real(jnp.sum(jnp.abs(evolved) ** 2))

    grad = jax.grad(monitored_energy)(density)
    grad_np = np.asarray(grad)
    assert grad_np.shape == density.shape
    assert np.all(np.isfinite(grad_np))
