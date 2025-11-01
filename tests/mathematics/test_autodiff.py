"""Automatic differentiation checks for optional backends."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics.backend import get_backend


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
