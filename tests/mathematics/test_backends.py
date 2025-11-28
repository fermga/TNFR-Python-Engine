"""Cross-backend numerical consistency checks."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics import CoherenceOperator, HilbertSpace
from tnfr.mathematics.backend import ensure_array, ensure_numpy, get_backend
from tnfr.mathematics.dynamics import (
    ContractiveDynamicsEngine,
    MathematicalDynamicsEngine,
)

_BACKEND_NAMES = ("numpy", "jax", "torch")


def _require_backend(name: str) -> object:
    backend = get_backend(name)
    if backend.name != name:
        pytest.skip(f"Backend '{name}' is unavailable; installed: {backend.name!r}.")
    return backend


def _to_numpy(value: object, *, backend: object) -> np.ndarray:
    return np.asarray(ensure_numpy(value, backend=backend))


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
def test_coherence_operator_matches_numpy(
    backend_name: str, structural_tolerances: dict[str, float]
) -> None:
    """Coherence operators must agree across available numerical backends."""

    backend = _require_backend(backend_name)
    reference_backend = get_backend("numpy")

    matrix = np.array([[0.9, 0.2 - 0.05j], [0.2 + 0.05j, 0.4]], dtype=np.complex128)
    state = np.array([0.6 + 0.1j, 0.3 - 0.2j], dtype=np.complex128)

    reference = CoherenceOperator(matrix, backend=reference_backend)
    operator = CoherenceOperator(matrix, backend=backend)

    np.testing.assert_allclose(
        operator.matrix,
        reference.matrix,
        rtol=structural_tolerances["rtol"],
        atol=structural_tolerances["atol"],
    )
    np.testing.assert_allclose(
        operator.eigenvalues,
        reference.eigenvalues,
        rtol=structural_tolerances["rtol"],
        atol=structural_tolerances["atol"],
    )
    assert (
        pytest.approx(
            reference.c_min,
            rel=structural_tolerances["rtol"],
            abs=structural_tolerances["atol"],
        )
        == operator.c_min
    )

    expectation_backend = operator.expectation(state)
    expectation_reference = reference.expectation(state)
    assert expectation_backend == pytest.approx(
        expectation_reference,
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
def test_mathematical_dynamics_matches_numpy(
    backend_name: str, structural_tolerances: dict[str, float]
) -> None:
    """Unitary trajectories should be backend agnostic within tolerance."""

    backend = _require_backend(backend_name)
    reference_backend = get_backend("numpy")

    hilbert = HilbertSpace(dimension=2)
    generator = np.array([[1.0, 0.25 - 0.15j], [0.25 + 0.15j, -0.5]], dtype=np.complex128)
    state = np.array([0.8 + 0.1j, 0.3 - 0.2j], dtype=np.complex128)

    reference_engine = MathematicalDynamicsEngine(
        generator,
        hilbert,
        backend=reference_backend,
        use_scipy=False,
    )
    engine = MathematicalDynamicsEngine(
        generator,
        hilbert,
        backend=backend,
        use_scipy=False,
    )

    trajectory_reference = reference_engine.evolve(state, steps=3, dt=0.05)
    trajectory_backend = engine.evolve(state, steps=3, dt=0.05)

    np.testing.assert_allclose(
        _to_numpy(trajectory_backend, backend=backend),
        trajectory_reference,
        rtol=structural_tolerances["rtol"],
        atol=structural_tolerances["atol"],
    )


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
def test_contractive_dynamics_matches_numpy(
    backend_name: str, structural_tolerances: dict[str, float]
) -> None:
    """Contractive trajectories should remain invariant across backends."""

    backend = _require_backend(backend_name)
    reference_backend = get_backend("numpy")

    hilbert = HilbertSpace(dimension=2)
    lindblad_generator = -0.2 * np.eye(4, dtype=np.complex128)
    density = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=np.complex128)

    reference_engine = ContractiveDynamicsEngine(
        lindblad_generator,
        hilbert,
        backend=reference_backend,
        use_scipy=False,
    )
    engine = ContractiveDynamicsEngine(
        lindblad_generator,
        hilbert,
        backend=backend,
        use_scipy=False,
    )

    evolved_reference = reference_engine.step(density, dt=0.1)
    evolved_backend = engine.step(density, dt=0.1)

    np.testing.assert_allclose(
        _to_numpy(evolved_backend, backend=backend),
        evolved_reference,
        rtol=structural_tolerances["rtol"],
        atol=structural_tolerances["atol"],
    )
    assert engine.last_contractivity_gap == pytest.approx(
        reference_engine.last_contractivity_gap,
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )


def test_torch_backend_handles_numpy_complex_dtype() -> None:
    """Torch backend must convert NumPy dtypes into ``torch.dtype`` instances."""

    backend = _require_backend("torch")

    if not hasattr(backend, "_torch"):
        pytest.skip("Torch backend unavailable for dtype inspection")

    matrix = np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex128)

    tensor = ensure_array(matrix, dtype=np.complex128, backend=backend)

    assert tensor.dtype == backend._torch.complex128
