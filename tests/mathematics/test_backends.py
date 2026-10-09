"""Cross-backend numerical consistency checks."""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

import tnfr.mathematics.backend as backend_registry
from tnfr.errors import TNFRValueError
from tnfr.mathematics import CoherenceOperator, HilbertSpace
from tnfr.mathematics.backend import (
    BackendUnavailableError,
    MathematicsBackend,
    ensure_array,
    ensure_numpy,
    get_backend,
)
from tnfr.mathematics.dynamics import (
    ContractiveDynamicsEngine,
    MathematicalDynamicsEngine,
)

_BACKEND_NAMES = ("numpy", "jax", "torch")


def test_numpy_matrix_exponential_retains_nilpotent_part():
    # A^2=0, so exp(A)=I+A; a diagonalization formula loses its Jordan part.
    matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        get_backend("numpy").matrix_exp(matrix), np.eye(2) + matrix, atol=1e-15
    )


def test_numpy_matrix_exponential_requires_scipy_instead_of_false_diagonalization():
    backend = backend_registry._NumpyBackend(np, None)
    matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    before = matrix.copy()

    with pytest.raises(BackendUnavailableError, match="SciPy.*matrix exponential"):
        backend.matrix_exp(matrix)

    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_array_equal(backend.matmul(matrix, np.eye(2)), matrix)


@pytest.fixture
def isolated_backend_registry(monkeypatch):
    """Keep registration tests independent of cached optional adapters."""
    monkeypatch.setattr(backend_registry, "_BACKEND_FACTORIES", {})
    monkeypatch.setattr(backend_registry, "_BACKEND_ALIASES", {})
    monkeypatch.setattr(backend_registry, "_BACKEND_CACHE", {})
    return backend_registry


def _registry_state(registry):
    return tuple(
        dict(getattr(registry, name))
        for name in ("_BACKEND_FACTORIES", "_BACKEND_ALIASES", "_BACKEND_CACHE")
    )


def test_backend_override_refreshes_cached_instance_and_preserves_aliases(
    isolated_backend_registry,
):
    registry = isolated_backend_registry
    old = backend_registry._NumpyBackend(np, None)
    replacement = backend_registry._NumpyBackend(np, None)
    unrelated = backend_registry._NumpyBackend(np, None)
    registry.register_backend("sample", lambda: old, aliases=("short",))
    registry.register_backend("other", lambda: unrelated)
    assert registry.get_backend("short") is old
    assert registry.get_backend("other") is unrelated

    registry.register_backend("sample", lambda: replacement, override=True)

    assert registry.get_backend("sample") is replacement
    assert registry.get_backend("short") is replacement
    assert registry.get_backend("other") is unrelated


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("collision", ["canonical_as_alias", "alias_as_canonical"])
def test_backend_names_and_aliases_cannot_shadow_each_other(
    isolated_backend_registry, override, collision
):
    registry = isolated_backend_registry
    original = backend_registry._NumpyBackend(np, None)
    registry.register_backend("numpy", lambda: original, aliases=("np",))
    assert registry.get_backend("np") is original
    before = _registry_state(registry)

    with pytest.raises(TNFRValueError):
        if collision == "canonical_as_alias":
            registry.register_backend(
                "new", lambda: original, aliases=(" NumPy ",), override=override
            )
        else:
            registry.register_backend(" NP ", lambda: original, override=override)

    assert _registry_state(registry) == before
    assert registry.get_backend("numpy") is original


@pytest.mark.parametrize("failure", ["existing_alias", "duplicate_alias", "iterator"])
def test_failed_backend_registration_leaves_all_state_unchanged(
    isolated_backend_registry, failure
):
    registry = isolated_backend_registry
    original = backend_registry._NumpyBackend(np, None)
    registry.register_backend("original", lambda: original, aliases=("used",))
    assert registry.get_backend("used") is original
    before = _registry_state(registry)

    def aliases():
        yield "new_alias"
        if failure == "iterator":
            raise RuntimeError("alias source failed")
        yield "used" if failure == "existing_alias" else " NEW_ALIAS "

    expected_error = RuntimeError if failure == "iterator" else TNFRValueError
    with pytest.raises(expected_error):
        registry.register_backend("new", lambda: original, aliases=aliases())

    assert _registry_state(registry) == before
    with pytest.raises(LookupError):
        registry.get_backend("new_alias")


def test_backend_alias_override_rebinds_without_reconstructing_other_backends(
    isolated_backend_registry,
):
    registry = isolated_backend_registry
    first = backend_registry._NumpyBackend(np, None)
    second = backend_registry._NumpyBackend(np, None)
    registry.register_backend("first", lambda: first, aliases=("shared", "retained"))
    assert registry.get_backend("shared") is first

    registry.register_backend(
        "second", lambda: second, aliases=("shared",), override=True
    )

    assert registry.get_backend("shared") is second
    assert registry.get_backend("retained") is first
    assert registry.get_backend("first") is first


@pytest.mark.parametrize("bad", ["", " \t ", "auto", " AuTo "])
@pytest.mark.parametrize("identifier", ["canonical", "alias"])
@pytest.mark.parametrize("override", [False, True])
def test_unreachable_backend_identifiers_are_rejected_without_mutation(
    isolated_backend_registry, bad, identifier, override
):
    registry = isolated_backend_registry
    original = backend_registry._NumpyBackend(np, None)
    replacement = backend_registry._NumpyBackend(np, None)
    registry.register_backend("original", lambda: original, aliases=("retained",))
    assert registry.get_backend("original") is original
    before = _registry_state(registry)

    with pytest.raises(TNFRValueError, match="nonempty.*auto"):
        if identifier == "canonical":
            registry.register_backend(bad, lambda: replacement, override=override)
        else:
            registry.register_backend(
                "original" if override else "new",
                lambda: replacement,
                aliases=("new_alias", bad),
                override=override,
            )

    assert _registry_state(registry) == before
    assert registry.get_backend("retained") is original


def _require_backend(name: str) -> MathematicsBackend:
    backend = get_backend(name)
    if backend.name != name:
        pytest.skip(f"Backend '{name}' is unavailable; installed: {backend.name!r}.")
    return cast(MathematicsBackend, backend)


def _adjust_tolerances_for_backend(
    backend_name: str, tolerances: dict[str, float]
) -> dict[str, float]:
    """Relax tolerances for backends that might default to 32-bit precision."""
    if backend_name == "jax":
        try:
            import jax.numpy as jnp

            if jnp.array([1.0]).dtype == jnp.float32:
                return {"rtol": 1e-5, "atol": 1e-6}
        except ImportError:
            pass
    return tolerances


def _to_numpy(value: object, *, backend: MathematicsBackend) -> np.ndarray:
    return np.asarray(ensure_numpy(value, backend=backend))


@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
def test_coherence_operator_matches_numpy(
    backend_name: str, structural_tolerances: dict[str, float]
) -> None:
    """Coherence operators must agree across available numerical backends."""

    backend = _require_backend(backend_name)
    structural_tolerances = _adjust_tolerances_for_backend(
        backend_name, structural_tolerances
    )
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

    expectation_backend = operator.expectation(
        state, atol=structural_tolerances["atol"]
    )
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
    structural_tolerances = _adjust_tolerances_for_backend(
        backend_name, structural_tolerances
    )
    reference_backend = get_backend("numpy")

    hilbert = HilbertSpace(dimension=2)
    generator = np.array(
        [[1.0, 0.25 - 0.15j], [0.25 + 0.15j, -0.5]], dtype=np.complex128
    )
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
    structural_tolerances = _adjust_tolerances_for_backend(
        backend_name, structural_tolerances
    )
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

    # Use getattr to avoid mypy errors since _torch is not in the interface
    torch_module = getattr(backend, "_torch", None)
    if torch_module is None:
        pytest.skip("Torch backend unavailable for dtype inspection")

    assert torch_module is not None

    matrix = np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex128)

    tensor = ensure_array(matrix, dtype=np.complex128, backend=backend)

    assert tensor.dtype == torch_module.complex128


@pytest.fixture
def cpu_torch_backend():
    torch = pytest.importorskip("torch")
    return backend_registry._TorchBackend(
        torch, torch.linalg, torch.device("cpu"), False
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("view", [False, True])
def test_torch_readonly_numpy_conversion_owns_storage(cpu_torch_backend, dtype, view):
    storage = np.arange(8, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        storage += 2j
    source = storage[::2] if view else storage
    source.setflags(write=False)
    before = storage.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        tensor = cpu_torch_backend.as_array(source, dtype=dtype)

    np.testing.assert_array_equal(tensor.numpy(), source)
    assert not np.shares_memory(tensor.numpy(), storage)
    tensor[0] = -99
    np.testing.assert_array_equal(storage, before)
    assert not source.flags.writeable


def test_torch_writable_numpy_conversion_retains_alias(cpu_torch_backend):
    source = np.array([1.0, 2.0])
    tensor = cpu_torch_backend.as_array(source)

    assert np.shares_memory(tensor.numpy(), source)
    tensor[0] = 3.0
    np.testing.assert_array_equal(source, [3.0, 2.0])


def test_torch_native_conversion_preserves_identity_and_gradient(cpu_torch_backend):
    torch = cpu_torch_backend._torch
    source = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

    tensor = cpu_torch_backend.as_array(source)

    assert tensor is source
    (3 * tensor).sum().backward()
    np.testing.assert_array_equal(source.grad.numpy(), [3.0, 3.0])
