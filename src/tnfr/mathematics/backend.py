"""Backend abstraction for TNFR mathematical kernels.

This module introduces a unified interface that maps core linear algebra
operations to concrete numerical libraries.  Keeping this layer small and
canonical guarantees we can switch implementations without diluting the
structural semantics required by TNFR (coherence, phase, νf, ΔNFR, etc.).

The canonical entry point is :func:`get_backend`, which honours three lookup
mechanisms in order of precedence:

1. Explicit ``name`` argument.
2. ``TNFR_MATH_BACKEND`` environment variable.
3. ``tnfr.config.get_flags().math_backend``.

If none of these provide a value we default to the NumPy backend.  Optional
backends are registered lazily so downstream environments without JAX or
PyTorch remain functional.
"""

from __future__ import annotations

from ..compat.dataclass import dataclass
import os
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    runtime_checkable,
)

from ..utils import cached_import, get_logger

logger = get_logger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when a registered backend cannot be constructed."""


@runtime_checkable
class MathematicsBackend(Protocol):
    """Structural numerical backend interface.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    backend implementations conform to the expected mathematical operations interface.
    """

    name: str
    supports_autodiff: bool

    def as_array(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Convert ``value`` into a backend-native dense array."""

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        """Return eigenvalues and eigenvectors for a general matrix."""

    def eigh(self, matrix: Any) -> tuple[Any, Any]:
        """Return eigenpairs for a Hermitian/symmetric matrix."""

    def matrix_exp(self, matrix: Any) -> Any:
        """Compute the matrix exponential of ``matrix``."""

    def norm(self, value: Any, *, ord: Any | None = None, axis: Any | None = None) -> Any:
        """Return the matrix or vector norm according to ``ord``."""

    def einsum(self, pattern: str, *operands: Any, **kwargs: Any) -> Any:
        """Evaluate an Einstein summation expression."""

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication that respects backend broadcasting rules."""

    def conjugate_transpose(self, matrix: Any) -> Any:
        """Hermitian conjugate of ``matrix`` († operator)."""

    def stack(self, arrays: Iterable[Any], *, axis: int = 0) -> Any:
        """Stack arrays along a new ``axis``."""

    def to_numpy(self, value: Any) -> Any:
        """Convert ``value`` to a ``numpy.ndarray`` when possible."""


BackendFactory = Callable[[], MathematicsBackend]


@dataclass(slots=True)
class _NumpyBackend:
    """NumPy backed implementation."""

    _np: Any
    _scipy_linalg: Any | None

    name: ClassVar[str] = "numpy"
    supports_autodiff: ClassVar[bool] = False

    def as_array(self, value: Any, *, dtype: Any | None = None) -> Any:
        return self._np.asarray(value, dtype=dtype)

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        return self._np.linalg.eig(matrix)

    def eigh(self, matrix: Any) -> tuple[Any, Any]:
        return self._np.linalg.eigh(matrix)

    def matrix_exp(self, matrix: Any) -> Any:
        if self._scipy_linalg is not None:
            return self._scipy_linalg.expm(matrix)
        eigvals, eigvecs = self._np.linalg.eig(matrix)
        inv = self._np.linalg.inv(eigvecs)
        exp_vals = self._np.exp(eigvals)
        return eigvecs @ self._np.diag(exp_vals) @ inv

    def norm(self, value: Any, *, ord: Any | None = None, axis: Any | None = None) -> Any:
        return self._np.linalg.norm(value, ord=ord, axis=axis)

    def einsum(self, pattern: str, *operands: Any, **kwargs: Any) -> Any:
        return self._np.einsum(pattern, *operands, **kwargs)

    def matmul(self, a: Any, b: Any) -> Any:
        return self._np.matmul(a, b)

    def conjugate_transpose(self, matrix: Any) -> Any:
        return self._np.conjugate(matrix).T

    def stack(self, arrays: Iterable[Any], *, axis: int = 0) -> Any:
        return self._np.stack(tuple(arrays), axis=axis)

    def to_numpy(self, value: Any) -> Any:
        return self._np.asarray(value)


@dataclass(slots=True)
class _JaxBackend:
    """JAX backed implementation."""

    _jnp: Any
    _jax_linalg: Any
    _jax: Any

    name: ClassVar[str] = "jax"
    supports_autodiff: ClassVar[bool] = True

    def as_array(self, value: Any, *, dtype: Any | None = None) -> Any:
        return self._jnp.asarray(value, dtype=dtype)

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        return self._jnp.linalg.eig(matrix)

    def eigh(self, matrix: Any) -> tuple[Any, Any]:
        return self._jnp.linalg.eigh(matrix)

    def matrix_exp(self, matrix: Any) -> Any:
        return self._jax_linalg.expm(matrix)

    def norm(self, value: Any, *, ord: Any | None = None, axis: Any | None = None) -> Any:
        return self._jnp.linalg.norm(value, ord=ord, axis=axis)

    def einsum(self, pattern: str, *operands: Any, **kwargs: Any) -> Any:
        return self._jnp.einsum(pattern, *operands, **kwargs)

    def matmul(self, a: Any, b: Any) -> Any:
        return self._jnp.matmul(a, b)

    def conjugate_transpose(self, matrix: Any) -> Any:
        return self._jnp.conjugate(matrix).T

    def stack(self, arrays: Iterable[Any], *, axis: int = 0) -> Any:
        return self._jnp.stack(tuple(arrays), axis=axis)

    def to_numpy(self, value: Any) -> Any:
        np_mod = cached_import("numpy")
        if np_mod is None:
            raise BackendUnavailableError("NumPy is required to export JAX arrays")
        return np_mod.asarray(self._jax.device_get(value))


@dataclass(slots=True)
class _TorchBackend:
    """PyTorch backed implementation."""

    _torch: Any
    _torch_linalg: Any

    name: ClassVar[str] = "torch"
    supports_autodiff: ClassVar[bool] = True

    def as_array(self, value: Any, *, dtype: Any | None = None) -> Any:
        tensor = self._torch.as_tensor(value)
        if dtype is None:
            return tensor

        target_dtype = self._normalise_dtype(dtype)
        if target_dtype is None:
            return tensor.to(dtype=dtype)

        if tensor.dtype == target_dtype:
            return tensor

        return tensor.to(dtype=target_dtype)

    def _normalise_dtype(self, dtype: Any) -> Any | None:
        """Return a ``torch.dtype`` equivalent for ``dtype`` when available."""

        if isinstance(dtype, self._torch.dtype):
            return dtype

        np_mod = cached_import("numpy")
        if np_mod is None:
            return None

        try:
            np_dtype = np_mod.dtype(dtype)
        except TypeError:
            return None

        numpy_name = np_dtype.name
        numpy_to_torch = {
            "bool": self._torch.bool,
            "uint8": self._torch.uint8,
            "int8": self._torch.int8,
            "int16": self._torch.int16,
            "int32": self._torch.int32,
            "int64": self._torch.int64,
            "float16": self._torch.float16,
            "float32": self._torch.float32,
            "float64": self._torch.float64,
            "complex64": getattr(self._torch, "complex64", None),
            "complex128": getattr(self._torch, "complex128", None),
            "bfloat16": getattr(self._torch, "bfloat16", None),
        }

        torch_dtype = numpy_to_torch.get(numpy_name)
        return torch_dtype

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        eigenvalues, eigenvectors = self._torch.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    def eigh(self, matrix: Any) -> tuple[Any, Any]:
        eigenvalues, eigenvectors = self._torch.linalg.eigh(matrix)
        return eigenvalues, eigenvectors

    def matrix_exp(self, matrix: Any) -> Any:
        return self._torch_linalg.matrix_exp(matrix)

    def norm(self, value: Any, *, ord: Any | None = None, axis: Any | None = None) -> Any:
        if axis is None:
            return self._torch.linalg.norm(value, ord=ord)
        return self._torch.linalg.norm(value, ord=ord, dim=axis)

    def einsum(self, pattern: str, *operands: Any, **kwargs: Any) -> Any:
        return self._torch.einsum(pattern, *operands, **kwargs)

    def matmul(self, a: Any, b: Any) -> Any:
        return self._torch.matmul(a, b)

    def conjugate_transpose(self, matrix: Any) -> Any:
        return matrix.mH if hasattr(matrix, "mH") else matrix.conj().transpose(-2, -1)

    def stack(self, arrays: Iterable[Any], *, axis: int = 0) -> Any:
        return self._torch.stack(tuple(arrays), dim=axis)

    def to_numpy(self, value: Any) -> Any:
        np_mod = cached_import("numpy")
        if np_mod is None:
            raise BackendUnavailableError("NumPy is required to export Torch tensors")
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np_mod.asarray(value)


def _normalise_name(name: str) -> str:
    return name.strip().lower()


_BACKEND_FACTORIES: MutableMapping[str, BackendFactory] = {}
_BACKEND_ALIASES: MutableMapping[str, str] = {}
_BACKEND_CACHE: MutableMapping[str, MathematicsBackend] = {}


def ensure_array(
    value: Any,
    *,
    dtype: Any | None = None,
    backend: MathematicsBackend | None = None,
) -> Any:
    """Return ``value`` as a backend-native dense array."""

    resolved = backend or get_backend()
    return resolved.as_array(value, dtype=dtype)


def ensure_numpy(value: Any, *, backend: MathematicsBackend | None = None) -> Any:
    """Export ``value`` from the backend into :class:`numpy.ndarray`."""

    resolved = backend or get_backend()
    return resolved.to_numpy(value)


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    aliases: Iterable[str] | None = None,
    override: bool = False,
) -> None:
    """Register a backend factory under ``name``.

    Parameters
    ----------
    name:
        Canonical backend identifier.
    factory:
        Callable that returns a :class:`MathematicsBackend` instance.
    aliases:
        Optional alternative identifiers that will resolve to ``name``.
    override:
        When ``True`` replaces existing registrations.
    """

    key = _normalise_name(name)
    if not override and key in _BACKEND_FACTORIES:
        raise ValueError(f"Backend '{name}' already registered")
    _BACKEND_FACTORIES[key] = factory
    if aliases:
        for alias in aliases:
            alias_key = _normalise_name(alias)
            if not override and alias_key in _BACKEND_ALIASES:
                raise ValueError(f"Backend alias '{alias}' already registered")
            _BACKEND_ALIASES[alias_key] = key


def _resolve_backend_name(name: str | None) -> str:
    if name:
        return _normalise_name(name)

    env_choice = os.getenv("TNFR_MATH_BACKEND")
    if env_choice:
        return _normalise_name(env_choice)

    backend_from_flags: str | None = None
    try:
        from ..config import get_flags  # Local import avoids circular dependency

        backend_from_flags = getattr(get_flags(), "math_backend", None)
    except Exception:  # pragma: no cover - defensive; config must not break selection
        backend_from_flags = None

    if backend_from_flags:
        return _normalise_name(backend_from_flags)

    return "numpy"


def _resolve_factory(name: str) -> BackendFactory:
    canonical = _BACKEND_ALIASES.get(name, name)
    try:
        return _BACKEND_FACTORIES[canonical]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise LookupError(f"Unknown mathematics backend: {name}") from exc


def get_backend(name: str | None = None) -> MathematicsBackend:
    """Return a backend instance using the configured resolution order."""

    resolved_name = _resolve_backend_name(name)
    canonical = _BACKEND_ALIASES.get(resolved_name, resolved_name)
    if canonical in _BACKEND_CACHE:
        return _BACKEND_CACHE[canonical]

    factory = _resolve_factory(canonical)
    try:
        backend = factory()
    except BackendUnavailableError as exc:
        logger.warning("Backend '%s' unavailable: %s", canonical, exc)
        if canonical != "numpy":
            logger.warning("Falling back to NumPy backend")
            return get_backend("numpy")
        raise

    _BACKEND_CACHE[canonical] = backend
    return backend


def available_backends() -> Mapping[str, BackendFactory]:
    """Return the registered backend factories."""

    return dict(_BACKEND_FACTORIES)


def _make_numpy_backend() -> MathematicsBackend:
    np_module = cached_import("numpy")
    if np_module is None:
        raise BackendUnavailableError("NumPy is not installed")
    scipy_linalg = cached_import("scipy.linalg")
    if scipy_linalg is None:
        logger.debug("SciPy not available; falling back to eigen decomposition for expm")
    return _NumpyBackend(np_module, scipy_linalg)


def _make_jax_backend() -> MathematicsBackend:
    jnp_module = cached_import("jax.numpy")
    if jnp_module is None:
        raise BackendUnavailableError("jax.numpy is not available")
    jax_scipy = cached_import("jax.scipy.linalg")
    if jax_scipy is None:
        raise BackendUnavailableError("jax.scipy.linalg is required for matrix_exp")
    jax_module = cached_import("jax")
    if jax_module is None:
        raise BackendUnavailableError("jax core module is required")
    return _JaxBackend(jnp_module, jax_scipy, jax_module)


def _make_torch_backend() -> MathematicsBackend:
    torch_module = cached_import("torch")
    if torch_module is None:
        raise BackendUnavailableError("PyTorch is not installed")
    torch_linalg = cached_import("torch.linalg")
    if torch_linalg is None:
        raise BackendUnavailableError("torch.linalg is required for linear algebra operations")
    return _TorchBackend(torch_module, torch_linalg)


register_backend("numpy", _make_numpy_backend, aliases=("np",))
register_backend("jax", _make_jax_backend)
register_backend("torch", _make_torch_backend, aliases=("pytorch",))

__all__ = [
    "MathematicsBackend",
    "BackendUnavailableError",
    "register_backend",
    "get_backend",
    "available_backends",
    "ensure_array",
    "ensure_numpy",
]
