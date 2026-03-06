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

If none of these provide a value we auto-select the first available backend in
the GPU-preferential order (JAX → PyTorch → NumPy).  Optional backends are
registered lazily so downstream environments without JAX or PyTorch remain
functional while still benefiting from acceleration when present.
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
    cast,
)

from ..errors import TNFRValueError

from ..utils import cached_import, get_logger
from ..core.exceptions import BackendUnavailableError

logger = get_logger(__name__)

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

    def is_gpu_available(self) -> bool:
        """Return True if the backend is currently using a GPU."""

    def get_device_name(self) -> str:
        """Return the name of the device being used (e.g. 'cpu', 'cuda:0')."""

    def get_backend_info(self) -> Mapping[str, Any]:
        """Return detailed backend information."""

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

    def is_gpu_available(self) -> bool:
        return False

    def get_device_name(self) -> str:
        return "cpu"

    def get_backend_info(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "version": self._np.__version__,
            "device": "cpu",
            "accelerated": False
        }

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

    def is_gpu_available(self) -> bool:
        try:
            # Check if any device is a GPU or TPU
            return any(d.platform in ("gpu", "tpu") for d in self._jax.devices())
        except Exception:
            return False

    def get_device_name(self) -> str:
        try:
            return str(self._jax.devices()[0])
        except Exception:
            return "unknown"

    def get_backend_info(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "version": self._jax.__version__,
            "device": self.get_device_name(),
            "accelerated": self.is_gpu_available()
        }

@dataclass(slots=True)
class _TorchBackend:
    """PyTorch backed implementation with CUDA support."""

    _torch: Any
    _torch_linalg: Any
    _device: Any  # torch.device for CUDA/CPU placement
    _use_cuda: bool  # Whether CUDA is available and enabled

    name: ClassVar[str] = "torch"
    supports_autodiff: ClassVar[bool] = True

    def as_array(self, value: Any, *, dtype: Any | None = None) -> Any:
        tensor = self._torch.as_tensor(value, device=self._device)
        if dtype is None:
            return tensor

        target_dtype = self._normalise_dtype(dtype)
        if target_dtype is None:
            return tensor.to(dtype=dtype, device=self._device)

        if tensor.dtype == target_dtype:
            return tensor.to(device=self._device)

        return tensor.to(dtype=target_dtype, device=self._device)

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

    def is_gpu_available(self) -> bool:
        return self._use_cuda

    def get_device_name(self) -> str:
        return str(self._device)

    def get_backend_info(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "version": self._torch.__version__,
            "device": self.get_device_name(),
            "accelerated": self.is_gpu_available(),
            "details": self.get_device_info()
        }

    def get_device_info(self) -> dict[str, Any]:
        """Get CUDA device information."""
        info = {
            "device": str(self._device),
            "use_cuda": self._use_cuda,
            "cuda_available": self._torch.cuda.is_available() if hasattr(self._torch, 'cuda') else False
        }
        
        if self._use_cuda and hasattr(self._torch, 'cuda'):
            info.update({
                "device_count": self._torch.cuda.device_count(),
                "current_device": self._torch.cuda.current_device(),
                "device_name": self._torch.cuda.get_device_name(self._torch.cuda.current_device()),
                "memory_allocated": self._torch.cuda.memory_allocated(),
                "memory_reserved": self._torch.cuda.memory_reserved(),
            })
        
        return info

    def to_cuda(self, value: Any) -> Any:
        """Move tensor to CUDA device if available."""
        if self._use_cuda and hasattr(value, "to"):
            return value.to(self._device)
        return value

    def to_cpu(self, value: Any) -> Any:
        """Move tensor to CPU device."""
        if hasattr(value, "cpu"):
            return value.cpu()
        return value

def _normalise_name(name: str) -> str:
    return name.strip().lower()

_BACKEND_FACTORIES: MutableMapping[str, BackendFactory] = {}
_BACKEND_ALIASES: MutableMapping[str, str] = {}
_BACKEND_CACHE: MutableMapping[str, MathematicsBackend] = {}

_AUTO_BACKEND_SENTINEL = "auto"
_AUTO_BACKEND_PRIORITY = ("jax", "torch", "numpy")

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
        raise TNFRValueError(
            f"Backend '{name}' already registered",
            context={"name": name, "existing": list(_BACKEND_FACTORIES.keys())},
            suggestion="Use override=True to replace the existing backend registration."
        )
    _BACKEND_FACTORIES[key] = factory
    if aliases:
        for alias in aliases:
            alias_key = _normalise_name(alias)
            if not override and alias_key in _BACKEND_ALIASES:
                raise TNFRValueError(
                    f"Backend alias '{alias}' already registered",
                    context={"alias": alias, "existing_aliases": list(_BACKEND_ALIASES.keys())},
                    suggestion="Use override=True or choose a different alias."
                )
            _BACKEND_ALIASES[alias_key] = key

def _resolve_backend_name(name: str | None) -> str:
    if name:
        normalised = _normalise_name(name)
        return _AUTO_BACKEND_SENTINEL if normalised == _AUTO_BACKEND_SENTINEL else normalised

    env_choice = os.getenv("TNFR_MATH_BACKEND")
    if env_choice:
        normalised = _normalise_name(env_choice)
        return _AUTO_BACKEND_SENTINEL if normalised == _AUTO_BACKEND_SENTINEL else normalised

    backend_from_config: str | None = None
    try:
        from ..config import get_config  # Use unified TNFR configuration

        config = get_config()
        backend_from_config = config.math_backend
    except Exception:  # pragma: no cover - defensive; config must not break selection
        backend_from_config = None

    if backend_from_config and backend_from_config != "auto":
        return _normalise_name(backend_from_config)

    return _AUTO_BACKEND_SENTINEL

def _resolve_factory(name: str) -> BackendFactory:
    canonical = _BACKEND_ALIASES.get(name, name)
    try:
        return _BACKEND_FACTORIES[canonical]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise LookupError(f"Unknown mathematics backend: {name}") from exc

def _construct_backend(name: str) -> MathematicsBackend | None:
    canonical = _BACKEND_ALIASES.get(name, name)
    if canonical in _BACKEND_CACHE:
        return _BACKEND_CACHE[canonical]

    factory = _resolve_factory(canonical)
    try:
        backend = factory()
    except BackendUnavailableError as exc:
        logger.warning("Backend '%s' unavailable: %s", canonical, exc)
        return None

    _BACKEND_CACHE[canonical] = backend
    return backend

def get_backend(name: str | None = None) -> MathematicsBackend:
    """Return a backend instance using the configured resolution order."""

    resolved_name = _resolve_backend_name(name)
    if resolved_name == _AUTO_BACKEND_SENTINEL:
        # First pass: Look for GPU-accelerated backend
        for candidate in _AUTO_BACKEND_PRIORITY:
            backend = _construct_backend(candidate)
            if backend is not None and backend.is_gpu_available():
                logger.info("Auto-selected GPU-accelerated backend '%s' (%s)", candidate, backend.get_device_name())
                return backend

        # Second pass: Fallback to any available backend
        for candidate in _AUTO_BACKEND_PRIORITY:
            backend = _construct_backend(candidate)
            if backend is not None:
                if candidate != "numpy":
                    logger.info("Auto-selected CPU backend '%s'", candidate)
                return backend
        raise BackendUnavailableError(
            "No mathematical backend available; tried: " + ", ".join(_AUTO_BACKEND_PRIORITY)
        )

    backend = _construct_backend(resolved_name)
    if backend is not None:
        return backend

    if resolved_name != "numpy":
        logger.warning("Falling back to NumPy backend")
        fallback = _construct_backend("numpy")
        if fallback is not None:
            return fallback

    raise BackendUnavailableError(f"Unable to initialise mathematics backend '{resolved_name}'")

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
    backend = _NumpyBackend(np_module, scipy_linalg)  # type: ignore[call-arg]
    return cast(MathematicsBackend, backend)

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
    backend = _JaxBackend(jnp_module, jax_scipy, jax_module)  # type: ignore[call-arg]
    return cast(MathematicsBackend, backend)

def _make_torch_backend() -> MathematicsBackend:
    torch_module = cached_import("torch")
    if torch_module is None:
        raise BackendUnavailableError("PyTorch is not installed")
    torch_linalg = cached_import("torch.linalg")
    if torch_linalg is None:
        raise BackendUnavailableError("torch.linalg is required for linear algebra operations")
    
    # CUDA device detection and configuration
    use_cuda = False
    device = torch_module.device("cpu")
    
    if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
        # Check environment variable for CUDA preference
        cuda_enabled = os.getenv("TNFR_CUDA_ENABLED", "true").lower() in ("true", "1", "yes")
        if cuda_enabled:
            try:
                device = torch_module.device("cuda")
                use_cuda = True
                logger.info(f"CUDA enabled: {torch_module.cuda.get_device_name()} with {torch_module.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            except Exception as e:
                logger.warning(f"CUDA available but failed to initialize: {e}. Falling back to CPU.")
                device = torch_module.device("cpu")
        else:
            logger.info("CUDA available but disabled via TNFR_CUDA_ENABLED=false")
    else:
        logger.info("CUDA not available, using CPU backend")
    
    backend = _TorchBackend(torch_module, torch_linalg, device, use_cuda)  # type: ignore[call-arg]
    return cast(MathematicsBackend, backend)

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
