"""TNFR computation backends for high-performance ΔNFR and Si evaluation.

This module provides pluggable backend implementations that optimize the core
TNFR computational kernels (ΔNFR, Si) using different numerical libraries.
Each backend maintains TNFR semantic fidelity while leveraging library-specific
optimizations like JIT compilation or GPU acceleration.

The backend system ensures that the nodal equation ∂EPI/∂t = νf · ΔNFR(t) and
all structural invariants remain intact regardless of which backend executes
the computation.

Examples
--------
Use the NumPy backend explicitly:

>>> from tnfr.backends import get_backend
>>> backend = get_backend("numpy")
>>> backend.name
'numpy'

Select backend via environment variable:

```bash
export TNFR_BACKEND=jax
python your_simulation.py
```

Available backends
------------------
- **numpy**: Vectorized NumPy implementation (default, stable)
- **jax**: JIT-compiled JAX with autodiff support (experimental)
- **torch**: PyTorch GPU-accelerated implementation (experimental)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Mapping, MutableMapping

from ..types import TNFRGraph
from ..utils import get_logger

__all__ = [
    "TNFRBackend",
    "get_backend",
    "set_backend",
    "available_backends",
]

logger = get_logger(__name__)


class TNFRBackend(ABC):
    """Base class for TNFR computation backends.

    All backends must implement the core computational methods while
    preserving TNFR structural semantics and the canonical nodal equation.

    Structural Invariants
    ---------------------
    1. ΔNFR semantics: sign and magnitude must modulate reorganization rate
    2. Phase verification: coupling requires explicit phase synchrony check
    3. Operator closure: all transformations map to valid TNFR states
    4. Determinism: computations must be reproducible with fixed seeds
    5. Si stability: sense index must correlate with network coherence

    Attributes
    ----------
    name : str
        Backend identifier (e.g., "numpy", "jax", "torch")
    supports_gpu : bool
        Whether this backend can utilize GPU acceleration
    supports_jit : bool
        Whether this backend supports JIT compilation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend identifier."""
        ...

    @property
    def supports_gpu(self) -> bool:
        """Return True if this backend can use GPU acceleration."""
        return False

    @property
    def supports_jit(self) -> bool:
        """Return True if this backend supports JIT compilation."""
        return False

    @abstractmethod
    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        """Compute ΔNFR for all nodes in the graph.

        This method must preserve the canonical ΔNFR computation semantics:
        - Weighted combination of phase, EPI, νf, and topology gradients
        - Proper phase dispersion calculation via neighbor phase means
        - Coherent handling of isolated nodes (ΔNFR = 0)

        Parameters
        ----------
        graph : TNFRGraph
            Graph with node attributes (phase, EPI, νf) to compute ΔNFR
        cache_size : int or None, optional
            Maximum cached state entries. None means unlimited.
        n_jobs : int or None, optional
            Parallelism hint for backends that support it
        profile : MutableMapping[str, float] or None, optional
            Dict to accumulate timing metrics for profiling

        Notes
        -----
        The computation must write ΔNFR values back to nodes using the
        appropriate alias and maintain consistency with graph metadata.
        """
        ...

    @abstractmethod
    def compute_si(
        self,
        graph: TNFRGraph,
        *,
        inplace: bool = True,
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> dict[Any, float] | Any:
        """Compute the sense index (Si) for all nodes.

        Si blends structural frequency (νf), phase alignment, and ΔNFR
        attenuation according to the weights configured in graph metadata.

        Parameters
        ----------
        graph : TNFRGraph
            Graph with node attributes (νf, ΔNFR, phase)
        inplace : bool, default=True
            Whether to write Si values back to nodes
        n_jobs : int or None, optional
            Parallelism hint for backends that support it
        chunk_size : int or None, optional
            Batch size for chunked processing
        profile : MutableMapping[str, Any] or None, optional
            Dict to accumulate timing and execution path metrics

        Returns
        -------
        dict[Any, float] or numpy.ndarray
            Node-to-Si mapping or array of Si values

        Notes
        -----
        The Si computation must respect the structural sensitivity weights
        (alpha, beta, gamma) configured in the graph's SI_WEIGHTS metadata.
        """
        ...


# Backend registry
_BACKEND_REGISTRY: MutableMapping[str, type[TNFRBackend]] = {}
_BACKEND_CACHE: MutableMapping[str, TNFRBackend] = {}
_DEFAULT_BACKEND: str = "numpy"
_CURRENT_BACKEND: str | None = None


def register_backend(name: str, backend_class: type[TNFRBackend]) -> None:
    """Register a TNFR backend implementation.

    Parameters
    ----------
    name : str
        Backend identifier (will be normalized to lowercase)
    backend_class : type[TNFRBackend]
        Backend class implementing the TNFRBackend interface

    Raises
    ------
    ValueError
        If name is already registered
    TypeError
        If backend_class doesn't implement TNFRBackend
    """
    name_lower = name.lower().strip()
    if not name_lower:
        raise ValueError("Backend name cannot be empty")

    if name_lower in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' is already registered")

    if not issubclass(backend_class, TNFRBackend):
        raise TypeError(f"Backend class must inherit from TNFRBackend, got {backend_class}")

    _BACKEND_REGISTRY[name_lower] = backend_class
    logger.debug("Registered TNFR backend: %s", name)


def get_backend(name: str | None = None) -> TNFRBackend:
    """Get a TNFR backend instance by name.

    Resolution order:
    1. Explicit `name` parameter
    2. Previously set backend via set_backend()
    3. TNFR_BACKEND environment variable
    4. Default backend ("numpy")

    Parameters
    ----------
    name : str or None, optional
        Backend name to retrieve. If None, uses resolution order above.

    Returns
    -------
    TNFRBackend
        Backend instance ready for computation

    Raises
    ------
    ValueError
        If the requested backend is not registered
    RuntimeError
        If backend initialization fails

    Examples
    --------
    >>> backend = get_backend("numpy")
    >>> backend.name
    'numpy'

    >>> import os
    >>> os.environ["TNFR_BACKEND"] = "numpy"
    >>> backend = get_backend()
    >>> backend.name
    'numpy'
    """
    # Resolve backend name
    if name is None:
        if _CURRENT_BACKEND is not None:
            name = _CURRENT_BACKEND
        else:
            name = os.environ.get("TNFR_BACKEND", _DEFAULT_BACKEND)

    name_lower = name.lower().strip()

    # Return cached instance if available
    if name_lower in _BACKEND_CACHE:
        return _BACKEND_CACHE[name_lower]

    # Get backend class from registry
    if name_lower not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available backends: {available}")

    # Instantiate backend
    backend_class = _BACKEND_REGISTRY[name_lower]
    try:
        backend = backend_class()
        _BACKEND_CACHE[name_lower] = backend
        logger.info("Initialized TNFR backend: %s", name_lower)
        return backend
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize backend '{name}': {exc}") from exc


def set_backend(name: str) -> None:
    """Set the default TNFR backend for subsequent operations.

    Parameters
    ----------
    name : str
        Backend name to set as default

    Raises
    ------
    ValueError
        If the backend name is not registered

    Examples
    --------
    >>> set_backend("numpy")
    >>> get_backend().name
    'numpy'
    """
    global _CURRENT_BACKEND

    name_lower = name.lower().strip()
    if name_lower not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available backends: {available}")

    _CURRENT_BACKEND = name_lower
    logger.info("Set default TNFR backend to: %s", name_lower)


def available_backends() -> Mapping[str, type[TNFRBackend]]:
    """Return mapping of registered backend names to their classes.

    Returns
    -------
    Mapping[str, type[TNFRBackend]]
        Read-only view of registered backends

    Examples
    --------
    >>> backends = available_backends()
    >>> "numpy" in backends
    True
    """
    return dict(_BACKEND_REGISTRY)


# Import and register backends
# This is done at module level to ensure backends are available immediately
try:
    from . import numpy_backend

    register_backend("numpy", numpy_backend.NumPyBackend)
except ImportError as exc:
    logger.warning("NumPy backend unavailable: %s", exc)

try:
    from . import optimized_numpy

    register_backend("optimized_numpy", optimized_numpy.OptimizedNumPyBackend)
    register_backend("optimized", optimized_numpy.OptimizedNumPyBackend)
except ImportError as exc:
    logger.debug("Optimized NumPy backend not available: %s", exc)

try:
    from . import jax_backend

    register_backend("jax", jax_backend.JAXBackend)
except ImportError as exc:
    logger.debug("JAX backend not available (optional dependency): %s", exc)

try:
    from . import torch_backend

    register_backend("torch", torch_backend.TorchBackend)
except ImportError as exc:
    logger.debug("PyTorch backend not available (optional dependency): %s", exc)
