"""JAX-based JIT-compiled backend for TNFR computations (Experimental).

This module provides a JIT-compiled JAX implementation of TNFR computational
kernels. JAX enables:

- Just-in-time (JIT) compilation for optimized machine code
- Automatic differentiation for gradient-based analysis
- GPU acceleration for large-scale networks
- XLA compiler optimizations

**Status**: Experimental - API may change in future releases.

The JAX backend currently delegates to the NumPy implementation but provides
infrastructure for future JIT-optimized kernels.

Examples
--------
>>> from tnfr.backends import get_backend
>>> backend = get_backend("jax")  # doctest: +SKIP
>>> backend.supports_jit  # doctest: +SKIP
True
"""

from __future__ import annotations

from typing import Any, MutableMapping

from . import TNFRBackend
from ..types import TNFRGraph


class JAXBackend(TNFRBackend):
    """JIT-compiled JAX implementation of TNFR kernels (Experimental).

    This backend provides a foundation for JIT-optimized TNFR computations
    using JAX. Current implementation delegates to NumPy backend while
    maintaining interface compatibility for future JIT implementations.

    Future optimizations planned:
    - JIT-compiled ΔNFR computation with @jax.jit
    - Vectorized operations using jax.numpy
    - GPU acceleration via JAX device placement
    - Automatic differentiation for sensitivity analysis

    Attributes
    ----------
    name : str
        Returns "jax"
    supports_gpu : bool
        True (JAX supports GPU acceleration)
    supports_jit : bool
        True (JAX provides JIT compilation)

    Notes
    -----
    Requires JAX to be installed: `pip install jax jaxlib`

    For GPU support, install appropriate JAX GPU build for your platform.
    """

    def __init__(self) -> None:
        """Initialize JAX backend."""
        try:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp
        except ImportError as exc:
            raise RuntimeError(
                "JAX backend requires jax to be installed. " "Install with: pip install jax jaxlib"
            ) from exc

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "jax"

    @property
    def supports_gpu(self) -> bool:
        """JAX supports GPU acceleration."""
        return True

    @property
    def supports_jit(self) -> bool:
        """JAX supports JIT compilation."""
        return True

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        """Compute ΔNFR using JAX backend.

        **Current implementation**: Delegates to NumPy backend while maintaining
        interface compatibility.

        **Planned**: JIT-compiled vectorized computation using jax.numpy with
        automatic XLA optimization and optional GPU acceleration.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        cache_size : int or None, optional
            Cache size hint (currently passed to NumPy backend)
        n_jobs : int or None, optional
            Ignored (JAX uses vectorization instead of multiprocessing)
        profile : MutableMapping[str, float] or None, optional
            Dict to collect timing metrics
        """
        # JAX implementation planned for v2.0 - high-performance JIT compilation
        # Currently delegates to NumPy backend for compatibility
        from ..dynamics.dnfr import default_compute_delta_nfr

        default_compute_delta_nfr(
            graph,
            cache_size=cache_size,
            n_jobs=n_jobs,
            profile=profile,
        )

    def compute_si(
        self,
        graph: TNFRGraph,
        *,
        inplace: bool = True,
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> dict[Any, float] | Any:
        """Compute sense index using JAX backend.

        **Current implementation**: Delegates to NumPy backend while maintaining
        interface compatibility.

        **Planned**: JIT-compiled vectorized Si computation using jax.numpy with
        optimized phase dispersion and normalization kernels.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        inplace : bool, default=True
            Whether to write Si values back to graph
        n_jobs : int or None, optional
            Ignored (JAX uses vectorization)
        chunk_size : int or None, optional
            Chunk size hint (currently passed to NumPy backend)
        profile : MutableMapping[str, Any] or None, optional
            Dict to collect timing metrics

        Returns
        -------
        dict[Any, float] or numpy.ndarray
            Node-to-Si mapping or array of Si values
        """
        # JAX implementation planned for v2.0 - high-performance JIT compilation  
        # Currently delegates to NumPy backend for compatibility
        from ..metrics.sense_index import compute_Si

        return compute_Si(
            graph,
            inplace=inplace,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            profile=profile,
        )
