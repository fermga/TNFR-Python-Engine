"""GPU acceleration for TNFR computations.

Optional module providing JAX and CuPy integration for GPU-accelerated
vectorized operations. Requires installation of optional dependencies:
    pip install tnfr[jax]  # or
    pip install tnfr[cupy]
"""

from __future__ import annotations

from typing import Any, Optional

# Check for optional GPU backends
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # type: ignore

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    jit = None  # type: ignore


class TNFRGPUEngine:
    """GPU acceleration engine for TNFR computations.

    Provides vectorized GPU implementations of ΔNFR and other TNFR operations
    using JAX or CuPy backends.

    Parameters
    ----------
    backend : {"auto", "jax", "cupy", "numpy"}, default="auto"
        GPU backend to use. "auto" prefers JAX, then CuPy, then NumPy fallback.

    Raises
    ------
    ImportError
        If requested GPU backend is not installed

    Examples
    --------
    >>> # Requires JAX or CuPy installation
    >>> try:
    ...     from tnfr.parallel import TNFRGPUEngine
    ...     engine = TNFRGPUEngine(backend="auto")
    ...     # engine.backend in ["jax", "cupy", "numpy"]
    ... except ImportError:
    ...     pass  # Optional dependency not installed

    Notes
    -----
    GPU acceleration provides significant speedup for large dense networks
    but requires compatible hardware and drivers. For sparse networks or
    small graphs, multiprocessing may be more efficient.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_gpu_backend(backend)

    def _select_gpu_backend(self, backend: str) -> str:
        """Select available GPU backend."""
        if backend == "auto":
            if HAS_JAX:
                return "jax"
            elif HAS_CUPY:
                return "cupy"
            else:
                return "numpy"  # Fallback

        if backend == "jax" and not HAS_JAX:
            raise ImportError(
                "JAX not available. Install with: pip install jax[cuda]"
            )
        if backend == "cupy" and not HAS_CUPY:
            raise ImportError(
                "CuPy not available. Install with: pip install cupy"
            )

        return backend

    def compute_delta_nfr_gpu(
        self,
        adjacency_matrix: Any,
        epi_vector: Any,
        vf_vector: Any,
        phase_vector: Any,
    ) -> Any:
        """Compute ΔNFR using vectorized GPU operations.

        Parameters
        ----------
        adjacency_matrix : array-like
            Network adjacency matrix (N x N)
        epi_vector : array-like
            EPI values for all nodes (N,)
        vf_vector : array-like
            Structural frequencies νf for all nodes (N,)
        phase_vector : array-like
            Phase values θ for all nodes (N,)

        Returns
        -------
        array-like
            ΔNFR values for all nodes (N,)

        Notes
        -----
        This is a placeholder for future GPU-accelerated implementations.
        Actual GPU computation requires careful optimization and testing.
        Current implementation raises NotImplementedError.
        """
        if self.backend == "jax" and HAS_JAX:
            return self._compute_delta_nfr_jax(
                adjacency_matrix, epi_vector, vf_vector, phase_vector
            )
        elif self.backend == "cupy" and HAS_CUPY:
            return self._compute_delta_nfr_cupy(
                adjacency_matrix, epi_vector, vf_vector, phase_vector
            )
        else:
            return self._compute_delta_nfr_numpy(
                adjacency_matrix, epi_vector, vf_vector, phase_vector
            )

    def _compute_delta_nfr_jax(
        self, adj_matrix: Any, epi_vec: Any, vf_vec: Any, phase_vec: Any
    ) -> Any:
        """JAX implementation with JIT compilation.

        Notes
        -----
        Placeholder for future implementation. JAX provides excellent
        GPU acceleration with automatic differentiation and JIT compilation.
        """
        raise NotImplementedError(
            "JAX GPU acceleration requires custom implementation. "
            "Use standard parallelization for now."
        )

    def _compute_delta_nfr_cupy(
        self, adj_matrix: Any, epi_vec: Any, vf_vec: Any, phase_vec: Any
    ) -> Any:
        """CuPy implementation for CUDA GPUs.

        Notes
        -----
        Placeholder for future implementation. CuPy provides NumPy-compatible
        GPU arrays with CUDA acceleration.
        """
        raise NotImplementedError(
            "CuPy GPU acceleration requires custom implementation. "
            "Use standard parallelization for now."
        )

    def _compute_delta_nfr_numpy(
        self, adj_matrix: Any, epi_vec: Any, vf_vec: Any, phase_vec: Any
    ) -> Any:
        """NumPy fallback implementation (CPU-only).

        Notes
        -----
        Placeholder for future implementation. This would use existing
        NumPy-based ΔNFR computation as fallback.
        """
        raise NotImplementedError(
            "NumPy fallback requires integration with existing vectorized code. "
            "Use default_compute_delta_nfr for standard computation."
        )

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is actually available."""
        if self.backend == "jax" and HAS_JAX:
            try:
                # Check if JAX has GPU backend
                return len(jax.devices("gpu")) > 0
            except Exception:
                return False
        elif self.backend == "cupy" and HAS_CUPY:
            try:
                # Check if CuPy can access GPU
                return cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                return False
        return False
