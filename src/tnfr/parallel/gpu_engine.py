"""GPU acceleration for TNFR computations.

Optional module providing JAX and CuPy integration for GPU-accelerated
vectorized operations. Requires installation of optional dependencies:
    pip install tnfr[jax]  # or
    pip install tnfr[cupy]
"""

from __future__ import annotations

from typing import Any, Dict

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
            raise ImportError("JAX not available. Install with: pip install jax[cuda]")
        if backend == "cupy" and not HAS_CUPY:
            raise ImportError("CuPy not available. Install with: pip install cupy")

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
        """JAX implementation with JIT compilation for GPU acceleration.

        Implements vectorized ΔNFR computation using JAX for automatic
        GPU acceleration and JIT compilation.

        Parameters
        ----------
        adj_matrix : array-like
            Adjacency matrix (N x N)
        epi_vec : array-like
            EPI values (N,)
        vf_vec : array-like
            Structural frequencies (N,)
        phase_vec : array-like
            Phase values (N,)

        Returns
        -------
        jax.numpy.ndarray
            ΔNFR values for all nodes

        Notes
        -----
        Uses the canonical TNFR nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t)

        ΔNFR is computed from:
        - Topological gradient (EPI differences with neighbors)
        - Phase gradient (phase synchronization)
        - Weighted by structural frequency
        """
        if not HAS_JAX:
            raise ImportError("JAX required for GPU acceleration")

        # Convert inputs to JAX arrays
        adj = jnp.asarray(adj_matrix)
        epi = jnp.asarray(epi_vec)
        vf = jnp.asarray(vf_vec)
        phase = jnp.asarray(phase_vec)

        # Define JIT-compiled ΔNFR computation
        @jit
        def compute_dnfr_vectorized(adj, epi, vf, phase):
            """Vectorized ΔNFR computation (JIT compiled)."""
            # Topological gradient: difference in EPI with neighbors
            # epi_diff[i,j] = epi[j] - epi[i]
            epi_diff = epi[None, :] - epi[:, None]  # (N, N) matrix
            topo_gradient = jnp.sum(adj * epi_diff, axis=1)  # (N,) vector

            # Phase gradient: phase difference with neighbors
            # phase_diff[i,j] = sin(phase[j] - phase[i])
            phase_diff = jnp.sin(phase[None, :] - phase[:, None])  # (N, N)
            phase_gradient = jnp.sum(adj * phase_diff, axis=1)  # (N,)

            # Normalize by degree (number of neighbors)
            degree = jnp.sum(adj, axis=1)
            # Avoid division by zero
            degree_safe = jnp.where(degree > 0, degree, 1.0)

            topo_gradient = topo_gradient / degree_safe
            phase_gradient = phase_gradient / degree_safe

            # Combine gradients with TNFR weights
            # Emphasize topological structure (0.7) over phase (0.3)
            combined_gradient = 0.7 * topo_gradient + 0.3 * phase_gradient

            # Apply structural frequency modulation (canonical equation)
            delta_nfr = vf * combined_gradient

            return delta_nfr

        # Execute JIT-compiled computation (GPU accelerated if available)
        result = compute_dnfr_vectorized(adj, epi, vf, phase)

        return result

    def _compute_delta_nfr_cupy(
        self, adj_matrix: Any, epi_vec: Any, vf_vec: Any, phase_vec: Any
    ) -> Any:
        """CuPy implementation for CUDA GPUs.

        Implements vectorized ΔNFR computation using CuPy for CUDA GPU
        acceleration with NumPy-compatible interface.

        Parameters
        ----------
        adj_matrix : array-like
            Adjacency matrix (N x N)
        epi_vec : array-like
            EPI values (N,)
        vf_vec : array-like
            Structural frequencies (N,)
        phase_vec : array-like
            Phase values (N,)

        Returns
        -------
        cupy.ndarray
            ΔNFR values for all nodes (on GPU)
        """
        if not HAS_CUPY:
            raise ImportError("CuPy required for CUDA GPU acceleration")

        # Transfer to GPU
        adj = cp.asarray(adj_matrix)
        epi = cp.asarray(epi_vec)
        vf = cp.asarray(vf_vec)
        phase = cp.asarray(phase_vec)

        # Topological gradient (vectorized on GPU)
        epi_diff = epi[None, :] - epi[:, None]
        topo_gradient = cp.sum(adj * epi_diff, axis=1)

        # Phase gradient (vectorized on GPU)
        phase_diff = cp.sin(phase[None, :] - phase[:, None])
        phase_gradient = cp.sum(adj * phase_diff, axis=1)

        # Normalize by degree
        degree = cp.sum(adj, axis=1)
        degree_safe = cp.where(degree > 0, degree, 1.0)

        topo_gradient = topo_gradient / degree_safe
        phase_gradient = phase_gradient / degree_safe

        # Combine with TNFR weights
        combined_gradient = 0.7 * topo_gradient + 0.3 * phase_gradient

        # Apply structural frequency
        delta_nfr = vf * combined_gradient

        return delta_nfr

    def _compute_delta_nfr_numpy(
        self, adj_matrix: Any, epi_vec: Any, vf_vec: Any, phase_vec: Any
    ) -> Any:
        """NumPy fallback implementation (CPU-only).

        Provides CPU-based vectorized computation when GPU is unavailable.

        Parameters
        ----------
        adj_matrix : array-like
            Adjacency matrix (N x N)
        epi_vec : array-like
            EPI values (N,)
        vf_vec : array-like
            Structural frequencies (N,)
        phase_vec : array-like
            Phase values (N,)

        Returns
        -------
        numpy.ndarray
            ΔNFR values for all nodes
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy required for CPU computation")

        # Convert to numpy arrays
        adj = np.asarray(adj_matrix)
        epi = np.asarray(epi_vec)
        vf = np.asarray(vf_vec)
        phase = np.asarray(phase_vec)

        # Topological gradient
        epi_diff = epi[None, :] - epi[:, None]
        topo_gradient = np.sum(adj * epi_diff, axis=1)

        # Phase gradient
        phase_diff = np.sin(phase[None, :] - phase[:, None])
        phase_gradient = np.sum(adj * phase_diff, axis=1)

        # Normalize by degree
        degree = np.sum(adj, axis=1)
        degree_safe = np.where(degree > 0, degree, 1.0)

        topo_gradient = topo_gradient / degree_safe
        phase_gradient = phase_gradient / degree_safe

        # Combine with TNFR weights
        combined_gradient = 0.7 * topo_gradient + 0.3 * phase_gradient

        # Apply structural frequency
        delta_nfr = vf * combined_gradient

        return delta_nfr

    def compute_delta_nfr_from_graph(self, graph: Any) -> Dict[Any, float]:
        """Compute ΔNFR directly from a TNFR graph using GPU acceleration.

        Convenience method that extracts matrices from graph and computes
        ΔNFR using GPU backend.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph with TNFR attributes

        Returns
        -------
        Dict[Any, float]
            Mapping from node IDs to ΔNFR values

        Examples
        --------
        >>> import networkx as nx
        >>> from tnfr.parallel import TNFRGPUEngine
        >>> G = nx.Graph([(0, 1), (1, 2)])
        >>> for node in G.nodes():
        ...     G.nodes[node]['epi'] = 0.5
        ...     G.nodes[node]['nu_f'] = 1.0
        ...     G.nodes[node]['phase'] = 0.0
        >>> engine = TNFRGPUEngine(backend="numpy")  # Use numpy for testing
        >>> result = engine.compute_delta_nfr_from_graph(G)
        >>> len(result) == 3
        True
        """

        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy required for graph processing")

        # Extract node list (maintain order)
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Build adjacency matrix
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        for i, j in graph.edges():
            idx_i = node_to_idx[i]
            idx_j = node_to_idx[j]
            adj_matrix[idx_i, idx_j] = 1.0
            adj_matrix[idx_j, idx_i] = 1.0  # Undirected

        # Extract node attributes
        def get_attr(node, attr_names, default):
            """Get attribute with fallbacks."""
            for name in (attr_names if isinstance(attr_names, (list, tuple)) else [attr_names]):
                if name in graph.nodes[node]:
                    return float(graph.nodes[node][name])
            return default

        epi_vec = np.array([get_attr(node, ["epi", "EPI"], 0.5) for node in nodes])
        vf_vec = np.array([get_attr(node, ["nu_f", "vf", "νf"], 1.0) for node in nodes])
        phase_vec = np.array([get_attr(node, ["phase", "theta"], 0.0) for node in nodes])

        # Compute ΔNFR using GPU
        delta_nfr_array = self.compute_delta_nfr_gpu(adj_matrix, epi_vec, vf_vec, phase_vec)

        # Convert back to dictionary
        if self.backend == "cupy" and HAS_CUPY:
            delta_nfr_array = cp.asnumpy(delta_nfr_array)  # Transfer from GPU
        elif self.backend == "jax" and HAS_JAX:
            delta_nfr_array = np.array(delta_nfr_array)  # Convert from JAX

        result = {node: float(delta_nfr_array[idx]) for idx, node in enumerate(nodes)}

        return result

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
