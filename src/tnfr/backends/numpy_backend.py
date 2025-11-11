"""NumPy-based vectorized backend for TNFR computations.

This module provides the canonical NumPy implementation of TNFR computational
kernels. It leverages the existing vectorized functions in `dynamics.dnfr` and
`metrics.sense_index` while providing a clean backend interface.

The NumPy backend is the default and most stable implementation, thoroughly
tested across all TNFR operations. It provides significant speedup over pure
Python fallback (~1.3-1.6x for typical graphs) through vectorized operations.

Examples
--------
>>> from tnfr.backends.numpy_backend import NumPyBackend
>>> import networkx as nx
>>> G = nx.erdos_renyi_graph(50, 0.2)
>>> backend = NumPyBackend()
>>> backend.compute_delta_nfr(G)  # Computes ΔNFR for all nodes
"""

from __future__ import annotations

from typing import Any, MutableMapping

from . import TNFRBackend
from ..types import TNFRGraph


class NumPyBackend(TNFRBackend):
    """Vectorized NumPy implementation of TNFR computational kernels.

    This backend wraps the highly-optimized NumPy-based implementations
    in `dynamics.dnfr` and `metrics.sense_index`, providing:

    - Vectorized neighbor accumulation via np.bincount and matrix operations
    - Cached buffer reuse to minimize allocations
    - Automatic sparse/dense strategy selection based on graph density
    - Optional multiprocessing for pure-Python fallback paths

    Performance characteristics:
    - 1.3-1.6x faster than Python fallback for typical graphs
    - Scales efficiently to 10,000+ nodes
    - Memory-efficient through strategic buffer caching

    Attributes
    ----------
    name : str
        Always returns "numpy"
    supports_gpu : bool
        Always False (NumPy is CPU-only)
    supports_jit : bool
        Always False (NumPy doesn't use JIT)
    """

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "numpy"

    @property
    def supports_gpu(self) -> bool:
        """NumPy backend is CPU-only."""
        return False

    @property
    def supports_jit(self) -> bool:
        """NumPy doesn't support JIT compilation."""
        return False

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        """Compute ΔNFR using vectorized NumPy operations.

        This implementation uses the canonical `default_compute_delta_nfr`
        function from `dynamics.dnfr`, which provides:

        - Automatic vectorization when NumPy is available
        - Weighted combination of phase, EPI, νf, and topology gradients
        - Intelligent sparse/dense strategy selection based on graph density
        - Optional parallel processing for large graphs

        The computation maintains all TNFR structural invariants:
        - ΔNFR = w_phase·g_phase + w_epi·g_epi + w_vf·g_vf + w_topo·g_topo
        - Phase gradients use circular mean of neighbor phases
        - Isolated nodes receive ΔNFR = 0
        - Results are deterministic with fixed graph topology

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes (phase, EPI, νf)
        cache_size : int or None, optional
            Maximum number of cached configurations. None = unlimited.
            Defaults to 1 for single-configuration optimization.
        n_jobs : int or None, optional
            Number of parallel workers for pure-Python fallback.
            Ignored when NumPy vectorization is active.
            None = serial execution, >1 = parallel processing.
        profile : MutableMapping[str, float] or None, optional
            Dict to collect timing metrics:
            - "dnfr_cache_rebuild": Time spent refreshing cached vectors
            - "dnfr_neighbor_accumulation": Time in neighbor sum computation
            - "dnfr_neighbor_means": Time computing phase/EPI/νf means
            - "dnfr_gradient_assembly": Time combining gradient components
            - "dnfr_inplace_write": Time writing ΔNFR to graph
            - "dnfr_path": "vectorized" or "fallback" execution mode

        Notes
        -----
        The implementation automatically detects graph density and selects
        between sparse (edge-based) and dense (matrix-based) accumulation:
        - Sparse path: Density ≤ 0.25, uses np.bincount on edge indices
        - Dense path: Density > 0.25, uses adjacency matrix multiplication

        Users can force dense mode by setting graph.graph["dnfr_force_dense"] = True.

        Examples
        --------
        Basic usage with profiling:

        >>> import networkx as nx
        >>> from tnfr.backends.numpy_backend import NumPyBackend
        >>> G = nx.erdos_renyi_graph(100, 0.2)
        >>> for node in G.nodes():
        ...     G.nodes[node]['phase'] = 0.0
        ...     G.nodes[node]['nu_f'] = 1.0
        ...     G.nodes[node]['epi'] = 0.5
        >>> backend = NumPyBackend()
        >>> profile = {}
        >>> backend.compute_delta_nfr(G, profile=profile)
        >>> profile['dnfr_path']
        'vectorized'
        """
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
        """Compute sense index (Si) using vectorized NumPy operations.

        This implementation uses the canonical `compute_Si` function from
        `metrics.sense_index`, which provides:

        - Vectorized computation of νf normalization, phase dispersion, and ΔNFR
        - Efficient bulk neighbor phase mean calculation
        - Strategic buffer caching to minimize allocations
        - Optional chunked processing for memory-constrained environments

        The Si metric blends three structural contributions:
        - **alpha * νf_norm**: Rewards fast structural reorganization
        - **beta * (1 - phase_disp)**: Rewards phase alignment with neighbors
        - **gamma * (1 - |ΔNFR|_norm)**: Rewards low internal turbulence

        Weights (alpha, beta, gamma) are read from graph.graph["SI_WEIGHTS"]
        and automatically normalized to sum to 1.0.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes (νf, ΔNFR, phase)
        inplace : bool, default=True
            If True, writes Si values to graph.nodes[n]['Si']
            If False, only returns the computed mapping
        n_jobs : int or None, optional
            Number of parallel workers for pure-Python fallback.
            Ignored when NumPy vectorization is active.
        chunk_size : int or None, optional
            Maximum nodes per processing batch.
            None = automatic sizing based on available memory.
            Useful for controlling memory footprint on large graphs.
        profile : MutableMapping[str, Any] or None, optional
            Dict to collect timing metrics:
            - "cache_rebuild": Time building/refreshing cached arrays
            - "neighbor_phase_mean_bulk": Time computing neighbor phase means
            - "normalize_clamp": Time normalizing and clamping Si values
            - "inplace_write": Time writing Si to graph (if inplace=True)
            - "path": "vectorized" or "fallback" execution mode
            - "fallback_chunks": Number of chunks processed (fallback only)

        Returns
        -------
        dict[Any, float] or numpy.ndarray
            If inplace=False: dict mapping node IDs to Si values
            If inplace=True and NumPy available: numpy array of Si values
            If inplace=True and fallback: dict mapping node IDs to Si values

        Notes
        -----
        The vectorized implementation achieves significant speedup through:
        1. Batch neighbor accumulation via edge index arrays
        2. Vectorized phase dispersion with angle_diff_array
        3. Cached buffer reuse across invocations
        4. Efficient normalization with np.clip and in-place operations

        Examples
        --------
        Compute Si with custom weights:

        >>> import networkx as nx
        >>> from tnfr.backends.numpy_backend import NumPyBackend
        >>> G = nx.erdos_renyi_graph(50, 0.3)
        >>> for node in G.nodes():
        ...     G.nodes[node]['phase'] = 0.0
        ...     G.nodes[node]['nu_f'] = 0.8
        ...     G.nodes[node]['delta_nfr'] = 0.1
        >>> G.graph['SI_WEIGHTS'] = {'alpha': 0.4, 'beta': 0.4, 'gamma': 0.2}
        >>> backend = NumPyBackend()
        >>> si_values = backend.compute_si(G, inplace=False)
        >>> all(0.0 <= v <= 1.0 for v in si_values.values())
        True
        """
        from ..metrics.sense_index import compute_Si

        return compute_Si(
            graph,
            inplace=inplace,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            profile=profile,
        )
