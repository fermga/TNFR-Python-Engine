"""PyTorch-based GPU-accelerated backend for TNFR computations (Experimental).

This module provides a PyTorch implementation of TNFR computational kernels
with support for:

- GPU acceleration via CUDA/ROCm
- Automatic differentiation with autograd
- Optimized tensor operations
- Mixed precision training support

**Status**: Experimental - API may change in future releases.

The Torch backend currently delegates to the NumPy implementation but provides
infrastructure for future GPU-optimized kernels.

Examples
--------
>>> from tnfr.backends import get_backend
>>> backend = get_backend("torch")  # doctest: +SKIP
>>> backend.supports_gpu  # doctest: +SKIP
True
"""

from __future__ import annotations

from typing import Any, MutableMapping

from . import TNFRBackend
from ..types import TNFRGraph


class TorchBackend(TNFRBackend):
    """PyTorch GPU-accelerated implementation of TNFR kernels (Experimental).

    This backend provides a foundation for GPU-accelerated TNFR computations
    using PyTorch. Current implementation delegates to NumPy backend while
    maintaining interface compatibility for future GPU implementations.

    Future optimizations planned:
    - GPU-accelerated ΔNFR computation using torch tensors
    - Sparse tensor operations for large-scale graphs
    - Mixed precision support (FP16/BF16) for memory efficiency
    - Automatic device placement (CPU/CUDA/ROCm)
    - Integration with PyTorch Geometric for graph operations

    Attributes
    ----------
    name : str
        Returns "torch"
    supports_gpu : bool
        True (PyTorch supports GPU acceleration)
    supports_jit : bool
        False (TorchScript not yet integrated)

    Notes
    -----
    Requires PyTorch to be installed: `pip install torch`

    For GPU support, install PyTorch with CUDA:
    `pip install torch --index-url https://download.pytorch.org/whl/cu118`
    """

    def __init__(self) -> None:
        """Initialize PyTorch backend."""
        try:
            import torch

            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch backend requires torch to be installed. " "Install with: pip install torch"
            ) from exc

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "torch"

    @property
    def supports_gpu(self) -> bool:
        """PyTorch supports GPU acceleration."""
        return True

    @property
    def supports_jit(self) -> bool:
        """TorchScript not yet integrated."""
        return False

    @property
    def device(self) -> Any:
        """Return the current PyTorch device (CPU or CUDA)."""
        return self._device

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        """Compute ΔNFR using PyTorch backend with GPU acceleration.

        Implements vectorized ΔNFR computation using PyTorch tensors with
        automatic device placement (CPU/CUDA). For large graphs (>1000 nodes),
        uses GPU if available for significant speedup.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        cache_size : int or None, optional
            Cache size hint (ignored, PyTorch manages memory)
        n_jobs : int or None, optional
            Ignored (PyTorch uses GPU parallelism)
        profile : MutableMapping[str, float] or None, optional
            Dict to collect timing metrics

        Notes
        -----
        Automatically moves tensors to GPU if available (backend.device).
        For small graphs (<1000 nodes), may use NumPy backend to avoid
        overhead of tensor conversion and device transfer.
        """
        import time
        import numpy as np

        if profile is not None:
            profile["dnfr_backend"] = "torch"
            profile["dnfr_device"] = str(self._device)

        n_nodes = graph.number_of_nodes()
        graph.number_of_edges()

        # For very small graphs, delegate to NumPy (tensor overhead not worth it)
        if n_nodes < 1000:
            if profile is not None:
                profile["dnfr_path"] = "numpy_fallback"
            from ..dynamics.dnfr import default_compute_delta_nfr

            default_compute_delta_nfr(graph, cache_size=cache_size, n_jobs=n_jobs, profile=profile)
            return

        if profile is not None:
            profile["dnfr_path"] = "torch_gpu"
            t0 = time.perf_counter()

        # Extract graph data
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Get node attributes as numpy arrays first
        phase = np.array(
            [graph.nodes[node].get("phase", 0.0) for node in node_list],
            dtype=np.float32,
        )
        epi = np.array([graph.nodes[node].get("EPI", 0.5) for node in node_list], dtype=np.float32)
        vf = np.array([graph.nodes[node].get("nu_f", 1.0) for node in node_list], dtype=np.float32)

        # Get edge list
        edges = list(graph.edges())
        if not edges:
            # No edges - all nodes get zero ΔNFR
            for node in node_list:
                graph.nodes[node]["ΔNFR"] = 0.0
            return

        edge_src = np.array([node_to_idx[src] for src, _ in edges], dtype=np.int64)
        edge_dst = np.array([node_to_idx[dst] for _, dst in edges], dtype=np.int64)

        # Get weights
        weights = graph.graph.get("DNFR_WEIGHTS", {})
        w_phase = float(weights.get("phase", 0.0))
        w_epi = float(weights.get("epi", 0.0))
        w_vf = float(weights.get("vf", 0.0))
        w_topo = float(weights.get("topo", 0.0))

        if profile is not None:
            profile["dnfr_data_prep"] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Convert to PyTorch tensors and move to device
        phase_t = self._torch.tensor(phase, device=self._device, dtype=self._torch.float32)
        epi_t = self._torch.tensor(epi, device=self._device, dtype=self._torch.float32)
        vf_t = self._torch.tensor(vf, device=self._device, dtype=self._torch.float32)
        edge_src_t = self._torch.tensor(edge_src, device=self._device, dtype=self._torch.int64)
        edge_dst_t = self._torch.tensor(edge_dst, device=self._device, dtype=self._torch.int64)

        if profile is not None:
            profile["dnfr_to_device"] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Compute ΔNFR using PyTorch operations
        delta_nfr_t = self._compute_delta_nfr_torch(
            phase_t,
            epi_t,
            vf_t,
            edge_src_t,
            edge_dst_t,
            w_phase,
            w_epi,
            w_vf,
            w_topo,
            graph.is_directed(),
        )

        if profile is not None:
            profile["dnfr_compute"] = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Convert back to numpy and write to graph
        delta_nfr = delta_nfr_t.cpu().numpy()

        if profile is not None:
            profile["dnfr_from_device"] = time.perf_counter() - t0
            t0 = time.perf_counter()

        for idx, node in enumerate(node_list):
            graph.nodes[node]["ΔNFR"] = float(delta_nfr[idx])

        if profile is not None:
            profile["dnfr_write_back"] = time.perf_counter() - t0

    def _compute_delta_nfr_torch(
        self,
        phase: Any,
        epi: Any,
        vf: Any,
        edge_src: Any,
        edge_dst: Any,
        w_phase: float,
        w_epi: float,
        w_vf: float,
        w_topo: float,
        is_directed: bool,
    ) -> Any:
        """Compute ΔNFR using PyTorch tensor operations.

        Implements the TNFR canonical formula:
        ΔNFR = νf · (w_phase·g_phase + w_epi·g_epi + w_vf·g_vf + w_topo·g_topo)

        Where:
        - g_phase = angle_diff(phase_mean, phase) / π (circular mean)
        - g_epi = epi_mean - epi
        - g_vf = vf_mean - vf
        - g_topo = neighbor_count · w_topo

        Parameters
        ----------
        phase, epi, vf : torch.Tensor
            Node attribute tensors on device
        edge_src, edge_dst : torch.Tensor
            Edge index tensors
        w_phase, w_epi, w_vf, w_topo : float
            Component weights
        is_directed : bool
            Whether graph is directed

        Returns
        -------
        torch.Tensor
            ΔNFR values for all nodes
        """
        n_nodes = phase.shape[0]
        torch = self._torch

        # Initialize accumulators
        neighbor_cos_sum = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        neighbor_sin_sum = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        neighbor_epi_sum = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        neighbor_vf_sum = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        neighbor_count = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)

        # Accumulate neighbor statistics
        # For each edge, dst receives contributions from src
        neighbor_cos_sum.scatter_add_(0, edge_dst, torch.cos(phase[edge_src]))
        neighbor_sin_sum.scatter_add_(0, edge_dst, torch.sin(phase[edge_src]))
        neighbor_epi_sum.scatter_add_(0, edge_dst, epi[edge_src])
        neighbor_vf_sum.scatter_add_(0, edge_dst, vf[edge_src])
        neighbor_count.scatter_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=torch.float32))

        # For undirected graphs, also accumulate in reverse
        if not is_directed:
            neighbor_cos_sum.scatter_add_(0, edge_src, torch.cos(phase[edge_dst]))
            neighbor_sin_sum.scatter_add_(0, edge_src, torch.sin(phase[edge_dst]))
            neighbor_epi_sum.scatter_add_(0, edge_src, epi[edge_dst])
            neighbor_vf_sum.scatter_add_(0, edge_src, vf[edge_dst])
            neighbor_count.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float32))

        # Compute means
        has_neighbors = neighbor_count > 0

        # Circular mean for phase (using atan2)
        phase_mean = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        phase_mean[has_neighbors] = torch.atan2(
            neighbor_sin_sum[has_neighbors], neighbor_cos_sum[has_neighbors]
        )

        # Arithmetic means for EPI and vf
        epi_mean = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        vf_mean = torch.zeros(n_nodes, device=self._device, dtype=torch.float32)
        epi_mean[has_neighbors] = neighbor_epi_sum[has_neighbors] / neighbor_count[has_neighbors]
        vf_mean[has_neighbors] = neighbor_vf_sum[has_neighbors] / neighbor_count[has_neighbors]

        # Compute gradients using TNFR canonical formula
        # Phase: angle_diff with wrapping to [-π, π]
        phase_diff = (phase_mean - phase + torch.pi) % (2 * torch.pi) - torch.pi
        g_phase = phase_diff / torch.pi
        g_phase[~has_neighbors] = 0.0

        # EPI and vf gradients
        g_epi = epi_mean - epi
        g_epi[~has_neighbors] = 0.0

        g_vf = vf_mean - vf
        g_vf[~has_neighbors] = 0.0

        # Topology gradient
        g_topo = neighbor_count * w_topo

        # Combine gradients
        delta_nfr = w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + g_topo

        # Apply structural frequency scaling (canonical TNFR)
        delta_nfr = vf * delta_nfr

        return delta_nfr

    def compute_si(
        self,
        graph: TNFRGraph,
        *,
        inplace: bool = True,
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> dict[Any, float] | Any:
        """Compute sense index using PyTorch backend.

        **Current implementation**: Delegates to NumPy backend while maintaining
        interface compatibility.

        **Planned**: GPU-accelerated vectorized Si computation using torch tensors
        with optimized phase dispersion kernels and mixed precision support.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        inplace : bool, default=True
            Whether to write Si values back to graph
        n_jobs : int or None, optional
            Ignored (PyTorch uses GPU parallelism)
        chunk_size : int or None, optional
            Chunk size hint (currently passed to NumPy backend)
        profile : MutableMapping[str, Any] or None, optional
            Dict to collect timing metrics

        Returns
        -------
        dict[Any, float] or numpy.ndarray
            Node-to-Si mapping or array of Si values

        Notes
        -----
        When implemented, will support mixed precision (FP16/BF16) for
        memory-efficient computation on large graphs, selectable via
        graph.graph["TORCH_DTYPE"] = torch.float16
        """
        # PyTorch GPU implementation planned for v2.0 - mixed precision support
        # Currently delegates to NumPy backend for compatibility
        from ..metrics.sense_index import compute_Si

        return compute_Si(
            graph,
            inplace=inplace,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            profile=profile,
        )
