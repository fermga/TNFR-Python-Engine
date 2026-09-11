"""Memory-optimized sparse representations for TNFR graphs.

Implements sparse storage strategies that minimize memory footprint while
maintaining computational efficiency and TNFR semantic fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Sequence

from scipy import sparse

from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..metrics.common import finite_mean_absolute, structural_coherence
from ..types import NodeId
from ..utils import get_logger

logger = get_logger(__name__)


def _finite_real_scalar(value: Any, *, name: str) -> float:
    """Return a finite real scalar while rejecting truth values and arrays."""
    if isinstance(value, (bool, np.bool_)):
        raise TNFRValueError(f"{name} must be a finite real scalar, not bool")
    if isinstance(value, (str, bytes)) or not bool(np.isscalar(value)):
        raise TNFRValueError(f"{name} must be a finite real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(normalized):
        raise TNFRValueError(f"{name} must be finite")
    return normalized


def _bounded_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Normalize an integral scalar within inclusive bounds."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TNFRValueError(f"{name} must be an integer, not bool or a fraction")
    normalized = int(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        interval = (
            f"[{minimum}, {maximum}]"
            if maximum is not None
            else f"[{minimum}, infinity)"
        )
        raise TNFRValueError(
            f"{name} must be in {interval}",
            context={name: value},
        )
    return normalized


def _node_id(value: Any, *, node_count: int) -> int:
    """Return an integer node ID in the sparse graph's fixed node domain."""
    return _bounded_integer(
        value,
        name="node_id",
        minimum=0,
        maximum=node_count - 1,
    )


def _node_ids(
    values: Sequence[NodeId],
    *,
    node_count: int,
) -> tuple[int, ...]:
    """Validate a requested sequence before any cache access or mutation."""
    try:
        return tuple(_node_id(value, node_count=node_count) for value in values)
    except TypeError as exc:
        raise TNFRValueError("node_ids must be an iterable of node IDs") from exc


def _float32_value(value: Any, *, name: str) -> np.float32:
    """Validate a scalar that the compact store can represent finitely."""
    normalized = _finite_real_scalar(value, name=name)
    if abs(normalized) > float(np.finfo(np.float32).max):
        raise TNFRValueError(f"{name} must be representable as finite float32")
    stored = np.float32(normalized)
    if normalized != 0.0 and stored == 0.0:
        raise TNFRValueError(f"{name} must be representable as nonzero float32")
    return stored


def _mean_absolute_channel(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    name: str,
) -> float:
    """Validate a full sparse channel and delegate its scalar reduction."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{name} must be a one-dimensional real channel") from exc
    if raw.shape != (expected_size,):
        raise TNFRValueError(
            f"{name} must contain exactly one value per node",
            context={"expected_shape": (expected_size,), "actual_shape": raw.shape},
        )
    try:
        return finite_mean_absolute(raw, name=name)
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{name} must contain only finite real values"
        ) from exc


@dataclass
class MemoryReport:
    """Memory usage report for sparse TNFR graphs.

    Attributes
    ----------
    total_mb : float
        Estimated tracked component usage in megabytes.
    per_node_kb : float
        Estimated tracked component usage per node in kilobytes.
    breakdown : dict[str, int]
        Estimated payload and populated-entry usage by component in bytes.
    """

    total_mb: float
    per_node_kb: float
    breakdown: dict[str, int]


class SparseCache:
    """Bounded write-age cache for sparse computation results.

    Entries expire after a configured number of evolution steps. Capacity
    eviction removes the oldest write; reads do not refresh entry age, so this
    is deliberately not an LRU cache.

    Parameters
    ----------
    capacity : int
        Maximum number of cached entries. Zero disables storage.
    ttl_steps : int
        Positive time-to-live in evolution steps before invalidation.
    """

    def __init__(self, capacity: int, ttl_steps: int = 10):
        self.capacity = _bounded_integer(
            capacity,
            name="capacity",
            minimum=0,
        )
        self.ttl_steps = _bounded_integer(
            ttl_steps,
            name="ttl_steps",
            minimum=1,
        )
        self._cache: dict[NodeId, tuple[float, int]] = {}
        self._current_step = 0

    def get(self, node_id: NodeId) -> float | None:
        """Get a cached value if its write age remains inside the TTL."""
        cached = self._cache.get(node_id)
        if cached is None:
            return None
        value, cached_step = cached
        if self._current_step - cached_step < self.ttl_steps:
            return value
        del self._cache[node_id]
        return None

    def update(self, values: dict[NodeId, float]) -> None:
        """Atomically validate and add values while respecting capacity."""
        normalized = tuple(
            (node_id, _finite_real_scalar(value, name="cache value"))
            for node_id, value in values.items()
        )

        for node_id, value in normalized:
            # Reinsert an existing key so equal-age eviction retains the latest
            # insertion order deterministically.
            self._cache.pop(node_id, None)
            self._cache[node_id] = (value, self._current_step)

        overflow = len(self._cache) - self.capacity
        if overflow > 0:
            # Python's stable sort preserves insertion order for equal ages.
            oldest_keys = sorted(
                self._cache,
                key=lambda key: self._cache[key][1],
            )[:overflow]
            for oldest_key in oldest_keys:
                del self._cache[oldest_key]

    def step(self) -> None:
        """Advance the evolution-step counter."""
        self._current_step += 1

    def clear(self) -> None:
        """Clear all cached values and reset their age origin."""
        self._cache.clear()
        self._current_step = 0

    def memory_usage(self) -> int:
        """Return the estimated populated-entry usage in bytes."""
        # Per entry: integer ID, binary64 value, step and estimated dict slot.
        bytes_per_entry = 8 + 8 + 8 + 112
        return len(self._cache) * bytes_per_entry


class CompactAttributeStore:
    """Compressed float32 storage for a fixed integer node domain.

    Only values whose float32 representation differs from the declared default
    occupy sparse dictionaries. Every setter rejects non-finite values and
    values that cannot be represented by this storage type.

    Parameters
    ----------
    node_count : int
        Positive number of nodes in the fixed domain ``range(node_count)``.
    on_theta_change : callable, optional
        Callback invoked after a stored phase value actually changes.
    """

    def __init__(
        self,
        node_count: int,
        *,
        on_theta_change: Callable[[], None] | None = None,
    ):
        self.node_count = _bounded_integer(
            node_count,
            name="node_count",
            minimum=1,
        )
        if on_theta_change is not None and not callable(on_theta_change):
            raise TNFRValueError("on_theta_change must be callable or None")
        self._on_theta_change = on_theta_change

        # TNFR canonical defaults
        self.default_vf = 1.0  # Hz_str
        self.default_theta = 0.0  # radians
        self.default_si = 0.0
        self.default_epi = 0.0
        self.default_dnfr = 0.0

        # Only non-default float32 values occupy sparse dictionaries.
        self._vf_sparse: dict[int, np.float32] = {}
        self._theta_sparse: dict[int, np.float32] = {}
        self._si_sparse: dict[int, np.float32] = {}
        self._epi_sparse: dict[int, np.float32] = {}
        self._dnfr_sparse: dict[int, np.float32] = {}

    @staticmethod
    def _assign(
        target: dict[int, np.float32],
        node_id: int,
        value: np.float32,
        *,
        default: float,
    ) -> None:
        """Store exactly the non-default float32 representation."""
        if value == np.float32(default):
            target.pop(node_id, None)
        else:
            target[node_id] = value

    def _node(self, node_id: NodeId) -> int:
        return _node_id(node_id, node_count=self.node_count)

    def _nodes(self, node_ids: Sequence[NodeId]) -> tuple[int, ...]:
        return _node_ids(node_ids, node_count=self.node_count)

    def set_vf(self, node_id: NodeId, vf: float) -> None:
        """Set a finite nonnegative structural frequency in Hz_str."""
        node = self._node(node_id)
        stored = _float32_value(vf, name="vf")
        if stored < 0.0:
            raise TNFRValueError("vf must be nonnegative")
        self._assign(self._vf_sparse, node, stored, default=self.default_vf)

    def get_vf(self, node_id: NodeId) -> float:
        """Get structural frequency with the canonical default fallback."""
        node = self._node(node_id)
        return float(self._vf_sparse.get(node, self.default_vf))

    def get_vfs(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Get structural frequencies in the requested node order."""
        nodes = self._nodes(node_ids)
        result = np.full(len(nodes), self.default_vf, dtype=np.float32)
        for index, node in enumerate(nodes):
            if node in self._vf_sparse:
                result[index] = self._vf_sparse[node]
        return result

    def set_theta(self, node_id: NodeId, theta: float) -> None:
        """Set finite phase modulo ``2*pi`` and notify on stored change."""
        node = self._node(node_id)
        normalized = _finite_real_scalar(theta, name="theta") % math.tau
        stored = _float32_value(normalized, name="theta")
        # The top float32 bin below 2*pi can round above the binary64 bound;
        # zero is its equivalent canonical circular representative.
        if float(stored) >= math.tau:
            stored = np.float32(0.0)
        previous = np.float32(
            self._theta_sparse.get(node, self.default_theta)
        )
        self._assign(
            self._theta_sparse,
            node,
            stored,
            default=self.default_theta,
        )
        if stored != previous and self._on_theta_change is not None:
            self._on_theta_change()

    def get_theta(self, node_id: NodeId) -> float:
        """Get phase with the canonical default fallback."""
        node = self._node(node_id)
        return float(self._theta_sparse.get(node, self.default_theta))

    def get_thetas(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Get phases in the requested node order."""
        nodes = self._nodes(node_ids)
        result = np.full(len(nodes), self.default_theta, dtype=np.float32)
        for index, node in enumerate(nodes):
            if node in self._theta_sparse:
                result[index] = self._theta_sparse[node]
        return result

    def set_si(self, node_id: NodeId, si: float) -> None:
        """Set a finite nonnegative sense index."""
        node = self._node(node_id)
        stored = _float32_value(si, name="si")
        if stored < 0.0:
            raise TNFRValueError("si must be nonnegative")
        self._assign(self._si_sparse, node, stored, default=self.default_si)

    def get_si(self, node_id: NodeId) -> float:
        """Get sense index with the canonical default fallback."""
        node = self._node(node_id)
        return float(self._si_sparse.get(node, self.default_si))

    def set_epi(self, node_id: NodeId, epi: float) -> None:
        """Set a finite representable EPI coordinate."""
        node = self._node(node_id)
        stored = _float32_value(epi, name="epi")
        self._assign(self._epi_sparse, node, stored, default=self.default_epi)

    def get_epi(self, node_id: NodeId) -> float:
        """Get EPI with the canonical default fallback."""
        node = self._node(node_id)
        return float(self._epi_sparse.get(node, self.default_epi))

    def get_epis(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Get EPI coordinates in the requested node order."""
        nodes = self._nodes(node_ids)
        result = np.full(len(nodes), self.default_epi, dtype=np.float32)
        for index, node in enumerate(nodes):
            if node in self._epi_sparse:
                result[index] = self._epi_sparse[node]
        return result

    def set_dnfr(self, node_id: NodeId, dnfr: float) -> None:
        """Set a finite representable structural pressure."""
        node = self._node(node_id)
        stored = _float32_value(dnfr, name="dnfr")
        self._assign(
            self._dnfr_sparse,
            node,
            stored,
            default=self.default_dnfr,
        )

    def get_dnfr(self, node_id: NodeId) -> float:
        """Get structural pressure with the canonical default fallback."""
        node = self._node(node_id)
        return float(self._dnfr_sparse.get(node, self.default_dnfr))

    def memory_usage(self) -> int:
        """Estimate populated attribute-entry usage in bytes."""
        # Per entry: integer key, float32 value and estimated dict slot.
        bytes_per_entry = 8 + 4 + 112
        sparse_channels = (
            self._vf_sparse,
            self._theta_sparse,
            self._si_sparse,
            self._epi_sparse,
            self._dnfr_sparse,
        )
        return sum(len(channel) * bytes_per_entry for channel in sparse_channels)


class SparseTNFRGraph:
    """Memory-optimized TNFR graph using sparse representations.

    Its reported component footprint can remain below 1 KB per node for
    genuinely sparse workloads by using:
    - Mutable LIL adjacency with transient CSR materialization
    - Compact float32 attributes that store only non-default values
    - Bounded write-age caching with dependency invalidation

    The representation implements this scoped contract:
    - Explicit Euler update of the nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Deterministic computation with reproducible seeds
    - Cache invalidation for graph-owned topology and phase mutation

    Canonical operator grammar, operator-only mutation, and U3 admission are
    higher-level engine contracts and are outside this storage type.

    Parameters
    ----------
    node_count : int
        Positive size of the fixed integer node domain ``range(node_count)``.
    expected_density : float, optional
        Edge probability for seeded Erdős-Rényi initialization. It is not a
        preallocation setting or a bound on the realized memory footprint.
    seed : int, optional
        RandomState-compatible seed in ``[0, 2**32 - 1]``. When omitted, the
        graph starts empty and ``expected_density`` is not sampled.

    Examples
    --------
    Create a sparse graph with about ten expected neighbours per node:

    >>> from tnfr.sparse import SparseTNFRGraph
    >>> graph = SparseTNFRGraph(1000, expected_density=0.01, seed=42)
    >>> graph.node_count
    1000
    >>> report = graph.memory_footprint()
    >>> report.per_node_kb < 1.0
    True

    The estimate tracks array payloads and populated entries. Realized
    edges, non-default attributes, cache occupancy, and unreported interpreter
    object overhead determine actual process memory.
    """

    def __init__(
        self,
        node_count: int,
        expected_density: float = 0.1,
        seed: int | None = None,
    ):
        node_count_value = _bounded_integer(
            node_count,
            name="node_count",
            minimum=1,
        )
        density_value = _finite_real_scalar(
            expected_density,
            name="expected_density",
        )
        if not 0.0 <= density_value <= 1.0:
            raise TNFRValueError(
                "expected_density must be in [0, 1]",
                context={"expected_density": expected_density},
            )
        seed_value = (
            None
            if seed is None
            else _bounded_integer(
                seed,
                name="seed",
                minimum=0,
                maximum=2**32 - 1,
            )
        )

        self.node_count = node_count_value
        self.expected_density = density_value
        self.seed = seed_value

        # LIL supports graph-owned edge mutation; computation materializes CSR.
        self.adjacency = sparse.lil_matrix(
            (node_count_value, node_count_value),
            dtype=np.float32,
        )

        # ΔNFR depends on graph topology and phase, so both mutation paths
        # invalidate this cache through the graph-owned attribute store.
        self._dnfr_cache = SparseCache(node_count_value, ttl_steps=10)

        # Compact node attributes
        self.node_attributes = CompactAttributeStore(
            node_count_value, on_theta_change=self._dnfr_cache.clear
        )

        # Initialize with random values if seed provided
        if seed_value is not None:
            self._initialize_random(seed_value)

        logger.info(
            f"Created sparse TNFR graph: {node_count_value} nodes, "
            f"density={density_value:.2f}"
        )

    def _initialize_random(self, seed: int) -> None:
        """Initialize graph with random Erdős-Rényi structure and attributes."""
        rng = np.random.RandomState(seed)

        # Generate random edges efficiently using NetworkX
        import networkx as nx

        G_temp = nx.erdos_renyi_graph(self.node_count, self.expected_density, seed=seed)

        # Copy edges to sparse matrix
        for u, v in G_temp.edges():
            weight = rng.uniform(0.5, 1.0)
            self.adjacency[u, v] = weight
            self.adjacency[v, u] = weight

        # Initialize node attributes
        for node_id in range(self.node_count):
            self.node_attributes.set_epi(node_id, rng.uniform(0.0, 1.0))
            self.node_attributes.set_vf(node_id, rng.uniform(0.5, 1.5))
            self.node_attributes.set_theta(node_id, rng.uniform(0.0, 2 * np.pi))

    def add_edge(self, u: NodeId, v: NodeId, weight: float = 1.0) -> None:
        """Add or replace an undirected edge with positive finite weight.

        Integer self-loops are supported and count as one undirected edge.
        Every successful graph-owned topology mutation invalidates cached
        pressure values.

        Parameters
        ----------
        u, v : NodeId
            Integer node identifiers in ``range(node_count)``.
        weight : float
            Positive coupling weight representable as finite float32.
        """
        u_value = _node_id(u, node_count=self.node_count)
        v_value = _node_id(v, node_count=self.node_count)
        stored_weight = _float32_value(weight, name="weight")
        if stored_weight <= 0.0:
            raise TNFRValueError("weight must be positive")

        self.adjacency[u_value, v_value] = stored_weight
        self.adjacency[v_value, u_value] = stored_weight
        self._dnfr_cache.clear()

    def compute_dnfr_sparse(
        self,
        node_ids: Sequence[NodeId] | None = None,
    ) -> np.ndarray:
        """Compute phase pressure with sparse matrix operations.

        Parameters
        ----------
        node_ids : sequence of NodeId, optional
            Integer nodes to compute, in output order. ``None`` selects all
            nodes. The complete request is validated before cache access.

        Returns
        -------
        numpy.ndarray
            Float32 pressure values in the requested order.
        """
        requested = (
            tuple(range(self.node_count))
            if node_ids is None
            else _node_ids(node_ids, node_count=self.node_count)
        )

        dnfr_values = np.zeros(len(requested), dtype=np.float32)
        uncached_indices: list[int] = []
        uncached_ids: list[int] = []

        for index, node_id in enumerate(requested):
            cached = self._dnfr_cache.get(node_id)
            if cached is not None:
                dnfr_values[index] = cached
            else:
                uncached_indices.append(index)
                uncached_ids.append(node_id)

        if uncached_ids:
            adj_csr = self.adjacency.tocsr()
            if not bool(np.all(np.isfinite(adj_csr.data))) or bool(
                np.any(adj_csr.data <= 0.0)
            ):
                raise TNFRValueError(
                    "adjacency weights must be positive finite values"
                )

            all_phases = self.node_attributes.get_thetas(
                range(self.node_count)
            )
            for index, node_id in zip(
                uncached_indices,
                uncached_ids,
                strict=True,
            ):
                row_start = adj_csr.indptr[node_id]
                row_end = adj_csr.indptr[node_id + 1]
                neighbor_indices = adj_csr.indices[row_start:row_end]

                if len(neighbor_indices) > 0:
                    neighbor_phases = all_phases[neighbor_indices]
                    neighbor_weights = adj_csr.data[row_start:row_end]
                    phase_differences = np.sin(
                        float(all_phases[node_id]) - neighbor_phases
                    )
                    weighted_differences = (
                        neighbor_weights.astype(np.float64)
                        * phase_differences.astype(np.float64)
                    )
                    pressure = float(
                        np.sum(weighted_differences, dtype=np.float64)
                        / len(neighbor_indices)
                    )
                else:
                    pressure = 0.0

                dnfr_values[index] = pressure

            cache_update = dict(
                zip(
                    uncached_ids,
                    dnfr_values[uncached_indices],
                    strict=True,
                )
            )
            self._dnfr_cache.update(cache_update)

        return dnfr_values

    def evolve_sparse(self, dt: float = 0.1, steps: int = 10) -> dict[str, Any]:
        """Evolve graph using sparse operations.

        Applies nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

        Parameters
        ----------
        dt : float
            Time step
        steps : int
            Number of evolution steps

        Returns
        -------
        dict[str, Any]
            Evolution metrics. ``coherence_depi_source`` distinguishes a
            measured last-step nodal rate from the static zero-rate convention
            used when ``steps=0``.
        """
        if isinstance(steps, (bool, np.bool_)) or not isinstance(
            steps, Integral
        ):
            raise TNFRValueError("steps must be a nonnegative integer")
        steps_value = int(steps)
        if steps_value < 0:
            raise TNFRValueError("steps must be a nonnegative integer")
        dt_value = _finite_real_scalar(dt, name="dt")
        if steps_value and dt_value <= 0.0:
            raise TNFRValueError("dt must be positive when steps is nonzero")

        all_node_ids = tuple(range(self.node_count))
        last_dnfr_values: np.ndarray | None = None
        last_depi_values: np.ndarray | None = None

        for _ in range(steps_value):
            dnfr_values = self.compute_dnfr_sparse(all_node_ids)
            vf_values = self.node_attributes.get_vfs(all_node_ids).astype(
                np.float64
            )
            epi_values = self.node_attributes.get_epis(all_node_ids).astype(
                np.float64
            )

            # Compute the declared nodal step in binary64, then preflight every
            # compact float32 result before committing any node in this step.
            with np.errstate(over="ignore", invalid="ignore"):
                depi_values = vf_values * dnfr_values.astype(np.float64)
                new_epis = epi_values + depi_values * dt_value
            if not bool(np.all(np.isfinite(depi_values))) or not bool(
                np.all(np.isfinite(new_epis))
            ):
                raise TNFRValueError("nodal update must remain finite")
            stored_epis = tuple(
                _float32_value(value, name="updated epi")
                for value in new_epis
            )

            for node_id, stored_epi, dnfr in zip(
                all_node_ids,
                stored_epis,
                dnfr_values,
                strict=True,
            ):
                self.node_attributes.set_epi(node_id, float(stored_epi))
                self.node_attributes.set_dnfr(node_id, float(dnfr))

            # Only the final rate is needed for C(t); retaining the full
            # trajectory would defeat the compact representation.
            last_dnfr_values = dnfr_values
            last_depi_values = depi_values
            self._dnfr_cache.step()

        if last_dnfr_values is None:
            # With no evolution there is no observed EPI rate. Report the
            # canonical static-field read-out, which explicitly sets dEPI=0.
            last_dnfr_values = self.compute_dnfr_sparse(all_node_ids)

        coherence_result = self._compute_coherence(
            dnfr_values=last_dnfr_values,
            depi_values=last_depi_values,
            return_means=True,
        )
        if not isinstance(coherence_result, tuple):  # pragma: no cover - typing guard
            raise RuntimeError("sparse coherence read-out did not return components")
        coherence, mean_abs_dnfr, mean_abs_depi = coherence_result

        return {
            "final_coherence": coherence,
            "steps": steps_value,
            "mean_abs_dnfr": mean_abs_dnfr,
            "mean_abs_depi": mean_abs_depi,
            "coherence_depi_source": (
                "last_step_nodal_rate"
                if last_depi_values is not None
                else "static_default_zero"
            ),
        }

    def _compute_coherence(
        self,
        *,
        dnfr_values: Sequence[float] | np.ndarray,
        depi_values: Sequence[float] | np.ndarray | None,
        return_means: bool = False,
    ) -> float | tuple[float, float, float]:
        r"""Compute canonical sparse ``C(t)`` from explicit channel provenance.

        ``depi_values`` is the instantaneous ``νf·ΔNFR`` rate used by the last
        Euler update. Passing ``None`` declares a static field with no observed
        evolution and therefore applies the canonical ``dEPI=0`` convention.
        """
        mean_abs_dnfr = _mean_absolute_channel(
            dnfr_values,
            expected_size=self.node_count,
            name="dnfr_values",
        )
        mean_abs_depi = (
            0.0
            if depi_values is None
            else _mean_absolute_channel(
                depi_values,
                expected_size=self.node_count,
                name="depi_values",
            )
        )
        coherence = float(structural_coherence(mean_abs_dnfr, mean_abs_depi))
        if return_means:
            return coherence, mean_abs_dnfr, mean_abs_depi
        return coherence

    def memory_footprint(self) -> MemoryReport:
        """Estimate tracked array payload and populated-entry memory.

        The estimate excludes Python and SciPy object headers, allocator
        fragmentation, and imported-module memory.

        Returns
        -------
        MemoryReport
            Component estimate with explicit byte totals.
        """
        # CSR exposes its numeric array payloads directly.
        adj_csr = self.adjacency.tocsr()
        adjacency_memory = (
            adj_csr.data.nbytes + adj_csr.indices.nbytes + adj_csr.indptr.nbytes
        )

        attributes_memory = self.node_attributes.memory_usage()
        cache_memory = self._dnfr_cache.memory_usage()

        total_memory = adjacency_memory + attributes_memory + cache_memory
        memory_per_node = total_memory / self.node_count

        return MemoryReport(
            total_mb=total_memory / (1024 * 1024),
            per_node_kb=memory_per_node / 1024,
            breakdown={
                "adjacency": adjacency_memory,
                "attributes": attributes_memory,
                "caches": cache_memory,
            },
        )

    def number_of_edges(self) -> int:
        """Return the undirected edge count, counting each self-loop once."""
        diagonal_count = int(np.count_nonzero(self.adjacency.diagonal()))
        off_diagonal_count = int(self.adjacency.nnz) - diagonal_count
        return diagonal_count + off_diagonal_count // 2
