"""Exact-state cache for TNFR structural fields.

Entries are keyed by every graph channel consumed by the fields and C(t), and store the
canonical coherence C(t) separately from the Kuramoto phase-synchronization
order parameter. Cryptographic signatures are used only for equality; no field
interpolation is inferred from hash similarity.
"""

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from numbers import Integral
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..constants.operational import STRUCT_CACHE_INTERPOLATE_CANONICAL
from ..mathematics.unified_numerical import np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..utils.cache import _compute_dependency_hash

# Import Physics Fields
try:
    from ..physics.fields import (
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        estimate_coherence_length,
    )

    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False


@dataclass
class StructuralCacheEntry:
    """Cache entry for structural field computations."""

    phi_s: dict[Any, float] = field(default_factory=dict)
    grad_phi: dict[Any, float] = field(default_factory=dict)
    k_phi: dict[Any, float] = field(default_factory=dict)
    xi_c: float = 0.0
    coherence: float = 0.0
    phase_sync: float = 0.0
    state_time: float | None = None
    created_at: float = 0.0
    topology_hash: str = ""
    spectral_basis_signature: str = ""
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    coordination_nodes: list[Any] = field(default_factory=list)

    @property
    def timestamp(self) -> float | None:
        """Compatibility alias for graph-state time, never creation time."""

        return self.state_time


@dataclass
class ResonancePattern:
    """Cached resonance pattern for frequency optimization."""

    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    pattern_hash: str
    usage_count: int = 0


class StructuralCoherenceCache:
    """
    Specialized cache for TNFR structural computations.

    Leverages the mathematical structure of structural fields to
    provide intelligent caching with dependency tracking.
    """

    def __init__(self, max_entries: int = 500, enable_interpolation: bool = False):
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral):
            raise ValueError("max_entries must be a positive integer")
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = int(max_entries)
        self.enable_interpolation = enable_interpolation
        self._structural_cache: OrderedDict[str, StructuralCacheEntry] = OrderedDict()
        self._resonance_cache: dict[str, ResonancePattern] = {}

        # Performance counters
        self.hits = 0
        self.misses = 0
        self.interpolations = 0

        self._fft_cache = None
        self._fft_cache_checked = False

    def get_topology_hash(self, G: Any) -> str:
        """Hash every graph channel consumed by the structural tetrad.

        The historical name is retained for compatibility. This is a full
        structural-state signature: Φ_s depends on ΔNFR, the phase fields on
        theta, and ξ_C/coherence on EPI, νf and pressure. Exact dependency
        hashing avoids stale entries after small changes that rounding used to
        erase.
        """
        if not HAS_NETWORKX or G is None:
            return "empty"
        return _compute_dependency_hash(
            G,
            {
                "graph_topology",
                "node_epi",
                "node_vf",
                "node_phase",
                "node_dnfr",
                "node_depi",
            },
        )

    def get_structural_fields(
        self,
        G: Any,
        force_recompute: bool = False,
        interpolate_threshold: float = STRUCT_CACHE_INTERPOLATE_CANONICAL,  # = 0.1 (operational)
        spectral_basis: Any | None = None,
    ) -> StructuralCacheEntry:
        """
        Return a defensive structural snapshot under an exact state key.

        The interpolation arguments remain for compatibility but no approximate
        reuse occurs without a certified structural distance.
        """
        if not HAS_NETWORKX or not HAS_PHYSICS or G is None:
            return StructuralCacheEntry()

        topology_hash = self.get_topology_hash(G)
        if spectral_basis is None:
            spectral_basis = self._maybe_fetch_spectral_basis(G)

        # Check direct cache hit
        if not force_recompute and topology_hash in self._structural_cache:
            self.hits += 1
            entry = self._structural_cache[topology_hash]
            self._structural_cache.move_to_end(topology_hash)
            self._attach_spectral_basis(entry, spectral_basis, G)
            entry.state_time = self._read_state_time(G)
            return self._copy_entry(entry)

        # Check for interpolation opportunities
        if self.enable_interpolation and not force_recompute:
            interpolated = self._try_interpolate_fields(
                G, topology_hash, interpolate_threshold
            )
            if interpolated is not None:
                self.interpolations += 1
                self._attach_spectral_basis(interpolated, spectral_basis, G)
                return interpolated

        # Compute from scratch
        self.misses += 1
        entry = self._compute_structural_fields(G, topology_hash, spectral_basis)

        # Cache with LRU eviction
        self._cache_with_eviction(topology_hash, entry)

        return self._copy_entry(entry)

    def _compute_structural_fields(
        self, G: Any, topology_hash: str, spectral_basis: Any | None = None
    ) -> StructuralCacheEntry:
        """Compute and validate all structural fields for the graph."""

        if not HAS_PHYSICS:
            return StructuralCacheEntry(topology_hash=topology_hash)

        phi_s = compute_structural_potential(G, alpha=2.0)
        grad_phi = compute_phase_gradient(G)
        k_phi = compute_phase_curvature(G)
        xi_c = estimate_coherence_length(G)

        self._validate_field_map(phi_s, label="structural potential")
        self._validate_field_map(grad_phi, label="phase gradient")
        self._validate_field_map(k_phi, label="phase curvature")
        xi_c_value = float(xi_c)
        if math.isnan(xi_c_value) or xi_c_value < 0.0:
            raise ValueError("coherence length must be non-negative and not NaN")

        from ..metrics.common import compute_coherence

        coherence = float(compute_coherence(G))
        if not math.isfinite(coherence):
            raise ValueError("canonical coherence must be finite")
        phase_sync = self._compute_phase_sync(G)
        entry = StructuralCacheEntry(
            phi_s=phi_s,
            grad_phi=grad_phi,
            k_phi=k_phi,
            xi_c=xi_c_value,
            coherence=coherence,
            phase_sync=phase_sync,
            state_time=self._read_state_time(G),
            created_at=time.time(),
            topology_hash=topology_hash,
        )
        self._attach_spectral_basis(entry, spectral_basis, G)
        return entry

    def register_coordination_nodes(
        self, G: Any, coordination_nodes: list[Any], spectral_basis: Any | None = None
    ) -> None:
        """Register nodes that coordinate cache distribution."""
        if not HAS_NETWORKX or G is None:
            return

        topology_hash = self.get_topology_hash(G)
        entry = self._structural_cache.get(topology_hash)
        if entry is None:
            self.get_structural_fields(
                G, force_recompute=False, spectral_basis=spectral_basis
            )
            entry = self._structural_cache[topology_hash]
        self._attach_spectral_basis(entry, spectral_basis, G)
        entry.coordination_nodes = list(coordination_nodes)

    @staticmethod
    def _copy_entry(entry: StructuralCacheEntry) -> StructuralCacheEntry:
        """Return a defensive snapshot so callers cannot poison the cache."""

        def copied_array(value: np.ndarray | None) -> np.ndarray | None:
            if value is None:
                return None
            result = np.array(value, copy=True)
            result.setflags(write=False)
            return result

        return StructuralCacheEntry(
            phi_s=dict(entry.phi_s),
            grad_phi=dict(entry.grad_phi),
            k_phi=dict(entry.k_phi),
            xi_c=entry.xi_c,
            coherence=entry.coherence,
            phase_sync=entry.phase_sync,
            state_time=entry.state_time,
            created_at=entry.created_at,
            topology_hash=entry.topology_hash,
            spectral_basis_signature=entry.spectral_basis_signature,
            eigenvalues=copied_array(entry.eigenvalues),
            eigenvectors=copied_array(entry.eigenvectors),
            coordination_nodes=list(entry.coordination_nodes),
        )

    def _maybe_fetch_spectral_basis(self, G: Any) -> Any | None:
        """Fetch spectral basis from FFT cache if available."""
        if G is None:
            return None

        fft_cache = self._get_fft_cache()
        if fft_cache is None:
            return None

        try:
            return fft_cache.get_spectral_basis(G)
        except Exception:
            return None

    def _get_fft_cache(self) -> Any | None:
        """Lazily instantiate FFT cache coordinator."""
        if self._fft_cache_checked:
            return self._fft_cache

        try:
            from .fft_cache_coordinator import get_fft_cache_coordinator

            self._fft_cache = get_fft_cache_coordinator()
        except ImportError:
            self._fft_cache = None

        self._fft_cache_checked = True
        return self._fft_cache

    def _attach_spectral_basis(
        self,
        entry: StructuralCacheEntry | None,
        spectral_basis: Any | None,
        G: Any,
    ) -> None:
        """Attach a graph-authenticated full symmetric basis to an entry."""
        if entry is None or spectral_basis is None:
            return

        signature, eigenvalues, eigenvectors = self._validated_spectral_basis(
            G, spectral_basis
        )
        entry.spectral_basis_signature = signature
        entry.eigenvalues = eigenvalues
        entry.eigenvectors = eigenvectors

    def _validated_spectral_basis(
        self, G: Any, spectral_basis: Any
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Authenticate and validate a complete L_sym eigenbasis for the graph."""

        fft_cache = self._get_fft_cache()
        signature_reader = getattr(fft_cache, "_graph_signature", None)
        if not callable(signature_reader):
            raise ValueError("spectral basis authentication is unavailable")
        expected_signature = str(signature_reader(G))
        signature = getattr(spectral_basis, "signature", None)
        if not isinstance(signature, str) or signature != expected_signature:
            raise ValueError(
                "spectral basis signature does not match the live graph and node order"
            )

        def numeric_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
            raw = np.asarray(getattr(spectral_basis, name, None))
            if raw.dtype.kind not in "iuf":
                raise ValueError(f"spectral {name} must be a real numeric array")
            result = np.asarray(raw, dtype=float)
            if result.shape != shape:
                raise ValueError(
                    f"spectral {name} must have shape {shape}, got {result.shape}"
                )
            if not np.all(np.isfinite(result)):
                raise ValueError(f"spectral {name} must be finite")
            return result

        count = len(G)
        eigenvalues = numeric_array("eigenvalues", (count,))
        eigenvectors = numeric_array("eigenvectors", (count, count))

        from ..physics.structural_diffusion import symmetric_normalized_laplacian

        nodes, laplacian = symmetric_normalized_laplacian(G, list(G.nodes()))
        if tuple(nodes) != tuple(G.nodes()):
            raise ValueError("spectral basis node order differs from the live graph")
        identity = np.eye(count, dtype=float)
        if not np.allclose(
            eigenvectors.T @ eigenvectors,
            identity,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError("spectral eigenvectors must form an orthonormal basis")
        residual = np.asarray(laplacian, dtype=float) @ eigenvectors
        if not np.allclose(
            residual,
            eigenvectors * eigenvalues[np.newaxis, :],
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError("spectral basis does not diagonalize the live L_sym")
        if count > 1 and np.any(np.diff(eigenvalues) < -1e-10):
            raise ValueError("spectral eigenvalues must be ordered nondecreasingly")

        eigenvalue_copy = np.array(eigenvalues, copy=True)
        eigenvector_copy = np.array(eigenvectors, copy=True)
        eigenvalue_copy.setflags(write=False)
        eigenvector_copy.setflags(write=False)
        return signature, eigenvalue_copy, eigenvector_copy

    @staticmethod
    def _read_state_time(G: Any) -> float | None:
        """Read optional graph evolution time separately from creation time."""

        value = G.graph.get("_t")
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("graph state time must be a finite real scalar")
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("graph state time must be a finite real scalar") from exc
        if not math.isfinite(result):
            raise ValueError("graph state time must be a finite real scalar")
        return result

    @staticmethod
    def _validate_field_map(values: Any, *, label: str) -> None:
        """Reject non-mapping or non-finite structural-field results."""

        if not isinstance(values, dict):
            raise TypeError(f"{label} must be a node-value mapping")
        for node, value in values.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"{label} at node {node!r} must be finite")

    def _compute_phase_sync(self, G: Any) -> float:
        """Compute the Kuramoto phase-synchronization order parameter."""
        if not HAS_NETWORKX or G is None:
            return 0.0

        phases = []
        for node in G.nodes():
            phase = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
            if not math.isfinite(phase):
                raise ValueError(f"phase at node {node!r} must be finite")
            phases.append(phase)

        if not phases:
            return 0.0

        # Kuramoto order parameter
        phases = np.array(phases)
        z = np.mean(np.exp(1j * phases))
        return float(np.abs(z))

    def _try_interpolate_fields(
        self, G: Any, new_hash: str, threshold: float
    ) -> StructuralCacheEntry | None:
        """Decline interpolation without an explicit structural state metric.

        Cryptographic hash-prefix similarity has no relation to distance in
        EPI, phase, pressure, or topology. Reusing a field snapshot on that
        basis can violate U6 and return stale telemetry, so exact signatures
        are the only accepted cache keys until a physically defined
        interpolation certificate exists.
        """
        del G, new_hash, threshold
        return None

    @staticmethod
    def _resonance_array(values: Any, *, label: str) -> np.ndarray:
        """Return a finite, one-dimensional float64 resonance channel."""

        raw = np.asarray(values)
        if raw.dtype.kind == "b":
            raise ValueError(f"{label} must be a numeric one-dimensional array")
        try:
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a numeric one-dimensional array"
            ) from exc
        if result.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain only finite values")
        detached = np.array(result, copy=True)
        detached.setflags(write=False)
        return detached

    @staticmethod
    def _resonance_pattern_copy(pattern: ResonancePattern) -> ResonancePattern:
        """Return a detached immutable-array view of one cached pattern."""

        def copy_array(values: np.ndarray) -> np.ndarray:
            result = np.array(values, copy=True)
            result.setflags(write=False)
            return result

        return ResonancePattern(
            frequencies=copy_array(pattern.frequencies),
            amplitudes=copy_array(pattern.amplitudes),
            phases=copy_array(pattern.phases),
            pattern_hash=pattern.pattern_hash,
            usage_count=pattern.usage_count,
        )

    def cache_resonance_pattern(
        self, frequencies: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray
    ) -> str:
        """Cache three aligned finite resonance channels and return their key."""

        channels = (
            self._resonance_array(frequencies, label="frequencies"),
            self._resonance_array(amplitudes, label="amplitudes"),
            self._resonance_array(phases, label="phases"),
        )
        if not (channels[0].shape == channels[1].shape == channels[2].shape):
            raise ValueError(
                "frequencies, amplitudes, and phases must have identical shapes"
            )

        digest = hashlib.sha256()
        for channel in channels:
            descriptor = (channel.dtype.str, channel.shape)
            digest.update(repr(descriptor).encode("utf-8"))
            digest.update(np.ascontiguousarray(channel).tobytes())
        pattern_hash = digest.hexdigest()

        self._resonance_cache[pattern_hash] = ResonancePattern(
            frequencies=channels[0],
            amplitudes=channels[1],
            phases=channels[2],
            pattern_hash=pattern_hash,
            usage_count=1,
        )
        if len(self._resonance_cache) > self.max_entries // 2:
            self._evict_resonance_patterns()
        return pattern_hash

    def get_resonance_pattern(self, pattern_hash: str) -> ResonancePattern | None:
        """Return a defensive pattern snapshot and record one internal hit."""

        pattern = self._resonance_cache.get(pattern_hash)
        if pattern is None:
            return None
        pattern.usage_count += 1
        return self._resonance_pattern_copy(pattern)

    def _cache_with_eviction(self, key: str, entry: StructuralCacheEntry) -> None:
        """Cache entry with LRU eviction."""
        self._structural_cache[key] = entry
        self._structural_cache.move_to_end(key)
        while len(self._structural_cache) > self.max_entries:
            self._structural_cache.popitem(last=False)

    def _evict_resonance_patterns(self) -> None:
        """Evict least-used resonance patterns."""
        if not self._resonance_cache:
            return

        # Sort by usage count and keep top 50%
        patterns = sorted(
            self._resonance_cache.items(), key=lambda x: x[1].usage_count, reverse=True
        )
        keep_count = len(patterns) // 2

        new_cache = {}
        for i in range(keep_count):
            key, pattern = patterns[i]
            new_cache[key] = pattern

        self._resonance_cache = new_cache

    def get_cache_stats(self) -> dict[str, Any]:
        """Get caching performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "interpolations": self.interpolations,
            "hit_rate": hit_rate,
            "structural_entries": len(self._structural_cache),
            "resonance_patterns": len(self._resonance_cache),
            "cache_enabled": True,
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._structural_cache.clear()
        self._resonance_cache.clear()
        self.hits = 0
        self.misses = 0
        self.interpolations = 0


# Global cache instance
_global_structural_cache = None


def get_structural_cache() -> StructuralCoherenceCache:
    """Get or create the global structural cache."""
    global _global_structural_cache
    if _global_structural_cache is None:
        _global_structural_cache = StructuralCoherenceCache()
    return _global_structural_cache


def cached_structural_fields(G: Any, **kwargs) -> StructuralCacheEntry:
    """Convenience function for cached structural field computation."""
    cache = get_structural_cache()
    return cache.get_structural_fields(G, **kwargs)


# Decorator for automatic structural field caching
def cache_structural_computation(func):
    """Decorator to automatically cache structural computations."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract graph from arguments (assume first argument)
        if args:
            G = args[0]
            cache = get_structural_cache()

            # Try to use cached fields if the function needs them
            if hasattr(func, "_uses_structural_fields"):
                cached_entry = cache.get_structural_fields(G)
                kwargs["_cached_fields"] = cached_entry

        return func(*args, **kwargs)

    return wrapper
