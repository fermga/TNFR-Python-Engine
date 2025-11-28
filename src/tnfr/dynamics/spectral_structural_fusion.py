"""TNFR Spectral-Structural Fusion Engine.

This module unifies FFT arithmetic caches and structural field caches so
spectral information obtained from Laplacian eigendecompositions powers
both advanced FFT operations and physics field evaluations. The fusion is
fully derived from the nodal equation ∂EPI/∂t = νf · ΔNFR(t):

- FFT arithmetic exposes harmonic structure used to predict structural field
  evolution.
- Structural fields Φ_s, |∇φ|, K_φ, ξ_C benefit from cached eigensystems,
  avoiding redundant Laplacian decompositions.
- Centralization engines reuse spectral signatures to coordinate distributed
  cache roles based on spectral centrality and νf hierarchies.

The engine orchestrates cache warm-up, structural field evaluation with
shared bases, and coordination-node registration so that cache topology
emerges naturally from the underlying physics.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:  # Optional dependency for typing and validation only
    import networkx as nx  # noqa: F401
    HAS_NETWORKX = True
except ImportError:  # pragma: no cover - optional at runtime
    HAS_NETWORKX = False

try:
    from .fft_cache_coordinator import get_fft_cache_coordinator
    HAS_FFT_CACHE = True
except ImportError:  # pragma: no cover
    HAS_FFT_CACHE = False

try:
    from .advanced_cache_optimizer import (
        CacheOptimizationStrategy,
        get_cache_optimizer,
    )
    HAS_CACHE_OPTIMIZER = True
except ImportError:  # pragma: no cover
    HAS_CACHE_OPTIMIZER = False

from .structural_cache import StructuralCacheEntry, get_structural_cache


class TNFRSpectralStructuralFusionEngine:
    """Bridge FFT spectral caches with structural field caches."""

    def __init__(self, enable_cache_optimization: bool = True) -> None:
        self.enable_cache_optimization = enable_cache_optimization
        self.structural_cache = get_structural_cache()
        self.fft_cache = get_fft_cache_coordinator() if HAS_FFT_CACHE else None
        self.cache_optimizer = (
            get_cache_optimizer() if (HAS_CACHE_OPTIMIZER and enable_cache_optimization) else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_structural_fields(self, G: Any, *, force_recompute: bool = False) -> StructuralCacheEntry:
        """Return structural fields using shared spectral basis."""
        spectral_basis = self._ensure_spectral_basis(G, force_recompute=force_recompute)
        return self.structural_cache.get_structural_fields(
            G,
            force_recompute=force_recompute,
            spectral_basis=spectral_basis,
        )

    def prewarm_state(self, G: Any) -> StructuralCacheEntry:
        """Alias for :meth:`compute_structural_fields` used by orchestrators."""
        return self.compute_structural_fields(G, force_recompute=False)

    def coordinate_cache_with_central_nodes(
        self,
        G: Any,
        coordination_nodes: Sequence[Any],
        *,
        strategy: str = "spectral",
    ) -> None:
        """Assign coordination nodes as cache anchors and optimize cache state."""
        if not coordination_nodes:
            return

        spectral_basis = self._ensure_spectral_basis(G)
        node_ids = [getattr(node, "node_id", node) for node in coordination_nodes]
        self.structural_cache.register_coordination_nodes(
            G,
            node_ids,
            spectral_basis=spectral_basis,
        )

        if self.cache_optimizer is None:
            return

        strategies = [
            CacheOptimizationStrategy.SPECTRAL_PERSISTENCE,
            CacheOptimizationStrategy.CROSS_ENGINE_SHARING,
        ]
        if strategy == "spectral":
            strategies.append(CacheOptimizationStrategy.PREDICTIVE_PREFETCH)
        else:
            strategies.append(CacheOptimizationStrategy.TEMPORAL_LOCALITY)

        self.cache_optimizer.optimize_cache_strategy(G, strategies)

    def synchronize_structural_and_fft_caches(self, G: Any) -> Optional[str]:
        """Ensure spectral signatures match across caches and return signature."""
        entry = self.compute_structural_fields(G, force_recompute=False)
        return entry.spectral_basis_signature

    def get_cached_spectral_signature(self, G: Any) -> str:
        """Expose cached spectral signature for diagnostics/tests."""
        entry = self.structural_cache.get_structural_fields(G)
        return entry.spectral_basis_signature

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_spectral_basis(self, G: Any, *, force_recompute: bool = False) -> Optional[Any]:
        if self.fft_cache is None or G is None:
            return None

        try:
            return self.fft_cache.get_spectral_basis(G, force_recompute=force_recompute)
        except Exception:
            return None


__all__ = ["TNFRSpectralStructuralFusionEngine"]
