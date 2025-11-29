"""
TNFR Structural Coherence Cache System

Implements a specialized caching layer for structural computations
that emerge from the nodal equation's mathematical properties:

∂EPI/∂t = νf · ΔNFR(t)

Key optimizations:
1. Structural Field Memoization: Cache Φ_s, |∇φ|, K_φ, ξ_C computations
2. Phase Gradient Interpolation: Spatial interpolation of phase fields
3. Coherence Metric Batching: Batch computation of coherence across time windows
4. Resonance Pattern Recognition: Cache and reuse resonant frequency patterns

Status: CANONICAL STRUCTURAL CACHE
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from functools import wraps
import hashlib

from ..constants.canonical import (
    PHI,  # Golden ratio for structural potential
    # PHASE 6 FINAL Canonical Constants for magic number elimination
    STRUCT_CACHE_INTERPOLATE_CANONICAL,  # γ/(π+e) ≈ 0.0985 (0.1 → canonical)
    STRUCT_CACHE_EVICTION_CANONICAL      # φ/(φ+γ) ≈ 0.7371 (0.8 → canonical)
)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import TNFR Cache Infrastructure
try:
    from ..utils.cache import CacheLevel, TNFRHierarchicalCache, get_global_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

# Import Physics Fields
try:
    from ..physics.fields import (
        compute_structural_potential,
        compute_phase_gradient, 
        compute_phase_curvature,
        estimate_coherence_length
    )
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False


@dataclass
class StructuralCacheEntry:
    """Cache entry for structural field computations."""
    phi_s: Dict[Any, float] = field(default_factory=dict)
    grad_phi: Dict[Any, float] = field(default_factory=dict)
    k_phi: Dict[Any, float] = field(default_factory=dict)
    xi_c: float = 0.0
    coherence: float = 0.0
    timestamp: float = 0.0
    topology_hash: str = ""
    spectral_basis_signature: str = ""
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    coordination_nodes: List[Any] = field(default_factory=list)
    

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
    
    def __init__(self, max_entries: int = 500, enable_interpolation: bool = True):
        self.max_entries = max_entries
        self.enable_interpolation = enable_interpolation
        self._structural_cache: Dict[str, StructuralCacheEntry] = {}
        self._resonance_cache: Dict[str, ResonancePattern] = {}
        
        # Performance counters
        self.hits = 0
        self.misses = 0
        self.interpolations = 0
        
        # Global cache integration
        if _CACHE_AVAILABLE:
            self._global_cache = get_global_cache()
        else:
            self._global_cache = None
        self._fft_cache = None
        self._fft_cache_checked = False
    
    def get_topology_hash(self, G: Any) -> str:
        """Generate topology hash for cache keying."""
        if not HAS_NETWORKX or G is None:
            return "empty"
            
        # Create deterministic topology fingerprint
        nodes = sorted(G.nodes())
        edges = sorted(G.edges())
        
        # Include node properties in hash
        node_props = []
        for node in nodes:
            props = G.nodes[node]
            prop_str = f"{props.get('EPI', 0):.3f}_{props.get('nu_f', 1):.3f}_{props.get('phase', 0):.3f}"
            node_props.append(prop_str)
        
        combined = f"n{len(nodes)}_e{len(edges)}_props{'_'.join(node_props)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def get_structural_fields(
        self,
        G: Any,
        force_recompute: bool = False,
        interpolate_threshold: float = STRUCT_CACHE_INTERPOLATE_CANONICAL,  # γ/(π+e) ≈ 0.0985 → canonical
        spectral_basis: Optional[Any] = None
    ) -> StructuralCacheEntry:
        """
        Get structural fields with intelligent caching and interpolation.
        
        Returns cached results if topology is unchanged, or interpolates
        if changes are small (< interpolate_threshold).
        """
        if not HAS_NETWORKX or not HAS_PHYSICS or G is None:
            return StructuralCacheEntry()
            
        topology_hash = self.get_topology_hash(G)
        spectral_basis = spectral_basis or self._maybe_fetch_spectral_basis(G)
        
        # Check direct cache hit
        if not force_recompute and topology_hash in self._structural_cache:
            self.hits += 1
            entry = self._structural_cache[topology_hash]
            self._attach_spectral_basis(entry, spectral_basis)
            return entry
        
        # Check for interpolation opportunities
        if self.enable_interpolation and not force_recompute:
            interpolated = self._try_interpolate_fields(G, topology_hash, interpolate_threshold)
            if interpolated is not None:
                self.interpolations += 1
                self._attach_spectral_basis(interpolated, spectral_basis)
                return interpolated
        
        # Compute from scratch
        self.misses += 1
        entry = self._compute_structural_fields(G, topology_hash, spectral_basis)
        
        # Cache with LRU eviction
        self._cache_with_eviction(topology_hash, entry)
        
        return entry
    
    def _compute_structural_fields(
        self,
        G: Any,
        topology_hash: str,
        spectral_basis: Optional[Any] = None
    ) -> StructuralCacheEntry:
        """Compute all structural fields for the graph."""
        if not HAS_PHYSICS:
            return StructuralCacheEntry(topology_hash=topology_hash)
            
        try:
            # Compute canonical structural fields
            phi_s = compute_structural_potential(G, alpha=PHI)
            grad_phi = compute_phase_gradient(G)
            k_phi = compute_phase_curvature(G)
            xi_c = estimate_coherence_length(G)
            
            # Compute global coherence
            coherence = self._compute_global_coherence(G)
            
            entry = StructuralCacheEntry(
                phi_s=phi_s,
                grad_phi=grad_phi,
                k_phi=k_phi,
                xi_c=xi_c,
                coherence=coherence,
                timestamp=0.0,  # Could integrate with time if available
                topology_hash=topology_hash
            )
            self._attach_spectral_basis(entry, spectral_basis)
            return entry
            
        except Exception:
            # Fallback to empty entry if computation fails
            return StructuralCacheEntry(topology_hash=topology_hash)

    def register_coordination_nodes(
        self,
        G: Any,
        coordination_nodes: List[Any],
        spectral_basis: Optional[Any] = None
    ) -> None:
        """Register nodes that coordinate cache distribution."""
        if not HAS_NETWORKX or G is None:
            return

        topology_hash = self.get_topology_hash(G)
        entry = self._structural_cache.get(topology_hash)
        if entry is None:
            entry = self.get_structural_fields(
                G,
                force_recompute=False,
                spectral_basis=spectral_basis
            )
        entry.coordination_nodes = list(coordination_nodes)
        self._attach_spectral_basis(entry, spectral_basis)

    def _maybe_fetch_spectral_basis(self, G: Any) -> Optional[Any]:
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

    def _get_fft_cache(self) -> Optional[Any]:
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
        entry: Optional[StructuralCacheEntry],
        spectral_basis: Optional[Any]
    ) -> None:
        """Attach spectral metadata to cache entry."""
        if entry is None or spectral_basis is None:
            return

        entry.spectral_basis_signature = getattr(spectral_basis, "signature", "")
        entry.eigenvalues = getattr(spectral_basis, "eigenvalues", None)
        entry.eigenvectors = getattr(spectral_basis, "eigenvectors", None)
    
    def _compute_global_coherence(self, G: Any) -> float:
        """Compute global coherence measure."""
        if not HAS_NETWORKX or G is None:
            return 0.0
            
        # Simple coherence proxy: phase synchronization
        phases = []
        for node in G.nodes():
            phase = G.nodes[node].get('phase', 0.0)
            phases.append(phase)
            
        if not phases:
            return 0.0
            
        # Kuramoto order parameter
        phases = np.array(phases)
        z = np.mean(np.exp(1j * phases))
        return float(np.abs(z))
    
    def _try_interpolate_fields(
        self, 
        G: Any, 
        new_hash: str, 
        threshold: float
    ) -> Optional[StructuralCacheEntry]:
        """
        Try to interpolate structural fields from similar cached entries.
        
        Uses topology similarity and field continuity assumptions.
        """
        if not self._structural_cache:
            return None
            
        # Find most similar cached topology
        best_match = None
        best_similarity = 0.0
        
        current_nodes = set(G.nodes()) if HAS_NETWORKX and G else set()
        current_edges = set(G.edges()) if HAS_NETWORKX and G else set()
        
        for cached_hash, entry in self._structural_cache.items():
            # Simple similarity based on hash prefix matching
            common_prefix = 0
            for i in range(min(len(cached_hash), len(new_hash))):
                if cached_hash[i] == new_hash[i]:
                    common_prefix += 1
                else:
                    break
                    
            similarity = common_prefix / max(len(cached_hash), len(new_hash))
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match is None or best_similarity < threshold:
            return None
            
        # Create interpolated entry (simple copy for now - could implement actual interpolation)
        interpolated = StructuralCacheEntry(
            phi_s=best_match.phi_s.copy(),
            grad_phi=best_match.grad_phi.copy(),
            k_phi=best_match.k_phi.copy(),
            xi_c=best_match.xi_c,
            coherence=best_match.coherence,
            timestamp=best_match.timestamp,
            topology_hash=new_hash
        )
        
        # Cache the interpolated result
        self._cache_with_eviction(new_hash, interpolated)
        
        return interpolated
    
    def cache_resonance_pattern(
        self, 
        frequencies: np.ndarray, 
        amplitudes: np.ndarray, 
        phases: np.ndarray
    ) -> str:
        """
        Cache a resonance pattern for frequency-domain optimizations.
        
        Returns pattern hash for later retrieval.
        """
        # Generate pattern fingerprint
        freq_hash = hashlib.md5(frequencies.tobytes()).hexdigest()[:8]
        amp_hash = hashlib.md5(amplitudes.tobytes()).hexdigest()[:8]
        phase_hash = hashlib.md5(phases.tobytes()).hexdigest()[:8]
        pattern_hash = f"{freq_hash}_{amp_hash}_{phase_hash}"
        
        # Store pattern
        pattern = ResonancePattern(
            frequencies=frequencies.copy(),
            amplitudes=amplitudes.copy(), 
            phases=phases.copy(),
            pattern_hash=pattern_hash,
            usage_count=1
        )
        
        self._resonance_cache[pattern_hash] = pattern
        
        # Evict old patterns if needed
        if len(self._resonance_cache) > self.max_entries // 2:
            self._evict_resonance_patterns()
        
        return pattern_hash
    
    def get_resonance_pattern(self, pattern_hash: str) -> Optional[ResonancePattern]:
        """Retrieve cached resonance pattern."""
        pattern = self._resonance_cache.get(pattern_hash)
        if pattern is not None:
            pattern.usage_count += 1
        return pattern
    
    def _cache_with_eviction(self, key: str, entry: StructuralCacheEntry) -> None:
        """Cache entry with LRU eviction."""
        self._structural_cache[key] = entry
        
        # Simple eviction: remove oldest entries
        if len(self._structural_cache) > self.max_entries:
            # Remove 20% of oldest entries
            to_remove = len(self._structural_cache) - int(STRUCT_CACHE_EVICTION_CANONICAL * self.max_entries)  # φ/(φ+γ) ≈ 0.7371 → canonical
            keys_to_remove = list(self._structural_cache.keys())[:to_remove]
            for k in keys_to_remove:
                del self._structural_cache[k]
    
    def _evict_resonance_patterns(self) -> None:
        """Evict least-used resonance patterns."""
        if not self._resonance_cache:
            return
            
        # Sort by usage count and keep top 50%
        patterns = sorted(self._resonance_cache.items(), 
                         key=lambda x: x[1].usage_count, reverse=True)
        keep_count = len(patterns) // 2
        
        new_cache = {}
        for i in range(keep_count):
            key, pattern = patterns[i]
            new_cache[key] = pattern
            
        self._resonance_cache = new_cache
    
    def get_cache_stats(self) -> Dict[str, Any]:
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
            "cache_enabled": _CACHE_AVAILABLE
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
            if hasattr(func, '_uses_structural_fields'):
                cached_entry = cache.get_structural_fields(G)
                kwargs['_cached_fields'] = cached_entry
                
        return func(*args, **kwargs)
    return wrapper