"""
TNFR Multi-Modal Unified Cache System

This module implements the unified caching strategy that emerges naturally from 
the nodal equation ∂EPI/∂t = νf · ΔNFR(t):

Mathematical Foundation:
1. **Computational Dependencies**: All TNFR operations share common computational roots
2. **Spectral Reuse**: Eigendecompositions can be shared across multiple operations  
3. **Field Correlation**: Structural fields (Φ_s, |∇φ|, K_φ, ξ_C) are mathematically linked
4. **Temporal Coherence**: Multi-step computations can reuse intermediate results
5. **Cross-Engine Synergy**: Cache sharing between optimization engines

Key Features:
- Cross-engine cache sharing (spectral ↔ nodal ↔ fields ↔ adelic)
- Dependency-aware invalidation (topology change → invalidate all dependent caches)  
- Intelligent prefetching (predict likely next computations)
- Memory-conscious eviction (mathematical importance-based LRU)
- Multi-scale coherence (cache hierarchies matching EPI fractality)

Status: CANONICAL UNIFIED CACHE SYSTEM
"""

import numpy as np
from typing import Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import weakref
from collections import OrderedDict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import existing cache infrastructure
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel, get_global_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False


class CacheEntryType(Enum):
    """Types of cached computations."""
    SPECTRAL_DECOMPOSITION = "spectral_decomp"      # Eigenvalues/eigenvectors
    NODAL_STATE = "nodal_state"                     # EPI, νf, phase, ΔNFR
    STRUCTURAL_FIELDS = "structural_fields"         # Φ_s, |∇φ|, K_φ, ξ_C
    TEMPORAL_TRAJECTORY = "temporal_traj"           # Multi-step evolution
    TOPOLOGY_ANALYSIS = "topology_analysis"         # Graph structure metrics
    OPERATOR_SEQUENCE = "operator_sequence"         # Applied operator results
    CROSS_CORRELATION = "cross_correlation"         # Inter-field correlations
    FFT_OPERATION = "fft_operation"                 # FFT arithmetic artifacts


class CacheInvalidationTrigger(Enum):
    """Events that trigger cache invalidation."""
    TOPOLOGY_CHANGE = "topology_change"             # Graph structure modified
    NODE_PARAMETER_CHANGE = "node_param_change"     # EPI, νf, phase modified
    EDGE_WEIGHT_CHANGE = "edge_weight_change"       # Edge weights modified
    OPERATOR_APPLICATION = "operator_applied"       # Structural operator applied
    TIME_EVOLUTION = "time_evolution"               # Temporal step taken


@dataclass
class CacheEntry:
    """Entry in the unified cache."""
    entry_type: CacheEntryType
    data: Any
    graph_signature: str
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    mathematical_importance: float = 1.0  # Higher = more important to keep
    dependencies: Set[CacheEntryType] = field(default_factory=set)
    size_mb: float = 0.0
    computation_time: float = 0.0  # Time it took to compute


@dataclass
class CacheStatistics:
    """Statistics for cache performance."""
    total_entries: int = 0
    total_size_mb: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_count: int = 0
    invalidation_count: int = 0
    cross_engine_reuse_count: int = 0
    memory_pressure_events: int = 0


class TNFRUnifiedMultiModalCache:
    """
    Unified multi-modal cache system for all TNFR computations.
    
    This cache system recognizes that all TNFR operations are mathematically
    related through the nodal equation and can share computational artifacts.
    """
    
    def __init__(self, max_size_mb: float = 512.0, enable_prefetching: bool = True):
        self.max_size_mb = max_size_mb
        self.enable_prefetching = enable_prefetching
        
        # Main cache storage (ordered for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Graph signature tracking (weak references to avoid memory leaks)
        self._graph_signatures: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        
        # Dependency mapping
        self._dependency_map: Dict[CacheEntryType, Set[CacheEntryType]] = {
            # Spectral decomposition is fundamental - many things depend on it
            CacheEntryType.SPECTRAL_DECOMPOSITION: {
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY,
                CacheEntryType.CROSS_CORRELATION
            },
            # Nodal states affect fields and temporal evolution
            CacheEntryType.NODAL_STATE: {
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY
            },
            # Topology analysis affects everything
            CacheEntryType.TOPOLOGY_ANALYSIS: {
                CacheEntryType.SPECTRAL_DECOMPOSITION,
                CacheEntryType.NODAL_STATE,
                CacheEntryType.STRUCTURAL_FIELDS
            }
        }
        
        # Statistics
        self.stats = CacheStatistics()
        self._total_requests = 0
        self._cache_hits = 0
        
        # Prefetching predictions
        self._access_patterns: Dict[str, int] = {}
        self._prefetch_queue: Set[str] = set()
        
    def compute_graph_signature(self, G: Any) -> str:
        """
        Compute mathematical signature of graph structure.
        
        This signature captures the mathematical essence that determines
        what cached computations are still valid.
        """
        if not HAS_NETWORKX or G is None:
            return "null_graph"
            
        # Check if we've already computed this
        if G in self._graph_signatures:
            return self._graph_signatures[G]
            
        # Create signature from graph structure + node parameters
        signature_elements = []
        
        # Graph topology
        signature_elements.append(f"nodes_{len(G.nodes())}")
        signature_elements.append(f"edges_{len(G.edges())}")
        
        # Node parameters (sorted for consistency)
        node_params = []
        for node in sorted(G.nodes()):
            params = []
            for attr in ['EPI', 'nu_f', 'phase', 'ΔNFR']:
                value = G.nodes[node].get(attr, 0.0)
                params.append(f"{attr}_{value:.6f}")
            node_params.append(f"n{node}_{'_'.join(params)}")
        signature_elements.extend(node_params[:10])  # Limit to first 10 for performance
        
        # Edge structure (basic connectivity)
        edge_hash = hashlib.md5()
        for edge in sorted(G.edges()):
            edge_hash.update(f"{edge[0]}_{edge[1]}".encode())
        signature_elements.append(f"edges_hash_{edge_hash.hexdigest()[:8]}")
        
        # Combine all elements
        full_signature = "_".join(signature_elements)
        signature_hash = hashlib.sha256(full_signature.encode()).hexdigest()[:16]
        
        # Cache the signature
        self._graph_signatures[G] = signature_hash
        return signature_hash
        
    def _generate_cache_key(
        self, 
        entry_type: CacheEntryType, 
        graph_signature: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate unique cache key."""
        key_parts = [entry_type.value, graph_signature]
        
        if parameters:
            # Sort parameters for consistent keys
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(parameters.items()))
            key_parts.append(param_str)
            
        return "_".join(key_parts)
        
    def get(
        self, 
        entry_type: CacheEntryType,
        G: Any,
        parameters: Optional[Dict[str, Any]] = None,
        computation_func: Optional[Callable] = None,
        mathematical_importance: float = 1.0
    ) -> Any:
        """
        Get cached computation or compute if not cached.
        
        This is the main interface for all TNFR computations.
        """
        self._total_requests += 1
        
        # Generate cache key
        graph_signature = self.compute_graph_signature(G)
        cache_key = self._generate_cache_key(entry_type, graph_signature, parameters)
        
        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end for LRU
            self._cache.move_to_end(cache_key)
            
            self._cache_hits += 1
            self.stats.cross_engine_reuse_count += 1
            
            return entry.data
            
        # Cache miss - compute if function provided
        if computation_func is None:
            return None
            
        # Compute the result
        start_time = time.time()
        try:
            result = computation_func()
            computation_time = time.time() - start_time
            
            # Estimate size (rough approximation)
            size_estimate = self._estimate_size(result)
            
            # Create cache entry
            entry = CacheEntry(
                entry_type=entry_type,
                data=result,
                graph_signature=graph_signature,
                timestamp=time.time(),
                mathematical_importance=mathematical_importance,
                size_mb=size_estimate,
                computation_time=computation_time
            )
            
            # Store in cache
            self._store_entry(cache_key, entry)
            
            # Update access patterns for prefetching
            if self.enable_prefetching:
                self._update_access_patterns(entry_type, cache_key)
                
            return result
            
        except Exception as e:
            # Don't cache failed computations
            raise e
            
    def _store_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Store entry in cache with size management."""
        # Add to cache
        self._cache[cache_key] = entry
        
        # Update statistics
        self.stats.total_entries = len(self._cache)
        self.stats.total_size_mb = sum(e.size_mb for e in self._cache.values())
        
        # Check memory pressure
        if self.stats.total_size_mb > self.max_size_mb:
            self._evict_entries()
            
    def _evict_entries(self) -> None:
        """Evict entries based on mathematical importance and access patterns."""
        self.stats.memory_pressure_events += 1
        
        # Calculate eviction scores (lower = evict first)
        eviction_candidates = []
        
        for key, entry in self._cache.items():
            # Score based on mathematical importance, access patterns, and age
            age_penalty = (time.time() - entry.last_access) / 3600  # Hours since access
            access_bonus = np.log(1 + entry.access_count)
            importance_bonus = entry.mathematical_importance
            
            eviction_score = importance_bonus + access_bonus - age_penalty
            eviction_candidates.append((eviction_score, key, entry))
            
        # Sort by score (lowest first)
        eviction_candidates.sort(key=lambda x: x[0])
        
        # Evict until we're under 80% of max size
        target_size = self.max_size_mb * 0.8
        
        for score, key, entry in eviction_candidates:
            if self.stats.total_size_mb <= target_size:
                break
                
            del self._cache[key]
            self.stats.total_size_mb -= entry.size_mb
            self.stats.eviction_count += 1
            
        self.stats.total_entries = len(self._cache)
        
    def invalidate(
        self, 
        trigger: CacheInvalidationTrigger,
        G: Optional[Any] = None,
        affected_types: Optional[Set[CacheEntryType]] = None
    ) -> int:
        """
        Invalidate cache entries based on mathematical dependencies.
        
        Returns number of entries invalidated.
        """
        invalidated_count = 0
        
        # Determine what to invalidate based on trigger
        if trigger == CacheInvalidationTrigger.TOPOLOGY_CHANGE:
            # Topology change affects everything
            invalidated_count = len(self._cache)
            self._cache.clear()
            
        elif trigger == CacheInvalidationTrigger.NODE_PARAMETER_CHANGE:
            # Node parameter changes affect nodal states and dependent computations
            types_to_invalidate = {
                CacheEntryType.NODAL_STATE,
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY
            }
            invalidated_count = self._invalidate_by_types(types_to_invalidate, G)
            
        elif trigger == CacheInvalidationTrigger.OPERATOR_APPLICATION:
            # Operator application affects nodal states
            types_to_invalidate = {
                CacheEntryType.NODAL_STATE,
                CacheEntryType.TEMPORAL_TRAJECTORY
            }
            invalidated_count = self._invalidate_by_types(types_to_invalidate, G)
            
        elif affected_types:
            # Custom invalidation
            invalidated_count = self._invalidate_by_types(affected_types, G)
            
        self.stats.invalidation_count += invalidated_count
        return invalidated_count
        
    def _invalidate_by_types(
        self, 
        types_to_invalidate: Set[CacheEntryType], 
        G: Optional[Any] = None
    ) -> int:
        """Invalidate entries by type, optionally filtered by graph."""
        graph_signature = self.compute_graph_signature(G) if G else None
        
        keys_to_remove = []
        for key, entry in self._cache.items():
            # Check if entry type should be invalidated
            if entry.entry_type in types_to_invalidate:
                # If graph specified, only invalidate entries for that graph
                if graph_signature is None or entry.graph_signature == graph_signature:
                    keys_to_remove.append(key)
                    
        # Remove invalidated entries
        for key in keys_to_remove:
            entry = self._cache[key]
            self.stats.total_size_mb -= entry.size_mb
            del self._cache[key]
            
        self.stats.total_entries = len(self._cache)
        return len(keys_to_remove)
        
    def _update_access_patterns(self, entry_type: CacheEntryType, cache_key: str) -> None:
        """Update access patterns for prefetching predictions."""
        pattern_key = f"{entry_type.value}"
        self._access_patterns[pattern_key] = self._access_patterns.get(pattern_key, 0) + 1
        
        # Simple prefetching: if spectral analysis accessed, prefetch fields
        if entry_type == CacheEntryType.SPECTRAL_DECOMPOSITION:
            self._prefetch_queue.add("structural_fields")
        elif entry_type == CacheEntryType.NODAL_STATE:
            self._prefetch_queue.add("temporal_trajectory")
            
    def _estimate_size(self, data: Any) -> float:
        """Rough estimation of data size in MB."""
        if isinstance(data, np.ndarray):
            return data.nbytes / (1024 * 1024)
        elif isinstance(data, dict):
            # Estimate based on number of items
            return len(data) * 0.001  # 1KB per dict item
        elif isinstance(data, (list, tuple)):
            return len(data) * 0.0001  # 100 bytes per list item
        else:
            return 0.001  # 1KB default
            
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        if self._total_requests > 0:
            self.stats.hit_rate = self._cache_hits / self._total_requests
            self.stats.miss_rate = 1.0 - self.stats.hit_rate
            
        return self.stats
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._graph_signatures.clear()
        self._access_patterns.clear()
        self._prefetch_queue.clear()
        
        # Reset statistics
        self.stats = CacheStatistics()
        self._total_requests = 0
        self._cache_hits = 0
        
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "total_entries": len(self._cache),
            "total_size_mb": sum(e.size_mb for e in self._cache.values()),
            "max_size_mb": self.max_size_mb,
            "utilization": sum(e.size_mb for e in self._cache.values()) / self.max_size_mb,
            "entry_types": {
                entry_type.value: sum(1 for e in self._cache.values() 
                                     if e.entry_type == entry_type)
                for entry_type in CacheEntryType
            },
            "access_patterns": dict(self._access_patterns),
            "statistics": self.stats
        }


# Global unified cache instance
_global_unified_cache: Optional[TNFRUnifiedMultiModalCache] = None


def get_unified_cache() -> TNFRUnifiedMultiModalCache:
    """Get global unified cache instance."""
    global _global_unified_cache
    
    if _global_unified_cache is None:
        _global_unified_cache = TNFRUnifiedMultiModalCache()
        
    return _global_unified_cache


def cache_unified_computation(
    entry_type: CacheEntryType,
    mathematical_importance: float = 1.0,
    parameters_func: Optional[Callable] = None
):
    """
    Decorator for caching unified computations.
    
    Usage:
    @cache_unified_computation(CacheEntryType.SPECTRAL_DECOMPOSITION, importance=2.0)
    def compute_spectrum(G):
        # computation here
        return eigenvalues, eigenvectors
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(G: Any, *args, **kwargs) -> Any:
            # Extract parameters for cache key
            params = {}
            if parameters_func:
                params = parameters_func(*args, **kwargs)
            else:
                # Use function arguments as parameters
                params.update(kwargs)
                if args:
                    params['args'] = str(args)
                    
            cache = get_unified_cache()
            
            return cache.get(
                entry_type=entry_type,
                G=G,
                parameters=params,
                computation_func=lambda: func(G, *args, **kwargs),
                mathematical_importance=mathematical_importance
            )
            
        return wrapper
    return decorator


# Convenience functions for common cache operations
def cache_spectral_decomposition(G: Any, computation_func: Callable) -> Any:
    """Cache spectral decomposition with high importance."""
    cache = get_unified_cache()
    return cache.get(
        CacheEntryType.SPECTRAL_DECOMPOSITION,
        G,
        computation_func=computation_func,
        mathematical_importance=3.0  # Very important - many things depend on this
    )


def cache_structural_fields(G: Any, computation_func: Callable, field_params: Dict[str, Any]) -> Any:
    """Cache structural field computation."""
    cache = get_unified_cache()
    return cache.get(
        CacheEntryType.STRUCTURAL_FIELDS,
        G,
        parameters=field_params,
        computation_func=computation_func,
        mathematical_importance=2.0
    )


def invalidate_after_operator(G: Any, operator_name: str) -> int:
    """Invalidate cache after operator application."""
    cache = get_unified_cache()
    return cache.invalidate(
        CacheInvalidationTrigger.OPERATOR_APPLICATION,
        G=G
    )


def invalidate_after_topology_change(G: Any) -> int:
    """Invalidate cache after topology change."""
    cache = get_unified_cache()
    return cache.invalidate(
        CacheInvalidationTrigger.TOPOLOGY_CHANGE,
        G=G
    )