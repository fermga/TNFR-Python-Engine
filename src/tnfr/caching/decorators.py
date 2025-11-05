"""Decorators for transparent caching of TNFR computations.

This module provides decorator-based caching that integrates seamlessly with
existing TNFR functions, automatically managing cache keys, dependencies,
and invalidation.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from .hierarchical_cache import CacheLevel, TNFRHierarchicalCache

__all__ = ["cache_tnfr_computation", "get_global_cache", "set_global_cache"]

# Global cache instance shared across all decorated functions
_global_cache: Optional[TNFRHierarchicalCache] = None

F = TypeVar("F", bound=Callable[..., Any])


def get_global_cache() -> TNFRHierarchicalCache:
    """Get or create the global TNFR cache instance.
    
    Returns
    -------
    TNFRHierarchicalCache
        The global cache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = TNFRHierarchicalCache(max_memory_mb=512)
    return _global_cache


def set_global_cache(cache: Optional[TNFRHierarchicalCache]) -> None:
    """Set the global cache instance.
    
    Parameters
    ----------
    cache : TNFRHierarchicalCache or None
        The cache instance to use globally, or None to reset to default.
    """
    global _global_cache
    _global_cache = cache


def _generate_cache_key(
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Generate deterministic cache key from function and arguments.
    
    Parameters
    ----------
    func_name : str
        Name of the function being cached.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    
    Returns
    -------
    str
        Cache key string.
    """
    # Build key components
    key_parts = [func_name]
    
    # Add positional args
    for arg in args:
        if hasattr(arg, '__name__'):  # For graph objects, use name
            key_parts.append(f"graph:{arg.__name__}")
        elif hasattr(arg, 'graph'):  # NetworkX graphs have .graph attribute
            # Use graph id for identity
            key_parts.append(f"graph:{id(arg)}")
        else:
            # For simple types, include value
            key_parts.append(str(arg))
    
    # Add keyword args (sorted for consistency)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        key_parts.append(f"{k}={v}")
    
    # Create deterministic hash
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_tnfr_computation(
    level: CacheLevel,
    dependencies: set[str],
    cost_estimator: Optional[Callable[..., float]] = None,
    cache_instance: Optional[TNFRHierarchicalCache] = None,
) -> Callable[[F], F]:
    """Decorator for automatic caching of TNFR computations.
    
    Caches function results based on arguments and invalidates when
    dependencies change. Transparently integrates with existing functions.
    
    Parameters
    ----------
    level : CacheLevel
        Cache level for storing results.
    dependencies : set[str]
        Set of structural properties this computation depends on.
        Examples: {'graph_topology', 'node_epi', 'node_vf', 'node_phase'}
    cost_estimator : callable, optional
        Function that takes same arguments as decorated function and returns
        estimated computational cost as float. Used for eviction priority.
    cache_instance : TNFRHierarchicalCache, optional
        Specific cache instance to use. If None, uses global cache.
    
    Returns
    -------
    callable
        Decorated function with caching.
    
    Examples
    --------
    >>> from tnfr.caching import cache_tnfr_computation, CacheLevel
    >>> @cache_tnfr_computation(
    ...     level=CacheLevel.DERIVED_METRICS,
    ...     dependencies={'node_vf', 'node_phase'},
    ...     cost_estimator=lambda graph, node_id: len(list(graph.neighbors(node_id)))
    ... )
    ... def compute_metric(graph, node_id):
    ...     # Expensive computation
    ...     return 0.85
    
    With custom cache instance:
    
    >>> from tnfr.caching import TNFRHierarchicalCache
    >>> my_cache = TNFRHierarchicalCache(max_memory_mb=256)
    >>> @cache_tnfr_computation(
    ...     level=CacheLevel.NODE_PROPERTIES,
    ...     dependencies={'node_data'},
    ...     cache_instance=my_cache
    ... )
    ... def get_node_property(graph, node_id):
    ...     return graph.nodes[node_id]
    """
    def decorator(func: F) -> F:
        func_name = func.__name__
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache = cache_instance if cache_instance is not None else get_global_cache()
            
            # Generate cache key
            cache_key = _generate_cache_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key, level)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Estimate computational cost
            comp_cost = 1.0
            if cost_estimator is not None:
                try:
                    comp_cost = float(cost_estimator(*args, **kwargs))
                except (TypeError, ValueError):
                    comp_cost = 1.0
            
            # Store in cache
            cache.set(cache_key, result, level, dependencies, comp_cost)
            
            return result
        
        # Attach metadata for introspection
        wrapper._cache_level = level  # type: ignore
        wrapper._cache_dependencies = dependencies  # type: ignore
        wrapper._is_cached = True  # type: ignore
        
        return wrapper  # type: ignore
    
    return decorator


def invalidate_function_cache(func: Callable[..., Any]) -> int:
    """Invalidate cache entries for a specific decorated function.
    
    Parameters
    ----------
    func : callable
        The decorated function whose cache entries should be invalidated.
    
    Returns
    -------
    int
        Number of entries invalidated.
    
    Raises
    ------
    ValueError
        If the function is not decorated with @cache_tnfr_computation.
    """
    if not hasattr(func, '_is_cached'):
        raise ValueError(f"Function {func.__name__} is not cached")
    
    cache = get_global_cache()
    dependencies = getattr(func, '_cache_dependencies', set())
    
    total = 0
    for dep in dependencies:
        total += cache.invalidate_by_dependency(dep)
    
    return total
