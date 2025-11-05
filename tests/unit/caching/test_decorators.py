"""Tests for caching decorators."""

from __future__ import annotations

import pytest

from tnfr.caching import (
    CacheLevel,
    TNFRHierarchicalCache,
    cache_tnfr_computation,
)
from tnfr.caching.decorators import (
    get_global_cache,
    set_global_cache,
    invalidate_function_cache,
)


class TestCacheDecorator:
    """Test @cache_tnfr_computation decorator."""
    
    def setup_method(self):
        """Reset global cache before each test."""
        set_global_cache(None)
        set_global_cache(TNFRHierarchicalCache(max_memory_mb=128))
    
    def test_basic_caching(self):
        """Test that decorator caches results."""
        call_count = 0
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies={'test_dep'},
        )
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - should compute
        result1 = compute(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = compute(5)
        assert result2 == 10
        assert call_count == 1  # Not called again
    
    def test_different_args_cached_separately(self):
        """Test that different arguments create separate cache entries."""
        call_count = 0
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies={'test_dep'},
        )
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = compute(5)
        result2 = compute(10)
        
        assert result1 == 10
        assert result2 == 20
        assert call_count == 2  # Both computed
        
        # Cache hits
        assert compute(5) == 10
        assert compute(10) == 20
        assert call_count == 2  # No new computations
    
    def test_cost_estimator(self):
        """Test that cost estimator is used."""
        @cache_tnfr_computation(
            level=CacheLevel.DERIVED_METRICS,
            dependencies={'node_data'},
            cost_estimator=lambda x: float(x),
        )
        def expensive_compute(x: int) -> int:
            return x * x
        
        expensive_compute(100)
        
        cache = get_global_cache()
        # Check that entry was cached (can't easily verify cost without internals)
        stats = cache.get_stats()
        assert stats['entry_counts']['derived_metrics'] >= 1
    
    def test_custom_cache_instance(self):
        """Test using custom cache instance."""
        custom_cache = TNFRHierarchicalCache(max_memory_mb=64)
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies={'test'},
            cache_instance=custom_cache,
        )
        def compute(x: int) -> int:
            return x + 1
        
        compute(5)
        
        # Should be in custom cache
        assert custom_cache.get_stats()['entry_counts']['temporary'] >= 1
        
        # Should not be in global cache
        global_cache = get_global_cache()
        assert global_cache.get_stats()['entry_counts']['temporary'] == 0
    
    def test_metadata_attached(self):
        """Test that metadata is attached to decorated function."""
        @cache_tnfr_computation(
            level=CacheLevel.NODE_PROPERTIES,
            dependencies={'node_epi', 'node_vf'},
        )
        def get_property(node_id: str) -> float:
            return 0.5
        
        assert hasattr(get_property, '_is_cached')
        assert get_property._is_cached is True
        assert get_property._cache_level == CacheLevel.NODE_PROPERTIES
        assert get_property._cache_dependencies == {'node_epi', 'node_vf'}
    
    def test_invalidate_function_cache(self):
        """Test invalidating cache for specific function."""
        call_count = 0
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies={'dep1', 'dep2'},
        )
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Cache result
        compute(5)
        assert call_count == 1
        
        # Invalidate function's cache
        invalidate_function_cache(compute)
        
        # Should recompute
        compute(5)
        assert call_count == 2
    
    def test_invalidate_non_cached_function_raises(self):
        """Test that invalidating non-cached function raises error."""
        def regular_function():
            return 42
        
        with pytest.raises(ValueError, match="not cached"):
            invalidate_function_cache(regular_function)
    
    def test_kwargs_in_cache_key(self):
        """Test that kwargs are part of cache key."""
        call_count = 0
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies=set(),
        )
        def compute(x: int, multiplier: int = 2) -> int:
            nonlocal call_count
            call_count += 1
            return x * multiplier
        
        result1 = compute(5, multiplier=2)
        result2 = compute(5, multiplier=3)
        result3 = compute(5, multiplier=2)
        
        assert result1 == 10
        assert result2 == 15
        assert result3 == 10
        assert call_count == 2  # Third call was cached
    
    def test_preserves_function_signature(self):
        """Test that decorator preserves function metadata."""
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies=set(),
        )
        def my_function(x: int) -> int:
            """Docstring for my_function."""
            return x
        
        assert my_function.__name__ == "my_function"
        assert "Docstring" in my_function.__doc__


class TestGlobalCache:
    """Test global cache management."""
    
    def test_get_global_cache_creates_default(self):
        """Test that global cache is created on first access."""
        set_global_cache(None)
        cache = get_global_cache()
        assert isinstance(cache, TNFRHierarchicalCache)
    
    def test_set_global_cache(self):
        """Test setting custom global cache."""
        custom = TNFRHierarchicalCache(max_memory_mb=256)
        set_global_cache(custom)
        
        retrieved = get_global_cache()
        assert retrieved is custom
    
    def test_reset_global_cache(self):
        """Test resetting global cache to None."""
        set_global_cache(None)
        cache1 = get_global_cache()
        
        set_global_cache(None)
        cache2 = get_global_cache()
        
        # Should create new instance
        assert cache1 is not cache2


class TestCacheKeyGeneration:
    """Test cache key generation logic."""
    
    def test_different_functions_different_keys(self):
        """Test that different functions get different cache keys."""
        cache = TNFRHierarchicalCache()
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies=set(),
            cache_instance=cache,
        )
        def func1(x: int) -> int:
            return x
        
        @cache_tnfr_computation(
            level=CacheLevel.TEMPORARY,
            dependencies=set(),
            cache_instance=cache,
        )
        def func2(x: int) -> int:
            return x
        
        func1(5)
        func2(5)
        
        # Both should be cached separately
        stats = cache.get_stats()
        assert stats['entry_counts']['temporary'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
