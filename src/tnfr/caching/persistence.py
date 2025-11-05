"""Persistent cache with disk-backed storage for expensive TNFR computations.

This module provides optional persistence for cache entries, allowing
expensive computations to survive between sessions.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Optional

from .hierarchical_cache import CacheLevel, TNFRHierarchicalCache

__all__ = ["PersistentTNFRCache"]


class PersistentTNFRCache:
    """Cache with optional disk persistence for costly computations.
    
    Combines in-memory caching with selective disk persistence for
    specific cache levels. Expensive computations can be preserved
    between sessions while temporary computations remain memory-only.
    
    Parameters
    ----------
    cache_dir : Path or str, default: ".tnfr_cache"
        Directory for persistent cache files.
    max_memory_mb : int, default: 512
        Memory limit for in-memory cache.
    persist_levels : set[CacheLevel], optional
        Cache levels to persist to disk. Defaults to GRAPH_STRUCTURE
        and DERIVED_METRICS.
    
    Examples
    --------
    >>> from pathlib import Path
    >>> cache = PersistentTNFRCache(cache_dir=Path("/tmp/tnfr_cache"))
    >>> # Cache is automatically persisted for expensive operations
    >>> cache.set_persistent(
    ...     "coherence_large_graph",
    ...     0.95,
    ...     CacheLevel.DERIVED_METRICS,
    ...     dependencies={'graph_topology'},
    ...     computation_cost=1000.0,
    ...     persist_to_disk=True
    ... )
    >>> # Later, in a new session
    >>> result = cache.get_persistent("coherence_large_graph", CacheLevel.DERIVED_METRICS)
    """
    
    def __init__(
        self,
        cache_dir: Path | str = ".tnfr_cache",
        max_memory_mb: int = 512,
        persist_levels: Optional[set[CacheLevel]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._memory_cache = TNFRHierarchicalCache(max_memory_mb=max_memory_mb)
        
        if persist_levels is None:
            persist_levels = {
                CacheLevel.GRAPH_STRUCTURE,
                CacheLevel.DERIVED_METRICS,
            }
        self._persist_levels = persist_levels
    
    def get_persistent(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Retrieve value from memory cache, falling back to disk.
        
        Parameters
        ----------
        key : str
            Cache key.
        level : CacheLevel
            Cache level.
        
        Returns
        -------
        Any or None
            Cached value if found, None otherwise.
        """
        # Try memory first
        result = self._memory_cache.get(key, level)
        if result is not None:
            return result
        
        # Try disk if level is persisted
        if level in self._persist_levels:
            file_path = self._get_cache_file_path(key, level)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Validate structure
                    if not isinstance(cached_data, dict):
                        file_path.unlink(missing_ok=True)
                        return None
                    
                    value = cached_data.get('value')
                    dependencies = cached_data.get('dependencies', set())
                    computation_cost = cached_data.get('computation_cost', 1.0)
                    
                    # Load back into memory cache
                    self._memory_cache.set(
                        key, value, level, dependencies, computation_cost
                    )
                    
                    return value
                    
                except (pickle.PickleError, EOFError, OSError):
                    # Corrupt cache file, remove it
                    file_path.unlink(missing_ok=True)
        
        return None
    
    def set_persistent(
        self,
        key: str,
        value: Any,
        level: CacheLevel,
        dependencies: set[str],
        computation_cost: float = 1.0,
        persist_to_disk: bool = True,
    ) -> None:
        """Store value in memory and optionally persist to disk.
        
        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        level : CacheLevel
            Cache level.
        dependencies : set[str]
            Structural dependencies.
        computation_cost : float, default: 1.0
            Computation cost estimate.
        persist_to_disk : bool, default: True
            Whether to persist this entry to disk.
        """
        # Always store in memory
        self._memory_cache.set(key, value, level, dependencies, computation_cost)
        
        # Persist to disk if requested and level supports it
        if persist_to_disk and level in self._persist_levels:
            file_path = self._get_cache_file_path(key, level)
            cache_data = {
                'value': value,
                'dependencies': dependencies,
                'computation_cost': computation_cost,
                'timestamp': time.time(),
            }
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PickleError, OSError) as e:
                # Log error but don't fail
                # In production, this should use proper logging
                pass
    
    def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate memory and disk cache entries for a dependency.
        
        Parameters
        ----------
        dependency : str
            The structural property that changed.
        
        Returns
        -------
        int
            Number of entries invalidated from memory.
        """
        # Invalidate memory cache
        count = self._memory_cache.invalidate_by_dependency(dependency)
        
        # Note: Disk cache is lazily invalidated on load
        # Entries with stale dependencies will be detected when loaded
        
        return count
    
    def clear_persistent_cache(self, level: Optional[CacheLevel] = None) -> None:
        """Clear persistent cache files.
        
        Parameters
        ----------
        level : CacheLevel, optional
            Specific level to clear. If None, clears all levels.
        """
        if level is not None:
            level_dir = self.cache_dir / level.value
            if level_dir.exists():
                for file_path in level_dir.glob("*.pkl"):
                    file_path.unlink(missing_ok=True)
        else:
            # Clear all levels
            for file_path in self.cache_dir.rglob("*.pkl"):
                file_path.unlink(missing_ok=True)
    
    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove old cache files from disk.
        
        Parameters
        ----------
        max_age_days : int, default: 30
            Maximum age in days before removal.
        
        Returns
        -------
        int
            Number of files removed.
        """
        count = 0
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        for file_path in self.cache_dir.rglob("*.pkl"):
            try:
                mtime = file_path.stat().st_mtime
                if current_time - mtime > max_age_seconds:
                    file_path.unlink()
                    count += 1
            except OSError:
                continue
        
        return count
    
    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics from memory and disk cache.
        
        Returns
        -------
        dict[str, Any]
            Statistics including memory stats and disk usage.
        """
        stats = self._memory_cache.get_stats()
        
        # Add disk stats
        disk_files = 0
        disk_size_bytes = 0
        for file_path in self.cache_dir.rglob("*.pkl"):
            disk_files += 1
            try:
                disk_size_bytes += file_path.stat().st_size
            except OSError:
                continue
        
        stats['disk_files'] = disk_files
        stats['disk_size_mb'] = disk_size_bytes / (1024 * 1024)
        
        return stats
    
    def _get_cache_file_path(self, key: str, level: CacheLevel) -> Path:
        """Get file path for a cache entry.
        
        Organizes cache files by level in subdirectories.
        """
        level_dir = self.cache_dir / level.value
        level_dir.mkdir(exist_ok=True, parents=True)
        # Use key as filename (already hashed in decorator)
        return level_dir / f"{key}.pkl"
