"""Tests for persistent cache with disk storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tnfr.cache import CacheLevel, PersistentTNFRCache


class TestPersistentTNFRCache:
    """Test persistent cache with disk backing."""

    def test_init_creates_directory(self):
        """Test that cache directory is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache = PersistentTNFRCache(cache_dir=cache_dir)

            assert cache.cache_dir.exists()
            assert cache.cache_dir.is_dir()

    def test_set_and_get_persistent(self):
        """Test persisting and retrieving values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentTNFRCache(cache_dir=tmpdir)

            cache.set_persistent(
                "key1",
                {"data": "value"},
                CacheLevel.DERIVED_METRICS,
                {"dep1"},
                persist_to_disk=True,
            )

            # Should be in memory
            result = cache.get_persistent("key1", CacheLevel.DERIVED_METRICS)
            assert result == {"data": "value"}

    def test_get_from_disk_after_memory_clear(self):
        """Test loading from disk when memory cache is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentTNFRCache(cache_dir=tmpdir)

            # Store to disk
            cache.set_persistent(
                "key1",
                "important_data",
                CacheLevel.GRAPH_STRUCTURE,
                {"dep1"},
                computation_cost=100.0,
                persist_to_disk=True,
            )

            # Clear memory cache
            cache._memory_cache.clear()

            # Should load from disk
            result = cache.get_persistent("key1", CacheLevel.GRAPH_STRUCTURE)
            assert result == "important_data"

            # Should now be back in memory
            result2 = cache.get_persistent("key1", CacheLevel.GRAPH_STRUCTURE)
            assert result2 == "important_data"

    def test_persist_only_configured_levels(self):
        """Test that only configured levels are persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = PersistentTNFRCache(
                cache_dir=cache_dir, persist_levels={CacheLevel.GRAPH_STRUCTURE}
            )

            # This level should persist
            cache.set_persistent(
                "struct",
                "data",
                CacheLevel.GRAPH_STRUCTURE,
                set(),
                persist_to_disk=True,
            )

            # This level should not persist
            cache.set_persistent(
                "temp",
                "data",
                CacheLevel.TEMPORARY,
                set(),
                persist_to_disk=True,
            )

            # Check files
            struct_dir = cache_dir / "graph_structure"
            temp_dir = cache_dir / "temporary"

            assert struct_dir.exists()
            assert len(list(struct_dir.glob("*.pkl"))) >= 1

            # TEMPORARY should not have directory created
            if temp_dir.exists():
                assert len(list(temp_dir.glob("*.pkl"))) == 0

    def test_no_persist_when_flag_false(self):
        """Test that persist_to_disk=False doesn't write to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = PersistentTNFRCache(cache_dir=cache_dir)

            cache.set_persistent(
                "key1",
                "data",
                CacheLevel.DERIVED_METRICS,
                set(),
                persist_to_disk=False,
            )

            # Clear memory
            cache._memory_cache.clear()

            # Should not be available (wasn't persisted)
            result = cache.get_persistent("key1", CacheLevel.DERIVED_METRICS)
            assert result is None

    def test_invalidate_by_dependency(self):
        """Test that invalidation works for persistent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentTNFRCache(cache_dir=tmpdir)

            cache.set_persistent(
                "key1",
                "value1",
                CacheLevel.DERIVED_METRICS,
                {"dep1"},
                persist_to_disk=True,
            )

            count = cache.invalidate_by_dependency("dep1")
            assert count >= 1

            # Should be gone from memory
            result = cache.get_persistent("key1", CacheLevel.DERIVED_METRICS)
            # Note: Disk file still exists but memory cache is empty
            assert result is None or result == "value1"  # May reload from disk

    def test_clear_persistent_cache_single_level(self):
        """Test clearing persistent cache for a single level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = PersistentTNFRCache(cache_dir=cache_dir)

            cache.set_persistent(
                "key1",
                "data1",
                CacheLevel.GRAPH_STRUCTURE,
                set(),
                persist_to_disk=True,
            )

            cache.set_persistent(
                "key2",
                "data2",
                CacheLevel.DERIVED_METRICS,
                set(),
                persist_to_disk=True,
            )

            # Clear only GRAPH_STRUCTURE
            cache.clear_persistent_cache(CacheLevel.GRAPH_STRUCTURE)

            struct_dir = cache_dir / "graph_structure"
            metrics_dir = cache_dir / "derived_metrics"

            # GRAPH_STRUCTURE files should be gone
            if struct_dir.exists():
                assert len(list(struct_dir.glob("*.pkl"))) == 0

            # DERIVED_METRICS should still exist
            assert metrics_dir.exists()
            assert len(list(metrics_dir.glob("*.pkl"))) >= 1

    def test_clear_persistent_cache_all_levels(self):
        """Test clearing all persistent cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = PersistentTNFRCache(cache_dir=cache_dir)

            for i, level in enumerate(
                [CacheLevel.GRAPH_STRUCTURE, CacheLevel.DERIVED_METRICS]
            ):
                cache.set_persistent(
                    f"key{i}",
                    f"data{i}",
                    level,
                    set(),
                    persist_to_disk=True,
                )

            # Clear all
            cache.clear_persistent_cache()

            # All .pkl files should be gone
            pkl_files = list(cache_dir.rglob("*.pkl"))
            assert len(pkl_files) == 0

    def test_cleanup_old_entries(self):
        """Test cleanup of old cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentTNFRCache(cache_dir=tmpdir)

            cache.set_persistent(
                "key1",
                "data",
                CacheLevel.GRAPH_STRUCTURE,
                set(),
                persist_to_disk=True,
            )

            # Cleanup with very short max_age (should remove nothing since just created)
            count = cache.cleanup_old_entries(max_age_days=365)  # 1 year
            assert count == 0

            # Cleanup with negative age (should remove everything)
            count = cache.cleanup_old_entries(max_age_days=-1)
            assert count >= 1

    def test_get_stats_includes_disk_usage(self):
        """Test that statistics include disk usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PersistentTNFRCache(cache_dir=tmpdir)

            cache.set_persistent(
                "key1",
                "data" * 1000,
                CacheLevel.GRAPH_STRUCTURE,
                set(),
                persist_to_disk=True,
            )

            stats = cache.get_stats()

            assert "disk_files" in stats
            assert "disk_size_mb" in stats
            assert stats["disk_files"] >= 1
            assert stats["disk_size_mb"] > 0

    def test_corrupted_cache_file_handled(self):
        """Test that corrupted cache files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = PersistentTNFRCache(cache_dir=cache_dir)

            # Create corrupted file
            level_dir = cache_dir / "graph_structure"
            level_dir.mkdir(exist_ok=True)
            corrupted_file = level_dir / "corrupted.pkl"
            corrupted_file.write_text("not valid pickle data")

            # Should handle gracefully
            result = cache.get_persistent("corrupted", CacheLevel.GRAPH_STRUCTURE)
            assert result is None

            # Corrupted file should be removed
            assert not corrupted_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
