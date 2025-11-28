"""Tests for cache configuration and telemetry utilities."""

from __future__ import annotations

import logging

import networkx as nx
import pytest

from tnfr.metrics.cache_utils import (
    CacheStats,
    configure_hot_path_caches,
    get_cache_config,
    log_cache_metrics,
)


class TestCacheStats:
    """Test CacheStats aggregate statistics."""

    def test_init_defaults(self):
        """Test CacheStats initialization with defaults."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_accesses == 0
        assert stats.hit_rate == 0.0

    def test_init_with_values(self):
        """Test CacheStats initialization with explicit values."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.total_accesses == 15
        assert abs(stats.hit_rate - (10 / 15)) < 1e-9

    def test_hit_rate_calculation(self):
        """Test hit rate calculation for various scenarios."""
        # Perfect hit rate
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

        # Zero hit rate
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0

        # Mixed hit rate
        stats = CacheStats(hits=75, misses=25)
        assert abs(stats.hit_rate - 0.75) < 1e-9

    def test_merge(self):
        """Test merging statistics from multiple caches."""
        stats1 = CacheStats(hits=10, misses=5, evictions=2)
        stats2 = CacheStats(hits=20, misses=10, evictions=3)

        merged = stats1.merge(stats2)
        assert merged.hits == 30
        assert merged.misses == 15
        assert merged.evictions == 5
        assert merged.total_accesses == 45
        assert abs(merged.hit_rate - (30 / 45)) < 1e-9

    def test_repr(self):
        """Test string representation."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        repr_str = repr(stats)
        assert "CacheStats" in repr_str
        assert "hits=10" in repr_str
        assert "misses=5" in repr_str
        assert "evictions=2" in repr_str
        assert "hit_rate=" in repr_str


class TestGetCacheConfig:
    """Test cache configuration retrieval."""

    def test_empty_config(self):
        """Test retrieving config from graph without cache settings."""
        G = nx.Graph()
        config = get_cache_config(G)
        assert isinstance(config, dict)
        assert len(config) == 0

    def test_existing_config(self):
        """Test retrieving existing cache configuration."""
        G = nx.Graph()
        G.graph["_cache_config"] = {"buffer_max_entries": 256, "trig_cache_size": 512}
        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 256
        assert config["trig_cache_size"] == 512

    def test_invalid_config_type(self):
        """Test handling of invalid config type."""
        G = nx.Graph()
        G.graph["_cache_config"] = "invalid"
        config = get_cache_config(G)
        assert isinstance(config, dict)
        assert len(config) == 0

    def test_custom_key(self):
        """Test using custom configuration key."""
        G = nx.Graph()
        G.graph["my_cache_config"] = {"value": 42}
        config = get_cache_config(G, key="my_cache_config")
        assert config["value"] == 42


class TestConfigureHotPathCaches:
    """Test unified cache configuration interface."""

    def test_configure_buffer_max_entries(self):
        """Test configuring buffer cache capacity."""
        G = nx.Graph()
        configure_hot_path_caches(G, buffer_max_entries=512)
        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 512

    def test_configure_si_chunk_size(self):
        """Test configuring Si computation chunk size."""
        G = nx.Graph()
        configure_hot_path_caches(G, si_chunk_size=2000)
        assert G.graph["SI_CHUNK_SIZE"] == 2000

    def test_configure_trig_cache_size(self):
        """Test configuring trigonometric cache size."""
        G = nx.Graph()
        configure_hot_path_caches(G, trig_cache_size=256)
        config = get_cache_config(G)
        assert config["trig_cache_size"] == 256

    def test_configure_coherence_cache_size(self):
        """Test configuring coherence cache size."""
        G = nx.Graph()
        configure_hot_path_caches(G, coherence_cache_size=128)
        config = get_cache_config(G)
        assert config["coherence_cache_size"] == 128

    def test_configure_multiple_settings(self):
        """Test configuring multiple cache settings simultaneously."""
        G = nx.Graph()
        configure_hot_path_caches(
            G,
            buffer_max_entries=1024,
            si_chunk_size=5000,
            trig_cache_size=512,
            coherence_cache_size=256,
        )

        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 1024
        assert config["trig_cache_size"] == 512
        assert config["coherence_cache_size"] == 256
        assert G.graph["SI_CHUNK_SIZE"] == 5000

    def test_configure_preserves_existing(self):
        """Test that configuration preserves unrelated settings."""
        G = nx.Graph()
        G.graph["_cache_config"] = {"custom_setting": "preserved"}
        configure_hot_path_caches(G, buffer_max_entries=256)

        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 256
        assert config["custom_setting"] == "preserved"

    def test_configure_none_values_ignored(self):
        """Test that None values don't create config entries."""
        G = nx.Graph()
        configure_hot_path_caches(
            G,
            buffer_max_entries=None,
            si_chunk_size=None,
            trig_cache_size=None,
        )
        config = get_cache_config(G)
        assert "buffer_max_entries" not in config
        assert "trig_cache_size" not in config
        # SI_CHUNK_SIZE also should not be set
        assert "SI_CHUNK_SIZE" not in G.graph


class TestLogCacheMetrics:
    """Test cache metrics logging."""

    def test_log_no_manager(self, caplog):
        """Test logging when no cache manager exists."""
        G = nx.Graph()
        with caplog.at_level(logging.INFO):
            log_cache_metrics(G)
        assert "No cache manager found" in caplog.text

    def test_log_with_custom_logger(self, caplog):
        """Test logging with custom logger instance."""
        G = nx.Graph()
        custom_logger = logging.getLogger("test.cache")
        with caplog.at_level(logging.INFO, logger="test.cache"):
            log_cache_metrics(G, logger_instance=custom_logger)
        # Should log with custom logger
        assert any(record.name == "test.cache" for record in caplog.records)

    def test_log_with_custom_level(self, caplog):
        """Test logging at different levels."""
        G = nx.Graph()
        with caplog.at_level(logging.DEBUG):
            log_cache_metrics(G, level=logging.DEBUG)
        # Should have debug-level records
        assert any(record.levelno == logging.DEBUG for record in caplog.records)


class TestCacheUtilsIntegration:
    """Integration tests for cache utilities."""

    def test_config_flow(self):
        """Test complete configuration workflow."""
        G = nx.Graph()

        # Configure caches
        configure_hot_path_caches(
            G,
            buffer_max_entries=512,
            trig_cache_size=256,
        )

        # Retrieve and verify
        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 512
        assert config["trig_cache_size"] == 256

        # Update configuration
        configure_hot_path_caches(G, buffer_max_entries=1024)
        config = get_cache_config(G)
        assert config["buffer_max_entries"] == 1024
        assert config["trig_cache_size"] == 256  # Should be preserved

    def test_stats_aggregation_workflow(self):
        """Test cache statistics aggregation workflow."""
        # Simulate multiple cache regions
        region1 = CacheStats(hits=100, misses=20, evictions=5)
        region2 = CacheStats(hits=80, misses=30, evictions=3)
        region3 = CacheStats(hits=120, misses=10, evictions=2)

        # Aggregate
        total = region1.merge(region2).merge(region3)
        assert total.hits == 300
        assert total.misses == 60
        assert total.evictions == 10
        assert abs(total.hit_rate - (300 / 360)) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
