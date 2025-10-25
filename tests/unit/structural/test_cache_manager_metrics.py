"""Telemetry tests for :class:`tnfr.cache.CacheManager`."""

from __future__ import annotations

import logging

import pytest

from tnfr.cache import CacheManager, CacheStatistics, prune_lock_mapping


def test_cache_manager_aggregate_metrics_combines_counters():
    manager = CacheManager()
    manager.register("primary", lambda: object(), create=False)
    manager.increment_hit("primary", amount=2)
    manager.increment_miss("primary", amount=3)

    with manager.timer("telemetry"):
        pass

    primary = manager.get_metrics("primary")
    telemetry = manager.get_metrics("telemetry")
    aggregate = manager.aggregate_metrics()

    assert aggregate == primary.merge(telemetry)
    assert aggregate.hits == 2
    assert aggregate.misses == 3
    assert telemetry.timings == 1
    assert telemetry.total_time > 0


def test_cache_manager_publish_metrics_dispatches_and_handles_errors(caplog: pytest.LogCaptureFixture):
    manager = CacheManager()
    manager.register("primary", lambda: object(), create=False)
    manager.increment_hit("primary")
    manager.increment_miss("primary")

    received: list[tuple[str, CacheStatistics]] = []

    def spy(name: str, stats: CacheStatistics) -> None:
        received.append((name, stats))

    def raiser(name: str, stats: CacheStatistics) -> None:
        raise RuntimeError("boom")

    manager.register_metrics_publisher(spy)
    manager.register_metrics_publisher(raiser)

    with caplog.at_level(logging.ERROR, logger="tnfr.cache"):
        manager.publish_metrics()

    assert received
    assert all(isinstance(name, str) and isinstance(stats, CacheStatistics) for name, stats in received)
    assert any("Cache metrics publisher failed for" in record.getMessage() for record in caplog.records)

    explicit: list[tuple[str, CacheStatistics]] = []

    def collector(name: str, stats: CacheStatistics) -> None:
        explicit.append((name, stats))

    manager.publish_metrics(publisher=collector)
    assert explicit == received


def test_cache_manager_logs_metrics(caplog: pytest.LogCaptureFixture):
    manager = CacheManager()
    manager.register("primary", lambda: object(), create=False)
    manager.increment_hit("primary", amount=5)
    manager.increment_miss("primary", amount=4)
    manager.increment_eviction("primary", amount=2)
    manager.record_timing("primary", 0.5)

    logger = logging.getLogger("tests.cache.metrics")

    with caplog.at_level(logging.INFO, logger="tests.cache.metrics"):
        manager.log_metrics(logger)

    aggregate = manager.aggregate_metrics()
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message == (
        "cache=primary hits=5 misses=4 evictions=2 timings=1 total_time=0.500000"
    )
    assert aggregate.hits == 5
    assert aggregate.misses == 4
    assert pytest.approx(aggregate.total_time, rel=0, abs=1e-9) == 0.5
    assert aggregate.timings == 1


def test_prune_lock_mapping_removes_stale_locks():
    shared_lock = object()
    stale_lock = object()
    cache = {"shared": 1}
    locks = {"shared": shared_lock, "stale": stale_lock}

    prune_lock_mapping(cache, locks)

    assert locks == {"shared": shared_lock}
