"""Unit tests for :class:`tnfr.cache.InstrumentedLRUCache`."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from tnfr.cache import CacheManager, InstrumentedLRUCache


class _EventRecorder:
    """Helper that records callback invocations in a deterministic list."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, Any]] = []

    def telemetry(self, label: str) -> Callable[[Any, Any], None]:
        def _callback(key: Any, value: Any) -> None:
            self.events.append((f"telemetry:{label}", key, value))

        return _callback

    def eviction(self, label: str) -> Callable[[Any, Any], None]:
        def _callback(key: Any, value: Any) -> None:
            self.events.append((f"eviction:{label}", key, value))

        return _callback


@pytest.fixture()
def cache_state() -> (
    tuple[InstrumentedLRUCache[str, int], CacheManager, dict[str, str], _EventRecorder]
):
    manager = CacheManager()
    locks: dict[str, str] = {}
    recorder = _EventRecorder()
    cache = InstrumentedLRUCache[
        str,
        int,
    ](
        2,
        manager=manager,
        metrics_key="instrumented",
        telemetry_callbacks=(recorder.telemetry("first"), recorder.telemetry("second")),
        eviction_callbacks=(recorder.eviction("first"), recorder.eviction("second")),
        locks=locks,
    )
    return cache, manager, locks, recorder


def test_eviction_updates_metrics_and_callbacks(
    cache_state: tuple[
        InstrumentedLRUCache[str, int], CacheManager, dict[str, str], _EventRecorder
    ],
) -> None:
    cache, manager, locks, recorder = cache_state
    locks["a"] = "lock-a"
    locks["b"] = "lock-b"

    cache["a"] = 1
    cache["a"] = 2
    cache["b"] = 3

    stats_before = manager.get_metrics("instrumented")
    assert stats_before.hits == 1
    assert stats_before.misses == 2
    assert stats_before.evictions == 0

    cache["c"] = 4

    stats_after = manager.get_metrics("instrumented")
    assert stats_after.hits == 1
    assert stats_after.misses == 3
    assert stats_after.evictions == 1

    assert recorder.events == [
        ("telemetry:first", "a", 2),
        ("telemetry:second", "a", 2),
        ("eviction:first", "a", 2),
        ("eviction:second", "a", 2),
    ]
    assert "a" not in locks
    assert "b" in locks


def test_pop_records_hits_misses_and_lock_cleanup(
    cache_state: tuple[
        InstrumentedLRUCache[str, int], CacheManager, dict[str, str], _EventRecorder
    ],
) -> None:
    cache, manager, locks, recorder = cache_state
    recorder.events.clear()

    locks["x"] = "lock-x"
    cache["x"] = 10
    cache["y"] = 20

    popped = cache.pop("x")
    assert popped == 10
    assert "x" not in locks
    assert recorder.events == [
        ("telemetry:first", "x", 10),
        ("telemetry:second", "x", 10),
        ("eviction:first", "x", 10),
        ("eviction:second", "x", 10),
    ]

    stats = manager.get_metrics("instrumented")
    assert stats.hits == 1  # from updating "x" via pop
    assert stats.misses == 2  # two insertions
    assert stats.evictions == 1

    recorder.events.clear()
    result = cache.pop("missing", 99)
    assert result == 99
    assert recorder.events == []

    stats_after = manager.get_metrics("instrumented")
    assert stats_after.misses == 3
    assert stats_after.evictions == 1

    with pytest.raises(KeyError):
        cache.pop("another-missing")

    final_stats = manager.get_metrics("instrumented")
    assert final_stats.misses == 4
    assert final_stats.evictions == 1


def test_clear_flushes_cache_and_metrics(
    cache_state: tuple[
        InstrumentedLRUCache[str, int], CacheManager, dict[str, str], _EventRecorder
    ],
) -> None:
    cache, manager, locks, recorder = cache_state
    recorder.events.clear()

    locks["m"] = "lock-m"
    locks["n"] = "lock-n"
    cache["m"] = 1
    cache["n"] = 2

    cache.clear()

    assert len(cache) == 0
    assert locks == {}
    assert recorder.events == [
        ("telemetry:first", "m", 1),
        ("telemetry:second", "m", 1),
        ("eviction:first", "m", 1),
        ("eviction:second", "m", 1),
        ("telemetry:first", "n", 2),
        ("telemetry:second", "n", 2),
        ("eviction:first", "n", 2),
        ("eviction:second", "n", 2),
    ]

    stats = manager.get_metrics("instrumented")
    assert stats.evictions == 2


def test_callback_registration_runtime_updates() -> None:
    manager = CacheManager()
    recorder = _EventRecorder()
    cache = InstrumentedLRUCache[str, int](
        2,
        manager=manager,
        metrics_key="instrumented",
    )

    cache.set_telemetry_callbacks(recorder.telemetry("alpha"))
    cache.set_telemetry_callbacks(recorder.telemetry("beta"), append=True)
    cache.set_eviction_callbacks(recorder.eviction("alpha"))
    cache.set_eviction_callbacks(recorder.eviction("beta"), append=True)

    cache["k"] = 1
    cache.pop("k")
    assert recorder.events == [
        ("telemetry:alpha", "k", 1),
        ("telemetry:beta", "k", 1),
        ("eviction:alpha", "k", 1),
        ("eviction:beta", "k", 1),
    ]

    recorder.events.clear()
    cache.set_telemetry_callbacks(recorder.telemetry("gamma"))
    cache.set_eviction_callbacks(None)
    cache["p"] = 5
    cache.pop("p")
    assert recorder.events == [("telemetry:gamma", "p", 5)]


def test_overwrite_hit_tracking_can_be_disabled() -> None:
    manager = CacheManager()
    cache = InstrumentedLRUCache[str, int](
        4,
        manager=manager,
        metrics_key="instrumented",
        count_overwrite_hit=False,
    )

    cache["k"] = 1
    stats_after_insert = manager.get_metrics("instrumented")
    assert stats_after_insert.misses == 1
    assert stats_after_insert.hits == 0

    cache["k"] = 2
    stats_after_overwrite = manager.get_metrics("instrumented")
    assert stats_after_overwrite.misses == 1
    assert stats_after_overwrite.hits == 0
