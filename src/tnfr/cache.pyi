from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Hashable, TypeVar

from cachetools import LRUCache

from .types import TimingContext

__all__ = [
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "LockMapCleaner",
]

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class CacheCapacityConfig:
    default_capacity: int | None
    overrides: dict[str, int | None]


@dataclass(frozen=True)
class CacheStatistics:
    hits: int = ...
    misses: int = ...
    evictions: int = ...
    total_time: float = ...
    timings: int = ...

    def merge(self, other: CacheStatistics) -> CacheStatistics: ...


class CacheManager:
    _MISSING: ClassVar[object]

    def __init__(
        self,
        storage: MutableMapping[str, Any] | None = ...,
        *,
        default_capacity: int | None = ...,
        overrides: Mapping[str, int | None] | None = ...,
    ) -> None: ...

    @staticmethod
    def _normalise_capacity(value: int | None) -> int | None: ...

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        lock_factory: Callable[[], threading.Lock | threading.RLock] | None = ...,
        reset: Callable[[Any], Any] | None = ...,
        create: bool = ...,
    ) -> None: ...

    def configure(
        self,
        *,
        default_capacity: int | None | object = ...,
        overrides: Mapping[str, int | None] | None = ...,
        replace_overrides: bool = ...,
    ) -> None: ...

    def configure_from_mapping(self, config: Mapping[str, Any]) -> None: ...

    def export_config(self) -> CacheCapacityConfig: ...

    def get_capacity(
        self,
        name: str,
        *,
        requested: int | None = ...,
        fallback: int | None = ...,
        use_default: bool = ...,
    ) -> int | None: ...

    def has_override(self, name: str) -> bool: ...

    def get_lock(self, name: str) -> threading.Lock | threading.RLock: ...

    def names(self) -> Iterator[str]: ...

    def get(self, name: str, *, create: bool = ...) -> Any: ...

    def peek(self, name: str) -> Any: ...

    def store(self, name: str, value: Any) -> None: ...

    def update(
        self,
        name: str,
        updater: Callable[[Any], Any],
        *,
        create: bool = ...,
    ) -> Any: ...

    def clear(self, name: str | None = ...) -> None: ...

    def increment_hit(
        self,
        name: str,
        *,
        amount: int = ...,
        duration: float | None = ...,
    ) -> None: ...

    def increment_miss(
        self,
        name: str,
        *,
        amount: int = ...,
        duration: float | None = ...,
    ) -> None: ...

    def increment_eviction(self, name: str, *, amount: int = ...) -> None: ...

    def record_timing(self, name: str, duration: float) -> None: ...

    def timer(self, name: str) -> TimingContext: ...

    def get_metrics(self, name: str) -> CacheStatistics: ...

    def iter_metrics(self) -> Iterator[tuple[str, CacheStatistics]]: ...

    def aggregate_metrics(self) -> CacheStatistics: ...

    def register_metrics_publisher(
        self, publisher: Callable[[str, CacheStatistics], None]
    ) -> None: ...

    def publish_metrics(
        self,
        *,
        publisher: Callable[[str, CacheStatistics], None] | None = ...,
    ) -> None: ...

    def log_metrics(
        self, logger: logging.Logger, *, level: int = ...
    ) -> None: ...


class LockMapCleaner(Generic[K]):
    def __init__(self, locks: MutableMapping[K, Any]) -> None: ...

    @property
    def locks(self) -> MutableMapping[K, Any]: ...

    def on_remove(self, key: K, _value: Any | None = ...) -> None: ...

    def prune(self, keys: Iterable[K]) -> None: ...

    def clear(self) -> None: ...


class InstrumentedLRUCache(LRUCache[K, V], Generic[K, V]):
    _MISSING: ClassVar[object]

    def __init__(
        self,
        maxsize: int,
        *,
        lock_cleaner: LockMapCleaner[K] | None = ...,
        telemetry: Sequence[tuple[CacheManager, str]] | tuple[CacheManager, str] | None = ...,
        on_evict: Iterable[Callable[[K, V], None]] | None = ...,
        logger: logging.Logger | None = ...,
    ) -> None: ...

    def register_evict_callback(self, callback: Callable[[K, V], None]) -> None: ...

    def register_telemetry(self, manager: CacheManager, metrics_key: str) -> None: ...

    def popitem(self) -> tuple[K, V]: ...

    def pop(self, key: K, default: V | object = ...) -> V: ...

    def __delitem__(self, key: K) -> None: ...

    def clear(self) -> None: ...
