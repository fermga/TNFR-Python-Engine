"""Central cache registry infrastructure for TNFR services."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, MutableMapping

__all__ = ["CacheManager", "CacheCapacityConfig"]


@dataclass(frozen=True)
class CacheCapacityConfig:
    """Configuration snapshot for cache capacity policies."""

    default_capacity: int | None
    overrides: dict[str, int | None]


@dataclass
class _CacheEntry:
    factory: Callable[[], Any]
    lock: threading.Lock
    reset: Callable[[Any], Any] | None = None


class CacheManager:
    """Coordinate named caches guarded by per-entry locks."""

    _MISSING = object()

    def __init__(
        self,
        storage: MutableMapping[str, Any] | None = None,
        *,
        default_capacity: int | None = None,
        overrides: Mapping[str, int | None] | None = None,
    ) -> None:
        self._storage: MutableMapping[str, Any]
        if storage is None:
            self._storage = {}
        else:
            self._storage = storage
        self._entries: dict[str, _CacheEntry] = {}
        self._registry_lock = threading.RLock()
        self._default_capacity = self._normalise_capacity(default_capacity)
        self._capacity_overrides: dict[str, int | None] = {}
        if overrides:
            self.configure(overrides=overrides)

    @staticmethod
    def _normalise_capacity(value: int | None) -> int | None:
        if value is None:
            return None
        size = int(value)
        if size < 0:
            raise ValueError("capacity must be non-negative or None")
        return size

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        lock_factory: Callable[[], threading.Lock | threading.RLock] | None = None,
        reset: Callable[[Any], Any] | None = None,
        create: bool = True,
    ) -> None:
        """Register ``name`` with ``factory`` and optional lifecycle hooks."""

        if lock_factory is None:
            lock_factory = threading.RLock
        with self._registry_lock:
            entry = self._entries.get(name)
            if entry is None:
                entry = _CacheEntry(factory=factory, lock=lock_factory(), reset=reset)
                self._entries[name] = entry
            else:
                # Update hooks when re-registering the same cache name.
                entry.factory = factory
                entry.reset = reset
        if create:
            self.get(name)

    def configure(
        self,
        *,
        default_capacity: int | None | object = _MISSING,
        overrides: Mapping[str, int | None] | None = None,
        replace_overrides: bool = False,
    ) -> None:
        """Update the cache capacity policy shared by registered entries."""

        with self._registry_lock:
            if default_capacity is not self._MISSING:
                self._default_capacity = self._normalise_capacity(
                    default_capacity if default_capacity is not None else None
                )
            if overrides is not None:
                if replace_overrides:
                    self._capacity_overrides.clear()
                for key, value in overrides.items():
                    self._capacity_overrides[key] = self._normalise_capacity(value)

    def configure_from_mapping(self, config: Mapping[str, Any]) -> None:
        """Load configuration produced by :meth:`export_config`."""

        default = config.get("default_capacity", self._MISSING)
        overrides = config.get("overrides")
        overrides_mapping: Mapping[str, int | None] | None
        overrides_mapping = overrides if isinstance(overrides, Mapping) else None
        self.configure(default_capacity=default, overrides=overrides_mapping)

    def export_config(self) -> CacheCapacityConfig:
        """Return a copy of the current capacity configuration."""

        with self._registry_lock:
            return CacheCapacityConfig(
                default_capacity=self._default_capacity,
                overrides=dict(self._capacity_overrides),
            )

    def get_capacity(
        self,
        name: str,
        *,
        requested: int | None = None,
        fallback: int | None = None,
        use_default: bool = True,
    ) -> int | None:
        """Return capacity for ``name`` considering overrides and defaults."""

        with self._registry_lock:
            override = self._capacity_overrides.get(name, self._MISSING)
            default = self._default_capacity
        if override is not self._MISSING:
            return override
        values: tuple[int | None, ...]
        if use_default:
            values = (requested, default, fallback)
        else:
            values = (requested, fallback)
        for value in values:
            if value is self._MISSING:
                continue
            normalised = self._normalise_capacity(value)
            if normalised is not None:
                return normalised
        return None

    def has_override(self, name: str) -> bool:
        """Return ``True`` if ``name`` has an explicit capacity override."""

        with self._registry_lock:
            return name in self._capacity_overrides

    def get_lock(self, name: str) -> threading.Lock | threading.RLock:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        return entry.lock

    def names(self) -> Iterator[str]:
        with self._registry_lock:
            return iter(tuple(self._entries))

    def get(self, name: str, *, create: bool = True) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            value = self._storage.get(name)
            if create and value is None:
                value = entry.factory()
                self._storage[name] = value
            return value

    def peek(self, name: str) -> Any:
        return self.get(name, create=False)

    def store(self, name: str, value: Any) -> None:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            self._storage[name] = value

    def update(
        self,
        name: str,
        updater: Callable[[Any], Any],
        *,
        create: bool = True,
    ) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            current = self._storage.get(name)
            if create and current is None:
                current = entry.factory()
            new_value = updater(current)
            self._storage[name] = new_value
            return new_value

    def clear(self, name: str | None = None) -> None:
        if name is not None:
            names = (name,)
        else:
            with self._registry_lock:
                names = tuple(self._entries)
        for cache_name in names:
            entry = self._entries.get(cache_name)
            if entry is None:
                continue
            with entry.lock:
                current = self._storage.get(cache_name)
                new_value = None
                if entry.reset is not None:
                    new_value = entry.reset(current)
                if new_value is None:
                    try:
                        new_value = entry.factory()
                    except Exception:
                        self._storage.pop(cache_name, None)
                        continue
                self._storage[cache_name] = new_value
