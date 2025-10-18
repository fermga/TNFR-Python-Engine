"""Central cache registry infrastructure for TNFR services."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, MutableMapping

__all__ = ["CacheManager"]


@dataclass
class _CacheEntry:
    factory: Callable[[], Any]
    lock: threading.Lock
    reset: Callable[[Any], Any] | None = None


class CacheManager:
    """Coordinate named caches guarded by per-entry locks."""

    def __init__(self, storage: MutableMapping[str, Any] | None = None) -> None:
        self._storage: MutableMapping[str, Any]
        if storage is None:
            self._storage = {}
        else:
            self._storage = storage
        self._entries: dict[str, _CacheEntry] = {}
        self._registry_lock = threading.RLock()

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
