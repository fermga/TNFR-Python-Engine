"""Utilities for named locks.

This module provides helpers to obtain process-wide ``threading.Lock``
instances identified by name. Locks are created lazily and reused,
allowing different modules to synchronise on shared resources without
redefining locks repeatedly.
"""

from __future__ import annotations

import threading
from weakref import WeakValueDictionary

# Registry of locks by name guarded by ``_REGISTRY_LOCK``.
# Using ``WeakValueDictionary`` ensures that once a lock is no longer
# referenced elsewhere, it is removed from the registry automatically,
# keeping the catalogue aligned with active coherence nodes.
_locks: WeakValueDictionary[str, threading.Lock] = WeakValueDictionary()
_REGISTRY_LOCK = threading.Lock()


def get_lock(name: str) -> threading.Lock:
    """Return a re-usable lock identified by ``name``.

    The same lock object is returned for identical names. Locks are
    created on first use and stored in a process-wide registry.
    """

    with _REGISTRY_LOCK:
        lock = _locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _locks[name] = lock
    return lock


__all__ = ["get_lock"]
