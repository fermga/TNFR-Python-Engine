"""Tests for the named lock registry using weak references."""

import gc
import uuid

from tnfr import locking


def test_lock_registry_shrinks_after_release():
    """Locks without external references disappear from the registry."""
    # Generate a unique lock name to avoid collisions with other tests.
    name = f"lock-{uuid.uuid4()}"
    start = len(locking._locks)

    lock = locking.get_lock(name)
    assert len(locking._locks) == start + 1

    # Remove the strong reference and force garbage collection so the
    # ``WeakValueDictionary`` drops the entry.
    del lock
    gc.collect()

    assert len(locking._locks) == start
    assert name not in locking._locks

