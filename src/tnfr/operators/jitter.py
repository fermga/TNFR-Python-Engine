"""Jitter operators for reproducible phase perturbations."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

from ..locking import get_lock
from ..rng import _compute_seed_hash, _sync_cache_size
from ..rng import clear_rng_cache as _clear_rng_cache
from ..rng import resolve_graph_seed, validate_graph_seed, validate_seed
from ..utils import (
    CacheManager,
    InstrumentedLRUCache,
    ScopedCounterCache,
    build_cache_manager,
)
from ..utils.cache import _scoped_node_offset

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodeProtocol

# Retained for callers of the legacy cache API. Runtime jitter progress lives
# on individual nodes, so eviction cannot change a graph's random trajectory.
_JITTER_MAX_ENTRIES = 1024
_JITTER_PROGRESS_KEY = "_rng_jitter_progress"
_JITTER_PROGRESS_LOCK = get_lock("jitter_progress")


@dataclass(frozen=True, slots=True)
class JitterProgress:
    """One immutable position in a node-local deterministic jitter stream."""

    seed: int
    offset: int
    draws: int

    def as_record(self) -> dict[str, int]:
        """Return the JSON-compatible node record used by the runtime."""

        return {
            "seed": self.seed,
            "offset": self.offset,
            "draws": self.draws,
        }


@dataclass(frozen=True, slots=True)
class JitterProposal:
    """Exact jitter value and compare-and-set progress transition."""

    value: float
    progress_before: JitterProgress | None
    progress_after: JitterProgress | None
    draw_index: int | None

    @property
    def advances(self) -> bool:
        """Whether accepting this proposal advances persistent progress."""

        return self.progress_after is not None


class JitterCache:
    """Compatibility container; runtime draw counts are persistent node data."""

    def __init__(
        self,
        max_entries: int = _JITTER_MAX_ENTRIES,
        *,
        manager: CacheManager | None = None,
    ) -> None:
        self._manager = manager or build_cache_manager()
        if not self._manager.has_override("scoped_counter:jitter"):
            self._manager.configure(
                overrides={"scoped_counter:jitter": int(max_entries)}
            )
        self._sequence = ScopedCounterCache(
            "jitter",
            max_entries=None,
            manager=self._manager,
            default_max_entries=int(max_entries),
        )
        self._settings_key = "jitter_settings"
        self._manager.register(
            self._settings_key,
            lambda: {"max_entries": self._sequence.max_entries},
            reset=self._reset_settings,
        )

    def _reset_settings(self, settings: dict[str, Any] | None) -> dict[str, Any]:
        return {"max_entries": self._sequence.max_entries}

    def _refresh_settings(self) -> None:
        self._manager.update(
            self._settings_key,
            lambda _: {"max_entries": self._sequence.max_entries},
        )

    @property
    def manager(self) -> CacheManager:
        """Expose the cache manager backing this cache."""

        return self._manager

    @property
    def seq(self) -> InstrumentedLRUCache[tuple[int, int], int]:
        """Expose the instrumented sequence cache for tests and diagnostics."""

        return self._sequence.cache

    @property
    def lock(self) -> threading.Lock | threading.RLock:
        """Return the lock protecting the sequence cache."""

        return self._sequence.lock

    @property
    def max_entries(self) -> int:
        """Return the maximum number of cached jitter sequences."""

        return self._sequence.max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """set the maximum number of cached jitter sequences."""

        self._sequence.configure(max_entries=int(value))
        self._refresh_settings()

    @property
    def settings(self) -> dict[str, Any]:
        """Return jitter cache settings stored on the manager."""

        return cast(dict[str, Any], self._manager.get(self._settings_key))

    def setup(self, force: bool = False, max_entries: int | None = None) -> None:
        """Ensure jitter cache matches the configured size."""

        self._sequence.configure(force=force, max_entries=max_entries)
        self._refresh_settings()

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""

        _clear_rng_cache()
        self._sequence.clear()
        self._manager.clear(self._settings_key)

    def bump(self, key: tuple[int, int]) -> int:
        """Return current jitter sequence counter for ``key`` and increment it."""

        return self._sequence.bump(key)


class JitterCacheManager:
    """Manager exposing the jitter cache without global reassignment."""

    def __init__(
        self,
        cache: JitterCache | None = None,
        *,
        manager: CacheManager | None = None,
    ) -> None:
        if cache is not None:
            self.cache = cache
            self._manager = cache.manager
        else:
            self._manager = manager or build_cache_manager()
            self.cache = JitterCache(manager=self._manager)

    # Convenience passthrough properties
    @property
    def seq(self) -> InstrumentedLRUCache[tuple[int, int], int]:
        """Expose the underlying instrumented jitter sequence cache."""

        return self.cache.seq

    @property
    def settings(self) -> dict[str, Any]:
        """Return persisted jitter cache configuration."""

        return self.cache.settings

    @property
    def lock(self) -> threading.Lock | threading.RLock:
        """Return the lock associated with the jitter cache."""

        return self.cache.lock

    @property
    def max_entries(self) -> int:
        """Return the maximum number of cached jitter entries."""

        return self.cache.max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """set the maximum number of cached jitter entries."""

        self.cache.max_entries = value

    def setup(self, force: bool = False, max_entries: int | None = None) -> None:
        """Ensure jitter cache matches the configured size.

        ``max_entries`` may be provided to explicitly resize the cache.
        When omitted the existing ``cache.max_entries`` is preserved.
        """

        if max_entries is not None:
            self.cache.setup(force=True, max_entries=max_entries)
        else:
            self.cache.setup(force=force)

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""

        self.cache.clear()

    def bump(self, key: tuple[int, int]) -> int:
        """Return and increment the jitter sequence counter for ``key``."""

        return self.cache.bump(key)


# Lazy manager instance
_JITTER_MANAGER: JitterCacheManager | None = None


def get_jitter_manager() -> JitterCacheManager:
    """Return the singleton jitter manager, initializing on first use."""
    global _JITTER_MANAGER
    if _JITTER_MANAGER is None:
        _JITTER_MANAGER = JitterCacheManager()
        _JITTER_MANAGER.setup(force=True)
    return _JITTER_MANAGER


def reset_jitter_manager() -> None:
    """Reset the global jitter manager (useful for tests)."""
    global _JITTER_MANAGER
    if _JITTER_MANAGER is not None:
        _JITTER_MANAGER.clear()
    _JITTER_MANAGER = None


def _validated_progress(state: Any) -> JitterProgress | None:
    """Return one validated immutable progress value without changing input."""

    if state is None:
        return None
    if isinstance(state, JitterProgress):
        state = state.as_record()
    if not isinstance(state, dict) or set(state) != {"seed", "offset", "draws"}:
        raise ValueError(
            "Invalid _rng_jitter_progress: expected seed, offset, and draws"
        )
    seed = validate_seed(state["seed"], allow_none=False)
    for name in ("offset", "draws"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(
                f"Invalid _rng_jitter_progress: {name} must be a nonnegative integer"
            )
    return JitterProgress(
        seed=seed,
        offset=int(state["offset"]),
        draws=int(state["draws"]),
    )


def propose_jitter_draw(
    amplitude: float,
    *,
    seed: Any = None,
    offset: Any = None,
    progress_state: Any = None,
) -> JitterProposal:
    """Preview the exact next node-local draw without mutating live state.

    Nonzero proposals require an already resolved graph seed and a validated
    node offset. The binary64 sample is generated by the same two-level BLAKE2b
    seed derivation used by the established runtime. A zero-amplitude proposal
    deliberately ignores seed, offset and progress, matching the no-draw path.
    """

    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    if amplitude == 0:
        return JitterProposal(
            value=0.0,
            progress_before=None,
            progress_after=None,
            draw_index=None,
        )

    progress = _validated_progress(progress_state)
    seed_root = validate_seed(seed, allow_none=False)
    if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0:
        raise ValueError("Jitter node offset must be a nonnegative integer")
    offset_value = int(offset)
    draw_index = (
        progress.draws
        if progress is not None
        and progress.seed == seed_root
        and progress.offset == offset_value
        else 0
    )

    node_stream_seed = _compute_seed_hash(seed_root, offset_value)
    draw_seed = _compute_seed_hash(node_stream_seed, draw_index)
    value = random.Random(draw_seed).uniform(-amplitude, amplitude)
    return JitterProposal(
        value=value,
        progress_before=progress,
        progress_after=JitterProgress(
            seed=seed_root,
            offset=offset_value,
            draws=draw_index + 1,
        ),
        draw_index=draw_index,
    )


def commit_jitter_proposal(storage: Any, proposal: JitterProposal) -> None:
    """Commit one current proposal or reject a stale progress transition.

    The caller owns synchronization across proposal construction and commit.
    ``random_jitter`` holds the shared jitter lock across both operations.
    A failed storage write restores the exact prior record when possible.
    """

    if not isinstance(proposal, JitterProposal):
        raise TypeError("proposal must be a JitterProposal")
    if not proposal.advances:
        return
    current = _validated_progress(storage.get(_JITTER_PROGRESS_KEY))
    if current != proposal.progress_before:
        raise RuntimeError("stale jitter proposal: progress changed before commit")
    assert proposal.progress_after is not None

    had_progress = _JITTER_PROGRESS_KEY in storage
    previous_record = storage.get(_JITTER_PROGRESS_KEY)
    try:
        storage[_JITTER_PROGRESS_KEY] = proposal.progress_after.as_record()
    except BaseException as exc:
        try:
            if had_progress:
                storage[_JITTER_PROGRESS_KEY] = previous_record
            else:
                storage.pop(_JITTER_PROGRESS_KEY, None)
        except BaseException as restore_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "Failed to restore _rng_jitter_progress after commit error: "
                    f"{restore_exc!r}"
                )
        raise


def random_jitter(
    node: NodeProtocol,
    amplitude: float,
) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for ``node``.

    The seed depends only on the recorded graph seed, canonical node offset,
    and per-node draw count. Each node's ``_rng_jitter_progress`` attribute
    records its seed, offset, and next draw count. Replacing this constant-size
    record isolates ordinary graph copies and keeps work per draw bounded.
    Cache eviction/clearing never restarts an active stream.

    Replay requires the same node ordering (including ``SORT_NODES`` policy)
    and operations; graph views intentionally share their parent's node data.
    Changing ``RANDOM_SEED`` or a node's offset starts its stream at draw zero.
    Explicit ``stable_node_offsets(graph)`` scopes amortize NodeNX offset
    validation under a caller-owned stable-order contract. Other node
    implementations keep their own offset semantics.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    validate_graph_seed(node)
    if amplitude == 0:
        return propose_jitter_draw(amplitude).value
    with _JITTER_PROGRESS_LOCK:
        storage = node._glyph_storage()
        progress = _validated_progress(storage.get(_JITTER_PROGRESS_KEY))
        seed_root = resolve_graph_seed(node)
        from ..node import NodeNX

        offset = (
            _scoped_node_offset(node.G, node.n) if type(node) is NodeNX else None
        )
        if offset is None:
            offset = node.offset()
        if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0:
            raise ValueError("Jitter node offset must be a nonnegative integer")
        offset = int(offset)
        # Preserve the established cache-configuration side effect while the
        # pure proposal derives the same sample without touching global RNG.
        _sync_cache_size(node)
        proposal = propose_jitter_draw(
            amplitude,
            seed=seed_root,
            offset=offset,
            progress_state=progress,
        )
        commit_jitter_proposal(storage, proposal)
        return proposal.value


__all__ = [
    "JitterProgress",
    "JitterProposal",
    "JitterCache",
    "JitterCacheManager",
    "commit_jitter_proposal",
    "get_jitter_manager",
    "propose_jitter_draw",
    "reset_jitter_manager",
    "random_jitter",
]
