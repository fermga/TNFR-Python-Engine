"""Attribute helpers supporting alias keys.

``AliasAccessor`` provides the main implementation for dealing with
alias-based attribute access. Legacy wrappers ``alias_get`` and
``alias_set`` have been removed; use :func:`get_attr` and
:func:`set_attr` instead.
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    TypeVar,
    Optional,
    Generic,
    Hashable,
    TYPE_CHECKING,
)

from functools import lru_cache, partial
from threading import Lock

from .constants import get_aliases
from .value_utils import convert_value

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_THETA = get_aliases("THETA")

if TYPE_CHECKING:  # pragma: no cover
    import networkx  # type: ignore[import-untyped]

T = TypeVar("T")

__all__ = (
    "set_attr_generic",
    "get_attr",
    "collect_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_attr_and_cache",
    "set_attr_with_max",
    "set_scalar",
    "set_vf",
    "set_dnfr",
    "set_theta",
    "recompute_abs_max",
    "multi_recompute_abs_max",
)


def _convert_default(
    default: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Convert ``default`` using ``conv`` with error handling.

    Behaves like :func:`convert_value` but uses a fixed ``key`` so the log
    message identifies the value as a default.
    """

    return convert_value(
        default,
        conv,
        strict=strict,
        key="default",
        log_level=log_level,
    )


@lru_cache(maxsize=128)
def _alias_cache(alias_tuple: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and cache alias tuples.

    ``functools.lru_cache`` protects this function with an internal lock,
    which is sufficient for thread-safe access; no explicit locking is
    required.
    """
    if not alias_tuple:
        raise ValueError("'aliases' must contain at least one key")
    if not all(isinstance(a, str) for a in alias_tuple):
        raise TypeError("'aliases' elements must be strings")
    return alias_tuple


class AliasAccessor(Generic[T]):
    """Helper providing ``get`` and ``set`` for alias-based attributes.

    This class implements all logic for resolving and assigning values
    using alias keys. Helper functions :func:`get_attr` and
    :func:`set_attr` delegate to a module-level instance of this class.
    """

    def __init__(
        self, conv: Callable[[Any], T] | None = None, default: T | None = None
    ) -> None:
        self._conv = conv
        self._default = default
        # expose cache for testing and manual control
        self._alias_cache = _alias_cache
        self._key_cache: dict[tuple[int, tuple[str, ...]], tuple[str, int]] = {}
        self._lock = Lock()

    def _prepare(
        self,
        aliases: Iterable[str],
        conv: Callable[[Any], T] | None,
        default: Optional[T] = None,
    ) -> tuple[tuple[str, ...], Callable[[Any], T], Optional[T]]:
        """Validate ``aliases`` and resolve ``conv`` and ``default``.

        Parameters
        ----------
        aliases:
            Iterable of alias strings. Must not be a single string.
        conv:
            Conversion callable. If ``None``, the accessor's default
            converter is used.
        default:
            Default value to use if no alias is found. If ``None``, the
            accessor's default is used.
        """

        if isinstance(aliases, str) or not isinstance(aliases, Iterable):
            raise TypeError("'aliases' must be a non-string iterable")
        aliases = _alias_cache(tuple(aliases))
        if conv is None:
            conv = self._conv
        if conv is None:
            raise TypeError("'conv' must be provided")
        if default is None:
            default = self._default
        return aliases, conv, default

    def get(
        self,
        d: dict[str, Any],
        aliases: Iterable[str],
        default: Optional[T] = None,
        *,
        strict: bool = False,
        log_level: int | None = None,
        conv: Callable[[Any], T] | None = None,
    ) -> Optional[T]:
        aliases, conv, default = self._prepare(aliases, conv, default)
        cache_key = (id(d), aliases)
        with self._lock:
            cached = self._key_cache.get(cache_key)
        if cached is not None:
            key, size = cached
            if size == len(d) and key in d:
                ok, value = convert_value(
                    d[key], conv, strict=strict, key=key, log_level=log_level
                )
                if ok:
                    return value
            else:
                with self._lock:
                    self._key_cache.pop(cache_key, None)
        for key in aliases:
            if key in d:
                ok, value = convert_value(
                    d[key], conv, strict=strict, key=key, log_level=log_level
                )
                if ok:
                    with self._lock:
                        self._key_cache[cache_key] = (key, len(d))
                    return value
        if default is not None:
            ok, value = _convert_default(
                default, conv, strict=strict, log_level=log_level
            )
            if ok:
                return value
        return None

    def set(
        self,
        d: dict[str, Any],
        aliases: Iterable[str],
        value: Any,
        conv: Callable[[Any], T] | None = None,
    ) -> T:
        aliases, conv, _ = self._prepare(aliases, conv)
        cache_key = (id(d), aliases)
        with self._lock:
            cached = self._key_cache.get(cache_key)
        if cached is not None:
            key, size = cached
            if size == len(d) and key in d:
                d[key] = conv(value)
                return d[key]
            else:
                with self._lock:
                    self._key_cache.pop(cache_key, None)
        key = next((k for k in aliases if k in d), aliases[0])
        val = conv(value)
        d[key] = val
        with self._lock:
            self._key_cache[cache_key] = (key, len(d))
        return val


_generic_accessor: AliasAccessor[Any] = AliasAccessor()


def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: T | None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], T] = float,
) -> T | None:
    """Return the value for the first key in ``aliases`` found in ``d``."""

    return _generic_accessor.get(
        d,
        aliases,
        default=default,
        strict=strict,
        log_level=log_level,
        conv=conv,
    )


def collect_attr(
    G: "networkx.Graph",
    nodes: Iterable[Any],
    aliases: Iterable[str],
    default: float = 0.0,
    *,
    np=None,
):
    """Collect attribute values for ``nodes`` from ``G`` using ``aliases``.

    Parameters
    ----------
    G:
        Graph containing node attribute mappings.
    nodes:
        Iterable of node identifiers to query.
    aliases:
        Sequence of alias keys passed to :func:`get_attr`.
    default:
        Fallback value when no alias is found for a node.
    np:
        Optional NumPy module. When provided, the result is returned as a
        NumPy array of ``float``; otherwise a Python ``list`` is returned.

    Returns
    -------
    list or numpy.ndarray
        Collected attribute values in the same order as ``nodes``.
    """

    if np is not None:
        if nodes is G.nodes:
            size = G.number_of_nodes()
        else:
            try:
                size = len(nodes)  # type: ignore[arg-type]
            except TypeError:
                nodes = list(nodes)
                size = len(nodes)
        return np.fromiter(
            (get_attr(G.nodes[n], aliases, default) for n in nodes),
            float,
            count=size,
        )
    return [get_attr(G.nodes[n], aliases, default) for n in nodes]


def set_attr_generic(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    *,
    conv: Callable[[Any], T],
) -> T:
    """Assign ``value`` to the first alias key found in ``d``."""

    return _generic_accessor.set(d, aliases, value, conv=conv)


set_attr = partial(set_attr_generic, conv=float)


get_attr_str = partial(get_attr, conv=str)
set_attr_str = partial(set_attr_generic, conv=str)


# -------------------------
# Máximos globales con caché
# -------------------------


def recompute_abs_max(
    G: "networkx.Graph", aliases: tuple[str, ...], *, key: str | None = None
) -> tuple[float, Hashable | None]:
    """Recalculate absolute maximum for ``aliases`` in ``G``.

    When ``key`` is provided, the graph caches ``G.graph[key]`` and
    ``G.graph[f"{key}_node"]`` are updated with the new maximum value and the
    node where it occurs.
    """
    node, max_val = max(
        ((n, abs(get_attr(G.nodes[n], aliases, 0.0))) for n in G.nodes()),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    if key is not None:
        G.graph[key] = max_val
        G.graph[f"{key}_node"] = node
    return max_val, node


def multi_recompute_abs_max(
    G: "networkx.Graph", alias_map: dict[str, tuple[str, ...]]
) -> dict[str, float]:
    """Return absolute maxima for each entry in ``alias_map``.

    ``G`` is a :class:`networkx.Graph`. ``alias_map`` maps result keys to
    alias tuples. The graph is traversed once and the absolute maximum for
    each alias tuple is recorded. The returned dictionary uses the same
    keys as ``alias_map``.
    """

    maxima: defaultdict[str, float] = defaultdict(float)
    items = list(alias_map.items())
    for _, nd in G.nodes(data=True):
        maxima.update(
            {
                key: max(maxima[key], abs(get_attr(nd, aliases, 0.0)))
                for key, aliases in items
            }
        )
    return {k: float(v) for k, v in maxima.items()}


def _update_cached_abs_max(
    G: "networkx.Graph",
    aliases: tuple[str, ...],
    n: Hashable,
    value: float,
    *,
    key: str,
) -> None:
    """Update ``G.graph[key]`` and ``G.graph[f"{key}_node"]``."""
    node_key = f"{key}_node"
    val = abs(value)
    cur = float(G.graph.get(key, 0.0))
    cur_node = G.graph.get(node_key)
    if val >= cur:
        G.graph[key] = val
        G.graph[node_key] = n
    elif cur_node == n and val < cur:
        recompute_abs_max(G, aliases, key=key)


def set_attr_and_cache(
    G: "networkx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str | None = None,
    extra: Callable[["networkx.Graph", Hashable, float], None] | None = None,
) -> float:
    """Assign ``value`` to node ``n`` and update caches if requested.

    Cache updates are performed via :func:`recompute_abs_max` when the
    existing maximum becomes invalid.
    """

    val = set_attr(G.nodes[n], aliases, value)
    if cache is not None:
        _update_cached_abs_max(G, aliases, n, val, key=cache)
    if extra is not None:
        extra(G, n, val)
    return val


def set_attr_with_max(
    G: "networkx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str,
) -> None:
    """Assign ``value`` to node ``n`` and update the global maximum.

    This is a convenience wrapper around :func:`set_attr_and_cache`.
    """
    set_attr_and_cache(G, n, aliases, value, cache=cache)


def set_scalar(
    G: "networkx.Graph",
    n: Hashable,
    alias: tuple[str, ...],
    value: float,
    *,
    cache: str | None = None,
    extra: Callable[["networkx.Graph", Hashable, float], None] | None = None,
) -> float:
    """Assign ``value`` to ``alias`` for node ``n`` and update caches."""
    return set_attr_and_cache(G, n, alias, value, cache=cache, extra=extra)


def set_vf(
    G: "networkx.Graph", n: Hashable, value: float, *, update_max: bool = True
) -> None:
    """Set ``νf`` for node ``n`` and optionally update the global maximum."""
    cache = "_vfmax" if update_max else None

    set_scalar(G, n, ALIAS_VF, value, cache=cache)


def set_dnfr(G: "networkx.Graph", n: Hashable, value: float) -> None:
    """Set ``ΔNFR`` for node ``n`` and update the global maximum."""

    set_scalar(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


def _increment_trig_version(
    G: "networkx.Graph", _: Hashable, __: float
) -> None:
    g = G.graph
    g["_trig_version"] = int(g.get("_trig_version", 0)) + 1
    # Clear cached trigonometric values to avoid stale data. Any existing
    # `_cos_th`, `_sin_th`, or `_thetas` entries are removed so that a fresh
    # cache will be built on the next access.
    for k in ("_cos_th", "_sin_th", "_thetas"):
        g.pop(k, None)


def set_theta(G: "networkx.Graph", n: Hashable, value: float) -> None:
    """Set ``θ`` for node ``n`` and invalidate trig caches."""

    set_scalar(
        G,
        n,
        ALIAS_THETA,
        value,
        cache=None,
        extra=_increment_trig_version,
    )
