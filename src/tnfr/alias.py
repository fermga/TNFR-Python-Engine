"""Attribute helpers supporting alias keys.

``AliasAccessor`` provides the main implementation for dealing with
alias-based attribute access. Legacy wrappers ``alias_get`` and
``alias_set`` have been removed; use :func:`get_attr` and
:func:`set_attr` instead.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Callable,
    TypeVar,
    Optional,
    overload,
    Generic,
    Hashable,
    TYPE_CHECKING,
)
from functools import lru_cache

from .value_utils import _convert_value

if TYPE_CHECKING:  # pragma: no cover
    import networkx  # type: ignore[import-untyped]

T = TypeVar("T")

__all__ = (
    "get_attr",
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

    Behaves like :func:`_convert_value` but uses a fixed ``key`` so the log
    message identifies the value as a default.
    """

    return _convert_value(
        default,
        conv,
        strict=strict,
        key="default",
        log_level=log_level,
    )


def _alias_resolve(
    d: dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    """Resolve the first matching key in ``aliases`` from ``d``."""

    sentinel = object()
    value = next(
        (
            v
            for key in aliases
            if key in d
            for ok, v in [
                _convert_value(
                    d[key],
                    conv,
                    strict=strict,
                    key=key,
                    log_level=log_level,
                )
            ]
            if ok
        ),
        sentinel,
    )
    if value is not sentinel:
        return value
    if default is not None:
        ok, value = _convert_default(
            default,
            conv,
            strict=strict,
            log_level=log_level,
        )
        if ok:
            return value
    return None


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

        if not hasattr(self, "_alias_cache"):
            @lru_cache(maxsize=128)
            def _alias_cache(alias_tuple: tuple[str, ...]) -> tuple[str, ...]:
                if not alias_tuple:
                    raise ValueError("'aliases' must contain at least one key")
                if not all(isinstance(a, str) for a in alias_tuple):
                    raise TypeError("'aliases' elements must be strings")
                return alias_tuple

            self._alias_cache = _alias_cache

        aliases = self._alias_cache(tuple(aliases))
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
        return _alias_resolve(
            d,
            aliases,
            conv=conv,
            default=default,
            strict=strict,
            log_level=log_level,
        )

    def set(
        self,
        d: dict[str, Any],
        aliases: Iterable[str],
        value: Any,
        conv: Callable[[Any], T] | None = None,
    ) -> T:
        aliases, conv, _ = self._prepare(aliases, conv)
        val = conv(value)
        key = next((k for k in aliases if k in d), aliases[0])
        d[key] = val
        return val


_float_accessor = AliasAccessor(float, default=0.0)
_str_accessor = AliasAccessor(str, default="")


@overload
def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: float,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], float] | None = None,
) -> float:
    ...


@overload
def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], float] | None = None,
) -> float | None:
    ...


def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: float | None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], float] | None = None,
) -> float | None:
    """Return the value for the first key in ``aliases`` found in ``d``."""

    return _float_accessor.get(
        d,
        aliases,
        default=default,
        strict=strict,
        log_level=log_level,
        conv=conv,
    )


def set_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    conv: Callable[[Any], float] | None = None,
) -> float:
    """Assign ``value`` to the first alias key found in ``d``."""

    return _float_accessor.set(d, aliases, value, conv=conv)


@overload
def get_attr_str(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: str,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], str] | None = None,
) -> str:
    ...


@overload
def get_attr_str(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], str] | None = None,
) -> str | None:
    ...


def get_attr_str(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: str | None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], str] | None = None,
) -> str | None:
    """Return the string value for the first key in ``aliases``."""

    return _str_accessor.get(
        d,
        aliases,
        default=default,
        strict=strict,
        log_level=log_level,
        conv=conv,
    )


def set_attr_str(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    conv: Callable[[Any], str] | None = None,
) -> str:
    """Assign ``value`` to the first alias key in ``d`` as ``str``."""

    return _str_accessor.set(d, aliases, value, conv=conv)


# -------------------------
# Máximos globales con caché
# -------------------------


def recompute_abs_max(
    G: "networkx.Graph", aliases: tuple[str, ...]
) -> tuple[float, Hashable | None]:
    """Recalculate and return ``(max_val, node)`` for ``aliases`` in ``G``."""
    node, max_val = max(
        (
            (n, abs(get_attr(G.nodes[n], aliases, 0.0)))
            for n in G.nodes()
        ),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
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

    maxima = {k: 0.0 for k in alias_map}
    # Micro-optimization: materialize alias_map items once to avoid
    # repeated dictionary lookups during iteration.
    items = list(alias_map.items())
    for _, nd in G.nodes(data=True):
        for key, aliases in items:
            val = abs(get_attr(nd, aliases, 0.0))
            if val > maxima[key]:
                maxima[key] = val
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
        max_val, max_node = recompute_abs_max(G, aliases)
        G.graph[key] = max_val
        G.graph[node_key] = max_node


def set_attr_and_cache(
    G: "networkx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str | None = None,
    extra: Callable[["networkx.Graph", Hashable, float], None] | None = None,
) -> float:
    """Assign ``value`` to node ``n`` and update caches if requested."""

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
    """Assign ``value`` to node ``n`` and update the global maximum."""
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
    from .constants import ALIAS_VF

    set_scalar(G, n, ALIAS_VF, value, cache=cache)


def set_dnfr(G: "networkx.Graph", n: Hashable, value: float) -> None:
    """Set ``ΔNFR`` for node ``n`` and update the global maximum."""
    from .constants import ALIAS_DNFR

    set_scalar(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


def _increment_trig_version(
    G: "networkx.Graph", _: Hashable, __: float
) -> None:
    g = G.graph
    g["_trig_version"] = int(g.get("_trig_version", 0)) + 1
    # Clear cached trigonometric values to avoid stale data. Any existing
    # `_cos_th`, `_sin_th`, or `_thetas` entries are removed so that a fresh
    # cache will be built on the next access.
    g.pop("_cos_th", None)
    g.pop("_sin_th", None)
    g.pop("_thetas", None)



def set_theta(G: "networkx.Graph", n: Hashable, value: float) -> None:
    """Set ``θ`` for node ``n`` and invalidate trig caches."""
    from .constants import ALIAS_THETA

    set_scalar(
        G,
        n,
        ALIAS_THETA,
        value,
        cache=None,
        extra=_increment_trig_version,
    )
