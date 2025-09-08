"""Attribute helpers supporting alias keys.

``AliasAccessor`` provides the main implementation for dealing with
alias-based attribute access. Legacy wrappers ``alias_get`` and
``alias_set`` have been removed; use :func:`get_attr` and
:func:`set_attr` instead.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import (
    Dict,
    Any,
    Callable,
    TypeVar,
    Optional,
    overload,
    Protocol,
    Generic,
    Hashable,
    TYPE_CHECKING,
)
import logging
from functools import lru_cache
from .logging_utils import get_logger

from .constants import ALIAS_VF, ALIAS_DNFR, ALIAS_THETA

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

logger = get_logger(__name__)

T = TypeVar("T")

__all__ = [
    "get_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_attr_with_max",
    "set_vf",
    "set_dnfr",
    "set_theta",
    "recompute_abs_max",
    "multi_recompute_abs_max",
]


@lru_cache(maxsize=128)
def _validate_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and cache ``aliases`` as a tuple of strings."""

    if not isinstance(aliases, tuple):
        raise TypeError("'aliases' must be a tuple of strings")
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    for a in aliases:
        if not isinstance(a, str):
            raise TypeError("'aliases' elements must be strings")
    return aliases


def _alias_resolve(
    d: Dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    """Resolve the first matching key in ``aliases`` from ``d``.

    ``aliases`` must already be validated with :func:`_validate_aliases`.
    """

    errors: list[tuple[str, Exception]] | None = None
    for key in aliases:
        if key not in d:
            continue
        try:
            return conv(d[key])
        except (ValueError, TypeError) as exc:
            if errors is None:
                errors = []
            errors.append((key, exc))
    if default is not None:
        try:
            return conv(default)
        except (ValueError, TypeError) as exc:
            if errors is None:
                errors = []
            errors.append(("default", exc))

    if errors is not None:
        if strict:
            err_msg = "; ".join(f"{k!r}: {e}" for k, e in errors)
            raise ValueError(f"Could not convert values for {err_msg}")
        if log_level is not None:
            lvl = log_level
        else:
            lvl = (
                logging.WARNING
                if any(k == "default" for k, _ in errors)
                else logging.DEBUG
            )
        summary = "; ".join(f"{k!r}: {e}" for k, e in errors)
        logger.log(lvl, "Could not convert values for %s", summary)

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
        aliases = _validate_aliases(tuple(aliases))
        if conv is None:
            conv = self._conv
        if conv is None:
            raise TypeError("'conv' must be provided")
        if default is None:
            default = self._default
        return aliases, conv, default

    def get(
        self,
        d: Dict[str, Any],
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
        d: Dict[str, Any],
        aliases: Iterable[str],
        value: Any,
        conv: Callable[[Any], T] | None = None,
    ) -> T:
        aliases, conv, _ = self._prepare(aliases, conv)
        val = conv(value)
        key = next((k for k in aliases if k in d), aliases[0])
        d[key] = val
        return val


class _Getter(Protocol[T]):
    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Iterable[str],
        default: T = ...,  # noqa: D401 - documented in get_attr
        *,
        strict: bool = False,
        log_level: int | None = None,
        conv: Callable[[Any], T] | None = ...,
    ) -> T: ...

    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Iterable[str],
        default: None = ...,  # noqa: D401 - documented in get_attr
        *,
        strict: bool = False,
        log_level: int | None = None,
        conv: Callable[[Any], T] | None = ...,
    ) -> Optional[T]: ...


class _Setter(Protocol[T]):
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Iterable[str],
        value: Any,
        conv: Callable[[Any], T] | None = ...,
    ) -> T: ...


_float_accessor = AliasAccessor(float, default=0.0)
get_attr, set_attr = _float_accessor.get, _float_accessor.set
_str_accessor = AliasAccessor(str, default="")
get_attr_str, set_attr_str = _str_accessor.get, _str_accessor.set


# -------------------------
# Máximos globales con caché
# -------------------------


def recompute_abs_max(
    G: "nx.Graph", aliases: tuple[str, ...]
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
    G: "nx.Graph", alias_map: Dict[str, tuple[str, ...]]
) -> Dict[str, float]:
    """Return absolute maxima for each entry in ``alias_map``.

    ``G`` is a :class:`networkx.Graph`. ``alias_map`` maps result keys to
    alias tuples. The graph is traversed once and the absolute maximum for
    each alias tuple is recorded. The returned dictionary uses the same
    keys as ``alias_map``.
    """

    maxima = {k: 0.0 for k in alias_map}
    for _, nd in G.nodes(data=True):
        for key, aliases in alias_map.items():
            val = abs(get_attr(nd, aliases, 0.0))
            if val > maxima[key]:
                maxima[key] = val
    return {k: float(v) for k, v in maxima.items()}


def _update_cached_abs_max(
    G: "nx.Graph",
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


def set_attr_with_max(
    G: "nx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str,
) -> None:
    """Assign ``value`` to node ``n`` and update the global maximum."""
    val = float(value)
    set_attr(G.nodes[n], aliases, val)
    _update_cached_abs_max(G, aliases, n, val, key=cache)


def set_vf(
    G: "nx.Graph", n: Hashable, value: float, *, update_max: bool = True
) -> None:
    """Set ``νf`` for node ``n`` and optionally update the global maximum."""
    val = float(value)
    set_attr(G.nodes[n], ALIAS_VF, val)
    if update_max:
        _update_cached_abs_max(G, ALIAS_VF, n, val, key="_vfmax")


def set_dnfr(G: "nx.Graph", n: Hashable, value: float) -> None:
    """Set ``ΔNFR`` for node ``n`` and update the global maximum."""
    set_attr_with_max(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


def set_theta(G: "nx.Graph", n: Hashable, value: float) -> None:
    """Set ``θ`` for node ``n`` and increment the trig cache version."""
    val = float(value)
    set_attr(G.nodes[n], ALIAS_THETA, val)
    g = G.graph
    g["_trig_version"] = int(g.get("_trig_version", 0)) + 1
