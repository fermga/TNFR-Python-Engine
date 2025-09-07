"""Attribute helpers supporting alias keys."""

from __future__ import annotations
from typing import (
    Sequence,
    Dict,
    Any,
    Callable,
    TypeVar,
    Optional,
    overload,
    Protocol,
)
import logging
from functools import lru_cache, partial

from .value_utils import _convert_value
from .constants import ALIAS_VF, ALIAS_DNFR

T = TypeVar("T")

__all__ = [
    "alias_get",
    "alias_set",
    "get_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_attr_with_max",
    "set_vf",
    "set_dnfr",
    "recompute_abs_max",
    "multi_recompute_abs_max",
]


@lru_cache(maxsize=None)
def _validate_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Return ``aliases`` as a validated tuple of strings."""
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("'aliases' must be a non-string sequence")
    seq = tuple(aliases)
    if not seq or any(not isinstance(a, str) for a in seq):
        if not seq:
            raise ValueError("'aliases' must contain at least one key")
        raise TypeError("'aliases' elements must be strings")
    return seq


def _alias_resolve(
    d: Dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    aliases = _validate_aliases(aliases)
    ok_def = False
    def_val = None
    if default is not None:
        ok_def, def_val = _convert_value(
            default,
            conv,
            strict=strict,
            key="default",
            log_level=logging.WARNING if not strict else log_level,
        )
    for key in aliases:
        if key in d:
            ok, val = _convert_value(
                d[key], conv, strict=strict, key=key, log_level=log_level
            )
            if ok:
                return val
    if default is None:
        return None
    return def_val if ok_def else None


@overload
def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: T,
    strict: bool = False,
    log_level: int | None = None,
) -> T: ...


@overload
def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: None = ...,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]: ...


def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    """Return the value for the first existing key in ``aliases``."""
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("'aliases' must be a non-string sequence")
    try:
        hash(aliases)
    except TypeError as exc:  # pragma: no cover - defensive programming
        raise TypeError("'aliases' must be a hashable sequence") from exc
    return _alias_resolve(
        d,
        aliases,
        conv=conv,
        default=default,
        strict=strict,
        log_level=log_level,
    )


def alias_set(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Assign ``value`` converted to the first available key in ``aliases``."""
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("'aliases' must be a non-string sequence")
    try:
        hash(aliases)
    except TypeError as exc:  # pragma: no cover - defensive programming
        raise TypeError("'aliases' must be a hashable sequence") from exc
    aliases = _validate_aliases(aliases)
    _, val = _convert_value(value, conv, strict=True)
    key = next((k for k in aliases if k in d), aliases[0])
    d[key] = val
    return val


class _Getter(Protocol[T]):
    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: T = ...,  # noqa: D401 - documented in alias_get
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> T: ...

    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: None = ...,  # noqa: D401 - documented in alias_get
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]: ...


class _Setter(Protocol[T]):
    def __call__(
        self, d: Dict[str, Any], aliases: Sequence[str], value: T
    ) -> T:
        ...


def _alias_get_set(
    conv: Callable[[Any], T],
    *,
    default: T | None = None,
) -> tuple[_Getter[T], _Setter[T]]:
    """Create alias ``get``/``set`` functions using ``conv``."""
    _base_get = partial(_alias_resolve, conv=conv)

    def _get(
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: Optional[T] = default,
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]:
        """Obtain an attribute using :func:`alias_get`."""
        return _base_get(
            d,
            aliases,
            default=default,
            strict=strict,
            log_level=log_level,
        )

    def _set(d: Dict[str, Any], aliases: Sequence[str], value: T) -> T:
        """Set an attribute using :func:`alias_set`."""
        return alias_set(d, aliases, conv, value)

    return _get, _set


get_attr, set_attr = _alias_get_set(float, default=0.0)
get_attr_str, set_attr_str = _alias_get_set(str, default="")


# -------------------------
# Máximos globales con caché
# -------------------------


def recompute_abs_max(G, aliases: tuple[str, ...]):
    """Recalculate and return ``(max_val, node)`` for ``aliases``."""
    node = max(
        G.nodes(),
        key=lambda m: abs(get_attr(G.nodes[m], aliases, 0.0)),
        default=None,
    )
    max_val = (
        abs(get_attr(G.nodes[node], aliases, 0.0)) if node is not None else 0.0
    )
    return max_val, node


def multi_recompute_abs_max(
    G, alias_map: Dict[str, tuple[str, ...]]
) -> Dict[str, float]:
    """Return absolute maxima for each entry in ``alias_map``.

    ``alias_map`` maps result keys to alias tuples. The graph is
    traversed once and the absolute maximum for each alias tuple is
    recorded. The returned dictionary uses the same keys as
    ``alias_map``.
    """

    maxima = {k: 0.0 for k in alias_map}
    for _, nd in G.nodes(data=True):
        for key, aliases in alias_map.items():
            val = abs(get_attr(nd, aliases, 0.0))
            if val > maxima[key]:
                maxima[key] = val
    return {k: float(v) for k, v in maxima.items()}


def _update_cached_abs_max(
    G, aliases: tuple[str, ...], n, value, *, key: str
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
    G, n, aliases: tuple[str, ...], value: float, *, cache: str
) -> None:
    """Assign ``value`` and update the global maximum."""
    val = float(value)
    set_attr(G.nodes[n], aliases, val)
    _update_cached_abs_max(G, aliases, n, val, key=cache)


def set_vf(G, n, value: float, *, update_max: bool = True) -> None:
    """Set ``νf`` and optionally update the global maximum."""
    val = float(value)
    set_attr(G.nodes[n], ALIAS_VF, val)
    if update_max:
        _update_cached_abs_max(G, ALIAS_VF, n, val, key="_vfmax")


def set_dnfr(G, n, value: float) -> None:
    """Set ``ΔNFR`` and update the global maximum."""
    set_attr_with_max(G, n, ALIAS_DNFR, value, cache="_dnfrmax")
