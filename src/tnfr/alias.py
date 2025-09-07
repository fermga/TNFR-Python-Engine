"""Attribute helpers supporting alias keys.

``AliasAccessor`` provides the main implementation for dealing with
alias-based attribute access.  The module-level :func:`alias_get` and
``alias_set`` functions are thin wrappers over a shared
``AliasAccessor`` instance kept for backward compatibility.
"""

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
    Generic,
)
import logging
from functools import lru_cache
from .logging_utils import get_logger

from .constants import ALIAS_VF, ALIAS_DNFR

logger = get_logger(__name__)

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


def _validate_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Return ``aliases`` as a validated tuple of strings."""
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("'aliases' must be a non-string sequence")
    return _cached_validate_aliases(tuple(aliases))


@lru_cache(maxsize=128)
def _cached_validate_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    for a in aliases:
        if not isinstance(a, str):
            raise TypeError("'aliases' elements must be strings")
    return aliases


# expose cache management helpers on the public function
_validate_aliases.cache_clear = _cached_validate_aliases.cache_clear  # type: ignore[attr-defined]
_validate_aliases.cache_info = _cached_validate_aliases.cache_info  # type: ignore[attr-defined]


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

    encountered_error = False
    for key in aliases:
        if key in d:
            try:
                return conv(d[key])
            except (ValueError, TypeError) as exc:
                encountered_error = True
                if not strict:
                    lvl = log_level if log_level is not None else logging.DEBUG
                    logger.log(
                        lvl, "No se pudo convertir el valor para %r: %s", key, exc
                    )
    if default is not None:
        try:
            return conv(default)
        except (ValueError, TypeError) as exc:
            encountered_error = True
            if not strict:
                lvl = (
                    logging.WARNING if log_level is None else log_level
                )
                logger.log(
                    lvl, "No se pudo convertir el valor para 'default': %s", exc
                )

    if not encountered_error:
        return None

    # At this point no conversion succeeded and at least one error occurred.
    # Build the error list only if we need to raise or log a final summary.
    errors: list[tuple[str, Exception]] = []
    for key in aliases:
        if key in d:
            try:
                conv(d[key])
            except (ValueError, TypeError) as exc:
                errors.append((key, exc))
    if default is not None:
        try:
            conv(default)
        except (ValueError, TypeError) as exc:
            errors.append(("default", exc))

    if errors and strict:
        err_msg = "; ".join(f"{k!r}: {e}" for k, e in errors)
        raise ValueError(f"No se pudieron convertir valores para {err_msg}")

    if errors and not strict:
        # In lax mode errors have already been logged individually; emit a summary
        lvl = log_level if log_level is not None else logging.DEBUG
        summary = "; ".join(f"{k!r}: {e}" for k, e in errors)
        logger.log(lvl, "No se pudieron convertir valores para %s", summary)

    return None


class AliasAccessor(Generic[T]):
    """Helper providing ``get`` and ``set`` for alias-based attributes.

    This class implements all logic for resolving and assigning values
    using alias keys.  Function helpers :func:`alias_get` and
    :func:`alias_set` simply delegate to a module-level instance of this
    class.
    """

    def __init__(
        self, conv: Callable[[Any], T] | None = None, default: T | None = None
    ) -> None:
        self._conv = conv
        self._default = default

    def get(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: Optional[T] = None,
        *,
        strict: bool = False,
        log_level: int | None = None,
        conv: Callable[[Any], T] | None = None,
    ) -> Optional[T]:
        aliases = _validate_aliases(aliases)
        if conv is None:
            conv = self._conv
        if conv is None:
            raise TypeError("'conv' must be provided")
        if default is None:
            default = self._default
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
        aliases: Sequence[str],
        value: Any,
        conv: Callable[[Any], T] | None = None,
    ) -> T:
        aliases = _validate_aliases(aliases)
        if conv is None:
            conv = self._conv
        if conv is None:
            raise TypeError("'conv' must be provided")
        val = conv(value)
        key = next((k for k in aliases if k in d), aliases[0])
        d[key] = val
        return val


# Shared accessor used by wrapper functions
_alias_accessor: AliasAccessor[Any] = AliasAccessor()


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
    """Return the value for the first existing key in ``aliases``.

    This is a convenience wrapper over a shared :class:`AliasAccessor`
    instance.
    """
    return _alias_accessor.get(
        d,
        aliases,
        default=default,
        strict=strict,
        log_level=log_level,
        conv=conv,
    )


def alias_set(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Assign ``value`` converted to the first available key in ``aliases``.

    This is a convenience wrapper over a shared :class:`AliasAccessor`
    instance.
    """
    return _alias_accessor.set(d, aliases, value, conv=conv)


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
        conv: Callable[[Any], T] | None = ...,
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
        conv: Callable[[Any], T] | None = ...,
    ) -> Optional[T]: ...


class _Setter(Protocol[T]):
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
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
