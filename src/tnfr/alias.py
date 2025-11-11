"""Attribute helpers supporting alias keys.

``AliasAccessor`` provides the main implementation for dealing with
alias-based attribute access. Legacy wrappers ``alias_get`` and
``alias_set`` have been removed; use :func:`get_attr` and
:func:`set_attr` instead.

CRITICAL: Canonical Attribute Access
=====================================

**ALWAYS use the alias system for reading/writing TNFR attributes.**

The TNFR canonical attribute keys use Unicode symbols (e.g., 'νf' for structural
frequency), but NetworkX and Python code often use ASCII equivalents (e.g., 'vf').
This creates a critical inconsistency:

**WRONG (breaks canonicity)**:
    >>> G.add_node(0, vf=1.0)           # Uses ASCII 'vf' key
    >>> value = G.nodes[0]['vf']         # Reads ASCII 'vf' - may not exist!
    >>> G.nodes[0]['vf'] = 2.0           # Writes ASCII 'vf' - wrong key!

**CORRECT (maintains canonicity)**:
    >>> from tnfr.alias import set_vf, get_attr
    >>> from tnfr.constants.aliases import ALIAS_VF
    >>> from tnfr.constants import VF_PRIMARY
    >>>
    >>> # For initialization, use canonical setters:
    >>> set_vf(G, 0, 1.0)                # Writes to 'νf' (Greek nu)
    >>>
    >>> # For reading, use canonical getters:
    >>> value = get_attr(G.nodes[0], ALIAS_VF, 0.0)  # Reads from 'νf'
    >>>
    >>> # Or use PRIMARY constants in add_node:
    >>> G.add_node(1, **{VF_PRIMARY: 1.0})  # Writes to 'νf' directly

**Why This Matters**:
- The alias system tries ALL aliases in order: ('νf', 'nu_f', 'nu-f', 'nu', 'freq', 'frequency')
- If you write to 'vf', the data is stored under a key NOT in the alias list
- Reading via get_attr() will return the default (0.0) instead of your value
- This breaks the nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

**For Tests**:
    >>> from tnfr.structural import create_nfr
    >>> # PREFERRED: Use create_nfr which handles canonicity
    >>> G, node = create_nfr("test", vf=1.0, epi=0.5, theta=0.0)
    >>>
    >>> # ALTERNATIVE: Manual initialization with canonical setters
    >>> from tnfr.alias import set_vf, get_attr
    >>> from tnfr.constants.aliases import ALIAS_VF
    >>> G = nx.Graph()
    >>> G.add_node(0, theta=0.0, EPI=1.0, Si=0.5)  # Other attrs OK
    >>> set_vf(G, 0, 1.0)                          # Use canonical setter for vf
    >>> value = get_attr(G.nodes[0], ALIAS_VF, 0.0)  # Use canonical getter

**Applies to**: νf (vf), θ (theta), ΔNFR (dnfr), and other aliased attributes.
See ALIAS_VF, ALIAS_THETA, ALIAS_DNFR in tnfr.constants.aliases for full lists.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sized
from functools import lru_cache, partial
from threading import Lock
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Optional,
    TypeVar,
    cast,
)

from .compat.dataclass import dataclass
from .constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from .types import FloatArray, NodeId
from .utils import convert_value

if TYPE_CHECKING:  # pragma: no cover
    import networkx

T = TypeVar("T")


def _bepi_to_float(value: Any) -> float:
    """Extract scalar from BEPIElement dict or convert value to float.

    When operators transform EPI from float to BEPIElement dict, this helper
    extracts the maximum magnitude from the 'continuous' component. This
    preserves ΔNFR semantics (§3.3) and structural metrics accuracy (§3.9).

    Parameters
    ----------
    value : Any
        Value to convert. If it's a dict with a 'continuous' key, extracts
        the maximum magnitude. Otherwise converts directly to float.

    Returns
    -------
    float
        Scalar representation of the value.
    """
    if isinstance(value, dict) and "continuous" in value:
        cont = value["continuous"]
        if isinstance(cont, tuple):
            return float(max(abs(c) for c in cont)) if cont else 0.0
        return float(abs(cont))
    return float(value)


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

    def __init__(self, conv: Callable[[Any], T] | None = None, default: T | None = None) -> None:
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

    def _resolve_cache_key(
        self, d: dict[str, Any], aliases: tuple[str, ...]
    ) -> tuple[tuple[int, tuple[str, ...]], str | None]:
        """Return cache entry for ``d`` and ``aliases`` if still valid.

        The mapping remains coherent only when the cached key exists in
        ``d`` and the dictionary size has not changed. Invalid entries are
        removed to preserve structural consistency.
        """

        cache_key = (id(d), aliases)
        with self._lock:
            cached = self._key_cache.get(cache_key)
        if cached is not None:
            key, size = cached
            if size == len(d) and key in d:
                return cache_key, key
            with self._lock:
                self._key_cache.pop(cache_key, None)
        return cache_key, None

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
        """Return ``value`` for the first alias present in ``d``."""

        aliases, conv, default = self._prepare(aliases, conv, default)
        cache_key, key = self._resolve_cache_key(d, aliases)
        if key is not None:
            ok, value = convert_value(d[key], conv, strict=strict, key=key, log_level=log_level)
            if ok:
                return value
        for key in aliases:
            if key in d:
                ok, value = convert_value(d[key], conv, strict=strict, key=key, log_level=log_level)
                if ok:
                    with self._lock:
                        self._key_cache[cache_key] = (key, len(d))
                    return value
        if default is not None:
            ok, value = convert_value(
                default,
                conv,
                strict=strict,
                key="default",
                log_level=log_level,
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
        """Write ``value`` under the first matching alias and cache the choice."""

        aliases, conv, _ = self._prepare(aliases, conv)
        cache_key, key = self._resolve_cache_key(d, aliases)
        if key is not None:
            d[key] = conv(value)
            return d[key]
        key = next((k for k in aliases if k in d), aliases[0])
        val = conv(value)
        d[key] = val
        with self._lock:
            self._key_cache[cache_key] = (key, len(d))
        return val


_generic_accessor: AliasAccessor[Any] = AliasAccessor()


def get_theta_attr(
    d: Mapping[str, Any],
    default: T | None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], T] = float,
) -> T | None:
    """Return ``theta``/``phase`` using the English alias set."""
    return _generic_accessor.get(
        cast(dict[str, Any], d),
        ALIAS_THETA,
        default,
        strict=strict,
        log_level=log_level,
        conv=conv,
    )


def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: T | None = None,
    *,
    strict: bool = False,
    log_level: int | None = None,
    conv: Callable[[Any], T] = _bepi_to_float,
) -> T | None:
    """Return the value for the first key in ``aliases`` found in ``d``.

    WARNING: This function searches for keys in alias order. If you manually
    wrote to a non-canonical key (e.g., 'vf' instead of 'νf'), this function
    will NOT find it and will return the default value instead.

    For structural frequency: ALWAYS use set_vf() to write, not d['vf'] = value.
    See module docstring for detailed guidance on canonical attribute access.
    """

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
    nodes: Iterable[NodeId],
    aliases: Iterable[str],
    default: float = 0.0,
    *,
    np: ModuleType | None = None,
) -> FloatArray | list[float]:
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

    def _nodes_iter_and_size(nodes: Iterable[NodeId]) -> tuple[Iterable[NodeId], int]:
        if nodes is G.nodes:
            return G.nodes, G.number_of_nodes()
        if isinstance(nodes, Sized):
            return nodes, len(nodes)  # type: ignore[arg-type]
        nodes_list = list(nodes)
        return nodes_list, len(nodes_list)

    nodes_iter, size = _nodes_iter_and_size(nodes)

    def _value(node: NodeId) -> float:
        return float(get_attr(G.nodes[node], aliases, default))

    if np is not None:
        values: FloatArray = np.fromiter((_value(n) for n in nodes_iter), float, count=size)
        return values
    return [_value(n) for n in nodes_iter]


def collect_theta_attr(
    G: "networkx.Graph",
    nodes: Iterable[NodeId],
    default: float = 0.0,
    *,
    np: ModuleType | None = None,
) -> FloatArray | list[float]:
    """Collect ``theta`` values honouring the English-only attribute contract."""

    def _nodes_iter_and_size(nodes: Iterable[NodeId]) -> tuple[Iterable[NodeId], int]:
        if nodes is G.nodes:
            return G.nodes, G.number_of_nodes()
        if isinstance(nodes, Sized):
            return nodes, len(nodes)  # type: ignore[arg-type]
        nodes_list = list(nodes)
        return nodes_list, len(nodes_list)

    nodes_iter, size = _nodes_iter_and_size(nodes)

    def _value(node: NodeId) -> float:
        return float(get_theta_attr(G.nodes[node], default))

    if np is not None:
        values: FloatArray = np.fromiter((_value(n) for n in nodes_iter), float, count=size)
        return values

    return [_value(n) for n in nodes_iter]


def set_attr_generic(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    *,
    conv: Callable[[Any], T],
) -> T:
    """Assign ``value`` to the FIRST (canonical) alias key in ``aliases``.

    CRITICAL: This function writes to the FIRST key in the alias tuple.
    For ALIAS_VF = ('νf', 'nu_f', ...), this writes to 'νf' (Greek nu), NOT 'vf'.

    If you later try to read with G.nodes[n]['vf'], you will NOT find the value.
    ALWAYS use get_attr() to read what set_attr() wrote.

    For high-level usage, prefer set_vf(), set_theta(), etc. which handle this correctly.
    See module docstring for detailed guidance on canonical attribute access.
    """

    return _generic_accessor.set(d, aliases, value, conv=conv)


set_attr = partial(set_attr_generic, conv=float)

get_attr_str = partial(get_attr, conv=str)
set_attr_str = partial(set_attr_generic, conv=str)


def set_theta_attr(d: MutableMapping[str, Any], value: Any) -> float:
    """Assign ``theta``/``phase`` using the English attribute names."""
    result = float(value)
    d["theta"] = result
    d["phase"] = result
    return result


# -------------------------
# Cached global maxima
# -------------------------


@dataclass(slots=True)
class AbsMaxResult:
    """Absolute maximum value and the node where it occurs."""

    max_value: float
    node: Hashable | None


def _coerce_abs_value(value: Any) -> float:
    """Return ``value`` as ``float`` treating ``None`` as ``0.0``."""

    if value is None:
        return 0.0
    try:
        return _bepi_to_float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_abs_max_result(
    G: "networkx.Graph",
    aliases: tuple[str, ...],
    *,
    key: str | None = None,
    candidate: tuple[Hashable, float] | None = None,
) -> AbsMaxResult:
    """Return the absolute maximum (and node) for ``aliases``.

    Parameters
    ----------
    G:
        Graph containing nodal data.
    aliases:
        Attribute aliases to inspect.
    key:
        Cache key to update. When ``None``, the graph cache is untouched.
    candidate:
        Optional ``(node, value)`` pair representing a candidate maximum.

    Returns
    -------
    AbsMaxResult
        Structure holding the absolute maximum and the node where it
        occurs. When ``candidate`` is provided, its value is treated as the
        current maximum and no recomputation is performed.
    """

    if candidate is not None:
        node, value = candidate
        max_val = abs(float(value))
    else:
        node, max_val = max(
            ((n, abs(get_attr(G.nodes[n], aliases, 0.0))) for n in G.nodes()),
            key=lambda item: item[1],
            default=(None, 0.0),
        )
        max_val = float(max_val)

    if key is not None:
        G.graph[key] = max_val
        G.graph[f"{key}_node"] = node

    return AbsMaxResult(max_value=max_val, node=node)


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
            {key: max(maxima[key], abs(get_attr(nd, aliases, 0.0))) for key, aliases in items}
        )
    return {k: float(v) for k, v in maxima.items()}


def _update_cached_abs_max(
    G: "networkx.Graph",
    aliases: tuple[str, ...],
    n: Hashable,
    value: float,
    *,
    key: str,
) -> AbsMaxResult:
    """Update cached absolute maxima for ``aliases``.

    The current cached value is updated when ``value`` becomes the new
    maximum or when the stored node matches ``n`` but its magnitude
    decreases. The returned :class:`AbsMaxResult` always reflects the
    cached maximum after applying the update.
    """

    node_key = f"{key}_node"
    val = abs(float(value))
    cur = _coerce_abs_value(G.graph.get(key))
    cur_node = cast(Hashable | None, G.graph.get(node_key))

    if val >= cur:
        return _compute_abs_max_result(G, aliases, key=key, candidate=(n, val))
    if cur_node == n:
        return _compute_abs_max_result(G, aliases, key=key)
    return AbsMaxResult(max_value=cur, node=cur_node)


def set_attr_and_cache(
    G: "networkx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str | None = None,
    extra: Callable[["networkx.Graph", Hashable, float], None] | None = None,
) -> AbsMaxResult | None:
    """Assign ``value`` to node ``n`` and optionally update cached maxima.

    Returns
    -------
    AbsMaxResult | None
        Absolute maximum information when ``cache`` is provided; otherwise
        ``None``.
    """

    val = set_attr(G.nodes[n], aliases, value)
    result: AbsMaxResult | None = None
    if cache is not None:
        result = _update_cached_abs_max(G, aliases, n, val, key=cache)
    if extra is not None:
        extra(G, n, val)
    return result


def set_attr_with_max(
    G: "networkx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str,
) -> AbsMaxResult:
    """Assign ``value`` to node ``n`` and update the global maximum.

    This is a convenience wrapper around :func:`set_attr_and_cache`.
    """
    return cast(
        AbsMaxResult,
        set_attr_and_cache(G, n, aliases, value, cache=cache),
    )


def set_scalar(
    G: "networkx.Graph",
    n: Hashable,
    alias: tuple[str, ...],
    value: float,
    *,
    cache: str | None = None,
    extra: Callable[["networkx.Graph", Hashable, float], None] | None = None,
) -> AbsMaxResult | None:
    """Assign ``value`` to ``alias`` for node ``n`` and update caches.

    Returns
    -------
    AbsMaxResult | None
        Updated absolute maximum details when ``cache`` is provided.
    """

    return set_attr_and_cache(G, n, alias, value, cache=cache, extra=extra)


def _increment_trig_version(G: "networkx.Graph", _: Hashable, __: float) -> None:
    """Increment cached trig version to invalidate trig caches."""
    g = G.graph
    g["_trig_version"] = int(g.get("_trig_version", 0)) + 1


SCALAR_SETTERS: dict[str, dict[str, Any]] = {
    "vf": {
        "alias": ALIAS_VF,
        "cache": "_vfmax",
        "doc": "Set ``νf`` for node ``n`` and optionally update the global maximum.",
        "update_max_param": True,
    },
    "dnfr": {
        "alias": ALIAS_DNFR,
        "cache": "_dnfrmax",
        "doc": "Set ``ΔNFR`` for node ``n`` and update the global maximum.",
    },
    "theta": {
        "alias": ALIAS_THETA,
        "extra": _increment_trig_version,
        "doc": "Set ``theta`` for node ``n`` and invalidate trig caches.",
    },
}


def _make_scalar_setter(name: str, spec: dict[str, Any]) -> Callable[..., AbsMaxResult | None]:
    alias = spec["alias"]
    cache = spec.get("cache")
    extra = spec.get("extra")
    doc = spec.get("doc")
    has_update = spec.get("update_max_param", False)

    if has_update:

        def setter(
            G: "networkx.Graph",
            n: Hashable,
            value: float,
            *,
            update_max: bool = True,
        ) -> AbsMaxResult | None:
            cache_key = cache if update_max else None
            return set_scalar(G, n, alias, value, cache=cache_key, extra=extra)

    else:

        def setter(G: "networkx.Graph", n: Hashable, value: float) -> AbsMaxResult | None:
            return set_scalar(G, n, alias, value, cache=cache, extra=extra)

    setter.__name__ = f"set_{name}"
    setter.__qualname__ = f"set_{name}"
    setter.__doc__ = doc
    return setter


for _name, _spec in SCALAR_SETTERS.items():
    globals()[f"set_{_name}"] = _make_scalar_setter(_name, _spec)

del _name, _spec, _make_scalar_setter

_set_theta_impl = cast(
    Callable[["networkx.Graph", Hashable, float], AbsMaxResult | None],
    globals()["set_theta"],
)


def _set_theta_with_compat(G: "networkx.Graph", n: Hashable, value: float) -> AbsMaxResult | None:
    nd = cast(MutableMapping[str, Any], G.nodes[n])
    result = _set_theta_impl(G, n, value)
    theta_val = get_theta_attr(nd, value)
    if theta_val is not None:
        float_theta = float(theta_val)
        nd["theta"] = float_theta
        nd["phase"] = float_theta
    return result


_set_theta_with_compat.__name__ = "set_theta"
_set_theta_with_compat.__qualname__ = "set_theta"
_set_theta_with_compat.__doc__ = _set_theta_impl.__doc__
globals()["set_theta"] = _set_theta_with_compat

__all__ = [
    "AbsMaxResult",
    "set_attr_generic",
    "get_attr",
    "get_theta_attr",
    "collect_attr",
    "collect_theta_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_theta_attr",
    "set_attr_and_cache",
    "set_attr_with_max",
    "set_scalar",
    "SCALAR_SETTERS",
    *[f"set_{name}" for name in SCALAR_SETTERS],
    "multi_recompute_abs_max",
]
