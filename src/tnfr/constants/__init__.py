"""Shared constants."""

from __future__ import annotations

from typing import Any, Dict
from collections.abc import Mapping
import threading
import copy
import warnings
from types import MappingProxyType
from functools import lru_cache
from dataclasses import asdict, is_dataclass
import weakref

from .core import CORE_DEFAULTS, REMESH_DEFAULTS
from .init import INIT_DEFAULTS
from .metric import (
    METRIC_DEFAULTS,
    SIGMA,
    TRACE,
    METRICS,
    GRAMMAR_CANON,
    COHERENCE,
    DIAGNOSIS,
)

try:  # pragma: no cover - optional dependency
    from ..helpers.cache import ensure_node_offset_map
except ImportError:  # noqa: BLE001 - allow any import error
    ensure_node_offset_map = None

# Valores que pueden asignarse directamente sin copiar
IMMUTABLE_SIMPLE = (
    int,
    float,
    complex,
    str,
    bool,
    bytes,
    type(None),
)


def _freeze(value: Any, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        raise ValueError("cycle detected")
    seen.add(obj_id)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            params = getattr(type(value), "__dataclass_params__", None)
            frozen = bool(params and params.frozen)
            value = asdict(value)
            tag = "mapping" if frozen else "dict"
            return (
                tag,
                tuple((k, _freeze(v, seen)) for k, v in value.items()),
            )
        if isinstance(value, IMMUTABLE_SIMPLE):
            return value
        if isinstance(value, tuple):
            return tuple(_freeze(v, seen) for v in value)
        if isinstance(value, list):
            return ("list", tuple(_freeze(v, seen) for v in value))
        if isinstance(value, set):
            return ("set", tuple(_freeze(v, seen) for v in value))
        if isinstance(value, frozenset):
            return frozenset(_freeze(v, seen) for v in value)
        if isinstance(value, bytearray):
            return ("bytearray", bytes(value))
        if isinstance(value, Mapping):
            tag = "dict" if hasattr(value, "__setitem__") else "mapping"
            return (
                tag,
                tuple((k, _freeze(v, seen)) for k, v in value.items()),
            )
        raise TypeError
    finally:
        seen.remove(obj_id)


@lru_cache(maxsize=1024)
def _is_immutable_inner(value: Any) -> bool:
    if isinstance(value, IMMUTABLE_SIMPLE):
        return True
    if isinstance(value, tuple):
        if value and isinstance(value[0], str):
            tag = value[0]
            if tag in {"list", "dict", "set", "bytearray"}:
                return False
            if tag == "mapping":
                return all(_is_immutable_inner(v) for v in value[1])
        return all(_is_immutable_inner(v) for v in value)
    if isinstance(value, frozenset):
        return all(_is_immutable_inner(v) for v in value)
    return False


# Cache of previous results keyed by object identity. Uses ``WeakKeyDictionary``
# so entries vanish automatically when objects are garbage collected.
_IMMUTABLE_CACHE: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
_IMMUTABLE_CACHE_LOCK = threading.Lock()


def _is_immutable(value: Any) -> bool:
    """Check recursively if ``value`` is immutable with caching.

    Results are memoised by object identity using weak references to avoid
    retaining objects that are no longer in use. Note that mutated objects may
    yield stale results; ``_is_immutable`` assumes immutability does not change
    for a given object ID.
    """

    # Try to fetch from cache using object identity. Objects that cannot be
    # weak-referenced will raise ``TypeError``; those simply bypass the cache.
    with _IMMUTABLE_CACHE_LOCK:
        try:
            return _IMMUTABLE_CACHE[value]
        except (KeyError, TypeError):
            pass

    try:
        frozen = _freeze(value)
    except (TypeError, ValueError):
        result = False
    else:
        result = _is_immutable_inner(frozen)

    # Store result in cache when possible.
    with _IMMUTABLE_CACHE_LOCK:
        try:
            _IMMUTABLE_CACHE[value] = result
        except TypeError:
            pass

    return result


# Diccionario combinado exportado
# Unimos los diccionarios en orden de menor a mayor prioridad para que los
# valores de ``METRIC_DEFAULTS`` sobrescriban al resto,
# como hacía ``ChainMap``.
_DEFAULTS_COMBINED: Dict[str, Any] = (
    CORE_DEFAULTS | INIT_DEFAULTS | REMESH_DEFAULTS | METRIC_DEFAULTS
)
DEFAULTS: Mapping[str, Any] = MappingProxyType(_DEFAULTS_COMBINED)

# -------------------------
# Retrocompatibilidad y aliases
# -------------------------
# "REMESH_TAU" era el nombre original para la memoria de REMESH. Hoy se
# desglosa en ``REMESH_TAU_GLOBAL`` y ``REMESH_TAU_LOCAL``.
ALIASES: Dict[str, tuple[str, ...]] = {
    "REMESH_TAU": ("REMESH_TAU_GLOBAL", "REMESH_TAU_LOCAL"),
}

_ALIAS_TARGET_TO_KEY: Dict[str, str] = {
    target: alias for alias, targets in ALIASES.items() for target in targets
}

# -------------------------
# Utilidades
# -------------------------


def inject_defaults(
    G, defaults: Mapping[str, Any] = DEFAULTS, override: bool = False
) -> None:
    """Inject ``defaults`` into ``G.graph``.

    ``defaults`` is usually ``DEFAULTS``, combining all sub-dictionaries.
    If ``override`` is ``True`` existing values are overwritten. Immutable
    values (numbers, strings, tuples, etc.) are assigned directly. Tuples are
    inspected recursively; if any element is mutable, a ``deepcopy`` is made
    to avoid shared state.
    """
    G.graph.setdefault("_tnfr_defaults_attached", False)
    for k, v in defaults.items():
        if override or k not in G.graph:
            G.graph[k] = v if _is_immutable(v) else copy.deepcopy(v)
    G.graph["_tnfr_defaults_attached"] = True
    if ensure_node_offset_map is not None:
        ensure_node_offset_map(G)


def merge_overrides(G, **overrides) -> None:
    """Apply specific changes to ``G.graph``."""
    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise KeyError(f"Parámetro desconocido: '{key}'")
        G.graph[key] = value


def get_param(G, key: str):
    """Retrieve a parameter from ``G.graph`` resolving legacy aliases."""
    if key in G.graph:
        return G.graph[key]
    alias = _ALIAS_TARGET_TO_KEY.get(key)
    if alias and alias in G.graph:
        warnings.warn(
            f"'{alias}' es alias legado; usa '{key}'",
            DeprecationWarning,
            stacklevel=2,
        )
        return G.graph[alias]
    if key not in DEFAULTS:
        raise KeyError(f"Parámetro desconocido: '{key}'")
    return DEFAULTS[key]


# Claves canónicas con nombres ASCII
VF_KEY = "νf"
THETA_KEY = "θ"

# Alias exportados por conveniencia (evita imports circulares)
ALIAS_VF = (VF_KEY, "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA = (THETA_KEY, "theta", "fase", "phi", "phase")
ALIAS_DNFR = ("ΔNFR", "delta_nfr", "dnfr")
ALIAS_EPI = ("EPI", "psi", "PSI", "value")
ALIAS_EPI_KIND = ("EPI_kind", "epi_kind", "source_glyph")
ALIAS_SI = ("Si", "sense_index", "S_i", "sense", "meaning_index")
ALIAS_dEPI = ("dEPI_dt", "dpsi_dt", "dEPI", "velocity")
ALIAS_D2EPI = ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel")
ALIAS_dVF = ("dνf_dt", "dvf_dt", "dnu_dt", "dvf")
ALIAS_D2VF = ("d2νf_dt2", "d2vf_dt2", "d2nu_dt2", "B")
ALIAS_dSI = ("δSi", "delta_Si", "dSi")

VF_PRIMARY = ALIAS_VF[0]
THETA_PRIMARY = ALIAS_THETA[0]
DNFR_PRIMARY = ALIAS_DNFR[0]
EPI_PRIMARY = ALIAS_EPI[0]
EPI_KIND_PRIMARY = ALIAS_EPI_KIND[0]
SI_PRIMARY = ALIAS_SI[0]
dEPI_PRIMARY = ALIAS_dEPI[0]
D2EPI_PRIMARY = ALIAS_D2EPI[0]
dVF_PRIMARY = ALIAS_dVF[0]
D2VF_PRIMARY = ALIAS_D2VF[0]
dSI_PRIMARY = ALIAS_dSI[0]

__all__ = [
    "CORE_DEFAULTS",
    "INIT_DEFAULTS",
    "REMESH_DEFAULTS",
    "METRIC_DEFAULTS",
    "SIGMA",
    "TRACE",
    "METRICS",
    "GRAMMAR_CANON",
    "COHERENCE",
    "DIAGNOSIS",
    "DEFAULTS",
    "ALIASES",
    "inject_defaults",
    "merge_overrides",
    "get_param",
    "VF_KEY",
    "THETA_KEY",
    "ALIAS_VF",
    "ALIAS_THETA",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_EPI_KIND",
    "ALIAS_SI",
    "ALIAS_dEPI",
    "ALIAS_D2EPI",
    "ALIAS_dVF",
    "ALIAS_D2VF",
    "ALIAS_dSI",
    "VF_PRIMARY",
    "THETA_PRIMARY",
    "DNFR_PRIMARY",
    "EPI_PRIMARY",
    "EPI_KIND_PRIMARY",
    "SI_PRIMARY",
    "dEPI_PRIMARY",
    "D2EPI_PRIMARY",
    "dVF_PRIMARY",
    "D2VF_PRIMARY",
    "dSI_PRIMARY",
]
