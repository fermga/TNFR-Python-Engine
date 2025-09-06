"""Shared constants."""

from __future__ import annotations

from typing import Any, Dict, Mapping
import copy
import warnings
from types import MappingProxyType
from weakref import ref

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
    from ..helpers import ensure_node_offset_map
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

_IMMUTABLE_CACHE: dict[int, tuple[ref, bool]] = {}


def _is_immutable(value: Any) -> bool:
    """Check recursively if ``value`` is immutable with caching."""
    oid = id(value)
    entry = _IMMUTABLE_CACHE.get(oid)
    if entry is not None:
        obj_ref, cached = entry
        if obj_ref() is value:
            return cached
    if isinstance(value, IMMUTABLE_SIMPLE):
        res = True
    elif isinstance(value, tuple):
        res = all(_is_immutable(item) for item in value)
    elif isinstance(value, frozenset):
        res = all(_is_immutable(item) for item in value)
    elif isinstance(value, MappingProxyType):
        res = all(_is_immutable(v) for v in value.values())
    else:
        res = False
    try:
        _IMMUTABLE_CACHE[oid] = (ref(value), res)
    except TypeError:
        pass
    return res

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


def attach_defaults(G, override: bool = False) -> None:
    """Write combined ``DEFAULTS`` into ``G.graph``.

    If ``override`` is ``True`` existing values are overwritten.
    """
    inject_defaults(G, DEFAULTS, override=override)


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
    G.graph.update(overrides)


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
    "attach_defaults",
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
]
