"""Shared constants."""

from __future__ import annotations

from typing import Any, Callable
from collections.abc import Mapping
import copy
import warnings
from types import MappingProxyType

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

from ..immutable import _is_immutable

try:  # pragma: no cover - optional dependency
    from ..helpers import ensure_node_offset_map
except ImportError:  # noqa: BLE001 - allow any import error
    ensure_node_offset_map = None

# Secciones individuales exportadas
DEFAULT_SECTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "core": CORE_DEFAULTS,
        "init": INIT_DEFAULTS,
        "remesh": REMESH_DEFAULTS,
        "metric": METRIC_DEFAULTS,
    }
)

# Diccionario combinado exportado
# Unimos los diccionarios en orden de menor a mayor prioridad para que los
# valores de ``METRIC_DEFAULTS`` sobrescriban al resto, como hacía
# ``ChainMap``.
DEFAULTS: Mapping[str, Any] = MappingProxyType(
    CORE_DEFAULTS | INIT_DEFAULTS | REMESH_DEFAULTS | METRIC_DEFAULTS
)

# -------------------------
# Retrocompatibilidad y aliases
# -------------------------
# "REMESH_TAU" era el nombre original para la memoria de REMESH. Hoy se
# desglosa en ``REMESH_TAU_GLOBAL`` y ``REMESH_TAU_LOCAL``.
PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "REMESH_TAU": ("REMESH_TAU_GLOBAL", "REMESH_TAU_LOCAL"),
}

_PARAM_ALIAS_TARGET_TO_KEY: dict[str, str] = {
    target: alias for alias, targets in PARAM_ALIASES.items() for target in targets
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
    """Apply specific changes to ``G.graph``.

    Non-immutable values are deep-copied to avoid shared state with
    :data:`DEFAULTS`.
    """
    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise KeyError(f"Parámetro desconocido: '{key}'")
        G.graph[key] = value if _is_immutable(value) else copy.deepcopy(value)


def get_param(G, key: str):
    """Retrieve a parameter from ``G.graph`` resolving legacy aliases."""
    if key in G.graph:
        return G.graph[key]
    alias = _PARAM_ALIAS_TARGET_TO_KEY.get(key)
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


def get_graph_param(G, key: str, cast: Callable[[Any], Any] = float):
    """Return ``key`` from ``G.graph`` applying ``cast``.

    The ``cast`` argument must be a function (e.g. ``float``, ``int``,
    ``bool``). If the stored value is ``None`` it is returned without
    casting.
    """
    val = get_param(G, key)
    return None if val is None else cast(val)


# Claves canónicas con nombres ASCII
VF_KEY = "νf"
THETA_KEY = "θ"

# Mapa de aliases para atributos nodales
ALIASES: dict[str, tuple[str, ...]] = {
    "VF": (VF_KEY, "nu_f", "nu-f", "nu", "freq", "frequency"),
    "THETA": (THETA_KEY, "theta", "fase", "phi", "phase"),
    "DNFR": ("ΔNFR", "delta_nfr", "dnfr"),
    "EPI": ("EPI", "psi", "PSI", "value"),
    "EPI_KIND": ("EPI_kind", "epi_kind", "source_glyph"),
    "SI": ("Si", "sense_index", "S_i", "sense", "meaning_index"),
    "DEPI": ("dEPI_dt", "dpsi_dt", "dEPI", "velocity"),
    "D2EPI": ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel"),
    "DVF": ("dνf_dt", "dvf_dt", "dnu_dt", "dvf"),
    "D2VF": ("d2νf_dt2", "d2vf_dt2", "d2nu_dt2", "B"),
    "DSI": ("δSi", "delta_Si", "dSi"),
}


def get_aliases(key: str) -> tuple[str, ...]:
    """Return alias tuple for canonical ``key``."""

    return ALIASES[key]


VF_PRIMARY = get_aliases("VF")[0]
THETA_PRIMARY = get_aliases("THETA")[0]
DNFR_PRIMARY = get_aliases("DNFR")[0]
EPI_PRIMARY = get_aliases("EPI")[0]
EPI_KIND_PRIMARY = get_aliases("EPI_KIND")[0]
SI_PRIMARY = get_aliases("SI")[0]
dEPI_PRIMARY = get_aliases("DEPI")[0]
D2EPI_PRIMARY = get_aliases("D2EPI")[0]
dVF_PRIMARY = get_aliases("DVF")[0]
D2VF_PRIMARY = get_aliases("D2VF")[0]
dSI_PRIMARY = get_aliases("DSI")[0]

__all__ = (
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
    "DEFAULT_SECTIONS",
    "PARAM_ALIASES",
    "ALIASES",
    "inject_defaults",
    "merge_overrides",
    "get_param",
    "get_graph_param",
    "get_aliases",
    "VF_KEY",
    "THETA_KEY",
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
)

# Deprecated names ---------------------------------------------------------
# Keep legacy alias constants for backwards compatibility.
_DEPRECATED_ALIASES = {
    "ALIAS_dEPI": "DEPI",
    "ALIAS_dVF": "DVF",
    "ALIAS_dSI": "DSI",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        key = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"'{name}' is deprecated; use get_aliases('{key}')",
            DeprecationWarning,
            stacklevel=2,
        )
        return ALIASES[key]
    if name.startswith("ALIAS_") and name[6:] in ALIASES:
        key = name[6:]
        warnings.warn(
            f"'{name}' is deprecated; use get_aliases('{key}')",
            DeprecationWarning,
            stacklevel=2,
        )
        return ALIASES[key]
    raise AttributeError(f"module {__name__} has no attribute {name}")
