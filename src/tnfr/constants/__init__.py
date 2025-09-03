"""Shared constants."""
from __future__ import annotations

from collections import ChainMap
from typing import Any, Dict
import copy
import warnings

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

# Valores que pueden asignarse directamente sin copiar
IMMUTABLE_TYPES = (
    int,
    float,
    complex,
    str,
    bool,
    tuple,
    frozenset,
    bytes,
    type(None),
)

# Diccionario combinado exportado
DEFAULTS: Dict[str, Any] = dict(
    ChainMap(
        dict(METRIC_DEFAULTS),
        dict(REMESH_DEFAULTS),
        dict(INIT_DEFAULTS),
        dict(CORE_DEFAULTS),
    )
)

# -------------------------
# Retrocompatibilidad y aliases
# -------------------------
# "REMESH_TAU" era el nombre original para la memoria de REMESH. Hoy se
# desglosa en ``REMESH_TAU_GLOBAL`` y ``REMESH_TAU_LOCAL``.
ALIASES: Dict[str, tuple[str, ...]] = {
    "REMESH_TAU": ("REMESH_TAU_GLOBAL", "REMESH_TAU_LOCAL"),
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
    G, defaults: Dict[str, Any] = DEFAULTS, override: bool = False
) -> None:
    """Inject ``defaults`` into ``G.graph``.

    ``defaults`` is usually ``DEFAULTS``, combining all sub-dictionaries.
    If ``override`` is ``True`` existing values are overwritten. Immutable
    values (numbers, strings, tuples, etc.) are assigned directly;
    ``copy.deepcopy`` is used only for mutable structures.
    """
    G.graph.setdefault("_tnfr_defaults_attached", False)
    for k, v in defaults.items():
        if override or k not in G.graph:
            G.graph[k] = (
                v if isinstance(v, IMMUTABLE_TYPES) else copy.deepcopy(v)
            )
    G.graph["_tnfr_defaults_attached"] = True
    try:  # local import para evitar dependencia circular
        from ..operators import _ensure_node_offset_map

        _ensure_node_offset_map(G)
    except ImportError:
        pass


def merge_overrides(G, **overrides) -> None:
    """Aplica cambios puntuales a ``G.graph``."""
    G.graph.update(overrides)


def get_param(G, key: str):
    """Recupera parámetro desde ``G.graph`` resolviendo aliases legados."""
    if key in G.graph:
        return G.graph[key]
    for alias, targets in ALIASES.items():
        if key in targets and alias in G.graph:
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
