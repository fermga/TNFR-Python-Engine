"""Shared constants."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Callable, TypeVar, cast

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
from ..types import GraphLike, TNFRConfigValue

T = TypeVar("T")

STATE_STABLE = "stable"
STATE_TRANSITION = "transition"
STATE_DISSONANT = "dissonant"

CANONICAL_STATE_TOKENS = frozenset(
    {STATE_STABLE, STATE_TRANSITION, STATE_DISSONANT}
)

def normalise_state_token(token: str) -> str:
    """Return the canonical English token for ``token``.

    The helper now enforces the English identifiers exclusively. Values that
    do not match the canonical set raise :class:`ValueError` so callers can
    surface explicit migration errors when legacy payloads are encountered.
    """

    if not isinstance(token, str):
        raise TypeError("state token must be a string")

    stripped = token.strip()
    lowered = stripped.lower()

    if stripped in CANONICAL_STATE_TOKENS:
        return stripped

    if lowered in CANONICAL_STATE_TOKENS:
        return lowered

    raise ValueError(
        "state token must be one of 'stable', 'transition', or 'dissonant'"
    )

try:  # pragma: no cover - optional dependency
    from ..utils import ensure_node_offset_map as _ensure_node_offset_map
except ImportError:  # noqa: BLE001 - allow any import error
    _ensure_node_offset_map = None

ensure_node_offset_map: Callable[[GraphLike], None] | None = _ensure_node_offset_map

# Exported sections
DEFAULT_SECTIONS: Mapping[str, Mapping[str, TNFRConfigValue]] = MappingProxyType(
    {
        "core": CORE_DEFAULTS,
        "init": INIT_DEFAULTS,
        "remesh": REMESH_DEFAULTS,
        "metric": METRIC_DEFAULTS,
    }
)

# Combined exported dictionary
# Merge the dictionaries from lowest to highest priority so that
# ``METRIC_DEFAULTS`` overrides the rest, mirroring the previous ``ChainMap``
# behaviour.
DEFAULTS: Mapping[str, TNFRConfigValue] = MappingProxyType(
    CORE_DEFAULTS | INIT_DEFAULTS | REMESH_DEFAULTS | METRIC_DEFAULTS
)

# -------------------------
# Utilities
# -------------------------


def inject_defaults(
    G: GraphLike,
    defaults: Mapping[str, TNFRConfigValue] = DEFAULTS,
    override: bool = False,
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
            G.graph[k] = (
                v
                if _is_immutable(v)
                else cast(TNFRConfigValue, copy.deepcopy(v))
            )
    G.graph["_tnfr_defaults_attached"] = True
    if ensure_node_offset_map is not None:
        ensure_node_offset_map(G)


def merge_overrides(G: GraphLike, **overrides: TNFRConfigValue) -> None:
    """Apply specific changes to ``G.graph``.

    Non-immutable values are deep-copied to avoid shared state with
    :data:`DEFAULTS`.
    """
    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise KeyError(f"Unknown parameter: '{key}'")
        G.graph[key] = (
            value
            if _is_immutable(value)
            else cast(TNFRConfigValue, copy.deepcopy(value))
        )


def get_param(G: GraphLike, key: str) -> TNFRConfigValue:
    """Retrieve a parameter from ``G.graph`` or fall back to defaults."""
    if key in G.graph:
        return G.graph[key]
    if key not in DEFAULTS:
        raise KeyError(f"Unknown parameter: '{key}'")
    return DEFAULTS[key]


def get_graph_param(
    G: GraphLike, key: str, cast: Callable[[object], T] = float
) -> T | None:
    """Return ``key`` from ``G.graph`` applying ``cast``.

    The ``cast`` argument must be a function (e.g. ``float``, ``int``,
    ``bool``). If the stored value is ``None`` it is returned without
    casting.
    """
    val = get_param(G, key)
    return None if val is None else cast(val)


# Canonical keys with ASCII spellings
VF_KEY = "νf"
THETA_KEY = "theta"

# Alias map for node attributes
ALIASES: dict[str, tuple[str, ...]] = {
    "VF": (VF_KEY, "nu_f", "nu-f", "nu", "freq", "frequency"),
    "THETA": (THETA_KEY, "phase"),
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
    "STATE_STABLE",
    "STATE_TRANSITION",
    "STATE_DISSONANT",
    "CANONICAL_STATE_TOKENS",
    "normalise_state_token",
)
