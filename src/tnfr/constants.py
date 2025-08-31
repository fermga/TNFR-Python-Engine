"""
constants.py — TNFR canónica

Centraliza parámetros por defecto organizados en sub-diccionarios temáticos.
Provee utilidades para inyectarlos en ``G.graph``.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
from types import MappingProxyType
import math
import warnings

# -------------------------
# Dataclasses de defaults
# -------------------------


@dataclass(frozen=True)
class CoreDefaults:
    DT: float = 1.0
    INTEGRATOR_METHOD: str = "euler"
    DT_MIN: float = 0.1
    EPI_MIN: float = -1.0
    EPI_MAX: float = 1.0
    EPI_MAX_GLOBAL: float = 1.0
    VF_MIN: float = 0.0
    VF_MAX: float = 1.0
    THETA_WRAP: bool = True
    DNFR_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {"phase": 0.34, "epi": 0.33, "vf": 0.33, "topo": 0.0}
    )
    SI_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {"alpha": 0.34, "beta": 0.33, "gamma": 0.33}
    )
    PHASE_K_GLOBAL: float = 0.05
    PHASE_K_LOCAL: float = 0.15
    PHASE_ADAPT: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "R_hi": 0.90,
            "R_lo": 0.60,
            "disr_hi": 0.50,
            "disr_lo": 0.25,
            "kG_min": 0.01,
            "kG_max": 0.20,
            "kL_min": 0.05,
            "kL_max": 0.25,
            "up": 0.10,
            "down": 0.07,
        }
    )
    UM_COMPAT_THRESHOLD: float = 0.75
    GLYPH_HYSTERESIS_WINDOW: int = 7
    AL_MAX_LAG: int = 5
    EN_MAX_LAG: int = 3
    GLYPH_SELECTOR_MARGIN: float = 0.05
    VF_ADAPT_TAU: int = 5
    VF_ADAPT_MU: float = 0.1
    GLYPH_FACTORS: Dict[str, float] = field(
        default_factory=lambda: {
            "AL_boost": 0.05,
            "EN_mix": 0.25,
            "IL_dnfr_factor": 0.7,
            "OZ_dnfr_factor": 1.3,
            "UM_theta_push": 0.25,
            "RA_epi_diff": 0.15,
            "SHA_vf_factor": 0.85,
            "VAL_scale": 1.15,
            "NUL_scale": 0.85,
            "THOL_accel": 0.10,
            "ZHIR_theta_shift": 1.57079632679,
            "NAV_jitter": 0.05,
            "NAV_eta": 0.5,
            "REMESH_alpha": 0.5,
        }
    )
    GLYPH_THRESHOLDS: Dict[str, float] = field(
        default_factory=lambda: {"hi": 0.66, "lo": 0.33, "dnfr": 1e-3}
    )
    NAV_RANDOM: bool = True
    NAV_STRICT: bool = False
    RANDOM_SEED: int = 0
    OZ_NOISE_MODE: bool = False
    OZ_SIGMA: float = 0.1
    GRAMMAR: Dict[str, Any] = field(
        default_factory=lambda: {
            "window": 3,
            "avoid_repeats": ["ZHIR", "OZ", "THOL"],
            "force_dnfr": 0.60,
            "force_accel": 0.60,
            "fallbacks": {"ZHIR": "NAV", "OZ": "ZHIR", "THOL": "NAV"},
        }
    )
    SELECTOR_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {"w_si": 0.5, "w_dnfr": 0.3, "w_accel": 0.2}
    )
    SELECTOR_THRESHOLDS: Dict[str, float] = field(
        default_factory=lambda: {
            "si_hi": 0.66,
            "si_lo": 0.33,
            "dnfr_hi": 0.50,
            "dnfr_lo": 0.10,
            "accel_hi": 0.50,
            "accel_lo": 0.10,
        }
    )
    GAMMA: Dict[str, Any] = field(
        default_factory=lambda: {"type": "none", "beta": 0.0, "R0": 0.0}
    )
    CALLBACKS_STRICT: bool = False
    VALIDATORS_STRICT: bool = False


@dataclass(frozen=True)
class InitDefaults:
    INIT_RANDOM_PHASE: bool = True
    INIT_THETA_MIN: float = -math.pi
    INIT_THETA_MAX: float = math.pi
    INIT_VF_MODE: str = "uniform"
    INIT_VF_MIN: float | None = None
    INIT_VF_MAX: float | None = None
    INIT_VF_MEAN: float = 0.5
    INIT_VF_STD: float = 0.15
    INIT_VF_CLAMP_TO_LIMITS: bool = True


@dataclass(frozen=True)
class RemeshDefaults:
    EPS_DNFR_STABLE: float = 1e-3
    EPS_DEPI_STABLE: float = 1e-3
    FRACTION_STABLE_REMESH: float = 0.80
    REMESH_COOLDOWN_VENTANA: int = 20
    REMESH_COOLDOWN_TS: float = 0.0
    REMESH_REQUIRE_STABILITY: bool = True
    REMESH_STABILITY_WINDOW: int = 25
    REMESH_MIN_PHASE_SYNC: float = 0.85
    REMESH_MAX_GLYPH_DISR: float = 0.35
    REMESH_MIN_SIGMA_MAG: float = 0.50
    REMESH_MIN_KURAMOTO_R: float = 0.80
    REMESH_MIN_SI_HI_FRAC: float = 0.50
    REMESH_LOG_EVENTS: bool = True
    REMESH_MODE: str = "knn"
    REMESH_COMMUNITY_K: int = 2
    REMESH_TAU_GLOBAL: int = 8
    REMESH_TAU_LOCAL: int = 4
    REMESH_ALPHA: float = 0.5
    REMESH_ALPHA_HARD: bool = False


@dataclass(frozen=True)
class MetricDefaults:
    PHASE_HISTORY_MAXLEN: int = 50
    STOP_EARLY: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "window": 25, "fraction": 0.90}
    )
    SIGMA: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "weight": "Si",      # "Si" | "EPI" | "1"
            "smooth": 0.0,        # EMA sobre el vector global (0=off)
            "history_key": "sigma_global",
            "per_node": False,
        }
    )
    TRACE: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "capture": [
                "gamma",
                "grammar",
                "selector",
                "dnfr_weights",
                "si_weights",
                "callbacks",
                "thol_state",
                "sigma",
                "kuramoto",
                "glifo_counts",
            ],
            "history_key": "trace_meta",
        }
    )
    METRICS: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "save_by_node": True,
            "normalize_series": False,
        }
    )
    GRAMMAR_CANON: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "zhir_requires_oz_window": 3,
            "zhir_dnfr_min": 0.05,
            "thol_min_len": 2,
            "thol_max_len": 6,
            "thol_close_dnfr": 0.15,
            "si_high": 0.66,
        }
    )
    COHERENCE: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "scope": "neighbors",
            "weights": {"phase": 0.34, "epi": 0.33, "vf": 0.20, "si": 0.13},
            "self_on_diag": True,
            "store_mode": "sparse",
            "threshold": 0.0,
            "history_key": "W_sparse",
            "Wi_history_key": "W_i",
            "stats_history_key": "W_stats",
        }
    )
    DIAGNOSIS: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "window": 16,
            "history_key": "nodal_diag",
            "stable": {"Rloc_hi": 0.80, "dnfr_lo": 0.20, "persist": 3},
            "dissonance": {"Rloc_lo": 0.40, "dnfr_hi": 0.50, "persist": 3},
            "transition": {"persist": 2},
            "compute_symmetry": True,
            "include_typology": False,
            "advice": {
                "stable": ["Coherencia", "Acoplamiento", "Resonancia"],
                "transition": ["Transición", "Resonancia", "Autoorganización"],
                "dissonant": ["Silencio", "Contracción", "Mutación"],
            },
        }
    )
    EPI_SUPPORT_THR: float = 0.05
    GLYPH_LOAD_WINDOW: int = 50
    WBAR_WINDOW: int = 25


# -------------------------
# Construcción de diccionarios
# -------------------------
CORE_DEFAULTS = asdict(CoreDefaults())
INIT_DEFAULTS = asdict(InitDefaults())
REMESH_DEFAULTS = asdict(RemeshDefaults())
_metric_defaults_obj = MetricDefaults()
METRIC_DEFAULTS = asdict(_metric_defaults_obj)

# Mapping proxies para acceso de solo lectura
SIGMA = MappingProxyType(METRIC_DEFAULTS["SIGMA"])
TRACE = MappingProxyType(METRIC_DEFAULTS["TRACE"])
METRICS = MappingProxyType(METRIC_DEFAULTS["METRICS"])
GRAMMAR_CANON = MappingProxyType(METRIC_DEFAULTS["GRAMMAR_CANON"])
COHERENCE = MappingProxyType(METRIC_DEFAULTS["COHERENCE"])
DIAGNOSIS = MappingProxyType(METRIC_DEFAULTS["DIAGNOSIS"])

# Diccionario combinado exportado
DEFAULTS: Dict[str, Any] = {}
DEFAULTS.update(CORE_DEFAULTS)
DEFAULTS.update(INIT_DEFAULTS)
DEFAULTS.update(REMESH_DEFAULTS)
DEFAULTS.update(METRIC_DEFAULTS)


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
    """Escribe ``DEFAULTS`` combinados en ``G.graph``.

    Si ``override`` es ``True`` se sobreescriben valores ya presentes.
    """
    inject_defaults(G, DEFAULTS, override=override)


def inject_defaults(
    G, defaults: Dict[str, Any] = DEFAULTS, override: bool = False
) -> None:
    """Inyecta ``defaults`` en ``G.graph``.

    ``defaults`` suele ser ``DEFAULTS``, que combina todos los sub-diccionarios.
    Si ``override`` es ``True`` se sobreescriben valores ya presentes.
    """
    G.graph.setdefault("_tnfr_defaults_attached", False)
    for k, v in defaults.items():
        if override or k not in G.graph:
            G.graph[k] = v
    G.graph["_tnfr_defaults_attached"] = True
    try:  # local import to evitar dependencia circular
        from .operators import _ensure_node_offset_map

        _ensure_node_offset_map(G)
    except ImportError:
        pass


def merge_overrides(G, **overrides) -> None:
    """Aplica cambios puntuales a ``G.graph``."""
    for k, v in overrides.items():
        G.graph[k] = v


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
    return DEFAULTS[key]


# Alias exportados por conveniencia (evita imports circulares)
ALIAS_VF = ("νf", "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA = ("θ", "theta", "fase", "phi", "phase")
ALIAS_DNFR = ("ΔNFR", "delta_nfr", "dnfr")
ALIAS_EPI = ("EPI", "psi", "PSI", "value")
ALIAS_EPI_KIND = ("EPI_kind", "epi_kind", "source_glifo")
ALIAS_SI = ("Si", "sense_index", "S_i", "sense", "meaning_index")
ALIAS_dEPI = ("dEPI_dt", "dpsi_dt", "dEPI", "velocity")
ALIAS_D2EPI = ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel")
ALIAS_dVF = ("dνf_dt", "dvf_dt", "dnu_dt", "dvf")
ALIAS_D2VF = ("d2νf_dt2", "d2vf_dt2", "d2nu_dt2", "B")
ALIAS_dSI = ("δSi", "delta_Si", "dSi")
