"""
constants.py — TNFR canónica

Centraliza parámetros por defecto y nombres que usan el resto de módulos.
Provee utilidades para inyectarlos en G.graph.
"""
from __future__ import annotations
from typing import Dict, Any
from types import MappingProxyType

# -------------------------
# Parámetros canónicos
# -------------------------
DEFAULTS: Dict[str, Any] = {
    # Discretización
    "DT": 1.0,
    "INTEGRATOR_METHOD": "euler",
    "DT_MIN": 0.1,

    # Rango de EPI (estructura primaria)
    "EPI_MIN": -1.0,
    "EPI_MAX": 1.0,
    # Alias histórico usado por algunos proyectos
    "EPI_MAX_GLOBAL": 1.0,

    # Rango de frecuencia estructural νf
    "VF_MIN": 0.0,
    "VF_MAX": 1.0,

    # Envolvente de fase
    "THETA_WRAP": True,

    # --- Inicialización (evitar simetrías) ---
    "INIT_RANDOM_PHASE": True,        # si True, θ ~ U[-π, π]
    "INIT_THETA_MIN": -3.141592653589793,
    "INIT_THETA_MAX":  3.141592653589793,

    "INIT_VF_MODE": "uniform",        # "uniform" | "normal"
    # para uniform:
    "INIT_VF_MIN": None,              # si None → usa VF_MIN
    "INIT_VF_MAX": None,              # si None → usa VF_MAX
    # para normal:
    "INIT_VF_MEAN": 0.5,
    "INIT_VF_STD":  0.15,
    # clamp de νf a límites canónicos
    "INIT_VF_CLAMP_TO_LIMITS": True,


    # Mezcla para ΔNFR (campo nodal)
    # phase: dispersión de fase local; epi: gradiente de EPI; vf: desajuste de νf;
    # topo: término topológico (p. ej., centralidad). Pesos se normalizan.
    "DNFR_WEIGHTS": {"phase": 0.34, "epi": 0.33, "vf": 0.33, "topo": 0.0},

    # Índice de sentido Si = α·νf_norm + β·(1 - disp_fase) + γ·(1 - |ΔNFR|/max)
    "SI_WEIGHTS": {"alpha": 0.34, "beta": 0.33, "gamma": 0.33},

    # Coordinación de fase (U’M) global/vecinal por paso
    "PHASE_K_GLOBAL": 0.05,
    "PHASE_K_LOCAL": 0.15,

    # Coordinación de fase adaptativa (kG/kL dinámicos por estado)
    "PHASE_ADAPT": {
        "enabled": True,     # activar adaptación automática si no se pasan fuerzas explícitas a la función
        "R_hi": 0.90,        # sincronía alta (Kuramoto R) => estado ESTABLE
        "R_lo": 0.60,        # sincronía baja => estado DISONANTE
        "disr_hi": 0.50,     # carga glífica disruptiva alta => DISONANTE
        "disr_lo": 0.25,     # carga glífica disruptiva baja + R alta => ESTABLE
        "kG_min": 0.01, "kG_max": 0.20,
        "kL_min": 0.05, "kL_max": 0.25,
        "up": 0.10,          # velocidad de subida hacia el objetivo
        "down": 0.07         # velocidad de bajada hacia el objetivo
    },

    "STOP_EARLY": {"enabled": False, "window": 25, "fraction": 0.90},

    # Criterios de estabilidad (para activar RE’MESH de red)
    "EPS_DNFR_STABLE": 1e-3,
    "EPS_DEPI_STABLE": 1e-3,
    "FRACTION_STABLE_REMESH": 0.80,   # fracción de nodos estables requerida
    "REMESH_COOLDOWN_VENTANA": 20,    # pasos mínimos entre RE’MESH
    "REMESH_COOLDOWN_TS": 0.0,        # cooldown adicional por tiempo simulado
    # Gating adicional basado en observadores (conmutador + ventana)
    "REMESH_REQUIRE_STABILITY": True,  # si True, exige ventana de estabilidad multi-métrica
    "REMESH_STABILITY_WINDOW": 25,     # tamaño de ventana para evaluar estabilidad
    "REMESH_MIN_PHASE_SYNC": 0.85,     # media mínima de sincronía de fase en ventana
    "REMESH_MAX_GLYPH_DISR": 0.35,     # media máxima de carga glífica disruptiva en ventana
    "REMESH_MIN_SIGMA_MAG": 0.50,      # magnitud mínima de σ en ventana
    "REMESH_MIN_KURAMOTO_R": 0.80,    # R de Kuramoto mínimo en ventana
    "REMESH_MIN_SI_HI_FRAC": 0.50,    # fracción mínima de nodos con Si alto
    "REMESH_LOG_EVENTS": True,         # guarda eventos y metadatos del RE’MESH
    "REMESH_MODE": "knn",            # modo de remallado topológico
    "REMESH_COMMUNITY_K": 2,          # conexiones por comunidad

    # RE’MESH: memoria τ y mezcla α (global/local)
    "REMESH_TAU_GLOBAL": 8,           # pasos hacia atrás (escala global)
    "REMESH_TAU_LOCAL": 4,            # pasos hacia atrás (escala local)
    "REMESH_ALPHA": 0.5,              # mezcla con pasado
    "REMESH_ALPHA_HARD": False,       # si True ignora GLYPH_FACTORS['REMESH_alpha']

    # Soporte y norma de la EPI
    "EPI_SUPPORT_THR": 0.05,          # umbral para Supp(EPI)

    # U'M — compatibilidad mínima para crear/reforzar enlaces funcionales
    "UM_COMPAT_THRESHOLD": 0.75,

    # Histéresis glífica
    "GLYPH_HYSTERESIS_WINDOW": 7,

    # Lags máximos sin emisión (A’L) y recepción (E’N)
    "AL_MAX_LAG": 5,
    "EN_MAX_LAG": 3,

    # Margen de histéresis del selector (cuánto "aguanta" sin cambiar glifo si está cerca de un umbral)
    "GLYPH_SELECTOR_MARGIN": 0.05,

    # Ventana para estimar la carga glífica en history/plots
    "GLYPH_LOAD_WINDOW": 50,

    # Tamaño de ventana para coherencia promedio W̄
    "WBAR_WINDOW": 25,

    # Adaptación de frecuencia estructural por coherencia
    "VF_ADAPT_TAU": 5,   # pasos estables antes de ajustar νf
    "VF_ADAPT_MU": 0.1,  # velocidad de ajuste hacia la media vecinal

    # Factores suaves por glifo (operadores)
    "GLYPH_FACTORS": {
        "AL_boost": 0.05,   # A’L — pequeña emisión
        "EN_mix": 0.25,     # E’N — mezcla con vecindad
        "IL_dnfr_factor": 0.7,  # I’L — reduce ΔNFR
        "OZ_dnfr_factor": 1.3,  # O’Z — aumenta ΔNFR
        "UM_theta_push": 0.25,  # U’M — empuje adicional de fase local
        "RA_epi_diff": 0.15,    # R’A — difusión EPI
        "SHA_vf_factor": 0.85,  # SH’A — baja νf
        "VAL_scale": 1.15,      # VA’L — expande EPI
        "NUL_scale": 0.85,      # NU’L — contrae EPI
        "THOL_accel": 0.10,     # T’HOL — acelera (seg. deriv.) si hay umbral
        "ZHIR_theta_shift": 1.57079632679,  # Z’HIR — desplazamiento ~π/2
        "NAV_jitter": 0.05,     # NA’V — pequeña inestabilidad creativa
        "NAV_eta": 0.5,         # NA’V — peso de convergencia hacia νf
        "REMESH_alpha": 0.5,    # RE’MESH — mezcla si no se usa REMESH_ALPHA
    },

    # Umbrales para el selector glífico por defecto
    "GLYPH_THRESHOLDS": {"hi": 0.66, "lo": 0.33, "dnfr": 1e-3},

    # Comportamiento NA’V
    "NAV_RANDOM": True,   # si True, usa jitter aleatorio en [-j, j]; si False, jitter determinista por signo
    "NAV_STRICT": False,  # si True, fuerza ΔNFR ← νf (sin mezcla)
    "RANDOM_SEED": 0,     # semilla base para reproducibilidad del jitter

    # Modo ruido para O’Z
    "OZ_NOISE_MODE": False,  # si True, añade ruido aditivo en ΔNFR
    "OZ_SIGMA": 0.1,         # amplitud del ruido uniforme [-σ, σ]

    # Gramática glífica (suave): evita repetir ciertos glifos salvo que el campo lo exija
    "GRAMMAR": {
        "window": 3,                       # cuántos pasos recientes miramos por nodo
        "avoid_repeats": ["Z’HIR", "O’Z", "T’HOL"],
        "force_dnfr": 0.60,                # si |ΔNFR|_norm ≥ este valor, se permite repetir
        "force_accel": 0.60,               # o si |accel|_norm ≥ este valor
        "fallbacks": {                     # a qué glifo caer si se bloquea el candidato
            "Z’HIR": "NA’V",
            "O’Z":   "Z’HIR",
            "T’HOL": "NA’V"
        }
    },

    # --- Selector multiobjetivo ---
    # Ponderaciones (se reescalan a 1 automáticamente)
    "SELECTOR_WEIGHTS": {"w_si": 0.5, "w_dnfr": 0.3, "w_accel": 0.2},
    # Umbrales hi/lo para decidir categorías (trabajan con |ΔNFR|_norm y |accel|_norm)
    "SELECTOR_THRESHOLDS": {
        "si_hi": 0.66, "si_lo": 0.33,
        "dnfr_hi": 0.50, "dnfr_lo": 0.10,
        "accel_hi": 0.50, "accel_lo": 0.10
    },
    # Callbacks Γ(R)
    "GAMMA": {
        "type": "none",  # 'none' | 'kuramoto_linear' | 'kuramoto_bandpass'
        "beta": 0.0,
        "R0": 0.0,
    },
    "CALLBACKS_STRICT": False,  # si True, un error en callback detiene; si False, se loguea y continúa
    "VALIDATORS_STRICT": False, # si True, alerta si se clampa fuera de rango
}

# Retrocompatibilidad: alias deprecado
DEFAULTS["REMESH_TAU"] = DEFAULTS["REMESH_TAU_GLOBAL"]

# --- Configuraciones por defecto exportadas ---
SIGMA = MappingProxyType(DEFAULTS.setdefault("SIGMA", {
    "enabled": True,
    "weight": "Si",      # "Si" | "EPI" | "1"
    "smooth": 0.0,        # EMA sobre el vector global (0=off)
    "history_key": "sigma_global",   # dónde guardar en G.graph['history']
    "per_node": False,    # si True, guarda trayectoria σ por nodo (más pesado)
}))

TRACE = MappingProxyType(DEFAULTS.setdefault("TRACE", {
    "enabled": True,
    "capture": ["gamma", "grammar", "selector", "dnfr_weights", "si_weights", "callbacks", "thol_state", "sigma", "kuramoto", "glifo_counts"],
    "history_key": "trace_meta",
}))

METRICS = MappingProxyType(DEFAULTS.setdefault("METRICS", {
    "enabled": True,
    "save_by_node": True,     # guarda Tg por nodo (más pesado)
    "normalize_series": False # glifograma normalizado a fracción por paso
}))

# Gramática glífica canónica
GRAMMAR_CANON = MappingProxyType(DEFAULTS.setdefault("GRAMMAR_CANON", {
    "enabled": True,                # activar la gramática canónica
    "zhir_requires_oz_window": 3,   # cuántos pasos atrás buscamos O’Z
    "zhir_dnfr_min": 0.05,          # si |ΔNFR|_norm < este valor, no permitimos Z’HIR sin O’Z
    "thol_min_len": 2,
    "thol_max_len": 6,
    "thol_close_dnfr": 0.15,        # si el campo calma, cerramos con SH’A/NU’L
    "si_high": 0.66,                # umbral para elegir NU’L vs SH’A al cerrar
}))

# --- Coherencia (W) ---
COHERENCE = MappingProxyType(DEFAULTS.setdefault("COHERENCE", {
    "enabled": True,
    "scope": "neighbors",      # "neighbors" | "all"
    "weights": {
        "phase": 0.34,
        "epi": 0.33,
        "vf": 0.20,
        "si": 0.13,
    },
    "self_on_diag": True,      # W_ii = 1.0
    "store_mode": "sparse",   # "sparse" | "dense"
    "threshold": 0.0,
    "history_key": "W_sparse",
    "Wi_history_key": "W_i",
    "stats_history_key": "W_stats",
}))

# --- Diagnóstico nodal ---
DIAGNOSIS = MappingProxyType(DEFAULTS.setdefault("DIAGNOSIS", {
    "enabled": True,
    "window": 16,
    "history_key": "nodal_diag",
    "stable":     {"Rloc_hi": 0.80, "dnfr_lo": 0.20, "persist": 3},
    "dissonance": {"Rloc_lo": 0.40, "dnfr_hi": 0.50, "persist": 3},
    "transition": {"persist": 2},
    "compute_symmetry": True,
    "include_typology": False,
    "advice": {
        "stable":     ["Coherencia", "Acoplamiento", "Resonancia"],
        "transition": ["Transición", "Resonancia", "Autoorganización"],
        "dissonant":  ["Silencio", "Contracción", "Mutación"],
    },
}))


# -------------------------
# Utilidades
# -------------------------

def attach_defaults(G, override: bool = False) -> None:
    """Escribe DEFAULTS en G.graph (sin sobreescribir si override=False)."""
    inject_defaults(G, DEFAULTS, override=override)


def inject_defaults(G, defaults: Dict[str, Any] = DEFAULTS, override: bool = False) -> None:
    """Alias de conveniencia para inyectar ``DEFAULTS`` en ``G.graph``.

    Permite pasar un diccionario de *defaults* alternativo y mantiene la
    semántica de ``attach_defaults`` existente. Si ``override`` es ``True`` se
    sobreescriben valores ya presentes.
    """
    G.graph.setdefault("_tnfr_defaults_attached", False)
    for k, v in defaults.items():
        if override or k not in G.graph:
            G.graph[k] = v
    G.graph["_tnfr_defaults_attached"] = True
    # Prepare deterministic node offset mapping once defaults are attached.
    try:  # local import to avoid circular dependency at module import time
        from .operators import _ensure_node_offset_map

        _ensure_node_offset_map(G)
    except ImportError:
        pass


def merge_overrides(G, **overrides) -> None:
    """Aplica cambios puntuales a G.graph.
    Útil para ajustar pesos sin tocar DEFAULTS globales.
    """
    for k, v in overrides.items():
        G.graph[k] = v


# Alias exportados por conveniencia (evita imports circulares)
ALIAS_VF     = ("νf", "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA  = ("θ", "theta", "fase", "phi", "phase")
ALIAS_DNFR   = ("ΔNFR", "delta_nfr", "dnfr")
ALIAS_EPI    = ("EPI", "psi", "PSI", "value")
ALIAS_EPI_KIND = ("EPI_kind", "epi_kind", "source_glifo")
ALIAS_SI     = ("Si", "sense_index", "S_i", "sense", "meaning_index")
ALIAS_dEPI   = ("dEPI_dt", "dpsi_dt", "dEPI", "velocity")
ALIAS_D2EPI  = ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel")
ALIAS_dVF    = ("dνf_dt", "dvf_dt", "dnu_dt", "dvf")
ALIAS_D2VF   = ("d2νf_dt2", "d2vf_dt2", "d2nu_dt2", "B")
ALIAS_dSI    = ("δSi", "delta_Si", "dSi")
