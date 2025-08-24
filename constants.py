"""
constants.py — TNFR canónica

Centraliza parámetros por defecto y nombres que usan el resto de módulos.
Provee utilidades para inyectarlos en G.graph.
"""
from __future__ import annotations
from typing import Dict, Any

# -------------------------
# Parámetros canónicos
# -------------------------
DEFAULTS: Dict[str, Any] = {
    # Discretización
    "DT": 1.0,

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
    # phase: dispersión de fase local; epi: gradiente de EPI; vf: desajuste de νf
    "DNFR_WEIGHTS": {"phase": 0.34, "epi": 0.33, "vf": 0.33},

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
    # Gating adicional basado en observadores (conmutador + ventana)
    "REMESH_REQUIRE_STABILITY": False, # si True, exige ventana de estabilidad multi-métrica
    "REMESH_STABILITY_WINDOW": 25,     # tamaño de ventana para evaluar estabilidad
    "REMESH_MIN_PHASE_SYNC": 0.85,     # media mínima de sincronía de fase en ventana
    "REMESH_MAX_GLYPH_DISR": 0.35,     # media máxima de carga glífica disruptiva en ventana
    "REMESH_LOG_EVENTS": True,         # guarda eventos y metadatos del RE’MESH

    # RE’MESH: memoria τ y mezcla α
    "REMESH_TAU": 8,                  # pasos hacia atrás
    "REMESH_ALPHA": 0.5,              # mezcla con pasado

    # Histéresis glífica
    "GLYPH_HYSTERESIS_WINDOW": 7,

    # Margen de histéresis del selector (cuánto "aguanta" sin cambiar glifo si está cerca de un umbral)
    "GLYPH_SELECTOR_MARGIN": 0.05,

    # Ventana para estimar la carga glífica en history/plots
    "GLYPH_LOAD_WINDOW": 50,

    # Tamaño de ventana para coherencia promedio W̄
    "WBAR_WINDOW": 25,  

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
        "REMESH_alpha": 0.5,    # RE’MESH — mezcla si no se usa REMESH_ALPHA
    },

    # Umbrales para el selector glífico por defecto
    "GLYPH_THRESHOLDS": {"hi": 0.66, "lo": 0.33, "dnfr": 1e-3},

    # Comportamiento NA’V
    "NAV_RANDOM": True,   # si True, usa jitter aleatorio en [-j, j]; si False, jitter determinista por signo
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
    "CALLBACKS_STRICT": False,  # si True, un error en callback detiene; si False, se loguea y continúa
}


# -------------------------
# Utilidades
# -------------------------

def attach_defaults(G, override: bool = False) -> None:
    """Escribe DEFAULTS en G.graph (sin sobreescribir si override=False)."""
    G.graph.setdefault("_tnfr_defaults_attached", False)
    for k, v in DEFAULTS.items():
        if override or k not in G.graph:
            G.graph[k] = v
    G.graph["_tnfr_defaults_attached"] = True


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
ALIAS_SI     = ("Si", "sense_index", "S_i", "sense", "meaning_index")
ALIAS_dEPI   = ("dEPI_dt", "dpsi_dt", "dEPI", "velocity")
ALIAS_D2EPI  = ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel")
