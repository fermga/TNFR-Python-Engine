"""
constants.py — TNFR canónico (umbrales centralizados)

Este módulo concentra constantes y mapas glíficos de uso transversal.
Notas canónicas:
- θ es FASE estructural (sincronía), no un umbral.
- Umbrales de estabilidad se centralizan aquí para evitar "magic numbers".
- Se incluyen mapas de categorías y códigos para compatibilidad con historiales antiguos.
"""
from __future__ import annotations

# -------------------------------
# Categorías glíficas (tabla operativa TNFR)
# -------------------------------
glifo_categoria = {
    "AL": "activador",      # Emisión
    "EN": "receptor",       # Recepción
    "IL": "estabilizador",  # Coherencia
    "OZ": "disonante",      # Disonancia
    "UM": "acoplador",      # Acoplamiento
    "RA": "resonador",      # Resonancia
    "SHA": "latente",       # Silencio
    "VAL": "expansivo",     # Expansión
    "NUL": "contractivo",   # Contracción
    "THOL": "autoorganizador", # Autoorganización
    "ZHIR": "mutante",      # Mutación
    "NAV": "transicional",  # Nacimiento/Transición
    "REMESH": "recursivo",  # Recursividad/Memoria estructural
}

# -------------------------------
# Límites globales
# -------------------------------
EPI_MAX_GLOBAL = 3.5
SI_MIN, SI_MAX = 0.0, 1.0
VF_MIN, VF_MAX = 0.1, 5.0
DELTA_NFR_MAX_ABS = 5.0

# -------------------------------
# Umbrales de estabilidad y criterios globales (canónicos)
# -------------------------------
# Estabilidad de forma y reorganización (usados para REMESH, etc.)
EPS_EPI_STABLE = 0.01       # |EPI - EPI_prev| < EPS_EPI_STABLE
EPS_DNFR_STABLE = 0.05      # |ΔNFR| < EPS_DNFR_STABLE
EPS_DERIV_STABLE = 0.01     # |dEPI/dt| < EPS_DERIV_STABLE
EPS_ACCEL_STABLE = 0.01     # |d²EPI/dt²| < EPS_ACCEL_STABLE

# Fracción mínima de nodos estables para desencadenar REMESH global
FRACTION_STABLE_REMESH = 0.8

# -------------------------------
# Pesos glíficos para el índice de sentido (Si)
# -------------------------------
# Estos pesos expresan la función estructural relativa de cada glifo sobre Si.
PESOS_GLIFO = {
    "AL": 1.0,
    "EN": 1.1,
    "IL": 1.3,
    "OZ": 0.6,
    "UM": 1.2,
    "RA": 1.5,
    "SHA": 0.4,
    "VAL": 1.4,
    "NUL": 0.8,
    "THOL": 1.6,
    "ZHIR": 1.7,
    "NAV": 1.0,
    "REMESH": 1.3,
    "NINGUNO": 1.0,
}

# -------------------------------
# Códigos glíficos (compatibilidad con historiales antiguos)
# -------------------------------
CODIGO_GLIFO = {
    "AL": 1, "EN": 2, "IL": 3, "OZ": 4, "UM": 5, "RA": 6, "SHA": 7,
    "VAL": 8, "NUL": 9, "THOL": 10, "ZHIR": 11, "NAV": 12, "REMESH": 13,
}
CODIGO_GLIFO_INV = {v: k for k, v in CODIGO_GLIFO.items()}

__all__ = [
    # mapas y códigos
    "glifo_categoria", "PESOS_GLIFO", "CODIGO_GLIFO", "CODIGO_GLIFO_INV",
    # límites globales
    "EPI_MAX_GLOBAL", "SI_MIN", "SI_MAX", "VF_MIN", "VF_MAX", "DELTA_NFR_MAX_ABS",
    # umbrales de estabilidad
    "EPS_EPI_STABLE", "EPS_DNFR_STABLE", "EPS_DERIV_STABLE", "EPS_ACCEL_STABLE",
    "FRACTION_STABLE_REMESH",
]
