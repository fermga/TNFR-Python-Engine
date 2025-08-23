"""
constants.py — TNFR canónica

Propósito
---------
Centraliza valores por defecto usados por los módulos TNFR y ofrece utilidades
para inyectarlos en el grafo (G.graph). No es obligatorio: `helpers.compute_Si`
y `dynamics.step` funcionan con sus propios defaults, pero si llamas a
`attach_defaults(G)` todos leerán estos valores de un único lugar.

Qué expone
----------
- DEFAULTS: diccionario con parámetros canónicos.
- attach_defaults(G, override=False): escribe DEFAULTS en G.graph.
- merge_overrides(G, **overrides): aplica cambios puntuales a G.graph.

Claves principales (G.graph)
----------------------------
- "Si_weights": {"alpha", "beta", "gamma"} → ponderaciones de Si (suman 1).
- "tau": entero ≥ 0 para RE’MESH.
- "dt": tamaño de paso sugerido (dynamics.run/step lo aceptan como parámetro).
- "dnfr_mix": pesos para ΔNFR por defecto (phase vs epi).
- "glyph_selector": callback opcional para selección glífica.
- "glyph_factors": factores suaves usados por operators.py (si deseas que los
  operadores lean desde aquí, basta con consultarlos en cada op_*).

Uso mínimo
---------
>>> import networkx as nx
>>> from constants import attach_defaults
>>> G = nx.Graph()
>>> attach_defaults(G)  # inyecta DEFAULTS sin sobreescribir lo ya definido

Autor: TNFR | Teoría de la naturaleza fractal resonante
"""
from __future__ import annotations

from typing import Dict, Any
import math

# ------------------
# Valores por defecto
# ------------------
DEFAULTS: Dict[str, Any] = {
    # Índice de sentido (Mejora 1)
    "Si_weights": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2},

    # Dinámica / integración
    "dt": 1.0,
    "tau": 1,  # usado por RE’MESH en operators

    # Mezcla para ΔNFR por defecto (dynamics.default_compute_delta_nfr)
    # Nota: dynamics.py actual usa 0.6/0.4 de forma interna; si quieres que lea
    # estos pesos, bastará con consultar G.graph["dnfr_mix"].
    "dnfr_mix": {"phase_weight": 0.6, "epi_weight": 0.4},

    # Selector glífico (callback opcional)
    "glyph_selector": None,

    # Factores suaves para glifos (por si los operadores los consultan)
    "glyph_factors": {
        # I’L
        "IL_dnfr_factor": 0.7,          # multiplica ΔNFR (reducción)
        "IL_dEPI_dt_factor": 0.6,      # amortigua derivada
        # O’Z
        "OZ_scale": 1.25,
        "OZ_bias": 0.10,
        "OZ_floor_nu_ratio": 0.8,      # lleva ΔNFR al 80% de νf si quedó muy bajo
        # U’M
        "UM_phase_step": 0.35,          # paso de sincronización hacia la media
        # R’A
        "RA_gain": 0.20,                # refuerzo de EPI proporcional a sincronía
        # SH’A
        "SHA_nu_factor": 0.50,
        "SHA_dEPI_dt_factor": 0.25,
        # VA’L / NU’L
        "VAL_scale": 1.15,
        "NUL_scale": 0.92,
        "NUL_supp_scale": 0.95,
        # T’HOL
        "THOL_dEPI_dt_factor": 0.50,
        "THOL_attractor_gain": 0.30,
        # Z’HIR
        "ZHIR_phase_rot": 0.5 * math.pi,
        "ZHIR_nu_mix_a": 0.90,         # ν' = a*ν + b*(ν + c)
        "ZHIR_nu_mix_b": 0.10,
        "ZHIR_nu_mix_c": 0.25,
        # NA’V
        "NAV_rate": 0.40,
        # RE’MESH
        "REMESH_gain": 0.50,
    },

    # Umbrales del selector básico sesgado por Si (dynamics.default_glyph_selector)
    "selector_thresholds": {"Si_high": 0.66, "Si_mid": 0.33},
}


# ----------------------------
# Utilidades de inicialización
# ----------------------------

def attach_defaults(G, *, override: bool = False) -> None:
    """Copia DEFAULTS en G.graph. Si override=False, no pisa claves existentes.
    """
    if not hasattr(G, "graph"):
        return
    for k, v in DEFAULTS.items():
        if k not in G.graph or override:
            # copiar profundo simple para dicts
            if isinstance(v, dict):
                G.graph[k] = {**v}
            else:
                G.graph[k] = v


def merge_overrides(G, **overrides) -> None:
    """Mezcla overrides superficiales en G.graph (nivel 1). Ej:
    merge_overrides(G, Si_weights={"alpha":0.5, "beta":0.3, "gamma":0.2})
    """
    if not hasattr(G, "graph"):
        return
    for k, v in overrides.items():
        cur = G.graph.get(k)
        if isinstance(cur, dict) and isinstance(v, dict):
            cur.update(v)
        else:
            G.graph[k] = v


# ----------------------------
# Notas de integración rápida
# ----------------------------
# 1) Llama a attach_defaults(G) tras crear el grafo.
# 2) Si usas tu propio compute_delta_nfr, asígnalo en G.graph["compute_delta_nfr"]
#    antes de correr dynamics.step/run.
# 3) Si quieres centralizar factores de glifos, en operators.py puedes leer:
#       gf = G.graph.get("glyph_factors", {})
#       factor = float(gf.get("IL_dnfr_factor", 0.7))
#    (Actualmente operators.py incluye valores internos; esta centralización es
#     opt-in y retrocompatible.)
