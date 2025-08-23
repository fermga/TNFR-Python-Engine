"""
ontosim.py — Orquestador TNFR (canónico, con histéresis glífica y observador)
-----------------------------------------------------------------------------
Flujo por paso:
  1) ΔNFR de campo (local) con pesos normalizados
  2) Hook Γ(R) opcional (campo de red) + Observador nodal (si existe)
  3) Ecuación nodal: ∂EPI/∂t = νf · ΔNFR  (Euler dt=1)
  4) Selección/aplicación de glifos (con histéresis para OZ/IL) → historial
  5) Clamps canónicos (EPI, νf, θ, ΔNFR, Si)
  6) Coordinación temporal U’M (global + vecinal suaves)
  7) REMESH selectivo y global (si estabilidad sostenida)
  8) Métricas de coherencia (G.graph)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple
import random
import networkx as nx

# Helpers canónicos
from helpers import (
    _safe_float, clamp, clamp01, clamp_abs,
    list_mean, media_vecinal, fase_media,
    push_glifo, reciente_glifo
)

# Operadores y dinámica
from operators import (
    aplicar_glifo, glifo_por_estructura,
    aplicar_remesh_red, aplicar_remesh_si_estabilizacion_global
)
from dynamics import (
    aplicar_clamps_canonicos, aplicar_dnfr_campo, integrar_epi_euler, coordinar_fase_global_vecinal
)

# Constantes
try:
    from constants import (
        EPI_MAX_GLOBAL, VF_MIN, VF_MAX, SI_MIN, SI_MAX, DELTA_NFR_MAX_ABS,
        EPS_EPI_STABLE, EPS_DNFR_STABLE, EPS_DERIV_STABLE, EPS_ACCEL_STABLE,
        FUERZA_GLOBAL_DEF, FUERZA_VECINAL_DEF,
        W_THETA_DEF, W_EPI_DEF, W_VF_DEF,
        FRACTION_STABLE_REMESH, REMESH_COOLDOWN_VENTANA, PASOS_ESTABLES_CONSEC_DEF,
    )
except Exception:
    EPI_MAX_GLOBAL = 1.0
    VF_MIN, VF_MAX = 0.1, 2.0
    SI_MIN, SI_MAX = 0.0, 1.0
    DELTA_NFR_MAX_ABS = 5.0
    EPS_EPI_STABLE = 0.01
    EPS_DNFR_STABLE = 0.05
    EPS_DERIV_STABLE = 0.01
    EPS_ACCEL_STABLE = 0.015
    FUERZA_GLOBAL_DEF, FUERZA_VECINAL_DEF = 0.02, 0.01
    W_THETA_DEF, W_EPI_DEF, W_VF_DEF = 0.5, 0.35, 0.15
    FRACTION_STABLE_REMESH = 0.75
    REMESH_COOLDOWN_VENTANA = 8
    PASOS_ESTABLES_CONSEC_DEF = 6

# Observador
try:
    from observers import ObservadorBase
except Exception:
    class ObservadorBase:  # fallback mínimo
        def step(self, G, t: int, context: Optional[Dict[str, Any]] = None):
            return

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
@dataclass
class ConfigTNFR:
    n: int = 64
    k: int = 4
    seed: int = 7
    topologia: str = "ws"  # "ws" | "erdos" | "grid"
    grid_m: int = 8
    grid_n: int = 8

    # Pesos ΔNFR
    w_theta: float = W_THETA_DEF
    w_epi: float = W_EPI_DEF
    w_vf: float = W_VF_DEF

    # Coordinación U’M
    fuerza_global: float = FUERZA_GLOBAL_DEF
    fuerza_vecinal: float = FUERZA_VECINAL_DEF

    # Histéresis glífica
    ventana_histeresis_glifos: int = 8  # evita repetir OZ/IL
    pasos_estables_consecutivos: int = PASOS_ESTABLES_CONSEC_DEF

    # Clamps por nodo
    epi_max_global: float = EPI_MAX_GLOBAL  # copia a _EPI_MAX por nodo

    # Hooks opcionales
    gamma_hook: Optional[Callable[[nx.Graph], None]] = None
    observador: Optional[ObservadorBase] = None

# -----------------------------------------------------------------------------
# Construcción e inicialización de la red
# -----------------------------------------------------------------------------

def construir_red(cfg: ConfigTNFR) -> nx.Graph:
    rng = random.Random(cfg.seed)
    if cfg.topologia == "ws":
        G = nx.watts_strogatz_graph(cfg.n, cfg.k, 0.1, seed=cfg.seed)
    elif cfg.topologia == "erdos":
        p = min(1.0, cfg.k/max(cfg.n-1,1))
        G = nx.erdos_renyi_graph(cfg.n, p, seed=cfg.seed)
    elif cfg.topologia == "grid":
        G = nx.grid_2d_graph(cfg.grid_m, cfg.grid_n)
        mapping = {node:i for i,node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    else:
        raise ValueError(f"Topología desconocida: {cfg.topologia}")
    return G


def _init_nodo_random(nd: Dict[str, Any], rng: random.Random, epi_max: float) -> None:
    nd["EPI"] = float(1.0 + 0.2*(rng.random()-0.5))
    nd["νf"]  = float(1.0 + 0.4*(rng.random()-0.5))
    nd["θ"]   = float(rng.random())
    nd["ΔNFR"] = 0.0
    nd["Si"]  = float(rng.random())
    nd["dEPI_dt"] = 0.0
    nd["d2EPI_dt2"] = 0.0
    nd["hist_glifos"] = []
    nd["_EPI_MAX"] = float(epi_max)


def inicializar(cfg: ConfigTNFR) -> nx.Graph:
    G = construir_red(cfg)
    rng = random.Random(cfg.seed)
    for _, nd in G.nodes(data=True):
        _init_nodo_random(nd, rng, cfg.epi_max_global)
    G.graph["t"] = 0
    G.graph["_consec_estables"] = 0
    return G

# -----------------------------------------------------------------------------
# Paso de simulación
# -----------------------------------------------------------------------------

def paso(G: nx.Graph, cfg: ConfigTNFR) -> nx.Graph:
    # 1) ΔNFR de campo
    aplicar_dnfr_campo(G, w_theta=cfg.w_theta, w_epi=cfg.w_epi, w_vf=cfg.w_vf)

    # 2) Hooks: campo Γ(R) y observador
    if cfg.gamma_hook is not None:
        try:
            cfg.gamma_hook(G)
        except Exception:
            pass
    if cfg.observador is not None:
        try:
            cfg.observador.step(G, int(G.graph.get("t", 0)), context=None)
        except Exception:
            pass

    # 3) Integración nodal
    integrar_epi_euler(G)

    # 4) Selección/aplicación de glifos con histéresis para OZ/IL
    for n in G.nodes():
        nd = G.nodes[n]
        glifo = glifo_por_estructura(nd)
        # histéresis simple
        if glifo in ("OZ", "IL") and reciente_glifo(nd, glifo, cfg.ventana_histeresis_glifos):
            glifo = "UM"  # alterna con acoplamiento si estaba muy reciente
        aplicar_glifo(nd, glifo)

    # 5) Clamps canónicos
    for _, nd in G.nodes(data=True):
        aplicar_clamps_canonicos(nd)

    # 6) Coordinación U’M
    coordinar_fase_global_vecinal(G, fuerza_global=cfg.fuerza_global, fuerza_vecinal=cfg.fuerza_vecinal)

    # 7) REMESH local/global si corresponde
    aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos=cfg.pasos_estables_consecutivos)

    # 8) Métricas globales simples
    _actualizar_metricas(G)

    # avanzar tiempo
    G.graph["t"] = int(G.graph.get("t", 0)) + 1
    return G


def _actualizar_metricas(G: nx.Graph) -> None:
    epis = [_safe_float(nd.get("EPI", 0.0), 0.0) for _, nd in G.nodes(data=True)]
    dnfrs = [abs(_safe_float(nd.get("ΔNFR", 0.0), 0.0)) for _, nd in G.nodes(data=True)]
    d1s = [abs(_safe_float(nd.get("dEPI_dt", 0.0), 0.0)) for _, nd in G.nodes(data=True)]
    d2s = [abs(_safe_float(nd.get("d2EPI_dt2", 0.0), 0.0)) for _, nd in G.nodes(data=True)]
    G.graph["C_epi_mean"] = list_mean(epis, 0.0)
    G.graph["C_dnfr_mean"] = list_mean(dnfrs, 0.0)
    G.graph["C_d1_mean"] = list_mean(d1s, 0.0)
    G.graph["C_d2_mean"] = list_mean(d2s, 0.0)

# -----------------------------------------------------------------------------
# API de corrida
# -----------------------------------------------------------------------------

def run(cfg: ConfigTNFR, pasos: int = 32) -> nx.Graph:
    G = inicializar(cfg)
    for _ in range(int(max(0, pasos))):
        paso(G, cfg)
    return G
