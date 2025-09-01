"""
dynamics.py — TNFR canónica

Bucle de dinámica con la ecuación nodal y utilidades:
    ∂EPI/∂t = νf · ΔNFR(t)
Incluye:
- default_compute_delta_nfr (mezcla de fase/EPI/νf)
- update_epi_via_nodal_equation (Euler explícito)
- aplicar_dnfr_campo, integrar_epi_euler, aplicar_clamps_canonicos
- coordinar_fase_global_vecinal
- default_glyph_selector, step, run
"""
from __future__ import annotations
from typing import Dict, Any, Literal
import math
from collections import deque
import logging

import networkx as nx

logger = logging.getLogger(__name__)

try:  # Optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    logger.error(
        "Fallo al importar numpy, el modo vectorizado no estará disponible",
        exc_info=True,
    )
    np = None  # type: ignore

from .observers import sincronía_fase, carga_glifica, orden_kuramoto
from .sense import sigma_vector
# Importar compute_Si y aplicar_glifo a nivel de módulo evita el coste de
# realizar la importación en cada paso de la dinámica. Como los módulos de
# origen no dependen de ``dynamics``, no se introducen ciclos.
from .operators import aplicar_remesh_si_estabilizacion_global, aplicar_glifo
from .grammar import (
    enforce_canonical_grammar,
    on_applied_glifo,
    AL,
    EN,
)
from .constants import (
    DEFAULTS,
    REMESH_DEFAULTS,
    METRIC_DEFAULTS,
    ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI,
    ALIAS_dEPI, ALIAS_D2EPI, ALIAS_dVF, ALIAS_D2VF, ALIAS_dSI,
    ALIAS_EPI_KIND,
    get_param,
)
from .gamma import eval_gamma
from .helpers import (
     clamp, clamp01, list_mean, angle_diff,
     get_attr, set_attr, get_attr_str, set_attr_str, media_vecinal, fase_media,
     invoke_callbacks, reciente_glifo, set_vf, set_dnfr, compute_Si, normalize_weights
)

# -------------------------
# ΔNFR por defecto (campo) + utilidades de hook/metadata
# -------------------------

def _write_dnfr_metadata(G, *, weights: dict, hook_name: str, note: str | None = None) -> None:
    """Escribe en G.graph un bloque _DNFR_META con la mezcla y el nombre del hook.

    `weights` puede incluir componentes arbitrarias (phase/epi/vf/topo/etc.)."""
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        # si no hay pesos, normalizamos a componentes iguales
        n = max(1, len(weights))
        weights = {k: 1.0 / n for k in weights}
        total = 1.0
    meta = {
        "hook": hook_name,
        "weights_raw": dict(weights),
        "weights_norm": {k: float(v) / total for k, v in weights.items()},
        "components": [k for k, v in weights.items() if float(v) != 0.0],
        "doc": "ΔNFR = Σ w_i·g_i",
    }
    if note:
        meta["note"] = str(note)
    G.graph["_DNFR_META"] = meta
    G.graph["_dnfr_hook_name"] = hook_name  # string friendly


def _configure_dnfr_weights(G) -> dict:
    """Normaliza y almacena los pesos de ΔNFR en ``G.graph['_dnfr_weights']``.

    Utiliza ``G.graph['DNFR_WEIGHTS']`` o los valores por defecto. El resultado
    es un diccionario con las componentes normalizadas para ser reutilizado en
    cada paso de la simulación sin recalcular la mezcla."""
    w = {**DEFAULTS["DNFR_WEIGHTS"], **G.graph.get("DNFR_WEIGHTS", {})}
    weights = normalize_weights(w, ("phase", "epi", "vf", "topo"), default=0.0)
    G.graph["_dnfr_weights"] = weights
    return weights


def _prepare_dnfr_data(G) -> dict:
    """Precalcula datos comunes para las estrategias de ΔNFR."""
    weights = G.graph.get("_dnfr_weights")
    if weights is None:
        weights = _configure_dnfr_weights(G)
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    theta = [get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in nodes]
    epi = [get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes]
    vf = [get_attr(G.nodes[n], ALIAS_VF, 0.0) for n in nodes]
    w_phase = float(weights.get("phase", 0.0))
    w_epi = float(weights.get("epi", 0.0))
    w_vf = float(weights.get("vf", 0.0))
    w_topo = float(weights.get("topo", 0.0))
    degs = dict(G.degree()) if w_topo != 0 else None
    return {
        "weights": weights,
        "nodes": nodes,
        "idx": idx,
        "theta": theta,
        "epi": epi,
        "vf": vf,
        "w_phase": w_phase,
        "w_epi": w_epi,
        "w_vf": w_vf,
        "w_topo": w_topo,
        "degs": degs,
    }


def _compute_dnfr_numpy(G, data) -> None:
    """Estrategia vectorizada usando ``numpy``."""
    if np is None:  # pragma: no cover - check at runtime
        raise RuntimeError("numpy no disponible para la versión vectorizada")
    nodes = data["nodes"]
    if not nodes:
        return
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
    count = A.sum(axis=1)
    theta = np.array(data["theta"], dtype=float)
    epi = np.array(data["epi"], dtype=float)
    vf = np.array(data["vf"], dtype=float)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    x = A @ cos_th
    y = A @ sin_th
    epi_sum = A @ epi
    vf_sum = A @ vf
    w_topo = data["w_topo"]
    if w_topo != 0.0:
        degs = count
        deg_sum = A @ degs
    else:
        degs = deg_sum = None
    mask = count > 0
    th_bar = theta.copy()
    epi_bar = epi.copy()
    vf_bar = vf.copy()
    if np.any(mask):
        th_bar[mask] = np.arctan2(y[mask] / count[mask], x[mask] / count[mask])
        epi_bar[mask] = epi_sum[mask] / count[mask]
        vf_bar[mask] = vf_sum[mask] / count[mask]
        if w_topo != 0.0 and degs is not None:
            deg_bar = degs.copy()
            deg_bar[mask] = deg_sum[mask] / count[mask]
    else:
        deg_bar = degs
    g_phase = np.array(
        [-angle_diff(theta[i], th_bar[i]) / math.pi for i in range(len(nodes))],
        dtype=float,
    )
    g_epi = epi_bar - epi
    g_vf = vf_bar - vf
    if w_topo != 0.0 and degs is not None and deg_bar is not None:
        g_topo = deg_bar - degs
    else:
        g_topo = np.zeros_like(g_phase)
    dnfr = (
        data["w_phase"] * g_phase
        + data["w_epi"] * g_epi
        + data["w_vf"] * g_vf
        + w_topo * g_topo
    )
    for i, n in enumerate(nodes):
        set_dnfr(G, n, float(dnfr[i]))


def _compute_dnfr_loops(G, data) -> None:
    """Estrategia basada en bucles estándar."""
    nodes = data["nodes"]
    idx = data["idx"]
    theta = data["theta"]
    epi = data["epi"]
    vf = data["vf"]
    w_phase = data["w_phase"]
    w_epi = data["w_epi"]
    w_vf = data["w_vf"]
    w_topo = data["w_topo"]
    degs = data["degs"]
    cos_th = [math.cos(t) for t in theta]
    sin_th = [math.sin(t) for t in theta]
    for i, n in enumerate(nodes):
        th_i = theta[i]
        epi_i = epi[i]
        vf_i = vf[i]
        x = y = epi_sum = vf_sum = 0.0
        count = 0
        if w_topo != 0 and degs is not None:
            deg_i = float(degs.get(n, 0))
            deg_sum = 0.0
        for v in G.neighbors(n):
            j = idx[v]
            x += cos_th[j]
            y += sin_th[j]
            epi_sum += epi[j]
            vf_sum += vf[j]
            if w_topo != 0 and degs is not None:
                deg_sum += degs.get(v, deg_i)
            count += 1
        if count:
            th_bar = math.atan2(y / count, x / count)
            epi_bar = epi_sum / count
            vf_bar = vf_sum / count
            if w_topo != 0 and degs is not None:
                deg_bar = deg_sum / count
        else:
            th_bar = th_i
            epi_bar = epi_i
            vf_bar = vf_i
            if w_topo != 0 and degs is not None:
                deg_bar = deg_i
        g_phase = -angle_diff(th_i, th_bar) / math.pi
        g_epi = epi_bar - epi_i
        g_vf = vf_bar - vf_i
        if w_topo != 0 and degs is not None:
            g_topo = deg_bar - deg_i
        else:
            g_topo = 0.0
        dnfr = w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + w_topo * g_topo
        set_dnfr(G, n, dnfr)


def default_compute_delta_nfr(G) -> None:
    """Calcula ΔNFR mezclando gradientes de fase, EPI, νf y un término topológico."""
    data = _prepare_dnfr_data(G)
    _write_dnfr_metadata(
        G,
        weights=data["weights"],
        hook_name="default_compute_delta_nfr",
    )
    if np is not None and G.graph.get("vectorized_dnfr"):
        _compute_dnfr_numpy(G, data)
    else:
        _compute_dnfr_loops(G, data)

def set_delta_nfr_hook(G, func, *, name: str | None = None, note: str | None = None) -> None:
    """Fija un hook estable para calcular ΔNFR. Firma requerida: func(G)->None y debe
    escribir ALIAS_DNFR en cada nodo. Actualiza metadatos básicos en G.graph."""
    G.graph["compute_delta_nfr"] = func
    G.graph["_dnfr_hook_name"] = str(name or getattr(func, "__name__", "custom_dnfr"))
    if "_dnfr_weights" not in G.graph:
        _configure_dnfr_weights(G)
    if note:
        meta = G.graph.get("_DNFR_META", {})
        meta["note"] = str(note)
        G.graph["_DNFR_META"] = meta

# --- Hooks de ejemplo (opcionales) ---
def dnfr_phase_only(G) -> None:
    """Ejemplo: ΔNFR solo desde fase (tipo Kuramoto-like)."""
    for n, nd in G.nodes(data=True):
        th_i = get_attr(nd, ALIAS_THETA, 0.0)
        th_bar = fase_media(G, n)
        g_phase = -angle_diff(th_i, th_bar) / math.pi
        set_dnfr(G, n, g_phase)
    _write_dnfr_metadata(G, weights={"phase": 1.0}, hook_name="dnfr_phase_only", note="Hook de ejemplo.")

def dnfr_epi_vf_mixed(G) -> None:
    """Ejemplo: ΔNFR sin fase, mezclando EPI y νf."""
    for n, nd in G.nodes(data=True):
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)
        epi_bar = media_vecinal(G, n, ALIAS_EPI, default=epi_i)
        g_epi = (epi_bar - epi_i)
        vf_i = get_attr(nd, ALIAS_VF, 0.0)
        vf_bar = media_vecinal(G, n, ALIAS_VF, default=vf_i)
        g_vf = (vf_bar - vf_i)
        set_dnfr(G, n, 0.5*g_epi + 0.5*g_vf)
    _write_dnfr_metadata(G, weights={"phase":0.0, "epi":0.5, "vf":0.5}, hook_name="dnfr_epi_vf_mixed", note="Hook de ejemplo.")


def dnfr_laplacian(G) -> None:
    """Gradiente topológico explícito usando Laplaciano sobre EPI y νf."""
    wE = float(G.graph.get("DNFR_WEIGHTS", {}).get("epi", 0.33))
    wV = float(G.graph.get("DNFR_WEIGHTS", {}).get("vf", 0.33))
    for n, nd in G.nodes(data=True):
        epi = get_attr(nd, ALIAS_EPI, 0.0)
        vf = get_attr(nd, ALIAS_VF, 0.0)
        neigh = list(G.neighbors(n))
        deg = len(neigh) or 1
        epi_bar = sum(get_attr(G.nodes[v], ALIAS_EPI, epi) for v in neigh) / deg
        vf_bar = sum(get_attr(G.nodes[v], ALIAS_VF, vf) for v in neigh) / deg
        g_epi = epi_bar - epi
        g_vf = vf_bar - vf
        set_dnfr(G, n, wE * g_epi + wV * g_vf)
    _write_dnfr_metadata(
        G,
        weights={"epi": wE, "vf": wV},
        hook_name="dnfr_laplacian",
        note="Gradiente topológico",
    )

# -------------------------
# Ecuación nodal
# -------------------------

def update_epi_via_nodal_equation(
    G,
    *,
    dt: float = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
) -> None:
    """Ecuación nodal TNFR.

    Implementa la forma extendida de la ecuación nodal:
        ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)

    Donde:
      - EPI es la Estructura Primaria de Información del nodo.
      - νf es la frecuencia estructural del nodo (Hz_str).
      - ΔNFR(t) es el gradiente nodal (necesidad de reorganización),
        típicamente una mezcla de componentes (p. ej. fase θ, EPI, νf).
      - Γi(R) es el acoplamiento de red opcional en función del orden de Kuramoto R
        (ver gamma.py), usado para modular la integración en red.

    Referencias TNFR: ecuación nodal (manual), glosario νf/ΔNFR/EPI, operador Γ.
    Efectos secundarios: cachea dEPI y actualiza EPI por integración explícita.
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("G must be a networkx graph instance")
    if dt is None:
        dt = float(G.graph.get("DT", DEFAULTS["DT"]))
    else:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a number")
        if dt < 0:
            raise ValueError("dt must be non-negative")
        dt = float(dt)
    if t is None:
        t = float(G.graph.get("_t", 0.0))
    else:
        t = float(t)

    method = (method or G.graph.get("INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler"))).lower()
    dt_min = float(G.graph.get("DT_MIN", DEFAULTS.get("DT_MIN", 0.0)))
    if dt_min > 0 and dt > dt_min:
        steps = int(math.ceil(dt / dt_min))
    else:
        steps = 1
    dt_step = dt / steps if steps else 0.0

    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")

    t_local = t
    for _ in range(steps):
        if method == "rk4":
            t_mid = t_local + dt_step / 2.0
            t_end = t_local + dt_step
            g1_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
            g_mid_map = {n: eval_gamma(G, n, t_mid) for n in G.nodes}
            g4_map = {n: eval_gamma(G, n, t_end) for n in G.nodes}
        else:
            gamma_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}

        for n, nd in G.nodes(data=True):
            vf = get_attr(nd, ALIAS_VF, 0.0)
            dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
            dEPI_dt_prev = get_attr(nd, ALIAS_dEPI, 0.0)
            epi_i = get_attr(nd, ALIAS_EPI, 0.0)

            base = vf * dnfr

            if method == "rk4":
                g1 = g1_map.get(n, 0.0)
                g_mid = g_mid_map.get(n, 0.0)
                g4 = g4_map.get(n, 0.0)
                k1 = base + g1
                k2 = base + g_mid
                k3 = base + g_mid
                k4 = base + g4
                epi = epi_i + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                dEPI_dt = k4
            else:
                gamma_local = gamma_map.get(n, 0.0)
                dEPI_dt = base + gamma_local
                epi = epi_i + dt_step * dEPI_dt

            epi_kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
            set_attr(nd, ALIAS_EPI, epi)
            if epi_kind:
                set_attr_str(nd, ALIAS_EPI_KIND, epi_kind)
            set_attr(nd, ALIAS_dEPI, dEPI_dt)
            set_attr(nd, ALIAS_D2EPI, (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0)

        t_local += dt_step

    G.graph["_t"] = t_local


# -------------------------
# Wrappers nombrados (compatibilidad)
# -------------------------

def aplicar_dnfr_campo(G, w_theta=None, w_epi=None, w_vf=None) -> None:
    if any(v is not None for v in (w_theta, w_epi, w_vf)):
        mix = G.graph.get("DNFR_WEIGHTS", DEFAULTS["DNFR_WEIGHTS"]).copy()
        if w_theta is not None: mix["phase"] = float(w_theta)
        if w_epi is not None: mix["epi"] = float(w_epi)
        if w_vf is not None: mix["vf"] = float(w_vf)
        G.graph["DNFR_WEIGHTS"] = mix
    default_compute_delta_nfr(G)


def integrar_epi_euler(G, dt: float | None = None) -> None:
    update_epi_via_nodal_equation(G, dt=dt, method="euler")


def aplicar_clamps_canonicos(nd: Dict[str, Any], G=None, node=None) -> None:
    eps_min = float((G.graph.get("EPI_MIN") if G is not None else DEFAULTS["EPI_MIN"]))
    eps_max = float((G.graph.get("EPI_MAX") if G is not None else DEFAULTS["EPI_MAX"]))
    vf_min = float((G.graph.get("VF_MIN") if G is not None else DEFAULTS["VF_MIN"]))
    vf_max = float((G.graph.get("VF_MAX") if G is not None else DEFAULTS["VF_MAX"]))

    epi = get_attr(nd, ALIAS_EPI, 0.0)
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th = get_attr(nd, ALIAS_THETA, 0.0)

    strict = bool((G.graph.get("VALIDATORS_STRICT") if G is not None else DEFAULTS.get("VALIDATORS_STRICT", False)))
    if strict and G is not None:
        hist = G.graph.setdefault("history", {}).setdefault("clamp_alerts", [])
        if epi < eps_min or epi > eps_max:
            hist.append({"node": node, "attr": "EPI", "value": float(epi)})
        if vf < vf_min or vf > vf_max:
            hist.append({"node": node, "attr": "VF", "value": float(vf)})

    set_attr(nd, ALIAS_EPI, clamp(epi, eps_min, eps_max))
    if G is not None and node is not None:
        set_vf(G, node, clamp(vf, vf_min, vf_max))
    else:
        set_attr(nd, ALIAS_VF, clamp(vf, vf_min, vf_max))
    if (G.graph.get("THETA_WRAP") if G is not None else DEFAULTS["THETA_WRAP"]):
        # envolver fase
        set_attr(nd, ALIAS_THETA, ((th + math.pi) % (2*math.pi) - math.pi))


def validate_canon(G) -> None:
    """Aplica clamps canónicos a todos los nodos de ``G``.

    Envuelve fase y restringe ``EPI`` y ``νf`` a los rangos en ``G.graph``.
    Si ``VALIDATORS_STRICT`` está activo, registra alertas en ``history``.
    """
    for n in G.nodes():
        aplicar_clamps_canonicos(G.nodes[n], G, n)
    return G


def coordinar_fase_global_vecinal(G, fuerza_global: float | None = None, fuerza_vecinal: float | None = None) -> None:
    """
    Ajusta fase con mezcla GLOBAL+VECINAL.
    Si no se pasan fuerzas explícitas, adapta kG/kL según estado (disonante / transición / estable).
    Estado se decide por R (Kuramoto) y carga glífica disruptiva reciente.
    """
    g = G.graph
    defaults = DEFAULTS
    hist = g.setdefault("history", {})
    maxlen = int(g.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"]))
    hist_state = hist.setdefault("phase_state", deque(maxlen=maxlen))
    if not isinstance(hist_state, deque):
        hist_state = deque(hist_state, maxlen=maxlen)
        hist["phase_state"] = hist_state
    hist_R = hist.setdefault("phase_R", deque(maxlen=maxlen))
    if not isinstance(hist_R, deque):
        hist_R = deque(hist_R, maxlen=maxlen)
        hist["phase_R"] = hist_R
    hist_disr = hist.setdefault("phase_disr", deque(maxlen=maxlen))
    if not isinstance(hist_disr, deque):
        hist_disr = deque(hist_disr, maxlen=maxlen)
        hist["phase_disr"] = hist_disr
    # 0) Si hay fuerzas explícitas, usar y salir del modo adaptativo
    if (fuerza_global is not None) or (fuerza_vecinal is not None):
        kG = float(
            fuerza_global
            if fuerza_global is not None
            else g.get("PHASE_K_GLOBAL", defaults["PHASE_K_GLOBAL"])
        )
        kL = float(
            fuerza_vecinal
            if fuerza_vecinal is not None
            else g.get("PHASE_K_LOCAL", defaults["PHASE_K_LOCAL"])
        )
    else:
        # 1) Lectura de configuración
        cfg = g.get("PHASE_ADAPT", defaults.get("PHASE_ADAPT", {}))
        kG = float(g.get("PHASE_K_GLOBAL", defaults["PHASE_K_GLOBAL"]))
        kL = float(g.get("PHASE_K_LOCAL", defaults["PHASE_K_LOCAL"]))

        if bool(cfg.get("enabled", False)):
            # 2) Métricas actuales (no dependemos de history)
            R = orden_kuramoto(G)
            win = int(g.get("GLYPH_LOAD_WINDOW", METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"]))
            dist = carga_glifica(G, window=win)
            disr = float(dist.get("_disruptivos", 0.0)) if dist else 0.0

            # 3) Decidir estado
            R_hi = float(cfg.get("R_hi", 0.90)); R_lo = float(cfg.get("R_lo", 0.60))
            disr_hi = float(cfg.get("disr_hi", 0.50)); disr_lo = float(cfg.get("disr_lo", 0.25))
            if (R >= R_hi) and (disr <= disr_lo):
                state = "estable"
            elif (R <= R_lo) or (disr >= disr_hi):
                state = "disonante"
            else:
                state = "transicion"

            # 4) Objetivos y actualización suave (con saturación)
            kG_min = float(cfg.get("kG_min", 0.01)); kG_max = float(cfg.get("kG_max", 0.20))
            kL_min = float(cfg.get("kL_min", 0.05)); kL_max = float(cfg.get("kL_max", 0.25))

            if state == "disonante":
                kG_t = kG_max
                kL_t = 0.5 * (kL_min + kL_max)   # local medio para no perder plasticidad
            elif state == "estable":
                kG_t = kG_min
                kL_t = kL_min
            else:
                kG_t = 0.5 * (kG_min + kG_max)
                kL_t = 0.5 * (kL_min + kL_max)

            up = float(cfg.get("up", 0.10))
            down = float(cfg.get("down", 0.07))

            def _step(curr, target, mn, mx):
                gain = up if target > curr else down
                nxt = curr + gain * (target - curr)
                return max(mn, min(mx, nxt))

            kG = _step(kG, kG_t, kG_min, kG_max)
            kL = _step(kL, kL_t, kL_min, kL_max)

            # 5) Persistir en G.graph y log de serie
            hist_state.append(state)
            hist_R.append(float(R))
            hist_disr.append(float(disr))

    g["PHASE_K_GLOBAL"] = kG
    g["PHASE_K_LOCAL"] = kL
    hist.setdefault("phase_kG", []).append(float(kG))
    hist.setdefault("phase_kL", []).append(float(kL))

    # 6) Fase GLOBAL (centroide) para empuje
    x_sum = y_sum = 0.0
    for n in G.nodes():
        th = get_attr(G.nodes[n], ALIAS_THETA, 0.0)
        x_sum += math.cos(th)
        y_sum += math.sin(th)
    num_nodes = G.number_of_nodes()
    if num_nodes:
        thG = math.atan2(y_sum / num_nodes, x_sum / num_nodes)
    else:
        thG = 0.0

    # 7) Aplicar corrección global+vecinal
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thL = fase_media(G, n)
        dG = angle_diff(thG, th)
        dL = angle_diff(thL, th)
        set_attr(nd, ALIAS_THETA, th + kG*dG + kL*dL)

# -------------------------
# Adaptación de νf por coherencia
# -------------------------

def adaptar_vf_por_coherencia(G) -> None:
    """Ajusta νf hacia la media vecinal en nodos con estabilidad sostenida."""
    tau = int(G.graph.get("VF_ADAPT_TAU", DEFAULTS.get("VF_ADAPT_TAU", 5)))
    mu = float(G.graph.get("VF_ADAPT_MU", DEFAULTS.get("VF_ADAPT_MU", 0.1)))
    eps_dnfr = float(G.graph.get("EPS_DNFR_STABLE", REMESH_DEFAULTS["EPS_DNFR_STABLE"]))
    thr_sel = G.graph.get("SELECTOR_THRESHOLDS", DEFAULTS.get("SELECTOR_THRESHOLDS", {}))
    thr_def = G.graph.get("GLYPH_THRESHOLDS", DEFAULTS.get("GLYPH_THRESHOLDS", {"hi": 0.66}))
    si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
    vf_min = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))

    updates = {}
    for n, nd in G.nodes(data=True):
        Si = get_attr(nd, ALIAS_SI, 0.0)
        dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0))
        if Si >= si_hi and dnfr <= eps_dnfr:
            nd["stable_count"] = nd.get("stable_count", 0) + 1
        else:
            nd["stable_count"] = 0
            continue

        if nd["stable_count"] >= tau:
            vf = get_attr(nd, ALIAS_VF, 0.0)
            vf_bar = media_vecinal(G, n, ALIAS_VF, default=vf)
            updates[n] = vf + mu * (vf_bar - vf)

    for n, vf_new in updates.items():
        set_vf(G, n, clamp(vf_new, vf_min, vf_max))

# -------------------------
# Umbrales de selector
# -------------------------

def _selector_thresholds(G) -> dict:
    """Retorna umbrales normalizados hi/lo para Si, ΔNFR y aceleración.

    Combina ``SELECTOR_THRESHOLDS`` con ``GLYPH_THRESHOLDS`` (legado) para
    los cortes de Si. Todos los valores se claman a [0,1]."""
    thr_sel = dict(DEFAULTS.get("SELECTOR_THRESHOLDS", {}))
    thr_sel.update(G.graph.get("SELECTOR_THRESHOLDS", {}))
    thr_def = G.graph.get("GLYPH_THRESHOLDS", DEFAULTS.get("GLYPH_THRESHOLDS", {}))

    si_hi = clamp01(float(thr_sel.get("si_hi", thr_def.get("hi", 0.66))))
    si_lo = clamp01(float(thr_sel.get("si_lo", thr_def.get("lo", 0.33))))
    dnfr_hi = clamp01(float(thr_sel.get("dnfr_hi", 0.5)))
    dnfr_lo = clamp01(float(thr_sel.get("dnfr_lo", 0.1)))
    acc_hi = clamp01(float(thr_sel.get("accel_hi", 0.5)))
    acc_lo = clamp01(float(thr_sel.get("accel_lo", 0.1)))

    return {
        "si_hi": si_hi,
        "si_lo": si_lo,
        "dnfr_hi": dnfr_hi,
        "dnfr_lo": dnfr_lo,
        "accel_hi": acc_hi,
        "accel_lo": acc_lo,
    }

# -------------------------
# Selector glífico por defecto
# -------------------------

def default_glyph_selector(G, n) -> str:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    hi, lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]

    norms = G.graph.get("_sel_norms")
    if norms is not None:
        dnfr_max = float(norms.get("dnfr_max", 1.0))
    else:
        dnfr_max = 0.0
        for _, nd2 in G.nodes(data=True):
            dnfr_max = max(dnfr_max, abs(get_attr(nd2, ALIAS_DNFR, 0.0)))
        if dnfr_max <= 0:
            dnfr_max = 1.0

    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max

    if Si >= hi:
        return "IL"
    if Si <= lo:
        return "OZ" if dnfr > dnfr_hi else "ZHIR"
    return "NAV" if dnfr > dnfr_hi else "RA"


# -------------------------
# Selector glífico multiobjetivo (paramétrico)
# -------------------------
def _norms_para_selector(G) -> dict:
    """Calcula y guarda en G.graph los máximos para normalizar |ΔNFR| y |d2EPI/dt2|."""
    dnfr_max = 0.0
    accel_max = 0.0
    for n, nd in G.nodes(data=True):
        dnfr_max = max(dnfr_max, abs(get_attr(nd, ALIAS_DNFR, 0.0)))
        accel_max = max(accel_max, abs(get_attr(nd, ALIAS_D2EPI, 0.0)))
    if dnfr_max <= 0: dnfr_max = 1.0
    if accel_max <= 0: accel_max = 1.0
    norms = {"dnfr_max": float(dnfr_max), "accel_max": float(accel_max)}
    G.graph["_sel_norms"] = norms
    return norms


def _soft_grammar_prefilter(G, n, cand, dnfr, accel):
    """Gramática suave: evita repeticiones antes de la canónica."""
    gram = G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {}))
    gwin = int(gram.get("window", 3))
    avoid = set(gram.get("avoid_repeats", []))
    force_dn = float(gram.get("force_dnfr", 0.60))
    force_ac = float(gram.get("force_accel", 0.60))
    fallbacks = gram.get("fallbacks", {})
    nd = G.nodes[n]
    if cand in avoid and reciente_glifo(nd, cand, gwin):
        if not (dnfr >= force_dn or accel >= force_ac):
            cand = fallbacks.get(cand, cand)
    return cand

def parametric_glyph_selector(G, n) -> str:
    """Multiobjetivo: combina Si, |ΔNFR|_norm y |accel|_norm + histéresis.
    Reglas base:
      - Si alto  ⇒ IL
      - Si bajo  ⇒ OZ si |ΔNFR| alto; ZHIR si |ΔNFR| bajo; THOL si hay mucha aceleración
      - Si medio ⇒ NAV si |ΔNFR| alto (o accel alta), si no RA
    """
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    si_hi, si_lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi, dnfr_lo = thr["dnfr_hi"], thr["dnfr_lo"]
    acc_hi, acc_lo = thr["accel_hi"], thr["accel_lo"]
    margin = float(G.graph.get("GLYPH_SELECTOR_MARGIN", DEFAULTS["GLYPH_SELECTOR_MARGIN"]))

    # Normalizadores por paso
    norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
    dnfr_max = float(norms.get("dnfr_max", 1.0))
    acc_max  = float(norms.get("accel_max", 1.0))

    # Lecturas nodales
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max

    W = G.graph.get("SELECTOR_WEIGHTS", DEFAULTS["SELECTOR_WEIGHTS"])
    w_si = float(W.get("w_si", 0.5)); w_dn = float(W.get("w_dnfr", 0.3)); w_ac = float(W.get("w_accel", 0.2))
    s = max(1e-9, w_si + w_dn + w_ac)
    w_si, w_dn, w_ac = w_si/s, w_dn/s, w_ac/s
    score = w_si*Si + w_dn*(1.0 - dnfr) + w_ac*(1.0 - accel)
    # usar score como desempate/override suave: si score>0.66 ⇒ inclinar a IL; <0.33 ⇒ inclinar a OZ/ZHIR

    # Decisión base
    if Si >= si_hi:
        cand = "IL"
    elif Si <= si_lo:
        if accel >= acc_hi:
            cand = "THOL"
        else:
            cand = "OZ" if dnfr >= dnfr_hi else "ZHIR"
    else:
        # Zona intermedia: transición si el campo "pide" reorganizar (dnfr/accel altos)
        if dnfr >= dnfr_hi or accel >= acc_hi:
            cand = "NAV"
        else:
            cand = "RA"

    # --- Histéresis del selector: si está cerca de umbrales, conserva el glifo reciente ---
    # Medimos "certeza" como distancia mínima a los umbrales relevantes
    d_si = min(abs(Si - si_hi), abs(Si - si_lo))
    d_dn = min(abs(dnfr - dnfr_hi), abs(dnfr - dnfr_lo))
    d_ac = min(abs(accel - acc_hi), abs(accel - acc_lo))
    certeza = min(d_si, d_dn, d_ac)
    if certeza < margin:
        hist = nd.get("hist_glifos")
        if hist:
            prev = hist[-1]
            if isinstance(prev, str) and prev in ("IL","OZ","ZHIR","THOL","NAV","RA"):
                return prev

    # Penalización por falta de avance en σ/Si si se repite glifo
    prev = None
    hist_prev = nd.get("hist_glifos")
    if hist_prev:
        prev = hist_prev[-1]
    if prev == cand:
        delta_si = get_attr(nd, ALIAS_dSI, 0.0)
        h = G.graph.get("history", {})
        sig = h.get("sense_sigma_mag", [])
        delta_sigma = sig[-1] - sig[-2] if len(sig) >= 2 else 0.0
        if delta_si <= 0.0 and delta_sigma <= 0.0:
            score -= 0.05
            
    # Override suave guiado por score (solo si NO cayó la histéresis arriba)
    # Regla: score>=0.66 inclina a IL; score<=0.33 inclina a OZ/ZHIR
    try:
        if score >= 0.66 and cand in ("NAV","RA","ZHIR","OZ"):
            cand = "IL"
        elif score <= 0.33 and cand in ("NAV","RA","IL"):
            cand = "OZ" if dnfr >= dnfr_lo else "ZHIR"
    except NameError:
        pass

    cand = _soft_grammar_prefilter(G, n, cand, dnfr, accel)
    return cand

# -------------------------
# Step / run
# -------------------------

def step(G, *, dt: float | None = None, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    # Contexto inicial
    _hist0 = G.graph.setdefault("history", {"C_steps": []})
    step_idx = len(_hist0.get("C_steps", []))
    invoke_callbacks(G, "before_step", {"step": step_idx, "dt": dt, "use_Si": use_Si, "apply_glyphs": apply_glyphs})

    # 1) ΔNFR (campo)
    compute_dnfr_cb = G.graph.get("compute_delta_nfr", default_compute_delta_nfr)
    compute_dnfr_cb(G)

    # 2) (opcional) Si
    if use_Si:
        compute_Si(G, inplace=True)

    # 2b) Normalizadores para selector paramétrico (por paso)
    selector = G.graph.get("glyph_selector", default_glyph_selector)
    if selector is parametric_glyph_selector:
        _norms_para_selector(G)

    # 3) Selección glífica + aplicación (con lags obligatorios AL/EN)
    if apply_glyphs:
        window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
        use_canon = bool(G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})).get("enabled", False))

        al_max = int(G.graph.get("AL_MAX_LAG", DEFAULTS["AL_MAX_LAG"]))
        en_max = int(G.graph.get("EN_MAX_LAG", DEFAULTS["EN_MAX_LAG"]))
        h_al = _hist0.setdefault("since_AL", {})
        h_en = _hist0.setdefault("since_EN", {})

        for n in G.nodes():
            h_al[n] = int(h_al.get(n, 0)) + 1
            h_en[n] = int(h_en.get(n, 0)) + 1

            if h_al[n] > al_max:
                g = AL
            elif h_en[n] > en_max:
                g = EN
            else:
                g = selector(G, n)
                if use_canon:
                    g = enforce_canonical_grammar(G, n, g)

            aplicar_glifo(G, n, g, window=window)
            if use_canon:
                on_applied_glifo(G, n, g)

            if g == AL:
                h_al[n] = 0
                h_en[n] = min(h_en[n], en_max)
            elif g == EN:
                h_en[n] = 0

    # 4) Ecuación nodal
    _dt = float(G.graph.get("DT", DEFAULTS["DT"])) if dt is None else float(dt)
    method = G.graph.get("INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler"))
    update_epi_via_nodal_equation(G, dt=_dt, method=method)

    # 5) Clamps
    for n in G.nodes():
        aplicar_clamps_canonicos(G.nodes[n], G, n)

    # 6) Coordinación de fase
    coordinar_fase_global_vecinal(G, None, None)

    # 6b) Adaptación de νf por coherencia
    adaptar_vf_por_coherencia(G)

    # 7) Observadores ligeros
    _update_history(G)
    # dynamics.py — dentro de step(), justo antes del punto 8)
    # REMESH_TAU: alias legado resuelto por ``get_param``
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append({n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()})

    # 8) REMESH condicionado
    aplicar_remesh_si_estabilizacion_global(G)

    # 8b) Validadores de invariantes
    from .validators import run_validators
    run_validators(G)

    # Contexto final (últimas métricas del paso)
    h = G.graph.get("history", {})
    ctx = {"step": step_idx}
    if h.get("C_steps"):         ctx["C"] = h["C_steps"][-1]
    if h.get("stable_frac"):     ctx["stable_frac"] = h["stable_frac"][-1]
    if h.get("phase_sync"):      ctx["phase_sync"] = h["phase_sync"][-1]
    if h.get("glyph_load_disr"): ctx["glyph_disr"] = h["glyph_load_disr"][-1]
    if h.get("Si_mean"):         ctx["Si_mean"] = h["Si_mean"][-1]
    invoke_callbacks(G, "after_step", ctx)


def run(G, steps: int, *, dt: float | None = None, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    for _ in range(int(steps)):
        step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
        # Early-stop opcional
        stop_cfg = G.graph.get("STOP_EARLY", METRIC_DEFAULTS.get("STOP_EARLY", {"enabled": False}))
        if stop_cfg and stop_cfg.get("enabled", False):
            w = int(stop_cfg.get("window", 25))
            frac = float(stop_cfg.get("fraction", 0.90))
            hist = G.graph.setdefault("history", {"stable_frac": []})
            series = hist.get("stable_frac", [])
            if len(series) >= w and all(v >= frac for v in series[-w:]):
                break


# -------------------------
# Historial simple
# -------------------------


def _update_coherence(G, hist) -> None:
    """Actualizar la coherencia global y su media móvil."""
    dnfr_mean = list_mean(abs(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes())
    dEPI_mean = list_mean(abs(get_attr(G.nodes[n], ALIAS_dEPI, 0.0)) for n in G.nodes())
    C = 1.0 / (1.0 + dnfr_mean + dEPI_mean)
    hist["C_steps"].append(C)

    wbar_w = int(G.graph.get("WBAR_WINDOW", METRIC_DEFAULTS.get("WBAR_WINDOW", 25)))
    cs = hist["C_steps"]
    if cs:
        w = min(len(cs), max(1, wbar_w))
        wbar = sum(cs[-w:]) / w
        hist.setdefault("W_bar", []).append(wbar)


def _update_phase_sync(G, hist) -> None:
    """Registrar sincronía de fase y el orden de Kuramoto."""
    ps = sincronía_fase(G)
    hist["phase_sync"].append(ps)
    R = orden_kuramoto(G)
    hist.setdefault("kuramoto_R", []).append(R)


def _update_sigma(G, hist) -> None:
    """Registrar carga glífica y el vector Σ⃗ asociado."""
    win = int(G.graph.get("GLYPH_LOAD_WINDOW", METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"]))
    gl = carga_glifica(G, window=win)
    hist["glyph_load_estab"].append(gl.get("_estabilizadores", 0.0))
    hist["glyph_load_disr"].append(gl.get("_disruptivos", 0.0))

    sig = sigma_vector(gl)
    hist.setdefault("sense_sigma_x", []).append(sig.get("x", 0.0))
    hist.setdefault("sense_sigma_y", []).append(sig.get("y", 0.0))
    hist.setdefault("sense_sigma_mag", []).append(sig.get("mag", 0.0))
    hist.setdefault("sense_sigma_angle", []).append(sig.get("angle", 0.0))


def _update_history(G) -> None:
    hist = G.graph.setdefault("history", {})
    for k in (
        "C_steps", "stable_frac", "phase_sync", "glyph_load_estab", "glyph_load_disr",
        "Si_mean", "Si_hi_frac", "Si_lo_frac", "delta_Si", "B"
    ):
        hist.setdefault(k, [])

    _update_coherence(G, hist)

    eps_dnfr = float(G.graph.get("EPS_DNFR_STABLE", REMESH_DEFAULTS["EPS_DNFR_STABLE"]))
    eps_depi = float(G.graph.get("EPS_DEPI_STABLE", REMESH_DEFAULTS["EPS_DEPI_STABLE"]))
    stables = 0
    total = max(1, G.number_of_nodes())
    dt = float(G.graph.get("DT", DEFAULTS.get("DT", 1.0))) or 1.0
    delta_si_sum = 0.0
    delta_si_count = 0
    B_sum = 0.0
    B_count = 0
    for n, nd in G.nodes(data=True):
        if abs(get_attr(nd, ALIAS_DNFR, 0.0)) <= eps_dnfr and abs(get_attr(nd, ALIAS_dEPI, 0.0)) <= eps_depi:
            stables += 1

        # δSi por nodo
        Si_curr = get_attr(nd, ALIAS_SI, 0.0)
        Si_prev = nd.get("_prev_Si", Si_curr)
        dSi = Si_curr - Si_prev
        nd["_prev_Si"] = Si_curr
        set_attr(nd, ALIAS_dSI, dSi)
        delta_si_sum += dSi
        delta_si_count += 1

        # Bifurcación B = ∂²νf/∂t²
        vf_curr = get_attr(nd, ALIAS_VF, 0.0)
        vf_prev = nd.get("_prev_vf", vf_curr)
        dvf_dt = (vf_curr - vf_prev) / dt
        dvf_prev = nd.get("_prev_dvf", dvf_dt)
        B = (dvf_dt - dvf_prev) / dt
        nd["_prev_vf"] = vf_curr
        nd["_prev_dvf"] = dvf_dt
        set_attr(nd, ALIAS_dVF, dvf_dt)
        set_attr(nd, ALIAS_D2VF, B)
        B_sum += B
        B_count += 1

    hist["stable_frac"].append(stables/total)
    hist["delta_Si"].append(delta_si_sum / delta_si_count if delta_si_count else 0.0)
    hist["B"].append(B_sum / B_count if B_count else 0.0)
    try:
        _update_phase_sync(G, hist)
        _update_sigma(G, hist)
        if hist.get("C_steps") and hist.get("stable_frac"):
            hist.setdefault("iota", []).append(
                hist["C_steps"][-1] * hist["stable_frac"][-1]
            )
    except (KeyError, ValueError, TypeError):
        # observadores son opcionales; si fallan se ignoran
        pass
  
    # --- nuevas series: Si agregado (media y colas) ---
    try:
        sis = []
        for n in G.nodes():
            sis.append(get_attr(G.nodes[n], ALIAS_SI, float("nan")))
        sis = [s for s in sis if not math.isnan(s)]
        if sis:
            si_mean = list_mean(sis, 0.0)
            hist["Si_mean"].append(si_mean)
            # umbrales preferentes del selector paramétrico; fallback a los del selector simple
            thr_sel = G.graph.get(
                "SELECTOR_THRESHOLDS", DEFAULTS.get("SELECTOR_THRESHOLDS", {})
            )
            thr_def = G.graph.get(
                "GLYPH_THRESHOLDS", DEFAULTS.get("GLYPH_THRESHOLDS", {"hi": 0.66, "lo": 0.33})
            )
            si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
            si_lo = float(thr_sel.get("si_lo", thr_def.get("lo", 0.33)))
            n = len(sis)
            hist["Si_hi_frac"].append(sum(1 for s in sis if s >= si_hi) / n)
            hist["Si_lo_frac"].append(sum(1 for s in sis if s <= si_lo) / n)
        else:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
    except (KeyError, ValueError, TypeError):
        # si aún no se calculó Si este paso, no interrumpimos
        pass
