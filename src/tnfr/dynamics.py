"""Dinámica del sistema."""
from __future__ import annotations

import logging
import math
import random
from collections import deque, OrderedDict
from typing import Dict, Any, Literal

import networkx as nx

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
     invoke_callbacks, reciente_glifo, set_vf, set_dnfr, compute_Si, normalize_weights,
     ensure_history, compute_coherence, compute_dnfr_accel_max,
)
from .selector import (
    _selector_thresholds,
    _norms_para_selector,
    _calc_selector_score,
    _apply_selector_hysteresis,
)

logger = logging.getLogger(__name__)

# ``numpy`` is an optional dependency.  It is loaded lazily to avoid emitting
# warnings when the vectorized path is not used.
np: Any | None = None

def _ensure_numpy(*, warn: bool = False) -> bool:
    """Load ``numpy`` on demand.

    Parameters
    ----------
    warn:
        If ``True`` the failure to import ``numpy`` is logged as a warning,
        otherwise a debug message is emitted. Returns ``True`` if ``numpy``
        was imported successfully.
    """

    global np
    if np is not None:  # pragma: no cover - already loaded
        return True
    try:  # Optional dependency
        import numpy as _np  # type: ignore
    except ImportError:  # pragma: no cover - handled gracefully
        log = logger.warning if warn else logger.debug
        log(
            "Fallo al importar numpy, se continuará con el modo no vectorizado",
            exc_info=True,
        )
        np = None
        return False
    np = _np
    return True

# Cacheo de nodos y matriz de adyacencia asociado a cada grafo
def _cached_nodes_and_A(
    G: nx.Graph, *, cache_size: int | None = 1
) -> tuple[list[int], Any]:
    """Devuelve la lista de nodos y la matriz de adyacencia para ``G``.

    La información se almacena en ``G.graph`` bajo la clave ``"_dnfr_cache"`` y
    se reutiliza mientras la estructura del grafo permanezca igual. ``cache_size``
    limita el número de entradas por grafo (``None`` o valores <= 0 implican sin
    límite). Cuando se supera el tamaño, se elimina explícitamente la entrada más
    antigua."""

    cache: OrderedDict = G.graph.setdefault("_dnfr_cache", OrderedDict())
    # El checksum depende del conjunto de nodos, ignorando el orden.
    nodes_list = list(G.nodes())
    checksum = hash(tuple(sorted(hash(n) for n in nodes_list)))

    last_checksum = G.graph.get("_dnfr_nodes_checksum")
    if last_checksum != checksum:
        cache.clear()
        G.graph["_dnfr_nodes_checksum"] = checksum

    key = (int(G.graph.get("_edge_version", 0)), len(nodes_list), checksum)
    nodes_and_A = cache.get(key)
    if nodes_and_A is None:
        nodes = nodes_list
        if np is not None:
            A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        else:
            A = None
        nodes_and_A = (nodes, A)
        cache[key] = nodes_and_A
        # Purga explícita si excede el tamaño permitido
        if cache_size is not None and cache_size > 0 and len(cache) > cache_size:
            cache.popitem(last=False)
    else:
        # Mantener orden de uso reciente
        cache.move_to_end(key)

    return nodes_and_A


def _update_node_sample(G, *, step: int) -> None:
    """Refresh ``G.graph['_node_sample']`` with a random subset of nodes.

    The sample is limited by ``UM_CANDIDATE_COUNT`` and refreshed every
    simulation step. When the network is small (``< 50`` nodes) or the limit
    is non‑positive, the full node set is used and sampling is effectively
    disabled.
    """

    limit = int(G.graph.get("UM_CANDIDATE_COUNT", 0))
    nodes = list(G.nodes())
    if limit <= 0 or len(nodes) < 50 or limit >= len(nodes):
        G.graph["_node_sample"] = nodes
        return

    seed = int(G.graph.get("RANDOM_SEED", 0))
    rng = random.Random(f"{seed}:{step}")
    G.graph["_node_sample"] = rng.sample(nodes, limit)

# -------------------------
# ΔNFR por defecto (campo) + utilidades de hook/metadata
# -------------------------

def _write_dnfr_metadata(G, *, weights: dict, hook_name: str, note: str | None = None) -> None:
    """Escribe en G.graph un bloque _DNFR_META con la mezcla y el nombre del hook.

    `weights` puede incluir componentes arbitrarias (phase/epi/vf/topo/etc.)."""
    total = math.fsum(float(v) for v in weights.values())
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


def _prepare_dnfr_data(G, *, cache_size: int | None = 1) -> dict:
    """Precalcula datos comunes para las estrategias de ΔNFR."""
    weights = G.graph.get("_dnfr_weights")
    if weights is None:
        weights = _configure_dnfr_weights(G)

    use_numpy = _ensure_numpy() and G.graph.get("vectorized_dnfr")

    # Cacheo de la lista de nodos y la matriz de adyacencia
    nodes, A = _cached_nodes_and_A(G, cache_size=cache_size)
    if not use_numpy:
        A = None

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
        "A": A,
        "cache_size": cache_size,
    }


def _compute_dnfr_numpy(G, data) -> None:
    """Estrategia vectorizada usando ``numpy``."""
    if not _ensure_numpy(warn=True):  # pragma: no cover - check at runtime
        raise RuntimeError("numpy no disponible para la versión vectorizada")
    nodes = data["nodes"]
    if not nodes:
        return
    A = data.get("A")
    if A is None:
        _, A = _cached_nodes_and_A(G, cache_size=data.get("cache_size"))
        data["A"] = A
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


def default_compute_delta_nfr(G, *, cache_size: int | None = 1) -> None:
    """Calcula ΔNFR mezclando gradientes de fase, EPI, νf y un término topológico.

    Parameters
    ----------
    G : nx.Graph
        Grafo sobre el que se realiza el cálculo.
    cache_size : int | None, opcional
        Número máximo de configuraciones de aristas que se cachean en
        ``G.graph``. Valores ``None`` o <= 0 implican caché sin límite. Por
        defecto ``1`` para mantener el comportamiento previo.
    """
    data = _prepare_dnfr_data(G, cache_size=cache_size)
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

def prepare_integration_params(
    G,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
):
    """Valida y normaliza ``dt``, ``t`` y ``method`` para la integración.

    Devuelve ``(dt_step, steps, t0, method)`` donde ``dt_step`` es el paso
    efectivo, ``steps`` la cantidad de subpasos y ``t0`` el tiempo inicial
    preparado.
    """
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

    method = (
        method
        or G.graph.get("INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler"))
    ).lower()
    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")

    dt_min = float(G.graph.get("DT_MIN", DEFAULTS.get("DT_MIN", 0.0)))
    if dt_min > 0 and dt > dt_min:
        steps = int(math.ceil(dt / dt_min))
    else:
        steps = 1
    dt_step = dt / steps if steps else 0.0

    return dt_step, steps, t, method


def _integrate_euler(G, dt_step: float, t_local: float):
    """Un paso de integración explícita de Euler."""
    gamma_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
    new_states: Dict[Any, tuple[float, float, float]] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dEPI_dt_prev = get_attr(nd, ALIAS_dEPI, 0.0)
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)

        base = vf * dnfr
        dEPI_dt = base + gamma_map.get(n, 0.0)
        epi = epi_i + dt_step * dEPI_dt
        d2epi = (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0
        new_states[n] = (epi, dEPI_dt, d2epi)
    return new_states


def _integrate_rk4(G, dt_step: float, t_local: float):
    """Un paso de integración con Runge-Kutta de orden 4."""
    t_mid = t_local + dt_step / 2.0
    t_end = t_local + dt_step
    g1_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
    g_mid_map = {n: eval_gamma(G, n, t_mid) for n in G.nodes}
    g4_map = {n: eval_gamma(G, n, t_end) for n in G.nodes}

    new_states: Dict[Any, tuple[float, float, float]] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dEPI_dt_prev = get_attr(nd, ALIAS_dEPI, 0.0)
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)

        base = vf * dnfr
        g1 = g1_map.get(n, 0.0)
        g_mid = g_mid_map.get(n, 0.0)
        g4 = g4_map.get(n, 0.0)
        k1 = base + g1
        k2 = base + g_mid
        k3 = base + g_mid
        k4 = base + g4
        epi = epi_i + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        dEPI_dt = k4
        d2epi = (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0
        new_states[n] = (epi, dEPI_dt, d2epi)
    return new_states


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

    dt_step, steps, t0, method = prepare_integration_params(G, dt, t, method)

    t_local = t0
    for _ in range(steps):
        if method == "rk4":
            updates = _integrate_rk4(G, dt_step, t_local)
        else:
            updates = _integrate_euler(G, dt_step, t_local)

        for n, (epi, dEPI_dt, d2epi) in updates.items():
            nd = G.nodes[n]
            epi_kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
            set_attr(nd, ALIAS_EPI, epi)
            if epi_kind:
                set_attr_str(nd, ALIAS_EPI_KIND, epi_kind)
            set_attr(nd, ALIAS_dEPI, dEPI_dt)
            set_attr(nd, ALIAS_D2EPI, d2epi)

        t_local += dt_step

    G.graph["_t"] = t_local


# -------------------------
# Wrappers nombrados (compatibilidad)
# -------------------------

def aplicar_dnfr_campo(G, w_theta=None, w_epi=None, w_vf=None) -> None:
    if any(v is not None for v in (w_theta, w_epi, w_vf)):
        mix = G.graph.get("DNFR_WEIGHTS", DEFAULTS["DNFR_WEIGHTS"]).copy()
        if w_theta is not None:
            mix["phase"] = float(w_theta)
        if w_epi is not None:
            mix["epi"] = float(w_epi)
        if w_vf is not None:
            mix["vf"] = float(w_vf)
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


def _leer_parametros_adaptativos(g: Dict[str, Any]) -> tuple[Dict[str, Any], float, float]:
    """Obtiene configuración y valores actuales para adaptación de fase."""
    cfg = g.get("PHASE_ADAPT", DEFAULTS.get("PHASE_ADAPT", {}))
    kG = float(g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"]))
    kL = float(g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"]))
    return cfg, kG, kL


def _calcular_estado(G, cfg: Dict[str, Any]) -> tuple[str, float, float]:
    """Devuelve estado actual (estable/disonante/transicion) y métricas."""
    R = orden_kuramoto(G)
    win = int(G.graph.get("GLYPH_LOAD_WINDOW", METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"]))
    dist = carga_glifica(G, window=win)
    disr = float(dist.get("_disruptivos", 0.0)) if dist else 0.0

    R_hi = float(cfg.get("R_hi", 0.90))
    R_lo = float(cfg.get("R_lo", 0.60))
    disr_hi = float(cfg.get("disr_hi", 0.50))
    disr_lo = float(cfg.get("disr_lo", 0.25))
    if (R >= R_hi) and (disr <= disr_lo):
        state = "estable"
    elif (R <= R_lo) or (disr >= disr_hi):
        state = "disonante"
    else:
        state = "transicion"
    return state, float(R), disr


def _ajustar_k_suave(kG: float, kL: float, state: str, cfg: Dict[str, Any]) -> tuple[float, float]:
    """Actualiza suavemente kG/kL hacia sus objetivos según el estado."""
    kG_min = float(cfg.get("kG_min", 0.01))
    kG_max = float(cfg.get("kG_max", 0.20))
    kL_min = float(cfg.get("kL_min", 0.05))
    kL_max = float(cfg.get("kL_max", 0.25))

    if state == "disonante":
        kG_t = kG_max
        kL_t = 0.5 * (kL_min + kL_max)  # local medio para no perder plasticidad
    elif state == "estable":
        kG_t = kG_min
        kL_t = kL_min
    else:
        kG_t = 0.5 * (kG_min + kG_max)
        kL_t = 0.5 * (kL_min + kL_max)

    up = float(cfg.get("up", 0.10))
    down = float(cfg.get("down", 0.07))

    def _step(curr: float, target: float, mn: float, mx: float) -> float:
        gain = up if target > curr else down
        nxt = curr + gain * (target - curr)
        return max(mn, min(mx, nxt))

    return _step(kG, kG_t, kG_min, kG_max), _step(kL, kL_t, kL_min, kL_max)


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
        cfg, kG, kL = _leer_parametros_adaptativos(g)

        if bool(cfg.get("enabled", False)):
            state, R, disr = _calcular_estado(G, cfg)
            kG, kL = _ajustar_k_suave(kG, kL, state, cfg)

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
# Selector glífico por defecto
# -------------------------
def default_glyph_selector(G, n) -> str:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    hi, lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]

    norms = G.graph.get("_sel_norms")
    if norms is None:
        norms = compute_dnfr_accel_max(G)
        G.graph["_sel_norms"] = norms
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0

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


def _selector_normalized_metrics(nd, norms):
    """Extrae y normaliza Si, ΔNFR y aceleración para el selector."""
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    acc_max = float(norms.get("accel_max", 1.0)) or 1.0
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max
    return Si, dnfr, accel


def _selector_base_choice(Si, dnfr, accel, thr):
    """Decisión base según umbrales de Si, ΔNFR y aceleración."""
    si_hi, si_lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]
    acc_hi = thr["accel_hi"]
    if Si >= si_hi:
        return "IL"
    if Si <= si_lo:
        if accel >= acc_hi:
            return "THOL"
        return "OZ" if dnfr >= dnfr_hi else "ZHIR"
    if dnfr >= dnfr_hi or accel >= acc_hi:
        return "NAV"
    return "RA"


def _compute_selector_score(G, nd, Si, dnfr, accel, cand):
    """Calcula la puntuación y aplica penalizaciones por estancamiento."""
    W = G.graph.get("SELECTOR_WEIGHTS", DEFAULTS["SELECTOR_WEIGHTS"])
    score = _calc_selector_score(Si, dnfr, accel, W)
    hist_prev = nd.get("hist_glifos")
    if hist_prev and hist_prev[-1] == cand:
        delta_si = get_attr(nd, ALIAS_dSI, 0.0)
        h = G.graph.get("history", {})
        sig = h.get("sense_sigma_mag", [])
        delta_sigma = sig[-1] - sig[-2] if len(sig) >= 2 else 0.0
        if delta_si <= 0.0 and delta_sigma <= 0.0:
            score -= 0.05
    return score


def _apply_score_override(cand, score, dnfr, dnfr_lo):
    """Ajusta el candidato final de forma suave según la puntuación."""
    try:
        if score >= 0.66 and cand in ("NAV", "RA", "ZHIR", "OZ"):
            return "IL"
        if score <= 0.33 and cand in ("NAV", "RA", "IL"):
            return "OZ" if dnfr >= dnfr_lo else "ZHIR"
    except NameError:
        pass
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
    margin = float(G.graph.get("GLYPH_SELECTOR_MARGIN", DEFAULTS["GLYPH_SELECTOR_MARGIN"]))

    norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
    Si, dnfr, accel = _selector_normalized_metrics(nd, norms)

    cand = _selector_base_choice(Si, dnfr, accel, thr)

    hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thr, margin)
    if hist_cand is not None:
        return hist_cand

    score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)

    cand = _apply_score_override(cand, score, dnfr, thr["dnfr_lo"])

    return _soft_grammar_prefilter(G, n, cand, dnfr, accel)

# -------------------------
# Step / run
# -------------------------

def _run_before_callbacks(G, *, step_idx: int, dt: float | None, use_Si: bool, apply_glyphs: bool) -> None:
    invoke_callbacks(
        G,
        "before_step",
        {"step": step_idx, "dt": dt, "use_Si": use_Si, "apply_glyphs": apply_glyphs},
    )


def _update_nodes(
    G,
    *,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
    step_idx: int,
    hist,
) -> None:
    _update_node_sample(G, step=step_idx)
    compute_dnfr_cb = G.graph.get("compute_delta_nfr", default_compute_delta_nfr)
    compute_dnfr_cb(G)
    if use_Si:
        compute_Si(G, inplace=True)
    selector = G.graph.get("glyph_selector", default_glyph_selector)
    if selector is parametric_glyph_selector:
        _norms_para_selector(G)
    if apply_glyphs:
        window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
        use_canon = bool(
            G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})).get(
                "enabled", False
            )
        )
        al_max = int(G.graph.get("AL_MAX_LAG", DEFAULTS["AL_MAX_LAG"]))
        en_max = int(G.graph.get("EN_MAX_LAG", DEFAULTS["EN_MAX_LAG"]))
        h_al = hist.setdefault("since_AL", {})
        h_en = hist.setdefault("since_EN", {})
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
    _dt = float(G.graph.get("DT", DEFAULTS["DT"])) if dt is None else float(dt)
    method = G.graph.get(
        "INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler")
    )
    update_epi_via_nodal_equation(G, dt=_dt, method=method)
    for n in G.nodes():
        aplicar_clamps_canonicos(G.nodes[n], G, n)
    coordinar_fase_global_vecinal(G, None, None)
    adaptar_vf_por_coherencia(G)


def _update_metrics(G) -> None:
    _update_history(G)
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append({n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()})


def _maybe_remesh(G) -> None:
    aplicar_remesh_si_estabilizacion_global(G)


def _run_validators(G) -> None:
    from .validators import run_validators

    run_validators(G)


def _run_after_callbacks(G, *, step_idx: int) -> None:
    h = G.graph.get("history", {})
    ctx = {"step": step_idx}
    metric_pairs = [
        ("C", "C_steps"),
        ("stable_frac", "stable_frac"),
        ("phase_sync", "phase_sync"),
        ("glyph_disr", "glyph_load_disr"),
        ("Si_mean", "Si_mean"),
    ]
    for dst, src in metric_pairs:
        values = h.get(src)
        if values:
            ctx[dst] = values[-1]
    invoke_callbacks(G, "after_step", ctx)


def step(
    G,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    hist = ensure_history(G)
    step_idx = len(hist.setdefault("C_steps", []))
    _run_before_callbacks(
        G, step_idx=step_idx, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs
    )
    _update_nodes(
        G,
        dt=dt,
        use_Si=use_Si,
        apply_glyphs=apply_glyphs,
        step_idx=step_idx,
        hist=hist,
    )
    _update_metrics(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)


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
    C = compute_coherence(G)
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
    hist = ensure_history(G)
    for k in (
        "C_steps", "stable_frac", "phase_sync", "glyph_load_estab", "glyph_load_disr",
        "Si_mean", "Si_hi_frac", "Si_lo_frac", "delta_Si", "B"
    ):
        hist.setdefault(k, [])

    _update_coherence(G, hist)

    eps_dnfr = float(G.graph.get("EPS_DNFR_STABLE", REMESH_DEFAULTS["EPS_DNFR_STABLE"]))
    eps_depi = float(G.graph.get("EPS_DEPI_STABLE", REMESH_DEFAULTS["EPS_DEPI_STABLE"]))
    # contadores y acumuladores
    stables = 0  # nodos que cumplen criterios de estabilidad
    total = max(1, G.number_of_nodes())
    dt = float(G.graph.get("DT", DEFAULTS.get("DT", 1.0))) or 1.0
    delta_si_sum = 0.0  # suma de variaciones δSi
    delta_si_count = 0
    B_sum = 0.0  # suma de bifurcaciones B
    B_count = 0

    for n, nd in G.nodes(data=True):
        # --- estabilidad ---
        if (
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) <= eps_dnfr
            and abs(get_attr(nd, ALIAS_dEPI, 0.0)) <= eps_depi
        ):
            stables += 1  # acumulamos nodos estables

        # --- δSi: cambio de sensibilidad ---
        Si_curr = get_attr(nd, ALIAS_SI, 0.0)
        Si_prev = nd.get("_prev_Si", Si_curr)
        dSi = Si_curr - Si_prev
        nd["_prev_Si"] = Si_curr
        set_attr(nd, ALIAS_dSI, dSi)
        delta_si_sum += dSi  # acumulamos δSi total
        delta_si_count += 1

        # --- bifurcación B = ∂²νf/∂t² ---
        vf_curr = get_attr(nd, ALIAS_VF, 0.0)
        vf_prev = nd.get("_prev_vf", vf_curr)
        dvf_dt = (vf_curr - vf_prev) / dt
        dvf_prev = nd.get("_prev_dvf", dvf_dt)
        B = (dvf_dt - dvf_prev) / dt
        nd["_prev_vf"] = vf_curr
        nd["_prev_dvf"] = dvf_dt
        set_attr(nd, ALIAS_dVF, dvf_dt)
        set_attr(nd, ALIAS_D2VF, B)
        B_sum += B  # acumulamos B total
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
