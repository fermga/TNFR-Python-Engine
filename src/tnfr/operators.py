# operators.py — TNFR canónica (ASCII-safe)
from __future__ import annotations
from typing import Dict, Any, Optional, Iterable
import math
import random
import hashlib
import networkx as nx
from networkx.algorithms import community as nx_comm

from .constants import DEFAULTS, ALIAS_EPI
from .helpers import (
    clamp,
    clamp01,
    list_mean,
    invoke_callbacks,
    angle_diff,
    _get_attr,
    _set_attr,
    _get_attr_str,
    _set_attr_str,
    fase_media,
)
from .node import NodoProtocol, NodoNX
from collections import deque

"""
Este módulo implementa:
- Los 13 glifos como operadores locales suaves.
- Un dispatcher `aplicar_glifo` que mapea el nombre del glifo (con apóstrofo tipográfico) a su función.
- RE’MESH de red: `aplicar_remesh_red` y `aplicar_remesh_si_estabilización_global`.

Nota sobre α (alpha) de RE’MESH: se toma por prioridad de
1) G.graph["GLYPH_FACTORS"]["REMESH_alpha"]
2) G.graph["REMESH_ALPHA"]
3) DEFAULTS["REMESH_ALPHA"]
"""


def _ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached deterministic node→index mapping for ``G``.

    The mapping is built once and stored in ``G.graph['_node_offset_map']``.
    It is rebuilt only if the node count changes.
    """

    node_count = G.number_of_nodes()
    mapping = G.graph.get("_node_offset_map")
    if mapping is None or G.graph.get("_node_offset_count") != node_count:
        mapping = {
            node: idx for idx, node in enumerate(sorted(G.nodes(), key=lambda x: str(x)))
        }
        G.graph["_node_offset_map"] = mapping
        G.graph["_node_offset_count"] = node_count
    return mapping


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = _ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def random_jitter(node: NodoProtocol, amplitude: float) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for ``node``."""
    base_seed = int(node.graph.get("RANDOM_SEED", 0))
    step_idx = len(node.graph.get("history", {}).get("C_steps", []))
    G = getattr(node, "G", None)
    mapping = _ensure_node_offset_map(G) if G is not None else {}
    node_id = getattr(node, "n", None)
    offset = mapping.get(node_id, 0)
    rnd = random.Random(base_seed + step_idx * 1000003 + offset % 1009)
    return amplitude * (2.0 * rnd.random() - 1.0)

# -------------------------
# Glifos (operadores locales)
# -------------------------


def _select_dominant_glifo(node: NodoProtocol, neigh: Iterable[NodoProtocol]) -> Optional[str]:
    """Return the epi_kind with the highest |EPI| among node and its neighbors."""
    candidatos = [(abs(node.EPI), node.epi_kind)]
    for v in neigh:
        candidatos.append((abs(v.EPI), v.epi_kind))
    return max(candidatos, key=lambda x: x[0])[1]


def _op_AL(node: NodoProtocol) -> None:  # A’L — Emisión
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    f = float(gf.get("AL_boost", 0.05))
    node.EPI = node.EPI + f
    node.epi_kind = "A’L"


def _op_EN(node: NodoProtocol) -> None:  # E’N — Recepción
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    mix = float(gf.get("EN_mix", 0.25))
    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        # Aunque no haya vecinos, etiquetar el nodo con el glifo E’N
        node.epi_kind = "E’N"
        return
    epi_bar = list_mean(v.EPI for v in neigh)
    node.EPI = (1 - mix) * epi + mix * epi_bar
    node.epi_kind = _select_dominant_glifo(node, neigh) or "E’N"


def _op_IL(node: NodoProtocol) -> None:  # I’L — Coherencia
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    factor = float(gf.get("IL_dnfr_factor", 0.7))
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodoProtocol) -> None:  # O’Z — Disonancia
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    factor = float(gf.get("OZ_dnfr_factor", 1.3))
    dnfr = getattr(node, "dnfr", 0.0)
    if bool(node.graph.get("OZ_NOISE_MODE", False)):
        sigma = float(node.graph.get("OZ_SIGMA", 0.1))
        node.dnfr = dnfr + random_jitter(node, sigma)
    else:
        node.dnfr = factor * dnfr if abs(dnfr) > 1e-9 else 0.1


def _op_UM(node: NodoProtocol) -> None:  # U’M — Acoplamiento
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    k = float(gf.get("UM_theta_push", 0.25))
    th = node.theta
    thL = fase_media(node)
    d = angle_diff(thL, th)
    node.theta = th + k * d

    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", False)):
        thr = float(node.graph.get("UM_COMPAT_THRESHOLD", DEFAULTS.get("UM_COMPAT_THRESHOLD", 0.75)))
        epi_i = node.EPI
        si_i = node.Si
        for j in node.all_nodes():
            if j is node or node.has_edge(j):
                continue
            th_j = j.theta
            dphi = abs(angle_diff(th_j, th)) / math.pi
            epi_j = j.EPI
            si_j = j.Si
            epi_sim = 1.0 - abs(epi_i - epi_j) / (abs(epi_i) + abs(epi_j) + 1e-9)
            si_sim = 1.0 - abs(si_i - si_j)
            compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodoProtocol) -> None:  # R’A — Resonancia
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    diff = float(gf.get("RA_epi_diff", 0.15))
    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        return
    epi_bar = list_mean(v.EPI for v in neigh)
    node.EPI = epi + diff * (epi_bar - epi)
    node.epi_kind = _select_dominant_glifo(node, neigh) or "R’A"


def _op_SHA(node: NodoProtocol) -> None:  # SH’A — Silencio
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    factor = float(gf.get("SHA_vf_factor", 0.85))
    node.vf = factor * node.vf


def _op_VAL(node: NodoProtocol) -> None:  # VA’L — Expansión
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    s = float(gf.get("VAL_scale", 1.15))
    node.EPI = s * node.EPI
    node.epi_kind = "VA’L"


def _op_NUL(node: NodoProtocol) -> None:  # NU’L — Contracción
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    s = float(gf.get("NUL_scale", 0.85))
    node.EPI = s * node.EPI
    node.epi_kind = "NU’L"


def _op_THOL(node: NodoProtocol) -> None:  # T’HOL — Autoorganización
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    a = float(gf.get("THOL_accel", 0.10))
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(node: NodoProtocol) -> None:  # Z’HIR — Mutación
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    shift = float(gf.get("ZHIR_theta_shift", math.pi / 2))
    node.theta = node.theta + shift


def _op_NAV(node: NodoProtocol) -> None:  # NA’V — Transición
    gf = node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])
    dnfr = node.dnfr
    vf = node.vf
    eta = float(gf.get("NAV_eta", 0.5))
    strict = bool(node.graph.get("NAV_STRICT", False))
    if strict:
        base = vf
    else:
        sign = 1.0 if dnfr >= 0 else -1.0
        target = sign * vf
        base = (1.0 - eta) * dnfr + eta * target
    j = float(gf.get("NAV_jitter", 0.05))
    if bool(node.graph.get("NAV_RANDOM", True)):
        jitter = random_jitter(node, j)
    else:
        jitter = j * (1 if base >= 0 else -1)
    node.dnfr = base + jitter


def _op_REMESH(node: NodoProtocol) -> None:  # RE’MESH — aviso
    step_idx = len(node.graph.get("history", {}).get("C_steps", []))
    last_warn = node.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = "RE’MESH es a escala de red. Usa aplicar_remesh_si_estabilizacion_global(G) o aplicar_remesh_red(G)."
        node.graph.setdefault("history", {}).setdefault("events", []).append(("warn", {"step": step_idx, "node": None, "msg": msg}))
        node.graph["_remesh_warn_step"] = step_idx
    return

# -------------------------
# Dispatcher
# -------------------------

_NAME_TO_OP = {
    "A’L": _op_AL, "E’N": _op_EN, "I’L": _op_IL, "O’Z": _op_OZ, "U’M": _op_UM,
    "R’A": _op_RA, "SH’A": _op_SHA, "VA’L": _op_VAL, "NU’L": _op_NUL,
    "T’HOL": _op_THOL, "Z’HIR": _op_ZHIR, "NA’V": _op_NAV, "RE’MESH": _op_REMESH,
}


def _wrap(fn):
    def inner(obj, n=None):
        node = obj if n is None else NodoNX(obj, n)
        return fn(node)
    return inner

op_AL = _wrap(_op_AL)
op_EN = _wrap(_op_EN)
op_IL = _wrap(_op_IL)
op_OZ = _wrap(_op_OZ)
op_UM = _wrap(_op_UM)
op_RA = _wrap(_op_RA)
op_SHA = _wrap(_op_SHA)
op_VAL = _wrap(_op_VAL)
op_NUL = _wrap(_op_NUL)
op_THOL = _wrap(_op_THOL)
op_ZHIR = _wrap(_op_ZHIR)
op_NAV = _wrap(_op_NAV)
op_REMESH = _wrap(_op_REMESH)


def aplicar_glifo_obj(node: NodoProtocol, glifo: str, *, window: Optional[int] = None) -> None:
    """Aplica ``glifo`` a un objeto que cumple :class:`NodoProtocol`."""

    glifo = str(glifo)
    op = _NAME_TO_OP.get(glifo)
    if not op:
        return
    if window is None:
        window = int(node.graph.get("GLYPH_HYSTERESIS_WINDOW", DEFAULTS["GLYPH_HYSTERESIS_WINDOW"]))
    node.push_glifo(glifo, window)
    op(node)


def aplicar_glifo(G, n, glifo: str, *, window: Optional[int] = None) -> None:
    """Adaptador para operar sobre grafos ``networkx``."""
    node = NodoNX(G, n)
    aplicar_glifo_obj(node, glifo, window=window)


# -------------------------
# RE’MESH de red (usa _epi_hist capturado en dynamics.step)
# -------------------------

def _remesh_alpha_info(G):
    """Devuelve `(alpha, source)` con precedencia explícita."""
    if bool(G.graph.get("REMESH_ALPHA_HARD", DEFAULTS.get("REMESH_ALPHA_HARD", False))):
        val = float(G.graph.get("REMESH_ALPHA", DEFAULTS["REMESH_ALPHA"]))
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return float(DEFAULTS["REMESH_ALPHA"]), "DEFAULTS.REMESH_ALPHA"


def aplicar_remesh_red(G) -> None:
    """RE’MESH a escala de red usando _epi_hist con memoria multi-escala."""
    # REMESH_TAU: alias legado de REMESH_TAU_GLOBAL
    tau_g = int(
        G.graph.get(
            "REMESH_TAU_GLOBAL",
            G.graph.get("REMESH_TAU", DEFAULTS["REMESH_TAU_GLOBAL"]),
        )
    )
    tau_l = int(
        G.graph.get(
            "REMESH_TAU_LOCAL",
            G.graph.get("REMESH_TAU", DEFAULTS["REMESH_TAU_LOCAL"]),
        )
    )
    tau_req = max(tau_g, tau_l)
    alpha, alpha_src = _remesh_alpha_info(G)
    G.graph["_REMESH_ALPHA_SRC"] = alpha_src
    hist = G.graph.get("_epi_hist", deque())
    if len(hist) < tau_req + 1:
        return

    past_g = hist[-(tau_g + 1)]
    past_l = hist[-(tau_l + 1)]

    # --- Topología + snapshot EPI (ANTES) ---
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        topo_hash = hashlib.sha1(topo_str.encode()).hexdigest()[:12]
    except Exception:
        topo_hash = None

    def _epi_items():
        for node in G.nodes():
            yield node, _get_attr(G.nodes[node], ALIAS_EPI, 0.0)

    epi_mean_before = list_mean(v for _, v in _epi_items())
    epi_checksum_before = hashlib.sha1(
        str(sorted((str(k), round(v, 6)) for k, v in _epi_items())).encode()
    ).hexdigest()[:12]

    # --- Mezcla (1-α)·now + α·old ---
    for n in G.nodes():
        nd = G.nodes[n]
        epi_now = _get_attr(nd, ALIAS_EPI, 0.0)
        epi_old_l = float(past_l.get(n, epi_now))
        epi_old_g = float(past_g.get(n, epi_now))
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        _set_attr(nd, ALIAS_EPI, mixed)

    # --- Snapshot EPI (DESPUÉS) ---
    epi_mean_after = list_mean(_get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes())
    epi_checksum_after = hashlib.sha1(
        str(sorted((str(n), round(_get_attr(G.nodes[n], ALIAS_EPI, 0.0), 6)) for n in G.nodes())).encode()
    ).hexdigest()[:12]

    # --- Metadatos y logging de evento ---
    step_idx = len(G.graph.get("history", {}).get("C_steps", []))
    meta = {
        "alpha": alpha,
        "alpha_source": alpha_src,
        "tau_global": tau_g,
        "tau_local": tau_l,
        "step": step_idx,
        # firmas
        "topo_hash": topo_hash,
        "epi_mean_before": float(epi_mean_before),
        "epi_mean_after": float(epi_mean_after),
        "epi_checksum_before": epi_checksum_before,
        "epi_checksum_after": epi_checksum_after,
    }

    # Snapshot opcional de métricas recientes
    h = G.graph.get("history", {})
    if h:
        if h.get("stable_frac"): meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):  meta["phase_sync_last"]  = h["phase_sync"][-1]
        if h.get("glyph_load_disr"): meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", DEFAULTS["REMESH_LOG_EVENTS"]):
        ev = G.graph.setdefault("history", {}).setdefault("remesh_events", [])
        ev.append(dict(meta))

    # Callbacks Γ(R)
    invoke_callbacks(G, "on_remesh", dict(meta))


def aplicar_remesh_red_topologico(
    G,
    mode: Optional[str] = None,
    *,
    k: Optional[int] = None,
    p_rewire: float = 0.2,
    seed: Optional[int] = None,
) -> None:
    """Remallado topológico aproximado.

    - mode="knn": conecta cada nodo con sus ``k`` vecinos más similares en EPI
      con probabilidad ``p_rewire``.
    - mode="mst": sólo preserva un árbol de expansión mínima según distancia EPI.
    - mode="community": agrupa por comunidades modulares y las conecta por
      similitud intercomunidad.

    Siempre preserva conectividad añadiendo un MST base.
    """
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    rnd = random.Random(seed)

    if mode is None:
        mode = str(G.graph.get("REMESH_MODE", DEFAULTS.get("REMESH_MODE", "knn")))
    mode = str(mode)

    # Similaridad basada en EPI (distancia absoluta)
    epi = {n: _get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes}
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            w = abs(epi[u] - epi[v])
            H.add_edge(u, v, weight=w)
    mst = nx.minimum_spanning_tree(H, weight="weight")

    if mode == "community":
        # Detectar comunidades y reconstruir la red con metanodos
        comms = list(nx_comm.greedy_modularity_communities(G))
        if len(comms) <= 1:
            new_edges = set(mst.edges())
        else:
            k_val = (
                int(k)
                if k is not None
                else int(G.graph.get("REMESH_COMMUNITY_K", DEFAULTS.get("REMESH_COMMUNITY_K", 2)))
            )
            # Grafo de comunidades basado en medias de EPI
            C = nx.Graph()
            for idx, comm in enumerate(comms):
                members = list(comm)
                epi_mean = list_mean(epi[n] for n in members)
                C.add_node(idx)
                _set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
                C.nodes[idx]["members"] = members
            for i in C.nodes():
                for j in C.nodes():
                    if i < j:
                        w = abs(
                            _get_attr(C.nodes[i], ALIAS_EPI, 0.0)
                            - _get_attr(C.nodes[j], ALIAS_EPI, 0.0)
                        )
                        C.add_edge(i, j, weight=w)
            mst_c = nx.minimum_spanning_tree(C, weight="weight")
            new_edges = set(mst_c.edges())
            for u in C.nodes():
                epi_u = _get_attr(C.nodes[u], ALIAS_EPI, 0.0)
                others = [v for v in C.nodes() if v != u]
                others.sort(key=lambda v: abs(epi_u - _get_attr(C.nodes[v], ALIAS_EPI, 0.0)))
                for v in others[:k_val]:
                    if rnd.random() < p_rewire:
                        new_edges.add(tuple(sorted((u, v))))

            # Reemplazar nodos y aristas del grafo original por comunidades
            G.remove_edges_from(list(G.edges()))
            G.remove_nodes_from(list(G.nodes()))
            for idx in C.nodes():
                data = dict(C.nodes[idx])
                G.add_node(idx, **data)
            G.add_edges_from(new_edges)

            if G.graph.get("REMESH_LOG_EVENTS", DEFAULTS["REMESH_LOG_EVENTS"]):
                ev = G.graph.setdefault("history", {}).setdefault("remesh_events", [])
                mapping = {idx: C.nodes[idx].get("members", []) for idx in C.nodes()}
                ev.append({
                    "mode": "community",
                    "n_before": n_before,
                    "n_after": G.number_of_nodes(),
                    "mapping": mapping,
                })
            return

    # Default/mode knn/mst operate on nodos originales
    new_edges = set(mst.edges())
    if mode == "knn":
        k_val = int(k) if k is not None else int(G.graph.get("REMESH_COMMUNITY_K", DEFAULTS.get("REMESH_COMMUNITY_K", 2)))
        k_val = max(1, k_val)
        for u in nodes:
            sims = sorted(nodes, key=lambda v: abs(epi[u] - epi[v]))
            for v in sims[1 : k_val + 1]:
                if rnd.random() < p_rewire:
                    new_edges.add(tuple(sorted((u, v))))

    G.remove_edges_from(list(G.edges()))
    G.add_edges_from(new_edges)

def aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos: Optional[int] = None) -> None:
    # Ventanas y umbrales
    w_estab = (
        pasos_estables_consecutivos
        if pasos_estables_consecutivos is not None
        else int(G.graph.get("REMESH_STABILITY_WINDOW", DEFAULTS["REMESH_STABILITY_WINDOW"]))
    )
    frac_req = float(G.graph.get("FRACTION_STABLE_REMESH", DEFAULTS["FRACTION_STABLE_REMESH"]))
    req_extra = bool(G.graph.get("REMESH_REQUIRE_STABILITY", DEFAULTS["REMESH_REQUIRE_STABILITY"]))
    min_sync = float(G.graph.get("REMESH_MIN_PHASE_SYNC", DEFAULTS["REMESH_MIN_PHASE_SYNC"]))
    max_disr = float(G.graph.get("REMESH_MAX_GLYPH_DISR", DEFAULTS["REMESH_MAX_GLYPH_DISR"]))
    min_sigma = float(G.graph.get("REMESH_MIN_SIGMA_MAG", DEFAULTS["REMESH_MIN_SIGMA_MAG"]))
    min_R = float(G.graph.get("REMESH_MIN_KURAMOTO_R", DEFAULTS["REMESH_MIN_KURAMOTO_R"]))
    min_sihi = float(G.graph.get("REMESH_MIN_SI_HI_FRAC", DEFAULTS["REMESH_MIN_SI_HI_FRAC"]))

    hist = G.graph.setdefault("history", {"stable_frac": []})
    sf = hist.get("stable_frac", [])
    if len(sf) < w_estab:
        return
    # 1) Estabilidad por fracción de nodos estables
    win_sf = sf[-w_estab:]
    cond_sf = all(v >= frac_req for v in win_sf)
    if not cond_sf:
        return
    # 2) Gating adicional (si está activado)
    if req_extra:
        # sincronía de fase (mayor mejor)
        ps_ok = True
        if "phase_sync" in hist and len(hist["phase_sync"]) >= w_estab:
            win_ps = hist["phase_sync"][-w_estab:]
            ps_ok = (sum(win_ps)/len(win_ps)) >= min_sync
        # carga glífica disruptiva (menor mejor)
        disr_ok = True
        if "glyph_load_disr" in hist and len(hist["glyph_load_disr"]) >= w_estab:
            win_disr = hist["glyph_load_disr"][-w_estab:]
            disr_ok = (sum(win_disr)/len(win_disr)) <= max_disr
        # magnitud de sigma (mayor mejor)
        sig_ok = True
        if "sense_sigma_mag" in hist and len(hist["sense_sigma_mag"]) >= w_estab:
            win_sig = hist["sense_sigma_mag"][-w_estab:]
            sig_ok = (sum(win_sig)/len(win_sig)) >= min_sigma
        # orden de Kuramoto R (mayor mejor)
        R_ok = True
        if "kuramoto_R" in hist and len(hist["kuramoto_R"]) >= w_estab:
            win_R = hist["kuramoto_R"][-w_estab:]
            R_ok = (sum(win_R)/len(win_R)) >= min_R
        # fracción de nodos con Si alto (mayor mejor)
        sihi_ok = True
        if "Si_hi_frac" in hist and len(hist["Si_hi_frac"]) >= w_estab:
            win_sihi = hist["Si_hi_frac"][-w_estab:]
            sihi_ok = (sum(win_sihi)/len(win_sihi)) >= min_sihi
        if not (ps_ok and disr_ok and sig_ok and R_ok and sihi_ok):
            return
    # 3) Cooldown
    last = G.graph.get("_last_remesh_step", -10**9)
    step_idx = len(sf)
    cooldown = int(G.graph.get("REMESH_COOLDOWN_VENTANA", DEFAULTS["REMESH_COOLDOWN_VENTANA"]))
    if step_idx - last < cooldown:
        return
    t_now = float(G.graph.get("_t", 0.0))
    last_ts = float(G.graph.get("_last_remesh_ts", -1e12))
    cooldown_ts = float(G.graph.get("REMESH_COOLDOWN_TS", DEFAULTS.get("REMESH_COOLDOWN_TS", 0.0)))
    if cooldown_ts > 0 and (t_now - last_ts) < cooldown_ts:
        return
    # 4) Aplicar y registrar
    aplicar_remesh_red(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now
