"""Operadores de la red."""
# operators.py — TNFR canónica (ASCII-safe)
from __future__ import annotations
from typing import Dict, Any, Optional, Iterable, TYPE_CHECKING
import math
import random
import hashlib
import heapq
import weakref
import networkx as nx
from networkx.algorithms import community as nx_comm

from .constants import DEFAULTS, REMESH_DEFAULTS, ALIAS_EPI, get_param
from .helpers import (
    list_mean,
    invoke_callbacks,
    angle_diff,
    get_attr,
    set_attr,
    fase_media,
    increment_edge_version,
)
if TYPE_CHECKING:
    from .node import NodoProtocol
from .types import Glyph
from collections import deque

"""
Este módulo implementa:
- Los 13 glifos como operadores locales suaves.
- Un dispatcher `aplicar_glifo` que mapea el nombre del glifo (con apóstrofo tipográfico) a su función.
- REMESH de red: `aplicar_remesh_red` y `aplicar_remesh_si_estabilización_global`.

Nota sobre α (alpha) de REMESH: se toma por prioridad de
1) G.graph["GLYPH_FACTORS"]["REMESH_alpha"]
2) G.graph["REMESH_ALPHA"]
3) REMESH_DEFAULTS["REMESH_ALPHA"]
"""


def _ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``.

    The mapping follows the natural insertion order of ``G.nodes`` for speed.
    When ``G.graph['SORT_NODES']`` is true a deterministic sort is applied.
    A checksum of the node set is stored so the mapping is recomputed only
    when the nodes change.
    """

    nodes = list(G.nodes())
    checksum = hash(tuple(nodes))
    mapping = G.graph.get("_node_offset_map")
    if mapping is None or G.graph.get("_node_offset_checksum") != checksum:
        if bool(G.graph.get("SORT_NODES", False)):
            nodes.sort(key=lambda x: str(x))
        mapping = {node: idx for idx, node in enumerate(nodes)}
        G.graph["_node_offset_map"] = mapping
        G.graph["_node_offset_checksum"] = checksum
    return mapping


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = _ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def _jitter_base(seed: int, key: int) -> random.Random:
    """Return a ``random.Random`` instance seeded from ``seed`` and ``key``."""
    seed_input = (seed, key)
    try:
        # Python's ``random`` module does not officially support tuples as seeds,
        # but future versions may. Attempt to use the tuple directly first and
        # fall back to a string representation when unsupported.
        return random.Random(seed_input)
    except TypeError:
        return random.Random(str(seed_input))


_rng_cache: "weakref.WeakKeyDictionary[object, Dict[tuple[int, int], random.Random]]" = (
    weakref.WeakKeyDictionary()
)


def _get_rng(scope: object, seed: int, key: int) -> random.Random:
    """Return cached ``random.Random`` for ``(seed, key)`` scoped to ``scope``."""
    cache = _rng_cache.setdefault(scope, {})
    pair = (int(seed), int(key))
    rng = cache.get(pair)
    if rng is None:
        rng = _jitter_base(seed, key)
        cache[pair] = rng
    return rng


def clear_rng_cache() -> None:
    """Clear all cached RNGs."""
    _rng_cache.clear()


# Backwards compatibility
clear_jitter_cache = clear_rng_cache


_NodoNX = None


def _get_NodoNX():
    """Lazy importer for ``NodoNX`` to avoid circular dependencies."""
    global _NodoNX
    if _NodoNX is None:
        from .node import NodoNX as _NodoNX_cls
        _NodoNX = _NodoNX_cls
    return _NodoNX


def random_jitter(
    node: NodoProtocol, amplitude: float, cache: Optional[Dict[int, random.Random]] = None
) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for ``node``.

    The value is derived from ``(RANDOM_SEED, node.offset())`` and does not store
    references to nodes. By default a global cache of ``(seed, key) → random.Random``
    instances, scoped by graph via weak references, advances deterministic
    sequences across calls. When ``cache`` is provided, it is used instead and
    must handle its own purging policy.
    """

    if amplitude <= 0:
        if amplitude < 0:
            raise ValueError("amplitude must be positive")
        return 0.0

    base_seed = int(node.graph.get("RANDOM_SEED", 0))

    if hasattr(node, "n") and hasattr(node, "G"):
        seed_key = _node_offset(node.G, node.n)
        scope = node.G
    else:
        uid = getattr(node, "_noise_uid", None)
        if uid is None:
            uid = id(node)
            setattr(node, "_noise_uid", uid)
        seed_key = int(uid)
        scope = node

    if cache is None:
        rng = _get_rng(scope, base_seed, seed_key)
    else:
        rng = cache.get(seed_key)
        if rng is None:
            rng = _jitter_base(base_seed, seed_key)
            cache[seed_key] = rng

    base = rng.uniform(-1.0, 1.0)
    return amplitude * base


def get_glyph_factors(node: NodoProtocol) -> Dict[str, Any]:
    """Return glyph factors for ``node`` with defaults."""
    return node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"])

# -------------------------
# Glifos (operadores locales)
# -------------------------


def _select_dominant_glifo(node: NodoProtocol, neigh: Iterable[NodoProtocol]) -> Optional[str]:
    """Return the epi_kind with the highest |EPI| among node and its neighbors."""
    best_mag = abs(node.EPI)
    best_kind = node.epi_kind
    for v in neigh:
        epi_abs = abs(v.EPI)
        if epi_abs > best_mag:
            best_mag = epi_abs
            best_kind = v.epi_kind
    return best_kind


def _mix_epi_with_neighbors(
    node: NodoProtocol, mix: float, default_glyph: Glyph | str
) -> None:
    """Mezcla ``EPI`` del ``node`` con la media de sus vecinos.

    ``mix`` controla la fracción de influencia vecinal y ``default_glyph`` es el
    glifo que se asignará si no hay vecinos o si no se detecta uno dominante.
    """

    default_kind = (
        default_glyph.value if isinstance(default_glyph, Glyph) else str(default_glyph)
    )
    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        node.epi_kind = default_kind
        return
    epi_bar = list_mean(v.EPI for v in neigh)
    node.EPI = (1 - mix) * epi + mix * epi_bar
    node.epi_kind = _select_dominant_glifo(node, neigh) or default_kind


def _op_AL(node: NodoProtocol) -> None:  # AL — Emisión
    gf = get_glyph_factors(node)
    f = float(gf.get("AL_boost", 0.05))
    node.EPI = node.EPI + f


def _op_EN(node: NodoProtocol) -> None:  # EN — Recepción
    gf = get_glyph_factors(node)
    mix = float(gf.get("EN_mix", 0.25))
    _mix_epi_with_neighbors(node, mix, Glyph.EN)


def _op_IL(node: NodoProtocol) -> None:  # IL — Coherencia
    gf = get_glyph_factors(node)
    factor = float(gf.get("IL_dnfr_factor", 0.7))
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodoProtocol) -> None:  # OZ — Disonancia
    gf = get_glyph_factors(node)
    factor = float(gf.get("OZ_dnfr_factor", 1.3))
    dnfr = getattr(node, "dnfr", 0.0)
    if bool(node.graph.get("OZ_NOISE_MODE", False)):
        sigma = float(node.graph.get("OZ_SIGMA", 0.1))
        if sigma <= 0:
            node.dnfr = dnfr
            return
        node.dnfr = dnfr + random_jitter(node, sigma)
    else:
        node.dnfr = factor * dnfr if abs(dnfr) > 1e-9 else 0.1


def _op_UM(node: NodoProtocol) -> None:  # UM — Acoplamiento
    """Alinea fase y opcionalmente crea enlaces funcionales.

    La búsqueda de enlaces puede reducirse evaluando sólo un subconjunto de
    candidatos. ``UM_CANDIDATE_COUNT`` establece cuántos nodos considerar y
    ``UM_CANDIDATE_MODE`` define la estrategia:

    * ``"proximity"``: selecciona los nodos más cercanos en fase.
    * ``"sample"``: toma una muestra determinista.

    Desde ``dynamics.step`` se mantiene en ``G.graph['_node_sample']`` una
    muestra aleatoria renovada en cada paso. Cuando el grafo es pequeño
    (``<50`` nodos) la muestra contiene todos los nodos y el muestreo se
    desactiva.

    Esto preserva la lógica de acoplamiento sin revisar todos los nodos.
    """

    gf = get_glyph_factors(node)
    k = float(gf.get("UM_theta_push", 0.25))
    th = node.theta
    thL = fase_media(node)
    d = angle_diff(thL, th)
    node.theta = th + k * d

    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", False)):
        thr = float(node.graph.get("UM_COMPAT_THRESHOLD", DEFAULTS.get("UM_COMPAT_THRESHOLD", 0.75)))
        epi_i = node.EPI
        si_i = node.Si

        sample_ids = node.graph.get("_node_sample")
        if sample_ids is not None and hasattr(node, "G"):
            NodoNX = _get_NodoNX()
            iter_nodes = (NodoNX(node.G, j) for j in sample_ids)
        else:
            iter_nodes = node.all_nodes()

        candidates = []
        for j in iter_nodes:
            same = (j is node) or (getattr(node, "n", None) == getattr(j, "n", None))
            if same or node.has_edge(j):
                continue
            candidates.append(j)
        limit = int(node.graph.get("UM_CANDIDATE_COUNT", 0))
        mode = str(node.graph.get("UM_CANDIDATE_MODE", "sample")).lower()
        if limit > 0 and len(candidates) > limit:
            if mode == "proximity":
                candidates.sort(key=lambda j: abs(angle_diff(j.theta, th)))
                candidates = candidates[:limit]
            else:
                rng = _jitter_base(int(node.graph.get("RANDOM_SEED", 0)), node.offset())
                candidates = rng.sample(candidates, limit)

        for j in candidates:
            th_j = j.theta
            dphi = abs(angle_diff(th_j, th)) / math.pi
            epi_j = j.EPI
            si_j = j.Si
            epi_sim = 1.0 - abs(epi_i - epi_j) / (abs(epi_i) + abs(epi_j) + 1e-9)
            si_sim = 1.0 - abs(si_i - si_j)
            compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodoProtocol) -> None:  # RA — Resonancia
    gf = get_glyph_factors(node)
    diff = float(gf.get("RA_epi_diff", 0.15))
    _mix_epi_with_neighbors(node, diff, Glyph.RA)


def _op_SHA(node: NodoProtocol) -> None:  # SHA — Silencio
    gf = get_glyph_factors(node)
    factor = float(gf.get("SHA_vf_factor", 0.85))
    node.vf = factor * node.vf


def _scale_epi(node: NodoProtocol, factor: float, glyph: Glyph) -> None:
    """Escala ``EPI`` del nodo y actualiza ``epi_kind``."""
    node.EPI = factor * node.EPI
    node.epi_kind = glyph.value if isinstance(glyph, Glyph) else str(glyph)


def _op_VAL(node: NodoProtocol) -> None:  # VAL — Expansión
    gf = get_glyph_factors(node)
    s = float(gf.get("VAL_scale", 1.15))
    _scale_epi(node, s, Glyph.VAL)


def _op_NUL(node: NodoProtocol) -> None:  # NUL — Contracción
    gf = get_glyph_factors(node)
    s = float(gf.get("NUL_scale", 0.85))
    _scale_epi(node, s, Glyph.NUL)


def _op_THOL(node: NodoProtocol) -> None:  # THOL — Autoorganización
    gf = get_glyph_factors(node)
    a = float(gf.get("THOL_accel", 0.10))
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(node: NodoProtocol) -> None:  # ZHIR — Mutación
    gf = get_glyph_factors(node)
    shift = float(gf.get("ZHIR_theta_shift", math.pi / 2))
    node.theta = node.theta + shift


def _op_NAV(node: NodoProtocol) -> None:  # NAV — Transición
    gf = get_glyph_factors(node)
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


def _op_REMESH(node: NodoProtocol) -> None:  # REMESH — aviso
    step_idx = len(node.graph.get("history", {}).get("C_steps", []))
    last_warn = node.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = "REMESH es a escala de red. Usa aplicar_remesh_si_estabilizacion_global(G) o aplicar_remesh_red(G)."
        node.graph.setdefault("history", {}).setdefault("events", []).append(("warn", {"step": step_idx, "node": None, "msg": msg}))
        node.graph["_remesh_warn_step"] = step_idx
    return

# -------------------------
# Dispatcher
# -------------------------

_NAME_TO_OP = {
    Glyph.AL: _op_AL,
    Glyph.EN: _op_EN,
    Glyph.IL: _op_IL,
    Glyph.OZ: _op_OZ,
    Glyph.UM: _op_UM,
    Glyph.RA: _op_RA,
    Glyph.SHA: _op_SHA,
    Glyph.VAL: _op_VAL,
    Glyph.NUL: _op_NUL,
    Glyph.THOL: _op_THOL,
    Glyph.ZHIR: _op_ZHIR,
    Glyph.NAV: _op_NAV,
    Glyph.REMESH: _op_REMESH,
}


def _wrap(fn):
    def inner(obj, n=None):
        NodoNX = _get_NodoNX()
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


def aplicar_glifo_obj(node: NodoProtocol, glifo: Glyph | str, *, window: Optional[int] = None) -> None:
    """Aplica ``glifo`` a un objeto que cumple :class:`NodoProtocol`."""

    try:
        g = glifo if isinstance(glifo, Glyph) else Glyph(str(glifo))
    except ValueError:
        step_idx = len(node.graph.get("history", {}).get("C_steps", []))
        node.graph.setdefault("history", {}).setdefault("events", []).append(
            (
                "warn",
                {
                    "step": step_idx,
                    "node": getattr(node, "n", None),
                    "msg": f"glifo desconocido: {glifo}",
                },
            )
        )
        raise ValueError(f"glifo desconocido: {glifo}")

    op = _NAME_TO_OP.get(g)
    if op is None:
        raise ValueError(f"glifo sin operador: {g}")
    if window is None:
        window = int(get_param(node, "GLYPH_HYSTERESIS_WINDOW"))
    op(node)
    node.push_glifo(g.value, window)


def aplicar_glifo(G, n, glifo: Glyph | str, *, window: Optional[int] = None) -> None:
    """Adaptador para operar sobre grafos ``networkx``."""
    NodoNX = _get_NodoNX()
    node = NodoNX(G, n)
    aplicar_glifo_obj(node, glifo, window=window)


# -------------------------
# REMESH de red (usa _epi_hist capturado en dynamics.step)
# -------------------------

def _remesh_alpha_info(G):
    """Devuelve ``(alpha, source)`` con precedencia explícita."""
    if bool(
        G.graph.get("REMESH_ALPHA_HARD", REMESH_DEFAULTS["REMESH_ALPHA_HARD"])
    ):
        val = float(G.graph.get("REMESH_ALPHA", REMESH_DEFAULTS["REMESH_ALPHA"]))
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return float(REMESH_DEFAULTS["REMESH_ALPHA"]), "REMESH_DEFAULTS.REMESH_ALPHA"


def aplicar_remesh_red(G) -> None:
    """REMESH a escala de red usando _epi_hist con memoria multi-escala."""
    # REMESH_TAU: alias legado resuelto por ``get_param``
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
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
    except (AttributeError, TypeError, nx.NetworkXError):
        topo_hash = None

    def _epi_items():
        for node in G.nodes():
            yield node, get_attr(G.nodes[node], ALIAS_EPI, 0.0)

    epi_items = list(_epi_items())
    epi_mean_before = list_mean(v for _, v in epi_items)
    epi_checksum_before = hashlib.sha1(
        str(sorted((str(k), round(v, 6)) for k, v in epi_items)).encode()
    ).hexdigest()[:12]

    # --- Mezcla (1-α)·now + α·old ---
    for n, nd in G.nodes(data=True):
        epi_now = get_attr(nd, ALIAS_EPI, 0.0)
        epi_old_l = float(past_l.get(n, epi_now))
        epi_old_g = float(past_g.get(n, epi_now))
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        set_attr(nd, ALIAS_EPI, mixed)

    # --- Snapshot EPI (DESPUÉS) ---
    epi_items_after = [
        (n, get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in G.nodes()
    ]
    epi_mean_after = list_mean(v for _, v in epi_items_after)
    epi_checksum_after = hashlib.sha1(
        str(sorted((str(n), round(v, 6)) for n, v in epi_items_after)).encode()
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
        if h.get("stable_frac"):
            meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):
            meta["phase_sync_last"] = h["phase_sync"][-1]
        if h.get("glyph_load_disr"):
            meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
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
        mode = str(G.graph.get("REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn")))
    mode = str(mode)

    # Similaridad basada en EPI (distancia absoluta)
    epi = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes}

    # Construir el MST de forma incremental (Prim) evitando grafo completo
    start = nodes[0]
    unvisited = set(nodes[1:])
    closest = {}
    for v in unvisited:
        closest[v] = (abs(epi[start] - epi[v]), start)
    mst_edges = set()
    while unvisited:
        v = min(unvisited, key=lambda x: closest[x][0])
        w, parent = closest[v]
        mst_edges.add(tuple(sorted((parent, v))))
        unvisited.remove(v)
        for u in unvisited:
            w = abs(epi[v] - epi[u])
            if w < closest[u][0]:
                closest[u] = (w, v)

    if mode == "community":
        # Detectar comunidades y reconstruir la red con metanodos
        comms = list(nx_comm.greedy_modularity_communities(G))
        if len(comms) <= 1:
            new_edges = set(mst_edges)
        else:
            k_val = (
                int(k)
                if k is not None
                else int(G.graph.get("REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2)))
            )
            # Grafo de comunidades basado en medias de EPI
            C = nx.Graph()
            for idx, comm in enumerate(comms):
                members = list(comm)
                epi_mean = list_mean(epi[n] for n in members)
                C.add_node(idx)
                set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
                C.nodes[idx]["members"] = members
            for i in C.nodes():
                for j in C.nodes():
                    if i < j:
                        w = abs(
                            get_attr(C.nodes[i], ALIAS_EPI, 0.0)
                            - get_attr(C.nodes[j], ALIAS_EPI, 0.0)
                        )
                        C.add_edge(i, j, weight=w)
            mst_c = nx.minimum_spanning_tree(C, weight="weight")
            new_edges = set(mst_c.edges())
            for u in C.nodes():
                epi_u = get_attr(C.nodes[u], ALIAS_EPI, 0.0)
                others = [v for v in C.nodes() if v != u]
                others.sort(key=lambda v: abs(epi_u - get_attr(C.nodes[v], ALIAS_EPI, 0.0)))
                for v in others[:k_val]:
                    if rnd.random() < p_rewire:
                        new_edges.add(tuple(sorted((u, v))))

            # Reemplazar nodos y aristas del grafo original por comunidades
            # clear_edges está disponible desde NetworkX 2.4 y evita materializar la lista
            # completa de aristas; tnfr requiere NetworkX>=2.6 (ver pyproject.toml)
            G.clear_edges()
            increment_edge_version(G)
            G.remove_nodes_from(list(G.nodes()))
            increment_edge_version(G)
            for idx in C.nodes():
                data = dict(C.nodes[idx])
                G.add_node(idx, **data)
            G.add_edges_from(new_edges)
            increment_edge_version(G)

            if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
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
    new_edges = set(mst_edges)
    if mode == "knn":
        k_val = (
            int(k)
            if k is not None
            else int(
                G.graph.get(
                    "REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2)
                )
            )
        )
        k_val = max(1, k_val)

        for u in nodes:
            epi_u = epi[u]
            dist_pairs = [
                (abs(epi_u - epi[v]), v) for v in nodes if v != u
            ]
            for _, v in heapq.nsmallest(k_val, dist_pairs):
                if rnd.random() < p_rewire:
                    new_edges.add(tuple(sorted((u, v))))

    # clear_edges disponible en NetworkX >=2.4; tnfr depende de >=2.6
    G.clear_edges()
    increment_edge_version(G)
    G.add_edges_from(new_edges)
    increment_edge_version(G)

def aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos: Optional[int] = None) -> None:
    # Ventanas y umbrales
    w_estab = (
        pasos_estables_consecutivos
        if pasos_estables_consecutivos is not None
        else int(G.graph.get("REMESH_STABILITY_WINDOW", REMESH_DEFAULTS["REMESH_STABILITY_WINDOW"]))
    )
    frac_req = float(
        G.graph.get("FRACTION_STABLE_REMESH", REMESH_DEFAULTS["FRACTION_STABLE_REMESH"])
    )
    req_extra = bool(
        G.graph.get("REMESH_REQUIRE_STABILITY", REMESH_DEFAULTS["REMESH_REQUIRE_STABILITY"])
    )
    min_sync = float(
        G.graph.get("REMESH_MIN_PHASE_SYNC", REMESH_DEFAULTS["REMESH_MIN_PHASE_SYNC"])
    )
    max_disr = float(
        G.graph.get("REMESH_MAX_GLYPH_DISR", REMESH_DEFAULTS["REMESH_MAX_GLYPH_DISR"])
    )
    min_sigma = float(
        G.graph.get("REMESH_MIN_SIGMA_MAG", REMESH_DEFAULTS["REMESH_MIN_SIGMA_MAG"])
    )
    min_R = float(
        G.graph.get("REMESH_MIN_KURAMOTO_R", REMESH_DEFAULTS["REMESH_MIN_KURAMOTO_R"])
    )
    min_sihi = float(
        G.graph.get("REMESH_MIN_SI_HI_FRAC", REMESH_DEFAULTS["REMESH_MIN_SI_HI_FRAC"])
    )

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
    cooldown = int(
        G.graph.get("REMESH_COOLDOWN_VENTANA", REMESH_DEFAULTS["REMESH_COOLDOWN_VENTANA"])
    )
    if step_idx - last < cooldown:
        return
    t_now = float(G.graph.get("_t", 0.0))
    last_ts = float(G.graph.get("_last_remesh_ts", -1e12))
    cooldown_ts = float(
        G.graph.get("REMESH_COOLDOWN_TS", REMESH_DEFAULTS.get("REMESH_COOLDOWN_TS", 0.0))
    )
    if cooldown_ts > 0 and (t_now - last_ts) < cooldown_ts:
        return
    # 4) Aplicar y registrar
    aplicar_remesh_red(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now
