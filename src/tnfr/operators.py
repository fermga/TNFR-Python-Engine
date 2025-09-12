"""Network operators.

This module implements:
- The 13 glyphs as smooth local operators.
- A dispatcher ``apply_glyph`` that maps the glyph name (with typographic
  apostrophe) to its function.
- Network remeshing: ``apply_network_remesh``,
  ``apply_topological_remesh`` and ``apply_remesh_if_globally_stable``.

Note on REMESH α (alpha) precedence:
1) ``G.graph["GLYPH_FACTORS"]["REMESH_alpha"]``
2) ``G.graph["REMESH_ALPHA"]``
3) ``REMESH_DEFAULTS["REMESH_ALPHA"]``
"""

# operators.py — TNFR canónica (ASCII-safe)
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import math
import hashlib
import heapq
from operator import ge, le
from functools import cache
from itertools import combinations, islice
from io import StringIO
from weakref import WeakKeyDictionary, WeakSet
from collections import deque
from cachetools import LRUCache

from .constants import DEFAULTS, REMESH_DEFAULTS, ALIAS_EPI, get_param
from .helpers.numeric import (
    list_mean,
    angle_diff,
    neighbor_phase_mean,
    neighbor_mean,
    kahan_sum,
)
from .helpers.cache import (
    increment_edge_version,
    ensure_node_offset_map,
)
from .alias import get_attr, set_attr
from .rng import (
    make_rng,
    base_seed,
    cache_enabled,
    clear_rng_cache as _clear_rng_cache,
    seed_hash,
)
from .callback_utils import invoke_callbacks
from .glyph_history import append_metric, ensure_history, current_step_idx
from .import_utils import import_nodonx, optional_import
from .types import Glyph
from .locking import get_lock

# Guarded by ``JITTER_CACHE.lock`` to ensure thread-safe access.
# ``JITTER_CACHE.seq`` stores per-scope jitter sequence counters in an LRU cache
# bounded to avoid unbounded memory usage.
_JITTER_MAX_ENTRIES = 1024


class JitterCache:
    """Container for jitter-related caches."""

    def __init__(self, max_entries: int = _JITTER_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self.seq: LRUCache[tuple[int, int], int] = LRUCache(maxsize=max_entries)
        self.graphs: WeakSet[Any] = WeakSet()
        self.settings: dict[str, Any] = {"max_entries": max_entries}
        self.lock = get_lock("jitter")

    def setup(self, force: bool = False) -> None:
        """Ensure ``seq`` matches the configured size."""
        max_entries = self.max_entries
        if force or self.settings.get("max_entries") != max_entries:
            self.seq = LRUCache(maxsize=max_entries)
            self.settings["max_entries"] = max_entries

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""
        with self.lock:
            _clear_rng_cache()
            self.setup(force=True)
            for G in list(self.graphs):
                cache = G.graph.get("_jitter_seed_hash")
                if cache is not None:
                    cache.clear()
            self.graphs.clear()


# Module-level singleton
JITTER_CACHE = JitterCache()

# Backward-compatibility aliases; updated by ``_update_cache_refs``.
_JITTER_SEQ = JITTER_CACHE.seq
_JITTER_GRAPHS = JITTER_CACHE.graphs
_JITTER_SETTINGS = JITTER_CACHE.settings


def _update_cache_refs(cache: JitterCache) -> None:
    global _JITTER_SEQ, _JITTER_GRAPHS, _JITTER_SETTINGS
    _JITTER_SEQ = cache.seq
    _JITTER_GRAPHS = cache.graphs
    _JITTER_SETTINGS = cache.settings


def setup_jitter_cache(
    force: bool = False, cache: JitterCache = JITTER_CACHE
) -> None:
    """Ensure jitter cache matches the configured size.

    Parameters
    ----------
    force:
        When ``True`` the cache is always recreated. Otherwise it is only
        reset when the configured ``_JITTER_MAX_ENTRIES`` value changes.
    """

    cache.max_entries = _JITTER_MAX_ENTRIES
    cache.setup(force)
    _update_cache_refs(cache)


# Initialize jitter cache on module import.
setup_jitter_cache(force=True)

if TYPE_CHECKING:
    from .node import NodoProtocol

__all__ = (
    "JitterCache",
    "JITTER_CACHE",
    "setup_jitter_cache",
    "clear_rng_cache",
    "random_jitter",
    "get_glyph_factors",
    "apply_glyph_obj",
    "apply_glyph",
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
)


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def clear_rng_cache(cache: JitterCache = JITTER_CACHE) -> None:
    """Clear cached RNGs and jitter state."""
    cache.clear()
    _update_cache_refs(cache)


@cache
def _get_networkx_modules():
    nx = optional_import("networkx")
    if nx is None:
        raise ImportError(
            "networkx is required for network operators; install 'networkx' "
            "to enable this feature"
        )
    nx_comm = optional_import("networkx.algorithms.community")
    if nx_comm is None:
        raise ImportError(
            "networkx.algorithms.community is required for community-based "
            "operations; install 'networkx' to enable this feature"
        )
    return nx, nx_comm


def _resolve_jitter_seed(node: NodoProtocol) -> tuple[int, int]:
    NodoNX = import_nodonx()  # cache import to avoid repeated lookups
    if isinstance(node, NodoNX):
        return _node_offset(node.G, node.n), id(node.G)
    uid = getattr(node, "_noise_uid", None)
    if uid is None:
        uid = id(node)
        setattr(node, "_noise_uid", uid)
    return int(uid), id(node)


def _get_jitter_cache(
    node: NodoProtocol, cache: JitterCache = JITTER_CACHE
) -> dict:
    """Return the jitter cache for ``node``.

    If the node cannot store attributes, fall back to a graph-level
    ``WeakKeyDictionary`` and ensure the graph is tracked in
    ``_JITTER_GRAPHS`` so its cache can be cleared when needed.
    """

    cache = getattr(node, "_jitter_seed_hash", None)
    if cache is not None:
        return cache

    try:
        cache = {}
        setattr(node, "_jitter_seed_hash", cache)
        return cache
    except AttributeError:
        graph_cache = node.graph.get("_jitter_seed_hash")
        if graph_cache is None:
            graph_cache = WeakKeyDictionary()
            node.graph["_jitter_seed_hash"] = graph_cache
            with cache.lock:
                cache.graphs.add(node.graph)
        cache = graph_cache.get(node)
        if cache is None:
            cache = {}
            graph_cache[node] = cache
        return cache


def random_jitter(node: NodoProtocol, amplitude: float) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for
    ``node``.

    The value is derived from ``(RANDOM_SEED, node.offset())`` and does
    not store references to nodes. ``make_rng`` provides a global LRU
    cache keyed by ``(seed, key)`` so sequences advance deterministically
    across calls. The Blake2 hash used to derive ``seed`` is cached per
    node in ``_jitter_seed_hash`` keyed by ``(seed_root, scope_id)`` to
    avoid recomputation. Clear this cache if the base seed changes to
    prevent stale values from being reused. Per-scope sequence numbers are
    stored in ``_JITTER_SEQ``, a bounded LRU cache limited to
    ``_JITTER_MAX_ENTRIES`` entries; older scopes are discarded when the
    limit is exceeded.
    """

    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    if amplitude == 0:
        return 0.0

    seed_root = base_seed(node.G)
    seed_key, scope_id = _resolve_jitter_seed(node)

    cache = _get_jitter_cache(node)

    cache_key = (seed_root, scope_id)
    seed = cache.get(cache_key)
    if seed is None:
        seed = seed_hash(seed_root, scope_id)
        cache[cache_key] = seed
    seq = 0
    if cache_enabled(node.G):
        with JITTER_CACHE.lock:
            seq = JITTER_CACHE.seq.get(cache_key, 0)
            JITTER_CACHE.seq[cache_key] = seq + 1
    rng = make_rng(seed, seed_key + seq, node.G)
    return rng.uniform(-amplitude, amplitude)


def get_glyph_factors(node: NodoProtocol) -> dict[str, Any]:
    """Return glyph factors for ``node`` with defaults."""
    return node.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"].copy())


# -------------------------
# Glyphs (operadores locales)
# -------------------------


def _any_neighbor_has(node: NodoProtocol, aliases: tuple[str, ...]) -> bool:
    """Return ``True`` if any neighbour defines one of ``aliases``."""
    if hasattr(node, "G"):
        G = node.G
        return any(
            any(a in G.nodes[v] for a in aliases) for v in G.neighbors(node.n)
        )
    return any(
        any(hasattr(neigh, a) for a in aliases) for neigh in node.neighbors()
    )


def _gather_neighbors(node: NodoProtocol) -> tuple[list[NodoProtocol], float]:
    """Return neighbour list and their mean ``EPI``.

    When ``node`` is bound to a graph and none of its neighbours defines an
    ``ALIAS_EPI`` attribute, the neighbours list is returned empty and the
    mean defaults to ``node.EPI``.  This allows callers to fall back to the
    node's own values without additional checks.
    """

    epi = node.EPI
    neigh = list(node.neighbors())
    if not neigh:
        return [], epi

    if hasattr(node, "G"):
        if not _any_neighbor_has(node, ALIAS_EPI):
            return [], epi
        epi_bar = neighbor_mean(node.G, node.n, ALIAS_EPI, default=epi)
        NodoNX = import_nodonx()
        neigh = [
            v if hasattr(v, "EPI") else NodoNX.from_graph(node.G, v)
            for v in neigh
        ]
    else:
        epi_bar = list_mean((v.EPI for v in neigh), default=epi)

    return neigh, epi_bar


def _determine_dominant(
    neigh: list[NodoProtocol], default_kind: str
) -> tuple[str, float]:
    """Return dominant ``epi_kind`` among ``neigh`` and its absolute ``EPI``.

    Falls back to ``default_kind`` when neighbours lack ``epi_kind``.
    """

    best_kind: str | None = None
    best_abs = 0.0
    for v in neigh:
        abs_v = abs(v.EPI)
        if abs_v > best_abs:
            best_abs = abs_v
            best_kind = v.epi_kind

    if not best_kind:
        return default_kind, 0.0
    return best_kind, best_abs


def _mix_epi_with_neighbors(
    node: NodoProtocol, mix: float, default_glyph: Glyph | str
) -> tuple[float, str]:
    """Mix ``EPI`` of ``node`` with the mean of its neighbours.

    ``mix`` controls the neighbour influence fraction and ``default_glyph``
    is assigned when there are no neighbours or no dominant one.  When no
    neighbour defines an ``ALIAS_EPI``, the costly mean computation is
    skipped and ``node.EPI`` is used directly.

    Returns
    -------
    epi_bar: float
        Mean ``EPI`` of the neighbours (``node.EPI`` if there are none or
        they lack ``ALIAS_EPI``).
    dominant: str
        ``epi_kind`` with the highest absolute ``EPI`` between ``node`` and
        its neighbours. Falls back to ``default_glyph`` when undefined.
    """

    default_kind = (
        default_glyph.value
        if isinstance(default_glyph, Glyph)
        else str(default_glyph)
    )
    epi = node.EPI
    neigh, epi_bar = _gather_neighbors(node)

    if not neigh:
        node.epi_kind = default_kind
        return epi, default_kind

    dominant, best_abs = _determine_dominant(neigh, default_kind)
    new_epi = (1 - mix) * epi + mix * epi_bar
    node.EPI = new_epi
    final = dominant if best_abs > abs(new_epi) else node.epi_kind
    if not final:
        final = default_kind
    node.epi_kind = final
    return epi_bar, final


def _op_AL(node: NodoProtocol, gf: dict[str, Any]) -> None:  # AL — Emisión
    f = float(gf.get("AL_boost", 0.05))
    node.EPI = node.EPI + f


def _op_EN(node: NodoProtocol, gf: dict[str, Any]) -> None:  # EN — Recepción
    mix = float(gf.get("EN_mix", 0.25))
    _mix_epi_with_neighbors(node, mix, Glyph.EN)


def _op_IL(node: NodoProtocol, gf: dict[str, Any]) -> None:  # IL — Coherencia
    factor = float(gf.get("IL_dnfr_factor", 0.7))
    node.dnfr = factor * getattr(node, "dnfr", 0.0)


def _op_OZ(node: NodoProtocol, gf: dict[str, Any]) -> None:  # OZ — Disonancia
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


def _um_candidate_iter(node: NodoProtocol):
    sample_ids = node.graph.get("_node_sample")
    if sample_ids is not None and hasattr(node, "G"):
        NodoNX = import_nodonx()
        base = (NodoNX.from_graph(node.G, j) for j in sample_ids)
    else:
        base = node.all_nodes()
    for j in base:
        same = (j is node) or (
            getattr(node, "n", None) == getattr(j, "n", None)
        )
        if same or node.has_edge(j):
            continue
        yield j


def _um_select_candidates(
    node: NodoProtocol,
    candidates,
    limit: int,
    mode: str,
    th: float,
):
    """Select a subset of ``candidates`` for UM coupling.

    ``candidates`` may be a large or lazy iterable. This function consumes
    it incrementally to avoid loading every element into memory.
    """
    rng = make_rng(int(node.graph.get("RANDOM_SEED", 0)), node.offset(), node.G)

    if limit <= 0:
        # No limit requested; fully materialize the iterable.
        return list(candidates)

    if mode == "proximity":
        # ``nsmallest`` only keeps ``limit`` elements in memory.
        return heapq.nsmallest(
            limit, candidates, key=lambda j: abs(angle_diff(j.theta, th))
        )

    # Deterministic reservoir sampling for large iterables.
    reservoir = list(islice(candidates, limit))
    for i, cand in enumerate(candidates, start=limit):
        j = rng.randint(0, i)
        if j < limit:
            reservoir[j] = cand

    if mode == "sample":
        rng.shuffle(reservoir)

    return reservoir


def _op_UM(node: NodoProtocol, gf: dict[str, Any]) -> None:  # UM — Coupling
    """Align phase and optionally create functional links.

    Link search can be reduced by evaluating only a subset of candidates.
    ``UM_CANDIDATE_COUNT`` sets how many nodes to consider and
    ``UM_CANDIDATE_MODE`` selects the strategy:

    * ``"proximity"``: choose nodes closest in phase.
    * ``"sample"``: take a deterministic sample.

    ``dynamics.step`` keeps a refreshed random sample in
    ``G.graph['_node_sample']``. When el grafo es pequeño (``<50``
    nodes) the sample contains all nodes and sampling is disabled.

    This preserves coupling logic without scanning all nodes.
    """

    k = float(gf.get("UM_theta_push", 0.25))
    th = node.theta
    thL = neighbor_phase_mean(node)
    d = angle_diff(thL, th)
    node.theta = th + k * d

    if bool(node.graph.get("UM_FUNCTIONAL_LINKS", False)):
        thr = float(
            node.graph.get(
                "UM_COMPAT_THRESHOLD",
                DEFAULTS.get("UM_COMPAT_THRESHOLD", 0.75),
            )
        )
        epi_i = node.EPI
        si_i = node.Si

        limit = int(node.graph.get("UM_CANDIDATE_COUNT", 0))
        mode = str(node.graph.get("UM_CANDIDATE_MODE", "sample")).lower()
        candidates = _um_select_candidates(
            node, _um_candidate_iter(node), limit, mode, th
        )

        for j in candidates:
            th_j = j.theta
            dphi = abs(angle_diff(th_j, th)) / math.pi
            epi_j = j.EPI
            si_j = j.Si
            epi_sim = 1.0 - abs(epi_i - epi_j) / (
                abs(epi_i) + abs(epi_j) + 1e-9
            )
            si_sim = 1.0 - abs(si_i - si_j)
            compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
            if compat >= thr:
                node.add_edge(j, compat)


def _op_RA(node: NodoProtocol, gf: dict[str, Any]) -> None:  # RA — Resonancia
    diff = float(gf.get("RA_epi_diff", 0.15))
    _mix_epi_with_neighbors(node, diff, Glyph.RA)


def _op_SHA(node: NodoProtocol, gf: dict[str, Any]) -> None:  # SHA — Silencio
    factor = float(gf.get("SHA_vf_factor", 0.85))
    node.vf = factor * node.vf


factor_val = 1.15
factor_nul = 0.85
_SCALE_FACTORS = {Glyph.VAL: factor_val, Glyph.NUL: factor_nul}


def _op_scale(node: NodoProtocol, glyph: Glyph, factor: float) -> None:
    """Scale node ``EPI`` and update ``epi_kind``."""
    node.EPI = factor * node.EPI
    node.epi_kind = glyph.value if isinstance(glyph, Glyph) else str(glyph)


def _make_scale_op(glyph: Glyph):
    def _op(node: NodoProtocol, gf: dict[str, Any]) -> None:
        factor_val = float(gf.get("VAL_scale", _SCALE_FACTORS[Glyph.VAL]))
        factor_nul = float(gf.get("NUL_scale", _SCALE_FACTORS[Glyph.NUL]))
        factor = factor_val if glyph is Glyph.VAL else factor_nul
        _op_scale(node, glyph, factor)

    return _op


def _op_THOL(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # THOL — Autoorganización
    a = float(gf.get("THOL_accel", 0.10))
    node.dnfr = node.dnfr + a * getattr(node, "d2EPI", 0.0)


def _op_ZHIR(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # ZHIR — Mutación
    shift = float(gf.get("ZHIR_theta_shift", math.pi / 2))
    node.theta = node.theta + shift


def _op_NAV(
    node: NodoProtocol, gf: dict[str, Any]
) -> None:  # NAV — Transición
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


def _op_REMESH(
    node: NodoProtocol, gf: dict[str, Any] | None = None
) -> None:  # REMESH — aviso
    step_idx = current_step_idx(node)
    last_warn = node.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = (
            "REMESH es a escala de red. Usa apply_remesh_if_globally_"
            "stable(G) o apply_network_remesh(G)."
        )
        hist = ensure_history(node)
        append_metric(
            hist,
            "events",
            ("warn", {"step": step_idx, "node": None, "msg": msg}),
        )
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
    Glyph.VAL: _make_scale_op(Glyph.VAL),
    Glyph.NUL: _make_scale_op(Glyph.NUL),
    Glyph.THOL: _op_THOL,
    Glyph.ZHIR: _op_ZHIR,
    Glyph.NAV: _op_NAV,
    Glyph.REMESH: _op_REMESH,
}


def apply_glyph_obj(
    node: NodoProtocol, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Apply ``glyph`` to an object satisfying :class:`NodoProtocol`."""

    try:
        g = glyph if isinstance(glyph, Glyph) else Glyph(str(glyph))
    except ValueError:
        step_idx = current_step_idx(node)
        hist = ensure_history(node)
        append_metric(
            hist,
            "events",
            (
                "warn",
                {
                    "step": step_idx,
                    "node": getattr(node, "n", None),
                    "msg": f"glyph desconocido: {glyph}",
                },
            ),
        )
        raise ValueError(f"glyph desconocido: {glyph}")

    op = _NAME_TO_OP.get(g)
    if op is None:
        raise ValueError(f"glyph sin operador: {g}")
    if window is None:
        window = int(get_param(node, "GLYPH_HYSTERESIS_WINDOW"))
    gf = get_glyph_factors(node)
    op(node, gf)
    node.push_glyph(g.value, window)


def apply_glyph(
    G, n, glyph: Glyph | str, *, window: int | None = None
) -> None:
    """Adapter to operate on ``networkx`` graphs."""
    NodoNX = import_nodonx()
    node = NodoNX(G, n)
    apply_glyph_obj(node, glyph, window=window)


# -------------------------
# REMESH de red (usa _epi_hist capturado en dynamics.step)
# -------------------------


def _remesh_alpha_info(G):
    """Return ``(alpha, source)`` with explicit precedence."""
    if bool(
        G.graph.get("REMESH_ALPHA_HARD", REMESH_DEFAULTS["REMESH_ALPHA_HARD"])
    ):
        val = float(
            G.graph.get("REMESH_ALPHA", REMESH_DEFAULTS["REMESH_ALPHA"])
        )
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return (
        float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        "REMESH_DEFAULTS.REMESH_ALPHA",
    )


def _snapshot_topology(G, nx):
    """Return a hash representing the current graph topology."""
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        return hashlib.sha1(topo_str.encode()).hexdigest()[:12]
    except (AttributeError, TypeError, nx.NetworkXError):
        return None


def _snapshot_epi(G):
    """Return ``(mean, checksum)`` of the node EPI values."""
    buf = StringIO()
    values = []
    for n, data in G.nodes(data=True):
        v = float(get_attr(data, ALIAS_EPI, 0.0))
        values.append(v)
        buf.write(f"{str(n)}:{round(v, 6)};")
    total = kahan_sum(values)
    mean_val = total / len(values) if values else 0.0
    checksum = hashlib.sha1(buf.getvalue().encode()).hexdigest()[:12]
    return float(mean_val), checksum


def _log_remesh_event(G, meta):
    """Store remesh metadata and optionally log and trigger callbacks."""
    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        append_metric(hist, "remesh_events", dict(meta))
    invoke_callbacks(G, "on_remesh", dict(meta))


def apply_network_remesh(G) -> None:
    """Network-scale REMESH using ``_epi_hist`` with multi-scale memory."""
    # REMESH_TAU: alias legado resuelto por ``get_param``
    nx, _ = _get_networkx_modules()
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
    topo_hash = _snapshot_topology(G, nx)
    epi_mean_before, epi_checksum_before = _snapshot_epi(G)

    # --- Mezcla (1-α)·now + α·old ---
    for n, nd in G.nodes(data=True):
        epi_now = get_attr(nd, ALIAS_EPI, 0.0)
        epi_old_l = float(past_l.get(n, epi_now))
        epi_old_g = float(past_g.get(n, epi_now))
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        set_attr(nd, ALIAS_EPI, mixed)

    # --- Snapshot EPI (DESPUÉS) ---
    epi_mean_after, epi_checksum_after = _snapshot_epi(G)

    # --- Metadatos y logging de evento ---
    step_idx = current_step_idx(G)
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
    h = ensure_history(G)
    if h:
        if h.get("stable_frac"):
            meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):
            meta["phase_sync_last"] = h["phase_sync"][-1]
        if h.get("glyph_load_disr"):
            meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    _log_remesh_event(G, meta)


def _mst_edges_from_epi(nx, nodes, epi):
    """Return MST edges based on absolute EPI distance."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from(
        (u, v, abs(epi[u] - epi[v])) for u, v in combinations(nodes, 2)
    )
    return {
        tuple(sorted((u, v)))
        for u, v in nx.minimum_spanning_edges(H, data=False)
    }


def _knn_edges(nodes, epi, k_val, p_rewire, rnd):
    """Edges linking each node to its k nearest neighbours in EPI."""
    new_edges = set()
    for u in nodes:
        epi_u = epi[u]
        dist_pairs = [(abs(epi_u - epi[v]), v) for v in nodes if v != u]
        for _, v in heapq.nsmallest(k_val, dist_pairs):
            if rnd.random() < p_rewire:
                new_edges.add(tuple(sorted((u, v))))
    return new_edges


def _community_graph(comms, epi, nx):
    """Return community graph ``C`` with mean EPI per community."""
    C = nx.Graph()
    for idx, comm in enumerate(comms):
        members = list(comm)
        epi_mean = list_mean(epi[n] for n in members)
        C.add_node(idx)
        set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
        C.nodes[idx]["members"] = members
    for i, j in combinations(C.nodes(), 2):
        w = abs(
            get_attr(C.nodes[i], ALIAS_EPI, 0.0)
            - get_attr(C.nodes[j], ALIAS_EPI, 0.0)
        )
        C.add_edge(i, j, weight=w)
    return C


def _community_k_neighbor_edges(C, k_val, p_rewire, rnd):
    """Edges linking each community to its ``k`` nearest neighbours."""
    epi_vals = {n: get_attr(C.nodes[n], ALIAS_EPI, 0.0) for n in C.nodes()}
    ordered = sorted(C.nodes(), key=lambda v: epi_vals[v])
    new_edges = set()
    for idx, u in enumerate(ordered):
        epi_u = epi_vals[u]
        left = idx - 1
        right = idx + 1
        added = 0
        while added < k_val and (left >= 0 or right < len(ordered)):
            if left < 0:
                v = ordered[right]
                right += 1
            elif right >= len(ordered):
                v = ordered[left]
                left -= 1
            else:
                if abs(epi_u - epi_vals[ordered[left]]) <= abs(
                    epi_vals[ordered[right]] - epi_u
                ):
                    v = ordered[left]
                    left -= 1
                else:
                    v = ordered[right]
                    right += 1
            if rnd.random() < p_rewire:
                new_edges.add(tuple(sorted((u, v))))
            added += 1
    return new_edges


def _community_remesh(
    G,
    epi,
    k_val,
    p_rewire,
    rnd,
    nx,
    nx_comm,
    mst_edges,
    n_before,
):
    """Remesh ``G`` replacing nodes by modular communities."""
    comms = list(nx_comm.greedy_modularity_communities(G))
    if len(comms) <= 1:
        G.clear_edges()
        increment_edge_version(G)
        G.add_edges_from(mst_edges)
        increment_edge_version(G)
        return
    C = _community_graph(comms, epi, nx)
    mst_c = nx.minimum_spanning_tree(C, weight="weight")
    new_edges = set(mst_c.edges())
    new_edges |= _community_k_neighbor_edges(C, k_val, p_rewire, rnd)

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
        hist = G.graph.setdefault("history", {})
        mapping = {idx: C.nodes[idx].get("members", []) for idx in C.nodes()}
        append_metric(
            hist,
            "remesh_events",
            {
                "mode": "community",
                "n_before": n_before,
                "n_after": G.number_of_nodes(),
                "mapping": mapping,
            },
        )


def apply_topological_remesh(
    G,
    mode: str | None = None,
    *,
    k: int | None = None,
    p_rewire: float = 0.2,
    seed: int | None = None,
) -> None:
    """Approximate topological remeshing.

    - ``mode="knn"``: connect each node with its ``k`` most similar
      neighbours in EPI with probability ``p_rewire``.
    - ``mode="mst"``: preserve only a minimum spanning tree according to
      EPI distance.
    - ``mode="community"``: group by modular communities and connect
      them by inter-community similarity.

    Connectivity is always preserved by adding a base MST.
    """
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    base_seed = 0 if seed is None else int(seed)
    rnd = make_rng(base_seed, -2, G)
    rnd.seed(base_seed)

    if mode is None:
        mode = str(
            G.graph.get(
                "REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn")
            )
        )
    mode = str(mode)
    nx, nx_comm = _get_networkx_modules()
    # Similaridad basada en EPI (distancia absoluta)
    epi = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes}

    mst_edges = _mst_edges_from_epi(nx, nodes, epi)

    # Valor por defecto para ``k`` en los modos "community" y "knn"
    # (2 si no se especifica)
    default_k = int(
        G.graph.get(
            "REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2)
        )
    )
    # ``k_val`` se calcula una sola vez, asegurando un mínimo de 1
    k_val = max(1, int(k) if k is not None else default_k)

    if mode == "community":
        _community_remesh(
            G,
            epi,
            k_val,
            p_rewire,
            rnd,
            nx,
            nx_comm,
            mst_edges,
            n_before,
        )
        return

    new_edges = set(mst_edges)
    if mode == "knn":
        new_edges |= _knn_edges(nodes, epi, k_val, p_rewire, rnd)

    G.clear_edges()
    increment_edge_version(G)
    G.add_edges_from(new_edges)
    increment_edge_version(G)


def _extra_gating_ok(hist, cfg, w_estab):
    """Check additional stability gating conditions."""
    checks = [
        ("phase_sync", "REMESH_MIN_PHASE_SYNC", ge),
        ("glyph_load_disr", "REMESH_MAX_GLYPH_DISR", le),
        ("sense_sigma_mag", "REMESH_MIN_SIGMA_MAG", ge),
        ("kuramoto_R", "REMESH_MIN_KURAMOTO_R", ge),
        ("Si_hi_frac", "REMESH_MIN_SI_HI_FRAC", ge),
    ]
    for hist_key, cfg_key, op in checks:
        series = hist.get(hist_key)
        if series is not None and len(series) >= w_estab:
            win = series[-w_estab:]
            avg = sum(win) / len(win)
            if not op(avg, cfg[cfg_key]):
                return False
    return True


def apply_remesh_if_globally_stable(
    G, pasos_estables_consecutivos: int | None = None
) -> None:
    params = [
        (
            "REMESH_STABILITY_WINDOW",
            int,
            REMESH_DEFAULTS["REMESH_STABILITY_WINDOW"],
        ),
        (
            "REMESH_REQUIRE_STABILITY",
            bool,
            REMESH_DEFAULTS["REMESH_REQUIRE_STABILITY"],
        ),
        (
            "REMESH_MIN_PHASE_SYNC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_PHASE_SYNC"],
        ),
        (
            "REMESH_MAX_GLYPH_DISR",
            float,
            REMESH_DEFAULTS["REMESH_MAX_GLYPH_DISR"],
        ),
        (
            "REMESH_MIN_SIGMA_MAG",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SIGMA_MAG"],
        ),
        (
            "REMESH_MIN_KURAMOTO_R",
            float,
            REMESH_DEFAULTS["REMESH_MIN_KURAMOTO_R"],
        ),
        (
            "REMESH_MIN_SI_HI_FRAC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SI_HI_FRAC"],
        ),
        (
            "REMESH_COOLDOWN_VENTANA",
            int,
            REMESH_DEFAULTS["REMESH_COOLDOWN_VENTANA"],
        ),
        ("REMESH_COOLDOWN_TS", float, REMESH_DEFAULTS["REMESH_COOLDOWN_TS"]),
    ]
    cfg = {}
    for key, conv, _default in params:
        cfg[key] = conv(get_param(G, key))
    # Parámetros de remallado: ventana de estabilidad, umbrales y cooldowns.
    frac_req = float(get_param(G, "FRACTION_STABLE_REMESH"))
    w_estab = (
        pasos_estables_consecutivos
        if pasos_estables_consecutivos is not None
        else cfg["REMESH_STABILITY_WINDOW"]
    )

    hist = ensure_history(G)
    sf = hist.setdefault("stable_frac", [])
    if len(sf) < w_estab:
        return
    win_sf = sf[-w_estab:]
    if not all(v >= frac_req for v in win_sf):
        return
    if cfg["REMESH_REQUIRE_STABILITY"] and not _extra_gating_ok(
        hist, cfg, w_estab
    ):
        return

    last = G.graph.get("_last_remesh_step", -(10**9))
    step_idx = len(sf)
    if step_idx - last < cfg["REMESH_COOLDOWN_VENTANA"]:
        return
    t_now = float(G.graph.get("_t", 0.0))
    last_ts = float(G.graph.get("_last_remesh_ts", -1e12))
    if (
        cfg["REMESH_COOLDOWN_TS"] > 0
        and (t_now - last_ts) < cfg["REMESH_COOLDOWN_TS"]
    ):
        return

    apply_network_remesh(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now
