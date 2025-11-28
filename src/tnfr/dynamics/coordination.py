"""Phase coordination helpers for TNFR dynamics."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar, cast
from ..alias import get_theta_attr, set_theta
from ..constants import (
    DEFAULTS,
    METRIC_DEFAULTS,
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    normalise_state_token,
)
from ..glyph_history import append_metric
from ..utils import angle_diff, resolve_chunk_size
from ..metrics.common import ensure_neighbors_map
from ..metrics.trig import neighbor_phase_mean_list
from ..metrics.trig_cache import get_trig_cache
from ..observers import DEFAULT_GLYPH_LOAD_SPAN, glyph_load, kuramoto_order
from ..types import FloatArray, NodeId, Phase, TNFRGraph
from ..utils import get_numpy

_DequeT = TypeVar("_DequeT")

ChunkArgs = tuple[
    Sequence[NodeId],
    Mapping[NodeId, Phase],
    Mapping[NodeId, float],
    Mapping[NodeId, float],
    Mapping[NodeId, Sequence[NodeId]],
    float,
    float,
    float,
]

__all__ = ("coordinate_global_local_phase",)


def _ensure_hist_deque(hist: MutableMapping[str, Any], key: str, maxlen: int) -> deque[_DequeT]:
    """Ensure history entry ``key`` is a deque with ``maxlen``."""

    dq = hist.setdefault(key, deque(maxlen=maxlen))
    if not isinstance(dq, deque):
        dq = deque(dq, maxlen=maxlen)
        hist[key] = dq
    return cast("deque[_DequeT]", dq)


def _read_adaptive_params(
    g: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float, float]:
    """Obtain configuration and current values for phase adaptation."""

    cfg = g.get("PHASE_ADAPT", DEFAULTS.get("PHASE_ADAPT", {}))
    kG = float(g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"]))
    kL = float(g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"]))
    return cast(Mapping[str, Any], cfg), kG, kL


def _compute_state(G: TNFRGraph, cfg: Mapping[str, Any]) -> tuple[str, float, float]:
    """Return the canonical network state and supporting metrics."""

    R = kuramoto_order(G)
    dist = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
    disr = float(dist.get("_disruptors", 0.0)) if dist else 0.0

    R_hi = float(cfg.get("R_hi", 0.90))
    R_lo = float(cfg.get("R_lo", 0.60))
    disr_hi = float(cfg.get("disr_hi", 0.50))
    disr_lo = float(cfg.get("disr_lo", 0.25))
    if (R >= R_hi) and (disr <= disr_lo):
        state = STATE_STABLE
    elif (R <= R_lo) or (disr >= disr_hi):
        state = STATE_DISSONANT
    else:
        state = STATE_TRANSITION
    return state, float(R), disr


def _smooth_adjust_k(
    kG: float, kL: float, state: str, cfg: Mapping[str, Any]
) -> tuple[float, float]:
    """Smoothly update kG/kL toward targets according to state."""

    kG_min = float(cfg.get("kG_min", 0.01))
    kG_max = float(cfg.get("kG_max", 0.20))
    kL_min = float(cfg.get("kL_min", 0.05))
    kL_max = float(cfg.get("kL_max", 0.25))

    state = normalise_state_token(state)

    if state == STATE_DISSONANT:
        kG_t = kG_max
        kL_t = 0.5 * (kL_min + kL_max)  # keep kL mid-range to preserve local plasticity
    elif state == STATE_STABLE:
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


def _phase_adjust_chunk(args: ChunkArgs) -> list[tuple[NodeId, Phase]]:
    """Return coordinated phase updates for the provided chunk."""

    (
        nodes,
        theta_map,
        cos_map,
        sin_map,
        neighbors_map,
        thG,
        kG,
        kL,
    ) = args
    updates: list[tuple[NodeId, Phase]] = []
    for node in nodes:
        th = float(theta_map.get(node, 0.0))
        neigh = neighbors_map.get(node, ())
        if neigh:
            thL = neighbor_phase_mean_list(
                neigh,
                cos_map,
                sin_map,
                np=None,
                fallback=th,
            )
        else:
            thL = th
        dG = angle_diff(thG, th)
        dL = angle_diff(thL, th)
        updates.append((node, cast(Phase, th + kG * dG + kL * dL)))
    return updates


def coordinate_global_local_phase(
    G: TNFRGraph,
    global_force: float | None = None,
    local_force: float | None = None,
    *,
    n_jobs: int | None = None,
) -> None:
    """Coordinate phase using a blend of global and neighbour coupling.

    This operator harmonises a TNFR graph by iteratively nudging each node's
    phase toward the global Kuramoto mean while respecting the local
    neighbourhood attractor. The global (``kG``) and local (``kL``) coupling
    gains reshape phase coherence by modulating how strongly nodes follow the
    network-wide synchrony versus immediate neighbours. When explicit coupling
    overrides are not supplied, the gains adapt based on current ΔNFR telemetry
    and the structural state recorded in the graph history. Adaptive updates
    mutate the ``history`` buffers for phase state, order parameter, disruptor
    load, and the stored coupling gains.

    Parameters
    ----------
    G : TNFRGraph
        Graph whose nodes expose TNFR phase attributes and ΔNFR telemetry. The
        graph's ``history`` mapping is updated in-place when adaptive gain
        smoothing is active.
    global_force : float, optional
        Override for the global coupling gain ``kG``. When provided, adaptive
        gain estimation is skipped and the global history buffers are left
        untouched.
    local_force : float, optional
        Override for the local coupling gain ``kL``. Analogous to
        ``global_force``, the adaptive pathway is bypassed when supplied.
    n_jobs : int, optional
        Maximum number of worker processes for distributing local updates.
        Values of ``None`` or ``<=1`` perform updates sequentially. NumPy
        availability forces sequential execution because vectorised updates are
        faster than multiprocess handoffs.

    Returns
    -------
    None
        This operator updates node phases in-place and does not allocate a new
        graph structure.

    Examples
    --------
    Coordinate phase on a minimal TNFR network while inspecting ΔNFR telemetry
    and history traces::

        >>> import networkx as nx
        >>> from tnfr.dynamics.coordination import coordinate_global_local_phase
        >>> G = nx.Graph()
        >>> G.add_nodes_from(("a", {"theta": 0.0, "ΔNFR": 0.08}),
        ...                   ("b", {"theta": 1.2, "ΔNFR": -0.05}))
        >>> G.add_edge("a", "b")
        >>> G.graph["history"] = {}
        >>> coordinate_global_local_phase(G)
        >>> list(round(G.nodes[n]["theta"], 3) for n in G)
        [0.578, 0.622]
        >>> history = G.graph["history"]
        >>> sorted(history)
        ['phase_R', 'phase_disr', 'phase_kG', 'phase_kL', 'phase_state']
        >>> history["phase_kG"][-1] <= history["phase_kL"][-1]
        True

    The resulting history buffers allow downstream observers to correlate
    ΔNFR adjustments with phase telemetry snapshots.
    """

    g = cast(dict[str, Any], G.graph)
    hist = cast(dict[str, Any], g.setdefault("history", {}))
    maxlen = int(g.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"]))
    hist_state = cast(deque[str], _ensure_hist_deque(hist, "phase_state", maxlen))
    if hist_state:
        normalised_states = [normalise_state_token(item) for item in hist_state]
        if normalised_states != list(hist_state):
            hist_state.clear()
            hist_state.extend(normalised_states)
    hist_R = cast(deque[float], _ensure_hist_deque(hist, "phase_R", maxlen))
    hist_disr = cast(deque[float], _ensure_hist_deque(hist, "phase_disr", maxlen))

    if (global_force is not None) or (local_force is not None):
        kG = float(
            global_force
            if global_force is not None
            else g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"])
        )
        kL = float(
            local_force
            if local_force is not None
            else g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"])
        )
    else:
        cfg, kG, kL = _read_adaptive_params(g)

        if bool(cfg.get("enabled", False)):
            state, R, disr = _compute_state(G, cfg)
            kG, kL = _smooth_adjust_k(kG, kL, state, cfg)

            hist_state.append(state)
            hist_R.append(float(R))
            hist_disr.append(float(disr))

    g["PHASE_K_GLOBAL"] = kG
    g["PHASE_K_LOCAL"] = kL
    append_metric(hist, "phase_kG", float(kG))
    append_metric(hist, "phase_kL", float(kL))

    jobs: int | None
    try:
        jobs = None if n_jobs is None else int(n_jobs)
    except (TypeError, ValueError):
        jobs = None
    if jobs is not None and jobs <= 1:
        jobs = None

    np = get_numpy()
    if np is not None:
        jobs = None

    nodes: list[NodeId] = [cast(NodeId, node) for node in G.nodes()]
    num_nodes = len(nodes)
    if not num_nodes:
        return

    trig = get_trig_cache(G, np=np)
    theta_map = cast(dict[NodeId, Phase], trig.theta)
    cos_map = cast(dict[NodeId, float], trig.cos)
    sin_map = cast(dict[NodeId, float], trig.sin)

    neighbors_proxy = ensure_neighbors_map(G)
    neighbors_map: dict[NodeId, tuple[NodeId, ...]] = {}
    for n in nodes:
        try:
            neighbors_map[n] = tuple(cast(Sequence[NodeId], neighbors_proxy[n]))
        except KeyError:
            neighbors_map[n] = ()

    def _theta_value(node: NodeId) -> float:
        cached = theta_map.get(node)
        if cached is not None:
            return float(cached)
        attr_val = get_theta_attr(G.nodes[node], 0.0)
        return float(attr_val if attr_val is not None else 0.0)

    theta_vals = [_theta_value(n) for n in nodes]
    cos_vals = [float(cos_map.get(n, math.cos(theta_vals[idx]))) for idx, n in enumerate(nodes)]
    sin_vals = [float(sin_map.get(n, math.sin(theta_vals[idx]))) for idx, n in enumerate(nodes)]

    if np is not None:
        theta_arr = cast(FloatArray, np.fromiter(theta_vals, dtype=float))
        cos_arr = cast(FloatArray, np.fromiter(cos_vals, dtype=float))
        sin_arr = cast(FloatArray, np.fromiter(sin_vals, dtype=float))
        if cos_arr.size:
            mean_cos = float(np.mean(cos_arr))
            mean_sin = float(np.mean(sin_arr))
            thG = float(np.arctan2(mean_sin, mean_cos))
        else:
            thG = 0.0
        neighbor_means = [
            neighbor_phase_mean_list(
                neighbors_map.get(n, ()),
                cos_map,
                sin_map,
                np=np,
                fallback=theta_vals[idx],
            )
            for idx, n in enumerate(nodes)
        ]
        neighbor_arr = cast(FloatArray, np.fromiter(neighbor_means, dtype=float))
        theta_updates = theta_arr + kG * (thG - theta_arr) + kL * (neighbor_arr - theta_arr)
        for idx, node in enumerate(nodes):
            set_theta(G, node, float(theta_updates[int(idx)]))
        return

    mean_cos = math.fsum(cos_vals) / num_nodes
    mean_sin = math.fsum(sin_vals) / num_nodes
    thG = math.atan2(mean_sin, mean_cos)

    if jobs is None:
        for node in nodes:
            th = float(theta_map.get(node, 0.0))
            neigh = neighbors_map.get(node, ())
            if neigh:
                thL = neighbor_phase_mean_list(
                    neigh,
                    cos_map,
                    sin_map,
                    np=None,
                    fallback=th,
                )
            else:
                thL = th
            dG = angle_diff(thG, th)
            dL = angle_diff(thL, th)
            set_theta(G, node, float(th + kG * dG + kL * dL))
        return

    approx_chunk = math.ceil(len(nodes) / jobs) if jobs else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        len(nodes),
        minimum=1,
    )
    chunks = [nodes[idx : idx + chunk_size] for idx in range(0, len(nodes), chunk_size)]
    args: list[ChunkArgs] = [
        (
            chunk,
            theta_map,
            cos_map,
            sin_map,
            neighbors_map,
            thG,
            kG,
            kL,
        )
        for chunk in chunks
    ]
    results: dict[NodeId, Phase] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for res in executor.map(_phase_adjust_chunk, args):
            for node, value in res:
                results[node] = value
    for node in nodes:
        new_theta = results.get(node)
        base_theta = theta_map.get(node, 0.0)
        set_theta(G, node, float(new_theta if new_theta is not None else base_theta))
