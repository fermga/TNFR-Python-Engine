from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

if TYPE_CHECKING:  # pragma: no cover - typing imports only
    try:
        import numpy as np_typing
        import numpy.typing as npt
    except ImportError:  # pragma: no cover - optional typing dependency
        FloatArray: TypeAlias = Any
    else:
        FloatArray: TypeAlias = npt.NDArray[np_typing.float_]
else:  # pragma: no cover - runtime without numpy typing
    FloatArray: TypeAlias = Any

from ..types import (
    CoherenceMetric,
    DeltaNFR,
    EPIValue,
    GlyphSelector,
    Glyph,
    HistoryState,
    NodeId,
    Phase,
    SelectorPreselectionChoices,
    SelectorPreselectionMetrics,
    SelectorMetrics,
    SelectorNorms,
    SelectorThresholds,
    SelectorWeights,
    TNFRGraph,
)

# Importar compute_Si y apply_glyph a nivel de módulo evita el coste de
# realizar la importación en cada paso de la dinámica. Como los módulos de
# origen no dependen de ``dynamics``, no se introducen ciclos.
from ..operators import apply_remesh_if_globally_stable, apply_glyph
from ..validation.grammar import enforce_canonical_grammar, on_applied_glyph
from ..constants import (
    DEFAULTS,
    METRIC_DEFAULTS,
    get_aliases,
    get_param,
    get_graph_param,
)
from ..observers import DEFAULT_GLYPH_LOAD_SPAN, glyph_load, kuramoto_order

from ..helpers.numeric import (
    clamp,
    clamp01,
    angle_diff,
)
from ..metrics.trig import neighbor_phase_mean_list
from ..alias import (
    collect_attr,
    get_attr,
    get_theta_attr,
    set_vf,
    set_attr,
    set_theta,
    set_theta_attr,
    multi_recompute_abs_max,
)
from ..metrics.sense_index import compute_Si
from ..metrics.common import (
    compute_dnfr_accel_max,
    ensure_neighbors_map,
    merge_and_normalize_weights,
)
from ..metrics.trig_cache import get_trig_cache
from ..callback_utils import CallbackEvent, callback_manager
from ..glyph_history import recent_glyph, ensure_history, append_metric
from ..selector import (
    _selector_thresholds,
    _norms_para_selector,
    _calc_selector_score,
    _apply_selector_hysteresis,
)
from ..config.operator_names import TRANSITION
from ..utils import get_numpy

from .sampling import update_node_sample as _update_node_sample
from .dnfr import (
    _prepare_dnfr_data,
    _init_dnfr_cache,
    _refresh_dnfr_vectors,
    _compute_neighbor_means,
    _compute_dnfr,
    default_compute_delta_nfr,
    set_delta_nfr_hook,
    dnfr_phase_only,
    dnfr_epi_vf_mixed,
    dnfr_laplacian,
)
from .integrators import (
    prepare_integration_params,
    update_epi_via_nodal_equation,
)

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_EPI = get_aliases("EPI")
ALIAS_SI = get_aliases("SI")
ALIAS_D2EPI = get_aliases("D2EPI")
ALIAS_DSI = get_aliases("DSI")

GlyphCode: TypeAlias = Glyph | str

__all__ = (
    "default_compute_delta_nfr",
    "set_delta_nfr_hook",
    "dnfr_phase_only",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "prepare_integration_params",
    "update_epi_via_nodal_equation",
    "apply_canonical_clamps",
    "validate_canon",
    "coordinate_global_local_phase",
    "adapt_vf_by_coherence",
    "default_glyph_selector",
    "parametric_glyph_selector",
    "step",
    "run",
    "_prepare_dnfr_data",
    "_init_dnfr_cache",
    "_refresh_dnfr_vectors",
    "_compute_neighbor_means",
    "_compute_dnfr",
)


HistoryLog = MutableSequence[MutableMapping[str, object]]
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


def _log_clamp(
    hist: HistoryLog,
    node: NodeId | None,
    attr: str,
    value: float,
    lo: float,
    hi: float,
) -> None:
    if value < lo or value > hi:
        hist.append({"node": node, "attr": attr, "value": float(value)})


def _normalize_job_overrides(
    job_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a dict with canonical override keys (uppercase without suffix)."""
    if not job_overrides:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in job_overrides.items():
        if key is None:
            continue
        key_str = str(key).upper()
        if key_str.endswith("_N_JOBS"):
            key_str = key_str[: -len("_N_JOBS")]
        normalized[key_str] = value
    return normalized


def _coerce_jobs_value(raw: Any) -> int | None:
    """Best-effort conversion of ``raw`` to an integer ``n_jobs`` value."""
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sanitize_jobs(value: int | None, *, allow_non_positive: bool) -> int | None:
    if value is None:
        return None
    if not allow_non_positive and value <= 0:
        return None
    return value


def _resolve_jobs_override(
    overrides: Mapping[str, Any],
    key: str,
    graph_value: Any,
    *,
    allow_non_positive: bool,
) -> int | None:
    """Resolve ``n_jobs`` using overrides before consulting ``graph_value``."""

    norm_key = key.upper()
    if overrides and norm_key in overrides:
        return _sanitize_jobs(
            _coerce_jobs_value(overrides.get(norm_key)),
            allow_non_positive=allow_non_positive,
        )

    return _sanitize_jobs(
        _coerce_jobs_value(graph_value),
        allow_non_positive=allow_non_positive,
    )


def apply_canonical_clamps(
    nd: dict[str, Any],
    G: TNFRGraph | None = None,
    node: NodeId | None = None,
) -> None:
    if G is not None:
        graph_dict = cast(dict[str, Any], G.graph)
        graph_data: Mapping[str, Any] = graph_dict
    else:
        graph_dict = None
        graph_data = DEFAULTS
    eps_min = float(graph_data.get("EPI_MIN", DEFAULTS["EPI_MIN"]))
    eps_max = float(graph_data.get("EPI_MAX", DEFAULTS["EPI_MAX"]))
    vf_min = float(graph_data.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(graph_data.get("VF_MAX", DEFAULTS["VF_MAX"]))
    theta_wrap = bool(graph_data.get("THETA_WRAP", DEFAULTS["THETA_WRAP"]))

    epi = cast(EPIValue, get_attr(nd, ALIAS_EPI, 0.0))
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th_val = get_theta_attr(nd, 0.0)
    th = cast(Phase, 0.0 if th_val is None else th_val)

    strict = bool(
        graph_data.get("VALIDATORS_STRICT", DEFAULTS.get("VALIDATORS_STRICT", False))
    )
    if strict and graph_dict is not None:
        history = cast(dict[str, Any], graph_dict.setdefault("history", {}))
        hist = cast(
            HistoryLog,
            history.setdefault("clamp_alerts", []),
        )
        _log_clamp(hist, node, "EPI", float(epi), eps_min, eps_max)
        _log_clamp(hist, node, "VF", float(vf), vf_min, vf_max)

    set_attr(nd, ALIAS_EPI, clamp(epi, eps_min, eps_max))

    vf_val = float(clamp(vf, vf_min, vf_max))
    if G is not None and node is not None:
        set_vf(G, node, vf_val, update_max=False)
    else:
        set_attr(nd, ALIAS_VF, vf_val)

    if theta_wrap:
        new_th = (th + math.pi) % (2 * math.pi) - math.pi
        if G is not None and node is not None:
            set_theta(G, node, new_th)
        else:
            set_theta_attr(nd, new_th)


def validate_canon(G: TNFRGraph) -> TNFRGraph:
    """Apply canonical clamps to all nodes of ``G``.

    Wrap phase and constrain ``EPI`` and ``νf`` to the ranges in ``G.graph``.
    If ``VALIDATORS_STRICT`` is active, alerts are logged in ``history``.
    """
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(cast(dict[str, Any], nd), G, cast(NodeId, n))
    maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF})
    G.graph.update(maxes)
    return G


def _read_adaptive_params(
    g: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    """Obtain configuration and current values for phase adaptation."""
    cfg = g.get("PHASE_ADAPT", DEFAULTS.get("PHASE_ADAPT", {}))
    kG = float(g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"]))
    kL = float(g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"]))
    return cfg, kG, kL


def _compute_state(G: TNFRGraph, cfg: Mapping[str, Any]) -> tuple[str, float, float]:
    """Return current state (stable/dissonant/transition) and metrics."""
    R = kuramoto_order(G)
    dist = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
    disr = (
        float(dist.get("_disruptors", dist.get("_disruptivos", 0.0)))
        if dist
        else 0.0
    )

    R_hi = float(cfg.get("R_hi", 0.90))
    R_lo = float(cfg.get("R_lo", 0.60))
    disr_hi = float(cfg.get("disr_hi", 0.50))
    disr_lo = float(cfg.get("disr_lo", 0.25))
    if (R >= R_hi) and (disr <= disr_lo):
        state = "estable"
    elif (R <= R_lo) or (disr >= disr_hi):
        state = "disonante"
    else:
        state = TRANSITION
    return state, float(R), disr


def _smooth_adjust_k(
    kG: float, kL: float, state: str, cfg: dict[str, Any]
) -> tuple[float, float]:
    """Smoothly update kG/kL toward targets according to state."""
    kG_min = float(cfg.get("kG_min", 0.01))
    kG_max = float(cfg.get("kG_max", 0.20))
    kL_min = float(cfg.get("kL_min", 0.05))
    kL_max = float(cfg.get("kL_max", 0.25))

    if state == "disonante":
        kG_t = kG_max
        kL_t = 0.5 * (
            kL_min + kL_max
        )  # local medio para no perder plasticidad
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


def _ensure_hist_deque(
    hist: dict[str, Any], key: str, maxlen: int
) -> deque[_DequeT]:
    """Ensure history entry ``key`` is a deque with ``maxlen``."""
    dq = hist.setdefault(key, deque(maxlen=maxlen))
    if not isinstance(dq, deque):
        dq = deque(dq, maxlen=maxlen)
        hist[key] = dq
    return cast("deque[_DequeT]", dq)


def _phase_adjust_chunk(
    args: ChunkArgs,
) -> list[tuple[NodeId, Phase]]:
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
    """
    Ajusta fase con mezcla GLOBAL+VECINAL.
    Si no se pasan fuerzas explícitas, adapta kG/kL según estado
    (disonante / transición / estable).
    Estado se decide por R (Kuramoto) y carga glífica disruptiva reciente.

    ``n_jobs`` controla el uso opcional de evaluación paralela cuando NumPy
    no está disponible; un valor ``None`` o ``<= 1`` mantiene el recorrido
    secuencial clásico.
    """
    g = cast(dict[str, Any], G.graph)
    defaults = DEFAULTS
    hist = cast(dict[str, Any], g.setdefault("history", {}))
    maxlen = int(
        g.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"])
    )
    hist_state = cast(deque[str], _ensure_hist_deque(hist, "phase_state", maxlen))
    hist_R = cast(deque[float], _ensure_hist_deque(hist, "phase_R", maxlen))
    hist_disr = cast(deque[float], _ensure_hist_deque(hist, "phase_disr", maxlen))
    # 0) Si hay fuerzas explícitas, usar y salir del modo adaptativo
    if (global_force is not None) or (local_force is not None):
        kG = float(
            global_force
            if global_force is not None
            else g.get("PHASE_K_GLOBAL", defaults["PHASE_K_GLOBAL"])
        )
        kL = float(
            local_force
            if local_force is not None
            else g.get("PHASE_K_LOCAL", defaults["PHASE_K_LOCAL"])
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
    cos_vals = [
        float(cos_map.get(n, math.cos(theta_vals[idx])))
        for idx, n in enumerate(nodes)
    ]
    sin_vals = [
        float(sin_map.get(n, math.sin(theta_vals[idx])))
        for idx, n in enumerate(nodes)
    ]

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
            if neighbors_map.get(n)
            else theta_vals[idx]
            for idx, n in enumerate(nodes)
        ]
        neighbor_arr = cast(FloatArray, np.fromiter(neighbor_means, dtype=float))
        two_pi = 2.0 * math.pi
        diff_global = np.mod(thG - theta_arr + math.pi, two_pi) - math.pi
        diff_local = np.mod(neighbor_arr - theta_arr + math.pi, two_pi) - math.pi
        updated = theta_arr + kG * diff_global + kL * diff_local
        for node, new_th in zip(nodes, updated):
            set_theta(G, node, float(new_th))
        return

    mean_cos = sum(cos_vals) / num_nodes if num_nodes else 0.0
    mean_sin = sum(sin_vals) / num_nodes if num_nodes else 0.0
    thG = math.atan2(mean_sin, mean_cos) if num_nodes else 0.0

    if jobs is None:
        for idx, (node, th) in enumerate(zip(nodes, theta_vals)):
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

    chunk_size = max(1, (num_nodes + jobs - 1) // jobs)
    chunks = [nodes[i:i + chunk_size] for i in range(0, num_nodes, chunk_size)]
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


# -------------------------
# Adaptación de νf por coherencia
# -------------------------


def _vf_adapt_chunk(
    args: tuple[list[tuple[Any, int, tuple[int, ...]]], tuple[float, ...], float]
) -> list[tuple[Any, float]]:
    """Return proposed νf updates for ``chunk`` of stable nodes."""

    chunk, vf_values, mu = args
    updates: list[tuple[Any, float]] = []
    for node, idx, neighbor_idx in chunk:
        vf = vf_values[idx]
        if neighbor_idx:
            mean = math.fsum(vf_values[j] for j in neighbor_idx) / len(neighbor_idx)
        else:
            mean = vf
        updates.append((node, vf + mu * (mean - vf)))
    return updates


def adapt_vf_by_coherence(G: TNFRGraph, n_jobs: int | None = None) -> None:
    """Adjust νf toward neighbour mean in nodes with sustained stability.

    When ``n_jobs`` is greater than one and NumPy is unavailable the stable-node
    updates are computed in worker processes. The returned proposals are then
    clamped and applied in the caller to preserve determinism.
    """

    tau = get_graph_param(G, "VF_ADAPT_TAU", int)
    mu = float(get_graph_param(G, "VF_ADAPT_MU"))
    eps_dnfr: DeltaNFR = cast(DeltaNFR, float(get_graph_param(G, "EPS_DNFR_STABLE")))
    thr_sel = get_graph_param(G, "SELECTOR_THRESHOLDS", dict)
    thr_def = get_graph_param(G, "GLYPH_THRESHOLDS", dict)
    si_hi: CoherenceMetric = cast(
        CoherenceMetric, float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
    )
    vf_min = float(get_graph_param(G, "VF_MIN"))
    vf_max = float(get_graph_param(G, "VF_MAX"))

    nodes = list(G.nodes)
    if not nodes:
        return

    neighbors_map = ensure_neighbors_map(G)
    node_count = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    jobs: int | None
    if n_jobs is None:
        jobs = None
    else:
        try:
            jobs = int(n_jobs)
        except (TypeError, ValueError):
            jobs = None
        else:
            if jobs <= 1:
                jobs = None

    np_mod = get_numpy()
    use_np = np_mod is not None

    si_values = collect_attr(G, nodes, ALIAS_SI, 0.0, np=np_mod if use_np else None)
    dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np_mod if use_np else None)
    vf_values = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np_mod if use_np else None)

    if use_np:
        np = np_mod  # type: ignore[assignment]
        assert np is not None
        si_arr = si_values.astype(float, copy=False)
        dnfr_arr = np.abs(dnfr_values.astype(float, copy=False))
        vf_arr = vf_values.astype(float, copy=False)

        prev_counts = np.fromiter(
            (int(G.nodes[node].get("stable_count", 0)) for node in nodes),
            dtype=int,
            count=node_count,
        )
        stable_mask = (si_arr >= si_hi) & (dnfr_arr <= eps_dnfr)
        new_counts = np.where(stable_mask, prev_counts + 1, 0)

        for node, count in zip(nodes, new_counts.tolist()):
            G.nodes[node]["stable_count"] = int(count)

        eligible_mask = new_counts >= tau
        if not bool(eligible_mask.any()):
            return

        max_degree = 0
        if node_count:
            degree_counts = np.fromiter(
                (len(neighbors_map.get(node, ())) for node in nodes),
                dtype=int,
                count=node_count,
            )
            if degree_counts.size:
                max_degree = int(degree_counts.max())

        if max_degree > 0:
            neighbor_indices = np.zeros((node_count, max_degree), dtype=int)
            mask = np.zeros((node_count, max_degree), dtype=bool)
            for idx, node in enumerate(nodes):
                neigh = neighbors_map.get(node, ())
                if not neigh:
                    continue
                idxs = [node_index[nbr] for nbr in neigh if nbr in node_index]
                if not idxs:
                    continue
                length = len(idxs)
                neighbor_indices[idx, :length] = idxs
                mask[idx, :length] = True
            neighbor_values = vf_arr[neighbor_indices]
            sums = (neighbor_values * mask).sum(axis=1)
            counts = mask.sum(axis=1)
            neighbor_means = np.where(counts > 0, sums / counts, vf_arr)
        else:
            neighbor_means = vf_arr

        vf_updates = vf_arr + mu * (neighbor_means - vf_arr)
        for idx in np.nonzero(eligible_mask)[0]:
            node = nodes[int(idx)]
            vf_new = clamp(float(vf_updates[int(idx)]), vf_min, vf_max)
            set_vf(G, node, vf_new)
        return

    # Pure-Python fallback
    si_list = [float(val) for val in si_values]
    dnfr_list = [abs(float(val)) for val in dnfr_values]
    vf_list = [float(val) for val in vf_values]

    prev_counts = [int(G.nodes[node].get("stable_count", 0)) for node in nodes]
    stable_flags = [
        si >= si_hi and dnfr <= eps_dnfr
        for si, dnfr in zip(si_list, dnfr_list)
    ]
    new_counts = [prev + 1 if flag else 0 for prev, flag in zip(prev_counts, stable_flags)]

    for node, count in zip(nodes, new_counts):
        G.nodes[node]["stable_count"] = int(count)

    eligible_nodes = [node for node, count in zip(nodes, new_counts) if count >= tau]
    if not eligible_nodes:
        return

    if jobs is None:
        for node in eligible_nodes:
            idx = node_index[node]
            neigh_indices = [
                node_index[nbr]
                for nbr in neighbors_map.get(node, ())
                if nbr in node_index
            ]
            if neigh_indices:
                total = math.fsum(vf_list[i] for i in neigh_indices)
                mean = total / len(neigh_indices)
            else:
                mean = vf_list[idx]
            vf_new = vf_list[idx] + mu * (mean - vf_list[idx])
            set_vf(G, node, clamp(float(vf_new), vf_min, vf_max))
        return

    work_items: list[tuple[Any, int, tuple[int, ...]]] = []
    for node in eligible_nodes:
        idx = node_index[node]
        neigh_indices = tuple(
            node_index[nbr]
            for nbr in neighbors_map.get(node, ())
            if nbr in node_index
        )
        work_items.append((node, idx, neigh_indices))

    chunk_size = max(1, math.ceil(len(work_items) / jobs))
    chunks = [
        work_items[i:i + chunk_size]
        for i in range(0, len(work_items), chunk_size)
    ]
    vf_tuple = tuple(vf_list)
    updates: dict[Any, float] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        args = ((chunk, vf_tuple, mu) for chunk in chunks)
        for chunk_updates in executor.map(_vf_adapt_chunk, args):
            for node, value in chunk_updates:
                updates[node] = float(value)

    for node in eligible_nodes:
        vf_new = updates.get(node)
        if vf_new is None:
            continue
        set_vf(G, node, clamp(float(vf_new), vf_min, vf_max))


# -------------------------
# Selector glífico por defecto
# -------------------------
def default_glyph_selector(G: TNFRGraph, n: NodeId) -> GlyphCode:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    hi, lo, dnfr_hi = itemgetter("si_hi", "si_lo", "dnfr_hi")(thr)
    # Extract thresholds in one call to reduce dict lookups inside loops.

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
def _soft_grammar_prefilter(
    G: TNFRGraph,
    n: NodeId,
    cand: GlyphCode,
    dnfr: float,
    accel: float,
) -> GlyphCode:
    """Soft grammar: avoid repetitions before the canonical one."""
    gram = get_graph_param(G, "GRAMMAR", dict)
    gwin = int(gram.get("window", 3))
    avoid = {str(item) for item in gram.get("avoid_repeats", [])}
    force_dn = float(gram.get("force_dnfr", 0.60))
    force_ac = float(gram.get("force_accel", 0.60))
    fallbacks = cast(Mapping[str, GlyphCode], gram.get("fallbacks", {}))
    nd = G.nodes[n]
    cand_key = str(cand)
    if cand_key in avoid and recent_glyph(nd, cand_key, gwin):
        if not (dnfr >= force_dn or accel >= force_ac):
            cand = fallbacks.get(cand_key, cand)
    return cand


def _selector_normalized_metrics(
    nd: Mapping[str, Any], norms: SelectorNorms
) -> SelectorMetrics:
    """Extract and normalise Si, ΔNFR and acceleration for the selector."""
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    acc_max = float(norms.get("accel_max", 1.0)) or 1.0
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max
    return Si, dnfr, accel


def _selector_base_choice(
    Si: float, dnfr: float, accel: float, thr: SelectorThresholds
) -> GlyphCode:
    """Base decision according to thresholds of Si, ΔNFR and acceleration."""
    si_hi, si_lo, dnfr_hi, acc_hi = itemgetter(
        "si_hi", "si_lo", "dnfr_hi", "accel_hi"
    )(thr)  # Reduce dict lookups inside loops.
    if Si >= si_hi:
        return "IL"
    if Si <= si_lo:
        if accel >= acc_hi:
            return "THOL"
        return "OZ" if dnfr >= dnfr_hi else "ZHIR"
    if dnfr >= dnfr_hi or accel >= acc_hi:
        return "NAV"
    return "RA"


def _configure_selector_weights(G: TNFRGraph) -> SelectorWeights:
    """Normalise and store selector weights in ``G.graph``."""
    weights = merge_and_normalize_weights(
        G, "SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel")
    )
    cast_weights = cast(SelectorWeights, weights)
    G.graph["_selector_weights"] = cast_weights
    return cast_weights


def _compute_selector_score(
    G: TNFRGraph,
    nd: Mapping[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    cand: GlyphCode,
) -> float:
    """Compute score and apply stagnation penalties."""
    W = G.graph.get("_selector_weights")
    if W is None:
        W = _configure_selector_weights(G)
    score = _calc_selector_score(Si, dnfr, accel, cast(SelectorWeights, W))
    hist_prev = nd.get("glyph_history")
    if hist_prev and hist_prev[-1] == cand:
        delta_si = get_attr(nd, ALIAS_DSI, 0.0)
        h = ensure_history(G)
        sig = h.get("sense_sigma_mag", [])
        delta_sigma = sig[-1] - sig[-2] if len(sig) >= 2 else 0.0
        if delta_si <= 0.0 and delta_sigma <= 0.0:
            score -= 0.05
    return float(score)


def _apply_score_override(
    cand: GlyphCode, score: float, dnfr: float, dnfr_lo: float
) -> GlyphCode:
    """Adjust final candidate smoothly according to the score."""
    cand_key = str(cand)
    if score >= 0.66 and cand_key in ("NAV", "RA", "ZHIR", "OZ"):
        return "IL"
    if score <= 0.33 and cand_key in ("NAV", "RA", "IL"):
        return "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def parametric_glyph_selector(G: TNFRGraph, n: NodeId) -> GlyphCode:
    """Multiobjective: combine Si, |ΔNFR|_norm and |accel|_norm with
    hysteresis.

    Base rules:
      - High Si  ⇒ IL
      - Low Si   ⇒ OZ if |ΔNFR| high; ZHIR if |ΔNFR| low;
        THOL if acceleration is high
      - Medium Si ⇒ NAV if |ΔNFR| high (or acceleration high),
        otherwise RA
    """
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    margin = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")

    norms = cast(SelectorNorms | None, G.graph.get("_sel_norms"))
    if norms is None:
        norms = _norms_para_selector(G)
    Si, dnfr, accel = _selector_normalized_metrics(nd, norms)

    cand = _selector_base_choice(Si, dnfr, accel, thr)

    hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thr, margin)
    if hist_cand is not None:
        return hist_cand

    score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)

    cand = _apply_score_override(cand, score, dnfr, thr["dnfr_lo"])

    return _soft_grammar_prefilter(G, n, cand, dnfr, accel)


def _choose_glyph(
    G: TNFRGraph,
    n: NodeId,
    selector: GlyphSelector,
    use_canon: bool,
    h_al: MutableMapping[Any, int],
    h_en: MutableMapping[Any, int],
    al_max: int,
    en_max: int,
) -> GlyphCode:
    """Select the glyph to apply on node ``n``."""
    if h_al[n] > al_max:
        return Glyph.AL
    if h_en[n] > en_max:
        return Glyph.EN
    g = selector(G, n)
    if use_canon:
        g = enforce_canonical_grammar(G, n, g)
    return g


# -------------------------
# Step / run
# -------------------------


def _run_before_callbacks(
    G: TNFRGraph,
    *,
    step_idx: int,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
) -> None:
    callback_manager.invoke_callbacks(
        G,
        CallbackEvent.BEFORE_STEP.value,
        {
            "step": step_idx,
            "dt": dt,
            "use_Si": use_Si,
            "apply_glyphs": apply_glyphs,
        },
    )


def _prepare_dnfr(
    G: TNFRGraph,
    *,
    use_Si: bool,
    job_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Compute ΔNFR and optionally Si for the current graph state."""
    compute_dnfr_cb = G.graph.get(
        "compute_delta_nfr", default_compute_delta_nfr
    )
    overrides = job_overrides or {}
    n_jobs = _resolve_jobs_override(
        overrides,
        "DNFR",
        G.graph.get("DNFR_N_JOBS"),
        allow_non_positive=False,
    )

    supports_n_jobs = False
    try:
        signature = inspect.signature(compute_dnfr_cb)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        if "n_jobs" in params:
            kind = params["n_jobs"].kind
            supports_n_jobs = kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        elif any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            supports_n_jobs = True

    if supports_n_jobs:
        compute_dnfr_cb(G, n_jobs=n_jobs)
    else:
        try:
            compute_dnfr_cb(G, n_jobs=n_jobs)
        except TypeError as exc:
            if "n_jobs" in str(exc):
                compute_dnfr_cb(G)
            else:
                raise
    G.graph.pop("_sel_norms", None)
    if use_Si:
        si_jobs = _resolve_jobs_override(
            overrides,
            "SI",
            G.graph.get("SI_N_JOBS"),
            allow_non_positive=False,
        )
        compute_Si(G, inplace=True, n_jobs=si_jobs)


def _apply_selector(G: TNFRGraph) -> GlyphSelector:
    """Configure and return the glyph selector for this step."""
    selector = cast(
        GlyphSelector,
        G.graph.get("glyph_selector", default_glyph_selector),
    )
    if selector is parametric_glyph_selector:
        _norms_para_selector(G)
        _configure_selector_weights(G)
    return selector


@dataclass(slots=True)
class _SelectorPreselection:
    kind: str
    metrics: SelectorPreselectionMetrics
    base_choices: SelectorPreselectionChoices
    thresholds: Mapping[str, float] | None = None
    margin: float | None = None


def _selector_parallel_jobs(G: TNFRGraph) -> int | None:
    """Return number of parallel jobs for glyph selection if enabled."""
    raw_jobs = G.graph.get("GLYPH_SELECTOR_N_JOBS")
    try:
        n_jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        return None
    if n_jobs is None or n_jobs <= 1:
        return None
    return n_jobs


def _selector_metrics_chunk(
    args: tuple[list[float], list[float], list[float], float, float]
) -> tuple[list[float], list[float], list[float]]:
    """Normalize selector metrics for a chunk of nodes."""

    si_values, dnfr_values, accel_values, dnfr_max, accel_max = args
    si_seq = [clamp01(float(v)) for v in si_values]
    dnfr_seq = [abs(float(v)) / dnfr_max for v in dnfr_values]
    accel_seq = [abs(float(v)) / accel_max for v in accel_values]
    return si_seq, dnfr_seq, accel_seq


def _collect_selector_metrics(
    G,
    nodes: list[Any],
    norms: dict[str, float],
    n_jobs: int | None = None,
) -> dict[Any, tuple[float, float, float]]:
    """Collect normalised Si, |ΔNFR| and |d²EPI/dt²| for ``nodes``."""
    if not nodes:
        return {}

    np_mod = get_numpy()
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    accel_max = float(norms.get("accel_max", 1.0)) or 1.0

    if np_mod is not None:
        si_seq_np = collect_attr(G, nodes, ALIAS_SI, 0.5, np=np_mod).astype(float)
        si_seq_np = np_mod.clip(si_seq_np, 0.0, 1.0)
        dnfr_seq_np = np_mod.abs(
            collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np_mod).astype(float)
        ) / dnfr_max
        accel_seq_np = np_mod.abs(
            collect_attr(G, nodes, ALIAS_D2EPI, 0.0, np=np_mod).astype(float)
        ) / accel_max

        si_seq = si_seq_np.tolist()
        dnfr_seq = dnfr_seq_np.tolist()
        accel_seq = accel_seq_np.tolist()
    else:
        si_values = collect_attr(G, nodes, ALIAS_SI, 0.5)
        dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0)
        accel_values = collect_attr(G, nodes, ALIAS_D2EPI, 0.0)

        worker_count = n_jobs if n_jobs is not None and n_jobs > 1 else None
        if worker_count is None:
            si_seq = [clamp01(float(v)) for v in si_values]
            dnfr_seq = [abs(float(v)) / dnfr_max for v in dnfr_values]
            accel_seq = [abs(float(v)) / accel_max for v in accel_values]
        else:
            chunk_size = max(1, math.ceil(len(nodes) / worker_count))
            chunk_bounds = [
                (start, min(start + chunk_size, len(nodes)))
                for start in range(0, len(nodes), chunk_size)
            ]

            si_seq: list[float] = []
            dnfr_seq: list[float] = []
            accel_seq: list[float] = []

            def _args_iter() -> Iterator[
                tuple[list[float], list[float], list[float], float, float]
            ]:
                for start, end in chunk_bounds:
                    yield (
                        si_values[start:end],
                        dnfr_values[start:end],
                        accel_values[start:end],
                        dnfr_max,
                        accel_max,
                    )

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for si_chunk, dnfr_chunk, accel_chunk in executor.map(
                    _selector_metrics_chunk, _args_iter()
                ):
                    si_seq.extend(si_chunk)
                    dnfr_seq.extend(dnfr_chunk)
                    accel_seq.extend(accel_chunk)

    return {
        node: (si_seq[idx], dnfr_seq[idx], accel_seq[idx])
        for idx, node in enumerate(nodes)
    }


def _compute_default_base_choices(
    metrics: dict[Any, tuple[float, float, float]],
    thresholds: dict[str, float],
) -> dict[Any, str]:
    """Return base glyph decisions for the default selector."""
    si_hi = float(thresholds.get("si_hi", 0.66))
    si_lo = float(thresholds.get("si_lo", 0.33))
    dnfr_hi = float(thresholds.get("dnfr_hi", 0.50))

    base: dict[Any, str] = {}
    for node, (Si, dnfr, _) in metrics.items():
        if Si >= si_hi:
            base[node] = "IL"
        elif Si <= si_lo:
            base[node] = "OZ" if dnfr > dnfr_hi else "ZHIR"
        else:
            base[node] = "NAV" if dnfr > dnfr_hi else "RA"
    return base


def _param_base_worker(
    args: tuple[dict[str, float], list[tuple[Any, tuple[float, float, float]]]]
) -> list[tuple[Any, str]]:
    """Worker used to evaluate base rules for the parametric selector."""
    thresholds, chunk = args
    return [
        (node, _selector_base_choice(Si, dnfr, accel, thresholds))
        for node, (Si, dnfr, accel) in chunk
    ]


def _compute_param_base_choices(
    metrics: dict[Any, tuple[float, float, float]],
    thresholds: dict[str, float],
    n_jobs: int | None,
) -> dict[Any, str]:
    """Evaluate base rules for the parametric selector, optionally in parallel."""
    if not metrics:
        return {}

    items = list(metrics.items())
    if n_jobs is None:
        return {
            node: _selector_base_choice(Si, dnfr, accel, thresholds)
            for node, (Si, dnfr, accel) in items
        }

    chunk_size = max(1, math.ceil(len(items) / n_jobs))
    base: dict[Any, str] = {}
    args = (
        (thresholds, items[idx:idx + chunk_size])
        for idx in range(0, len(items), chunk_size)
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for result in executor.map(_param_base_worker, args):
            for node, cand in result:
                base[node] = cand
    return base


def _prepare_selector_preselection(
    G: TNFRGraph,
    selector: GlyphSelector,
    nodes: Sequence[NodeId],
) -> _SelectorPreselection | None:
    """Return preselection data for recognised selectors."""
    if selector is default_glyph_selector:
        norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
        thresholds = _selector_thresholds(G)
        n_jobs = _selector_parallel_jobs(G)
        metrics = _collect_selector_metrics(G, nodes, norms, n_jobs=n_jobs)
        base_choices = _compute_default_base_choices(metrics, thresholds)
        return _SelectorPreselection(
            "default", metrics, base_choices, thresholds=thresholds
        )
    if selector is parametric_glyph_selector:
        norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
        thresholds = _selector_thresholds(G)
        n_jobs = _selector_parallel_jobs(G)
        metrics = _collect_selector_metrics(G, nodes, norms, n_jobs=n_jobs)
        margin = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")
        base_choices = _compute_param_base_choices(
            metrics, thresholds, n_jobs
        )
        return _SelectorPreselection(
            "param", metrics, base_choices, thresholds=thresholds, margin=margin
        )
    return None


def _resolve_preselected_glyph(
    G: TNFRGraph,
    n: NodeId,
    selector: GlyphSelector,
    preselection: _SelectorPreselection | None,
) -> GlyphCode:
    """Return glyph for node ``n`` using ``preselection`` when available."""
    if preselection is None:
        return selector(G, n)

    metrics = preselection.metrics.get(n)
    if metrics is None:
        return selector(G, n)

    if preselection.kind == "default":
        cand = preselection.base_choices.get(n)
        return cand if cand is not None else selector(G, n)

    if preselection.kind == "param":
        Si, dnfr, accel = metrics
        thresholds = preselection.thresholds or _selector_thresholds(G)
        margin = preselection.margin
        if margin is None:
            margin = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")

        cand = preselection.base_choices.get(n)
        if cand is None:
            cand = _selector_base_choice(Si, dnfr, accel, thresholds)

        nd = G.nodes[n]
        hist_cand = _apply_selector_hysteresis(
            nd, Si, dnfr, accel, thresholds, margin
        )
        if hist_cand is not None:
            return hist_cand

        score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)
        cand = _apply_score_override(cand, score, dnfr, thresholds["dnfr_lo"])
        return _soft_grammar_prefilter(G, n, cand, dnfr, accel)

    return selector(G, n)


def _glyph_proposal_worker(
    args: tuple[
        list[NodeId],
        TNFRGraph,
        GlyphSelector,
        _SelectorPreselection | None,
    ]
) -> list[tuple[NodeId, GlyphCode]]:
    """Return glyph proposals for ``args[0]`` using ``_resolve_preselected_glyph``."""

    nodes, G, selector, preselection = args
    return [
        (n, _resolve_preselected_glyph(G, n, selector, preselection))
        for n in nodes
    ]


def _apply_glyphs(G: TNFRGraph, selector: GlyphSelector, hist: HistoryState) -> None:
    """Apply glyphs to nodes using ``selector`` and update history."""
    window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
    use_canon = bool(
        get_graph_param(G, "GRAMMAR_CANON", dict).get("enabled", False)
    )
    al_max = get_graph_param(G, "AL_MAX_LAG", int)
    en_max = get_graph_param(G, "EN_MAX_LAG", int)

    nodes_data = list(G.nodes(data=True))
    nodes = [n for n, _ in nodes_data]
    preselection = _prepare_selector_preselection(G, selector, nodes)

    h_al = hist.setdefault("since_AL", {})
    h_en = hist.setdefault("since_EN", {})
    forced: dict[Any, str | Glyph] = {}
    to_select: list[Any] = []

    for n, _ in nodes_data:
        h_al[n] = int(h_al.get(n, 0)) + 1
        h_en[n] = int(h_en.get(n, 0)) + 1

        if h_al[n] > al_max:
            forced[n] = Glyph.AL
        elif h_en[n] > en_max:
            forced[n] = Glyph.EN
        else:
            to_select.append(n)

    decisions: dict[Any, str | Glyph] = dict(forced)
    if to_select:
        n_jobs = _selector_parallel_jobs(G)
        if n_jobs is None:
            for n in to_select:
                decisions[n] = _resolve_preselected_glyph(
                    G, n, selector, preselection
                )
        else:
            chunk_size = max(1, math.ceil(len(to_select) / n_jobs))
            chunks = [
                to_select[idx:idx + chunk_size]
                for idx in range(0, len(to_select), chunk_size)
            ]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                args_iter = (
                    (chunk, G, selector, preselection) for chunk in chunks
                )
                for results in executor.map(_glyph_proposal_worker, args_iter):
                    for node, glyph in results:
                        decisions[node] = glyph

    for n, _ in nodes_data:
        g = decisions.get(n)
        if g is None:
            continue

        if use_canon:
            g = enforce_canonical_grammar(G, n, g)

        apply_glyph(G, n, g, window=window)
        if use_canon:
            on_applied_glyph(G, n, g)

        if g == Glyph.AL:
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif g == Glyph.EN:
            h_en[n] = 0


def _update_nodes(
    G: TNFRGraph,
    *,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
    step_idx: int,
    hist: HistoryState,
    job_overrides: Mapping[str, Any] | None = None,
) -> None:
    _update_node_sample(G, step=step_idx)
    overrides = job_overrides or {}
    _prepare_dnfr(G, use_Si=use_Si, job_overrides=overrides)
    selector = _apply_selector(G)
    if apply_glyphs:
        _apply_glyphs(G, selector, hist)
    _dt = get_graph_param(G, "DT") if dt is None else float(dt)
    method = get_graph_param(G, "INTEGRATOR_METHOD", str)
    n_jobs = _resolve_jobs_override(
        overrides,
        "INTEGRATOR",
        G.graph.get("INTEGRATOR_N_JOBS"),
        allow_non_positive=True,
    )
    update_epi_via_nodal_equation(G, dt=_dt, method=method, n_jobs=n_jobs)
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(nd, G, n)
    phase_jobs = _resolve_jobs_override(
        overrides,
        "PHASE",
        G.graph.get("PHASE_N_JOBS"),
        allow_non_positive=True,
    )
    coordinate_global_local_phase(G, None, None, n_jobs=phase_jobs)
    vf_jobs = _resolve_jobs_override(
        overrides,
        "VF_ADAPT",
        G.graph.get("VF_ADAPT_N_JOBS"),
        allow_non_positive=False,
    )
    adapt_vf_by_coherence(G, n_jobs=vf_jobs)


def _update_epi_hist(G) -> None:
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append(
        {n: get_attr(nd, ALIAS_EPI, 0.0) for n, nd in G.nodes(data=True)}
    )


def _maybe_remesh(G) -> None:
    apply_remesh_if_globally_stable(G)


def _run_validators(G) -> None:
    from ..utils import run_validators

    run_validators(G)


def _run_after_callbacks(G, *, step_idx: int) -> None:
    h = ensure_history(G)
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
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value, ctx)


def step(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    """Advance the dynamic state of ``G`` one step using optional job overrides."""
    job_overrides = _normalize_job_overrides(n_jobs)
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
        job_overrides=job_overrides,
    )
    _update_epi_hist(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)


def run(
    G: TNFRGraph,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    steps_int = int(steps)
    if steps_int < 0:
        raise ValueError("'steps' must be non-negative")
    stop_cfg = get_graph_param(G, "STOP_EARLY", dict)
    stop_enabled = False
    if stop_cfg and stop_cfg.get("enabled", False):
        w = int(stop_cfg.get("window", 25))
        frac = float(stop_cfg.get("fraction", 0.90))
        stop_enabled = True
    job_overrides = _normalize_job_overrides(n_jobs)
    for _ in range(steps_int):
        step(
            G,
            dt=dt,
            use_Si=use_Si,
            apply_glyphs=apply_glyphs,
            n_jobs=job_overrides,
        )
        # Early-stop opcional
        if stop_enabled:
            history = ensure_history(G)
            series = history.get("stable_frac", [])
            if not isinstance(series, list):
                series = list(series)
            if len(series) >= w and all(v >= frac for v in series[-w:]):
                break
