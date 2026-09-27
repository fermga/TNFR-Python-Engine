"""Phase coordination helpers for TNFR dynamics."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from .._exact_time import finite_represented_real
from ..alias import get_theta_attr, set_theta
from ..constants import (
    DEFAULTS,
    METRIC_DEFAULTS,
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    normalise_state_token,
)
from ..errors import TNFRValueError
from ..glyph_history import append_metric
from ..mathematics.phasor_resultant import (
    RepresentedPhasorResultant,
    reduce_phasor_components,
)
from ..mathematics.unified_numerical import compute_phase_difference, np
from ..metrics.common import ensure_neighbors_map
from ..metrics.trig import neighbor_phase_mean_list
from ..metrics.trig_cache import get_trig_cache
from ..observers import DEFAULT_GLYPH_LOAD_SPAN, glyph_load, kuramoto_order
from ..types import FloatArray, NodeId, Phase, TNFRGraph
from ..utils import angle_diff, resolve_chunk_size

_DequeT = TypeVar("_DequeT")

ChunkArgs = tuple[
    Sequence[NodeId],
    Mapping[NodeId, Phase],
    Mapping[NodeId, float],
    Mapping[NodeId, float],
    Mapping[NodeId, Sequence[NodeId]],
    float | None,
    float,
    float,
]

__all__ = (
    "coordinate_global_local_phase",
    "GlobalPhaseCoordinationEvidence",
    "UndefinedGlobalPhaseError",
)


class UndefinedGlobalPhaseError(TNFRValueError):
    """An active global coupling term has exactly zero represented resultant."""


@dataclass(frozen=True)
class GlobalPhaseCoordinationEvidence:
    """Public readout of one opt-in call, not a sealed execution certificate.

    Resultant arithmetic concerns the recorded cached components, not certified
    transcendental values. Local means, adaptive policy and normalization keep
    their existing semantics; global reduction alone is versioned here.
    Freezing is shallow: graph node identifiers are retained by reference.
    """

    version: str
    status: str
    nodes: tuple[NodeId, ...]
    neighbor_order: tuple[tuple[NodeId, tuple[NodeId, ...]], ...]
    primitive_phases: tuple[float, ...]
    resultant: RepresentedPhasorResultant | None
    requested_global_force: float | None
    requested_local_force: float | None
    effective_global_force: float
    effective_local_force: float
    gain_mode: str
    global_target: float | None
    global_term_active: bool
    local_targets: tuple[float, ...]
    raw_proposals: tuple[float, ...]
    realized_phases: tuple[float, ...]
    execution_path: str
    scope: str


def _ensure_hist_deque(
    hist: MutableMapping[str, Any], key: str, maxlen: int
) -> deque[_DequeT]:
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

    supplied = g.get("PHASE_ADAPT", {})
    if not isinstance(supplied, Mapping):
        raise TNFRValueError("PHASE_ADAPT must be a mapping")
    cfg = {**DEFAULTS["PHASE_ADAPT"], **supplied}
    if not isinstance(cfg["enabled"], bool):
        raise TNFRValueError("PHASE_ADAPT['enabled'] must be a boolean")
    if cfg["enabled"]:
        cfg.update(_finite_adaptive_config(cfg))
    kG = _finite_gain(g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"]))
    kL = _finite_gain(g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"]))
    return cfg, kG, kL


def _finite_gain(value: Any) -> float:
    return finite_represented_real(value, "phase coupling/configuration value")[0]


def _finite_adaptive_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate only numeric parameters consumed by the existing active policy."""
    resolved = {
        key: _finite_gain(cfg[key])
        for key in DEFAULTS["PHASE_ADAPT"]
        if key != "enabled"
    }
    for key in ("R_hi", "R_lo", "disr_hi", "disr_lo", "up", "down"):
        if not 0.0 <= resolved[key] <= 1.0:
            raise TNFRValueError(f"PHASE_ADAPT[{key!r}] must be in [0, 1]")
    for lower, upper in (
        ("R_lo", "R_hi"),
        ("disr_lo", "disr_hi"),
        ("kG_min", "kG_max"),
        ("kL_min", "kL_max"),
    ):
        if not 0.0 <= resolved[lower] <= resolved[upper]:
            raise TNFRValueError(f"PHASE_ADAPT requires 0 <= {lower} <= {upper}")
    return resolved


def _compute_state(G: TNFRGraph, cfg: Mapping[str, Any]) -> tuple[str, float, float]:
    """Return the configured controller state and supporting metrics."""

    R = kuramoto_order(G)
    dist = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
    disr = float(dist.get("_disruptors", 0.0)) if dist else 0.0

    R_hi = cfg["R_hi"]
    R_lo = cfg["R_lo"]
    disr_hi = cfg["disr_hi"]
    disr_lo = cfg["disr_lo"]
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

    kG_min = cfg["kG_min"]
    kG_max = cfg["kG_max"]
    kL_min = cfg["kL_min"]
    kL_max = cfg["kL_max"]

    state = normalise_state_token(state)

    def midpoint(lower: float, upper: float) -> float:
        candidate = 0.5 * (lower + upper)
        # Nonnegative finite bounds have a finite midpoint even if their sum
        # overflows. Preserve ordinary arithmetic away from that boundary.
        return candidate if math.isfinite(candidate) else lower / 2 + upper / 2

    if state == STATE_DISSONANT:
        kG_t = kG_max
        kL_t = midpoint(kL_min, kL_max)
    elif state == STATE_STABLE:
        kG_t = kG_min
        kL_t = kL_min
    else:
        kG_t = midpoint(kG_min, kG_max)
        kL_t = midpoint(kL_min, kL_max)

    up = cfg["up"]
    down = cfg["down"]

    def _step(curr: float, target: float, mn: float, mx: float) -> float:
        gain = up if target > curr else down
        nxt = curr + gain * (target - curr)
        if not math.isfinite(nxt):
            # Opposite-sign finite endpoints can overflow their difference.
            # The admitted gain is in [0, 1], so this convex form stays finite.
            nxt = (1.0 - gain) * curr + gain * target
        return max(mn, min(mx, nxt))

    return _step(kG, kG_t, kG_min, kG_max), _step(kL, kL_t, kL_min, kL_max)


def _phase_adjust_chunk(args: ChunkArgs) -> list[tuple[NodeId, Phase, Phase]]:
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
    updates: list[tuple[NodeId, Phase, Phase]] = []
    for node in nodes:
        th = float(theta_map.get(node, 0.0))
        neigh = neighbors_map.get(node, ())
        if neigh:
            thL = neighbor_phase_mean_list(
                neigh,
                cos_map,
                sin_map,
                fallback=th,
            )
        else:
            thL = th
        if thG is None and kG != 0.0:
            raise UndefinedGlobalPhaseError(
                "active global phase coupling requires a defined target"
            )
        dG = 0.0 if thG is None else angle_diff(thG, th)
        dL = angle_diff(thL, th)
        updates.append((node, cast(Phase, th + kG * dG + kL * dL), cast(Phase, thL)))
    return updates


def coordinate_global_local_phase(
    G: TNFRGraph,
    global_force: float | None = None,
    local_force: float | None = None,
    *,
    n_jobs: int | None = None,
    global_reduction: str = "legacy",
) -> GlobalPhaseCoordinationEvidence | None:
    """Coordinate phase using a blend of global and neighbour coupling.

    This operator harmonises a TNFR graph by iteratively nudging each node's
    phase toward the global Kuramoto mean while respecting the local
    neighbourhood attractor. The global (``kG``) and local (``kL``) coupling
    gains reshape phase coherence by modulating how strongly nodes follow the
    network-wide synchrony versus immediate neighbours. When explicit coupling
    overrides are not supplied, the gains adapt based on phase order and
    recent disruptive-glyph load. Adaptive updates
    mutate the ``history`` buffers for phase state, order parameter, disruptor
    load, and the stored coupling gains.

    This is a configured relaxation map per invocation, with no time-step
    argument or capacity-driven free phase advance. It is distinct from the
    model in ``phase_evolution.propose_u3_gated_phase_step``; the nodal EPI
    equation alone does not select either phase law. See
    ``theory/FORCED_SUPPORT_BALANCE.md`` section 23.

    Both reductions validate authoritative raw phases before history/cache
    work and all represented proposals before the first phase write. Only the
    exact reduction additionally restores caches, histories and arbitrary
    graph-owned setter side effects after a late failure. Neighbors are unique
    outgoing support neighbors, irrespective of conductance or parallel edges.

    Parameters
    ----------
    G : TNFRGraph
        Graph whose nodes expose TNFR phase attributes and ΔNFR telemetry. The
        graph's ``history`` mapping is updated in-place when adaptive gain
        smoothing is active.
    global_force : float, optional
        Override for the global coupling gain ``kG``. When provided, adaptive
        state/order/disruptor estimation and appends are skipped. History
        buffers are still initialized and effective gain histories appended.
    local_force : float, optional
        Override for the local coupling gain ``kL``. Analogous to
        ``global_force``, the adaptive pathway is bypassed when supplied.
    n_jobs : int, optional
        Maximum number of worker processes for distributing local updates.
        Values of ``None`` or ``<=1`` perform updates sequentially. NumPy
        availability forces sequential execution because vectorised updates are
        faster than multiprocess handoffs. The opt-in version propagates failed
        integer materialization and rolls back; legacy keeps its fallback.
    global_reduction : {"legacy", "exact_components_v1"}, optional
        Legacy keeps the existing global mean and return value. The opt-in
        version reduces the materialized global trig components exactly,
        retaining existing adaptive gains, local means and normalization.
        Exact zero with a nonzero effective global gain raises
        ``UndefinedGlobalPhaseError``; a disabled global term uses zero
        displacement without selecting a target.

    Returns
    -------
    None or GlobalPhaseCoordinationEvidence
        Legacy returns None. The opt-in version returns a frozen public
        readout (not a sealed certificate) and restores graph-owned state on
        any failure, including late commit/evidence failure. The transaction
        is captured before input conversion or history/cache mutation. External
        I/O and side effects outside graph ownership are not rolled back.

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

    if type(global_reduction) is not str or global_reduction not in (
        "legacy",
        "exact_components_v1",
    ):
        raise TNFRValueError(
            "global_reduction must be 'legacy' or 'exact_components_v1'"
        )
    if global_reduction == "legacy":
        return _coordinate_global_local_phase(
            G, global_force, local_force, n_jobs=n_jobs, exact=False
        )
    # Network stages import dynamics: resolve the transaction owner lazily.
    from ..operators.network_stage import GraphTransactionSnapshot

    transaction = GraphTransactionSnapshot(G)
    try:
        return _coordinate_global_local_phase(
            G, global_force, local_force, n_jobs=n_jobs, exact=True
        )
    except BaseException as failure:
        transaction.restore_after_failure(G, failure)
        raise


def _coordinate_global_local_phase(
    G: TNFRGraph,
    global_force: float | None,
    local_force: float | None,
    *,
    n_jobs: int | None,
    exact: bool,
) -> GlobalPhaseCoordinationEvidence | None:
    """Shared adaptive/local algorithm for legacy and opt-in global reduction."""
    g = cast(dict[str, Any], G.graph)
    nodes: list[NodeId] = [cast(NodeId, node) for node in G.nodes()]
    # The trig cache is a permissive observation reader. Admit its primitive
    # source first so a bad primary cannot become a fallback, text-derived
    # number or a zero before execution checks it.
    theta_vals = [
        get_theta_attr(
            G.nodes[node],
            0.0,
            strict=True,
            conv=lambda value: finite_represented_real(value, "primitive phase")[0],
        )
        for node in nodes
    ]
    # Resolve the consumed policy before allocating histories or writing gains.
    # A partial mapping means a default overlay, never a legacy policy switch.
    fixed_override = global_force is not None or local_force is not None
    cfg: Mapping[str, Any] = {}
    if not fixed_override:
        cfg, kG, kL = _read_adaptive_params(g)
    else:
        kG = _finite_gain(
            global_force
            if global_force is not None
            else g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"])
        )
        kL = _finite_gain(
            local_force
            if local_force is not None
            else g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"])
        )
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

    requested_global = requested_local = None
    gain_mode = "configured_fixed"
    if fixed_override:
        gain_mode = "fixed_override"
        requested_global = kG if global_force is not None else None
        requested_local = kL if local_force is not None else None
    else:
        if cfg["enabled"]:
            gain_mode = "adaptive"
            state, R, disr = _compute_state(G, cfg)
            kG, kL = _smooth_adjust_k(kG, kL, state, cfg)

            hist_state.append(state)
            hist_R.append(float(R))
            hist_disr.append(float(disr))

    if exact:
        kG, kL = _finite_gain(kG), _finite_gain(kL)

    g["PHASE_K_GLOBAL"] = kG
    g["PHASE_K_LOCAL"] = kL
    append_metric(hist, "phase_kG", float(kG))
    append_metric(hist, "phase_kL", float(kL))

    jobs: int | None
    try:
        jobs = None if n_jobs is None else int(n_jobs)
    except (TypeError, ValueError):
        if exact:
            raise
        jobs = None
    if jobs is not None and jobs <= 1:
        jobs = None

    if np is not None:
        jobs = None

    num_nodes = len(nodes)

    def evidence(
        path: str,
        *,
        phases: Sequence[float] = (),
        resultant: RepresentedPhasorResultant | None = None,
        neighbors: Mapping[NodeId, Sequence[NodeId]] | None = None,
        target: float | None = None,
        local_targets: Sequence[float] = (),
        proposals: Sequence[float] = (),
    ) -> GlobalPhaseCoordinationEvidence | None:
        if not exact:
            return None
        realized = tuple(
            finite_represented_real(get_theta_attr(G.nodes[n]), "realized phase")[0]
            for n in nodes
        )
        return GlobalPhaseCoordinationEvidence(
            version="exact_components_v1",
            status="applied" if nodes else "empty_graph",
            nodes=tuple(nodes),
            neighbor_order=tuple(
                (n, tuple((neighbors or {}).get(n, ()))) for n in nodes
            ),
            primitive_phases=tuple(phases),
            resultant=resultant,
            requested_global_force=requested_global,
            requested_local_force=requested_local,
            effective_global_force=kG,
            effective_local_force=kL,
            gain_mode=gain_mode,
            global_target=target,
            global_term_active=bool(nodes) and kG != 0.0,
            local_targets=tuple(local_targets),
            raw_proposals=tuple(proposals),
            realized_phases=realized,
            execution_path=path,
            scope="Exact global reduction of consumed represented components only. Local targets, adaptive gains and "
            "phase normalization keep existing semantics. Public readout, not a sealed execution or full-runtime "
            "invariance certificate; graph-owned rollback excludes external side effects.",
        )

    if not num_nodes:
        return evidence("empty_graph")

    trig = get_trig_cache(G)
    theta_map = cast(dict[NodeId, Phase], dict(zip(nodes, theta_vals)))
    cos_map = cast(dict[NodeId, float], trig.cos)
    sin_map = cast(dict[NodeId, float], trig.sin)

    neighbors_proxy = ensure_neighbors_map(G)
    neighbors_map: dict[NodeId, tuple[NodeId, ...]] = {}
    for n in nodes:
        try:
            neighbors_map[n] = tuple(cast(Sequence[NodeId], neighbors_proxy[n]))
        except KeyError:
            neighbors_map[n] = ()

    def component(value: Any) -> float:
        return finite_represented_real(value, "materialized trig component")[0]

    def commit_phases(proposals: Sequence[float]) -> None:
        # Admit every proposal first, including the legacy and worker routes.
        values = tuple(
            finite_represented_real(value, "raw phase proposal")[0]
            for value in proposals
        )
        if len(values) != num_nodes:
            raise TNFRValueError("phase proposal set must cover every node")
        for node, value in zip(nodes, values):
            set_theta(G, node, value)

    cos_vals = [
        component(cos_map[n] if n in cos_map else math.cos(theta_vals[idx]))
        for idx, n in enumerate(nodes)
    ]
    sin_vals = [
        component(sin_map[n] if n in sin_map else math.sin(theta_vals[idx]))
        for idx, n in enumerate(nodes)
    ]
    cos_map = dict(zip(nodes, cos_vals))
    sin_map = dict(zip(nodes, sin_vals))
    resultant = reduce_phasor_components(zip(cos_vals, sin_vals)) if exact else None
    thG: float | None = None
    if resultant is not None and kG != 0.0:
        if resultant.joint_zero:
            raise UndefinedGlobalPhaseError(
                "active global phase coupling has exactly zero represented resultant",
                context={
                    "global_reduction": "exact_components_v1",
                    "effective_global_force": kG,
                },
            )
        thG = resultant.angle

    if np is not None:
        theta_arr = cast(FloatArray, np.fromiter(theta_vals, dtype=float))
        cos_arr = cast(FloatArray, np.fromiter(cos_vals, dtype=float))
        sin_arr = cast(FloatArray, np.fromiter(sin_vals, dtype=float))
        if not exact and cos_arr.size:
            mean_cos = float(np.mean(cos_arr))
            mean_sin = float(np.mean(sin_arr))
            thG = float(np.arctan2(mean_sin, mean_cos))
        elif not exact:
            thG = 0.0
        neighbor_means = [
            neighbor_phase_mean_list(
                neighbors_map.get(n, ()),
                cos_map,
                sin_map,
                fallback=theta_vals[idx],
            )
            for idx, n in enumerate(nodes)
        ]
        neighbor_arr = cast(FloatArray, np.fromiter(neighbor_means, dtype=float))
        # Use the scalar/vector circular owner, including signed antipodal ties.
        global_difference = (
            np.zeros_like(theta_arr)
            if thG is None
            else compute_phase_difference(thG, theta_arr)
        )
        local_difference = compute_phase_difference(neighbor_arr, theta_arr)
        theta_updates = theta_arr + kG * global_difference + kL * local_difference
        commit_phases(tuple(map(float, theta_updates)))
        return evidence(
            "numpy",
            phases=theta_vals,
            resultant=resultant,
            neighbors=neighbors_map,
            target=thG,
            local_targets=neighbor_means,
            proposals=tuple(map(float, theta_updates)) if exact else (),
        )

    if not exact:
        mean_cos = math.fsum(cos_vals) / num_nodes
        mean_sin = math.fsum(sin_vals) / num_nodes
        thG = math.atan2(mean_sin, mean_cos)

    if jobs is None:
        updates = _phase_adjust_chunk(
            (nodes, theta_map, cos_map, sin_map, neighbors_map, thG, kG, kL)
        )
        proposals = tuple(float(value) for _, value, _ in updates)
        targets = tuple(float(target) for _, _, target in updates)
        commit_phases(proposals)
        return evidence(
            "scalar_sequential",
            phases=theta_vals,
            resultant=resultant,
            neighbors=neighbors_map,
            target=thG,
            local_targets=targets,
            proposals=proposals,
        )

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
    local_results: dict[NodeId, Phase] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for res in executor.map(_phase_adjust_chunk, args):
            for node, value, target in res:
                results[node] = value
                local_results[node] = target
    if set(results) != set(nodes) or set(local_results) != set(nodes):
        raise TNFRValueError("phase worker proposals must cover every node")
    proposals = tuple(float(results[node]) for node in nodes)
    commit_phases(proposals)
    return evidence(
        "scalar_multiprocessing",
        phases=theta_vals,
        resultant=resultant,
        neighbors=neighbors_map,
        target=thG,
        local_targets=tuple(float(local_results[n]) for n in nodes) if exact else (),
        proposals=proposals,
    )
