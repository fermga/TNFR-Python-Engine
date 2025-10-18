from __future__ import annotations

import math
from collections import deque
from operator import itemgetter
from typing import Any

# Importar compute_Si y apply_glyph a nivel de módulo evita el coste de
# realizar la importación en cada paso de la dinámica. Como los módulos de
# origen no dependen de ``dynamics``, no se introducen ciclos.
from ..operators import apply_remesh_if_globally_stable, apply_glyph
from ..validation.grammar import enforce_canonical_grammar, on_applied_glyph
from ..types import Glyph
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
from ..metrics.trig import neighbor_phase_mean
from ..alias import (
    get_attr,
    set_vf,
    set_attr,
    set_theta,
    multi_recompute_abs_max,
)
from ..metrics.sense_index import compute_Si
from ..metrics.common import compute_dnfr_accel_max, merge_and_normalize_weights
from ..metrics.trig_cache import compute_theta_trig
from ..callback_utils import CallbackEvent, callback_manager
from ..glyph_history import recent_glyph, ensure_history, append_metric
from ..selector import (
    _selector_thresholds,
    _norms_para_selector,
    _calc_selector_score,
    _apply_selector_hysteresis,
)
from ..config.operator_names import TRANSICION

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
ALIAS_THETA = get_aliases("THETA")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_EPI = get_aliases("EPI")
ALIAS_SI = get_aliases("SI")
ALIAS_D2EPI = get_aliases("D2EPI")
ALIAS_DSI = get_aliases("DSI")

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


def _log_clamp(hist, node, attr, value, lo, hi):
    if value < lo or value > hi:
        hist.append({"node": node, "attr": attr, "value": float(value)})


def apply_canonical_clamps(nd: dict[str, Any], G=None, node=None) -> None:
    g = G.graph if G is not None else DEFAULTS
    eps_min = float(g.get("EPI_MIN", DEFAULTS["EPI_MIN"]))
    eps_max = float(g.get("EPI_MAX", DEFAULTS["EPI_MAX"]))
    vf_min = float(g.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(g.get("VF_MAX", DEFAULTS["VF_MAX"]))
    theta_wrap = bool(g.get("THETA_WRAP", DEFAULTS["THETA_WRAP"]))

    epi = get_attr(nd, ALIAS_EPI, 0.0)
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th = get_attr(nd, ALIAS_THETA, 0.0)

    strict = bool(
        g.get("VALIDATORS_STRICT", DEFAULTS.get("VALIDATORS_STRICT", False))
    )
    if strict and G is not None:
        hist = g.setdefault("history", {}).setdefault("clamp_alerts", [])
        _log_clamp(hist, node, "EPI", epi, eps_min, eps_max)
        _log_clamp(hist, node, "VF", vf, vf_min, vf_max)

    set_attr(nd, ALIAS_EPI, clamp(epi, eps_min, eps_max))

    vf_val = clamp(vf, vf_min, vf_max)
    if G is not None and node is not None:
        set_vf(G, node, vf_val, update_max=False)
    else:
        set_attr(nd, ALIAS_VF, vf_val)

    if theta_wrap:
        new_th = (th + math.pi) % (2 * math.pi) - math.pi
        if G is not None and node is not None:
            set_theta(G, node, new_th)
        else:
            set_attr(nd, ALIAS_THETA, new_th)


def validate_canon(G) -> None:
    """Apply canonical clamps to all nodes of ``G``.

    Wrap phase and constrain ``EPI`` and ``νf`` to the ranges in ``G.graph``.
    If ``VALIDATORS_STRICT`` is active, alerts are logged in ``history``.
    """
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(nd, G, n)
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


def _compute_state(G, cfg: dict[str, Any]) -> tuple[str, float, float]:
    """Return current state (stable/dissonant/transition) and metrics."""
    R = kuramoto_order(G)
    dist = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
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
        state = TRANSICION
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


def _ensure_hist_deque(hist: dict[str, Any], key: str, maxlen: int) -> deque:
    """Ensure history entry ``key`` is a deque with ``maxlen``."""
    dq = hist.setdefault(key, deque(maxlen=maxlen))
    if not isinstance(dq, deque):
        dq = deque(dq, maxlen=maxlen)
        hist[key] = dq
    return dq


def coordinate_global_local_phase(
    G, global_force: float | None = None, local_force: float | None = None
) -> None:
    """
    Ajusta fase con mezcla GLOBAL+VECINAL.
    Si no se pasan fuerzas explícitas, adapta kG/kL según estado
    (disonante / transición / estable).
    Estado se decide por R (Kuramoto) y carga glífica disruptiva reciente.
    """
    g = G.graph
    defaults = DEFAULTS
    hist = g.setdefault("history", {})
    maxlen = int(
        g.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"])
    )
    hist_state = _ensure_hist_deque(hist, "phase_state", maxlen)
    hist_R = _ensure_hist_deque(hist, "phase_R", maxlen)
    hist_disr = _ensure_hist_deque(hist, "phase_disr", maxlen)
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

    # 6) Fase GLOBAL (centroide) para empuje
    trig = compute_theta_trig(G.nodes(data=True))
    num_nodes = G.number_of_nodes()
    if num_nodes:
        mean_cos = sum(trig.cos.values()) / num_nodes
        mean_sin = sum(trig.sin.values()) / num_nodes
        thG = math.atan2(mean_sin, mean_cos)
    else:
        thG = 0.0

    # 7) Aplicar corrección global+vecinal
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thL = neighbor_phase_mean(G, n)
        dG = angle_diff(thG, th)
        dL = angle_diff(thL, th)
        set_theta(G, n, th + kG * dG + kL * dL)


# -------------------------
# Adaptación de νf por coherencia
# -------------------------


def adapt_vf_by_coherence(G) -> None:
    """Adjust νf toward neighbour mean in nodes with sustained stability."""
    tau = get_graph_param(G, "VF_ADAPT_TAU", int)
    mu = get_graph_param(G, "VF_ADAPT_MU")
    eps_dnfr = get_graph_param(G, "EPS_DNFR_STABLE")
    thr_sel = get_graph_param(G, "SELECTOR_THRESHOLDS", dict)
    thr_def = get_graph_param(G, "GLYPH_THRESHOLDS", dict)
    si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
    vf_min = get_graph_param(G, "VF_MIN")
    vf_max = get_graph_param(G, "VF_MAX")

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
            neigh = list(G.neighbors(n))
            if neigh:
                total = 0.0
                for v in neigh:
                    total += float(get_attr(G.nodes[v], ALIAS_VF, vf))
                vf_bar = total / len(neigh)
            else:
                vf_bar = float(vf)
            updates[n] = vf + mu * (vf_bar - vf)

    for n, vf_new in updates.items():
        set_vf(G, n, clamp(vf_new, vf_min, vf_max))


# -------------------------
# Selector glífico por defecto
# -------------------------
def default_glyph_selector(G, n) -> str:
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
def _soft_grammar_prefilter(G, n, cand, dnfr, accel):
    """Soft grammar: avoid repetitions before the canonical one."""
    gram = get_graph_param(G, "GRAMMAR", dict)
    gwin = int(gram.get("window", 3))
    avoid = set(gram.get("avoid_repeats", []))
    force_dn = float(gram.get("force_dnfr", 0.60))
    force_ac = float(gram.get("force_accel", 0.60))
    fallbacks = gram.get("fallbacks", {})
    nd = G.nodes[n]
    if cand in avoid and recent_glyph(nd, cand, gwin):
        if not (dnfr >= force_dn or accel >= force_ac):
            cand = fallbacks.get(cand, cand)
    return cand


def _selector_normalized_metrics(nd, norms):
    """Extract and normalise Si, ΔNFR and acceleration for the selector."""
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    acc_max = float(norms.get("accel_max", 1.0)) or 1.0
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max
    return Si, dnfr, accel


def _selector_base_choice(Si, dnfr, accel, thr):
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


def _configure_selector_weights(G) -> dict:
    """Normalise and store selector weights in ``G.graph``."""
    weights = merge_and_normalize_weights(
        G, "SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel")
    )
    G.graph["_selector_weights"] = weights
    return weights


def _compute_selector_score(G, nd, Si, dnfr, accel, cand):
    """Compute score and apply stagnation penalties."""
    W = G.graph.get("_selector_weights")
    if W is None:
        W = _configure_selector_weights(G)
    score = _calc_selector_score(Si, dnfr, accel, W)
    hist_prev = nd.get("glyph_history")
    if hist_prev and hist_prev[-1] == cand:
        delta_si = get_attr(nd, ALIAS_DSI, 0.0)
        h = ensure_history(G)
        sig = h.get("sense_sigma_mag", [])
        delta_sigma = sig[-1] - sig[-2] if len(sig) >= 2 else 0.0
        if delta_si <= 0.0 and delta_sigma <= 0.0:
            score -= 0.05
    return score


def _apply_score_override(cand, score, dnfr, dnfr_lo):
    """Adjust final candidate smoothly according to the score."""
    if score >= 0.66 and cand in ("NAV", "RA", "ZHIR", "OZ"):
        cand = "IL"
    elif score <= 0.33 and cand in ("NAV", "RA", "IL"):
        cand = "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def parametric_glyph_selector(G, n) -> str:
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

    norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
    Si, dnfr, accel = _selector_normalized_metrics(nd, norms)

    cand = _selector_base_choice(Si, dnfr, accel, thr)

    hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thr, margin)
    if hist_cand is not None:
        return hist_cand

    score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)

    cand = _apply_score_override(cand, score, dnfr, thr["dnfr_lo"])

    return _soft_grammar_prefilter(G, n, cand, dnfr, accel)


def _choose_glyph(G, n, selector, use_canon, h_al, h_en, al_max, en_max):
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
    G, *, step_idx: int, dt: float | None, use_Si: bool, apply_glyphs: bool
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


def _prepare_dnfr(G, *, use_Si: bool) -> None:
    """Compute ΔNFR and optionally Si for the current graph state."""
    compute_dnfr_cb = G.graph.get(
        "compute_delta_nfr", default_compute_delta_nfr
    )
    compute_dnfr_cb(G)
    G.graph.pop("_sel_norms", None)
    if use_Si:
        compute_Si(G, inplace=True)


def _apply_selector(G):
    """Configure and return the glyph selector for this step."""
    selector = G.graph.get("glyph_selector", default_glyph_selector)
    if selector is parametric_glyph_selector:
        _norms_para_selector(G)
        _configure_selector_weights(G)
    return selector


def _apply_glyphs(G, selector, hist) -> None:
    """Apply glyphs to nodes using ``selector`` and update history."""
    window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
    use_canon = bool(
        get_graph_param(G, "GRAMMAR_CANON", dict).get("enabled", False)
    )
    al_max = get_graph_param(G, "AL_MAX_LAG", int)
    en_max = get_graph_param(G, "EN_MAX_LAG", int)
    h_al = hist.setdefault("since_AL", {})
    h_en = hist.setdefault("since_EN", {})
    for n, _ in G.nodes(data=True):
        h_al[n] = int(h_al.get(n, 0)) + 1
        h_en[n] = int(h_en.get(n, 0)) + 1
        g = _choose_glyph(
            G, n, selector, use_canon, h_al, h_en, al_max, en_max
        )
        apply_glyph(G, n, g, window=window)
        if use_canon:
            on_applied_glyph(G, n, g)
        if g == Glyph.AL:
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif g == Glyph.EN:
            h_en[n] = 0


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
    _prepare_dnfr(G, use_Si=use_Si)
    selector = _apply_selector(G)
    if apply_glyphs:
        _apply_glyphs(G, selector, hist)
    _dt = get_graph_param(G, "DT") if dt is None else float(dt)
    method = get_graph_param(G, "INTEGRATOR_METHOD", str)
    raw_jobs = G.graph.get("INTEGRATOR_N_JOBS")
    try:
        n_jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        n_jobs = None
    update_epi_via_nodal_equation(G, dt=_dt, method=method, n_jobs=n_jobs)
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(nd, G, n)
    coordinate_global_local_phase(G, None, None)
    adapt_vf_by_coherence(G)


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
    _update_epi_hist(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)


def run(
    G,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
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
    for _ in range(steps_int):
        step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
        # Early-stop opcional
        if stop_enabled:
            history = ensure_history(G)
            series = history.get("stable_frac", [])
            if not isinstance(series, list):
                series = list(series)
            if len(series) >= w and all(v >= frac for v in series[-w:]):
                break
