"""Basic metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from typing import Any, Dict, Callable

import heapq
import math

from ..constants import (
    ALIAS_EPI,
    ALIAS_DNFR,
    ALIAS_dEPI,
    ALIAS_SI,
    ALIAS_dSI,
    ALIAS_VF,
    ALIAS_dVF,
    ALIAS_D2VF,
    get_param,
)
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, last_glyph, append_metric
from ..alias import get_attr, set_attr
from ..metrics_utils import compute_coherence
from ..constants_glyphs import GLYPHS_CANONICAL, GLYPH_GROUPS
from ..types import Glyph
from .coherence import register_coherence_callbacks
from .diagnosis import register_diagnosis_callbacks
from ..observers import phase_sync, glyph_load, kuramoto_order
from ..sense import sigma_vector
from ..logging_utils import get_logger


logger = get_logger(__name__)


LATENT_GLYPH = Glyph.SHA.value
TgCurr = "curr"
TgRun = "run"


# -------------
# Utilidades internas
# -------------


def _tg_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Internal per-node structure for accumulating run times per glyph.
    Fields: curr (current glyph), run (accumulated time in current glyph)
    """
    return nd.setdefault("_Tg", {TgCurr: None, TgRun: 0.0})


def for_each_glyph(fn: Callable[[str], Any]) -> None:
    """Execute ``fn`` for each canonical glyph.

    ``fn`` is called with a single argument: the glyph identifier.
    """
    for g in GLYPHS_CANONICAL:
        fn(g)


# -------------
# Métricas legadas trasladadas desde ``dynamics``
# -------------


def _update_coherence(G, hist) -> None:
    """Update coherence and related means.

    Records instantaneous coherence ``C`` along with the mean absolute
    ``ΔNFR`` and ``dEPI`` values, and updates the moving average ``W̄``.
    """

    C, dnfr_mean, depi_mean = compute_coherence(G, return_means=True)
    _record_metrics(
        hist,
        (lambda: C, "C_steps"),
        (lambda: dnfr_mean, "dnfr_mean"),
        (lambda: depi_mean, "depi_mean"),
    )

    wbar_w = int(get_param(G, "WBAR_WINDOW"))
    cs = hist["C_steps"]
    if cs:
        w = min(len(cs), max(1, wbar_w))
        wbar = sum(cs[-w:]) / w
        _record_metrics(hist, (lambda: wbar, "W_bar"))


def _record_metrics(
    hist: Dict[str, Any], *pairs: tuple[Callable[[], Any], str]
) -> None:
    """Record metrics using pairs of callables and keys."""
    for fn, key in pairs:
        append_metric(hist, key, fn())


def _update_phase_sync(G, hist) -> None:
    """Record phase synchrony and Kuramoto order."""

    _record_metrics(
        hist,
        (lambda: phase_sync(G), "phase_sync"),
        (lambda: kuramoto_order(G), "kuramoto_R"),
    )


def _update_sigma(G, hist) -> None:
    """Record glyph load and associated Σ⃗ vector."""

    win = int(get_param(G, "GLYPH_LOAD_WINDOW"))
    gl = glyph_load(G, window=win)
    _record_metrics(
        hist,
        (lambda: gl.get("_estabilizadores", 0.0), "glyph_load_estab"),
        (lambda: gl.get("_disruptivos", 0.0), "glyph_load_disr"),
    )

    dist = {k: v for k, v in gl.items() if not k.startswith("_")}
    sig = sigma_vector(dist)
    _record_metrics(
        hist,
        (lambda: sig.get("x", 0.0), "sense_sigma_x"),
        (lambda: sig.get("y", 0.0), "sense_sigma_y"),
        (lambda: sig.get("mag", 0.0), "sense_sigma_mag"),
        (lambda: sig.get("angle", 0.0), "sense_sigma_angle"),
    )


# -------------
# Helpers de métricas
# -------------


def _update_tg_node(n, nd, dt, tg_total, tg_by_node):
    """Process a single node glyph transition.

    Returns ``(glyph, is_latent)`` or ``(None, False)`` when the node has no
    glyph. ``tg_total`` and ``tg_by_node`` are updated in-place.
    """
    g = last_glyph(nd)
    if not g:
        return None, False
    st = _tg_state(nd)
    curr = st[TgCurr]
    if curr is None:
        st[TgCurr] = g
        st[TgRun] = dt
    elif g == curr:
        st[TgRun] += dt
    else:
        dur = float(st[TgRun])
        tg_total[curr] += dur
        if tg_by_node is not None:
            tg_by_node[n][curr].append(dur)
        st[TgCurr] = g
        st[TgRun] = dt
    return g, g == LATENT_GLYPH


def _update_tg(G, hist, dt, save_by_node: bool):
    """Accumulate glyph times per node and return counts and latency."""
    counts = Counter()
    tg_total = hist.setdefault("Tg_total", defaultdict(float))
    tg_by_node = (
        hist.setdefault("Tg_by_node", defaultdict(lambda: defaultdict(list)))
        if save_by_node
        else None
    )

    n_total = 0
    n_latent = 0
    for n, nd in G.nodes(data=True):
        g, is_latent = _update_tg_node(n, nd, dt, tg_total, tg_by_node)
        if g is None:
            continue
        n_total += 1
        if is_latent:
            n_latent += 1
        counts[g] += 1
    return counts, n_total, n_latent


def _update_glyphogram(G, hist, counts, t, n_total):
    """Record the glyphogram for the step from counts."""
    normalize_series = bool(
        get_param(G, "METRICS").get("normalize_series", False)
    )
    row = {"t": t}
    total = max(1, n_total)
    for g in GLYPHS_CANONICAL:
        c = counts.get(g, 0)
        row[g] = (c / total) if normalize_series else c
    append_metric(hist, "glyphogram", row)


def _update_latency_index(G, hist, n_total, n_latent, t):
    """Add latency index to history."""
    li = n_latent / max(1, n_total)
    append_metric(hist, "latency_index", {"t": t, "value": li})


def _update_epi_support(G, hist, t, thr):
    """Compute EPI support and norm."""
    total = 0.0
    count = 0
    for _, nd in G.nodes(data=True):
        epi_val = abs(get_attr(nd, ALIAS_EPI, 0.0))
        if epi_val >= thr:
            total += epi_val
            count += 1
    epi_norm = (total / count) if count else 0.0
    append_metric(
        hist,
        "EPI_support",
        {"t": t, "size": count, "epi_norm": float(epi_norm)},
    )


def _update_morph_metrics(G, hist, counts, t):
    """Record morphosyntactic metrics based on glyph counts."""

    def get_count(keys):
        return sum(counts.get(k, 0) for k in keys)

    total = max(1, sum(counts.values()))
    id_val = get_count(GLYPH_GROUPS.get("ID", ())) / total
    cm_val = get_count(GLYPH_GROUPS.get("CM", ())) / total
    ne_val = get_count(GLYPH_GROUPS.get("NE", ())) / total
    num = get_count(GLYPH_GROUPS.get("PP_num", ()))
    den = get_count(GLYPH_GROUPS.get("PP_den", ()))
    pp_val = 0.0 if den == 0 else num / den
    append_metric(
        hist,
        "morph",
        {"t": t, "ID": id_val, "CM": cm_val, "NE": ne_val, "PP": pp_val},
    )


def _track_stability(G, hist, dt, eps_dnfr, eps_depi):
    """Track per-node stability and derivative metrics."""

    stables = 0
    total = max(1, G.number_of_nodes())
    delta_si_sum = 0.0
    delta_si_count = 0
    B_sum = 0.0
    B_count = 0

    for _, nd in G.nodes(data=True):
        if (
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) <= eps_dnfr
            and abs(get_attr(nd, ALIAS_dEPI, 0.0)) <= eps_depi
        ):
            stables += 1

        Si_curr = get_attr(nd, ALIAS_SI, 0.0)
        Si_prev = nd.get("_prev_Si", Si_curr)
        dSi = Si_curr - Si_prev
        nd["_prev_Si"] = Si_curr
        set_attr(nd, ALIAS_dSI, dSi)
        delta_si_sum += dSi
        delta_si_count += 1

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

    hist["stable_frac"].append(stables / total)
    hist["delta_Si"].append(
        delta_si_sum / delta_si_count if delta_si_count else 0.0
    )
    hist["B"].append(B_sum / B_count if B_count else 0.0)


def _aggregate_si(G, hist):
    """Aggregate Si statistics for all nodes."""

    try:
        thr_sel = get_param(G, "SELECTOR_THRESHOLDS")
        thr_def = get_param(G, "GLYPH_THRESHOLDS")
        si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
        si_lo = float(thr_sel.get("si_lo", thr_def.get("lo", 0.33)))

        sis = [
            s
            for _, nd in G.nodes(data=True)
            if not math.isnan(s := get_attr(nd, ALIAS_SI, float("nan")))
        ]

        total = 0.0
        hi_count = 0
        lo_count = 0
        for s in sis:
            total += s
            if s >= si_hi:
                hi_count += 1
            if s <= si_lo:
                lo_count += 1

        n = len(sis)
        if n:
            hist["Si_mean"].append(total / n)
            hist["Si_hi_frac"].append(hi_count / n)
            hist["Si_lo_frac"].append(lo_count / n)
        else:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("Si aggregation failed: %s", exc)


def _compute_advanced_metrics(G, hist, t, dt, cfg, thr):
    """Compute advanced glyph-based metrics."""

    save_by_node = bool(cfg.get("save_by_node", True))
    counts, n_total, n_latent = _update_tg(G, hist, dt, save_by_node)
    _update_glyphogram(G, hist, counts, t, n_total)
    _update_latency_index(G, hist, n_total, n_latent, t)
    _update_epi_support(G, hist, t, thr)
    _update_morph_metrics(G, hist, counts, t)


# -------------
# Callback principal: actualizar métricas por paso
# -------------


def _metrics_step(G, *args, **kwargs):
    """Update operational TNFR metrics per step.

    Coordinates updates of glyphogram, latency index, Tg, EPI support and
    morphosyntactic metrics. All results are stored in ``G.graph['history']``.
    """
    cfg = get_param(G, "METRICS")
    if not cfg.get("enabled", True):
        return

    hist = ensure_history(G)
    dt = float(get_param(G, "DT"))
    t = float(G.graph.get("_t", 0.0))
    thr = float(get_param(G, "EPI_SUPPORT_THR"))

    # -- Métricas básicas heredadas de ``dynamics`` --
    for k in (
        "C_steps",
        "stable_frac",
        "phase_sync",
        "glyph_load_estab",
        "glyph_load_disr",
        "Si_mean",
        "Si_hi_frac",
        "Si_lo_frac",
        "delta_Si",
        "B",
    ):
        hist.setdefault(k, [])

    _update_coherence(G, hist)

    eps_dnfr = float(get_param(G, "EPS_DNFR_STABLE"))
    eps_depi = float(get_param(G, "EPS_DEPI_STABLE"))
    _track_stability(G, hist, dt, eps_dnfr, eps_depi)
    try:
        _update_phase_sync(G, hist)
        _update_sigma(G, hist)
        if hist.get("C_steps") and hist.get("stable_frac"):
            append_metric(
                hist,
                "iota",
                hist["C_steps"][-1] * hist["stable_frac"][-1],
            )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("observer update failed: %s", exc)

    _aggregate_si(G, hist)
    _compute_advanced_metrics(G, hist, t, dt, cfg, thr)


# -------------
# Registro del callback
# -------------


def register_metrics_callbacks(G) -> None:
    register_callback(
        G, event="after_step", func=_metrics_step, name="metrics_step"
    )
    # Nuevas funcionalidades canónicas
    register_coherence_callbacks(G)
    register_diagnosis_callbacks(G)


# -------------
# Consultas / reportes
# -------------


def Tg_global(G, normalize: bool = True) -> Dict[str, float]:
    """Total glyph time per class. If ``normalize=True``, return fractions
    of the total."""
    hist = ensure_history(G)
    tg_total: Dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    out: Dict[str, float] = {}

    def add(g):
        val = float(tg_total.get(g, 0.0))
        out[g] = val / total if normalize else val

    for_each_glyph(add)
    return out


def Tg_by_node(
    G, n, normalize: bool = False
) -> Dict[str, float | list[float]]:
    """Per-node summary: if ``normalize`` return mean run per glyph;
    otherwise list runs."""
    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    if not normalize:
        # convertir default dict → list para serializar
        out: Dict[str, list[float]] = {}

        def copy_runs(g):
            out[g] = list(rec.get(g, []))

        for_each_glyph(copy_runs)
        return out
    out: Dict[str, float] = {}

    def add(g):
        runs = rec.get(g, [])
        out[g] = float(mean(runs)) if runs else 0.0

    for_each_glyph(add)
    return out


def latency_series(G) -> Dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glyphogram_series(G) -> Dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("glyphogram", [])
    if not xs:
        return {"t": []}
    out: Dict[str, list[float]] = {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)]
    }

    def add(g):
        out[g] = [float(x.get(g, 0.0)) for x in xs]

    for_each_glyph(add)
    return out


def glyph_top(G, k: int = 3) -> list[tuple[str, float]]:
    """Top-k structural operators by ``Tg_global`` (fraction)."""
    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    tg = Tg_global(G, normalize=True)
    return heapq.nlargest(k, tg.items(), key=lambda kv: kv[1])


def glyph_dwell_stats(G, n) -> Dict[str, Dict[str, float]]:
    """Per-node statistics: mean/median/max of runs per glyph."""
    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    out: Dict[str, Dict[str, float]] = {}

    def add(g):
        runs = list(rec.get(g, []))
        if not runs:
            out[g] = {"mean": 0.0, "median": 0.0, "max": 0.0, "count": 0}
        else:
            out[g] = {
                "mean": float(mean(runs)),
                "median": float(median(runs)),
                "max": float(max(runs)),
                "count": int(len(runs)),
            }

    for_each_glyph(add)
    return out
