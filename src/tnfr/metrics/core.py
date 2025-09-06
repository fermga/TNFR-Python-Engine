"""Basic metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any
import heapq
import logging
import math

from ..constants import (
    METRIC_DEFAULTS,
    ALIAS_EPI,
    METRICS,
    REMESH_DEFAULTS,
    DEFAULTS,
    ALIAS_DNFR,
    ALIAS_dEPI,
    ALIAS_SI,
    ALIAS_dSI,
    ALIAS_VF,
    ALIAS_dVF,
    ALIAS_D2VF,
)
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, last_glyph, append_metric
from ..alias import get_attr, set_attr
from ..helpers import list_mean
from ..metrics_utils import compute_coherence
from ..constants_glyphs import GLYPHS_CANONICAL, GLYPH_GROUPS
from ..types import Glyph
from .coherence import register_coherence_callbacks
from .diagnosis import register_diagnosis_callbacks
from ..observers import phase_sync, glyph_load, kuramoto_order
from ..sense import sigma_vector


logger = logging.getLogger(__name__)


LATENT_GLYPH = Glyph.SHA.value
TgCurr = "curr"
TgRun = "run"


# -------------
# Utilidades internas
# -------------


def _tg_state(nd: dict[str, Any]) -> dict[str, Any]:
    """Internal per-node structure for accumulating run times per glyph.
    Fields: curr (current glyph), run (accumulated time in current glyph)
    """
    return nd.setdefault("_Tg", {TgCurr: None, TgRun: 0.0})


def for_each_glyph(fn) -> None:
    """Execute ``fn`` for each canonical glyph.

    ``fn`` is called with a single argument: the glyph identifier.
    """
    for g in GLYPHS_CANONICAL:
        fn(g)


# -------------
# Métricas legadas trasladadas desde ``dynamics``
# -------------


def _update_coherence(G, hist) -> None:
    """Update global coherence and its moving average."""

    C = compute_coherence(G)
    append_metric(hist, "C_steps", C)

    wbar_w = int(
        G.graph.get("WBAR_WINDOW", METRIC_DEFAULTS.get("WBAR_WINDOW", 25))
    )
    cs = hist["C_steps"]
    if cs:
        w = min(len(cs), max(1, wbar_w))
        wbar = sum(cs[-w:]) / w
        append_metric(hist, "W_bar", wbar)


def _update_phase_sync(G, hist) -> None:
    """Record phase synchrony and Kuramoto order."""

    ps = phase_sync(G)
    append_metric(hist, "phase_sync", ps)
    R = kuramoto_order(G)
    append_metric(hist, "kuramoto_R", R)


def _update_sigma(G, hist) -> None:
    """Record glyph load and associated Σ⃗ vector."""

    win = int(
        G.graph.get("GLYPH_LOAD_WINDOW", METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"])
    )
    gl = glyph_load(G, window=win)
    append_metric(
        hist,
        "glyph_load_estab",
        gl.get("_estabilizadores", 0.0),
    )
    append_metric(hist, "glyph_load_disr", gl.get("_disruptivos", 0.0))

    dist = {k: v for k, v in gl.items() if not k.startswith("_")}
    sig, _ = sigma_vector(dist)
    append_metric(hist, "sense_sigma_x", sig.get("x", 0.0))
    append_metric(hist, "sense_sigma_y", sig.get("y", 0.0))
    append_metric(hist, "sense_sigma_mag", sig.get("mag", 0.0))
    append_metric(hist, "sense_sigma_angle", sig.get("angle", 0.0))


# -------------
# Helpers de métricas
# -------------


def _update_tg(G, hist, dt, save_by_node: bool):
    """Accumulate glyph times per node and return counts and latency.

    ``save_by_node`` controls whether per-node runs are recorded.
    """
    counts = Counter()
    n_total = 0
    n_latent = 0

    tg_total = hist.setdefault("Tg_total", defaultdict(float))
    tg_by_node = (
        hist.setdefault("Tg_by_node", defaultdict(lambda: defaultdict(list)))
        if save_by_node
        else None
    )

    last = last_glyph
    tg_state = _tg_state
    latent = LATENT_GLYPH
    curr_key = TgCurr
    run_key = TgRun

    nodes = G.nodes
    for n in nodes():
        nd = nodes[n]
        g = last(nd)
        if not g:
            continue

        n_total += 1
        if g == latent:
            n_latent += 1

        counts[g] += 1

        st = tg_state(nd)
        if st[curr_key] is None:
            st[curr_key] = g
            st[run_key] = dt
        elif g == st[curr_key]:
            st[run_key] += dt
        else:
            prev = st[curr_key]
            dur = float(st[run_key])
            tg_total[prev] += dur
            if save_by_node:
                tg_by_node[n][prev].append(dur)
            st[curr_key] = g
            st[run_key] = dt

    return counts, n_total, n_latent


def _update_glyphogram(G, hist, counts, t, n_total):
    """Record the glyphogram for the step from counts."""
    normalize_series = bool(
        G.graph.get("METRICS", METRICS).get("normalize_series", False)
    )
    row = {"t": t}
    total = max(1, n_total)

    def add_row(g):
        c = counts.get(g, 0)
        row[g] = (c / total) if normalize_series else c

    for_each_glyph(add_row)
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
        sis = [
            get_attr(nd, ALIAS_SI, float("nan")) for _, nd in G.nodes(data=True)
        ]
        sis = [s for s in sis if not math.isnan(s)]
        if sis:
            si_mean = list_mean(sis, 0.0)
            hist["Si_mean"].append(si_mean)
            thr_sel = G.graph.get(
                "SELECTOR_THRESHOLDS", DEFAULTS.get("SELECTOR_THRESHOLDS", {})
            )
            thr_def = G.graph.get(
                "GLYPH_THRESHOLDS",
                DEFAULTS.get("GLYPH_THRESHOLDS", {"hi": 0.66, "lo": 0.33}),
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
    cfg = G.graph.get("METRICS", METRICS)
    if not cfg.get("enabled", True):
        return

    hist = ensure_history(G)
    dt = float(G.graph.get("DT", 1.0))
    t = float(G.graph.get("_t", 0.0))
    thr = float(
        G.graph.get(
            "EPI_SUPPORT_THR", METRIC_DEFAULTS.get("EPI_SUPPORT_THR", 0.0)
        )
    )

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

    eps_dnfr = float(
        G.graph.get("EPS_DNFR_STABLE", REMESH_DEFAULTS["EPS_DNFR_STABLE"])
    )
    eps_depi = float(
        G.graph.get("EPS_DEPI_STABLE", REMESH_DEFAULTS["EPS_DEPI_STABLE"])
    )
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


def Tg_global(G, normalize: bool = True) -> dict[str, float]:
    """Total glyph time per class. If ``normalize=True``, return fractions
    of the total."""
    hist = ensure_history(G)
    tg_total: dict[str, float] = hist.get("Tg_total", {})
    total = sum(tg_total.values()) or 1.0
    out: dict[str, float] = {}

    def add(g):
        val = float(tg_total.get(g, 0.0))
        out[g] = val / total if normalize else val

    for_each_glyph(add)
    return out


def Tg_by_node(
    G, n, normalize: bool = False
) -> dict[str, float | list[float]]:
    """Per-node summary: if ``normalize`` return mean run per glyph;
    otherwise list runs."""
    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    if not normalize:
        # convertir default dict → list para serializar
        out: dict[str, list[float]] = {}

        def copy_runs(g):
            out[g] = list(rec.get(g, []))

        for_each_glyph(copy_runs)
        return out
    out: dict[str, float] = {}

    def add(g):
        runs = rec.get(g, [])
        out[g] = float(mean(runs)) if runs else 0.0

    for_each_glyph(add)
    return out


def latency_series(G) -> dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("latency_index", [])
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "value": [float(x.get("value", 0.0)) for x in xs],
    }


def glyphogram_series(G) -> dict[str, list[float]]:
    hist = ensure_history(G)
    xs = hist.get("glyphogram", [])
    if not xs:
        return {"t": []}
    out = {"t": [float(x.get("t", i)) for i, x in enumerate(xs)]}

    def add(g):
        out[g] = [float(x.get(g, 0.0)) for x in xs]

    for_each_glyph(add)
    return out


def glyph_top(G, k: int = 3) -> list[tuple[str, float]]:
    """Top-k structural operators by ``Tg_global`` (fraction)."""
    tg = Tg_global(G, normalize=True)
    return heapq.nlargest(max(1, int(k)), tg.items(), key=lambda kv: kv[1])


def glyph_dwell_stats(G, n) -> dict[str, dict[str, float]]:
    """Per-node statistics: mean/median/max of runs per glyph."""
    hist = ensure_history(G)
    rec = hist.get("Tg_by_node", {}).get(n, {})
    out: dict[str, dict[str, float]] = {}

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
