"""Glyph selection helpers for TNFR dynamics."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, TypeAlias, cast

from ..alias import collect_attr, get_attr
from ..constants import get_graph_param, get_param
from ..glyph_history import ensure_history, recent_glyph
from ..helpers.numeric import clamp01
from ..metrics.common import compute_dnfr_accel_max, merge_and_normalize_weights
from ..operators import apply_glyph
from ..selector import (
    _apply_selector_hysteresis,
    _calc_selector_score,
    _selector_norms,
    _selector_thresholds,
)
from ..types import Glyph, GlyphSelector, HistoryState, NodeId, TNFRGraph
from ..utils import get_numpy
from ..validation.grammar import enforce_canonical_grammar, on_applied_glyph
from .aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_DSI, ALIAS_SI

GlyphCode: TypeAlias = Glyph | str

__all__ = (
    "GlyphCode",
    "default_glyph_selector",
    "parametric_glyph_selector",
    "_SelectorPreselection",
    "_configure_selector_weights",
    "_apply_selector",
    "_apply_glyphs",
    "_selector_parallel_jobs",
    "_prepare_selector_preselection",
    "_resolve_preselected_glyph",
    "_choose_glyph",
)


def default_glyph_selector(G: TNFRGraph, n: NodeId) -> GlyphCode:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    hi, lo, dnfr_hi = itemgetter("si_hi", "si_lo", "dnfr_hi")(thr)

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
    nd: Mapping[str, Any], norms: Mapping[str, float]
) -> tuple[float, float, float]:
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    acc_max = float(norms.get("accel_max", 1.0)) or 1.0
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max
    return Si, dnfr, accel


def _selector_base_choice(
    Si: float, dnfr: float, accel: float, thr: Mapping[str, float]
) -> GlyphCode:
    si_hi, si_lo, dnfr_hi, acc_hi = itemgetter(
        "si_hi", "si_lo", "dnfr_hi", "accel_hi"
    )(thr)
    if Si >= si_hi:
        return "IL"
    if Si <= si_lo:
        if accel >= acc_hi:
            return "THOL"
        return "OZ" if dnfr >= dnfr_hi else "ZHIR"
    if dnfr >= dnfr_hi or accel >= acc_hi:
        return "NAV"
    return "RA"


def _configure_selector_weights(G: TNFRGraph) -> Mapping[str, float]:
    weights = merge_and_normalize_weights(
        G, "SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel")
    )
    cast_weights = cast(Mapping[str, float], weights)
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
    W = G.graph.get("_selector_weights")
    if W is None:
        W = _configure_selector_weights(G)
    score = _calc_selector_score(Si, dnfr, accel, cast(Mapping[str, float], W))
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
    cand_key = str(cand)
    if score >= 0.66 and cand_key in ("NAV", "RA", "ZHIR", "OZ"):
        return "IL"
    if score <= 0.33 and cand_key in ("NAV", "RA", "IL"):
        return "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def parametric_glyph_selector(G: TNFRGraph, n: NodeId) -> GlyphCode:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    margin = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")

    norms = cast(Mapping[str, float] | None, G.graph.get("_sel_norms"))
    if norms is None:
        norms = _selector_norms(G)
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
    if h_al[n] > al_max:
        return Glyph.AL
    if h_en[n] > en_max:
        return Glyph.EN
    g = selector(G, n)
    if use_canon:
        g = enforce_canonical_grammar(G, n, g)
    return g


@dataclass(slots=True)
class _SelectorPreselection:
    kind: str
    metrics: Mapping[Any, tuple[float, float, float]]
    base_choices: Mapping[Any, GlyphCode]
    thresholds: Mapping[str, float] | None = None
    margin: float | None = None


def _selector_parallel_jobs(G: TNFRGraph) -> int | None:
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
    si_values, dnfr_values, accel_values, dnfr_max, accel_max = args
    si_seq = [clamp01(float(v)) for v in si_values]
    dnfr_seq = [abs(float(v)) / dnfr_max for v in dnfr_values]
    accel_seq = [abs(float(v)) / accel_max for v in accel_values]
    return si_seq, dnfr_seq, accel_seq


def _collect_selector_metrics(
    G: TNFRGraph,
    nodes: list[Any],
    norms: Mapping[str, float],
    n_jobs: int | None = None,
) -> dict[Any, tuple[float, float, float]]:
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

            si_seq = []
            dnfr_seq = []
            accel_seq = []

            def _args_iter() -> Sequence[tuple[list[float], list[float], list[float], float, float]]:
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
    metrics: Mapping[Any, tuple[float, float, float]],
    thresholds: Mapping[str, float],
) -> dict[Any, str]:
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
    args: tuple[Mapping[str, float], list[tuple[Any, tuple[float, float, float]]]]
) -> list[tuple[Any, str]]:
    thresholds, chunk = args
    return [
        (node, _selector_base_choice(Si, dnfr, accel, thresholds))
        for node, (Si, dnfr, accel) in chunk
    ]


def _compute_param_base_choices(
    metrics: Mapping[Any, tuple[float, float, float]],
    thresholds: Mapping[str, float],
    n_jobs: int | None,
) -> dict[Any, str]:
    if not metrics:
        return {}

    items = list(metrics.items())
    if n_jobs is None or n_jobs <= 1:
        return {
            node: _selector_base_choice(Si, dnfr, accel, thresholds)
            for node, (Si, dnfr, accel) in items
        }

    chunk_size = max(1, math.ceil(len(items) / n_jobs))
    chunks = [
        items[i : i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]
    base: dict[Any, str] = {}
    args = ((thresholds, chunk) for chunk in chunks)
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
    if selector is default_glyph_selector:
        norms = G.graph.get("_sel_norms") or _selector_norms(G)
        thresholds = _selector_thresholds(G)
        n_jobs = _selector_parallel_jobs(G)
        metrics = _collect_selector_metrics(G, list(nodes), norms, n_jobs=n_jobs)
        base_choices = _compute_default_base_choices(metrics, thresholds)
        return _SelectorPreselection(
            "default", metrics, base_choices, thresholds=thresholds
        )
    if selector is parametric_glyph_selector:
        norms = G.graph.get("_sel_norms") or _selector_norms(G)
        thresholds = _selector_thresholds(G)
        n_jobs = _selector_parallel_jobs(G)
        metrics = _collect_selector_metrics(G, list(nodes), norms, n_jobs=n_jobs)
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
    nodes, G, selector, preselection = args
    return [
        (n, _resolve_preselected_glyph(G, n, selector, preselection))
        for n in nodes
    ]


def _apply_glyphs(G: TNFRGraph, selector: GlyphSelector, hist: HistoryState) -> None:
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
                to_select[idx : idx + chunk_size]
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


def _apply_selector(G: TNFRGraph) -> GlyphSelector:
    selector = cast(
        GlyphSelector,
        G.graph.get("glyph_selector", default_glyph_selector),
    )
    if selector is parametric_glyph_selector:
        _selector_norms(G)
        _configure_selector_weights(G)
    return selector

