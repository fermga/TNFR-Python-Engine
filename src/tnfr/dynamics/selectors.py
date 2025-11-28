"""Glyph selection helpers for TNFR dynamics."""

from __future__ import annotations

from ..compat.dataclass import dataclass
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from operator import itemgetter
from typing import Any, cast
from ..alias import collect_attr, get_attr
from ..constants import get_graph_param, get_param
from ..glyph_history import ensure_history
from ..utils import clamp01, resolve_chunk_size
from ..metrics.common import compute_dnfr_accel_max, merge_and_normalize_weights
from ..operators import apply_glyph
from ..validation import (
    GrammarContext,
    StructuralGrammarError,
    enforce_canonical_grammar,
    on_applied_glyph,
    record_grammar_violation,
)
from ..selector import (
    _apply_selector_hysteresis,
    _calc_selector_score,
    _selector_norms,
    _selector_parallel_jobs,
    _selector_thresholds,
)
from ..types import Glyph, GlyphCode, GlyphSelector, HistoryState, NodeId, TNFRGraph
from ..utils import get_numpy
from ..validation import soft_grammar_filters
from .aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_DSI, ALIAS_SI

__all__ = (
    "GlyphCode",
    "AbstractSelector",
    "DefaultGlyphSelector",
    "ParametricGlyphSelector",
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


class AbstractSelector(ABC):
    """Interface describing glyph selector lifecycle hooks."""

    def prepare(
        self, graph: TNFRGraph, nodes: Sequence[NodeId]
    ) -> None:  # pragma: no cover - default no-op
        """Prepare selector state before evaluating a glyph batch."""

    @abstractmethod
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Return the glyph to apply for ``node`` within ``graph``."""

    def __call__(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Allow selectors to be used as legacy callables."""

        return self.select(graph, node)


def _default_selector_logic(G: TNFRGraph, n: NodeId) -> GlyphCode:
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
) -> GlyphCode:
    """Soft grammar: avoid repetitions before the canonical one."""

    ctx = GrammarContext.from_graph(G)
    filtered = soft_grammar_filters(ctx, n, cand)
    return cast(GlyphCode, filtered)


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
    si_hi, si_lo, dnfr_hi, acc_hi = itemgetter("si_hi", "si_lo", "dnfr_hi", "accel_hi")(thr)
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
    """Load and cache selector weight configuration from graph parameters."""

    weights = merge_and_normalize_weights(G, "SELECTOR_WEIGHTS", ("w_si", "w_dnfr", "w_accel"))
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


def _apply_score_override(cand: GlyphCode, score: float, dnfr: float, dnfr_lo: float) -> GlyphCode:
    cand_key = str(cand)
    if score >= 0.66 and cand_key in ("NAV", "RA", "ZHIR", "OZ"):
        return "IL"
    if score <= 0.33 and cand_key in ("NAV", "RA", "IL"):
        return "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def _parametric_selector_logic(G: TNFRGraph, n: NodeId) -> GlyphCode:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    margin: float | None = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")

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

    return _soft_grammar_prefilter(G, n, cand)


@dataclass(slots=True)
class _SelectorPreselection:
    """Precomputed selector context shared across glyph decisions."""

    kind: str
    metrics: Mapping[Any, tuple[float, float, float]]
    base_choices: Mapping[Any, GlyphCode]
    thresholds: Mapping[str, float] | None = None
    margin: float | None = None


def _build_default_preselection(G: TNFRGraph, nodes: Sequence[NodeId]) -> _SelectorPreselection:
    node_list = list(nodes)
    thresholds = _selector_thresholds(G)
    if not node_list:
        return _SelectorPreselection("default", {}, {}, thresholds=thresholds)

    norms = G.graph.get("_sel_norms") or _selector_norms(G)
    n_jobs = _selector_parallel_jobs(G)
    metrics = _collect_selector_metrics(G, node_list, norms, n_jobs=n_jobs)
    base_choices = _compute_default_base_choices(metrics, thresholds)
    return _SelectorPreselection("default", metrics, base_choices, thresholds=thresholds)


def _build_param_preselection(G: TNFRGraph, nodes: Sequence[NodeId]) -> _SelectorPreselection:
    node_list = list(nodes)
    thresholds = _selector_thresholds(G)
    margin: float | None = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")
    if not node_list:
        return _SelectorPreselection("param", {}, {}, thresholds=thresholds, margin=margin)

    norms = G.graph.get("_sel_norms") or _selector_norms(G)
    n_jobs = _selector_parallel_jobs(G)
    metrics = _collect_selector_metrics(G, node_list, norms, n_jobs=n_jobs)
    base_choices = _compute_param_base_choices(metrics, thresholds, n_jobs)
    return _SelectorPreselection(
        "param",
        metrics,
        base_choices,
        thresholds=thresholds,
        margin=margin,
    )


class DefaultGlyphSelector(AbstractSelector):
    """Selector implementing the legacy default glyph heuristic."""

    __slots__ = ("_preselection", "_prepared_graph_id")

    def __init__(self) -> None:
        self._preselection: _SelectorPreselection | None = None
        self._prepared_graph_id: int | None = None

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None:
        """Precompute default selector metrics for ``nodes``."""

        self._preselection = _build_default_preselection(graph, nodes)
        self._prepared_graph_id = id(graph)

    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Return the canonical glyph for ``node`` using cached metrics when available."""

        if self._prepared_graph_id == id(graph):
            preselection = self._preselection
        else:
            preselection = None
        return _resolve_preselected_glyph(graph, node, _default_selector_logic, preselection)


class ParametricGlyphSelector(AbstractSelector):
    """Selector exposing the parametric scoring pipeline."""

    __slots__ = ("_preselection", "_prepared_graph_id")

    def __init__(self) -> None:
        self._preselection: _SelectorPreselection | None = None
        self._prepared_graph_id: int | None = None

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None:
        """Precompute parametric selector metrics and hysteresis thresholds."""

        _selector_norms(graph)
        _configure_selector_weights(graph)
        self._preselection = _build_param_preselection(graph, nodes)
        self._prepared_graph_id = id(graph)

    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Return the parametric glyph decision for ``node``."""

        if self._prepared_graph_id == id(graph):
            preselection = self._preselection
        else:
            preselection = None
        return _resolve_preselected_glyph(graph, node, _parametric_selector_logic, preselection)


default_glyph_selector = DefaultGlyphSelector()
parametric_glyph_selector = ParametricGlyphSelector()


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
    """Return glyph for ``n`` considering forced lags and canonical grammar."""

    if h_al[n] > al_max:
        return Glyph.AL
    if h_en[n] > en_max:
        return Glyph.EN
    g = selector(G, n)
    if use_canon:
        try:
            g = enforce_canonical_grammar(G, n, g)
        except StructuralGrammarError as err:
            nd = G.nodes[n]
            history = tuple(str(item) for item in nd.get("glyph_history", ()))
            selector_name = getattr(selector, "__name__", selector.__class__.__name__)
            err.attach_context(node=n, selector=selector_name, history=history, stage="selector")
            record_grammar_violation(G, n, err, stage="selector")
            raise
    return g


def _selector_metrics_chunk(
    args: tuple[list[float], list[float], list[float], float, float],
) -> tuple[list[float], list[float], list[float]]:
    """Normalise metric chunk values for multiprocessing execution."""

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
    """Return normalised (Si, ΔNFR, acceleration) triples for ``nodes``."""

    if not nodes:
        return {}

    dynamics_module = sys.modules.get("tnfr.dynamics")
    get_numpy_fn = get_numpy
    if dynamics_module is not None:
        get_numpy_fn = getattr(dynamics_module, "get_numpy", get_numpy)

    np_mod = get_numpy_fn()
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    accel_max = float(norms.get("accel_max", 1.0)) or 1.0

    if np_mod is not None:
        si_seq_np = collect_attr(G, nodes, ALIAS_SI, 0.5, np=np_mod).astype(float)
        si_seq_np = np_mod.clip(si_seq_np, 0.0, 1.0)
        dnfr_seq_np = (
            np_mod.abs(collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np_mod).astype(float)) / dnfr_max
        )
        accel_seq_np = (
            np_mod.abs(collect_attr(G, nodes, ALIAS_D2EPI, 0.0, np=np_mod).astype(float))
            / accel_max
        )

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
            approx_chunk = math.ceil(len(nodes) / worker_count) if worker_count else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(nodes),
                minimum=1,
            )
            chunk_bounds = [
                (start, min(start + chunk_size, len(nodes)))
                for start in range(0, len(nodes), chunk_size)
            ]

            si_seq = []
            dnfr_seq = []
            accel_seq = []

            def _args_iter() -> (
                Sequence[tuple[list[float], list[float], list[float], float, float]]
            ):
                for start, end in chunk_bounds:
                    yield (
                        si_values[start:end],
                        dnfr_values[start:end],
                        accel_values[start:end],
                        dnfr_max,
                        accel_max,
                    )

            executor_cls = ProcessPoolExecutor
            if dynamics_module is not None:
                executor_cls = getattr(dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor)
            with executor_cls(max_workers=worker_count) as executor:
                for si_chunk, dnfr_chunk, accel_chunk in executor.map(
                    _selector_metrics_chunk, _args_iter()
                ):
                    si_seq.extend(si_chunk)
                    dnfr_seq.extend(dnfr_chunk)
                    accel_seq.extend(accel_chunk)

    return {node: (si_seq[idx], dnfr_seq[idx], accel_seq[idx]) for idx, node in enumerate(nodes)}


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
    args: tuple[Mapping[str, float], list[tuple[Any, tuple[float, float, float]]]],
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

    approx_chunk = math.ceil(len(items) / n_jobs) if n_jobs else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        len(items),
        minimum=1,
    )
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    base: dict[Any, str] = {}
    args = ((thresholds, chunk) for chunk in chunks)
    executor_cls = ProcessPoolExecutor
    dynamics_module = sys.modules.get("tnfr.dynamics")
    if dynamics_module is not None:
        executor_cls = getattr(dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor)
    with executor_cls(max_workers=n_jobs) as executor:
        for result in executor.map(_param_base_worker, args):
            for node, cand in result:
                base[node] = cand
    return base


def _prepare_selector_preselection(
    G: TNFRGraph,
    selector: GlyphSelector,
    nodes: Sequence[NodeId],
) -> _SelectorPreselection | None:
    """Build cached selector metrics when ``selector`` supports them."""

    if selector is default_glyph_selector:
        return _build_default_preselection(G, nodes)
    if selector is parametric_glyph_selector:
        return _build_param_preselection(G, nodes)
    return None


def _resolve_preselected_glyph(
    G: TNFRGraph,
    n: NodeId,
    selector: GlyphSelector,
    preselection: _SelectorPreselection | None,
) -> GlyphCode:
    """Return glyph for ``n`` using ``preselection`` shortcuts when possible."""

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
        margin: float | None = preselection.margin
        if margin is None:
            margin = get_graph_param(G, "GLYPH_SELECTOR_MARGIN")

        cand = preselection.base_choices.get(n)
        if cand is None:
            cand = _selector_base_choice(Si, dnfr, accel, thresholds)

        nd = G.nodes[n]
        hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thresholds, margin)
        if hist_cand is not None:
            return hist_cand

        score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)
        cand = _apply_score_override(cand, score, dnfr, thresholds["dnfr_lo"])
        return _soft_grammar_prefilter(G, n, cand)

    return selector(G, n)


def _glyph_proposal_worker(
    args: tuple[
        list[NodeId],
        TNFRGraph,
        GlyphSelector,
        _SelectorPreselection | None,
    ],
) -> list[tuple[NodeId, GlyphCode]]:
    nodes, G, selector, preselection = args
    return [(n, _resolve_preselected_glyph(G, n, selector, preselection)) for n in nodes]


def _apply_glyphs(G: TNFRGraph, selector: GlyphSelector, hist: HistoryState) -> None:
    """Apply glyph decisions across the graph updating hysteresis trackers."""

    window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
    use_canon = bool(get_graph_param(G, "GRAMMAR_CANON", dict).get("enabled", False))
    al_max = get_graph_param(G, "AL_MAX_LAG", int)
    en_max = get_graph_param(G, "EN_MAX_LAG", int)

    nodes_data = list(G.nodes(data=True))
    nodes = [n for n, _ in nodes_data]
    if isinstance(selector, AbstractSelector):
        selector.prepare(G, nodes)
        preselection: _SelectorPreselection | None = None
    else:
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
    forced_al_nodes = {n for n, choice in forced.items() if choice == Glyph.AL}
    forced_en_nodes = {n for n, choice in forced.items() if choice == Glyph.EN}
    if to_select:
        n_jobs = _selector_parallel_jobs(G)
        if n_jobs is None:
            for n in to_select:
                decisions[n] = _resolve_preselected_glyph(G, n, selector, preselection)
        else:
            approx_chunk = math.ceil(len(to_select) / n_jobs) if n_jobs else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(to_select),
                minimum=1,
            )
            chunks = [
                to_select[idx : idx + chunk_size] for idx in range(0, len(to_select), chunk_size)
            ]
            dynamics_module = sys.modules.get("tnfr.dynamics")
            executor_cls = ProcessPoolExecutor
            if dynamics_module is not None:
                executor_cls = getattr(dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor)
            with executor_cls(max_workers=n_jobs) as executor:
                args_iter = ((chunk, G, selector, preselection) for chunk in chunks)
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

        if n in forced_al_nodes:
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
            continue
        if n in forced_en_nodes:
            h_en[n] = 0
            continue

        try:
            glyph_enum = g if isinstance(g, Glyph) else Glyph(str(g))
        except ValueError:
            glyph_enum = None

        if glyph_enum is Glyph.AL:
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif glyph_enum is Glyph.EN:
            h_en[n] = 0


def _apply_selector(G: TNFRGraph) -> GlyphSelector:
    """Resolve the glyph selector callable configured on ``G``."""

    raw_selector = G.graph.get("glyph_selector")

    selector: GlyphSelector
    if isinstance(raw_selector, AbstractSelector):
        selector = raw_selector
    elif isinstance(raw_selector, type) and issubclass(raw_selector, AbstractSelector):
        selector_obj = cast(AbstractSelector, raw_selector())
        G.graph["glyph_selector"] = selector_obj
        selector = selector_obj
    elif raw_selector is None:
        selector = default_glyph_selector
    elif callable(raw_selector):
        selector = cast(GlyphSelector, raw_selector)
    else:
        selector = default_glyph_selector

    if isinstance(selector, ParametricGlyphSelector) or selector is parametric_glyph_selector:
        _selector_norms(G)
        _configure_selector_weights(G)
    return selector
