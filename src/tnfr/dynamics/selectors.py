"""Configured glyph policies over admitted stored metrics and histories.

Selection uses the declared Si/pressure/acceleration bands, score overrides and
hysteresis. These policies do not derive an autonomous selection law. Grammar
and operator live-state admission remain separate at execution.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, cast

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..config.parsing import parse_bool
from ..constants import get_param
from ..mathematics.unified_numerical import np, safe_divide
from ..metrics.common import merge_and_normalize_weights
from ..operators import apply_glyph
from ..operators.grammar_debt import (
    canonical_operator_code,
    read_grammar_state,
    require_grammar_state,
)
from ..selector import (
    _apply_selector_hysteresis,
    _calc_selector_score,
    _selector_margin,
    _selector_norms,
    _selector_parallel_jobs,
    _selector_thresholds,
)
from ..types import Glyph, GlyphCode, GlyphSelector, HistoryState, NodeId, TNFRGraph
from ..utils import clamp01, is_non_string_sequence, resolve_chunk_size
from ..validation import (
    GrammarContext,
    StructuralGrammarError,
    enforce_canonical_grammar,
    record_grammar_violation,
    soft_grammar_filters,
)
from ..validation.window import validate_window
from .aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_DSI, ALIAS_SI

# ---------------------------------------------------------------------------
# Score override thresholds for glyph selector
# ---------------------------------------------------------------------------
_SCORE_HIGH_OVERRIDE = 0.66
_SCORE_LOW_OVERRIDE = 0.33

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
    Si, dnfr, _ = _selector_normalized_metrics(nd, _selector_norms(G))
    return _default_selector_base_choice(Si, dnfr, thr)


def _default_selector_base_choice(
    Si: float, dnfr: float, thr: Mapping[str, float]
) -> GlyphCode:
    """One owner of the configured default selector's comparison order."""
    hi, lo, dnfr_hi = itemgetter("si_hi", "si_lo", "dnfr_hi")(thr)
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
    """Soft heuristic prefilter: repetition avoidance and force thresholds.

    Configured incremental U1a/U2/U3/U4 admission remains separate at the apply
    stage via :func:`enforce_canonical_grammar`. This soft policy neither
    validates a complete word nor certifies all U1-U6 conditions.
    """
    ctx = GrammarContext.from_graph(G)
    return cast(GlyphCode, soft_grammar_filters(ctx, n, cand))


def _selector_normalized_metrics(
    nd: Mapping[str, Any], norms: Mapping[str, float]
) -> tuple[float, float, float]:
    dnfr_max, acc_max = _selector_divisors(norms)
    si, pressure, acceleration = _selector_metric_inputs(nd)
    return (
        clamp01(si),
        float(safe_divide(abs(pressure), dnfr_max)),
        float(safe_divide(abs(acceleration), acc_max)),
    )


def _metric_value(nd: Mapping[str, Any], aliases, default: float) -> float:
    """Validate the authoritative value before alias fallback or array coercion."""
    return get_attr(
        nd,
        aliases,
        default,
        strict=True,
        conv=lambda value: finite_represented_real(value, aliases[0])[0],
    )


def _selector_metric_inputs(nd: Mapping[str, Any]) -> tuple[float, float, float]:
    # Missing-channel values are configured selector baselines, not observations
    # of equilibrium. Finite Si keeps its historical [0,1] saturation policy.
    return (
        _metric_value(nd, ALIAS_SI, 0.5),
        _metric_value(nd, ALIAS_DNFR, 0.0),
        _metric_value(nd, ALIAS_D2EPI, 0.0),
    )


def _selector_divisors(norms: Mapping[str, float]) -> tuple[float, float]:
    values = tuple(
        finite_represented_real(norms.get(key, 1.0), key)[0]
        for key in ("dnfr_max", "accel_max")
    )
    if any(value < 0.0 for value in values):
        raise ValueError("selector normalization maxima must be nonnegative")
    return values[0] or 1.0, values[1] or 1.0


def _selector_base_choice(
    Si: float, dnfr: float, accel: float, thr: Mapping[str, float]
) -> GlyphCode:
    si_hi, si_lo, dnfr_hi, acc_hi = itemgetter("si_hi", "si_lo", "dnfr_hi", "accel_hi")(
        thr
    )
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
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    grammar_state = read_grammar_state(nd)
    W = _configure_selector_weights(G) if weights is None else weights
    score = _calc_selector_score(Si, dnfr, accel, cast(Mapping[str, float], W))
    hist_prev = grammar_state.history
    if hist_prev and hist_prev[-1] == canonical_operator_code(cand):
        delta_si = _metric_value(nd, ALIAS_DSI, 0.0)
        h = G.graph.get("history", {})
        if not isinstance(h, Mapping):
            raise ValueError("selector history must be a mapping")
        sig = h.get("sense_sigma_mag", [])
        if not is_non_string_sequence(sig):
            raise ValueError("sense_sigma_mag must be a numeric history sequence")
        sigma_nonincreasing = (
            finite_represented_real(sig[-1], "sense_sigma_mag latest")[0]
            <= finite_represented_real(sig[-2], "sense_sigma_mag previous")[0]
            if len(sig) >= 2
            else True
        )
        if delta_si <= 0.0 and sigma_nonincreasing:
            score -= 0.05
    return float(score)


def _apply_score_override(
    cand: GlyphCode, score: float, dnfr: float, dnfr_lo: float
) -> GlyphCode:
    cand_key = str(cand)
    if score >= _SCORE_HIGH_OVERRIDE and cand_key in ("NAV", "RA", "ZHIR", "OZ"):
        return "IL"
    if score <= _SCORE_LOW_OVERRIDE and cand_key in ("NAV", "RA", "IL"):
        return "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def _parametric_selector_logic(G: TNFRGraph, n: NodeId) -> GlyphCode:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    margin = _selector_margin(G)
    weights = _configure_selector_weights(G)
    norms = _selector_norms(G)
    Si, dnfr, accel = _selector_normalized_metrics(nd, norms)

    cand = _selector_base_choice(Si, dnfr, accel, thr)

    hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thr, margin)
    if hist_cand is not None:
        return hist_cand

    score = _compute_selector_score(G, nd, Si, dnfr, accel, cand, weights=weights)

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
    weights: Mapping[str, float] | None = None


def _build_default_preselection(
    G: TNFRGraph, nodes: Sequence[NodeId]
) -> _SelectorPreselection:
    node_list = list(nodes)
    thresholds = _selector_thresholds(G)
    if not node_list:
        return _SelectorPreselection("default", {}, {}, thresholds=thresholds)

    norms = _selector_norms(G)
    n_jobs = _selector_parallel_jobs(G)
    metrics = _collect_selector_metrics(G, node_list, norms, n_jobs=n_jobs)
    base_choices = _compute_default_base_choices(metrics, thresholds)
    return _SelectorPreselection(
        "default", metrics, base_choices, thresholds=thresholds
    )


def _build_param_preselection(
    G: TNFRGraph, nodes: Sequence[NodeId]
) -> _SelectorPreselection:
    node_list = list(nodes)
    thresholds = _selector_thresholds(G)
    margin = _selector_margin(G)
    weights = _configure_selector_weights(G)
    if not node_list:
        return _SelectorPreselection(
            "param", {}, {}, thresholds=thresholds, margin=margin, weights=weights
        )

    norms = _selector_norms(G)
    n_jobs = _selector_parallel_jobs(G)
    metrics = _collect_selector_metrics(G, node_list, norms, n_jobs=n_jobs)
    base_choices = _compute_param_base_choices(metrics, thresholds, n_jobs)
    return _SelectorPreselection(
        "param",
        metrics,
        base_choices,
        thresholds=thresholds,
        margin=margin,
        weights=weights,
    )


class DefaultGlyphSelector(AbstractSelector):
    """Configured default heuristic with an explicit prepared-batch snapshot.

    Without prepare, each call reads current metrics and normalizers. A manual
    prepare remains frozen until the next prepare or clear; engine batches
    clear it on exit. Its graph reference survives process serialization and
    is released with the snapshot. Selection does not certify live admission.
    """

    __slots__ = ("_preselection", "_prepared_graph")

    def __init__(self) -> None:
        self._preselection: _SelectorPreselection | None = None
        self._prepared_graph: TNFRGraph | None = None

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None:
        """Precompute default selector metrics for ``nodes``."""

        self.clear()
        self._preselection = _build_default_preselection(graph, nodes)
        self._prepared_graph = graph

    def clear(self) -> None:
        """End the prepared batch so later calls read the current graph."""
        self._preselection = None
        self._prepared_graph = None

    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Return the configured glyph, using an explicitly prepared snapshot."""

        if self._prepared_graph is graph:
            preselection = self._preselection
        else:
            preselection = None
        return _resolve_preselected_glyph(
            graph, node, _default_selector_logic, preselection
        )


class ParametricGlyphSelector(AbstractSelector):
    """Configured score policy with the same explicit batch lifecycle as default."""

    __slots__ = ("_preselection", "_prepared_graph")

    def __init__(self) -> None:
        self._preselection: _SelectorPreselection | None = None
        self._prepared_graph: TNFRGraph | None = None

    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None:
        """Precompute parametric selector metrics and hysteresis thresholds."""

        self.clear()
        self._preselection = _build_param_preselection(graph, nodes)
        self._prepared_graph = graph

    def clear(self) -> None:
        """End the prepared batch so later calls read the current graph."""
        self._preselection = None
        self._prepared_graph = None

    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode:
        """Return the parametric glyph decision for ``node``."""

        if self._prepared_graph is graph:
            preselection = self._preselection
        else:
            preselection = None
        return _resolve_preselected_glyph(
            graph, node, _parametric_selector_logic, preselection
        )


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
    """Choose from supplied lag counters, then apply configured live grammar.

    This helper neither increments counters nor executes the resulting glyph.
    Forced AL/EN requests use the same grammar path as ordinary proposals.
    """

    use_canon = parse_bool(use_canon)
    al_max = _lag_count(al_max, "AL_MAX_LAG")
    en_max = _lag_count(en_max, "EN_MAX_LAG")
    al_lag = _lag_count(h_al[n], "since_AL")
    en_lag = _lag_count(h_en[n], "since_EN")
    require_grammar_state(G.nodes[n])
    if al_lag > al_max:
        g = Glyph.AL
    elif en_lag > en_max:
        g = Glyph.EN
    else:
        g = selector(G, n)
    if use_canon:
        try:
            admitted = enforce_canonical_grammar(G, n, g)
            if admitted != g:
                g = admitted
        except StructuralGrammarError as err:
            nd = G.nodes[n]
            history = read_grammar_state(nd).history
            selector_name = getattr(selector, "__name__", selector.__class__.__name__)
            err.attach_context(
                node=n, selector=selector_name, history=history, stage="selector"
            )
            record_grammar_violation(G, n, err, stage="selector")
            raise
    return g


def _selector_metrics_chunk(
    args: tuple[list[float], list[float], list[float], float, float],
) -> tuple[list[float], list[float], list[float]]:
    """Normalise metric chunk values for multiprocessing execution."""

    si_values, dnfr_values, accel_values, dnfr_max, accel_max = args
    si_seq = [clamp01(float(v)) for v in si_values]
    dnfr_seq = list(safe_divide([abs(v) for v in dnfr_values], dnfr_max))
    accel_seq = list(safe_divide([abs(v) for v in accel_values], accel_max))
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
    dnfr_max, accel_max = _selector_divisors(norms)
    inputs = [_selector_metric_inputs(G.nodes[node]) for node in nodes]
    si_values, dnfr_values, accel_values = (list(values) for values in zip(*inputs))

    if np is not None:
        si_seq_np = np.asarray(si_values, dtype=float)
        si_seq_np = np.clip(si_seq_np, 0.0, 1.0)
        dnfr_seq_np = safe_divide(
            np.abs(np.asarray(dnfr_values, dtype=float)), dnfr_max
        )
        accel_seq_np = safe_divide(
            np.abs(np.asarray(accel_values, dtype=float)), accel_max
        )

        si_seq = si_seq_np.tolist()
        dnfr_seq = dnfr_seq_np.tolist()
        accel_seq = accel_seq_np.tolist()
    else:
        worker_count = n_jobs if n_jobs is not None and n_jobs > 1 else None
        if worker_count is None:
            si_seq, dnfr_seq, accel_seq = _selector_metrics_chunk(
                (si_values, dnfr_values, accel_values, dnfr_max, accel_max)
            )
        else:
            approx_chunk = (
                math.ceil(len(nodes) / worker_count) if worker_count else None
            )
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
                executor_cls = getattr(
                    dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor
                )
            with executor_cls(max_workers=worker_count) as executor:
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
    return {
        node: _default_selector_base_choice(Si, dnfr, thresholds)
        for node, (Si, dnfr, _) in metrics.items()
    }


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
        executor_cls = getattr(
            dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor
        )
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

        cand = preselection.base_choices.get(n)
        if cand is None:
            cand = _selector_base_choice(Si, dnfr, accel, thresholds)

        nd = G.nodes[n]
        hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thresholds, margin)
        if hist_cand is not None:
            return hist_cand

        score = _compute_selector_score(
            G, nd, Si, dnfr, accel, cand, weights=preselection.weights
        )
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
    return [
        (n, _resolve_preselected_glyph(G, n, selector, preselection)) for n in nodes
    ]


def _apply_glyphs(G: TNFRGraph, selector: GlyphSelector, hist: HistoryState) -> None:
    """Apply one decision batch, then release built-in prepared observations."""
    try:
        _apply_glyph_batch(G, selector, hist)
    finally:
        if isinstance(selector, (DefaultGlyphSelector, ParametricGlyphSelector)):
            selector.clear()


@dataclass(frozen=True, slots=True)
class _GlyphBatchAdmission:
    """Fresh admitted inputs, valid only until the execution boundary changes."""

    window: int
    use_canon: bool
    al_max: int
    en_max: int
    nodes: tuple[NodeId, ...]
    h_al: MutableMapping[Any, int]
    h_en: MutableMapping[Any, int]
    next_al: dict[NodeId, int]
    next_en: dict[NodeId, int]


def _preflight_glyph_batch(G: TNFRGraph, hist: HistoryState) -> _GlyphBatchAdmission:
    """Read batch policy/context without preparing a selector or writing state.

    The runtime uses this before callbacks; the batch re-admits after callbacks
    and pressure refresh. No earlier snapshot authorizes a changed live state.
    """

    if not isinstance(hist, MutableMapping):
        raise TypeError("selector history must be a mutable mapping")
    window = validate_window(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
    canon = get_param(G, "GRAMMAR_CANON")
    if not isinstance(canon, Mapping):
        raise TypeError("GRAMMAR_CANON must be a mapping")
    use_canon = parse_bool(canon.get("enabled", False))
    al_max = _lag_count(get_param(G, "AL_MAX_LAG"), "AL_MAX_LAG")
    en_max = _lag_count(get_param(G, "EN_MAX_LAG"), "EN_MAX_LAG")
    finite_represented_real(G.graph.get("_t", 0.0), "selector event clock")

    nodes = tuple(G)
    for n in nodes:
        require_grammar_state(G.nodes[n])
    # Validate the complete consumed counter snapshot before updating any node.
    # Subsequent live operation failures remain sequential, not transactional.
    # HistoryDict.get can materialize bounded telemetry; admission is read-only.
    if isinstance(hist, dict):
        h_al = dict.get(hist, "since_AL", {})
        h_en = dict.get(hist, "since_EN", {})
    else:
        h_al = hist.get("since_AL", {})
        h_en = hist.get("since_EN", {})
    if not isinstance(h_al, MutableMapping) or not isinstance(h_en, MutableMapping):
        raise TypeError("since_AL and since_EN must be mutable mappings")
    next_al = {n: _lag_count(h_al.get(n, 0), "since_AL") + 1 for n in nodes}
    next_en = {n: _lag_count(h_en.get(n, 0), "since_EN") + 1 for n in nodes}
    return _GlyphBatchAdmission(
        window, use_canon, al_max, en_max, nodes, h_al, h_en, next_al, next_en
    )


def _apply_glyph_batch(
    G: TNFRGraph, selector: GlyphSelector, hist: HistoryState
) -> None:
    """Apply glyph decisions across the graph updating hysteresis trackers."""

    admission = _preflight_glyph_batch(G, hist)
    window, use_canon = admission.window, admission.use_canon
    al_max, en_max, nodes = admission.al_max, admission.en_max, admission.nodes
    h_al, h_en = admission.h_al, admission.h_en
    if isinstance(selector, AbstractSelector):
        selector.prepare(G, nodes)
        preselection: _SelectorPreselection | None = None
    else:
        preselection = _prepare_selector_preselection(G, selector, nodes)

    hist.setdefault("since_AL", h_al)
    hist.setdefault("since_EN", h_en)
    h_al.update(admission.next_al)
    h_en.update(admission.next_en)
    forced: dict[Any, str | Glyph] = {}
    to_select: list[Any] = []

    for n in nodes:
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
                decisions[n] = _resolve_preselected_glyph(G, n, selector, preselection)
        else:
            approx_chunk = math.ceil(len(to_select) / n_jobs) if n_jobs else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(to_select),
                minimum=1,
            )
            chunks = [
                to_select[idx : idx + chunk_size]
                for idx in range(0, len(to_select), chunk_size)
            ]
            dynamics_module = sys.modules.get("tnfr.dynamics")
            executor_cls = ProcessPoolExecutor
            if dynamics_module is not None:
                executor_cls = getattr(
                    dynamics_module, "ProcessPoolExecutor", ProcessPoolExecutor
                )
            with executor_cls(max_workers=n_jobs) as executor:
                args_iter = ((chunk, G, selector, preselection) for chunk in chunks)
                for results in executor.map(_glyph_proposal_worker, args_iter):
                    for node, glyph in results:
                        decisions[node] = glyph

    for n in nodes:
        g = decisions.get(n)
        if g is None:
            continue

        try:
            g = canonical_operator_code(g)
        except ValueError:
            # Preserve the configured grammar/dispatcher rejection path for
            # unknown proposals, including grammar's explicit fallback mode.
            pass

        mutation_abstention: dict[str, Any] | None = None
        if g == "ZHIR":
            from ..operators._mutation_gate import (
                mutation_trigger_certificate,
                validate_mutation_runtime_gate,
            )
            from ..operators.preconditions import OperatorPreconditionError

            node_data = G.nodes[n]
            certificate = mutation_trigger_certificate(node_data, G.graph)
            reason: str | None = None
            if not certificate.capacity_active:
                reason = "inactive_structural_capacity"
            elif not certificate.evidence_valid:
                reason = certificate.reason or "invalid_mutation_evidence"
            elif not certificate.threshold_gate_satisfied:
                reason = "observed_threshold_not_crossed"
            else:
                try:
                    validate_mutation_runtime_gate(node_data, G.graph)
                except OperatorPreconditionError as exc:
                    # A finite positive capacity can still sit below an
                    # explicitly configured operational floor. Invalid
                    # configuration is not an admissible abstention and must
                    # remain visible to the caller.
                    if "configured minimum=" not in exc.reason:
                        raise
                    reason = "below_configured_capacity_floor"

            if reason is not None:
                requested = g
                g = "IL"
                mutation_abstention = {
                    "node": n,
                    "time": finite_represented_real(
                        G.graph.get("_t", 0.0), "selector event clock"
                    )[0],
                    "requested_glyph": requested,
                    "applied_glyph": g,
                    "reason": reason,
                    "observed_depi_dt": certificate.observed_depi_dt,
                    "xi": certificate.xi,
                    "evidence_source": certificate.source,
                    "time_basis": certificate.time_basis,
                    "physical_time_resolved": (certificate.physical_time_resolved),
                }

        if use_canon:
            g = enforce_canonical_grammar(G, n, g)
        if mutation_abstention is not None:
            mutation_abstention["applied_glyph"] = g

        abstentions: list[dict[str, Any]] | None = None
        if mutation_abstention is not None:
            raw_abstentions = hist.get("mutation_abstentions")
            if raw_abstentions is None:
                abstentions = []
            elif isinstance(raw_abstentions, list):
                abstentions = raw_abstentions
            else:
                raise TypeError("history['mutation_abstentions'] must be a list")

        apply_glyph(G, n, g, window=window)

        if mutation_abstention is not None and abstentions is not None:
            if "mutation_abstentions" not in hist:
                hist["mutation_abstentions"] = abstentions
            abstentions.append(mutation_abstention)
            del abstentions[: -max(1, window)]

        if g == "AL":
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif g == "EN":
            h_en[n] = 0


def _lag_count(value: Any, label: str) -> int:
    """Admit a nonnegative operator count using the shared integer boundary."""
    try:
        return validate_window(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a nonnegative integer") from exc
    except ValueError as exc:
        raise ValueError(f"{label} must be a nonnegative integer") from exc


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
        raise TypeError(
            "glyph_selector must be a selector, selector class, callable or None"
        )
    return selector
