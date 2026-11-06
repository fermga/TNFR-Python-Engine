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
from ._mutation_occurrence import (
    COHERENCE_ABSTENTION,
    CONSUME_POLICY,
    SKIP_ABSTENTION,
    admit_native_mutation_occurrence,
    commit_native_mutation_interval,
    consumed_mutation_intervals,
    native_mutation_abstention_policy,
    native_mutation_occurrence_policy,
    preflight_native_mutation_occurrence,
)
from ._operator_form_history import (
    capture_native_form_history,
    finish_native_form_history,
)
from .aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_DSI, ALIAS_SI
from .native_epi_source import NativeEPIBatchObservation, _NativeEPIBatchCapture
from .native_operator_observation import (
    NativeOperatorDecision,
    NativeSelectorObservation,
)

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
    "NativeOperatorDecision",
    "NativeSelectorObservation",
    "apply_native_operator_batch",
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
    *,
    _acceleration_score=None,
) -> GlyphCode:
    """Soft heuristic prefilter: repetition avoidance and force thresholds.

    Configured incremental U1a/U2/U3/U4 admission remains separate at the apply
    stage via :func:`enforce_canonical_grammar`. This soft policy neither
    validates a complete word nor certifies all U1-U6 conditions.
    """
    ctx = GrammarContext.from_graph(G)
    if _acceleration_score is None:
        return cast(GlyphCode, soft_grammar_filters(ctx, n, cand))
    return cast(
        GlyphCode,
        soft_grammar_filters(ctx, n, cand, _acceleration_score=_acceleration_score),
    )


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
    acceleration_source: str = "solver_diagnostic"
    acceleration_window: tuple[Any, Any] | None = None


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
    G: TNFRGraph, nodes: Sequence[NodeId], *, _episode_selection_context=None
) -> _SelectorPreselection:
    node_list = list(nodes)
    episode = (
        None
        if _episode_selection_context is None
        else _episode_selection_context.admit_preparation(G, node_list)
    )
    thresholds = _selector_thresholds(G)
    margin = _selector_margin(G)
    weights = _configure_selector_weights(G)
    if not node_list:
        return _SelectorPreselection(
            "param", {}, {}, thresholds=thresholds, margin=margin, weights=weights
        )

    if episode is None:
        norms = _selector_norms(G)
        n_jobs = _selector_parallel_jobs(G)
        metrics = _collect_selector_metrics(G, node_list, norms, n_jobs=n_jobs)
    else:
        n_jobs = _selector_parallel_jobs(G)
        inputs = [
            (
                _metric_value(G.nodes[node], ALIAS_SI, 0.5),
                _metric_value(G.nodes[node], ALIAS_DNFR, 0.0),
                finite_represented_real(acceleration, "episode interior acceleration")[
                    0
                ],
            )
            for node, acceleration in zip(
                node_list, episode.exact_interior_accelerations, strict=True
            )
        ]
        norms = {
            "dnfr_max": max(abs(value[1]) for value in inputs),
            "accel_max": max(abs(value[2]) for value in inputs),
        }
        metrics = _collect_selector_metrics(
            G, node_list, norms, n_jobs=n_jobs, _inputs=inputs
        )
    base_choices = _compute_param_base_choices(metrics, thresholds, n_jobs)
    if episode is not None:
        _episode_selection_context.finish_preparation(G)
    return _SelectorPreselection(
        "param",
        metrics,
        base_choices,
        thresholds=thresholds,
        margin=margin,
        weights=weights,
        acceleration_source=(
            "solver_diagnostic" if episode is None else "emission_interior"
        ),
        acceleration_window=(
            None
            if episode is None
            else (episode.exact_start_time, episode.exact_end_time)
        ),
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

    def select(
        self, graph: TNFRGraph, node: NodeId, *, _selection_details=None
    ) -> GlyphCode:
        """Return the configured glyph, using an explicitly prepared snapshot."""

        if self._prepared_graph is graph:
            preselection = self._preselection
        else:
            preselection = None
        if _selection_details is None:
            return _resolve_preselected_glyph(
                graph, node, _default_selector_logic, preselection
            )
        return _resolve_preselected_glyph(
            graph,
            node,
            _default_selector_logic,
            preselection,
            _selection_details=_selection_details,
        )


class ParametricGlyphSelector(AbstractSelector):
    """Configured score policy with the same explicit batch lifecycle as default."""

    __slots__ = ("_preselection", "_prepared_graph")

    def __init__(self) -> None:
        self._preselection: _SelectorPreselection | None = None
        self._prepared_graph: TNFRGraph | None = None

    def prepare(
        self,
        graph: TNFRGraph,
        nodes: Sequence[NodeId],
        *,
        _episode_selection_context=None,
    ) -> None:
        """Precompute parametric selector metrics and hysteresis thresholds."""

        self.clear()
        if _episode_selection_context is None:
            self._preselection = _build_param_preselection(graph, nodes)
        else:
            from ..operators.emission_observation import _EmissionSelectionContext

            if type(_episode_selection_context) is not _EmissionSelectionContext:
                raise TypeError(
                    "typed selection requires its private actual episode context"
                )
            self._preselection = _build_param_preselection(
                graph, nodes, _episode_selection_context=_episode_selection_context
            )
        self._prepared_graph = graph

    def clear(self) -> None:
        """End the prepared batch so later calls read the current graph."""
        self._preselection = None
        self._prepared_graph = None

    def select(
        self, graph: TNFRGraph, node: NodeId, *, _selection_details=None
    ) -> GlyphCode:
        """Return the parametric glyph decision for ``node``."""

        if self._prepared_graph is graph:
            preselection = self._preselection
        else:
            preselection = None
        if _selection_details is None:
            return _resolve_preselected_glyph(
                graph, node, _parametric_selector_logic, preselection
            )
        return _resolve_preselected_glyph(
            graph,
            node,
            _parametric_selector_logic,
            preselection,
            _selection_details=_selection_details,
        )


default_glyph_selector = DefaultGlyphSelector()
parametric_glyph_selector = ParametricGlyphSelector()

# Only the unchanged builtin method family supports branch-level provenance.
# Subclasses and replaced hooks continue through their actual callable once.
_BUILTIN_SELECTOR_BINDINGS = {
    DefaultGlyphSelector: (
        DefaultGlyphSelector.prepare,
        DefaultGlyphSelector.select,
        AbstractSelector.__call__,
        "default",
    ),
    ParametricGlyphSelector: (
        ParametricGlyphSelector.prepare,
        ParametricGlyphSelector.select,
        AbstractSelector.__call__,
        "parametric",
    ),
}


def _native_selector_kind(selector: GlyphSelector) -> str:
    bindings = _BUILTIN_SELECTOR_BINDINGS.get(type(selector))
    if bindings is not None:
        # AbstractSelector supplies an instance dictionary despite the builtin
        # subclasses' slots. A supplied bound hook must keep its old signature
        # and must not inherit builtin provenance from the class underneath.
        supplied = object.__getattribute__(selector, "__dict__")
        if "select" in supplied or "prepare" in supplied:
            return "opaque"
    if bindings is not None and (
        type(selector).prepare is bindings[0]
        and type(selector).select is bindings[1]
        and type(selector).__call__ is bindings[2]
    ):
        return bindings[3]
    return "opaque"


def _resolve_observed_glyph(G, n, selector, preselection, details):
    """Observe the actual builtin prepared branch, or one opaque invocation."""
    kind = _native_selector_kind(selector)
    details["kind"] = kind
    if kind != "opaque":
        return selector.select(G, n, _selection_details=details)
    return _resolve_preselected_glyph(G, n, selector, preselection)


def _glyph_observation_worker(args):
    nodes, G, selector, preselection = args
    result = []
    for n in nodes:
        details: dict[str, Any] = {}
        glyph = _resolve_observed_glyph(G, n, selector, preselection, details)
        result.append((n, glyph, NativeSelectorObservation(**details, selected=glyph)))
    return result


def apply_native_operator_batch(
    graph: TNFRGraph,
    history: HistoryState,
    *,
    decision_trace: list[NativeOperatorDecision] | None = None,
    epi_trace: list[NativeEPIBatchObservation] | None = None,
) -> None:
    """Apply one existing all-node selector batch without advancing time.

    Selection consumes configured stored metrics, histories and lag memory.
    An exact optional list receives passive observations even when a later
    target fails. Existing sequential effects are not rolled back. This is
    direct-glyph execution, not the public operator-class lifecycle.

    ``NATIVE_ZHIR_OCCURRENCE_POLICY='consume_timestamped_interval'`` opts into
    one returned native ZHIR action per retained own source/time pair. Missing
    policy or ``'level'`` retains the existing repeatable threshold trigger.

    ``NATIVE_ZHIR_ABSTENTION_POLICY='skip'`` leaves a gate-blocked request without
    a replacement glyph. Missing policy or ``'coherence'`` retains the real IL
    substitution. A skipped request advances selector lags, not operator
    history or receipts; other nodes' actual actions can still affect it.

    An exact ``epi_trace`` list receives one frozen whole-population endpoint
    account, including a failed prefix. It must differ from ``decision_trace``.
    Capture is optional, observes actual direct dispatch and supplies no growth
    rate or permission. Unsupported captured state leaves totals unavailable
    without changing the native operation's admission.
    """
    _apply_native_operator_batch(
        graph, history, decision_trace=decision_trace, epi_trace=epi_trace
    )


def _apply_native_operator_batch(
    graph,
    history,
    *,
    decision_trace=None,
    epi_trace=None,
    _source_guard=None,
    _episode_selection_context=None,
):
    """Shared native execution; only a live producer supplies the private guard."""
    _validate_decision_trace(decision_trace)
    if epi_trace is not None and type(epi_trace) is not list:
        raise TypeError("epi_trace must be an exact list or None")
    if epi_trace is not None and epi_trace is decision_trace:
        raise ValueError("epi_trace and decision_trace must be distinct lists")
    capture = None if epi_trace is None else _NativeEPIBatchCapture(graph)
    failure = None
    try:
        if _source_guard is not None:
            _source_guard.require_bindings()
        if native_mutation_occurrence_policy(graph.graph) == CONSUME_POLICY:
            # A supplied selector class can execute its constructor on resolution.
            # Admit consumed memory before that hook as well as before preparation.
            _preflight_glyph_batch(graph, history)
        selector = _apply_selector(graph)
        if _source_guard is not None:
            _source_guard.require_bindings(selector)
        if capture is not None or _episode_selection_context is not None:
            _apply_glyphs(
                graph,
                selector,
                history,
                _decision_trace=decision_trace,
                _epi_capture=capture,
                _source_guard=_source_guard,
                _episode_selection_context=_episode_selection_context,
            )
        elif decision_trace is None:
            _apply_glyphs(graph, selector, history)
        else:
            _apply_glyphs(graph, selector, history, _decision_trace=decision_trace)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if capture is not None:
            try:
                epi_trace.append(capture.finish(failure))
            except BaseException:
                if failure is None:
                    raise
                _note_epi_capture_failure(failure)


def _note_epi_capture_failure(
    failure: BaseException,
    note: str = "Native EPI capture also failed; original error preserved.",
) -> None:
    """Keep an original execution error if its secondary observation fails."""
    try:
        add_note = getattr(BaseException, "add_note", None)
        if add_note is not None:
            add_note(failure, note)
    except BaseException:
        # Even a failed diagnostic note must not replace the execution error.
        pass


def _validate_decision_trace(trace) -> None:
    if trace is not None and type(trace) is not list:
        raise TypeError("decision_trace must be an exact list or None")


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
    *,
    _inputs=None,
) -> dict[Any, tuple[float, float, float]]:
    """Return normalised (Si, ΔNFR, acceleration) triples for ``nodes``."""

    if not nodes:
        return {}

    dynamics_module = sys.modules.get("tnfr.dynamics")
    dnfr_max, accel_max = _selector_divisors(norms)
    inputs = (
        [_selector_metric_inputs(G.nodes[node]) for node in nodes]
        if _inputs is None
        else _inputs
    )
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
    *,
    _selection_details: dict[str, Any] | None = None,
) -> GlyphCode:
    """Return glyph for ``n`` using ``preselection`` shortcuts when possible."""

    if preselection is None:
        return selector(G, n)

    metrics = preselection.metrics.get(n)
    if metrics is None:
        return selector(G, n)

    if preselection.kind == "default":
        cand = preselection.base_choices.get(n)
        if _selection_details is not None:
            _selection_details.update(
                kind="default",
                normalized_metrics=metrics,
                thresholds=tuple((preselection.thresholds or {}).items()),
                base_candidate=cand,
                acceleration_source=preselection.acceleration_source,
                acceleration_window=preselection.acceleration_window,
            )
        return cand if cand is not None else selector(G, n)

    if preselection.kind == "param":
        Si, dnfr, accel = metrics
        thresholds = preselection.thresholds or _selector_thresholds(G)
        margin: float | None = preselection.margin

        cand = preselection.base_choices.get(n)
        if cand is None:
            cand = _selector_base_choice(Si, dnfr, accel, thresholds)
        if _selection_details is not None:
            _selection_details.update(
                kind="parametric",
                normalized_metrics=metrics,
                thresholds=tuple(thresholds.items()),
                base_candidate=cand,
                margin=margin,
                acceleration_source=preselection.acceleration_source,
                acceleration_window=preselection.acceleration_window,
            )

        nd = G.nodes[n]
        hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thresholds, margin)
        if _selection_details is not None:
            _selection_details["hysteresis_candidate"] = hist_cand
        if hist_cand is not None:
            return hist_cand

        score = _compute_selector_score(
            G, nd, Si, dnfr, accel, cand, weights=preselection.weights
        )
        cand = _apply_score_override(cand, score, dnfr, thresholds["dnfr_lo"])
        if _selection_details is not None:
            _selection_details.update(score=score, override_candidate=cand)
        if preselection.acceleration_source == "emission_interior":
            cand = _soft_grammar_prefilter(G, n, cand, _acceleration_score=accel)
        else:
            cand = _soft_grammar_prefilter(G, n, cand)
        if _selection_details is not None:
            _selection_details["soft_filter_candidate"] = cand
        return cand

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


def _apply_glyphs(
    G: TNFRGraph,
    selector: GlyphSelector,
    hist: HistoryState,
    *,
    _decision_trace: list[NativeOperatorDecision] | None = None,
    _epi_capture: _NativeEPIBatchCapture | None = None,
    _source_guard=None,
    _episode_selection_context=None,
) -> None:
    """Apply one decision batch, then release built-in prepared observations."""
    _validate_decision_trace(_decision_trace)
    source_failure = None
    try:
        if _epi_capture is not None or _episode_selection_context is not None:
            _apply_glyph_batch(
                G,
                selector,
                hist,
                _decision_trace=_decision_trace,
                _epi_capture=_epi_capture,
                _source_guard=_source_guard,
                _episode_selection_context=_episode_selection_context,
            )
        elif _decision_trace is None:
            _apply_glyph_batch(G, selector, hist)
        else:
            _apply_glyph_batch(G, selector, hist, _decision_trace=_decision_trace)
    except BaseException as failure:
        source_failure = failure
        raise
    finally:
        if _source_guard is None:
            if isinstance(selector, (DefaultGlyphSelector, ParametricGlyphSelector)):
                selector.clear()
        else:
            try:
                _source_guard.require_bindings(selector)
                selector.clear()
            except BaseException:
                if source_failure is None:
                    raise
                _note_epi_capture_failure(
                    source_failure,
                    "Native source finalization also failed; original error preserved.",
                )


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

    The runtime uses this before callbacks; the batch re-admits after callbacks,
    pressure refresh and selector hooks. No earlier snapshot authorizes a
    changed live admission state. This reader never advances lag counters.
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
    native_mutation_abstention_policy(G.graph)
    preflight_native_mutation_occurrence(G, nodes)
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


def _readmit_glyph_batch(
    G: TNFRGraph, hist: HistoryState, targets: tuple[NodeId, ...]
) -> _GlyphBatchAdmission:
    """Admit live hook output while retaining the original ordered targets.

    Selectors may prepare diagnostics or change a supplied policy; they cannot
    authorize execution using an earlier admission snapshot. A changed target
    population requires another batch. Hook effects are not rolled back here.
    """
    admission = _preflight_glyph_batch(G, hist)
    if len(admission.nodes) != len(targets) or any(
        actual is not expected
        for actual, expected in zip(admission.nodes, targets, strict=True)
    ):
        raise ValueError("selector hooks must preserve the ordered batch targets")
    return admission


def _apply_glyph_batch(
    G: TNFRGraph,
    selector: GlyphSelector,
    hist: HistoryState,
    *,
    _decision_trace: list[NativeOperatorDecision] | None = None,
    _epi_capture: _NativeEPIBatchCapture | None = None,
    _source_guard=None,
    _episode_selection_context=None,
) -> None:
    """Apply glyph decisions across the graph updating hysteresis trackers."""

    _validate_decision_trace(_decision_trace)
    admission = _preflight_glyph_batch(G, hist)
    if _decision_trace is None:
        _run_glyph_batch(
            G,
            selector,
            hist,
            admission,
            None,
            _epi_capture,
            _source_guard,
            _episode_selection_context,
        )
        return
    observation = _NativeBatchObservation(
        admission.nodes, _native_selector_kind(selector)
    )
    try:
        _run_glyph_batch(
            G,
            selector,
            hist,
            admission,
            observation,
            _epi_capture,
            _source_guard,
            _episode_selection_context,
        )
    except Exception as exc:
        observation.failed(exc)
        raise
    finally:
        _decision_trace.extend(
            NativeOperatorDecision(**fields) for fields in observation.fields.values()
        )


class _NativeBatchObservation:
    """Mutable local construction only; no graph storage or executable hooks."""

    def __init__(self, nodes, kind):
        self.fields = {
            node: dict(node=node, index=index, selector_kind=kind)
            for index, node in enumerate(nodes)
        }
        self.active = None
        self.stage = "prepare"

    def boundary(self, stage, node=None):
        self.stage, self.active = stage, node

    def failed(self, exc):
        try:
            message = str(exc)
        except Exception:
            # Observation must not replace the original execution exception.
            message = "<exception message unavailable>"
        failure = dict(
            failure_stage=self.stage,
            failure_type=type(exc).__name__,
            failure_message=message,
        )
        if self.active is not None:
            self.fields[self.active].update(status="failed", **failure)
        else:
            # A shared preparation/worker/admission failure has no authenticated
            # target attribution. Retain all targets as not reached.
            for fields in self.fields.values():
                fields.update(**failure)


def _record_mutation_abstention(history, abstentions, record, window, *, bind_sink):
    """Bind the admitted sink and append telemetry, including no-action records."""
    if bind_sink:
        history["mutation_abstentions"] = abstentions
    abstentions.append(record)
    del abstentions[: -max(1, window)]


def _run_glyph_batch(
    G,
    selector,
    hist,
    admission,
    observation,
    epi_capture=None,
    source_guard=None,
    episode_selection_context=None,
):
    """Shared batch body; observation never selects or executes independently."""
    nodes = admission.nodes
    if episode_selection_context is not None:
        from ..operators.emission_observation import (
            _require_episode_parametric_selector,
        )

        _require_episode_parametric_selector(G, selector)
        selector.prepare(G, nodes, _episode_selection_context=episode_selection_context)
        preselection = None
    elif isinstance(selector, AbstractSelector):
        selector.prepare(G, nodes)
        preselection: _SelectorPreselection | None = None
    else:
        preselection = _prepare_selector_preselection(G, selector, nodes)

    if observation is not None:
        observation.boundary("post_prepare_admission")
    admission = _readmit_glyph_batch(G, hist, nodes)
    al_max, en_max = admission.al_max, admission.en_max
    h_al, h_en = admission.h_al, admission.h_en
    hist.setdefault("since_AL", h_al)
    hist.setdefault("since_EN", h_en)
    h_al.update(admission.next_al)
    h_en.update(admission.next_en)
    forced: dict[Any, str | Glyph] = {}
    to_select: list[Any] = []

    for n in nodes:
        if observation is not None:
            observation.fields[n].update(
                since_al=h_al[n],
                since_en=h_en[n],
                al_max=al_max,
                en_max=en_max,
            )
        if h_al[n] > al_max:
            forced[n] = Glyph.AL
            if observation is not None:
                observation.fields[n].update(
                    lag_reason="al_lag", requested_glyph="AL", selector_evaluated=False
                )
        elif h_en[n] > en_max:
            forced[n] = Glyph.EN
            if observation is not None:
                observation.fields[n].update(
                    lag_reason="en_lag", requested_glyph="EN", selector_evaluated=False
                )
        else:
            to_select.append(n)

    decisions: dict[Any, str | Glyph] = dict(forced)
    if to_select:
        if observation is not None:
            observation.boundary("selection")
        n_jobs = _selector_parallel_jobs(G)
        if n_jobs is None:
            for n in to_select:
                if observation is None:
                    decisions[n] = _resolve_preselected_glyph(
                        G, n, selector, preselection
                    )
                else:
                    observation.boundary("selection", n)
                    observation.fields[n]["selector_evaluated"] = True
                    details: dict[str, Any] = {"kind": "opaque"}
                    try:
                        decisions[n] = _resolve_observed_glyph(
                            G, n, selector, preselection, details
                        )
                    finally:
                        observation.fields[n]["selection"] = NativeSelectorObservation(
                            **details, selected=decisions.get(n)
                        )
                    observation.fields[n]["requested_glyph"] = decisions[n]
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
                worker = (
                    _glyph_proposal_worker
                    if observation is None
                    else _glyph_observation_worker
                )
                for results in executor.map(worker, args_iter):
                    for result in results:
                        node, glyph = result[:2]
                        decisions[node] = glyph
                        if observation is not None:
                            observation.fields[node].update(
                                selector_evaluated=True,
                                selection=result[2],
                                requested_glyph=glyph,
                            )

    # The proposals and forced-lag choices retain their selection boundary.
    # Admission consumes the live post-selection context without replaying any
    # selector or incrementing its lag counters for a second time.
    if observation is not None:
        observation.boundary("post_selection_admission")
    if source_guard is not None:
        source_guard.require_bindings(selector)
    admission = _readmit_glyph_batch(G, hist, nodes)
    window, use_canon = admission.window, admission.use_canon
    al_max, en_max = admission.al_max, admission.en_max
    h_al, h_en = admission.h_al, admission.h_en

    for n in nodes:
        if observation is not None:
            observation.boundary("request", n)
        abstention_policy = native_mutation_abstention_policy(G.graph)
        occurrence_policy = native_mutation_occurrence_policy(G.graph)
        if occurrence_policy == CONSUME_POLICY:
            consumed_mutation_intervals(G.nodes[n])
        if observation is not None:
            observation.fields[n]["mutation_abstention_policy"] = abstention_policy
            observation.fields[n]["mutation_occurrence_policy"] = occurrence_policy
        occurrence = None
        g = decisions.get(n)
        if g is None:
            if observation is not None:
                observation.fields[n]["status"] = "skipped"
            continue

        try:
            g = canonical_operator_code(g)
        except ValueError:
            # Preserve the configured grammar/dispatcher rejection path for
            # unknown proposals, including grammar's explicit fallback mode.
            pass

        if observation is not None:
            observation.fields[n]["requested_glyph"] = g
            observation.boundary("mutation_gate", n)

        mutation_abstention: dict[str, Any] | None = None
        if g == "ZHIR":
            from ..operators._mutation_gate import (
                _read_mutation_capacity,
                mutation_trigger_certificate,
            )

            node_data = G.nodes[n]
            certificate = mutation_trigger_certificate(node_data, G.graph)
            if observation is not None:
                observation.fields[n]["mutation_certificate"] = certificate
            # Capacity and its configured minimum are consumed even when
            # another trigger condition already blocks the request. Invalid
            # authoritative values are errors, not a reason to substitute IL
            # or skip. Preserve the legacy precedence of valid blockers.
            nu_f, minimum_nu_f = _read_mutation_capacity(node_data, G.graph)
            reason: str | None = None
            if nu_f <= 0.0:
                reason = "inactive_structural_capacity"
            elif not certificate.evidence_valid:
                reason = certificate.reason or "invalid_mutation_evidence"
            elif not certificate.threshold_gate_satisfied:
                reason = "observed_threshold_not_crossed"
            elif nu_f < minimum_nu_f:
                reason = "below_configured_capacity_floor"

            if reason is None:
                occurrence = admit_native_mutation_occurrence(
                    node_data, certificate, occurrence_policy
                )
                reason = occurrence.reason
                if observation is not None:
                    observation.fields[n].update(
                        mutation_interval=occurrence.interval,
                        mutation_interval_previously_consumed=(
                            occurrence.previously_consumed
                        ),
                    )

            if reason is not None:
                requested = g
                g = "IL" if abstention_policy == COHERENCE_ABSTENTION else None
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

            if observation is not None:
                observation.fields[n]["mutation_abstention_reason"] = reason

        if observation is not None:
            observation.fields[n]["post_gate_glyph"] = g
            observation.boundary("grammar", n)
        if g is not None and use_canon:
            if observation is None:
                g = enforce_canonical_grammar(G, n, g)
            else:
                grammar_trace = []
                try:
                    g = enforce_canonical_grammar(
                        G, n, g, _decision_trace=grammar_trace
                    )
                finally:
                    if grammar_trace:
                        observation.fields[n]["grammar_decision"] = grammar_trace[-1]
        if mutation_abstention is not None:
            mutation_abstention["applied_glyph"] = g

        abstentions: list[dict[str, Any]] | None = None
        bind_abstentions = False
        if observation is not None:
            observation.boundary("abstention_admission", n)
        if mutation_abstention is not None:
            raw_abstentions = hist.get("mutation_abstentions")
            if raw_abstentions is None:
                abstentions = []
                bind_abstentions = True
            elif isinstance(raw_abstentions, list):
                abstentions = raw_abstentions
            else:
                raise TypeError("history['mutation_abstentions'] must be a list")

        if mutation_abstention is not None and abstention_policy == SKIP_ABSTENTION:
            # A blocked request under the explicit skip policy has no operator
            # occurrence. In particular, do not resolve grammar, dispatch a
            # fallback, clean form history or consume a Mutation interval.
            _record_mutation_abstention(
                hist,
                abstentions,
                mutation_abstention,
                window,
                bind_sink=bind_abstentions,
            )
            if observation is not None:
                observation.fields[n]["status"] = "abstained"
            continue

        # Grammar can replace a request. Bind occurrence to the actual final
        # ZHIR and fresh live evidence without replaying grammar. A new blocker
        # here rejects; it cannot authorize an unadmitted replacement glyph.
        native_mutation_abstention_policy(G.graph)
        occurrence_policy = native_mutation_occurrence_policy(G.graph)
        if observation is not None:
            observation.fields[n]["mutation_occurrence_policy"] = occurrence_policy
        if g == "ZHIR" and occurrence_policy == CONSUME_POLICY:
            if observation is not None:
                observation.boundary("mutation_occurrence", n)
            from ..operators._mutation_gate import validate_mutation_runtime_gate
            from ..operators.preconditions import OperatorPreconditionError

            runtime_gate = validate_mutation_runtime_gate(G.nodes[n], G.graph)
            certificate = runtime_gate.threshold.certificate
            occurrence = admit_native_mutation_occurrence(
                G.nodes[n], certificate, occurrence_policy
            )
            if observation is not None:
                observation.fields[n].update(
                    mutation_certificate=certificate,
                    mutation_interval=occurrence.interval,
                    mutation_interval_previously_consumed=(
                        occurrence.previously_consumed
                    ),
                )
            if occurrence.reason is not None:
                raise OperatorPreconditionError("Mutation", occurrence.reason)
        else:
            occurrence = None

        if observation is not None:
            observation.boundary("form_history_admission", n)
        form_before = capture_native_form_history(G.nodes[n], g)
        epi_before = None if epi_capture is None else epi_capture.before_dispatch()
        if source_guard is not None:
            source_guard.require_bindings(selector)
        if observation is not None:
            observation.boundary("dispatch", n)
        try:
            apply_glyph(G, n, g, window=window)
        except BaseException as dispatch_error:
            if epi_capture is not None:
                try:
                    epi_capture.after_dispatch(n, g, epi_before, dispatch_error)
                except BaseException:
                    epi_capture.coverage_failed = True
                    _note_epi_capture_failure(dispatch_error)
            # Native dispatch is sequential, not transactional. A partial
            # form jump must not leave an old continuous secant available.
            try:
                invalidated = finish_native_form_history(G.nodes[n], form_before)
                if observation is not None:
                    observation.fields[n]["physical_history_invalidated"] = invalidated
            except BaseException:
                # External mapping/dispatcher failures can prevent cleanup.
                # Preserve the original error rather than imply rollback.
                try:
                    dispatch_error.add_note("Native physical history cleanup failed.")
                except BaseException:
                    pass
            raise
        if observation is not None:
            observation.fields[n]["applied_glyph"] = g
        epi_capture_error = None
        if epi_capture is not None:
            if observation is not None:
                observation.boundary("epi_capture", n)
            try:
                epi_capture.after_dispatch(n, g, epi_before)
            except BaseException as capture_error:
                epi_capture.coverage_failed = True
                # Finish the real returned action's ordinary lifecycle before
                # surfacing a secondary observer error. In particular, stale
                # samples must be cleared and successful interval use recorded.
                # An actual cleanup/bookkeeping error retains its own priority.
                epi_capture_error = capture_error
        if observation is not None:
            observation.boundary("form_history_cleanup", n)
        invalidated = finish_native_form_history(G.nodes[n], form_before)
        if observation is not None:
            observation.fields[n]["physical_history_invalidated"] = invalidated
            observation.boundary("post_dispatch", n)
        if occurrence is not None:
            commit_native_mutation_interval(G.nodes[n], occurrence)
            if observation is not None:
                observation.fields[n]["mutation_interval_committed"] = True

        if mutation_abstention is not None and abstentions is not None:
            _record_mutation_abstention(
                hist,
                abstentions,
                mutation_abstention,
                window,
                bind_sink=bind_abstentions,
            )

        if g == "AL":
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif g == "EN":
            h_en[n] = 0
        if observation is not None:
            observation.fields[n]["status"] = "applied"
        if epi_capture_error is not None:
            if observation is not None:
                observation.boundary("epi_capture", n)
            raise epi_capture_error
        if source_guard is not None:
            source_guard.require_bindings(selector)


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

    # A configured selector class can execute a constructor. Reject malformed
    # abstention policy before that hook even under level occurrence policy.
    native_mutation_abstention_policy(G.graph)
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


# The restricted live producer checks these bindings without changing the
# ordinary native API's support for supplied selectors and dispatch functions.
_NATIVE_SOURCE_BINDINGS = tuple(
    (name, globals()[name])
    for name in (
        "apply_native_operator_batch",
        "_apply_native_operator_batch",
        "_apply_glyphs",
        "_apply_glyph_batch",
        "_run_glyph_batch",
        "_apply_selector",
        "apply_glyph",
    )
)
_NATIVE_SOURCE_SELECTOR_BINDINGS = tuple(
    (
        kind,
        tuple(
            (name, getattr(kind, name))
            for name in ("__init__", "prepare", "select", "__call__", "clear")
        ),
    )
    for kind in (DefaultGlyphSelector, ParametricGlyphSelector)
)
