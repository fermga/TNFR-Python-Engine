"""Restricted live native source followed by an optional canonical phase response.

The supplied order is held-form flow, one real sequential native batch, then
eligible Mutation. Source failures retain the native prefix; only the response
is transactional. Returned records are observations, never activation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
from numbers import Integral
from typing import Any

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..config.parsing import parse_bool
from ..constants.aliases import ALIAS_DNFR
from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._argument_validation import validate_common_execution_arguments
from ._mutation_gate import _read_mutation_capacity
from .event_runtime import OperatorEventExecutionResult, execute_operator_event_schedule
from .event_timing import OperatorEventSchedule, build_operator_event_schedule
from .network_stage import GraphTransactionSnapshot, NetworkStageResult
from .operator_growth import observe_operator_epi_growth
from .operator_growth_mutation import (
    OperatorGrowthMutationCandidate,
    _execute_growth_stage,
    _prepare_growth_request,
    _select_growth_candidates,
    _watermarks,
)

__all__ = (
    "NativePositiveWindowSource",
    "NativePositiveWindowGrowth",
    "NativePositiveWindowDispatch",
    "execute_native_positive_window",
)

_SOURCE = "native_positive_window_net_growth"
_TIME_BASIS = "native_positive_window_time"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class NativePositiveWindowSource:
    """Actual separate flow and native records; construction grants no authority."""

    flow_result: OperatorEventExecutionResult
    native_batch: Any
    native_decisions: tuple[Any, ...]
    initial: Any
    flow_endpoint: Any

    @property
    def final_time(self) -> float:
        return self.flow_result.final_time


@dataclass(frozen=True, slots=True)
class NativePositiveWindowGrowth:
    """Exact represented endpoints over the producer's actual positive interval."""

    nodes: tuple[Any, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    exact_epi_before: tuple[Fraction, ...]
    exact_epi_after: tuple[Fraction, ...]
    exact_flow_increment: tuple[Fraction, ...]
    exact_jump_increment: tuple[Fraction, ...]
    exact_jump_variation: tuple[Fraction, ...]
    exact_net_increment: tuple[Fraction, ...]
    exact_net_growth_rate: tuple[Fraction, ...]
    available: bool = field(default=True, init=False)
    unavailability_reasons: tuple[str, ...] = field(default=(), init=False)
    source: str = field(default=_SOURCE, init=False)


@dataclass(frozen=True, slots=True)
class NativePositiveWindowDispatch:
    """Completed native source and selected response, or explicit abstention."""

    source_result: NativePositiveWindowSource
    growth: NativePositiveWindowGrowth
    candidates: tuple[Any, ...]
    decisions: tuple[OperatorGrowthMutationCandidate, ...]
    selected_targets: tuple[Any, ...]
    stage_result: NetworkStageResult | None

    @property
    def executed(self) -> bool:
        return self.stage_result is not None


class _NativeSourceBinding:
    """Call-local binding to the unchanged native execution family and owner."""

    def __init__(self, graph, history):
        self.graph = graph
        self.history = history
        self.history_bound = graph.graph.get("history", _MISSING) is history
        self.nodes = tuple(graph.nodes)
        self.storage = tuple(graph.nodes[node] for node in self.nodes)

    def require_bindings(self, selector=None):
        from .. import operators
        from ..dynamics import selectors

        for name, expected in selectors._NATIVE_SOURCE_BINDINGS:
            if getattr(selectors, name) is not expected:
                raise ValueError("native positive-window dispatcher binding changed")
        for name, expected in operators._NATIVE_SOURCE_DISPATCH_BINDINGS:
            if getattr(operators, name) is not expected:
                raise ValueError("native direct dispatcher binding changed")
        if type(operators.GLYPH_OPERATIONS) is not dict or any(
            operators.GLYPH_OPERATIONS.get(glyph) is not operation
            for glyph, operation in operators._NATIVE_SOURCE_GLYPH_BINDINGS
        ):
            raise ValueError("native positive-window operator registry changed")
        if selectors.apply_glyph is not operators.apply_glyph:
            raise ValueError("native positive-window dispatcher is not canonical")
        for kind, bindings in selectors._NATIVE_SOURCE_SELECTOR_BINDINGS:
            if any(getattr(kind, name) is not value for name, value in bindings):
                raise ValueError("native builtin selector binding changed")
        configured = self.graph.graph.get("glyph_selector")
        allowed_types = tuple(
            kind for kind, _ in selectors._NATIVE_SOURCE_SELECTOR_BINDINGS
        )
        if configured is None:
            configured = selectors.default_glyph_selector
        if any(configured is kind for kind in allowed_types):
            if selector is None:
                self._require_owner()
                return
        elif type(configured) not in allowed_types:
            raise TypeError(
                "native positive-window requires an unchanged builtin selector"
            )
        actual = configured if selector is None else selector
        if (
            type(actual) not in allowed_types
            or selectors._native_selector_kind(actual) == "opaque"
        ):
            raise TypeError(
                "native positive-window requires an unchanged builtin selector"
            )
        namespace = object.__getattribute__(actual, "__dict__")
        if any(
            name in namespace for name in ("prepare", "select", "__call__", "clear")
        ):
            raise ValueError(
                "native positive-window selector instance overrides a hook"
            )
        self._require_owner()

    def _require_owner(self):
        actual = tuple(self.graph.nodes)
        if len(actual) != len(self.nodes) or any(
            node is not old or self.graph.nodes[node] is not data
            for node, old, data in zip(actual, self.nodes, self.storage, strict=True)
        ):
            raise ValueError(
                "native positive-window node population or storage changed"
            )
        if (
            self.history_bound
            and self.graph.graph.get("history", _MISSING) is not self.history
        ):
            raise ValueError("native positive-window history binding changed")


def _require_canonical_held_flow(graph, schedule, *, method):
    """Admit shared held rows without imposing a form-rate constitutive law."""
    from .. import gamma
    from ..dynamics import integrators
    from .event_runtime import (
        _CANONICAL_DEFAULT_INTEGRATE,
        _CanonicalDefaultIntegrator,
        _flow_runtime_metadata,
        _require_runtime_clock,
        _resolve_integrator_method,
        _resolve_schedule_integrator_instance,
        _validate_schedule_clock,
    )

    _validate_schedule_clock(schedule)
    _require_runtime_clock(graph, schedule.start_time, boundary="native_window.start")
    candidate = graph.graph.get("integrator")
    if (
        candidate is not None
        and candidate is not _CanonicalDefaultIntegrator
        and type(candidate) is not _CanonicalDefaultIntegrator
    ):
        raise TypeError(
            "native positive-window requires the canonical DefaultIntegrator"
        )
    if (
        integrators.DefaultIntegrator is not _CanonicalDefaultIntegrator
        or _CanonicalDefaultIntegrator.integrate is not _CANONICAL_DEFAULT_INTEGRATE
    ):
        raise ValueError("native positive-window integrator binding changed")
    integrator = _resolve_schedule_integrator_instance(graph, cache_result=False)
    metadata = _flow_runtime_metadata(
        graph,
        schedule.intervals[0],
        integrator,
        _resolve_integrator_method(integrator),
        method=method,
    )
    if not metadata.integrator_provenance_certified:
        raise ValueError("native positive-window integrator provenance is unavailable")
    integrators._integration_clip_policy(graph)
    if parse_bool(graph.graph.get("use_extended_dynamics", False)):
        raise ValueError("native positive-window excludes extended dynamics")
    raw_gamma = graph.graph.get("GAMMA")
    if raw_gamma is not None and (
        type(raw_gamma) is not dict or raw_gamma.get("type", "none") != "none"
    ):
        raise ValueError("native positive-window requires Gamma type none")
    if (
        not metadata.gamma_is_none
        or gamma.GAMMA_REGISTRY.get("none") is not gamma._BUILTIN_GAMMA_ENTRIES["none"]
    ):
        raise ValueError(
            "native positive-window requires the unchanged zero Gamma source"
        )
    for node in graph:
        integrators._node_state(graph.nodes[node])
    from ..dynamics.native_epi_source import _capture_endpoint

    if not _capture_endpoint(graph, []).available:
        raise ValueError("native positive-window initial EPI capture is unavailable")


def _require_held_form_flow(graph, schedule, *, method):
    """The original source retains its strictly zero-pressure domain."""
    _require_canonical_held_flow(graph, schedule, method=method)
    for node in graph:
        raw = get_attr(
            graph.nodes[node], ALIAS_DNFR, None, strict=True, conv=lambda value: value
        )
        pressure, _ = finite_represented_real(raw, "held-form DeltaNFR")
        if pressure != 0.0:
            raise ValueError("native positive-window requires zero held pressure")


def _observe_native_positive_window(result):
    """Rebuild owned arithmetic, never issue a live action gate."""
    return _observe_native_window_accounting(
        result,
        NativePositiveWindowSource,
        NativePositiveWindowGrowth,
        require_zero_flow=True,
    )


def _observe_native_window_accounting(
    result, source_type, growth_type, *, require_zero_flow
):
    """Shared endpoint balance after each producer admits its own flow law."""
    from ..dynamics.native_epi_source import (
        NativeEPIBatchObservation,
        NativeEPIDispatch,
        NativeEPIEndpoint,
        _batch_observation,
        _same_population,
    )

    if type(result) is not source_type:
        raise TypeError("native growth requires its separate produced source type")
    flow = observe_operator_epi_growth(result.flow_result)
    if (
        not flow.available
        or result.flow_result.schedule.event_count != 0
        or len(flow.contributions) != 1
    ):
        raise ValueError("native positive-window flow coverage is unavailable")
    batch = result.native_batch
    if (
        type(batch) is not NativeEPIBatchObservation
        or type(batch.dispatches) is not tuple
    ):
        raise TypeError("native positive-window requires its actual batch observation")
    if any(
        type(item) is not NativeEPIDispatch
        or not item.returned
        or item.failure_type is not None
        for item in batch.dispatches
    ):
        raise ValueError("native positive-window requires a completed dispatch source")
    if not batch.completed or batch.failure_type is not None:
        raise ValueError("native positive-window source did not complete")
    rebuilt = _batch_observation(
        batch.initial, batch.final, batch.dispatches, True, None, False
    )

    def batch_values(value):
        return tuple(
            object.__getattribute__(value, item.name)
            for item in fields(NativeEPIBatchObservation)
        )

    if not proof_stamps_are_identical(
        structural_proof_signature(batch_values(batch)),
        structural_proof_signature(batch_values(rebuilt)),
    ):
        raise ValueError("native positive-window capture fields are inconsistent")
    endpoints = (result.initial, result.flow_endpoint, batch.initial, batch.final)
    if any(
        type(item) is not NativeEPIEndpoint or not item.available for item in endpoints
    ):
        raise ValueError("native positive-window endpoint is unavailable")
    if any(not _same_population(result.initial, item) for item in endpoints):
        raise ValueError("native positive-window node population or storage changed")
    if not rebuilt.available or rebuilt.clock_preserved is not True:
        raise ValueError(
            "native positive-window batch coverage or clock is unavailable"
        )
    if any(rebuilt.exact_outside_dispatch_variation):
        raise ValueError("native positive-window has outside-dispatch EPI changes")
    if (
        result.initial.exact_time != flow.exact_start_time
        or any(item.exact_time != flow.exact_end_time for item in endpoints[1:])
        or result.initial.exact_epi != flow.exact_epi_before
        or result.flow_endpoint.exact_epi != flow.exact_epi_after
        or result.flow_endpoint.exact_epi != batch.initial.exact_epi
        or len(flow.nodes) != len(result.initial.nodes)
        or any(
            a is not b for a, b in zip(flow.nodes, result.initial.nodes, strict=True)
        )
    ):
        raise ValueError("native positive-window flow and batch are disconnected")
    if require_zero_flow and any(flow.exact_net_increment):
        raise ValueError("native positive-window flow did not hold EPI")
    net = tuple(
        b - a for a, b in zip(flow.exact_epi_before, batch.final.exact_epi, strict=True)
    )
    if net != tuple(
        a + b
        for a, b in zip(
            flow.exact_net_increment, rebuilt.exact_dispatch_increment, strict=True
        )
    ):
        raise ValueError("native positive-window source balance is inconsistent")
    return growth_type(
        flow.nodes,
        flow.exact_start_time,
        flow.exact_end_time,
        flow.exact_duration,
        flow.exact_epi_before,
        batch.final.exact_epi,
        flow.exact_net_increment,
        rebuilt.exact_dispatch_increment,
        rebuilt.exact_dispatch_variation,
        net,
        tuple(value / flow.exact_duration for value in net),
    )


def _admit_native_batch_domain(graph, schedule, history, *, method, n_jobs):
    """Admit the shared native batch owner independently of its flow law."""
    from ..dynamics.selectors import _preflight_glyph_batch

    if type(schedule) is not OperatorEventSchedule:
        raise TypeError("native positive-window requires an exact event schedule")
    schedule.__post_init__()
    if (
        schedule.event_count
        or len(schedule.intervals) != 1
        or schedule.exact_total_flow_duration <= 0
    ):
        raise ValueError(
            "native positive-window requires one positive empty-word interval"
        )
    if method is not None and type(method) is not str:
        raise TypeError("method must be a string or None")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)
    ):
        raise TypeError("n_jobs must be an integer or None")
    if type(history) is not dict:
        raise TypeError("native positive-window history must be an exact dictionary")
    stored = graph.graph.get("history", _MISSING)
    if stored is not _MISSING and stored is not history:
        raise ValueError(
            "native positive-window requires the actual graph-owned history"
        )
    guard = _NativeSourceBinding(graph, history)
    guard.require_bindings()
    admission = _preflight_glyph_batch(graph, history)
    if not admission.use_canon or graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise":
        raise ValueError(
            "native positive-window requires enabled grammar and raise mode"
        )
    return guard


def _admit_native_source_domain(graph, schedule, history, *, method, n_jobs):
    """The zero-pressure producer independently owns its constitutive premise."""
    guard = _admit_native_batch_domain(
        graph, schedule, history, method=method, n_jobs=n_jobs
    )
    _require_held_form_flow(graph, schedule, method=method)
    return guard


def _produce_native_positive_window(graph, schedule, history, *, method, n_jobs):
    from ..dynamics import selectors
    from ..dynamics.native_epi_source import _capture_endpoint

    guard = _admit_native_source_domain(
        graph, schedule, history, method=method, n_jobs=n_jobs
    )
    retained = []
    initial = _capture_endpoint(graph, retained)
    if not initial.available:
        raise ValueError("native positive-window initial EPI capture is unavailable")
    if not guard.history_bound:
        graph.graph["history"] = history
        guard.history_bound = True
    flow_result = execute_operator_event_schedule(
        graph,
        schedule,
        method=method,
        n_jobs=n_jobs,
        include_stage_certificates=True,
    )
    guard.require_bindings()
    flow_endpoint = _capture_endpoint(graph, retained)
    decisions, captures = [], []
    selectors._apply_native_operator_batch(
        graph,
        history,
        decision_trace=decisions,
        epi_trace=captures,
        _source_guard=guard,
    )
    guard.require_bindings()
    if len(captures) != 1:
        raise ValueError("native positive-window requires exactly one captured batch")
    result = NativePositiveWindowSource(
        flow_result, captures[0], tuple(decisions), initial, flow_endpoint
    )
    _observe_native_positive_window(result)
    return result


def execute_native_positive_window(
    graph: Any,
    *,
    duration: Any,
    candidates: tuple[Any, ...] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    mutation_kwargs=None,
) -> NativePositiveWindowDispatch:
    """Execute real held-form time, native source, then selected canonical ZHIR.

    The source retains its existing sequential failure prefix. A separate
    transaction covers only post-source selection and response. The fixed-domain
    route accepts unchanged builtin selectors and no detached growth evidence.
    Successful native Mutation has priority over an appended response; that
    branch is unreachable in this restricted zero-pressure single-batch domain.
    """
    from ._operator_growth_trigger import _native_positive_window_source_context

    request_snapshot = GraphTransactionSnapshot(graph)
    try:
        if method is not None and type(method) is not str:
            raise TypeError("method must be a string or None")
        if n_jobs is not None and (
            isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)
        ):
            raise TypeError("n_jobs must be an integer or None")
        _, exact_duration = finite_represented_real(duration, "native window duration")
        if exact_duration <= 0:
            raise ValueError("native positive-window duration must be positive")
        schedule = build_operator_event_schedule(
            (),
            start_time=graph.graph.get("_t", 0.0),
            flow_durations=(duration,),
        )
        candidates = tuple(graph.nodes) if candidates is None else candidates
        operator, options, before_watermarks = _prepare_growth_request(
            graph,
            schedule,
            candidates,
            mutation_kwargs,
            request_snapshot,
        )
        validate_common_execution_arguments(
            graph.graph, options, operator=operator.name
        )
        for node in candidates:
            _read_mutation_capacity(graph.nodes[node], graph.graph)
        raw_xi = graph.graph.get("ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL)
        _, exact_xi = finite_represented_real(raw_xi, "ZHIR_THRESHOLD_XI")
        if exact_xi < 0:
            raise ValueError("ZHIR_THRESHOLD_XI must be nonnegative")
        history = graph.graph.get("history", _MISSING)
        if history is _MISSING:
            history = {}
        _admit_native_source_domain(
            graph, schedule, history, method=method, n_jobs=n_jobs
        )
    except BaseException as failure:
        request_snapshot.restore_after_failure(graph, failure)
        raise

    # This lifetime produces the source itself. No outer transaction erases a
    # completed flow or a genuine partial native prefix on source failure.
    with _native_positive_window_source_context(
        graph,
        schedule,
        candidates,
        history,
        method=method,
        n_jobs=n_jobs,
    ) as source:
        response_snapshot = GraphTransactionSnapshot(graph)
        try:
            if _watermarks(graph, candidates) != before_watermarks:
                raise RuntimeError(
                    "Mutation consumption memory changed during native source"
                )
            native_mutated = tuple(
                item.node
                for item in source.result.native_batch.dispatches
                if item.returned and item.glyph == "ZHIR"
            )
            decisions = _select_growth_candidates(
                source,
                candidates,
                before_watermarks,
                native_mutated=native_mutated,
            )
            selected = tuple(item.node for item in decisions if item.selected)
            stage = None
            if selected:
                stage = _execute_growth_stage(
                    graph,
                    source,
                    operator,
                    selected,
                    options,
                    tuple(
                        item.previous_consumed_end
                        for item in decisions
                        if item.selected
                    ),
                )
            else:
                source.validate_live_before_commit(graph)
            return NativePositiveWindowDispatch(
                source.result,
                source.observation,
                candidates,
                decisions,
                selected,
                stage,
            )
        except BaseException as failure:
            response_snapshot.restore_after_failure(graph, failure)
            raise
