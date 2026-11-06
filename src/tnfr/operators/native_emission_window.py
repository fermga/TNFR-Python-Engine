"""An actual Emission episode, native selection and optional growth response.

The supplied orchestration is explicit. Its source commits before the separate
response transaction; no detached report can activate the shared Mutation gate.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from .._exact_time import finite_represented_real, represented_fraction
from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL
from ..dynamics._mutation_occurrence import (
    CONSUME_POLICY,
    admit_native_mutation_occurrence,
    native_mutation_occurrence_policy,
)
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._argument_validation import validate_common_execution_arguments
from ._mutation_gate import _read_mutation_capacity, mutation_trigger_certificate
from .emission_episode import (
    EmissionEpisodeResult,
    _admit_emission_episode,
    execute_emission_episode,
)
from .emission_observation import (
    EMISSION_INTERIOR,
    SOLVER_DIAGNOSTIC,
    EmissionEpisodeObservation,
    _execute_emission_for_selection,
    _observation_values,
    _require_episode_parametric_selector,
    _selector_acceleration_source,
    observe_emission_episode,
)
from .native_positive_window import (
    _MISSING,
    NativePositiveWindowDispatch,
    NativePositiveWindowGrowth,
    NativePositiveWindowSource,
    _admit_native_batch_domain,
    _observe_native_window_accounting,
)
from .network_stage import GraphTransactionSnapshot
from .operator_growth_mutation import (
    _execute_growth_stage,
    _prepare_growth_request,
    _select_growth_candidates,
    _watermarks,
)

__all__ = (
    "NativeEmissionWindowSource",
    "NativeEmissionWindowGrowth",
    "NativeEmissionWindowDispatch",
    "execute_native_emission_window",
)


@dataclass(frozen=True, slots=True)
class NativeEmissionWindowSource(NativePositiveWindowSource):
    """Separate actual episode/native records, with no activation constructor."""

    episode_result: EmissionEpisodeResult
    selector_acceleration_source: str = SOLVER_DIAGNOSTIC
    selector_observation: EmissionEpisodeObservation | None = None


@dataclass(frozen=True, slots=True)
class NativeEmissionWindowGrowth(NativePositiveWindowGrowth):
    """Exact endpoint accounting including the actual nonzero Emission flow."""

    source: str = field(default="native_emission_window_net_growth", init=False)


@dataclass(frozen=True, slots=True)
class NativeEmissionWindowDispatch(NativePositiveWindowDispatch):
    """Produced source and its optional phase-only response."""

    source_result: NativeEmissionWindowSource
    growth: NativeEmissionWindowGrowth


def _require_consumption_policy(graph):
    if native_mutation_occurrence_policy(graph.graph) != CONSUME_POLICY:
        raise ValueError("native Emission window requires consume_timestamped_interval")


def _observe_native_emission_window(result):
    if type(result) is not NativeEmissionWindowSource:
        raise TypeError("native Emission growth requires its exact produced source")
    episode = result.episode_result
    if (
        type(episode) is not EmissionEpisodeResult
        or not episode._proof_fields_are_intact()
        or result.flow_result is not episode.flow_result
        or result.initial.exact_epi != episode.exact_epi_before
        or result.flow_endpoint.exact_epi != episode.exact_epi_after
    ):
        raise ValueError("native Emission source lost its actual episode binding")
    mode = _selector_acceleration_source(result.selector_acceleration_source)
    if mode == EMISSION_INTERIOR:
        if type(
            result.selector_observation
        ) is not EmissionEpisodeObservation or not proof_stamps_are_identical(
            structural_proof_signature(
                _observation_values(result.selector_observation)
            ),
            structural_proof_signature(
                _observation_values(observe_emission_episode(episode))
            ),
        ):
            raise ValueError(
                "native Emission source lost its typed selector observation"
            )
    elif result.selector_observation is not None:
        raise ValueError(
            "solver diagnostic selection cannot claim typed episode consumption"
        )
    return _observe_native_window_accounting(
        result,
        NativeEmissionWindowSource,
        NativeEmissionWindowGrowth,
        require_zero_flow=False,
    )


def _matching_physical_interval(graph, node, observation):
    """Return only the existing own certificate for this exact source window.

    Neither a derivative estimate nor a source report creates physical samples.
    Missing or invalidated temporal evidence has no native receipt to mirror.
    """
    _require_consumption_policy(graph)
    certificate = mutation_trigger_certificate(graph.nodes[node], graph.graph)
    admission = admit_native_mutation_occurrence(
        graph.nodes[node], certificate, CONSUME_POLICY
    )
    if (
        admission.interval is None
        or certificate.current_endpoint_matches_state is not True
        or represented_fraction(admission.interval[1], "physical interval start")
        != observation.exact_start_time
        or represented_fraction(admission.interval[2], "physical interval end")
        != observation.exact_end_time
    ):
        return None
    return admission


def _produce_native_emission_window(
    graph,
    schedule,
    emission_targets,
    history,
    *,
    method=None,
    n_jobs=None,
    episode_window=None,
    selector_acceleration_source=SOLVER_DIAGNOSTIC,
):
    """Re-admit this source's entire domain before inserting history or flowing."""
    from ..dynamics import selectors
    from ..dynamics.native_epi_source import _capture_endpoint
    from .event_runtime import _require_runtime_clock

    guard = _admit_native_batch_domain(
        graph, schedule, history, method=method, n_jobs=n_jobs
    )
    mode = _selector_acceleration_source(selector_acceleration_source)
    if mode == EMISSION_INTERIOR:
        _require_episode_parametric_selector(graph)
    _require_consumption_policy(graph)
    _require_runtime_clock(graph, schedule.start_time, boundary="emission_window.start")
    duration = finite_represented_real(
        schedule.exact_total_flow_duration, "episode duration"
    )[0]
    _admit_emission_episode(
        graph,
        emission_targets,
        duration=duration,
        method=method,
        n_jobs=n_jobs,
        window=episode_window,
    )
    with ExitStack() as lifetime:
        retained = []
        initial = _capture_endpoint(graph, retained)
        selection_context = None
        if mode == EMISSION_INTERIOR:
            selection_context = lifetime.enter_context(
                _execute_emission_for_selection(
                    graph,
                    emission_targets,
                    duration=duration,
                    method=method,
                    n_jobs=n_jobs,
                    window=episode_window,
                    _history=history,
                )
            )
            episode = selection_context.result
            guard.history_bound = True
        else:
            episode_transaction = GraphTransactionSnapshot(graph)
            try:
                if not guard.history_bound:
                    graph.graph["history"] = history
                    guard.history_bound = True
                episode = execute_emission_episode(
                    graph,
                    emission_targets,
                    duration=duration,
                    method=method,
                    n_jobs=n_jobs,
                    window=episode_window,
                )
            except BaseException as failure:
                episode_transaction.restore_after_failure(graph, failure)
                raise
        guard.require_bindings()
        _require_consumption_policy(graph)
        flow_endpoint = _capture_endpoint(graph, retained)
        decisions, captures = [], []
        selectors._apply_native_operator_batch(
            graph,
            history,
            decision_trace=decisions,
            epi_trace=captures,
            _source_guard=guard,
            _episode_selection_context=selection_context,
        )
        guard.require_bindings()
        _require_consumption_policy(graph)
        if len(captures) != 1:
            raise ValueError(
                "native Emission source requires exactly one captured batch"
            )
        result = NativeEmissionWindowSource(
            episode.flow_result,
            captures[0],
            tuple(decisions),
            initial,
            flow_endpoint,
            episode,
            mode,
            None if selection_context is None else selection_context.observation,
        )
        _observe_native_emission_window(result)
        return result


def execute_native_emission_window(
    graph: Any,
    emission_targets: tuple[Any, ...],
    *,
    duration: Any,
    candidates: tuple[Any, ...] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    episode_window: int | None = None,
    mutation_kwargs=None,
    selector_acceleration_source: str = SOLVER_DIAGNOSTIC,
) -> NativeEmissionWindowDispatch:
    """Execute a real AL episode, native batch and selected canonical response.

    The episode is atomic; later native errors retain its completed source and
    actual sequential prefix. Selection/response failure rolls back only the
    response. Successful native Mutation wins over a supplementary response.
    Matching real physical evidence is consumed on either successful route.
    """
    from ._operator_growth_trigger import _native_emission_window_source_context

    request = GraphTransactionSnapshot(graph)
    try:
        mode = _selector_acceleration_source(selector_acceleration_source)
        if mode == EMISSION_INTERIOR:
            _require_episode_parametric_selector(graph)
        admitted = _admit_emission_episode(
            graph,
            emission_targets,
            duration=duration,
            method=method,
            n_jobs=n_jobs,
            window=episode_window,
        )
        schedule = admitted[0]
        candidates = tuple(graph.nodes) if candidates is None else candidates
        operator, options, before = _prepare_growth_request(
            graph,
            schedule,
            candidates,
            mutation_kwargs,
            request,
        )
        validate_common_execution_arguments(
            graph.graph, options, operator=operator.name
        )
        for node in candidates:
            _read_mutation_capacity(graph.nodes[node], graph.graph)
        _, xi = finite_represented_real(
            graph.graph.get("ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL),
            "ZHIR_THRESHOLD_XI",
        )
        if xi < 0:
            raise ValueError("ZHIR_THRESHOLD_XI must be nonnegative")
        history = graph.graph.get("history", _MISSING)
        if history is _MISSING:
            history = {}
        _admit_native_batch_domain(
            graph, schedule, history, method=method, n_jobs=n_jobs
        )
        _require_consumption_policy(graph)
    except BaseException as failure:
        request.restore_after_failure(graph, failure)
        raise

    with _native_emission_window_source_context(
        graph,
        schedule,
        emission_targets,
        candidates,
        history,
        method=method,
        n_jobs=n_jobs,
        episode_window=episode_window,
        selector_acceleration_source=mode,
    ) as source:
        response = GraphTransactionSnapshot(graph)
        try:
            if _watermarks(graph, candidates) != before:
                raise RuntimeError(
                    "Mutation consumption memory changed during native source"
                )
            mutated = tuple(
                item.node
                for item in source.result.native_batch.dispatches
                if item.returned and item.glyph == "ZHIR"
            )
            decisions = _select_growth_candidates(
                source,
                candidates,
                before,
                native_mutated=mutated,
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
            return NativeEmissionWindowDispatch(
                source.result,
                source.observation,
                candidates,
                decisions,
                selected,
                stage,
            )
        except BaseException as failure:
            response.restore_after_failure(graph, failure)
            raise
