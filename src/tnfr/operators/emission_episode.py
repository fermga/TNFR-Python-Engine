"""Explicit finite-duration Emission, with one completion lifecycle per call.

This constitutive lift is separate from instantaneous AL. The actual held-row
integrator supplies form; shared AL admission and completion supply its operator
meaning. Numerical substeps retain their own derivative caches and histories.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from numbers import Integral
from typing import Any

from .._exact_time import finite_represented_real, represented_fraction
from ..alias import get_attr
from ..config.parsing import parse_bool
from ..constants import get_param
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EMISSION_TIMESTAMP,
    ALIAS_VF,
)
from ..types import Glyph
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._argument_validation import validate_common_execution_arguments
from .al_sha_stage_proposals import (
    _LATENCY_KEYS,
    commit_emission_lifecycle,
    emission_epi_proposal,
    propose_emission_stage,
)
from .emission import Emission
from .event_runtime import execute_operator_event_schedule
from .event_timing import build_operator_event_schedule
from .grammar_debt import advance_debt, advance_prior_coherence, read_grammar_state
from .native_positive_window import _require_canonical_held_flow
from .network_stage import (
    GraphTransactionSnapshot,
    PointwiseStageProposal,
    _detached_stage_graph,
    _preflight_two_phase_stage,
    _record_histories_and_patterns,
    _run_postcommit_checks,
)
from .operator_growth import observe_operator_epi_growth
from .preconditions.emission import validate_emission_strict
from .retained_mutation import _held_target_state
from .retained_organization import _protected_state, _signature

__all__ = ("EmissionEpisodeResult", "execute_emission_episode")

_CANONICAL_EMISSION = Emission
_EMISSION_BINDINGS = tuple(
    (name, getattr(Emission, name))
    for name in (
        "__call__",
        "_execute",
        "_validate_preconditions",
        "_validate_requested_hard_invariants",
        "_validate_application_preconditions",
    )
)


def _episode_values(result):
    return (
        id(result.flow_result),
        result.nodes,
        result.targets,
        result.exact_rates,
        result.effective_boosts,
        result.exact_epi_before,
        result.exact_epi_after,
        result.solver_rates,
        result.solver_accelerations,
    )


@dataclass(frozen=True, slots=True)
class EmissionEpisodeResult:
    """Produced flow and one AL completion, never a later action permission."""

    nodes: tuple[Any, ...]
    targets: tuple[Any, ...]
    flow_result: Any
    exact_rates: tuple[Fraction, ...]
    effective_boosts: tuple[float, ...]
    exact_epi_before: tuple[Fraction, ...]
    exact_epi_after: tuple[Fraction, ...]
    solver_rates: tuple[float, ...] = ()
    solver_accelerations: tuple[float, ...] = ()
    _proof_stamp: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def start_time(self):
        return self.flow_result.schedule.start_time

    @property
    def final_time(self):
        return self.flow_result.final_time

    def _proof_fields_are_intact(self):
        return (
            self._proof_stamp is not None
            and self.flow_result._proof_fields_are_intact()
            and proof_stamps_are_identical(
                self._proof_stamp, structural_proof_signature(_episode_values(self))
            )
        )


def _admit_emission_episode(graph, targets, *, duration, method, n_jobs, window):
    """Admit the complete monotone latent chart before actual flow."""
    from ..dynamics.integrators import _node_state
    from ..dynamics.native_epi_source import _capture_endpoint
    from ..node import NodeNX

    if type(targets) is not tuple or not targets:
        raise TypeError("Emission episode targets must be a nonempty exact tuple")
    if len(set(targets)) != len(targets) or any(node not in graph for node in targets):
        raise ValueError("Emission episode targets must be unique existing nodes")
    if method is not None and type(method) is not str:
        raise TypeError("method must be a string or None")
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)
    ):
        raise TypeError("n_jobs must be an integer or None")
    duration_value, exact_duration = finite_represented_real(
        duration, "Emission episode duration"
    )
    if exact_duration <= 0:
        raise ValueError("Emission episode duration must be positive")
    canon = get_param(graph, "GRAMMAR_CANON")
    if not isinstance(canon, Mapping):
        raise TypeError("GRAMMAR_CANON must be a mapping")
    if not parse_bool(canon.get("enabled", False)) or (
        graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
    ):
        raise ValueError("Emission episode requires enabled grammar and raise mode")
    if Emission is not _CANONICAL_EMISSION or any(
        getattr(Emission, name) is not expected for name, expected in _EMISSION_BINDINGS
    ):
        raise ValueError("Emission episode canonical operator binding changed")
    schedule = build_operator_event_schedule(
        (), start_time=graph.graph.get("_t", 0.0), flow_durations=(duration_value,)
    )
    _require_canonical_held_flow(graph, schedule, method=method)
    options = {"window": window, "validate_preconditions": True, "dt": duration_value}
    validate_common_execution_arguments(graph.graph, options, operator="Emission")
    operator = Emission()
    preflight = _preflight_two_phase_stage(
        graph,
        operator,
        targets,
        sequence_context=None,
        execution_kwargs=options,
    )
    if not preflight.exact_stage:
        raise ValueError("Emission episode requires actual AL grammar admission")
    nodes = tuple(graph.nodes)
    rates = []
    proposals, boosts = [], []
    timestamp = datetime.now(timezone.utc).isoformat()
    endpoint_chart = _detached_stage_graph(preflight.snapshot)
    for node in nodes:
        vf, pressure, _, epi = _node_state(preflight.snapshot.nodes[node])
        if emission_epi_proposal(preflight.snapshot.graph, epi, 0.0)[1] != epi:
            raise ValueError("Emission episode starts outside the unprojected chart")
        rate, exact_rate = finite_represented_real(vf * pressure, "Emission held rate")
        rates.append(exact_rate)
        if node not in targets:
            if vf != 0 and pressure != 0:
                raise ValueError("Emission episode has an unowned nonzero form row")
            continue
        if exact_rate <= 0:
            raise ValueError("Emission episode targets require positive held form rate")
        boost, _ = finite_represented_real(
            duration_value * rate, "Emission effective boost"
        )
        if boost <= 0:
            raise ValueError("Emission episode positive increment underflowed")
        proposal = propose_emission_stage(
            preflight.snapshot,
            node,
            {"AL_boost": boost},
            timestamp=timestamp,
        )
        if not proposal.initialize_emission and not proposal.increment_lineage:
            raise ValueError(
                "Emission episode requires counted existing activation lineage"
            )
        if proposal.epi_after != proposal.epi_before + boost:
            raise ValueError("Emission episode excludes projected AL endpoints")
        validate_emission_strict(preflight.snapshot, node)
        NodeNX.from_graph(endpoint_chart, node).EPI = proposal.epi_after
        validate_emission_strict(endpoint_chart, node)
        proposals.append(PointwiseStageProposal(node, Glyph.AL, proposal))
        boosts.append((node, boost))
    # Targets retain their declared lifecycle order independently of graph order.
    by_node = {item.node: item for item in proposals}
    by_boost = dict(boosts)
    ordered = tuple(by_node[node] for node in targets)
    ordered_boosts = tuple(by_boost[node] for node in targets)
    initial = _capture_endpoint(graph, [])
    return schedule, operator, preflight, ordered, tuple(rates), ordered_boosts, initial


def _episode_temporal_state(graph, nodes):
    keys = (
        "epi_time_history",
        "epi_history",
        "_epi_history",
        *ALIAS_DEPI,
        *ALIAS_D2EPI,
    )
    return _signature(
        graph,
        tuple(
            (
                node,
                tuple(
                    (key, graph.nodes[node][key])
                    for key in keys
                    if key in graph.nodes[node]
                ),
            )
            for node in nodes
        ),
    )


def _episode_lifecycle_state(graph, targets):
    keys = (
        *_LATENCY_KEYS,
        *ALIAS_EMISSION_TIMESTAMP,
        "_emission_activated",
        "_emission_origin",
        "_structural_lineage",
    )
    return _signature(
        graph,
        tuple(
            (
                node,
                tuple(
                    (key, graph.nodes[node][key])
                    for key in keys
                    if key in graph.nodes[node]
                ),
            )
            for node in targets
        ),
    )


def execute_emission_episode(
    graph: Any,
    targets: tuple[Any, ...],
    *,
    duration: Any,
    method: str | None = None,
    n_jobs: int | None = None,
    window: int | None = None,
) -> EmissionEpisodeResult:
    """Integrate one positive AL episode, then commit its lifecycle exactly once.

    Admission and all flow/completion effects are graph-transactional. Selected
    nodes remain in AL's strict latent chart for the whole monotone interval.
    Instantaneous AL, native lags and actual solver diagnostics are unchanged.
    """
    from ..dynamics.native_epi_source import _capture_endpoint, _same_population

    transaction = GraphTransactionSnapshot(graph)
    try:
        schedule, operator, preflight, proposals, rates, boosts, initial = (
            _admit_emission_episode(
                graph,
                targets,
                duration=duration,
                method=method,
                n_jobs=n_jobs,
                window=window,
            )
        )
        nodes = initial.nodes
        retained = [graph.nodes[node] for node in nodes]
        protected = _protected_state(graph, nodes)
        held = _signature(
            graph,
            tuple(
                (
                    node,
                    tuple(
                        (
                            alias,
                            get_attr(
                                graph.nodes[node],
                                alias,
                                None,
                                strict=True,
                                conv=lambda value: value,
                            ),
                        )
                        for alias in (ALIAS_VF, ALIAS_DNFR)
                    ),
                )
                for node in nodes
            ),
        )
        grammar = tuple(read_grammar_state(graph.nodes[node]) for node in targets)
        flow_result = execute_operator_event_schedule(
            graph,
            schedule,
            method=method,
            n_jobs=n_jobs,
            include_stage_certificates=True,
        )
        flow = observe_operator_epi_growth(flow_result)
        if not flow.available or (
            flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
            is not True
        ):
            raise ValueError("Emission episode held-flow coverage is unavailable")
        expected = dict(zip(nodes, initial.exact_epi, strict=True))
        for proposal in proposals:
            expected[proposal.node] = represented_fraction(
                proposal.payload.epi_after, "AL endpoint"
            )
        endpoint = _capture_endpoint(graph, retained)
        if (
            not endpoint.available
            or not _same_population(initial, endpoint)
            or endpoint.exact_epi != tuple(expected[node] for node in nodes)
            or flow.exact_epi_before != initial.exact_epi
            or flow.exact_epi_after != endpoint.exact_epi
        ):
            raise ValueError(
                "Emission episode flow does not match prospective AL endpoint"
            )
        if not proof_stamps_are_identical(protected, _protected_state(graph, nodes)):
            raise RuntimeError("Emission episode changed held phase or support")
        current_held = _signature(
            graph,
            tuple(
                (
                    node,
                    tuple(
                        (
                            alias,
                            get_attr(
                                graph.nodes[node],
                                alias,
                                None,
                                strict=True,
                                conv=lambda value: value,
                            ),
                        )
                        for alias in (ALIAS_VF, ALIAS_DNFR)
                    ),
                )
                for node in nodes
            ),
        )
        if not proof_stamps_are_identical(held, current_held):
            raise RuntimeError("Emission episode changed held capacity or pressure")
        completion_held = _held_target_state(graph, nodes)
        temporal = _episode_temporal_state(graph, nodes)
        for proposal in proposals:
            commit_emission_lifecycle(graph, proposal.payload)
            data = graph.nodes[proposal.node]
            expected_count = (
                1
                if proposal.payload.initialize_emission
                else proposal.payload.lineage_activation_count_after
            )
            actual_count = data.get("_structural_lineage", {}).get("activation_count")
            if (
                data.get("_emission_activated") is not True
                or type(actual_count) is not int
                or actual_count != expected_count
                or any(key in data for key in proposal.payload.latency_keys_to_clear)
            ):
                raise RuntimeError(
                    "Emission episode lost its counted activation lifecycle"
                )
        lifecycle = _episode_lifecycle_state(graph, targets)
        _record_histories_and_patterns(graph, proposals, window=preflight.window)
        graph._last_operator_applied = operator.name
        _run_postcommit_checks(
            graph,
            preflight.snapshot,
            operator,
            proposals,
            preflight.states_before,
            preflight.execution_kwargs,
        )
        if not proof_stamps_are_identical(
            lifecycle, _episode_lifecycle_state(graph, targets)
        ):
            raise RuntimeError("Emission completion changed its activation lifecycle")
        if (
            not proof_stamps_are_identical(protected, _protected_state(graph, nodes))
            or not proof_stamps_are_identical(
                completion_held, _held_target_state(graph, nodes)
            )
            or not proof_stamps_are_identical(
                temporal, _episode_temporal_state(graph, nodes)
            )
            or represented_fraction(graph.graph["_t"], "episode clock")
            != flow.exact_end_time
        ):
            raise RuntimeError("Emission completion changed protected flow state")
        for node, prior in zip(targets, grammar, strict=True):
            current = read_grammar_state(graph.nodes[node])
            unbounded = (*prior.history, Glyph.AL.value)
            if (
                current.operator_step != prior.operator_step + 1
                or current.u2_debt != advance_debt(prior.u2_debt, Glyph.AL)
                or current.prior_coherence
                != advance_prior_coherence(prior.prior_coherence, Glyph.AL)
                or current.history
                != (unbounded[-preflight.window :] if preflight.window else ())
            ):
                raise RuntimeError(
                    "Emission episode lost its single grammar occurrence"
                )
        result = EmissionEpisodeResult(
            nodes,
            targets,
            flow_result,
            rates,
            boosts,
            initial.exact_epi,
            endpoint.exact_epi,
            tuple(
                finite_represented_real(
                    get_attr(
                        graph.nodes[node],
                        ALIAS_DEPI,
                        None,
                        strict=True,
                        conv=lambda value: value,
                    ),
                    "episode solver rate",
                )[0]
                for node in nodes
            ),
            tuple(
                finite_represented_real(
                    get_attr(
                        graph.nodes[node],
                        ALIAS_D2EPI,
                        None,
                        strict=True,
                        conv=lambda value: value,
                    ),
                    "episode solver acceleration",
                )[0]
                for node in nodes
            ),
        )
        object.__setattr__(
            result, "_proof_stamp", structural_proof_signature(_episode_values(result))
        )
        return result
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
