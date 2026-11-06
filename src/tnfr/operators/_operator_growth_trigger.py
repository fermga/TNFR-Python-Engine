"""Private live admission for an explicit operator-schedule Mutation source.

The source is minted only around an actual schedule execution. Exact net growth
decides the optional threshold; projected floats remain observational telemetry.
No continuous history is created or reinterpreted by this adapter.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL
from ..physics.mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerEvidence,
    certify_mutation_trigger,
)
from ..types import require_finite_real_scalar_epi
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)

_SOURCE = "operator_schedule_net_growth"
_TIME_BASIS = "operator_schedule_time"
_NATIVE_SOURCE = "native_positive_window_net_growth"
_NATIVE_TIME_BASIS = "native_positive_window_time"
_EMISSION_SOURCE = "native_emission_window_net_growth"
_EMISSION_TIME_BASIS = "native_emission_window_time"
_PRODUCTION_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class OperatorGrowthTriggerCertificate(MutationTriggerCertificate):
    """Exact trigger with compatible, explicitly projected float diagnostics.

    This public value type is observational. Constructing it cannot authorize
    a Mutation. Only the private live production context supplies a runtime
    gate. ``physical_time_resolved`` means the declared runtime clock, not an
    identification with laboratory time.
    """

    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    exact_previous_epi: Fraction
    exact_current_epi: Fraction
    exact_observed_rate: Fraction
    exact_xi: Fraction


def _validate_operator_growth_trigger_certificate(
    certificate: OperatorGrowthTriggerCertificate,
) -> None:
    """Rebuild the exact algebra and diagnostic projections of this value."""

    if type(certificate) is not OperatorGrowthTriggerCertificate:
        raise TypeError("operator growth requires its exact trigger certificate type")
    exact_values = (
        certificate.exact_start_time,
        certificate.exact_end_time,
        certificate.exact_duration,
        certificate.exact_previous_epi,
        certificate.exact_current_epi,
        certificate.exact_observed_rate,
        certificate.exact_xi,
    )
    if any(type(value) is not Fraction for value in exact_values):
        raise TypeError("operator growth trigger exact fields must be Fractions")
    if any(
        type(value.numerator) is not int or type(value.denominator) is not int
        for value in exact_values
    ):
        raise TypeError("operator growth fractions require Python integer components")
    if (
        certificate.exact_duration <= 0
        or certificate.exact_end_time - certificate.exact_start_time
        != certificate.exact_duration
        or certificate.exact_xi < 0
        or certificate.exact_observed_rate
        != (certificate.exact_current_epi - certificate.exact_previous_epi)
        / certificate.exact_duration
        or not certificate.exact_observed_rate > certificate.exact_xi
    ):
        raise ValueError("operator growth trigger exact threshold is inconsistent")
    evidence = certificate.evidence
    if type(evidence) is not MutationTriggerEvidence:
        raise TypeError("operator growth trigger requires canonical evidence values")
    expected_projection = (
        (certificate.current_epi, certificate.exact_current_epi),
        (certificate.xi, certificate.exact_xi),
        (certificate.observed_depi_dt, certificate.exact_observed_rate),
        (evidence.previous_epi, certificate.exact_previous_epi),
        (evidence.current_epi, certificate.exact_current_epi),
        (evidence.previous_time, certificate.exact_start_time),
        (evidence.current_time, certificate.exact_end_time),
        (evidence.sample_interval, certificate.exact_duration),
        (evidence.observed_depi_dt, certificate.exact_observed_rate),
    )
    for projected, exact in expected_projection:
        expected = finite_represented_real(exact, "operator growth projection")[0]
        if type(projected) is not float or projected != expected:
            raise ValueError("operator growth trigger projection is inconsistent")
    if (
        Fraction.from_float(certificate.current_epi) != certificate.exact_current_epi
        or Fraction.from_float(evidence.previous_epi) != certificate.exact_previous_epi
    ):
        raise ValueError("operator growth EPI endpoints must retain represented state")
    for value in (
        certificate.nu_f,
        certificate.delta_nfr,
        certificate.predicted_depi_dt,
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("operator growth prediction fields must be finite floats")
    predicted = certificate.nu_f * certificate.delta_nfr
    expected_gap = certificate.observed_depi_dt - predicted
    if not math.isfinite(expected_gap):
        expected_gap = None
    expected_prediction = predicted > certificate.xi
    if (
        certificate.nu_f <= 0
        or certificate.predicted_depi_dt != predicted
        or type(certificate.predicted_crossed) is not bool
        or certificate.predicted_crossed is not expected_prediction
        or certificate.rate_gap != expected_gap
        or (
            certificate.rate_gap is not None and type(certificate.rate_gap) is not float
        )
        or certificate.reason is not None
        or evidence.reason is not None
        or any(
            type(label) is not str
            for label in (
                certificate.source,
                evidence.source,
                certificate.time_basis,
                evidence.time_basis,
            )
        )
        or (certificate.source, certificate.time_basis)
        not in (
            (_SOURCE, _TIME_BASIS),
            (_NATIVE_SOURCE, _NATIVE_TIME_BASIS),
            (_EMISSION_SOURCE, _EMISSION_TIME_BASIS),
        )
        or evidence.source != certificate.source
        or evidence.time_basis != certificate.time_basis
        or any(
            flag is not True
            for flag in (
                certificate.observed_crossed,
                certificate.evidence_available,
                certificate.evidence_valid,
                certificate.capacity_active,
                certificate.threshold_gate_satisfied,
                certificate.physical_time_resolved,
                certificate.current_endpoint_matches_state,
                evidence.physical_time_resolved,
                evidence.current_endpoint_matches_state,
                evidence.is_valid,
            )
        )
    ):
        raise ValueError("operator growth trigger diagnostics are inconsistent")


def _graph_signature(graph: Any, references: list[Any] | None = None) -> Any:
    from .event_runtime import _schedule_preparation_state_signature

    return _schedule_preparation_state_signature(graph, retained_references=references)


def _materialize_graph_views(graph: Any) -> None:
    """Resolve inert NetworkX views before sealing complete instance state."""

    graph.nodes
    graph.edges
    graph.adj
    graph.degree
    if graph.is_directed():
        graph.pred
        graph.succ
        graph.in_edges
        graph.out_edges
        graph.in_degree
        graph.out_degree


def _observation_signature(observation: Any) -> Any:
    """Compare rebuilt values rather than newly allocated report identities."""

    from .native_emission_window import NativeEmissionWindowGrowth
    from .native_positive_window import NativePositiveWindowGrowth

    if type(observation) in (NativePositiveWindowGrowth, NativeEmissionWindowGrowth):
        return structural_proof_signature(
            (
                observation.nodes,
                observation.exact_start_time,
                observation.exact_end_time,
                observation.exact_duration,
                observation.exact_epi_before,
                observation.exact_epi_after,
                observation.exact_flow_increment,
                observation.exact_jump_increment,
                observation.exact_jump_variation,
                observation.exact_net_increment,
                observation.exact_net_growth_rate,
                observation.available,
                observation.unavailability_reasons,
                observation.source,
            )
        )

    return structural_proof_signature(
        (
            observation.nodes,
            observation.exact_start_time,
            observation.exact_end_time,
            observation.exact_duration,
            observation.available,
            observation.unavailability_reasons,
            observation.included_event_indices,
            observation.excluded_initial_event_indices,
            tuple(
                (
                    item.position,
                    item.operation_kind,
                    item.operation_index,
                    item.operator_name,
                    item.exact_start_time,
                    item.exact_end_time,
                    item.exact_epi_before,
                    item.exact_epi_after,
                    item.exact_increment,
                )
                for item in observation.contributions
            ),
            observation.exact_epi_before,
            observation.exact_epi_after,
            observation.exact_flow_increment,
            observation.exact_jump_increment,
            observation.exact_jump_variation,
            observation.exact_net_increment,
            observation.exact_net_growth_rate,
            observation.source,
        )
    )


def _read_produced_growth(result):
    """Accept only exact result families minted by live internal producers."""
    from .native_emission_window import (
        NativeEmissionWindowSource,
        _observe_native_emission_window,
    )
    from .native_positive_window import (
        NativePositiveWindowSource,
        _observe_native_positive_window,
    )
    from .operator_growth import observe_operator_epi_growth

    if type(result) is NativePositiveWindowSource:
        return _observe_native_positive_window(result)
    if type(result) is NativeEmissionWindowSource:
        return _observe_native_emission_window(result)
    return observe_operator_epi_growth(result)


def _native_source_stamp(result):
    from .native_emission_window import NativeEmissionWindowSource
    from .native_positive_window import NativePositiveWindowSource

    if type(result) in (NativePositiveWindowSource, NativeEmissionWindowSource):
        return structural_proof_signature(
            (
                type(result),
                ("flow_result_identity", id(result.flow_result)),
                (
                    result.episode_result
                    if type(result) is NativeEmissionWindowSource
                    else None
                ),
                (
                    (result.selector_acceleration_source, result.selector_observation)
                    if type(result) is NativeEmissionWindowSource
                    else None
                ),
                result.initial,
                result.flow_endpoint,
                result.native_batch,
                result.native_decisions,
            )
        )
    return None


@dataclass(frozen=True, slots=True)
class _OperatorGrowthInput:
    """Admitted observational inputs, not an execution gate or permission."""

    node: Any
    previous_epi: Fraction
    current_epi: Fraction
    exact_rate: Fraction
    live_epi: float
    nu_f: float
    minimum_nu_f: float
    xi: float
    exact_xi: Fraction


def _read_operator_growth_input(snapshot: Any, node: Any, observation: Any) -> Any:
    """One primitive owner shared by eligibility and strict live admission."""

    from ._mutation_gate import _read_mutation_capacity, _reject

    index = observation.nodes.index(node)
    previous = observation.exact_epi_before[index]
    current = observation.exact_epi_after[index]
    rate = observation.exact_net_growth_rate[index]
    data = snapshot.nodes[node]
    live_epi = require_finite_real_scalar_epi(
        get_attr(data, ALIAS_EPI, None, strict=True, conv=lambda value: value),
        "operator growth current EPI",
    )
    if Fraction.from_float(live_epi) != current:
        _reject("operator growth endpoint does not match the live target EPI")
    nu_f, minimum = _read_mutation_capacity(data, snapshot.graph)
    raw_xi = snapshot.graph.get("ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL)
    xi, exact_xi = finite_represented_real(raw_xi, "ZHIR_THRESHOLD_XI")
    if raw_xi < 0 or exact_xi < 0:
        _reject("ZHIR_THRESHOLD_XI must be nonnegative")
    return _OperatorGrowthInput(
        node, previous, current, rate, live_epi, nu_f, minimum, xi, exact_xi
    )


class _OperatorGrowthContext:
    """One private source-production lifetime, never stored in a graph."""

    __slots__ = (
        "_source_result",
        "_source_observation",
        "_observation_stamp",
        "_native_stamp",
        "_graph",
        "_targets",
        "_references",
        "_live_signature",
        "_snapshot",
        "_snapshot_signature",
        "_active",
    )

    def __init__(
        self, graph: Any, targets: tuple[Any, ...], result: Any, *, _authority: Any
    ) -> None:
        if _authority is not _PRODUCTION_AUTHORITY:
            raise TypeError("operator growth context requires live source production")
        observation = _read_produced_growth(result)
        if not observation.available:
            raise ValueError(
                "operator growth source unavailable: "
                + ", ".join(observation.unavailability_reasons)
            )
        self._source_result = result
        self._source_observation = observation
        self._observation_stamp = _observation_signature(observation)
        self._native_stamp = _native_source_stamp(result)
        self._graph = graph
        self._targets = targets
        self._references: list[Any] = []
        if self._native_stamp is not None:
            # Keep the original flow alive even if a caller corrupts the frozen
            # source field: equal detached flow values cannot replace this run.
            self._references.append(result.flow_result)
            if observation.source == _EMISSION_SOURCE:
                self._references.append(result.episode_result)
                if result.selector_observation is not None:
                    self._references.append(result.selector_observation)
        _materialize_graph_views(graph)
        self._live_signature = _graph_signature(graph, self._references)
        self._snapshot: Any = None
        self._snapshot_signature: Any = None
        self._active = True

    @property
    def result(self) -> Any:
        """The original actual source result, with no substitution setter."""

        return self._source_result

    @property
    def observation(self) -> Any:
        """The rebuilt source observation, with no substitution setter."""

        return self._source_observation

    def close(self) -> None:
        """End admission even when execution or a surrounding check failed."""

        self._active = False
        self._snapshot = None
        self._snapshot_signature = None

    def validate_live_before_commit(self, graph: Any) -> Any:
        """Require the actual live owner/state and return readmitted evidence."""

        if not self._active or graph is not self._graph:
            raise ValueError("operator growth source owner is foreign or inactive")
        if not proof_stamps_are_identical(
            self._live_signature, _graph_signature(graph)
        ):
            raise ValueError("operator growth source live endpoint is stale")
        return self.readmit_source()

    def readmit_source(self) -> Any:
        """Rebuild production evidence, including before publishing its receipt."""

        if not self._active:
            raise ValueError("operator growth source is inactive")
        observation = _read_produced_growth(self.result)
        current_native_stamp = _native_source_stamp(self.result)
        native_source_preserved = (
            current_native_stamp is None
            if self._native_stamp is None
            else proof_stamps_are_identical(self._native_stamp, current_native_stamp)
        )
        if not native_source_preserved:
            raise ValueError("native positive-window source changed during admission")
        if any(
            not proof_stamps_are_identical(
                self._observation_stamp, _observation_signature(candidate)
            )
            for candidate in (observation, self.observation)
        ):
            raise ValueError("operator growth source changed during admission")
        return observation

    def bind_snapshot(self, graph: Any, snapshot: Any) -> None:
        """Authorize only the stage's fresh detached copy of the live owner."""

        self.validate_live_before_commit(graph)
        if self._snapshot is not None:
            raise ValueError("operator growth context already bound a stage snapshot")
        _materialize_graph_views(snapshot)
        self._snapshot = snapshot
        self._snapshot_signature = _graph_signature(snapshot, self._references)

    @contextmanager
    def probe_candidates(self):
        """Read one source-bound candidate snapshot without authorizing a stage.

        The actual stage still owns its distinct, single snapshot binding and
        strictly re-admits every selected target. The probe supplies no gate.
        """

        from .event_runtime import _require_runtime_clock
        from .network_stage import _detached_stage_graph

        observation = self.validate_live_before_commit(self._graph)
        snapshot = _detached_stage_graph(self._graph)
        _materialize_graph_views(snapshot)
        signature = _graph_signature(snapshot)
        _require_runtime_clock(snapshot, self.result.final_time, boundary="growth.end")
        inputs = tuple(
            _read_operator_growth_input(snapshot, node, observation)
            for node in self._targets
        )
        yield snapshot, inputs
        if not proof_stamps_are_identical(signature, _graph_signature(snapshot)):
            raise ValueError(
                "operator growth candidate snapshot changed during selection"
            )
        self.validate_live_before_commit(self._graph)

    def gate_for(self, snapshot: Any, node: Any) -> Any:
        """Readmit the actual source and build one target's exact runtime gate."""

        from ._mutation_gate import (
            MutationRuntimeGate,
            MutationThresholdSample,
            _reject,
        )
        from .event_runtime import _require_runtime_clock

        observation = self.validate_live_before_commit(self._graph)
        if self._native_stamp is not None and any(
            item.returned and item.node is node and item.glyph == "ZHIR"
            for item in self.result.native_batch.dispatches
        ):
            raise ValueError("native_mutation_precedence")
        if snapshot is not self._snapshot or snapshot is None:
            raise ValueError(
                "operator growth gate requires its authorized stage snapshot"
            )
        if not proof_stamps_are_identical(
            self._snapshot_signature, _graph_signature(snapshot)
        ):
            raise ValueError("operator growth stage snapshot changed during admission")
        if node not in self._targets:
            raise ValueError("operator growth gate target was not requested")
        physical = self.response_interval_for(snapshot, node)
        if physical is not None and physical.previously_consumed:
            raise ValueError("mutation_interval_already_consumed")
        _require_runtime_clock(snapshot, self.result.final_time, boundary="growth.end")
        inputs = _read_operator_growth_input(snapshot, node, observation)
        previous, current, rate = (
            inputs.previous_epi,
            inputs.current_epi,
            inputs.exact_rate,
        )
        live_epi, nu_f, minimum = inputs.live_epi, inputs.nu_f, inputs.minimum_nu_f
        xi, exact_xi = inputs.xi, inputs.exact_xi
        if nu_f <= 0.0 or nu_f < minimum:
            _reject(
                "ZHIR requires active structural frequency; "
                f"got nu_f={nu_f!r}, configured minimum={minimum!r}"
            )
        if not rate > exact_xi:
            _reject(
                f"ZHIR requires exact signed operator growth > xi; got {rate} <= {exact_xi}"
            )
        projected_rate = finite_represented_real(rate, "operator growth rate")[0]
        duration = finite_represented_real(
            observation.exact_duration, "growth duration"
        )[0]
        pressure = get_attr(
            snapshot.nodes[node], ALIAS_DNFR, 0.0, strict=True, conv=lambda value: value
        )
        prediction = certify_mutation_trigger(
            current_epi=live_epi, nu_f=nu_f, delta_nfr=pressure, xi=xi
        )
        gap = projected_rate - prediction.predicted_depi_dt
        source_label = observation.source
        time_basis = {
            _SOURCE: _TIME_BASIS,
            _NATIVE_SOURCE: _NATIVE_TIME_BASIS,
            _EMISSION_SOURCE: _EMISSION_TIME_BASIS,
        }[source_label]
        evidence = MutationTriggerEvidence(
            previous_epi=float(previous),
            current_epi=live_epi,
            previous_time=float(observation.exact_start_time),
            current_time=float(observation.exact_end_time),
            sample_interval=duration,
            observed_depi_dt=projected_rate,
            source=source_label,
            time_basis=time_basis,
            physical_time_resolved=True,
            current_endpoint_matches_state=True,
            is_valid=True,
            reason=None,
        )
        certificate = OperatorGrowthTriggerCertificate(
            current_epi=live_epi,
            nu_f=nu_f,
            delta_nfr=prediction.delta_nfr,
            xi=xi,
            predicted_depi_dt=prediction.predicted_depi_dt,
            observed_depi_dt=projected_rate,
            observed_crossed=True,
            predicted_crossed=prediction.predicted_crossed,
            evidence_available=True,
            evidence_valid=True,
            capacity_active=True,
            threshold_gate_satisfied=True,
            source=source_label,
            time_basis=time_basis,
            physical_time_resolved=True,
            current_endpoint_matches_state=True,
            reason=None,
            rate_gap=gap if math.isfinite(gap) else None,
            evidence=evidence,
            exact_start_time=observation.exact_start_time,
            exact_end_time=observation.exact_end_time,
            exact_duration=observation.exact_duration,
            exact_previous_epi=previous,
            exact_current_epi=current,
            exact_observed_rate=rate,
            exact_xi=exact_xi,
        )
        _validate_operator_growth_trigger_certificate(certificate)
        return MutationRuntimeGate(
            threshold=MutationThresholdSample(
                previous_epi=evidence.previous_epi,
                current_epi=live_epi,
                depi_dt=projected_rate,
                xi=xi,
                history_key=source_label,
                sample_interval=duration,
                time_basis=time_basis,
                physical_time_resolved=True,
                certificate=certificate,
            ),
            nu_f=nu_f,
            minimum_nu_f=minimum,
        )

    def response_interval_for(self, graph, node):
        """New episode route alone shares the actual physical consumption owner."""
        from .native_emission_window import (
            NativeEmissionWindowSource,
            _matching_physical_interval,
        )

        if type(self.result) is NativeEmissionWindowSource:
            return _matching_physical_interval(graph, node, self.observation)
        return None


def _requested_operator_growth_context(value: Any) -> _OperatorGrowthContext:
    if type(value) is not _OperatorGrowthContext:
        raise TypeError("operator growth request requires its private live context")
    return value


@contextmanager
def _operator_growth_source_context(
    graph: Any, schedule: Any, targets: tuple[Any, ...], **schedule_kwargs: Any
):
    """Produce and retire admission around an actual captured schedule run."""

    from .event_runtime import execute_operator_event_schedule

    result = execute_operator_event_schedule(
        graph, schedule, include_stage_certificates=True, **schedule_kwargs
    )
    context = _OperatorGrowthContext(
        graph, targets, result, _authority=_PRODUCTION_AUTHORITY
    )
    try:
        yield context
    finally:
        context.close()


@contextmanager
def _native_positive_window_source_context(
    graph, schedule, targets, history, *, method=None, n_jobs=None
):
    """Produce real elapsed native evidence, then retire its single live use."""
    from .native_positive_window import _produce_native_positive_window

    result = _produce_native_positive_window(
        graph,
        schedule,
        history,
        method=method,
        n_jobs=n_jobs,
    )
    context = _OperatorGrowthContext(
        graph,
        targets,
        result,
        _authority=_PRODUCTION_AUTHORITY,
    )
    try:
        yield context
    finally:
        context.close()


@contextmanager
def _native_emission_window_source_context(
    graph,
    schedule,
    emission_targets,
    targets,
    history,
    *,
    method=None,
    n_jobs=None,
    episode_window=None,
    selector_acceleration_source="solver_diagnostic",
):
    """Produce the distinct AL-episode source and retire its one live authority."""
    from .native_emission_window import _produce_native_emission_window

    result = _produce_native_emission_window(
        graph,
        schedule,
        emission_targets,
        history,
        method=method,
        n_jobs=n_jobs,
        episode_window=episode_window,
        selector_acceleration_source=selector_acceleration_source,
    )
    context = _OperatorGrowthContext(
        graph,
        targets,
        result,
        _authority=_PRODUCTION_AUTHORITY,
    )
    try:
        yield context
    finally:
        context.close()
