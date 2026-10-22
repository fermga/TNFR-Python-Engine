"""Prior visible-rate enclosures for one supplied hidden sine-law mediator.

The input contains only visible state and known incidence. No hidden node,
state, capacity or full-law comparison is supplied or reconstructed as a live
graph. Necessary interval consistency checks enclose conditional candidates;
they do not certify existence, calibration, provenance or future prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from typing import Any

import networkx as nx

from .._exact_time import exact_or_represented_real
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.relational import (
    RelationalExchangeModel,
    _admit_graph,
    _epi,
    _finite,
    _raw,
)
from ..mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_sine_bounds,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ._sine_admission import _sine_model_coefficients
from .relational_observations import _interval, _ordered
from .relational_sine_comparison import _sine_work

__all__ = (
    "SineHiddenStateInference",
    "SineCapacityChannel",
    "SineHiddenCapacityInference",
    "infer_relational_sine_hidden_state",
)


@dataclass(frozen=True)
class SineHiddenStateInference:
    """Visible-only evidence and conditional hidden-state candidate bounds.

    Port arrays retain the requested order; zero-capacity ports have no form
    or phase projection estimate. A unit phase, when returned, is ordered
    (cos, sin) relative to phase_anchor. Projection rows are (-sin(delta),
    cos(delta)) in that chart. A circle-overlapping rectangle with residuals
    containing zero passes necessary checks only: existence or uniqueness of
    one jointly compatible state is not certified.
    """

    reference_model: RelationalExchangeModel
    source_id: str
    clock_id: str
    observation_time: Q
    evidence_window: tuple[Q, Q]
    forecast_start: Q
    visible_nodes: tuple[Any, ...]
    visible_edges: tuple[tuple[Any, Any], ...]
    visible_epi: tuple[Q, ...]
    visible_phase: tuple[Q, ...]
    visible_capacity: tuple[Q, ...]
    ports: tuple[Any, ...]
    port_degrees: tuple[int, ...]
    active_ports: tuple[Any, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    internal_form_gradients: tuple[Q, ...]
    internal_phase_currents: tuple[I, ...]
    hidden_form_by_port: tuple[I | None, ...]
    hidden_form_bounds: I | None
    hidden_projection_raw_bounds: tuple[I | None, ...]
    hidden_projection_bounds: tuple[I | None, ...]
    phase_rank: int | None
    phase_anchor: Any | None
    identifying_ports: tuple[Any, Any] | None
    phase_determinant_bounds: I | None
    phase_gram_determinant_bounds: I
    hidden_unit_phase_raw_bounds: tuple[I, I] | None
    hidden_unit_phase_relative_to_anchor_bounds: tuple[I, I] | None
    unit_norm_squared_bounds: I | None
    projection_rows: tuple[tuple[I, I], ...] | None
    projection_residuals: tuple[I | None, ...]
    status: str
    reasons: tuple[str, ...]
    hidden_capacity_identified: bool = False
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "visible_only_state_and_supplied_original_hidden_star_incidence",
        "same_declared_normalized_sine_law_and_clock_for_both_prior_rate_channels",
        "rate_bounds_must_include_instantaneous_rates_at_declared_observation_time",
        "strictly_prior_evidence_window_does_not_authenticate_calibration_or_source",
        "hidden_form_cancels_from_the_paired_rate_phase_projection",
        "phase_gram_determinant_is_observation_geometry_not_a_dynamical_resultant",
        "bounded_candidate_means_conditional_enclosure_not_existence_or_identification",
        "interval_overlap_and_zero_residual_containment_are_necessary_not_sufficient",
        "rank_one_returns_no_phase_estimate_without_claiming_universal_nonuniqueness",
        "instantaneous_visible_rates_do_not_identify_hidden_capacity",
        "no_hidden_state_read_graph_write_solver_phase_normalization_or_stationary_replacement",
        "no_state_at_forecast_start_future_prediction_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_inference_labels(self)
        return {
            "schema": "tnfr.relational-sine-hidden-state.v1",
            "report": _project(self),
        }

    def infer_capacity(
        self,
        *,
        form_acceleration_bounds,
        phase_acceleration_bounds,
        source_id,
        clock_id,
        observation_time,
        evidence_window,
    ):
        """Add prior acceleration evidence without rereading a hidden state.

        Each acceleration map is a subset of the supplied ports. Missing
        channels remain absent, and clock/time must match this first-rate
        inference. Nonport rates derive from its captured visible state.
        """
        result = _infer_hidden_capacity(
            self,
            form_acceleration_bounds=form_acceleration_bounds,
            phase_acceleration_bounds=phase_acceleration_bounds,
            source_id=source_id,
            clock_id=clock_id,
            observation_time=observation_time,
            evidence_window=evidence_window,
        )
        # Preserve an unchanged source association only after computations on
        # normalized primitives. Equality is not an input-admission rule.
        return (
            replace(result, state_inference=self)
            if self == result.state_inference
            else result
        )


@dataclass(frozen=True)
class SineCapacityChannel:
    """One affine acceleration enclosure observed=baseline+sensitivity*mu.

    ``capacity_bounds`` is the untruncated quotient when the sensitivity is
    separated from zero; negative values remain visible as inconsistency
    evidence. ``residual_bounds`` tests the common nonnegative candidate,
    when available, before the report's final consistency decision.
    """

    port: Any
    kind: str
    observed_bounds: I
    baseline_bounds: I
    sensitivity_bounds: I
    capacity_bounds: I | None = None
    residual_bounds: I | None = None


@dataclass(frozen=True)
class SineHiddenCapacityInference:
    """Conditional hidden-capacity bounds from prior visible accelerations."""

    state_inference: SineHiddenStateInference
    source_id: str
    clock_id: str
    observation_time: Q
    evidence_window: tuple[Q, Q]
    combined_evidence_window: tuple[Q, Q]
    forecast_start: Q
    form_acceleration_bounds: tuple[I | None, ...]
    phase_acceleration_bounds: tuple[I | None, ...]
    visible_form_rate_bounds: tuple[I, ...]
    visible_phase_rate_bounds: tuple[I, ...]
    visible_rate_provenance: tuple[str, ...]
    hidden_phase_available: bool
    hidden_form_contrast_bounds: I | None
    hidden_form_response_per_capacity_bounds: I | None
    hidden_phase_response_per_capacity_bounds: I | None
    hidden_phase_projection_bounds: tuple[I, ...] | None
    port_hidden_cosine_bounds: tuple[I, ...] | None
    port_form_gradient_rate_without_hidden_bounds: tuple[I, ...]
    port_phase_current_rate_without_hidden_bounds: tuple[I, ...] | None
    channels: tuple[SineCapacityChannel, ...]
    capacity_bounds: I | None
    status: str
    reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_declared_sine_law_held_capacities_support_and_clock",
        "supplied_port_first_rate_bounds_and_prior_acceleration_channels_are_retained",
        "nonport_first_rates_derive_only_from_captured_visible_internal_state",
        "missing_acceleration_channels_are_unavailable_not_zero",
        "paired_exchange_acceleration_cancels_the_form_gradient_rate_algebraically",
        "hidden_capacity_may_be_bounded_while_hidden_phase_remains_unresolved",
        "bounded_candidate_is_conditional_not_existence_or_exact_identification",
        "sensitivity_containing_zero_does_not_certify_exact_stationarity_or_blindness",
        "residual_zero_containment_is_necessary_not_sufficient_for_joint_consistency",
        "source_ids_windows_and_public_reports_do_not_authenticate_prior_evidence",
        "no_hidden_input_graph_recapture_solver_phase_normalization_or_future_prediction",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_inference_labels(self.state_inference)
        for channel in self.channels:
            _validate_label(channel.port)
        return {
            "schema": "tnfr.relational-sine-hidden-capacity.v1",
            "report": _project(self),
        }


def _validate_inference_labels(inference):
    from ..sdk.relational_reports import _validate_label

    for node in (*inference.visible_nodes, *inference.ports, *inference.active_ports):
        _validate_label(node)
    for edge in inference.visible_edges:
        for node in edge:
            _validate_label(node)
    if inference.phase_anchor is not None:
        _validate_label(inference.phase_anchor)
    if inference.identifying_ports is not None:
        for node in inference.identifying_ports:
            _validate_label(node)


def _prior_evidence(
    *, source_id, clock_id, observation_time, evidence_window, forecast_start
):
    """Share declared prior metadata admission without authenticating data."""
    for value, label in ((source_id, "source_id"), (clock_id, "clock_id")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonblank string")
    window = _ordered(evidence_window, "evidence_window", limit=3)
    if len(window) != 2:
        raise ValueError("evidence_window must contain two ordered times")
    start, end = (
        exact_or_represented_real(value, "evidence_window") for value in window
    )
    at = exact_or_represented_real(observation_time, "observation_time")
    forecast = exact_or_represented_real(forecast_start, "forecast_start")
    if not 0 <= start <= at <= end < forecast:
        raise ValueError(
            "require 0<=window_start<=observation_time<=window_end<forecast_start"
        )
    return at, (start, end), forecast


def _intersection(left, right):
    lower, upper = max(left.lo, right.lo), min(left.hi, right.hi)
    return I(lower, upper) if lower <= upper else None


def _prior_channel_map(ports, raw, label, *, optional=False):
    """Retain complete ordered channel association before rebuilding reports."""
    values = _ordered(raw, label, limit=len(ports) + 1)
    if len(values) != len(ports):
        raise ValueError(f"{label} must contain one entry per supplied port")
    result = {}
    for port, value in zip(ports, values):
        if optional and value is None:
            continue
        if isinstance(value, I):
            value = (value.lo, value.hi)
        result[port] = _interval(value, label)
    return {port: (bound.lo, bound.hi) for port, bound in result.items()}


def _rebuild_sine_hidden_state(inference):
    """Recompute the inverse from primitive visible evidence, never its caches."""
    if not isinstance(inference, SineHiddenStateInference):
        raise TypeError("a hidden-state inference report is required")
    ports = _ordered(inference.ports, "ports")
    if len(ports) < 2 or len(set(ports)) != len(ports):
        raise ValueError("at least two distinct supplied ports are required")
    return _infer_sine_hidden_state_data(
        nodes=inference.visible_nodes,
        edges=inference.visible_edges,
        epi=inference.visible_epi,
        phase=inference.visible_phase,
        capacity=inference.visible_capacity,
        ports=ports,
        form_rate_bounds=_prior_channel_map(
            ports, inference.form_rate_bounds, "form_rate_bounds"
        ),
        phase_rate_bounds=_prior_channel_map(
            ports, inference.phase_rate_bounds, "phase_rate_bounds"
        ),
        reference_model=inference.reference_model,
        source_id=inference.source_id,
        clock_id=inference.clock_id,
        observation_time=inference.observation_time,
        evidence_window=inference.evidence_window,
        forecast_start=inference.forecast_start,
    )


def _rebuild_sine_hidden_capacity(inference):
    """Recompute both inverse stages from their retained primitive evidence."""
    if not isinstance(inference, SineHiddenCapacityInference):
        raise TypeError("a hidden-capacity inference report is required")
    if not isinstance(inference.state_inference, SineHiddenStateInference):
        raise TypeError("a hidden-state inference report is required")
    ports = _ordered(inference.state_inference.ports, "ports")
    if len(ports) < 2 or len(set(ports)) != len(ports):
        raise ValueError("at least two distinct supplied ports are required")
    if inference.forecast_start != inference.state_inference.forecast_start:
        raise ValueError("capacity forecast_start must match its prior state")
    # Admit the associated time even though equality alone accepts Booleans.
    exact_or_represented_real(inference.forecast_start, "forecast_start")
    return _infer_hidden_capacity(
        inference.state_inference,
        form_acceleration_bounds=_prior_channel_map(
            ports,
            inference.form_acceleration_bounds,
            "form_acceleration_bounds",
            optional=True,
        ),
        phase_acceleration_bounds=_prior_channel_map(
            ports,
            inference.phase_acceleration_bounds,
            "phase_acceleration_bounds",
            optional=True,
        ),
        source_id=inference.source_id,
        clock_id=inference.clock_id,
        observation_time=inference.observation_time,
        evidence_window=inference.evidence_window,
    )


def _visible_prior_rates(inference):
    """Retain measured port rates and derive only unexposed-node rates."""
    nodes = inference.visible_nodes
    positions = {node: i for i, node in enumerate(nodes)}
    rows = [[] for _ in nodes]
    for left, right in inference.visible_edges:
        i, j = positions[left], positions[right]
        rows[i].append(j)
        rows[j].append(i)
    neighbors = tuple(tuple(row) for row in rows)
    port_indices = tuple(positions[port] for port in inference.ports)
    selected = set(port_indices)
    internal = tuple(i for i in range(len(nodes)) if i not in selected)
    forms, phases, provenance = (
        [None] * len(nodes),
        [None] * len(nodes),
        [None] * len(nodes),
    )
    for p, i in enumerate(port_indices):
        forms[i], phases[i] = (
            inference.form_rate_bounds[p],
            inference.phase_rate_bounds[p],
        )
        provenance[i] = "supplied_prior_port_rate_bounds"
    if internal:
        epi, phase = inference.visible_epi, inference.visible_phase
        gradients = tuple(
            sum((epi[i] - epi[j] for j in neighbors[i]), Q(0)) for i in internal
        )
        currents = tuple(
            sum(
                (I(*certified_sine_bounds(phase[j] - phase[i])) for j in neighbors[i]),
                I(0),
            )
            for i in internal
        )
        field = _sine_work(
            inference.reference_model,
            tuple(len(neighbors[i]) for i in internal),
            gradients,
            tuple(inference.visible_capacity[i] for i in internal),
            currents,
        )
        for position, i in enumerate(internal):
            forms[i], phases[i] = (
                field["form_rates"][position],
                field["phase_rates"][position],
            )
            provenance[i] = "derived_visible_internal_sine_law"
    return neighbors, port_indices, tuple(forms), tuple(phases), tuple(provenance)


def _capacity_intersections(channels):
    """Intersect signed affine readings with one nonnegative capacity."""
    retained, reasons, candidate = [], [], None

    def reason(value):
        if value not in reasons:
            reasons.append(value)

    for channel in channels:
        offset = channel.observed_bounds - channel.baseline_bounds
        slope = channel.sensitivity_bounds
        quotient = None
        if slope == I(0) and not offset.contains(0):
            reason("zero_sensitivity_offset_excludes_zero")
        if (slope.lo >= 0 and offset.hi < 0) or (slope.hi <= 0 and offset.lo > 0):
            reason("nonnegative_capacity_sign_conflict")
        if not slope.contains(0):
            quotient = offset / slope
            if quotient.hi < 0:
                reason("capacity_quotient_is_negative")
            else:
                nonnegative = I(max(Q(0), quotient.lo), quotient.hi)
                if candidate is None:
                    candidate = nonnegative
                else:
                    overlap = _intersection(candidate, nonnegative)
                    if overlap is None:
                        reason("capacity_channel_intersection_empty")
                    else:
                        candidate = overlap
        retained.append(replace(channel, capacity_bounds=quotient))
    if candidate is not None:
        checked = []
        for channel in retained:
            residual = (
                channel.observed_bounds
                - channel.baseline_bounds
                - channel.sensitivity_bounds * candidate
            )
            checked.append(replace(channel, residual_bounds=residual))
            if not residual.contains(0):
                reason("acceleration_residual_excludes_zero")
        retained = checked
    if reasons:
        status, candidate = "inconsistent", None
    elif candidate is None:
        status = "unavailable"
        reasons.append("no_capacity_sensitivity_separated_from_zero")
    else:
        status = "bounded_candidate"
    return tuple(retained), candidate, status, tuple(reasons)


def _infer_hidden_capacity(
    inference,
    *,
    form_acceleration_bounds,
    phase_acceleration_bounds,
    source_id,
    clock_id,
    observation_time,
    evidence_window,
):
    inference = _rebuild_sine_hidden_state(inference)
    at, window, forecast = _prior_evidence(
        source_id=source_id,
        clock_id=clock_id,
        observation_time=observation_time,
        evidence_window=evidence_window,
        forecast_start=inference.forecast_start,
    )
    if clock_id != inference.clock_id or at != inference.observation_time:
        raise ValueError(
            "capacity evidence must match the prior clock_id and observation_time"
        )
    selected = set(inference.ports)
    for values, label in (
        (form_acceleration_bounds, "form_acceleration_bounds"),
        (phase_acceleration_bounds, "phase_acceleration_bounds"),
    ):
        if not isinstance(values, Mapping) or not set(values).issubset(selected):
            raise ValueError(
                f"{label} must map a subset of the supplied ports to endpoint pairs"
            )
    if not form_acceleration_bounds and not phase_acceleration_bounds:
        raise ValueError("at least one prior acceleration channel is required")
    form_acc = tuple(
        (
            _interval(form_acceleration_bounds[port], "form_acceleration_bounds")
            if port in form_acceleration_bounds
            else None
        )
        for port in inference.ports
    )
    phase_acc = tuple(
        (
            _interval(phase_acceleration_bounds[port], "phase_acceleration_bounds")
            if port in phase_acceleration_bounds
            else None
        )
        for port in inference.ports
    )
    neighbors, port_indices, form_rates, phase_rates, provenance = _visible_prior_rates(
        inference
    )
    d_terms = tuple(
        degree * form_rates[i] - sum((form_rates[j] for j in neighbors[i]), I(0))
        for i, degree in zip(port_indices, inference.port_degrees)
    )
    context = dict(
        state_inference=inference,
        source_id=source_id,
        clock_id=clock_id,
        observation_time=at,
        evidence_window=window,
        combined_evidence_window=(
            min(window[0], inference.evidence_window[0]),
            max(window[1], inference.evidence_window[1]),
        ),
        forecast_start=forecast,
        form_acceleration_bounds=form_acc,
        phase_acceleration_bounds=phase_acc,
        visible_form_rate_bounds=form_rates,
        visible_phase_rate_bounds=phase_rates,
        visible_rate_provenance=provenance,
        hidden_phase_available=inference.hidden_unit_phase_relative_to_anchor_bounds
        is not None,
        port_form_gradient_rate_without_hidden_bounds=d_terms,
    )
    prior_reasons = []
    if inference.status == "inconsistent":
        prior_reasons.append("parent_state_inference_inconsistent")
    if any(
        inference.visible_capacity[i] == 0
        and any(
            bound is not None and not bound.contains(0)
            for bound in (form_acc[p], phase_acc[p])
        )
        for p, i in enumerate(port_indices)
    ):
        prior_reasons.append("inactive_port_acceleration_excludes_zero")
    if prior_reasons or inference.hidden_form_bounds is None:
        return SineHiddenCapacityInference(
            **context,
            hidden_form_contrast_bounds=None,
            hidden_form_response_per_capacity_bounds=None,
            hidden_phase_response_per_capacity_bounds=None,
            hidden_phase_projection_bounds=None,
            port_hidden_cosine_bounds=None,
            port_phase_current_rate_without_hidden_bounds=None,
            channels=(),
            capacity_bounds=None,
            status="inconsistent" if prior_reasons else "unavailable",
            reasons=(
                tuple(prior_reasons) if prior_reasons else ("hidden_form_unavailable",)
            ),
        )

    e, w, beta = _sine_model_coefficients(inference.reference_model)
    pi = pi_interval()
    a, b = w / pi, w / (beta * pi)
    k = len(port_indices)
    epi, phase = inference.visible_epi, inference.visible_phase
    mean_form = sum((epi[i] for i in port_indices), Q(0)) / k
    u = inference.hidden_form_bounds - mean_form
    phase_bounds = inference.hidden_unit_phase_relative_to_anchor_bounds
    projections, cosines, preparation_reasons = [], [], []
    anchor = (
        inference.visible_nodes.index(inference.phase_anchor)
        if phase_bounds is not None
        else None
    )
    for p, i in enumerate(port_indices):
        if phase_bounds is None:
            sine = inference.hidden_projection_bounds[p]
            sine = I(-1, 1) if sine is None else sine
            cosine = I(-1, 1)
        else:
            gap = phase[i] - phase[anchor]
            cos_gap = I(*certified_cosine_bounds(gap))
            sin_gap = I(*certified_sine_bounds(gap))
            cosine = _intersection(
                phase_bounds[0] * cos_gap + phase_bounds[1] * sin_gap, I(-1, 1)
            )
            sine = _intersection(
                phase_bounds[1] * cos_gap - phase_bounds[0] * sin_gap, I(-1, 1)
            )
            observed = inference.hidden_projection_bounds[p]
            if sine is not None and observed is not None:
                sine = _intersection(sine, observed)
            if sine is None or cosine is None:
                preparation_reasons.append("hidden_phase_projection_intersection_empty")
                sine = I(-1, 1) if sine is None else sine
                cosine = I(-1, 1) if cosine is None else cosine
        projections.append(sine)
        cosines.append(cosine)
    projections, cosines = tuple(projections), tuple(cosines)
    f = -e * u - a * sum(projections, I(0)) / k
    g = b * u
    c_terms = tuple(
        sum(
            (
                I(*certified_cosine_bounds(phase[j] - phase[i]))
                * (phase_rates[j] - phase_rates[i])
                for j in neighbors[i]
            ),
            I(0),
        )
        - cosine * phase_rates[i]
        for i, cosine in zip(port_indices, cosines)
    )
    channels = []
    for p, i in enumerate(port_indices):
        factor = inference.visible_capacity[i] / inference.port_degrees[p]
        if phase_acc[p] is not None:
            channels.append(
                SineCapacityChannel(
                    inference.ports[p],
                    "phase_acceleration",
                    phase_acc[p],
                    factor * b * d_terms[p],
                    -factor * b * f,
                )
            )
        if form_acc[p] is not None:
            channels.append(
                SineCapacityChannel(
                    inference.ports[p],
                    "form_acceleration",
                    form_acc[p],
                    factor * (-e * d_terms[p] + a * c_terms[p]),
                    factor * (e * f + a * cosines[p] * g),
                )
            )
        if form_acc[p] is not None and phase_acc[p] is not None:
            channels.append(
                SineCapacityChannel(
                    inference.ports[p],
                    "exchange_acceleration",
                    form_acc[p] + e * beta * pi * phase_acc[p] / w,
                    factor * a * c_terms[p],
                    factor * a * cosines[p] * g,
                )
            )
    channels, candidate, status, reasons = _capacity_intersections(channels)
    if preparation_reasons:
        status, candidate = "inconsistent", None
        reasons = tuple(dict.fromkeys((*preparation_reasons, *reasons)))
    return SineHiddenCapacityInference(
        **context,
        hidden_form_contrast_bounds=u,
        hidden_form_response_per_capacity_bounds=f,
        hidden_phase_response_per_capacity_bounds=g,
        hidden_phase_projection_bounds=projections,
        port_hidden_cosine_bounds=cosines,
        port_phase_current_rate_without_hidden_bounds=c_terms,
        channels=channels,
        capacity_bounds=candidate,
        status=status,
        reasons=reasons,
    )


def infer_relational_sine_hidden_state(
    visible_graph,
    *,
    ports,
    form_rate_bounds,
    phase_rate_bounds,
    reference_model,
    source_id,
    clock_id,
    observation_time,
    evidence_window,
    forecast_start,
):
    """Infer conditional hidden-state bounds from strictly prior visible data.

    Every supplied port has one additional unit incidence to the same hidden
    node. Visible-only support may be disconnected or isolated, provided each
    component touches a port. The hidden coordinate and capacity are never
    supplied. Rate maps must contain exactly the ports; their endpoint pairs
    enclose instantaneous form/phase rates at observation_time in clock_id.

    A nonempty source/clock identifier and an ordered prior window record the
    caller's declarations, not authenticated evidence. All times obey
    0<=start<=observation_time<=end<forecast_start. No propagation across the
    remaining time gap or numerical differentiation is performed.
    """
    if (
        not isinstance(reference_model, RelationalExchangeModel)
        or reference_model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular reference model is required")
    _admit_graph(visible_graph, reference_model, require_connected=False)
    nodes = tuple(visible_graph)
    epi, phase, capacity = [], [], []
    for node in nodes:
        data = visible_graph.nodes[node]
        epi.append(Q(_epi(_raw(data, ALIAS_EPI, "EPI"))))
        phase.append(Q(_finite(_raw(data, ALIAS_THETA, "phase"), "phase")))
        capacity.append(Q(_finite(_raw(data, ALIAS_VF, "capacity"), "capacity")))
    return _infer_sine_hidden_state_data(
        nodes=nodes,
        edges=tuple(visible_graph.edges()),
        epi=tuple(epi),
        phase=tuple(phase),
        capacity=tuple(capacity),
        ports=ports,
        form_rate_bounds=form_rate_bounds,
        phase_rate_bounds=phase_rate_bounds,
        reference_model=reference_model,
        source_id=source_id,
        clock_id=clock_id,
        observation_time=observation_time,
        evidence_window=evidence_window,
        forecast_start=forecast_start,
    )


def _infer_sine_hidden_state_data(
    *,
    nodes,
    edges,
    epi,
    phase,
    capacity,
    ports,
    form_rate_bounds,
    phase_rate_bounds,
    reference_model,
    source_id,
    clock_id,
    observation_time,
    evidence_window,
    forecast_start,
):
    """Admit captured primitives and rebuild the conditional inverse exactly.

    The graph entry point materializes represented values once. Detached
    consumers retain exact declared values without recapture, live Gamma
    access or a fitted hidden coordinate. Derived report fields are unused.
    """
    e, w, beta = _sine_model_coefficients(reference_model)
    at, (start, end), forecast = _prior_evidence(
        source_id=source_id,
        clock_id=clock_id,
        observation_time=observation_time,
        evidence_window=evidence_window,
        forecast_start=forecast_start,
    )

    nodes = _ordered(nodes, "visible_nodes")
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise ValueError("at least two distinct ordered visible nodes are required")
    positions = {node: i for i, node in enumerate(nodes)}
    edges = tuple(
        _ordered(edge, "visible edge", limit=3)
        for edge in _ordered(edges, "visible_edges")
    )
    if any(
        len(edge) != 2
        or any(node not in positions for node in edge)
        or edge[0] == edge[1]
        for edge in edges
    ):
        raise ValueError("visible edges must join distinct declared nodes")
    indices = tuple(
        tuple(sorted((positions[left], positions[right]))) for left, right in edges
    )
    if len(set(indices)) != len(indices):
        raise ValueError("visible support must be simple with unique edges")
    support = nx.Graph()
    support.add_nodes_from(nodes)
    support.add_edges_from(edges)

    def values(raw, label):
        raw = _ordered(raw, label, limit=len(nodes) + 1)
        if len(raw) != len(nodes):
            raise ValueError(f"{label} must contain one value per visible node")
        return tuple(exact_or_represented_real(value, label) for value in raw)

    epi, phase, capacity = (
        values(epi, "visible_epi"),
        values(phase, "visible_phase"),
        values(capacity, "visible_capacity"),
    )
    if any(nu < 0 for nu in capacity):
        raise ValueError("capacity must be nonnegative")
    ports = _ordered(ports, "ports")
    if len(ports) < 2 or len(set(ports)) != len(ports):
        raise ValueError("at least two distinct supplied ports are required")
    if any(port not in positions for port in ports):
        raise ValueError("every port must belong to the visible support")
    selected = set(ports)
    if any(
        not (component & selected) for component in nx.connected_components(support)
    ):
        raise ValueError("each visible component must touch a supplied port")
    for mapping, label in (
        (form_rate_bounds, "form_rate_bounds"),
        (phase_rate_bounds, "phase_rate_bounds"),
    ):
        if not isinstance(mapping, Mapping) or set(mapping) != selected:
            raise ValueError(
                f"{label} must map exactly the supplied ports to endpoint pairs"
            )
    form_rates = tuple(
        _interval(form_rate_bounds[port], "form_rate_bounds") for port in ports
    )
    phase_rates = tuple(
        _interval(phase_rate_bounds[port], "phase_rate_bounds") for port in ports
    )

    neighbors = tuple(tuple(positions[j] for j in support[node]) for node in nodes)
    port_indices = tuple(positions[port] for port in ports)
    degrees = tuple(len(neighbors[i]) + 1 for i in port_indices)
    gradients = tuple(
        sum((epi[i] - epi[j] for j in neighbors[i]), Q(0)) for i in port_indices
    )
    currents = tuple(
        sum(
            (I(*certified_sine_bounds(phase[j] - phase[i])) for j in neighbors[i]), I(0)
        )
        for i in port_indices
    )
    pi = pi_interval()
    active = tuple(p for p, i in enumerate(port_indices) if capacity[i] > 0)
    forms, raw_projections, projections = [], [], []
    reasons, incompatible = [], False

    def reason(value, *, inconsistent=False):
        nonlocal incompatible
        if value not in reasons:
            reasons.append(value)
        incompatible |= inconsistent

    for p, i in enumerate(port_indices):
        nu, degree = capacity[i], degrees[p]
        if not nu:
            forms.append(None)
            raw_projections.append(None)
            projections.append(None)
            if not form_rates[p].contains(0) or not phase_rates[p].contains(0):
                reason("inactive_port_rate_excludes_zero", inconsistent=True)
            continue
        forms.append(
            epi[i] + gradients[p] - beta * pi * degree * phase_rates[p] / (w * nu)
        )
        # Hidden form cancels algebraically between the two measured rows.
        projection = (
            pi * degree * form_rates[p] / (w * nu)
            + beta * e * pi**2 * degree * phase_rates[p] / (w**2 * nu)
            - currents[p]
        )
        raw_projections.append(projection)
        bounded = _intersection(projection, I(-1, 1))
        projections.append(bounded)
        if bounded is None:
            reason("hidden_phase_projection_outside_unit_interval", inconsistent=True)

    form_bounds = forms[active[0]] if active else None
    for p in active[1:]:
        if form_bounds is not None:
            form_bounds = _intersection(form_bounds, forms[p])
            if form_bounds is None:
                reason("hidden_form_intersection_empty", inconsistent=True)

    chosen, determinant, strongest, gram = None, None, Q(0), I(0)
    for offset, p in enumerate(active):
        for other in active[offset + 1 :]:
            difference = phase[port_indices[other]] - phase[port_indices[p]]
            bound = I(*certified_sine_bounds(difference))
            gram += bound**2
            separation = max(Q(0), bound.lo, -bound.hi)
            if separation > strongest:
                strongest, chosen, determinant = separation, (p, other), bound

    anchor = pair = phase_raw = phase_bounds = norm_bounds = projection_rows = None
    residuals = (None,) * len(ports)
    if chosen is not None:
        rank = 2
        p, other = chosen
        anchor, pair = ports[p], (ports[p], ports[other])
        delta = phase[port_indices[other]] - phase[port_indices[p]]
        first = projections[p] if projections[p] is not None else raw_projections[p]
        second = (
            projections[other]
            if projections[other] is not None
            else raw_projections[other]
        )
        sine = first
        cosine = (I(*certified_cosine_bounds(delta)) * first - second) / determinant
        phase_raw = (cosine, sine)
        cosine = _intersection(cosine, I(-1, 1))
        sine = _intersection(sine, I(-1, 1))
        if cosine is None or sine is None:
            reason(
                "hidden_unit_phase_coordinate_outside_unit_interval", inconsistent=True
            )
        else:
            phase_bounds = (cosine, sine)
            norm_bounds = cosine**2 + sine**2
            if not norm_bounds.contains(1):
                reason("unit_circle_excluded", inconsistent=True)
            rows, checks = [], []
            for port_pos, i in enumerate(port_indices):
                gap = phase[i] - phase[port_indices[p]]
                row = (
                    -I(*certified_sine_bounds(gap)),
                    I(*certified_cosine_bounds(gap)),
                )
                rows.append(row)
                check = None
                if port_pos in active:
                    check = row[0] * cosine + row[1] * sine - raw_projections[port_pos]
                    if not check.contains(0):
                        reason(
                            "port_phase_projection_residual_excludes_zero",
                            inconsistent=True,
                        )
                checks.append(check)
            projection_rows, residuals = tuple(rows), tuple(checks)
    elif not active:
        rank = 0
        reason("no_active_ports")
    elif len(active) == 1:
        rank = 1
        reason("single_active_port")
    elif all(phase[port_indices[p]] == phase[port_indices[active[0]]] for p in active):
        rank = 1
        reason("rank_one_equal_visible_phases")
        common_projection = projections[active[0]]
        for p in active[1:]:
            if common_projection is not None and projections[p] is not None:
                common_projection = _intersection(common_projection, projections[p])
                if common_projection is None:
                    reason("equal_phase_projections_are_disjoint", inconsistent=True)
    else:
        rank = None
        reason("phase_geometry_unresolved")

    if incompatible:
        status = "inconsistent"
        phase_bounds = None
    elif form_bounds is None or phase_bounds is None:
        status = "unavailable"
    else:
        status = "bounded_candidate"
    return SineHiddenStateInference(
        reference_model=reference_model,
        source_id=source_id,
        clock_id=clock_id,
        observation_time=at,
        evidence_window=(start, end),
        forecast_start=forecast,
        visible_nodes=nodes,
        visible_edges=edges,
        visible_epi=epi,
        visible_phase=phase,
        visible_capacity=capacity,
        ports=ports,
        port_degrees=degrees,
        active_ports=tuple(ports[p] for p in active),
        form_rate_bounds=form_rates,
        phase_rate_bounds=phase_rates,
        internal_form_gradients=gradients,
        internal_phase_currents=currents,
        hidden_form_by_port=tuple(forms),
        hidden_form_bounds=form_bounds,
        hidden_projection_raw_bounds=tuple(raw_projections),
        hidden_projection_bounds=tuple(projections),
        phase_rank=rank,
        phase_anchor=anchor,
        identifying_ports=pair,
        phase_determinant_bounds=determinant,
        phase_gram_determinant_bounds=gram,
        hidden_unit_phase_raw_bounds=phase_raw,
        hidden_unit_phase_relative_to_anchor_bounds=phase_bounds,
        unit_norm_squared_bounds=norm_bounds,
        projection_rows=projection_rows,
        projection_residuals=residuals,
        status=status,
        reasons=tuple(reasons),
    )
