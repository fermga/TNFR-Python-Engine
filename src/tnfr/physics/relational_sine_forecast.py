"""Prior-admitted finite future enclosures for the supplied smooth sine law.

Prior inference gives necessary bounds, not joint existence. A separate exact
dyadic witness must pass every observed first/second rate interval by forward
containment before a prior forecast is admitted. The witness never replaces
the complete initial uncertainty box used by the shared validated solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from functools import lru_cache

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._comparison_flow import MAX_COMPARISON_DIMENSION, _exact, _ordered
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import I, arg, cos, sin
from ..mathematics._validated_taylor import (
    ValidatedTaylorStep,
    flow_jets,
    validated_taylor_step,
)
from ._sine_admission import _sine_model_coefficients
from .relational_sine_comparison import _sine_rates
from .relational_sine_observation import (
    SineHiddenCapacityInference,
    _rebuild_sine_hidden_capacity,
    _validate_inference_labels,
)

__all__ = (
    "SinePriorAdmission",
    "SineForecast",
    "admit_sine_prior",
    "bound_sine_flow",
    "forecast_sine_prior",
)

MAX_FORECAST_STEPS = 256


@dataclass(frozen=True)
class SinePriorAdmission:
    """A jointly compatible witness and a distinct complete prior box.

    Coordinates are visible forms, hidden form, visible phase lifts, hidden
    phase lift, hidden capacity. The final capacity is held by a zero rate.
    A witness establishes existence within declared evidence intervals, not
    uniqueness, calibration, provenance authentication or physical truth.
    """

    capacity_inference: SineHiddenCapacityInference
    neighbors: tuple[tuple[int, ...], ...]
    visible_capacity: tuple[Q, ...]
    initial_box: tuple[I, ...] | None
    hidden_phase_lift_bounds: I | None
    joint_witness: tuple[Q, ...] | None
    witness_first_rate_bounds: tuple[I, ...] | None
    witness_acceleration_bounds: tuple[I, ...] | None
    status: str
    reasons: tuple[str, ...]
    witness_grid: Q = Q(1, 1 << 20)
    scope: tuple[str, ...] = (
        "prior_observation_time_not_evidence_end_or_forecast_start",
        "one_supplied_hidden_star_and_held_capacities_with_no_input_or_events",
        "certified_positive_cosine_chart_of_the_prior_hidden_unit_phase",
        "exact_dyadic_witness_checked_by_forward_rate_and_acceleration_containment",
        "witness_is_not_an_estimate_or_replacement_for_the_initial_uncertainty_box",
        "joint_existence_within_declared_intervals_not_uniqueness_or_authentication",
        "same_normalized_sine_reciprocal_law_not_native_Arg_dynamics",
        "no_graph_read_new_measurement_or_evaluated_future_response",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_prior_labels(self)
        return {
            "schema": "tnfr.relational-sine-prior-admission.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineForecast:
    """Fixed-budget Picard/Taylor certificates in the full state coordinates.

    Each endpoint encloses every solution from the supplied initial box.
    Every step retains its whole-time tube, initial-radius propagation and
    local remainder. An unresolved step is unavailable, never an adaptive
    shorter-step success. A prior source, when present, remains attached.
    """

    model: RelationalExchangeModel
    neighbors: tuple[tuple[int, ...], ...]
    visible_capacity: tuple[Q, ...]
    initial_box: tuple[I, ...]
    observation_time: Q
    end_time: Q
    time_step: Q
    order: int
    steps: tuple[ValidatedTaylorStep, ...]
    validated_end_time: Q
    endpoint: tuple[I, ...]
    failed_tube: tuple[I, ...] | None
    status: str
    reasons: tuple[str, ...]
    prior_admission: SinePriorAdmission | None = None
    forecast_start: Q | None = None
    freeze_hidden: bool = False
    method: str = "full_sine_held_capacity_Picard_Taylor_Metzler128_v1"
    scope: tuple[str, ...] = (
        "fixed_simple_connected_unit_support_and_explicit_smooth_sine_law",
        "full_phase_lifts_not_rounded_phase_midpoints_or_normalized_phasors",
        "held_nonnegative_capacity_uncertainty_is_an_augmented_zero_rate_coordinate",
        "exact_held_capacity_and_zero_capacity_row_invariants_tighten_endpoints",
        "every_initial_point_enclosed_using_whole_time_Picard_and_Taylor_certificates",
        "propagation_starts_at_actual_observation_time_including_the_preforecast_gap",
        "fixed_order_and_step_budget_without_adaptive_retries_or_empirical_error_estimates",
        "global_smooth_extension_admits_tubes_while_initial_held_capacity_stays_nonnegative",
        "no_runtime_dispatch_graph_mutation_events_or_physical_validation",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        if self.prior_admission is not None:
            _validate_prior_labels(self.prior_admission)
        return {
            "schema": "tnfr.relational-sine-forecast.v1",
            "report": _project(self),
        }


def _validate_prior_labels(admission):
    from ..sdk.relational_reports import _validate_label

    source = admission.capacity_inference
    _validate_inference_labels(source.state_inference)
    for channel in source.channels:
        _validate_label(channel.port)


def _sine_flow(state, *, neighbors, visible_capacity, model):
    """Shared sine rates for interval states or matched formal Taylor jets.

    Caller admission fixes support and dimensions. The added final capacity
    coordinate propagates its uncertainty through the same Jacobian and
    comparison flow as all other state coordinates; it is not a fitted rate.
    """
    size = len(neighbors)
    epi, phase, hidden_capacity = state[:size], state[size : 2 * size], state[-1]
    is_jet = isinstance(state[0], Jet)
    zero = Jet.constant(0, state[0].order) if is_jet else I(0)
    capacities = tuple(
        Jet.constant(value, state[0].order) if is_jet else I(value)
        for value in visible_capacity
    ) + (hidden_capacity,)
    sine = jet_sin if is_jet else sin
    gradient = tuple(
        sum((epi[i] - epi[j] for j in row), zero) for i, row in enumerate(neighbors)
    )
    currents = tuple(
        sum((sine(phase[j] - phase[i]) for j in row), zero)
        for i, row in enumerate(neighbors)
    )
    rates = _sine_rates(
        model, tuple(map(len, neighbors)), gradient, capacities, currents
    )
    return rates["form_rates"] + rates["phase_rates"] + (zero,)


def _admit_support(neighbors, visible_capacity, model):
    _sine_model_coefficients(model)
    neighbors = tuple(
        tuple(_ordered(row, "neighbor row")) for row in _ordered(neighbors, "neighbors")
    )
    size = len(neighbors)
    if not 2 <= size or 2 * size + 1 > MAX_COMPARISON_DIMENSION:
        raise ValueError(
            "require at least two nodes and at most "
            f"{MAX_COMPARISON_DIMENSION} state coordinates"
        )
    for i, row in enumerate(neighbors):
        if not row or any(type(j) is not int or not 0 <= j < size for j in row):
            raise ValueError("neighbor indices must be nonempty in-range integer rows")
        if len(set(row)) != len(row) or i in row:
            raise ValueError("support must be simple and loop-free")
        if any(i not in neighbors[j] for j in row):
            raise ValueError("support must be undirected")
    reached, pending = {0}, [0]
    while pending:
        for j in neighbors[pending.pop()]:
            if j not in reached:
                reached.add(j)
                pending.append(j)
    if len(reached) != size:
        raise ValueError("support must be connected")
    capacity = tuple(_exact(value) for value in _ordered(visible_capacity, "capacity"))
    if len(capacity) != size - 1 or any(value < 0 for value in capacity):
        raise ValueError(
            "one nonnegative exact held capacity per visible node required"
        )
    return neighbors, capacity


def _prior_neighbors(state):
    positions = {node: i for i, node in enumerate(state.visible_nodes)}
    hidden = len(positions)
    rows = [[] for _ in range(hidden + 1)]
    for left, right in state.visible_edges:
        i, j = positions[left], positions[right]
        rows[i].append(j)
        rows[j].append(i)
    for port in state.ports:
        index = positions[port]
        rows[index].append(hidden)
        rows[hidden].append(index)
    return tuple(tuple(row) for row in rows)


def _round_witness(value):
    """Choose an exact dyadic witness, independently of forecast uncertainty."""
    return Q(round(value * (1 << 20)), 1 << 20)


def admit_sine_prior(capacity_inference):
    """Check one joint witness without treating necessary overlap as existence.

    This bounded method uses only a certified positive-real relative phase
    chart and one fixed dyadic witness candidate. Failure to certify the
    candidate is unavailability, not proof that no compatible state exists.
    The full initial box remains a conservative enclosure of every compatible
    hidden state and capacity from the prior inference.

    Retained primitive observations and metadata are read again; cached
    inference bounds or verdicts cannot replace that declared evidence.
    """
    admission = _rebuild_sine_prior(capacity_inference)
    if capacity_inference == admission.capacity_inference:
        admission = replace(admission, capacity_inference=capacity_inference)
    return admission


def _rebuild_sine_prior(capacity_inference):
    """Return normalized joint admission for calculation, before association."""
    if not isinstance(capacity_inference, SineHiddenCapacityInference):
        raise TypeError("a hidden-capacity inference report is required")
    capacity_inference = _rebuild_sine_hidden_capacity(capacity_inference)
    state = capacity_inference.state_inference
    neighbors, capacities = _admit_support(
        _prior_neighbors(state), state.visible_capacity, state.reference_model
    )
    initial = angle = witness = first = acceleration = None
    reasons = []
    if state.status == "inconsistent" or capacity_inference.status == "inconsistent":
        reasons.append("prior_evidence_inconsistent")
    if state.hidden_form_bounds is None or capacity_inference.capacity_bounds is None:
        reasons.append("prior_hidden_form_or_capacity_unavailable")
    phase = state.hidden_unit_phase_relative_to_anchor_bounds
    if phase is None or state.phase_anchor is None:
        reasons.append("prior_hidden_unit_phase_unavailable")
    elif phase[0].lo <= 0:
        reasons.append("positive_real_hidden_phase_chart_unresolved")
    if not reasons:
        anchor = state.visible_phase[state.visible_nodes.index(state.phase_anchor)]
        angle = anchor + arg(*phase)
        initial = (
            tuple(I(value) for value in state.visible_epi)
            + (state.hidden_form_bounds,)
            + tuple(I(value) for value in state.visible_phase)
            + (angle, capacity_inference.capacity_bounds)
        )
        size = len(neighbors)
        candidate = (
            state.visible_epi
            + (_round_witness(state.hidden_form_bounds.midpoint),)
            + state.visible_phase
            + (_round_witness(angle.midpoint),)
            + (_round_witness(capacity_inference.capacity_bounds.midpoint),)
        )
        if not all(box.contains(value) for box, value in zip(initial, candidate)):
            reasons.append("dyadic_joint_witness_outside_prior_box")
        relative = I(candidate[2 * size - 1] - anchor)
        if not phase[0].contains(cos(relative)) or not phase[1].contains(sin(relative)):
            reasons.append("unit_phase_witness_not_contained_in_prior_rectangle")
        if not reasons:

            def flow(values):
                return _sine_flow(
                    values,
                    neighbors=neighbors,
                    visible_capacity=capacities,
                    model=state.reference_model,
                )

            series = flow_jets(tuple(I(value) for value in candidate), 2, flow)
            first = tuple(row[1] for row in series)
            acceleration = tuple(2 * row[2] for row in series)
            for p, port in enumerate(state.ports):
                index = state.visible_nodes.index(port)
                readings = (
                    (state.form_rate_bounds[p], first[index]),
                    (state.phase_rate_bounds[p], first[size + index]),
                    (
                        capacity_inference.form_acceleration_bounds[p],
                        acceleration[index],
                    ),
                    (
                        capacity_inference.phase_acceleration_bounds[p],
                        acceleration[size + index],
                    ),
                )
                if any(
                    obs is not None and not obs.contains(value)
                    for obs, value in readings
                ):
                    reasons.append(
                        "joint_witness_forward_bounds_not_contained_in_evidence"
                    )
                    break
            if not reasons:
                witness = candidate
    return SinePriorAdmission(
        capacity_inference,
        neighbors,
        capacities,
        initial,
        angle,
        witness,
        first,
        acceleration,
        "unavailable" if reasons else "admitted",
        tuple(reasons),
    )


def bound_sine_flow(
    initial,
    *,
    neighbors,
    visible_capacity,
    model,
    observation_time,
    end_time,
    time_step,
    order=6,
):
    """Enclose a supplied full initial box with a fixed rational time budget.

    Initial values are exact integers, Fractions or rational intervals in
    the documented layout. Phase coordinates are real lifts of unit phases,
    not Cartesian coordinates to normalize. The hidden capacity is the last
    coordinate and has identically zero rate. The smooth sine law has no
    resultant-chart boundary; its full real extension admits Picard tubes.
    The shared 24-coordinate work cap admits up to eleven complete nodes
    plus the held-capacity coordinate; it is not a physical dimension bound.
    Taylor order, step count and comparison-tail domains remain independent
    numerical admission limits.
    """
    neighbors, capacities = _admit_support(neighbors, visible_capacity, model)
    initial = tuple(I.coerce(value) for value in _ordered(initial, "initial state"))
    if len(initial) != 2 * len(neighbors) + 1:
        raise ValueError("state layout requires all forms, all phases, hidden capacity")
    if initial[-1].lo < 0:
        raise ValueError("initial hidden capacity must be nonnegative")
    at, end, step_size = map(_exact, (observation_time, end_time, time_step))
    if (
        not 0 <= at < end
        or step_size <= 0
        or (end - at) / step_size > MAX_FORECAST_STEPS
    ):
        raise ValueError(
            f"require 0<=observation_time<end_time and at most {MAX_FORECAST_STEPS} positive steps"
        )
    if type(order) is not int or not 1 <= order <= 16:
        raise ValueError("Taylor order must be an integer from 1 to 16")

    @lru_cache(maxsize=32)
    def flow(values):
        return _sine_flow(
            values, neighbors=neighbors, visible_capacity=capacities, model=model
        )

    def domain(values):
        # A real lift of sine is globally smooth, including zero resultants.
        return (Q(1),)

    time, box, steps, reasons, failed = at, initial, [], [], None
    size = len(neighbors)
    constant_indices = [2 * size]
    for index, capacity in enumerate(capacities + (initial[-1],)):
        if capacity == 0 or isinstance(capacity, I) and capacity.lo == capacity.hi == 0:
            constant_indices.extend((index, size + index))
    while time < end:
        duration = min(step_size, end - time)
        certificate, failed, reason = validated_taylor_step(
            box, duration, flow, domain, order=order, time=time
        )
        if certificate is None:
            reasons.append(reason)
            break
        endpoint = list(certificate.endpoint)
        for index in constant_indices:
            value, held = endpoint[index], initial[index]
            lower, upper = max(value.lo, held.lo), min(value.hi, held.hi)
            if lower > upper:
                raise ArithmeticError(
                    "validated endpoint contradicts an exact held invariant"
                )
            endpoint[index] = I(lower, upper)
        certificate = replace(certificate, endpoint=tuple(endpoint))
        steps.append(certificate)
        box = certificate.endpoint
        time += duration
    return SineForecast(
        model,
        neighbors,
        capacities,
        initial,
        at,
        end,
        step_size,
        order,
        tuple(steps),
        time,
        box,
        failed,
        "unavailable" if reasons else "admitted",
        tuple(reasons),
    )


def forecast_sine_prior(
    admission, *, end_time, time_step, order=6, freeze_hidden=False
):
    """Propagate the full admitted box from its actual prior observation time.

    ``freeze_hidden=True`` is an explicitly changed-capacity control: only
    the initial hidden-capacity coordinate is replaced by exact zero. It is
    not claimed to fit the original prior accelerations, and does not alter
    the attached original admission or re-estimate any coordinate.

    Joint admission and its full box are rebuilt from the retained primitive
    evidence before propagation. An equivalent original report stays attached;
    stale derived fields are replaced without authenticating the observations.
    """
    if not isinstance(admission, SinePriorAdmission):
        raise TypeError("a sine prior admission report is required")
    if type(freeze_hidden) is not bool:
        raise TypeError("freeze_hidden must be a Boolean")
    original = admission
    admission = _rebuild_sine_prior(admission.capacity_inference)
    if not admission.admitted or admission.initial_box is None:
        raise ValueError("a jointly witnessed prior admission is required")
    prior = admission.capacity_inference
    end = _exact(end_time)
    if end < prior.forecast_start:
        raise ValueError("end_time must be at or after the declared forecast_start")
    initial = admission.initial_box
    if freeze_hidden:
        initial = initial[:-1] + (I(0),)
    report = bound_sine_flow(
        initial,
        neighbors=admission.neighbors,
        visible_capacity=admission.visible_capacity,
        model=prior.state_inference.reference_model,
        observation_time=prior.observation_time,
        end_time=end,
        time_step=time_step,
        order=order,
    )
    return replace(
        report,
        prior_admission=original if original == admission else admission,
        forecast_start=prior.forecast_start,
        freeze_hidden=freeze_hidden,
    )
