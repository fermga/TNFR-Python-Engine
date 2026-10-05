"""Bounded continuous enclosures for unequal reflected source/receiver rings.

This proof computation consumes exact interval coordinates, not a graph to
project onto symmetry. It reuses the native-law restriction and shared Taylor
kernel; it is neither an Euler execution path nor an automatic capture test.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._comparison_flow import _exact, _ordered
from ..mathematics._rational_interval import I, arg, cos, pi_interval
from ..mathematics._validated_taylor import ValidatedTaylorStep, validated_taylor_step
from ._relational_reflected_flow import evaluate_reflected_regular_flow

__all__ = (
    "RelationalReflectedBarrierCertificate",
    "RelationalReflectedTransitCertificate",
    "certify_relational_reflected_barrier",
    "certify_relational_reflected_transit",
)


@dataclass(frozen=True)
class ReflectedTransitObservation:
    time: Q
    storage: I
    integrated_loss: I
    cumulative_loss: I
    target_budget: I
    source_argument: I
    receiver_winding_zero_on_tube: bool
    source_winding_one_on_tube: bool


@dataclass(frozen=True)
class RelationalReflectedTransitCertificate:
    initial_box: tuple[I, ...]
    model: RelationalExchangeModel
    horizon: Q
    time_step: Q
    order: int
    steps: tuple[ValidatedTaylorStep, ...]
    observations: tuple[ReflectedTransitObservation, ...]
    validated_horizon: Q
    endpoint: tuple[I, ...]
    initial_storage: I
    endpoint_storage: I
    cumulative_loss: I
    target_storage: I
    target_budget: I
    initial_source_argument: I
    endpoint_source_argument: I
    source_winding_one_throughout: bool
    receiver_winding_zero_throughout: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    failed_tube: tuple[I, ...] | None
    method: str = "reflected8_regular_centered_Taylor_Picard_Metzler128_v1"
    scope: tuple[str, ...] = (
        "two_unit_C5_rings_matching_adjacent_bridges_unit_held_capacity",
        "exact_joint_reflection_without_equal_source_receiver_states",
        "coordinates_p_r_P_R_a_b_A_B_no_graph_projection",
        "unforced_capacity_separable_regular_law_no_events",
        "outward_initial_uncertainty_and_whole_time_domain_evidence",
        "no_capture_global_continuation_or_physical_identification",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-reflected-transit.v1",
            "report": _project(self),
        }


def _intersection(left, right):
    lo, hi = max(left.lo, right.lo), min(left.hi, right.hi)
    if lo > hi:
        raise ArithmeticError("disjoint direct and integrated storage enclosures")
    return I(lo, hi)


def _winding(box, offset, sector):
    """Sufficient whole-box ordinary winding; false means not certified."""
    pi = pi_interval()
    a, b = box[4 + offset], box[5 + offset]
    if not (a - b).abs_max < pi.lo or not b.abs_max < pi.lo:
        return False
    if sector == 0:
        return (2 * a).abs_max < pi.lo
    return a.lo > pi.hi / 2 and a.hi < pi.lo


def _lift_margins(state):
    pi = pi_interval()
    return (
        pi.lo - state[4].abs_max,
        pi.lo / 2 - state[5].abs_max,
        pi.lo - state[6].abs_max,
        pi.lo / 2 - state[7].abs_max,
    )


@dataclass(frozen=True)
class RelationalReflectedBarrierCertificate:
    """Conditional future exclusion by a geometric separator, not evolution."""

    initial_box: tuple[I, ...]
    model: RelationalExchangeModel
    storage: I
    barrier_storage: I
    phase_sum: I
    separator: I
    energy_deficit: I
    separator_margin: I
    domain_lower_bounds: tuple[Q, ...]
    status: str
    unavailable_reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "two_unit_C5_rings_matching_adjacent_bridges_exact_joint_reflection",
        "unit_held_capacity_regular_unforced_fixed_support_law",
        "same_regular_continuation_cannot_reach_both_acute_winding_plus_one_rings",
        "no_exclusion_of_single_receiver_formation_nonacute_patterns_or_other_preparations",
        "snapshot_theorem_not_new_trajectory_or_physical_identification",
    )

    @property
    def obstructed(self):
        return self.status == "certified"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-reflected-barrier.v1",
            "report": _project(self),
        }


def certify_relational_reflected_barrier(initial_box, *, model):
    """Exclude the two acute +1 rings across the exact 7*beta separator.

    In the regular reflection lift, the face a+A=4*pi/3 has phase storage
    at least 7*beta, including the supplied bridges. Each target acute +1
    ring requires a>3*pi/4 or A>3*pi/4 respectively. Strictly lower storage
    on the lower side cannot cross this face under the unforced loss law.
    Zero resultants on lift faces prohibit evading the separator through
    another continuous regular lift. Failure of this sufficient test is
    unavailability, never a formation or reachability certificate.
    """
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular relational model is required")
    state = tuple(I.coerce(value) for value in _ordered(initial_box, "state"))
    e, w = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)
    field = evaluate_reflected_regular_flow(
        state, epi_weight=e, phase_weight=w, storage_scale=beta
    )
    margins = field.resultant_margin_lower_bounds + _lift_margins(state)
    if min(margins) <= 0:
        raise ValueError("initial reflection lift is not in the declared component")
    barrier, separator = I(7 * beta), 4 * pi_interval() / 3
    phase_sum = state[4] + state[6]
    deficit, distance = barrier - field.storage, separator - phase_sum
    reasons = []
    if deficit.lo <= 0:
        reasons.append("storage_not_strictly_below_separating_barrier")
    if distance.lo <= 0:
        reasons.append("lower_separator_component_not_certified")
    return RelationalReflectedBarrierCertificate(
        state,
        model,
        field.storage,
        barrier,
        phase_sum,
        separator,
        deficit,
        distance,
        margins,
        "unavailable" if reasons else "certified",
        tuple(reasons),
    )


def certify_relational_reflected_transit(
    initial_box, *, model, horizon, time_step, order=6
) -> RelationalReflectedTransitCertificate:
    """Certify a declared finite horizon or retain the first unresolved tube.

    Coordinates are exact integers, Fractions or outward intervals. The
    explicit regular model supplies effective coefficients; unit capacities,
    fixed support and reflection are premises of this restricted model.
    No graph is inferred or repaired, and no shorter-step retry is attempted.
    The 256-step cap is a numerical work policy, not a physical horizon.
    """
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular relational model is required")
    horizon, time_step = _exact(horizon), _exact(time_step)
    if horizon <= 0 or time_step <= 0 or horizon / time_step > 256:
        raise ValueError("positive horizon/step with at most 256 proof steps required")
    if type(order) is not int or not 1 <= order <= 16:
        raise ValueError("Taylor order must be an integer from 1 to 16")
    box = initial_box = tuple(
        I.coerce(value) for value in _ordered(initial_box, "state")
    )
    e, w = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)

    @lru_cache(maxsize=32)
    def field(state):
        return evaluate_reflected_regular_flow(
            state, epi_weight=e, phase_weight=w, storage_scale=beta
        )

    def flow(state):
        return field(state).rates

    def domain(state):
        return field(state).resultant_margin_lower_bounds + _lift_margins(state)

    initial = field(initial_box)
    if min(domain(initial_box)) <= 0:
        raise ValueError("initial reflection lift is not in the declared component")
    initial_storage = storage = initial.storage
    initial_argument = argument = arg(*initial.resultants[0])
    pi = pi_interval()
    target = beta * 10 * (1 - cos(2 * pi / 5))
    loss = I(0)
    steps, observations, reasons = [], [], []
    time, failed = Q(0), None
    source_kept = _winding(initial_box, 0, 1)
    receiver_kept = _winding(initial_box, 2, 0)
    while time < horizon:
        duration = min(time_step, horizon - time)
        step, failed, reason = validated_taylor_step(
            box, duration, flow, domain, order=order, time=time
        )
        if step is None:
            reasons.append(reason)
            break
        try:
            increment = field(step.tube).continuous_loss * duration
            next_loss = loss + increment
            endpoint_field = field(step.endpoint)
            next_storage = _intersection(
                endpoint_field.storage, initial_storage - next_loss
            )
            # The loss identity and nonnegative interval square terms justify
            # these intersections; they never assign a favorable zero defect.
            next_loss = _intersection(next_loss, initial_storage - next_storage)
            next_argument = arg(*endpoint_field.resultants[0])
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            reasons.append(f"storage_or_endpoint_observation_unavailable: {exc}")
            failed = step.tube
            break
        source_ok = _winding(step.tube, 0, 1)
        receiver_ok = _winding(step.tube, 2, 0)
        source_kept = source_kept and source_ok
        receiver_kept = receiver_kept and receiver_ok
        time += duration
        box, loss, storage, argument = (
            step.endpoint,
            next_loss,
            next_storage,
            next_argument,
        )
        steps.append(step)
        observations.append(
            ReflectedTransitObservation(
                time,
                storage,
                increment,
                loss,
                storage - target,
                argument,
                receiver_ok,
                source_ok,
            )
        )
    if time != horizon:
        reasons.append("requested_horizon_not_validated")
    return RelationalReflectedTransitCertificate(
        initial_box,
        model,
        horizon,
        time_step,
        order,
        tuple(steps),
        tuple(observations),
        time,
        box,
        initial_storage,
        storage,
        loss,
        target,
        storage - target,
        initial_argument,
        argument,
        source_kept,
        receiver_kept,
        "unavailable" if reasons else "admitted",
        tuple(reasons),
        failed,
    )
