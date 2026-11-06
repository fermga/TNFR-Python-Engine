"""Analytic contact of actual formed C9 families under one supplied sine law.

Uninterrupted isolated recovery precedes a supplied unit bridge. A correlated
fourth-order receiver comparison and a full-support barrier retain all source
uncertainty. No ideal reset, trajectory solver or phase-probe report is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_formed_contact import _joined_contact_bounds, _unprobed_handoff
from .relational_sine_formed_classes import SineFormedClassPair

__all__ = ("SineFormedClassContact", "assess_sine_formed_class_contact")


@dataclass(frozen=True)
class SineFormedClassContact:
    """Conditional receiver contrast and identity after a supplied contact.

    The receiver is class one; donor cases follow the fresh pair's order.
    Endpoint budgets bound the actual unprobed source images. Ideal targets
    enter only the analytic comparison. Unavailable handoffs leave actual
    response, event and retention bounds unavailable, not zero or passing.
    """

    formation_certificate: SineFormedClassPair
    relaxation_duration: Q
    contact_time: Q
    original_contact_time: Q
    phase_origin_difference: Q
    contact_duration: Q
    endpoint_radius: Q
    readout_error_bound: Q
    work_allowance: Q
    decay_power: int
    lyapunov_decay_rate: Q | None
    decay_exponent: Q | None
    exact_decay_upper_bound: Q | None
    initial_lyapunov_upper_bounds: tuple[Q, Q]
    endpoint_form_norm_squared_upper_bounds: tuple[Q, Q] | None
    endpoint_phase_norm_squared_upper_bounds: tuple[Q, Q] | None
    handoff_certified_by_class: tuple[bool, bool]
    ideal_fourth_derivative_contrast_bounds: I
    ideal_leading_contrast_bounds: I
    semigroup_tail_upper_bound: Q
    nonlinear_remainder_upper_bound: Q
    preparation_response_error_upper_bound: Q | None
    recorded_contrast_bounds: I | None
    disconnected_recorded_contrast_bounds: I | None
    joined_radius_squared_upper_bound: Q | None
    joined_excess_storage_upper_bound: Q | None
    joined_barrier_lower_bound: Q
    joined_radius_margin_bounds: I | None
    joined_storage_margin_bounds: I | None
    bridge_work_bounds: I | None
    work_margin_bounds: I | None
    joined_form_mean_bounds: I | None
    joined_phase_mean_bounds: I | None
    response_certified: bool
    identity_certified: bool
    work_within_allowance: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    receiver_class: str = "winding_one"
    bridge: tuple[int, int] = (4, 13)
    receiver_node: int = 13
    nodes: tuple[int, ...] = tuple(range(18))
    edges: tuple[tuple[int, int], ...] = tuple(
        sorted(
            ((4, 13),)
            + tuple(
                (min(s + j, s + (j + 1) % 9), max(s + j, s + (j + 1) % 9))
                for s in (0, 9)
                for j in range(9)
            )
        )
    )
    joined_degrees: tuple[int, ...] = tuple(3 if j in (4, 13) else 2 for j in range(18))
    joined_gap_lower_bound: Q = Q(2, 81)
    joined_cosine_lower_bound: Q = Q(1, 20)
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "actual_unprobed_original_source_families_with_separate_exact_zero_sums",
        "receiver_phase_origin_declared_from_preparation_not_a_contact_reset",
        "fresh_formation_and_uninterrupted_nonlinear_recovery_no_incoming_report",
        "unit_central_bridge_changes_degrees_and_memberwise_weighted_means",
        "same_interface_and_receiver_preparation_for_both_donor_classes",
        "correlated_fourth_order_ideal_comparison_with_full_nonlinear_remainder",
        "whole_actual_family_error_and_independent_readout_errors_retained",
        "full_support_barrier_retains_both_windings_and_gives_later_recovery",
        "disconnected_receiver_history_independent_of_donor_under_the_same_law",
        "supplied_contact_work_not_inferred_from_continuous_loss",
        "no_trajectory_solver_ideal_reset_parameter_search_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-formed-class-contact.v1", "report": _project(self)}


def assess_sine_formed_class_contact(
    *,
    formation_time,
    relaxation_duration,
    phase_origin_difference,
    contact_duration,
    form_error_bound,
    phase_error_bound,
    endpoint_radius,
    readout_error_bound,
    radius,
    work_allowance,
    decay_power,
) -> SineFormedClassContact:
    """Rebuild a fixed-family contact certificate from explicit primitives.

    All times use the scaled structural clock. The phase origin is receiver
    minus donor, fixed from initial preparation. Require 0<=phase<=1,
    0<=contact_duration<=1/4, 0<radius<=1/12, positive endpoint_radius and
    integer 0<=decay_power<=4096. Missing sufficient inequalities remain
    unavailable. The generic contrast rule is strict positivity; a frozen
    protocol may impose a stronger separately declared threshold.
    """
    raw = dict(
        formation_time=formation_time,
        relaxation_duration=relaxation_duration,
        phase_origin_difference=phase_origin_difference,
        contact_duration=contact_duration,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        endpoint_radius=endpoint_radius,
        readout_error_bound=readout_error_bound,
        radius=radius,
        work_allowance=work_allowance,
    )
    v = {name: exact_or_represented_real(value, name) for name, value in raw.items()}
    if any(value < 0 for value in v.values()):
        raise ValueError("contact primitives must be nonnegative")
    if not 0 < v["radius"] <= Q(1, 12) or v["endpoint_radius"] <= 0:
        raise ValueError("require 0 < radius <= 1/12 and positive endpoint_radius")
    if v["phase_origin_difference"] > 1 or v["contact_duration"] > Q(1, 4):
        raise ValueError(
            "require phase_origin_difference <= 1 and contact_duration <= 1/4"
        )
    if type(decay_power) is not int or not 0 <= decay_power <= 4096:
        raise ValueError("decay_power must be an integer between zero and4096")
    handoff = _unprobed_handoff(
        formation_time=v["formation_time"],
        relaxation_duration=v["relaxation_duration"],
        form_error_bound=v["form_error_bound"],
        phase_error_bound=v["phase_error_bound"],
        radius=v["radius"],
        endpoint_radius=v["endpoint_radius"],
        decay_power=decay_power,
    )
    fresh = handoff.formation_certificate
    gamma, eta = fresh.gamma_bounds, fresh.eta_bounds
    rate = handoff.lyapunov_decay_rate
    exponent = handoff.decay_exponent
    decay = handoff.exact_decay_upper_bound
    initial = handoff.initial_lyapunov_upper_bounds
    xs = handoff.endpoint_form_norm_squared_upper_bounds
    ys = handoff.endpoint_phase_norm_squared_upper_bounds
    flags = handoff.handoff_certified_by_class
    eps = v["endpoint_radius"]

    phi, h = v["phase_origin_difference"], v["contact_duration"]
    pi = pi_interval()
    dc, sine = cos(2 * pi / 9) - cos(4 * pi / 9), sin(I(phi))
    if not (dc.lo > 0 and gamma.hi < Q(1, 3000) and eta.hi * h**2 <= Q(3, 4)):
        raise ArithmeticError("fixed contact response theorem constants failed")
    fourth = Q(11, 81) * gamma**3 * sine * dc
    leading = fourth * (h**4 / 24)
    tail = gamma.hi**3 * dc.hi * 4 * sine.abs_max * h**5 / (45 * (1 - h / 2))
    nonlinear = Q(8, 15) * sine.abs_max * gamma.hi**5 * h**5
    barrier = Q(1, 1620) * v["radius"] ** 2
    prep = contrast = disconnected = z2 = energy = radius_margin = storage_margin = None
    work = work_margin = mean_x = mean_phase = None
    response = identity = allowed = False
    if all(flags):
        prep = 2 * eps / (1 - 3 * h)
        error = tail + nonlinear + prep + 2 * v["readout_error_bound"]
        contrast = leading + I(-error, error)
        null_error = prep + 2 * v["readout_error_bound"]
        disconnected = I(-null_error, null_error)
        response = contrast.lo > 0
        joined = _joined_contact_bounds(
            endpoint_radius=eps,
            phase_origin_difference=phi,
            radius=v["radius"],
            work_allowance=v["work_allowance"],
        )
        z2 = joined.joined_radius_squared_upper_bound
        energy = joined.joined_excess_storage_upper_bound
        radius_margin = joined.joined_radius_margin_bounds
        storage_margin = joined.joined_storage_margin_bounds
        work = joined.bridge_work_bounds
        work_margin = joined.work_margin_bounds
        mean_x = joined.joined_form_mean_bounds
        mean_phase = joined.joined_phase_mean_bounds
        identity = joined.identity_certified
        allowed = joined.work_within_allowance

    reasons = tuple(
        reason
        for condition, reason in (
            (fresh.status == "certified_two_formed_classes", "formation_unavailable"),
            (all(flags), "unprobed_endpoint_budget_not_certified"),
            (identity, "whole_joined_identity_not_certified"),
            (allowed, "supplied_contact_work_allowance_not_certified"),
            (response, "recorded_receiver_contrast_not_strictly_positive"),
        )
        if not condition
    )
    contact_time = v["formation_time"] + v["relaxation_duration"]
    return SineFormedClassContact(
        formation_certificate=fresh,
        relaxation_duration=v["relaxation_duration"],
        contact_time=contact_time,
        original_contact_time=contact_time / fresh.form_loss,
        phase_origin_difference=phi,
        contact_duration=h,
        endpoint_radius=eps,
        readout_error_bound=v["readout_error_bound"],
        work_allowance=v["work_allowance"],
        decay_power=decay_power,
        lyapunov_decay_rate=rate,
        decay_exponent=exponent,
        exact_decay_upper_bound=decay,
        initial_lyapunov_upper_bounds=initial,
        endpoint_form_norm_squared_upper_bounds=xs,
        endpoint_phase_norm_squared_upper_bounds=ys,
        handoff_certified_by_class=flags,
        ideal_fourth_derivative_contrast_bounds=fourth,
        ideal_leading_contrast_bounds=leading,
        semigroup_tail_upper_bound=tail,
        nonlinear_remainder_upper_bound=nonlinear,
        preparation_response_error_upper_bound=prep,
        recorded_contrast_bounds=contrast,
        disconnected_recorded_contrast_bounds=disconnected,
        joined_radius_squared_upper_bound=z2,
        joined_excess_storage_upper_bound=energy,
        joined_barrier_lower_bound=barrier,
        joined_radius_margin_bounds=radius_margin,
        joined_storage_margin_bounds=storage_margin,
        bridge_work_bounds=work,
        work_margin_bounds=work_margin,
        joined_form_mean_bounds=mean_x,
        joined_phase_mean_bounds=mean_phase,
        response_certified=response,
        identity_certified=identity,
        work_within_allowance=allowed,
        status="certified_formed_class_contact" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
