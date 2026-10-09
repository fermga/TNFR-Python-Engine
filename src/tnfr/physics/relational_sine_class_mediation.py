"""Finite class-mediated transmission between three acquired C9 patterns.

The two full-law comparisons differ only in the mediator's acquired class.
Each receiver response subtracts its own unprobed continuation from the same
reached state. Exact walk algebra and explicit nonlinear/family remainders do
not replace that baseline, replay a trajectory or reset a formation image.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._sine_formed_contact import _unprobed_handoff, _UnprobedHandoff
from ._sine_port_bounds import _joined_port_bounds, _JoinedPortBounds
from ._sine_port_geometry import (
    _central_port_geometry,
    _component_interior_matrix,
    _PortGeometry,
)
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassMediation", "assess_sine_class_mediation")

_CLASSES = ((1, 1, 1), (1, 2, 1))
_CONTACTS = ((0, 1), (1, 2))
_NODES = tuple(range(27))
_EDGES = tuple(
    (9 * component + j, 9 * component + (j + 1) % 9)
    for component in range(3)
    for j in range(9)
) + ((4, 13), (13, 22))


def _mediation_walks(geometry):
    """Rebuild central-donor to central-receiver walks on the shared quotient."""
    form = geometry.normalized_form_matrix
    mediator = _component_interior_matrix(geometry, 1)
    square = exact_matrix_product(form, form)
    transmitted = exact_matrix_product(exact_matrix_product(form, mediator), form)
    return square[10][0], transmitted[10][0]


@dataclass(frozen=True)
class _MediationReferenceBounds:
    """Shared ideal-reference algebra; callers admit the physical primitives."""

    gamma: I
    cosines: tuple[I, I]
    eta: I
    difference: I
    common: Q
    walk: Q
    leading: I
    tail: Q
    denominator: Q
    nonlinear: Q
    ideal: I


def _mediation_reference_bounds(geometry, a, h):
    """Retain the established outward operation order of the ideal contrast."""
    common, walk = _mediation_walks(geometry)
    gamma, cosines = _parameters((1, 2))
    eta = gamma**2
    difference = cosines[0] - cosines[1]
    leading = a * eta * difference * h**3 * walk / 6
    tail = 9 * a * eta.hi * difference.hi * h**4 / (1 - 3 * h / 4)
    denominator = 1 - 2 * eta.hi * h**2
    nonlinear = 16 * gamma.hi**3 * a**2 * h**3 / (3 * denominator**2 * (1 - 3 * h))
    ideal = leading + I(-tail - nonlinear, tail + nonlinear)
    return _MediationReferenceBounds(
        gamma,
        cosines,
        eta,
        difference,
        common,
        walk,
        leading,
        tail,
        denominator,
        nonlinear,
        ideal,
    )


@dataclass(frozen=True)
class SineClassMediation:
    """Conditional response, identity and event-work certificates.

    Source errors are componentwise original-preparation budgets with exact
    separate zero sums. The freshly rebuilt handoff bounds every component's
    reached Euclidean form/phase errors by endpoint_radius. Each actual
    response uses its same-state unprobed baseline. Readout error bounds one
    scalar reading; four readings enter the oriented class contrast.

    A phase-blind alternative retains both rows, support, clock and kick but
    removes phase-to-form feedback. Its ideal class contrast is zero; its
    actual-family enclosure retains preparation and readout errors. Identity,
    supplied-work policy and response separation have distinct flags.
    """

    formation_time: Q
    relaxation_duration: Q
    contact_duration: Q
    probe_amplitude: Q
    form_error_bound: Q
    phase_error_bound: Q
    endpoint_radius: Q
    radius: Q
    decay_power: int
    readout_error_bound: Q
    contact_work_allowance: Q
    probe_work_allowance: Q
    geometry: _PortGeometry
    source_handoff: _UnprobedHandoff
    joined_bounds: _JoinedPortBounds
    gamma_bounds: I
    eta_bounds: I
    mediator_cosine_bounds: tuple[I, I]
    mediator_cosine_difference_bounds: I
    common_diffusion_walk_coefficient: Q
    common_second_derivative_per_amplitude_bounds: I
    mediator_walk_coefficient: Q
    leading_contrast_bounds: I
    linear_tail_upper_bound: Q
    nonlinear_contrast_error_upper_bound: Q
    nonlinear_bootstrap_margin: Q
    ideal_nonlinear_contrast_bounds: I
    preparation_contrast_error_upper_bound: Q | None
    readout_contrast_error_upper_bound: Q | None
    actual_contrast_bounds: I | None
    recorded_contrast_bounds: I | None
    phase_blind_recorded_contrast_bounds: I | None
    phase_blind_exclusion_margin_bounds: I | None
    contact_work_bounds: I | None
    probe_work_bounds: I | None
    probe_work_margin: Q | None
    post_probe_radius_squared_upper_bound: Q | None
    post_probe_excess_storage_upper_bound: Q | None
    post_probe_radius_margin: Q | None
    post_probe_storage_margin: Q | None
    post_probe_form_mean_bounds: I | None
    post_probe_phase_mean_bounds: I | None
    source_handoff_certified: bool
    response_certified: bool
    phase_blind_alternative_excluded: bool
    baseline_identity_certified: bool
    probe_identity_certified: bool
    identity_certified: bool
    contact_work_within_allowance: bool
    probe_work_within_allowance: bool
    work_within_allowances: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    classes: tuple[tuple[int, ...], ...] = _CLASSES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    phase_origins: tuple[Q, ...] = (Q(0),) * 3
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    donor_node: int = 4
    mediator_node: int = 13
    receiver_node: int = 22
    held_capacities: tuple[Q, ...] = (Q(1),) * 27
    first_class_dependent_derivative_order: int = 3
    scaled_generator_norm_upper_bound: Q = Q(3)
    probe_form_mean_shift: Q = Q(0)
    form_loss: Q = Q(1023, 1024)
    exchange_weight: Q = Q(1, 1024)
    phase_exchange_beta: Q = Q(1)
    law: str = "x'=-KLx+gamma*KS(theta); theta'=gamma*KLx"
    clock: str = "tau=e*t; e=1023/1024"
    response_definition: str = "R_k=x_probe[22](h)-x_unprobed[22](h); contrast=R_1-R_2"
    phase_blind_law: str = "x'=-KLx; theta'=gamma*KLx"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_law_support_capacity_clock_probe_and_receiver_for_both_classes",
        "fresh_original_preparation_and_unprobed_dwell_no_cached_report_admission",
        "aligned_origins_are_original_preparation_choices_not_handoff_resets",
        "three_independent_component_errors_with_original_separate_zero_sums",
        "full_27_node_form_and_phase_state_retained_by_the_family_comparison",
        "reflection_even_quotient_used_only_for_ideal_tangent_walk_algebra",
        "actual_probe_and_unprobed_baseline_start_at_the_same_reached_state",
        "strict_positive_oriented_contrast_with_nonlinear_and_preparation_errors",
        "four_scalar_reading_errors_have_total_bound_four_delta",
        "phase_blind_complete_alternative_keeps_actual_source_error_allowance",
        "contact_and_form_jump_work_have_separate_supplied_allowances",
        "identity_and_recovery_are_on_the_actual_post_event_conserved_mean_leaf",
        "response_identity_and_work_flags_are_independent_obligations",
        "no_trajectory_solver_source_reset_autonomous_contact_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-class-mediation.v1", "report": _project(self)}


def assess_sine_class_mediation(
    *,
    formation_time,
    relaxation_duration,
    contact_duration,
    probe_amplitude,
    form_error_bound,
    phase_error_bound,
    endpoint_radius,
    radius,
    decay_power,
    readout_error_bound,
    contact_work_allowance,
    probe_work_allowance,
) -> SineClassMediation:
    """Rebuild an acquired-family transmission certificate from twelve primitives.

    All scalar inputs use shared exact-or-represented-real admission. Require
    nonnegative times, errors, amplitude and allowances, 0<endpoint_radius,
    0<radius<=1/12 and contact_duration<=1/4. Zero impulse or horizon is valid
    but cannot certify positive response. decay_power is an ordinary integer
    from zero through4096; shared formation and dwell exponential caps apply.

    Actual-family outputs require the fresh handoff for both classes. Ideal
    coefficients and remainders remain available when the handoff fails.
    Positive response, identity retention and closed work-allowance comparisons
    are separate tests; none is inferred from another channel's verdict.
    """
    raw = dict(
        formation_time=formation_time,
        relaxation_duration=relaxation_duration,
        contact_duration=contact_duration,
        probe_amplitude=probe_amplitude,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        endpoint_radius=endpoint_radius,
        radius=radius,
        readout_error_bound=readout_error_bound,
        contact_work_allowance=contact_work_allowance,
        probe_work_allowance=probe_work_allowance,
    )
    v = {key: exact_or_represented_real(value, key) for key, value in raw.items()}
    if any(value < 0 for value in v.values()):
        raise ValueError(
            "times, errors, probe amplitude and work allowances must be nonnegative"
        )
    if not 0 < v["radius"] <= Q(1, 12) or v["endpoint_radius"] <= 0:
        raise ValueError("require 0 < radius <= 1/12 and positive endpoint_radius")
    if v["contact_duration"] > Q(1, 4):
        raise ValueError("contact_duration must not exceed1/4")
    if v["formation_time"] > 20480:
        raise ValueError("formation_time/5 must not exceed4096")
    if type(decay_power) is not int or not 0 <= decay_power <= 4096:
        raise ValueError("decay_power must be an ordinary integer zero through4096")

    a, h, eps, r = (
        v[key]
        for key in ("probe_amplitude", "contact_duration", "endpoint_radius", "radius")
    )
    geometry = _central_port_geometry(3, _CONTACTS)
    reference = _mediation_reference_bounds(geometry, a, h)

    handoff = _unprobed_handoff(
        **{
            key: v[key]
            for key in (
                "formation_time",
                "relaxation_duration",
                "form_error_bound",
                "phase_error_bound",
                "endpoint_radius",
                "radius",
            )
        },
        decay_power=decay_power,
    )
    source = all(handoff.handoff_certified_by_class)
    joined = _joined_port_bounds(
        geometry=geometry,
        phase_origins=(Q(0),) * 3,
        endpoint_radius=eps,
        radius=r,
        work_allowance=v["contact_work_allowance"],
        handoff_available=source,
    )
    preparation = readout = actual = recorded = blind = blind_margin = None
    contact_work = probe_work = work_margin = z2 = energy = z_margin = e_margin = None
    form_mean = phase_mean = None
    response = excluded = identity_probe = allowed_probe = False
    shift = Q(3, 58) * a
    if source:
        preparation = 4 * eps / (1 - 3 * h)
        readout = 4 * v["readout_error_bound"]
        actual = reference.ideal + I(-preparation, preparation)
        recorded = actual + I(-readout, readout)
        blind = I(-preparation - readout, preparation + readout)
        blind_margin = recorded - blind.hi
        response = bool(a > 0 and h > 0 and recorded.lo > 0)
        excluded = bool(response and blind_margin.lo > 0)
        contact_work = I(0, joined.contact_work_upper_bound)
        probe_center, probe_error = Q(3, 2) * a**2, 6 * a * eps
        probe_work = I(probe_center - probe_error, probe_center + probe_error)
        work_margin = v["probe_work_allowance"] - probe_center - probe_error
        allowed_probe = work_margin >= 0
        z2 = 6 * eps**2 + 2 * a * eps + Q(26, 27) * a**2
        energy = 22 * eps**2 + 6 * a * eps + Q(3, 2) * a**2
        z_margin = r**2 - z2
        if joined.joined_barrier_lower_bound is not None:
            e_margin = joined.joined_barrier_lower_bound - energy
            identity_probe = z_margin > 0 and e_margin > 0
        mean_error = Q(4, 58) * eps
        form_mean = I(shift - mean_error, shift + mean_error)
        phase_mean = I(-mean_error, mean_error)
    identity = bool(joined.identity_certified and identity_probe)
    allowed = bool(joined.work_within_allowance and allowed_probe)
    reasons = tuple(
        label
        for passed, label in (
            (source, "actual_source_handoff_not_certified"),
            (response, "strict_positive_class_contrast_not_certified"),
            (excluded, "phase_blind_alternative_not_excluded"),
            (identity, "baseline_and_probe_identity_not_certified"),
            (allowed, "contact_and_probe_work_not_within_allowances"),
        )
        if not passed
    )
    status = "certified_class_mediation" if not reasons else "unavailable"
    return SineClassMediation(
        **v,
        decay_power=decay_power,
        geometry=geometry,
        source_handoff=handoff,
        joined_bounds=joined,
        gamma_bounds=reference.gamma,
        eta_bounds=reference.eta,
        mediator_cosine_bounds=reference.cosines,
        mediator_cosine_difference_bounds=reference.difference,
        common_diffusion_walk_coefficient=reference.common,
        common_second_derivative_per_amplitude_bounds=(1 - reference.eta)
        * reference.common,
        mediator_walk_coefficient=reference.walk,
        leading_contrast_bounds=reference.leading,
        linear_tail_upper_bound=reference.tail,
        nonlinear_contrast_error_upper_bound=reference.nonlinear,
        nonlinear_bootstrap_margin=reference.denominator,
        ideal_nonlinear_contrast_bounds=reference.ideal,
        preparation_contrast_error_upper_bound=preparation,
        readout_contrast_error_upper_bound=readout,
        actual_contrast_bounds=actual,
        recorded_contrast_bounds=recorded,
        phase_blind_recorded_contrast_bounds=blind,
        phase_blind_exclusion_margin_bounds=blind_margin,
        contact_work_bounds=contact_work,
        probe_work_bounds=probe_work,
        probe_work_margin=work_margin,
        post_probe_radius_squared_upper_bound=z2,
        post_probe_excess_storage_upper_bound=energy,
        post_probe_radius_margin=z_margin,
        post_probe_storage_margin=e_margin,
        post_probe_form_mean_bounds=form_mean,
        post_probe_phase_mean_bounds=phase_mean,
        source_handoff_certified=source,
        response_certified=response,
        phase_blind_alternative_excluded=excluded,
        baseline_identity_certified=joined.identity_certified,
        probe_identity_certified=identity_probe,
        identity_certified=identity,
        contact_work_within_allowance=joined.work_within_allowance,
        probe_work_within_allowance=allowed_probe,
        work_within_allowances=allowed,
        status=status,
        unavailable_reasons=reasons,
        probe_form_mean_shift=shift,
    )
