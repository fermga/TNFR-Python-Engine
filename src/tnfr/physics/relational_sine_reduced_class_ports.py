"""Reflection-even C9 port surrogates with an exact nonlinear sine bridge.

Each component retains five form and five target-phase displacement coordinates.
Its interior is the supplied class's linearized sine field; the bridge keeps
its full sine. This is a bounded approximation to the original nonlinear law,
not an invariant nonlinear quotient, a new contact selector or a trajectory
solver. Public calculations admit primitives and rebuild consumed coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import ExactSquareMatrix, exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_formed_contact import (
    _joined_contact_bounds,
    _JoinedContactBounds,
    _unprobed_handoff,
    _UnprobedHandoff,
)
from .relational_observations import _ordered

__all__ = (
    "SineReducedClassPortState",
    "SineReducedClassPorts",
    "evaluate_sine_reduced_class_ports",
    "assess_sine_reduced_class_ports",
)

_ORBITS = ((4,), (3, 5), (2, 6), (1, 7), (0, 8))
_MULTIPLICITIES = (1, 2, 2, 2, 2)
_DEGREES = (3, 2, 2, 2, 2)


def _reduced_port_matrices() -> (
    tuple[ExactSquareMatrix, ExactSquareMatrix, ExactSquareMatrix]
):
    """Build exact joined, donor-interior and receiver-interior rate matrices.

    The five-node chain represents reflection orbits, with two equal internal
    neighbors at the central port and a cancelling closing edge at the outer
    orbit. Mobility uses the actual degree three at either joined port. No
    class or sine value is hidden in these rational matrices.
    """
    internal = [[Q(0) for _ in range(5)] for _ in range(5)]
    internal[0][0], internal[0][1] = Q(2), Q(-2)
    for row in range(1, 4):
        internal[row][row - 1 : row + 2] = [Q(-1), Q(2), Q(-1)]
    internal[4][3], internal[4][4] = Q(-1), Q(1)
    donor = [[Q(0) for _ in range(10)] for _ in range(10)]
    receiver = [[Q(0) for _ in range(10)] for _ in range(10)]
    for i in range(5):
        for j in range(5):
            donor[i][j] = internal[i][j] / _DEGREES[i]
            receiver[5 + i][5 + j] = internal[i][j] / _DEGREES[i]
    joined = [[donor[i][j] + receiver[i][j] for j in range(10)] for i in range(10)]
    for i, j in ((0, 5), (5, 0)):
        joined[i][i] += Q(1, 3)
        joined[i][j] -= Q(1, 3)
    return (
        tuple(map(tuple, joined)),
        tuple(map(tuple, donor)),
        tuple(map(tuple, receiver)),
    )


def _class_index(value, label):
    if type(value) is not int or value not in (1, 2):
        raise ValueError(f"{label} must be an ordinary integer in {{1, 2}}")
    return value


def _state_row(values, label):
    raw = _ordered(values, label, limit=11)
    if len(raw) != 10:
        raise ValueError(f"{label} must contain exactly ten values")
    return tuple(
        exact_or_represented_real(value, f"{label}[{index}]")
        for index, value in enumerate(raw)
    )


def _exact_action(matrix, vector):
    """Apply an already admitted rational matrix through the shared product."""
    return tuple(
        row[0] for row in exact_matrix_product(matrix, tuple((x,) for x in vector))
    )


def _interval_action(matrix, vector):
    return tuple(
        sum((value * vector[j] for j, value in enumerate(row) if value), I(0))
        for row in matrix
    )


def _reduced_parameters(donor_class, receiver_class):
    """Rebuild fixed coefficients after the caller admits primitive classes."""
    a, donor, receiver = _reduced_port_matrices()
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coefficients = tuple(cos(2 * k * pi / 9) for k in (donor_class, receiver_class))
    return a, donor, receiver, coefficients, gamma


def _reduced_receiver_jets(*, a, donor, receiver, coefficients, gamma, phi):
    """Differentiate the same composed surrogate rows at its ideal zero state.

    The linearized bridge includes cos(phi); phase's initial derivative is
    zero, so bridge derivatives beyond its Jacobian do not enter orders1--4.
    Inputs are private, freshly rebuilt coefficient data, never supplied reports.
    """
    sine, bridge_cosine = sin(I(phi)), cos(I(phi))
    u = tuple(Q(1, 3) if i == 0 else Q(-1, 3) if i == 5 else Q(0) for i in range(10))
    au = _exact_action(a, u)
    a2u = _exact_action(a, au)
    a3u = _exact_action(a, a2u)

    def stiffness(vector):
        d, r = _exact_action(donor, vector), _exact_action(receiver, vector)
        bridge_gap = vector[0] - vector[5]
        return tuple(
            coefficients[0] * d[i]
            + coefficients[1] * r[i]
            + bridge_cosine * bridge_gap * u[i]
            for i in range(10)
        )

    bau, ba2u = stiffness(au), stiffness(a2u)
    abau = _interval_action(a, bau)
    common = gamma * sine
    feedback = gamma**3 * sine
    return (
        common * u[5],
        -common * au[5],
        common * a2u[5] - feedback * bau[5],
        -common * a3u[5] + feedback * (abau[5] + ba2u[5]),
    )


@dataclass(frozen=True)
class SineReducedClassPortState:
    """Detached instantaneous rows of the declared twenty-coordinate surrogate.

    Coordinates are donor orbits followed by receiver orbits. Phase coordinates
    are real displacements after subtracting both the chosen winding target
    and that component's separately supplied common origin. The donor origin
    is zero and the receiver origin is phase_origin_difference; thus the bridge
    phase is phase_origin_difference + phase_deviations[5] - phase_deviations[0].
    These are not absolute circular phases. Arbitrary admitted coordinates
    define surrogate rows; an error bound
    against full nonlinear dynamics requires the separate preparation and horizon
    hypotheses of the assessment API.
    """

    donor_class: int
    receiver_class: int
    forms: tuple[Q, ...]
    phase_deviations: tuple[Q, ...]
    phase_origin_difference: Q
    class_cosine_bounds: tuple[I, I]
    gamma_bounds: I
    normalized_form_matrix: ExactSquareMatrix
    normalized_donor_interior_matrix: ExactSquareMatrix
    normalized_receiver_interior_matrix: ExactSquareMatrix
    bridge_phase_difference: Q
    bridge_sine_bounds: I
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    reflection_orbits: tuple[tuple[int, ...], ...] = _ORBITS
    orbit_multiplicities: tuple[int, ...] = _MULTIPLICITIES
    component_degrees: tuple[int, ...] = _DEGREES
    component_coordinate_count: int = 10
    joined_coordinate_count: int = 20
    full_joined_coordinate_count: int = 36
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "class_one_or_two_linear_interior_on_five_reflection_orbits",
        "exact_sine_bridge_with_actual_degree_three_ports",
        "real_target_phase_displacements_not_circular_state_substitution",
        "instantaneous_surrogate_rows_not_an_exact_nonlinear_quotient",
        "no_solver_contact_selector_or_full_state_reconstruction_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-reduced-class-port-state.v1",
            "report": _project(self),
        }


def evaluate_sine_reduced_class_ports(
    *, donor_class, receiver_class, forms, phase_deviations, phase_origin_difference
) -> SineReducedClassPortState:
    """Evaluate the supplied reduced component composition from original values.

    Classes are ordinary integers one or two. The two ordered state rows contain
    exactly ten finite signed reals each, with donor coordinates first. Every
    primitive is admitted before interval construction. No report is an input.
    phase_deviations subtract the winding target and the component origin
    (zero for the donor, phase_origin_difference for the receiver). The bridge
    uses phase_origin_difference + phase_deviations[5] - phase_deviations[0].
    """
    donor_class = _class_index(donor_class, "donor_class")
    receiver_class = _class_index(receiver_class, "receiver_class")
    x = _state_row(forms, "forms")
    y = _state_row(phase_deviations, "phase_deviations")
    phi = exact_or_represented_real(phase_origin_difference, "phase_origin_difference")
    a, donor, receiver, coefficients, gamma = _reduced_parameters(
        donor_class, receiver_class
    )
    form_gradient = _exact_action(a, x)
    donor_gradient, receiver_gradient = _exact_action(donor, y), _exact_action(
        receiver, y
    )
    gap = phi + y[5] - y[0]
    bridge = sin(I(gap))
    rates = tuple(
        -form_gradient[i]
        - gamma
        * (coefficients[0] * donor_gradient[i] + coefficients[1] * receiver_gradient[i])
        + gamma * bridge * (Q(1, 3) if i == 0 else Q(-1, 3) if i == 5 else Q(0))
        for i in range(10)
    )
    return SineReducedClassPortState(
        donor_class=donor_class,
        receiver_class=receiver_class,
        forms=x,
        phase_deviations=y,
        phase_origin_difference=phi,
        class_cosine_bounds=coefficients,
        gamma_bounds=gamma,
        normalized_form_matrix=a,
        normalized_donor_interior_matrix=donor,
        normalized_receiver_interior_matrix=receiver,
        bridge_phase_difference=gap,
        bridge_sine_bounds=bridge,
        form_rate_bounds=rates,
        phase_rate_bounds=tuple(gamma * value for value in form_gradient),
    )


@dataclass(frozen=True)
class SineReducedClassPorts:
    """A bounded reduced-model transfer to the held-out class-two receiver.

    The ideal reduced comparison and its full-law discrepancy are distinct.
    Original uncertainty is carried by a fresh unprobed formation handoff;
    missing handoff leaves actual-response bounds unavailable. The fractional
    error test uses the final outward recorded lower bound, not its center.
    """

    formation_time: Q
    relaxation_duration: Q
    phase_origin_difference: Q
    contact_duration: Q
    form_error_bound: Q
    phase_error_bound: Q
    endpoint_radius: Q
    readout_error_bound: Q
    radius: Q
    work_allowance: Q
    decay_power: int
    error_fraction: Q
    unprobed_handoff: _UnprobedHandoff
    joined_contact_bounds: _JoinedContactBounds | None
    normalized_form_matrix: ExactSquareMatrix
    normalized_donor_interior_matrix: ExactSquareMatrix
    normalized_receiver_interior_matrix: ExactSquareMatrix
    class_cosine_bounds: tuple[I, I]
    receiver_derivative_bounds_by_donor: tuple[tuple[I, ...], ...]
    fourth_derivative_geometry_coefficient: Q
    ideal_fourth_derivative_contrast_bounds: I
    ideal_leading_contrast_bounds: I
    reduced_ideal_contrast_bounds: I
    reduced_semigroup_tail_upper_bound: Q
    reduced_nonlinear_remainder_upper_bound: Q
    surrogate_full_discrepancy_upper_bound: Q
    preparation_response_error_upper_bound: Q | None
    readout_contrast_error_upper_bound: Q
    total_error_upper_bound: Q | None
    recorded_contrast_bounds: I | None
    disconnected_recorded_contrast_bounds: I | None
    error_ratio_upper_bound: Q | None
    error_fraction_margin_bounds: I | None
    response_certified: bool
    approximation_certified: bool
    identity_certified: bool
    work_within_allowance: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    receiver_class: int = 2
    donor_classes: tuple[int, int] = (1, 2)
    derivative_orders: tuple[int, ...] = (1, 2, 3, 4)
    reflection_orbits: tuple[tuple[int, ...], ...] = _ORBITS
    orbit_multiplicities: tuple[int, ...] = _MULTIPLICITIES
    component_degrees: tuple[int, ...] = _DEGREES
    component_coordinate_count: int = 10
    joined_coordinate_count: int = 20
    full_joined_coordinate_count: int = 36
    clock: str = "tau=e*t; e=1023/1024"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_class_one_and_two_coefficients_no_fit_or_response_recalibration",
        "twenty_coordinate_reflection_even_surrogate_with_exact_sine_bridge",
        "held_out_class_two_receiver_uses_the_same_component_kernel",
        "receiver_jets_derived_from_reduced_matrices_and_changed_receiver_cosine",
        "discarded_reflection_odd_and_nonlinear_interior_effects_have_explicit_bounds",
        "actual_source_error_retained_through_fresh_unprobed_formation_handoff",
        "same_supplied_bridge_work_and_whole_joined_identity_hypotheses",
        "no_incoming_report_ideal_reset_solver_parameter_search_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-reduced-class-ports.v1", "report": _project(self)}


def assess_sine_reduced_class_ports(
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
    error_fraction,
) -> SineReducedClassPorts:
    """Certify a receiver-two contrast using freshly rebuilt reduced components.

    All inputs are required. Times and errors are nonnegative, endpoint_radius
    is positive, 0<radius<=1/12, phase_origin_difference<=1, contact_duration<=1/4,
    and 0<error_fraction<1. decay_power is an ordinary integer0..4096; shared
    formation and relaxation owners retain their separate exponential work caps.
    A zero phase difference or horizon is admitted but cannot certify response.
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
        error_fraction=error_fraction,
    )
    v = {name: exact_or_represented_real(value, name) for name, value in raw.items()}
    if any(value < 0 for value in v.values()):
        raise ValueError("reduced port primitives must be nonnegative")
    if not 0 < v["radius"] <= Q(1, 12) or v["endpoint_radius"] <= 0:
        raise ValueError("require 0 < radius <= 1/12 and positive endpoint_radius")
    if v["phase_origin_difference"] > 1 or v["contact_duration"] > Q(1, 4):
        raise ValueError(
            "require phase_origin_difference <= 1 and contact_duration <= 1/4"
        )
    if not 0 < v["error_fraction"] < 1:
        raise ValueError("error_fraction must be strictly between zero and one")
    if type(decay_power) is not int or not 0 <= decay_power <= 4096:
        raise ValueError("decay_power must be an ordinary integer between zero and4096")
    handoff = _unprobed_handoff(
        formation_time=v["formation_time"],
        relaxation_duration=v["relaxation_duration"],
        form_error_bound=v["form_error_bound"],
        phase_error_bound=v["phase_error_bound"],
        radius=v["radius"],
        endpoint_radius=v["endpoint_radius"],
        decay_power=decay_power,
    )
    a, donor, receiver, coefficients, gamma = _reduced_parameters(1, 2)
    phi, h = v["phase_origin_difference"], v["contact_duration"]
    derivatives = tuple(
        _reduced_receiver_jets(
            a=a,
            donor=donor,
            receiver=receiver,
            coefficients=(coefficients[k - 1], coefficients[1]),
            gamma=gamma,
            phi=phi,
        )
        for k in (1, 2)
    )
    u = tuple(Q(1, 3) if i == 0 else Q(-1, 3) if i == 5 else Q(0) for i in range(10))
    au = _exact_action(a, u)
    geometry = (
        -_exact_action(a, _exact_action(donor, au))[5]
        - _exact_action(donor, _exact_action(a, au))[5]
    )
    dc, sine = coefficients[0] - coefficients[1], sin(I(phi))
    if not (geometry > 0 and dc.lo > 0 and gamma.hi < Q(1, 3000)):
        raise ArithmeticError("fixed reduced-port theorem constants failed")
    fourth = geometry * gamma**3 * sine * dc
    leading = fourth * (h**4 / 24)
    tail = gamma.hi**3 * dc.hi * 4 * sine.abs_max * h**5 / (45 * (1 - h / 2))
    nonlinear = Q(8, 15) * sine.abs_max * gamma.hi**5 * h**5
    discrepancy = Q(16, 45) * gamma.hi**5 * sine.abs_max**2 * h**5 / (1 - 3 * h)
    reduced = leading + I(-tail - nonlinear, tail + nonlinear)
    readout = 2 * v["readout_error_bound"]
    joined = prep = total = recorded = disconnected = ratio = margin = None
    response = approximation = identity = allowed = False
    if all(handoff.handoff_certified_by_class):
        joined = _joined_contact_bounds(
            endpoint_radius=v["endpoint_radius"],
            phase_origin_difference=phi,
            radius=v["radius"],
            work_allowance=v["work_allowance"],
        )
        prep = 2 * v["endpoint_radius"] / (1 - 3 * h)
        total = tail + nonlinear + discrepancy + prep + readout
        recorded = leading + I(-total, total)
        disconnected = I(-prep - readout, prep + readout)
        response = recorded.lo > 0
        if response:
            ratio = total / recorded.lo
            margin = I(v["error_fraction"] * recorded.lo - total)
            approximation = margin.lo > 0
        identity, allowed = joined.identity_certified, joined.work_within_allowance
    reasons = tuple(
        reason
        for condition, reason in (
            (
                handoff.formation_certificate.status == "certified_two_formed_classes",
                "formation_unavailable",
            ),
            (
                all(handoff.handoff_certified_by_class),
                "unprobed_endpoint_budget_not_certified",
            ),
            (identity, "whole_joined_identity_not_certified"),
            (allowed, "supplied_contact_work_allowance_not_certified"),
            (response, "recorded_receiver_contrast_not_strictly_positive"),
            (approximation, "relative_error_budget_not_certified"),
        )
        if not condition
    )
    return SineReducedClassPorts(
        **v,
        decay_power=decay_power,
        unprobed_handoff=handoff,
        joined_contact_bounds=joined,
        normalized_form_matrix=a,
        normalized_donor_interior_matrix=donor,
        normalized_receiver_interior_matrix=receiver,
        class_cosine_bounds=coefficients,
        receiver_derivative_bounds_by_donor=derivatives,
        fourth_derivative_geometry_coefficient=geometry,
        ideal_fourth_derivative_contrast_bounds=fourth,
        ideal_leading_contrast_bounds=leading,
        reduced_ideal_contrast_bounds=reduced,
        reduced_semigroup_tail_upper_bound=tail,
        reduced_nonlinear_remainder_upper_bound=nonlinear,
        surrogate_full_discrepancy_upper_bound=discrepancy,
        preparation_response_error_upper_bound=prep,
        readout_contrast_error_upper_bound=readout,
        total_error_upper_bound=total,
        recorded_contrast_bounds=recorded,
        disconnected_recorded_contrast_bounds=disconnected,
        error_ratio_upper_bound=ratio,
        error_fraction_margin_bounds=margin,
        response_certified=response,
        approximation_certified=approximation,
        identity_certified=identity,
        work_within_allowance=allowed,
        status="certified_reduced_class_ports" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
