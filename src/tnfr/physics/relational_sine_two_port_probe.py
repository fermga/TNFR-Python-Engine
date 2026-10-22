"""Finite supplied form probes of a conditional two-port C9 endpoint family.

Exact heat-semigroup and complete-law comparison bounds retain every member
of the supplied metric balls. A fresh implicit target supplies criticality;
no acquisition history, incoming report or trajectory is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import pi_interval
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_sine_two_port_compatibility import (
    _EDGES,
    _NODES,
    SineTwoPortCompatibility,
    assess_sine_two_port_compatibility,
)

__all__ = ("SineTwoPortProbe", "assess_sine_two_port_probe")


def _probe_geometry():
    """Rebuild the actual joined and disconnected linear probe geometry."""
    geometry = _derive(_NODES, _EDGES)
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    incidence = tuple(tuple(map(Q, row)) for row in geometry.incidence)
    laplacian = exact_matrix_product(incidence, tuple(zip(*incidence)))
    normalized = tuple(
        tuple(value / degree for value in row)
        for degree, row in zip(degrees, laplacian)
    )
    mass = Q(sum(degrees))
    donor = tuple(Q(i < 9) for i in _NODES)
    receiver_mass = sum(degrees[9:])
    receiver = tuple(Q(degrees[i] * int(i >= 9), receiver_mass) for i in _NODES)
    centered = tuple(value - Q(1, 2) for value in donor)
    receiver_gradient = tuple(
        sum(weight * normalized[i][j] for i, weight in enumerate(receiver))
        for j in _NODES
    )
    pulse_gradient = tuple(
        sum(value * b for value, b in zip(row, donor)) for row in laplacian
    )
    lower_slack = tuple(
        tuple(
            laplacian[i][j]
            - Q(1, 90) * (degrees[i] * int(i == j) - Q(degrees[i] * degrees[j]) / mass)
            for j in _NODES
        )
        for i in _NODES
    )
    upper_slack = tuple(
        tuple(2 * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
        for i in _NODES
    )
    if not all(map(exact_symmetric_semidefinite, (lower_slack, upper_slack))):
        raise ArithmeticError("the fixed normalized diffusion bounds failed")
    identities = (
        (sum(degree * value**2 for degree, value in zip(degrees, centered)), Q(10)),
        (
            sum(value**2 / degree for degree, value in zip(degrees, receiver_gradient)),
            Q(1, 300),
        ),
        (
            sum(value**2 / degree for degree, value in zip(degrees, pulse_gradient)),
            Q(4, 3),
        ),
        (sum(value * b for value, b in zip(pulse_gradient, donor)), Q(2)),
        (-sum(value * b for value, b in zip(receiver_gradient, donor)), Q(1, 10)),
        (
            sum(
                (weight - Q(degree) / mass) ** 2 / degree
                for weight, degree in zip(receiver, degrees)
            ),
            Q(1, 40),
        ),
    )
    if any(actual != expected for actual, expected in identities):
        raise ArithmeticError("the fixed probe metric identities failed")
    control_edges = tuple(
        edge for edge in geometry.edges if edge not in ((0, 9), (1, 10))
    )
    control_degrees = tuple(sum(i in edge for edge in control_edges) for i in _NODES)
    control_receiver = tuple(
        Q(degree * int(i >= 9), sum(control_degrees[9:]))
        for i, degree in enumerate(control_degrees)
    )
    if any(
        donor[i] != donor[j]
        or control_receiver[i] / control_degrees[i]
        != control_receiver[j] / control_degrees[j]
        for i, j in control_edges
    ):
        raise ArithmeticError("the disconnected component invariants failed")
    return dict(
        geometry=geometry,
        degrees=degrees,
        laplacian=laplacian,
        normalized_laplacian=normalized,
        normalized_gap_slack_matrix=lower_slack,
        normalized_upper_slack_matrix=upper_slack,
        donor_mask=donor,
        receiver_weights=receiver,
        disconnected_edges=control_edges,
        disconnected_degrees=control_degrees,
        disconnected_receiver_weights=control_receiver,
    )


@dataclass(frozen=True)
class SineTwoPortProbe:
    """Conditional finite response, supplied work and joined-target recovery.

    ``form_radius`` bounds the degree norm of form minus its own conserved
    mean; ``phase_radius`` bounds real lifted phase relative to the compatible
    target shifted to the member's phase mean. They are endpoint hypotheses,
    not nodewise errors or a certificate that any earlier source reaches them.

    The event adds the same positive form amplitude to every donor node and
    changes no phase. Readouts are receiver degree-weighted mean increments,
    each using two observations with independent absolute error bounds.
    The disconnected comparator has its own degree weights and exact zero
    increment for every initial state under its unchanged component laws.

    All endpoint pairs are exact rational bounds, without clipping signed
    work or negative lower responses. Target failure leaves target-dependent
    bounds unavailable; failure of a sufficient gate does not imply failed
    dynamics. Recovery refers only to the joined target on the new mean leaf.
    """

    form_radius: Q
    phase_radius: Q
    pulse_amplitude: Q
    probe_duration: Q
    readout_error_bound: Q
    contrast_threshold: Q
    work_allowance: Q
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    laplacian: tuple[tuple[Q, ...], ...]
    normalized_laplacian: tuple[tuple[Q, ...], ...]
    normalized_gap_slack_matrix: tuple[tuple[Q, ...], ...]
    normalized_upper_slack_matrix: tuple[tuple[Q, ...], ...]
    donor_mask: tuple[Q, ...]
    receiver_weights: tuple[Q, ...]
    disconnected_edges: tuple[tuple[int, int], ...]
    disconnected_degrees: tuple[int, ...]
    disconnected_receiver_weights: tuple[Q, ...]
    target: SineTwoPortCompatibility
    target_acute_margin_lower_bound: Q | None
    target_admitted: bool
    joined_form_mean_increment: Q
    disconnected_form_mean_increments: tuple[Q, Q]
    post_probe_form_norm_upper_bound: Q
    post_probe_relative_norm_squared_upper_bound: Q
    coupled_loop_gain: Q
    coupled_denominator: Q
    whole_window_form_norm_candidate: Q
    whole_window_phase_norm_candidate: Q
    whole_window_form_norm_upper_bound: Q | None
    whole_window_phase_norm_upper_bound: Q | None
    ideal_heat_increment_bounds: tuple[Q, Q]
    initial_form_response_error_upper_bound: Q
    sine_response_error_candidate: Q
    response_error_upper_bound: Q | None
    joined_increment_bounds: tuple[Q, Q] | None
    recorded_joined_increment_bounds: tuple[Q, Q] | None
    disconnected_increment_bounds: tuple[Q, Q]
    recorded_disconnected_increment_bounds: tuple[Q, Q]
    recorded_contrast_bounds: tuple[Q, Q] | None
    response_margin: Q | None
    joined_work_bounds: tuple[Q, Q]
    disconnected_work_bounds: tuple[Q, Q]
    work_allowance_margin: Q
    positive_supplied_work_certified: bool
    post_probe_excess_storage_candidate: Q
    post_probe_excess_storage_upper_bound: Q | None
    post_probe_radius_margin: Q
    capture_storage_margin_candidate: Q
    capture_storage_margin: Q | None
    response_certified: bool
    work_certified: bool
    joined_identity_certified: bool
    recovery_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    root_outer_refinements: int = 32
    root_inner_refinements: int = 64
    capacity: tuple[Q, ...] = (Q(1),) * 18
    gamma_upper_bound: Q = Q(1, 3069)
    normalized_gap_lower_bound: Q = Q(1, 90)
    normalized_rate_upper_bound: Q = Q(2)
    local_radius: Q = Q(1, 12)
    local_cosine_lower_bound: Q = Q(1, 25)
    capture_barrier_lower_bound: Q = Q(1, 648000)
    pulse_law: str = "x_plus=x_minus+pulse_amplitude*donor_mask; theta_plus=theta_minus"
    readout: str = "receiver_degree_weighted_mean_form_after_minus_before_probe"
    clock: str = "tau=e*t; e=1023/1024; gamma=1/(1023*pi)"
    scope: tuple[str, ...] = (
        "conditional_endpoint_degree_metric_balls_no_acquisition_history_or_source_verdict",
        "full_joined_eighteen_node_positive_loss_sine_law_and_held_unit_capacities",
        "fresh_correlated_implicit_two_one_target_not_its_midpoint_or_interval_product",
        "arbitrary_common_form_and_lifted_phase_means_retained_member_by_member",
        "same_supplied_donor_uniform_form_jump_on_joined_and_disconnected_supports",
        "unforced_complete_law_after_the_instantaneous_supplied_jump",
        "finite_heat_semigroup_response_and_full_nonlinear_whole_family_error_bounds",
        "two_independent_errors_per_increment_four_per_joined_minus_control_contrast",
        "disconnected_receiver_increment_and_probe_work_vanish_for_every_initial_state",
        "signed_jump_work_is_separate_from_subsequent_continuous_storage_loss",
        "strict_joined_local_radius_and_storage_barrier_retain_the_acute_target_cell",
        "joined_recovery_to_the_same_phase_geometry_with_form_mean_shifted_by_half_amplitude",
        "heat_response_control_shows_the_transfer_is_not_specific_to_sine_or_winding",
        "no_probe_trajectory_capture_producer_event_selection_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-probe.v1", "report": _project(self)}


def assess_sine_two_port_probe(
    *,
    form_radius,
    phase_radius,
    pulse_amplitude,
    probe_duration,
    readout_error_bound,
    contrast_threshold,
    work_allowance,
) -> SineTwoPortProbe:
    """Assess a finite supplied pulse from primitive conditional endpoint radii.

    All seven scalars are mandatory. Exact rationals remain exact; other reals
    use shared represented-real admission. Radii, readout error, threshold and
    allowance are nonnegative; amplitude is positive and ``0<duration<=1``.
    Phase uses radians, duration uses tau, and the source norm uses the actual
    joined degree metric. Every primitive is admitted before the fresh target.

    Response requires the recorded contrast strictly above its threshold;
    supplied work requires its upper bound at most the allowance. Joined
    identity and recovery additionally require strict radius and storage
    barriers. This conditional endpoint-family theorem never establishes an
    earlier source-to-endpoint handoff and accepts no acquisition certificate.
    """
    labels = (
        "form_radius",
        "phase_radius",
        "pulse_amplitude",
        "probe_duration",
        "readout_error_bound",
        "contrast_threshold",
        "work_allowance",
    )
    values = tuple(
        exact_or_represented_real(value, name)
        for value, name in zip(
            (
                form_radius,
                phase_radius,
                pulse_amplitude,
                probe_duration,
                readout_error_bound,
                contrast_threshold,
                work_allowance,
            ),
            labels,
        )
    )
    x, y, amplitude, duration, noise, threshold, allowance = values
    if min(x, y, noise, threshold, allowance) < 0:
        raise ValueError(
            "radii, readout error, contrast threshold and work allowance must be nonnegative"
        )
    if amplitude <= 0 or not 0 < duration <= 1:
        raise ValueError("pulse_amplitude must be positive and 0 < probe_duration <= 1")
    geometry = _probe_geometry()
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    target = assess_sine_two_port_compatibility(
        classes=(2, 1), outer_refinements=32, inner_refinements=64
    )
    target_margin = (
        (2 * pi_interval() * target.acute_margin_turns_bounds).lo
        if target.acute_margin_turns_bounds is not None
        else None
    )
    target_admitted = (
        target.local_attraction_certified
        and target_margin is not None
        and target_margin > Q(1, 8)
    )
    q0 = x + Q(19, 6) * amplitude
    loop = Q(2, 3069) * duration
    denominator = 1 - loop**2
    qmax = (q0 + loop * y) / denominator
    pmax = y + loop * qmax
    heat = (amplitude * duration * (1 - duration) / 10, amplitude * duration / 10)
    background = Q(7, 120) * duration * x
    sine_error = duration * pmax / (3069 * 3)
    error = background + sine_error if target_admitted else None
    joined = (heat[0] - error, heat[1] + error) if error is not None else None
    recorded = (
        (joined[0] - 2 * noise, joined[1] + 2 * noise) if joined is not None else None
    )
    contrast = (
        (joined[0] - 4 * noise, joined[1] + 4 * noise) if joined is not None else None
    )
    response_margin = contrast[0] - threshold if contrast is not None else None
    response = response_margin is not None and response_margin > 0
    work_error = Q(7, 6) * amplitude * x
    work = (amplitude**2 - work_error, amplitude**2 + work_error)
    work_margin = allowance - work[1]
    energy = x**2 + y**2 + work[1]
    norm_squared = y**2 + q0**2
    radius_margin = Q(1, 144) - norm_squared
    storage_margin = Q(1, 648000) - energy
    identity = target_admitted and radius_margin > 0 and storage_margin > 0
    reasons = tuple(
        reason
        for flag, reason in (
            (target_admitted, "required_implicit_target_geometry_not_certified"),
            (response, "recorded_contrast_not_strictly_above_threshold"),
            (work_margin >= 0, "supplied_work_upper_bound_exceeds_allowance"),
            (radius_margin > 0, "strict_post_probe_local_radius_not_certified"),
            (storage_margin > 0, "strict_post_probe_capture_storage_not_certified"),
        )
        if not flag
    )
    return SineTwoPortProbe(
        **dict(zip(labels, values)),
        **geometry,
        reference_model=model,
        target=target,
        target_acute_margin_lower_bound=target_margin,
        target_admitted=target_admitted,
        joined_form_mean_increment=amplitude / 2,
        disconnected_form_mean_increments=(amplitude, Q(0)),
        post_probe_form_norm_upper_bound=q0,
        post_probe_relative_norm_squared_upper_bound=norm_squared,
        coupled_loop_gain=loop,
        coupled_denominator=denominator,
        whole_window_form_norm_candidate=qmax,
        whole_window_phase_norm_candidate=pmax,
        whole_window_form_norm_upper_bound=qmax if target_admitted else None,
        whole_window_phase_norm_upper_bound=pmax if target_admitted else None,
        ideal_heat_increment_bounds=heat,
        initial_form_response_error_upper_bound=background,
        sine_response_error_candidate=sine_error,
        response_error_upper_bound=error,
        joined_increment_bounds=joined,
        recorded_joined_increment_bounds=recorded,
        disconnected_increment_bounds=(Q(0), Q(0)),
        recorded_disconnected_increment_bounds=(-2 * noise, 2 * noise),
        recorded_contrast_bounds=contrast,
        response_margin=response_margin,
        joined_work_bounds=work,
        disconnected_work_bounds=(Q(0), Q(0)),
        work_allowance_margin=work_margin,
        positive_supplied_work_certified=work[0] > 0,
        post_probe_excess_storage_candidate=energy,
        post_probe_excess_storage_upper_bound=energy if target_admitted else None,
        post_probe_radius_margin=radius_margin,
        capture_storage_margin_candidate=storage_margin,
        capture_storage_margin=storage_margin if target_admitted else None,
        response_certified=response,
        work_certified=work_margin >= 0,
        joined_identity_certified=identity,
        recovery_certified=identity,
        status="certified_probe" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
