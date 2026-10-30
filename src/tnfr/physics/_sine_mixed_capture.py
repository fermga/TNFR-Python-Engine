"""Acquired joint registration of two unlike, independently formed C9s.

This fixed degree-three sine protocol retains both original winding classes,
the marked phase-compatible matching, and every fine-state source residual.
Its conditional full-state capture proof and disconnected comparison belong
to ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md. No event is executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_SOURCE_ALLOWANCE = Q(1, 10**10)
_READOUT_ALLOWANCE = Q(1, 10**6)
_EPSILON = Q(1, 10**70)
_ETA = 20 * _EPSILON
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 12)
_COERCIVITY = Q(1, 100)
_BARRIER = _COERCIVITY * _RADIUS**2
_DEADLINE = Q(2 * 10**10)
_GAMMA_BOUNDS = (Q(1, 3300), Q(1, 3000))
_CLOCK_MATRIX = Q(4, 3)
_DECAY_POWER = 12


@dataclass(frozen=True)
class _MixedConstitutivePair:
    """Fixed actual support and registered winding-one/winding-two carrier.

    The matching is not an automorphism of either marked cycle. Integer
    contact branches retain the difference of the original continuous lifts;
    phase equality modulo one turn does not discard that association.
    """

    nodes: tuple[int, ...]
    internal_edges: tuple[tuple[int, int, Q], ...]
    contact_edges: tuple[tuple[int, int, Q], ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    degrees: tuple[int, ...]
    matching_permutation: tuple[int, ...]
    target_phase_turns: tuple[Q, ...]
    contact_branch_turns: tuple[Q, ...]
    internal_cosine_bounds: tuple[I, I]
    gamma_bounds: I
    internal_storage_bounds: I
    phase_alignment_certified: bool
    support_certified: bool
    same_slot_offset_turns: tuple[Q, ...]
    same_slot_carrier_compatible: bool
    class_order: tuple[int, int] = (1, 2)
    degree: int = 3
    clock_factor: Q = Q(1, 3)
    held_capacities: tuple[Q, ...] = (Q(1),) * 18
    clock: str = "tau=(1023/1024)*t; joined factor1/3 in both rows"
    scope: tuple[str, ...] = (
        "two_different_acquired_winding_classes_with_fixed_marked_ports",
        "both_internal_cycles_and_the_phase_adapted_perfect_matching_retained",
        "exact_four_coordinate_carrier_not_generic_fine_state_closure",
        "same_slot_failure_only_excludes_this_declared_carrier",
        "support_event_and_positive_work_are_supplied_not_selected",
    )


def _derive_mixed_constitutive_pair():
    """Build the one fixed heterogeneous geometry from exact primitive data."""
    nodes = tuple(range(18))
    permutation = tuple((5 * (j - 4) + 4) % 9 for j in range(9))
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(2) for j in range(9)
    )
    contact = tuple((j, 9 + permutation[j], Q(1)) for j in range(9))
    edges = internal + contact
    degrees = tuple(sum(node in (a, b) for a, b, _ in edges) for node in nodes)
    turns = tuple(Q(k * (j - 4), 9) for k in (1, 2) for j in range(9))
    branches = tuple(turns[9 + permutation[j]] - turns[j] for j in range(9))
    same_slot = tuple(turns[9 + j] - turns[j] for j in range(9))
    alpha = 2 * pi_interval() / 9
    cosines = (cos(alpha), cos(2 * alpha))
    return _MixedConstitutivePair(
        nodes=nodes,
        internal_edges=internal,
        contact_edges=contact,
        weighted_edges=edges,
        degrees=degrees,
        matching_permutation=permutation,
        target_phase_turns=turns,
        contact_branch_turns=branches,
        internal_cosine_bounds=cosines,
        gamma_bounds=Q(1, 1023) / pi_interval(),
        internal_storage_bounds=9 * (2 - cosines[0] - cosines[1]),
        phase_alignment_certified=all(value.denominator == 1 for value in branches),
        support_certified=(
            tuple(sorted(permutation)) == tuple(range(9))
            and degrees == (3,) * 18
            and len({tuple(sorted((a, b))) for a, b, _ in edges}) == 27
        ),
        same_slot_offset_turns=same_slot,
        same_slot_carrier_compatible=all(
            (value - same_slot[0]).denominator == 1 for value in same_slot
        ),
    )


@dataclass(frozen=True)
class _SineMixedCapture:
    """Conditional all-future registration after one fixed recovery deadline.

    Every allowance concerns the same actual fine source. The disconnected
    alternative has identical initial coordinates and keeps factor one half
    in both rows. Readout errors are separately bounded and may correlate.
    The common metric's V=4/3 only rewrites the supplied degree-three clock;
    the physical held capacities remain one. No reduced trajectory is run.
    """

    form_error_bound: Q
    phase_error_bound: Q
    observation_error_bound: Q
    geometry: _MixedConstitutivePair
    recoveries: tuple[_ConstitutiveRecovery, _ConstitutiveRecovery]
    source_allowances_within_protocol: bool
    observation_allowance_within_protocol: bool
    source_certified: bool
    isolated_waiting_certified: bool
    initial_identity_absent: bool
    isolated_excess_upper_bound: Q
    isolated_waiting_storage_margin: Q
    initial_registered_gap_bounds: tuple[Q, Q]
    initial_joint_norm_squared_upper_bound: Q
    post_attachment_excess_upper_bound: Q
    phase_hessian_lower_bound: Q
    full_graph_gap_lower_bound: Q
    radius_margin: Q
    storage_margin: Q
    nominal_attachment_work_bounds: I
    attachment_work_correction_upper_bound: Q
    common_metric_lower_bound: Q
    common_metric_upper_bound: Q
    common_metric_decay_rate: Q
    initial_metric_upper_bound: Q
    recovery_exponent_lower_bound: Q
    decay_upper_bound: Q
    returned_joint_norm_squared_upper_bound: Q
    returned_registered_gap_squared_upper_bound: Q
    returned_mean_registration_squared_upper_bound: Q
    returned_relative_form_row_defect_upper_bound: Q
    joint_return_margin: Q
    registration_margin: Q
    connected_registration_bounds: tuple[Q, Q]
    disconnected_registration_bounds: tuple[Q, Q]
    actual_contrast_bounds: tuple[Q, Q]
    recorded_contrast_bounds: tuple[Q, Q]
    contrast_margin: Q
    constituent_identity_certified: bool
    joint_identity_certified: bool
    response_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    recovery_duration: Q = _DEADLINE
    original_recovery_duration: Q = _DEADLINE * Q(1024, 1023)
    recovered_radius: Q = _EPSILON
    waiting_radius: Q = _ETA
    relative_phase_gap: Q = _DELTA
    registration_target: Q = _DELTA / 2
    returned_gap_bound: Q = _DELTA / 8
    response_target: Q = _DELTA / 2
    identity_radius: Q = _RADIUS
    storage_coercivity_lower_bound: Q = _COERCIVITY
    storage_barrier: Q = _BARRIER
    decay_power: int = _DECAY_POWER
    common_metric_gamma_bounds: tuple[Q, Q] = _GAMMA_BOUNDS
    common_metric_clock_matrix_factor: Q = _CLOCK_MATRIX
    source_allowance_limit: Q = _SOURCE_ALLOWANCE
    observation_allowance_limit: Q = _READOUT_ALLOWANCE
    supplied_phase_origins: tuple[Q, Q] = (-_DELTA / 2, _DELTA / 2)
    initial_form_means: tuple[Q, Q] = (Q(0), Q(0))
    observation: str = "mean marked contact phase gap; disconnected minus connected"
    clock: str = "tau=(1023/1024)*t; isolated/cut factor1/2, joined factor1/3"
    scope: tuple[str, ...] = (
        "fresh_original_class_one_and_class_two_preparations_and_recoveries",
        "supplied_original_phase_origins_no_later_phase_or_form_reset",
        "arbitrary_further_isolated_waiting_with_all_fine_residuals_retained",
        "full_state_trapping_retains_both_constituent_windings_for_all_time",
        "all_fine_registered_contact_gaps_return_and_stay_small_after_deadline",
        "joint_registration_initially_absent_and_acquired_under_actual_contact",
        "cut_comparator_uses_same_initial_source_and_its_own_complete_degree_two_law",
        "unequal_support_work_explicitly_supplied_not_paid_from_prior_loss",
        "separate_additive_readout_errors_allow_arbitrary_correlation",
        "carrier_coordinates_do_not_replace_actual_fine_state_in_the_capture_proof",
        "no_new_outer_winding_autonomous_assembly_sensor_or_physical_binding_claim",
    )


def _assess_sine_mixed_capture(
    *, form_error_bound, phase_error_bound, observation_error_bound
):
    """Rebuild fixed mixed capture from primitive source/readout allowances.

    All raw values admit before geometry, sources or interval arithmetic.
    Valid nonnegative allowances beyond the frozen caps return unavailable;
    candidate numbers alone never establish their missing source premises.
    No supplied report, elapsed-time choice or interface override is accepted.
    """
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    noise = _exact(observation_error_bound, "observation_error_bound")
    if min(rx, rt, noise) < 0:
        raise ValueError("source and observation error bounds must be nonnegative")

    source_limits = max(rx, rt) <= _SOURCE_ALLOWANCE
    noise_limit = noise <= _READOUT_ALLOWANCE
    geometry = _derive_mixed_constitutive_pair()
    recoveries = tuple(
        _assess_constitutive_recovery(
            deformation=0,
            form_error_bound=rx,
            phase_error_bound=rt,
            target_winding=winding,
        )
        for winding in (1, 2)
    )
    source_ok = source_limits and all(row.recovery_certified for row in recoveries)
    isolated_excess = 4 * _EPSILON**2
    waiting_margin = _BARRIER - isolated_excess
    waiting_geometry = all(
        row.target_winding == winding
        and row.source.phase_hessian_lower_bound is not None
        and row.source.phase_hessian_lower_bound > Q(1, 20)
        and row.source.coercivity_lower_bound is not None
        and row.source.coercivity_lower_bound > _COERCIVITY
        and row.radius == _RADIUS
        and row.endpoint_radius == _EPSILON
        for winding, row in zip((1, 2), recoveries)
    )
    waiting_ok = source_ok and waiting_geometry and waiting_margin > 0
    initial_gaps = (_DELTA - 2 * _ETA, _DELTA + 2 * _ETA)
    initial_absence = waiting_ok and initial_gaps[0] > _DELTA / 2

    # The mixed target is stationary node by node. The perfect-matching
    # gap proof uses child means and zero-mean residuals, not a product graph.
    curvature = cos(4 * pi_interval() / 9 + sqrt(I(2)) * _RADIUS).lo
    geometry_ok = (
        geometry.support_certified
        and geometry.phase_alignment_certified
        and curvature > Q(1, 20)
        and 2 - 2 * geometry.internal_cosine_bounds[0].hi > Q(2, 5)
        and _GAMMA_BOUNDS[0] <= geometry.gamma_bounds.lo
        and geometry.gamma_bounds.hi <= _GAMMA_BOUNDS[1]
    )
    initial_norm = Q(9, 2) * _DELTA**2 + 4 * _ETA**2
    excess = 3 * initial_norm
    radius_margin = _RADIUS**2 - initial_norm
    storage_margin = _BARRIER - excess
    retained = waiting_ok and geometry_ok and radius_margin > 0 and storage_margin > 0

    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=_GAMMA_BOUNDS[0],
        gamma_upper=_GAMMA_BOUNDS[1],
        cosine_lower=Q(1, 20),
        gap_lower=Q(2, 5),
        capacity_lower=_CLOCK_MATRIX,
        capacity_upper=_CLOCK_MATRIX,
    )
    initial_metric = upper * initial_norm
    exponent = rate * _DEADLINE
    # e > 2 and exponent > 12 give this exact rational upper bound without
    # a large-duration exponential evaluator or a numerical settling trace.
    decay = Q(1, 2**_DECAY_POWER)
    returned = initial_metric * decay / lower
    gap_squared = 2 * returned
    mean_gap_squared = Q(2, 9) * returned
    # The exact live coordinates u=mean(x1)-mean(x0), J=mean(contact gap)
    # satisfy J'=2*gamma*u/3. Only the mean sine in u' needs this residual
    # allowance; it is not an independently evolved reduced trajectory.
    relative_form_defect = Q(2, 27) * _GAMMA_BOUNDS[1] * returned
    return_margin = (_DELTA / 16) ** 2 - returned
    registration_margin = (_DELTA / 8) ** 2 - gap_squared
    joint_ok = (
        retained
        and initial_absence
        and exponent > _DECAY_POWER
        and return_margin > 0
        and registration_margin > 0
    )
    connected = (-_DELTA / 8, _DELTA / 8)
    disconnected = (_DELTA, _DELTA)
    actual = (_DELTA - connected[1], _DELTA - connected[0])
    recorded = (actual[0] - 2 * noise, actual[1] + 2 * noise)
    contrast_margin = recorded[0] - _DELTA / 2
    response_ok = joint_ok and noise_limit and contrast_margin > 0
    reasons = tuple(
        label
        for passed, label in (
            (source_limits, "source_allowance_exceeds_fixed_protocol"),
            (source_ok, "both_original_class_recoveries_not_certified"),
            (waiting_ok, "isolated_actual_source_waiting_not_certified"),
            (initial_absence, "initial_joint_registration_absence_not_certified"),
            (geometry_ok, "mixed_full_state_geometry_not_certified"),
            (radius_margin > 0, "mixed_initial_radius_not_certified"),
            (storage_margin > 0, "mixed_storage_barrier_not_certified"),
            (exponent > _DECAY_POWER, "fixed_common_metric_decay_not_certified"),
            (return_margin > 0, "fixed_joint_return_not_certified"),
            (registration_margin > 0, "all_contact_registration_return_not_certified"),
            (noise_limit, "observation_allowance_exceeds_fixed_protocol"),
            (contrast_margin > 0, "fixed_recorded_contrast_target_not_certified"),
        )
        if not passed
    )
    return _SineMixedCapture(
        form_error_bound=rx,
        phase_error_bound=rt,
        observation_error_bound=noise,
        geometry=geometry,
        recoveries=recoveries,
        source_allowances_within_protocol=source_limits,
        observation_allowance_within_protocol=noise_limit,
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        initial_identity_absent=initial_absence,
        isolated_excess_upper_bound=isolated_excess,
        isolated_waiting_storage_margin=waiting_margin,
        initial_registered_gap_bounds=initial_gaps,
        initial_joint_norm_squared_upper_bound=initial_norm,
        post_attachment_excess_upper_bound=excess,
        phase_hessian_lower_bound=Q(1, 20),
        full_graph_gap_lower_bound=Q(2, 5),
        radius_margin=radius_margin,
        storage_margin=storage_margin,
        nominal_attachment_work_bounds=9 * (1 - cos(I(_DELTA))),
        attachment_work_correction_upper_bound=4 * _ETA**2,
        common_metric_lower_bound=lower,
        common_metric_upper_bound=upper,
        common_metric_decay_rate=rate,
        initial_metric_upper_bound=initial_metric,
        recovery_exponent_lower_bound=exponent,
        decay_upper_bound=decay,
        returned_joint_norm_squared_upper_bound=returned,
        returned_registered_gap_squared_upper_bound=gap_squared,
        returned_mean_registration_squared_upper_bound=mean_gap_squared,
        returned_relative_form_row_defect_upper_bound=relative_form_defect,
        joint_return_margin=return_margin,
        registration_margin=registration_margin,
        connected_registration_bounds=connected,
        disconnected_registration_bounds=disconnected,
        actual_contrast_bounds=actual,
        recorded_contrast_bounds=recorded,
        contrast_margin=contrast_margin,
        constituent_identity_certified=retained,
        joint_identity_certified=joint_ok,
        response_certified=response_ok,
        certified=response_ok,
        status="certified_mixed_organized_capture" if response_ok else "unavailable",
        unresolved_conditions=reasons,
    )
