"""Mixed acquired registration under independently held contact conductances.

This explicitly selected detached law keeps bare-neighbor sine current and
bare-degree normalization, while conductance enters the form gradient and
reciprocal phase transport. It is not native strength-normalized EPI flow.
The existing mixed unit-contact reader supplies only precontact evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, cos, sqrt
from ._sine_flow import _held_sine_neighbor_field
from ._sine_lyapunov import _sine_common_quadratic_coefficients
from ._sine_mixed_capture import (
    _assess_sine_mixed_capture,
    _derive_mixed_constitutive_pair,
    _SineMixedCapture,
)
from .relational_observations import _ordered

_CONTACT_RANGE = (Q(3, 4), Q(5, 4))
_LOSS = Q(1023, 1024)
_SOURCE_LIMIT = Q(1, 10**10)
_READOUT_LIMIT = Q(1, 10**6)
_EPSILON = Q(1, 10**70)
_ETA = 20 * _EPSILON
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 12)
_BARRIER = Q(1, 14400)
_DEADLINE = Q(2 * 10**10)
_PHASE_SCALE = Q(6, 5)
_SCALED_GAMMA = (Q(1, 2750), Q(1, 2500))
_GAMMA_UPPER = Q(1, 3000)


def _contact_conductances(values):
    """Normalize nine nonnegative primitive weights before numerical work."""
    raw = _ordered(values, "contact_conductances", limit=10)
    if len(raw) != 9:
        raise ValueError("contact_conductances requires nine weights in marked j order")
    weights = tuple(
        _exact(value, f"contact_conductances[{j}]") for j, value in enumerate(raw)
    )
    if any(value < 0 for value in weights):
        raise ValueError("contact conductances must be nonnegative")
    return weights


def _within_contact_box(weights):
    return all(_CONTACT_RANGE[0] <= value <= _CONTACT_RANGE[1] for value in weights)


def _weighted_edges(geometry, weights):
    return geometry.internal_edges + tuple(
        (a, b, weight) for (a, b, _), weight in zip(geometry.contact_edges, weights)
    )


def _mixed_conductance_field(*, contact_conductances):
    """Compile the complete selected law in tau, only on its protocol box.

    The callable consumes all 18 forms followed by all 18 lifted phases,
    using the shared field's exact/interval or common-order Taylor-jet domain.
    Conductance weights multiply the form gradient, never the sine current.
    Both original-clock rows are divided by e=1023/1024. This function performs
    no source reconstruction, trajectory, projection or native dispatch.
    """
    weights = _contact_conductances(contact_conductances)
    if not _within_contact_box(weights):
        raise ValueError("field requires every contact conductance in [3/4, 5/4]")
    geometry = _derive_mixed_constitutive_pair()
    edges = _weighted_edges(geometry, weights)
    rows = tuple(
        tuple((b if i == a else a, 1, w) for a, b, w in edges if i in (a, b))
        for i in geometry.nodes
    )
    model = RelationalExchangeModel(
        1, epi_weight=_LOSS, phase_weight=Q(1, 1024), phase_domain="regular"
    )
    evaluate = _held_sine_neighbor_field(
        model, geometry.degrees, geometry.held_capacities, rows
    )

    def flow(values):
        return tuple(value / _LOSS for value in evaluate(values))

    return flow


@dataclass(frozen=True)
class _SineMixedConductance:
    """Conditional uniform-box capture with point-associated collective errors.

    ``precontact_reference`` is rebuilt from primitive source allowances. Only
    its geometry, acquisitions, recoveries and isolated waiting are consumed;
    its unit joined work, metric, return and success verdict are not premises.
    Candidate bounds remain conditional when a protocol premise is unavailable.

    ``actual_source_aliases_certified`` is an existential nonclosure result at
    contact initialization for positive source-error interiors and unequal
    weights. It authenticates no explicit witness endpoint or inverse radius.
    The scaled phase and kernel matrix are coordinate/clock representations;
    physical held capacities remain one.
    """

    contact_conductances: tuple[Q, ...]
    form_error_bound: Q
    phase_error_bound: Q
    observation_error_bound: Q
    precontact_reference: _SineMixedCapture
    weighted_edges: tuple[tuple[int, int, Q], ...]
    bare_degrees: tuple[int, ...]
    conductance_strengths: tuple[Q, ...]
    mean_contact_conductance: Q
    contact_conductance_variance: Q
    conductances_within_protocol: bool
    source_certified: bool
    isolated_waiting_certified: bool
    initial_identity_absent: bool
    initial_joint_norm_squared_upper_bound: Q
    post_attachment_excess_upper_bound: Q
    radius_margin: Q
    storage_margin: Q
    nominal_attachment_work_bounds: I
    attachment_work_correction_upper_bound: Q
    common_metric_lower_bound: Q
    common_metric_upper_bound: Q
    common_metric_decay_rate: Q
    initial_scaled_norm_squared_upper_bound: Q
    initial_metric_upper_bound: Q
    recovery_exponent_lower_bound: Q
    decay_upper_bound: Q
    returned_joint_norm_squared_upper_bound: Q
    returned_registered_gap_squared_upper_bound: Q
    returned_mean_registration_squared_upper_bound: Q
    joint_return_margin: Q
    registration_margin: Q
    returned_weighted_correlation_squared_upper_bound: Q
    returned_weighted_correlation_upper_bound: Q
    returned_relative_form_row_defect_upper_bound: Q
    returned_relative_phase_row_defect_upper_bound: Q
    carrier_compatible: bool
    actual_source_aliases_certified: bool
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
    contact_conductance_range: tuple[Q, Q] = _CONTACT_RANGE
    source_allowance_limit: Q = _SOURCE_LIMIT
    observation_allowance_limit: Q = _READOUT_LIMIT
    recovered_radius: Q = _EPSILON
    waiting_radius: Q = _ETA
    relative_phase_gap: Q = _DELTA
    identity_radius: Q = _RADIUS
    registration_target: Q = _DELTA / 2
    returned_gap_bound: Q = _DELTA / 8
    response_target: Q = _DELTA / 2
    storage_coercivity_lower_bound: Q = Q(1, 100)
    storage_barrier: Q = _BARRIER
    full_graph_gap_lower_bound: Q = Q(3, 10)
    full_laplacian_upper_bound: Q = Q(13, 2)
    phase_hessian_lower_bound: Q = Q(1, 25)
    phase_hessian_upper_bound: Q = Q(4, 3)
    phase_scale: Q = _PHASE_SCALE
    common_metric_gamma_bounds: tuple[Q, Q] = _SCALED_GAMMA
    common_metric_cosine_lower_bound: Q = Q(1, 36)
    common_metric_clock_matrix_factor: Q = Q(4, 3)
    recovery_duration: Q = _DEADLINE
    original_recovery_duration: Q = _DEADLINE / _LOSS
    decay_power: int = 10
    law: str = "held_bare_degree_sine_weighted_form_loss_and_reciprocal_transport"
    clock: str = "tau=(1023/1024)*t; joined bare divisor3, isolated/cut divisor2"
    scope: tuple[str, ...] = (
        "fixed_independent_contact_weights_whole_box_not_time_varying_control",
        "existing_detached_neighbor_field_bare_sine_and_bare_degree_in_both_rows",
        "not_native_strength_normalized_EPI_or_native_resultant_phase",
        "not_weighted_sine_potential_or_uniquely_selected_extension_of_unit_law",
        "fresh_unit_reference_only_for_precontact_geometry_source_and_waiting",
        "actual_weighted_joined_trap_work_and_return_recomputed",
        "all_fine_residuals_and_original_phase_origins_retained_without_reset",
        "all_future_mixed_identity_after_fixed_deadline_and_same_state_cut_control",
        "point_weight_variance_controls_linear_fine_correlation_in_both_mean_rows",
        "nonuniform_source_aliases_existential_not_the_fixed_explicit_witness",
        "no_autonomous_reduced_trajectory_event_selection_or_physical_calibration",
    )


def _assess_sine_mixed_conductance(
    *,
    contact_conductances,
    form_error_bound,
    phase_error_bound,
    observation_error_bound,
):
    """Rebuild source and weighted capture after admitting every raw primitive.

    Nonnegative conductances outside the fixed box, including zero, retain
    their bare support and yield unavailable certification. They do not alter
    the sine incidence or select a different divisor. The separate field
    builder rejects such out-of-protocol weights before compiling its rows.
    """
    weights = _contact_conductances(contact_conductances)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    noise = _exact(observation_error_bound, "observation_error_bound")
    if min(rx, rt, noise) < 0:
        raise ValueError("source and observation error bounds must be nonnegative")

    in_box = _within_contact_box(weights)
    source_limits = max(rx, rt) <= _SOURCE_LIMIT
    noise_limit = noise <= _READOUT_LIMIT
    geometry = _derive_mixed_constitutive_pair()
    reference = _assess_sine_mixed_capture(
        form_error_bound=rx, phase_error_bound=rt, observation_error_bound=noise
    )
    source_ok = (
        source_limits
        and reference.source_certified
        and all(
            row.target_winding == winding and row.recovery_certified
            for winding, row in zip((1, 2), reference.recoveries)
        )
        and len(reference.recoveries) == 2
    )
    waiting_ok = (
        source_ok
        and reference.isolated_waiting_certified
        and reference.recovered_radius == _EPSILON
        and reference.waiting_radius == _ETA
    )
    geometry_ok = (
        reference.geometry == geometry
        and geometry.support_certified
        and geometry.phase_alignment_certified
        and Q(1, 3300) <= geometry.gamma_bounds.lo
        and geometry.gamma_bounds.hi <= _GAMMA_UPPER
    )
    edges = _weighted_edges(geometry, weights)
    strengths = tuple(
        sum((w for a, b, w in edges if i in (a, b)), Q(0)) for i in geometry.nodes
    )
    mean = sum(weights, Q(0)) / 9
    variance = sum(((w - mean) ** 2 for w in weights), Q(0)) / 9
    initial_absence = waiting_ok and _DELTA - 2 * _ETA > _DELTA / 2
    initial_norm = Q(9, 2) * _DELTA**2 + 4 * _ETA**2
    excess = Q(13, 4) * initial_norm
    radius_margin = _RADIUS**2 - initial_norm
    storage_margin = _BARRIER - excess
    retained = (
        in_box
        and waiting_ok
        and geometry_ok
        and radius_margin > 0
        and storage_margin > 0
    )

    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=_SCALED_GAMMA[0],
        gamma_upper=_SCALED_GAMMA[1],
        cosine_lower=Q(1, 36),
        gap_lower=Q(3, 10),
        capacity_lower=Q(4, 3),
        capacity_upper=Q(4, 3),
    )
    scaled_norm = _PHASE_SCALE**2 * initial_norm
    initial_metric = upper * scaled_norm
    exponent = rate * _DEADLINE
    # e > 27/10 and 27**7 > 1024*10**7 prove the exact factor below when
    # exponent > 7. No settling trace or exponential evaluation is needed.
    decay = Q(1, 1024)
    returned = initial_metric * decay / lower
    return_margin = _DELTA**2 / 128 - returned
    registration_margin = (_DELTA / 8) ** 2 - 2 * returned
    joint_ok = (
        retained
        and initial_absence
        and exponent > 7
        and return_margin > 0
        and registration_margin > 0
    )

    correlation_squared = Q(2, 9) * variance * returned
    spread = max(abs(w - mean) for w in weights)
    # Keep the exact conductance scale outside the display interval grid.
    # The squared enclosure is rational even for subgrid perturbations.
    correlation = (
        spread * sqrt(I(correlation_squared / spread**2)).hi if spread else Q(0)
    )
    form_defect = Q(2, 3) * correlation + Q(2, 27) * _GAMMA_UPPER * returned
    phase_defect = Q(2, 3) * _GAMMA_UPPER * correlation
    carrier = variance == 0
    aliases = (
        in_box and geometry_ok and waiting_ok and rx > 0 and rt > 0 and not carrier
    )
    connected, disconnected = (-_DELTA / 8, _DELTA / 8), (_DELTA, _DELTA)
    actual = (Q(7, 8) * _DELTA, Q(9, 8) * _DELTA)
    recorded = (actual[0] - 2 * noise, actual[1] + 2 * noise)
    contrast_margin = recorded[0] - _DELTA / 2
    response_ok = joint_ok and noise_limit and contrast_margin > 0
    reasons = tuple(
        label
        for passed, label in (
            (in_box, "contact_conductance_outside_fixed_positive_box"),
            (source_limits, "source_allowance_exceeds_fixed_protocol"),
            (source_ok, "both_original_class_recoveries_not_certified"),
            (waiting_ok, "isolated_actual_source_waiting_not_certified"),
            (geometry_ok, "precontact_mixed_geometry_not_certified"),
            (initial_absence, "initial_joint_registration_absence_not_certified"),
            (radius_margin > 0, "weighted_initial_radius_not_certified"),
            (storage_margin > 0, "weighted_storage_barrier_not_certified"),
            (exponent > 7, "fixed_scaled_metric_decay_not_certified"),
            (return_margin > 0, "weighted_joint_return_not_certified"),
            (registration_margin > 0, "all_contact_registration_return_not_certified"),
            (noise_limit, "observation_allowance_exceeds_fixed_protocol"),
            (contrast_margin > 0, "fixed_recorded_contrast_target_not_certified"),
        )
        if not passed
    )
    return _SineMixedConductance(
        contact_conductances=weights,
        form_error_bound=rx,
        phase_error_bound=rt,
        observation_error_bound=noise,
        precontact_reference=reference,
        weighted_edges=edges,
        bare_degrees=geometry.degrees,
        conductance_strengths=strengths,
        mean_contact_conductance=mean,
        contact_conductance_variance=variance,
        conductances_within_protocol=in_box,
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        initial_identity_absent=initial_absence,
        initial_joint_norm_squared_upper_bound=initial_norm,
        post_attachment_excess_upper_bound=excess,
        radius_margin=radius_margin,
        storage_margin=storage_margin,
        nominal_attachment_work_bounds=9 * (1 - cos(I(_DELTA))),
        attachment_work_correction_upper_bound=Q(9, 2) * _ETA**2,
        common_metric_lower_bound=lower,
        common_metric_upper_bound=upper,
        common_metric_decay_rate=rate,
        initial_scaled_norm_squared_upper_bound=scaled_norm,
        initial_metric_upper_bound=initial_metric,
        recovery_exponent_lower_bound=exponent,
        decay_upper_bound=decay,
        returned_joint_norm_squared_upper_bound=returned,
        returned_registered_gap_squared_upper_bound=2 * returned,
        returned_mean_registration_squared_upper_bound=Q(2, 9) * returned,
        joint_return_margin=return_margin,
        registration_margin=registration_margin,
        returned_weighted_correlation_squared_upper_bound=correlation_squared,
        returned_weighted_correlation_upper_bound=correlation,
        returned_relative_form_row_defect_upper_bound=form_defect,
        returned_relative_phase_row_defect_upper_bound=phase_defect,
        carrier_compatible=carrier,
        actual_source_aliases_certified=aliases,
        connected_registration_bounds=connected,
        disconnected_registration_bounds=disconnected,
        actual_contrast_bounds=actual,
        recorded_contrast_bounds=recorded,
        contrast_margin=contrast_margin,
        constituent_identity_certified=retained,
        joint_identity_certified=joint_ok,
        response_certified=response_ok,
        certified=response_ok,
        status="certified_mixed_contact_robustness" if response_ok else "unavailable",
        unresolved_conditions=reasons,
    )
