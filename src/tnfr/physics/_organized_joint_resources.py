"""Joint source, recovery, observation and finite-contact resource admission.

All source evidence is rebuilt from primitives. This conditional calculation
does not execute a trajectory or replace any frozen earlier preparation.
Proof and prospective corner: ORGANIZED_JOINT_RESOURCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_external_registration import _quadrature_norm
from ._organized_outer_observation import (
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
)
from ._organized_outer_winding import (
    _derive_organized_outer_resource_bounds,
    _OrganizedOuterResourceBounds,
)
from ._organized_sequential_observation import (
    _assess_organized_sequential_observation_resources,
    _OrganizedSequentialObservationProtocol,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from ._sine_recovery_resources import (
    _assess_sine_recovery_resources,
    _SineRecoveryResources,
)

_SOURCE_LIMIT = Q(1, 10**6)
_CONTACT_HORIZON = Q(1, 1000)
_UPPER, _LOWER = Q(1, 10**8), Q(3, 10**7)


@dataclass(frozen=True)
class _OrganizedJointResources:
    """Fresh sufficient evidence and separately scoped necessary restrictions.

    ``phase_compression_time_lower_bound`` concerns total isolated preparation
    time, including formation. Subtract 200 for a recovery-dwell restriction.
    None means the selected phase-pair argument gives no positive bound.
    ``historical_resource_upper_bound`` sums initial preparation storage and
    later absolute event work; it is not a derived laboratory energy cost.
    A nonpositive necessary contact signal margin upper bound excludes the
    requested witness; a positive upper bound alone does not admit it.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    recoveries: tuple[_SineRecoveryResources, ...]
    outer_resources: _OrganizedOuterResourceBounds
    observation: _OrganizedSequentialObservationProtocol
    recovered_radius: Q
    quarter_dwell: Q
    decay_power: int
    sequential_readout_error: Q
    contact_readout_error: Q
    relative_preparation_dimension: int
    source_inclusion_margins: tuple[tuple[str, Q], ...]
    contact_form_upper_bound: Q
    observation_phase_drift_upper_bound: Q
    contact_edge_drift_upper_bound: Q
    response_remainder_upper_bound: Q
    boundary_cosine_bounds: tuple[I, I]
    boundary_sine_bounds: tuple[I, I]
    boundary_amplitude_bounds: tuple[I, I]
    capability_bounds: tuple[I, I]
    fixed_origin_contrast_bounds: I
    contact_work_upper_bound: Q
    joined_storage_upper_bounds: tuple[Q, Q]
    primitive_storage_upper_bound: Q
    historical_resource_upper_bound: Q
    fixed_outer_work_lower_bound: Q
    winding_work_time_lower_bound: I
    phase_compression_time_lower_bound: Q | None
    necessary_contact_noise_margin: Q
    necessary_contact_signal_margin_upper_bound: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    admitted: bool
    unresolved_conditions: tuple[str, ...]
    contact_horizon: Q = _CONTACT_HORIZON
    scope: tuple[str, ...] = (
        "new_preparation_domain_not_a_rewrite_of_frozen_response_records",
        "same_sine_law_support_mean_correlations_and_supplied_observation_word",
        "primitive_component_widths_and_recovered_euclidean_radii_are_distinct",
        "independent_sources_share_bounds_not_reached_states",
        "all_ninety_six_relative_source_directions_at_the_positive_corner",
        "finite_contact_identity_and_capability_not_indefinite_outer_one_capture",
        "initial_storage_and_paid_event_work_remain_separate_ledger_terms",
        "necessary_resource_bounds_are_not_sufficient_capture_certificates",
        "no_trajectory_origin_scan_frozen_producer_replay_or_laboratory_bridge",
    )


def _assess_organized_joint_resources(
    *,
    form_error_bounds,
    phase_error_bounds,
    quarter_dwell,
    recovered_radius,
    decay_power,
    sequential_readout_error,
    contact_readout_error,
) -> _OrganizedJointResources:
    """Rebuild the coupled sufficient domain from all twelve source allowances."""
    rx = _vector(form_error_bounds, "form_error_bounds", 6)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 6)
    dwell = _exact(quarter_dwell, "quarter_dwell")
    q = _exact(recovered_radius, "recovered_radius")
    eta_o = _exact(sequential_readout_error, "sequential_readout_error")
    eta_c = _exact(contact_readout_error, "contact_readout_error")
    if min(*rx, *rt) < 0 or max(*rx, *rt) > _SOURCE_LIMIT:
        raise ValueError("source component allowances require [0, 1e-6]")
    if dwell < 0 or not 0 < q <= Q(1, 12) or min(eta_o, eta_c) < 0:
        raise ValueError(
            "nonnegative dwell/noise and 0 < recovered_radius <= 1/12 required"
        )
    if type(decay_power) is not int or not 1 <= decay_power <= 4096:
        raise ValueError("decay_power requires an exact integer in 1..4096")

    # Reuse equal proof bounds only within this fresh assessment. This does
    # not equate the independent source members or any reached coordinates.
    local_bounds = {}
    recoveries = []
    for a, b in zip(rx, rt):
        if (a, b) not in local_bounds:
            local_bounds[a, b] = _assess_sine_recovery_resources(
                form_error_bound=a,
                phase_error_bound=b,
                quarter_dwell=dwell,
                endpoint_radius=q,
                decay_power=decay_power,
            )
        recoveries.append(local_bounds[a, b])
    recoveries = tuple(recoveries)
    outer = _derive_organized_outer_resource_bounds(recovered_radius=q)
    observation = _assess_organized_sequential_observation_resources(
        raw_readout_error=eta_o
    )
    gamma, pi = outer.geometry.gamma_bounds, pi_interval()
    g, h = gamma.hi, _CONTACT_HORIZON
    r, z = outer.full_form_error_upper_bound, outer.full_phase_error_upper_bound
    primitive = outer.primitive_entry
    origin_gap = max(box.abs_max for box in primitive.capture.phase_edge_gap_bounds)
    inclusion = (
        (
            "outer_one_form",
            _FORM_BOUND - max(v.abs_max for v in primitive.endpoint_form_bounds) - r,
        ),
        ("outer_zero_form", _FORM_BOUND - r),
        ("reprojected_fine_phase", _FINE_PHASE_BOUND - 2 * z),
        ("outer_one_origin_gap", _OUTER_ANGLE_BOUND - origin_gap - 2 * z),
        ("outer_zero_origin_gap", _OUTER_ANGLE_BOUND - 2 * z),
    )
    b_obs = observation.observation_form_upper_bound
    carried = 2 * g * b_obs * observation.total_elapsed_scaled_time
    error_box = I(-z - carried - q, z + carried + q)
    phase_families = (
        tuple(box + error_box for box in primitive.endpoint_phase_bounds),
        (error_box,) * 5,
    )
    cosines = tuple(
        sum((cos(box) for box in phases), I(0)) for phases in phase_families
    )
    sines = tuple(-sum((sin(box) for box in phases), I(0)) for phases in phase_families)
    amplitudes = tuple(_quadrature_norm(c, s) for c, s in zip(cosines, sines))
    b = max(b_obs, q) + Q(7, 4) * g * h
    contact_drift = 7 * g * b * h
    remainder = Q(15, 16) * g * h**2 + Q(15, 4) * g**2 * b * h**2
    error = remainder + 2 * eta_c
    factor = gamma * h / 4
    capability = tuple(
        I(max(Q(0), (factor * a).lo - error), (factor * a).hi + error)
        for a in amplitudes
    )
    contrast = factor * (cosines[1] - cosines[0]) + I(-2 * error, 2 * error)
    work = 90 + Q(45, 2) * (b_obs + q) ** 2
    v9 = outer.internal_cycle_minimum_bounds
    host_storage = (
        outer.target_storage_bounds.hi + outer.endpoint_excess_upper_bound,
        5 * v9.hi + 40 * q**2,
    )
    observation_work = observation.total_absolute_event_work_upper_bound
    joined = tuple(
        value + observation_work + v9.hi + 4 * q**2 + work for value in host_storage
    )
    alpha = outer.geometry.alpha_bounds
    drift = observation.continuous_edge_drift_upper_bound
    identity = (
        (
            "host_internal_positive",
            alpha.lo - 2 * _FINE_PHASE_BOUND - drift - contact_drift,
        ),
        (
            "host_internal_acute",
            pi.lo / 2 - alpha.hi - 2 * _FINE_PHASE_BOUND - drift - contact_drift,
        ),
        ("donor_internal_positive", alpha.lo - 2 * q - contact_drift),
        ("donor_internal_acute", pi.lo / 2 - alpha.hi - 2 * q - contact_drift),
        (
            "outer_acute",
            pi.lo / 2
            - _OUTER_ANGLE_BOUND
            - 2 * _FINE_PHASE_BOUND
            - drift
            - contact_drift,
        ),
    )
    margins = (
        inclusion
        + observation.sufficient_margins
        + identity
        + (
            ("outer_capture", outer.capture_margin),
            ("outer_control_capture", outer.control_capture_margin),
            ("outer_attachment_work", outer.attachment_work_margin),
            ("outer_initial_storage", outer.initial_storage_margin),
            ("contact_attachment_work", Q(100) - work),
            ("outer_one_joined_storage", Q(200) - joined[0]),
            ("outer_zero_joined_storage", Q(200) - joined[1]),
            ("outer_one_all_origin_response", _UPPER - capability[0].hi),
            ("outer_zero_witness_response", capability[1].lo - _LOWER),
            ("fixed_origin_response_contrast", contrast.lo - Q(1, 10**7)),
        )
    )
    flags = (
        ("recovered_sources", all(s.recovery_certified for s in recoveries)),
        ("outer_full_fine_capture", outer.conditional_capture_certified),
        ("outer_control_capture_domain", outer.control_conditional_capture_certified),
        ("outer_attachment_budget", outer.attachment_budget_certified),
        ("sequential_observation", observation.admitted),
    )
    unresolved = tuple(name for name, ok in flags if not ok) + tuple(
        name for name, margin in margins if margin <= 0
    )
    storage = sum((s.source.initial_storage_upper_bounds[0] for s in recoveries), Q(0))
    phase_width = max(rt)
    compression = None
    if phase_width > q:
        u = 1 - q / phase_width
        compression = (2 * u / (2 + u)) / g**2
    work_time = (
        9 * outer.outer_cycle_minimum_bounds
        + 18 * (pi - 2 * q) ** 2 / (gamma**2 * outer.primitive_entry.scaled_time * 2)
        - 20 * q**2
    )
    return _OrganizedJointResources(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        recoveries=recoveries,
        outer_resources=outer,
        observation=observation,
        recovered_radius=q,
        quarter_dwell=dwell,
        decay_power=decay_power,
        sequential_readout_error=eta_o,
        contact_readout_error=eta_c,
        relative_preparation_dimension=sum(
            s.source.preparation_dimension for s in recoveries
        ),
        source_inclusion_margins=inclusion,
        contact_form_upper_bound=b,
        observation_phase_drift_upper_bound=carried,
        contact_edge_drift_upper_bound=contact_drift,
        response_remainder_upper_bound=remainder,
        boundary_cosine_bounds=cosines,
        boundary_sine_bounds=sines,
        boundary_amplitude_bounds=amplitudes,
        capability_bounds=capability,
        fixed_origin_contrast_bounds=contrast,
        contact_work_upper_bound=work,
        joined_storage_upper_bounds=joined,
        primitive_storage_upper_bound=storage,
        historical_resource_upper_bound=storage
        + outer.attachment_work_upper_bound
        + observation_work
        + work,
        fixed_outer_work_lower_bound=outer.attachment_nominal_work,
        winding_work_time_lower_bound=work_time,
        phase_compression_time_lower_bound=compression,
        necessary_contact_noise_margin=_UPPER - 2 * eta_c,
        necessary_contact_signal_margin_upper_bound=5 * g * h / 4 - _LOWER - 2 * eta_c,
        sufficient_margins=margins,
        source_certified=all(s.recovery_certified for s in recoveries),
        admitted=not unresolved,
        unresolved_conditions=unresolved,
    )
