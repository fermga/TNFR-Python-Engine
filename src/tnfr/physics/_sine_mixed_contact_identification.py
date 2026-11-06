"""Necessary receiver-record sets with independently unknown contact weights.

The two complete loss laws and the acquired source protocol are unchanged.
Each hypothesis may choose its own weights and fine source residuals from the
same supplied information. Interval inclusion is not physical calibration, and
overlap of necessary outer bounds is not observational equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, sin
from ._sine_mixed_capture import _derive_mixed_constitutive_pair
from ._sine_mixed_normalization import (
    _DELTA,
    _EARLY_TARGET,
    _EARLY_TIME,
    _ETA,
    _FORM_READOUT_LIMIT,
    _GAMMA_LOWER,
    _GAMMA_UPPER,
    _REGISTRATION_LIMIT,
    _SOURCE_LIMIT,
    _assess_sine_mixed_normalization,
    _SineMixedNormalization,
)
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered

_CONTACT_RANGE = (Q(3, 4), Q(5, 4))
_FORCING_LOWER = Q(1, 10240000)
_FORCING_WIDTH_LIMIT = Q(1, 10**12)


def _contact_conductance_bounds(values):
    """Detach nine ordered rational pairs without outward grid materialization."""
    raw = _ordered(values, "contact_conductance_bounds", limit=10)
    if len(raw) != 9:
        raise ValueError("contact_conductance_bounds requires nine marked pairs")
    bounds = tuple(
        _prepared_coordinate_vector(pair, f"contact_conductance_bounds[{j}]", 2)
        for j, pair in enumerate(raw)
    )
    if any(lower < 0 or lower > upper for lower, upper in bounds):
        raise ValueError("contact pairs must satisfy 0 <= lower <= upper")
    return bounds


def _scaled_response_bounds(coefficient, lower, upper):
    """Keep rational endpoint products exact, including unresolved signs."""
    products = tuple(
        value * endpoint
        for value in (coefficient.lo, coefficient.hi)
        for endpoint in (lower, upper)
    )
    return min(products), max(products)


@dataclass(frozen=True)
class _SineMixedContactIdentification:
    """Conditional outer record sets, identity and separate identification flags.

    ``lower_endpoint_reference`` rebuilds sources and the explicitly uniform
    identity theorem. Inclusion of every interval in that theorem's box is a
    separate obligation: no interpolation of corner verdicts is performed.
    Its paired early response and equality of actual work are not transferred
    to independently chosen nuisance values under the competing hypotheses.

    Response intervals are necessary enclosures, not realizable scalar ranges.
    ``recorded_gap`` has the fixed T-minus-D orientation; ``reverse_recorded_gap``
    permits a distinct disjointness flag without changing the ordered target.
    The unit-profile obstruction asserts that a common record exists, not that
    all independent fits have the same response. No actual reading is consumed.
    Candidate scalar bounds and public construction authenticate no premises.
    """

    contact_conductance_bounds: tuple[tuple[Q, Q], ...]
    form_error_bound: Q
    phase_error_bound: Q
    registration_error_bound: Q
    form_readout_error_bound: Q
    lower_endpoint_reference: _SineMixedNormalization
    contact_box_within_protocol: bool
    source_certified: bool
    isolated_waiting_certified: bool
    geometry_certified: bool
    both_joint_identities_certified: bool
    registration_response_certified: bool
    degree_coefficient_bounds: tuple[Q, Q]
    strength_coefficient_bounds: tuple[Q, Q]
    forcing_bounds: I
    forcing_lower_margin: Q
    forcing_width: Q
    forcing_width_margin: Q
    numerical_enclosure_certified: bool
    degree_remainder_coefficient: Q
    strength_remainder_coefficient: Q
    per_hypothesis_source_error_upper_bound: Q
    degree_nominal_response_bounds: tuple[Q, Q] | None
    strength_nominal_response_bounds: tuple[Q, Q] | None
    degree_actual_response_bounds: tuple[Q, Q] | None
    strength_actual_response_bounds: tuple[Q, Q] | None
    degree_recorded_response_bounds: tuple[Q, Q] | None
    strength_recorded_response_bounds: tuple[Q, Q] | None
    recorded_gap: Q | None
    reverse_recorded_gap: Q | None
    separation_margin: Q | None
    disjoint_response_sets_certified: bool
    target_separation_certified: bool
    contains_unit_profile: bool
    unit_profile_obstruction_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    contact_conductance_range: tuple[Q, Q] = _CONTACT_RANGE
    source_allowance_limit: Q = _SOURCE_LIMIT
    registration_allowance_limit: Q = _REGISTRATION_LIMIT
    form_readout_allowance_limit: Q = _FORM_READOUT_LIMIT
    waiting_radius: Q = _ETA
    relative_phase_gap: Q = _DELTA
    observation_time: Q = _EARLY_TIME
    separation_target: Q = _EARLY_TARGET
    forcing_lower_requirement: Q = _FORCING_LOWER
    forcing_width_limit: Q = _FORCING_WIDTH_LIMIT
    scope: tuple[str, ...] = (
        "same_declared_contact_information_with_independent_fits_under_each_law",
        "independent_actual_source_residuals_and_waiting_times_not_nominal_resets",
        "uniform_identity_theorem_requires_entire_box_inclusion_not_corner_sampling",
        "fixed_gamma_clock_capacities_origins_and_one_absolute_receiver_statistic",
        "monotone_coefficient_endpoints_with_separate_live_phase_and_source_errors",
        "exact_rational_endpoints_after_tight_shared_trigonometric_enclosure",
        "per_hypothesis_actual_work_remains_supplied_not_assumed_equal_across_fits",
        "outer_intervals_are_necessary_not_every_scalar_inside_is_realizable",
        "ordered_positive_target_distinct_from_interval_disjointness_in_either_order",
        "unit_obstruction_is_existential_not_equality_for_all_independent_sources",
        "overlapping_outer_bounds_alone_leave_identification_unresolved",
        "no_record_classifier_physical_calibration_trajectory_or_source_replay",
    )


def _assess_sine_mixed_contact_identification(
    *,
    contact_conductance_bounds,
    form_error_bound,
    phase_error_bound,
    registration_error_bound,
    form_readout_error_bound,
):
    """Rebuild conditional single-record sets from independently supplied bounds.

    All raw pairs and allowances are admitted before sources, geometry or
    interval work. Nonnegative boxes outside the proved positive domain give
    unavailable claims and no response enclosures. No cached report or measured
    response is accepted. Readout allowances concern separate observables;
    registration error never widens the early form-record intervals.
    """
    bounds = _contact_conductance_bounds(contact_conductance_bounds)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    registration_noise = _exact(registration_error_bound, "registration_error_bound")
    noise = _exact(form_readout_error_bound, "form_readout_error_bound")
    if min(rx, rt, registration_noise, noise) < 0:
        raise ValueError("source and readout error bounds must be nonnegative")

    lower_weights = tuple(pair[0] for pair in bounds)
    in_box = all(
        _CONTACT_RANGE[0] <= lower <= upper <= _CONTACT_RANGE[1]
        for lower, upper in bounds
    )
    contains_unit = all(lower <= 1 <= upper for lower, upper in bounds)
    source_limits = max(rx, rt) <= _SOURCE_LIMIT
    form_limit = noise <= _FORM_READOUT_LIMIT
    registration_limit = registration_noise <= _REGISTRATION_LIMIT
    reference = _assess_sine_mixed_normalization(
        contact_conductances=lower_weights,
        form_error_bound=rx,
        phase_error_bound=rt,
        registration_error_bound=registration_noise,
        form_readout_error_bound=noise,
    )
    geometry = _derive_mixed_constitutive_pair()
    geometry_ok = (
        reference.contact_conductances == lower_weights
        and reference.bare_degree_reference.precontact_reference.geometry == geometry
        and geometry.support_certified
        and geometry.phase_alignment_certified
        and _GAMMA_LOWER <= geometry.gamma_bounds.lo
        and geometry.gamma_bounds.hi <= _GAMMA_UPPER
    )
    source_ok = source_limits and reference.source_certified
    waiting_ok = (
        source_ok
        and reference.isolated_waiting_certified
        and reference.waiting_radius == _ETA
    )
    identity = (
        in_box
        and geometry_ok
        and waiting_ok
        and reference.both_joint_identities_certified
    )
    registration_ok = (
        identity and registration_limit and reference.registration_response_certified
    )

    # Monotonicity of w/(2+w) is exact: f(v)-f(u)=2(v-u)/((2+u)(2+v)).
    # These endpoint reductions cover the entire box, not just sampled corners.
    degree = tuple(sum((pair[k] for pair in bounds), Q(0)) / 27 for k in (0, 1))
    strength = tuple(
        sum((pair[k] / (2 + pair[k]) for pair in bounds), Q(0)) / 9 for k in (0, 1)
    )
    forcing = geometry.gamma_bounds * sin(I(_DELTA)) / 3
    lower_margin = forcing.lo - _FORCING_LOWER
    width = forcing.hi - forcing.lo
    width_margin = _FORCING_WIDTH_LIMIT - width
    feedback = Q(13, 9) * _GAMMA_UPPER**2
    beta_degree = Q(65, 216) + feedback
    beta_strength = Q(10, 39) + feedback
    source_error = 2 * _ETA
    degree_nominal = strength_nominal = None
    degree_actual = strength_actual = None
    degree_recorded = strength_recorded = None
    gap = reverse_gap = margin = None
    positive_coefficients = False
    if in_box:
        h = _EARLY_TIME
        coefficient_rows = tuple(
            (h - h**2 * r[1] - h**3 * beta, h - h**2 * r[0] + h**3 * beta)
            for r, beta in ((degree, beta_degree), (strength, beta_strength))
        )
        positive_coefficients = all(0 < lo <= hi <= h for lo, hi in coefficient_rows)
        degree_nominal, strength_nominal = tuple(
            _scaled_response_bounds(forcing, lo, hi) for lo, hi in coefficient_rows
        )
        degree_actual, strength_actual = tuple(
            (lo - source_error, hi + source_error)
            for lo, hi in (degree_nominal, strength_nominal)
        )
        degree_recorded, strength_recorded = tuple(
            (lo - noise, hi + noise) for lo, hi in (degree_actual, strength_actual)
        )
        gap = strength_recorded[0] - degree_recorded[1]
        reverse_gap = degree_recorded[0] - strength_recorded[1]
        margin = gap - _EARLY_TARGET
    arithmetic_ok = (
        in_box and lower_margin >= 0 and width_margin >= 0 and positive_coefficients
    )
    record_premises = identity and form_limit and arithmetic_ok
    disjoint = record_premises and (gap > 0 or reverse_gap > 0)
    target = record_premises and margin > 0
    # One admitted common unit-weight member suffices for a true intersection.
    # Unlike the sufficient envelopes, this argument needs no trig-width gate.
    unit_obstruction = identity and form_limit and contains_unit
    status = (
        "certified_ordered_target_separation"
        if target
        else (
            "certified_disjoint_without_ordered_target"
            if disjoint
            else (
                "certified_unit_profile_obstruction"
                if unit_obstruction
                else (
                    "unresolved_response_envelopes"
                    if record_premises
                    else "unavailable"
                )
            )
        )
    )
    reasons = tuple(
        label
        for passed, label in (
            (in_box, "contact_intervals_outside_proved_positive_box"),
            (source_limits, "source_allowance_exceeds_fixed_protocol"),
            (source_ok, "original_acquired_sources_not_certified"),
            (waiting_ok, "actual_isolated_waiting_not_certified"),
            (geometry_ok, "marked_geometry_or_reference_association_not_certified"),
            (identity, "whole_box_identity_not_certified"),
            (registration_limit, "registration_readout_allowance_exceeds_protocol"),
            (registration_ok, "inherited_registration_readout_not_certified"),
            (form_limit, "form_readout_allowance_exceeds_protocol"),
            (arithmetic_ok, "fixed_numerical_enclosure_budget_not_certified"),
            (
                target or unit_obstruction,
                "fixed_ordered_record_separation_target_not_certified",
            ),
            (
                disjoint or unit_obstruction or not record_premises,
                "outer_record_envelopes_do_not_resolve_identification",
            ),
        )
        if not passed
    )
    return _SineMixedContactIdentification(
        contact_conductance_bounds=bounds,
        form_error_bound=rx,
        phase_error_bound=rt,
        registration_error_bound=registration_noise,
        form_readout_error_bound=noise,
        lower_endpoint_reference=reference,
        contact_box_within_protocol=in_box,
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        geometry_certified=geometry_ok,
        both_joint_identities_certified=identity,
        registration_response_certified=registration_ok,
        degree_coefficient_bounds=degree,
        strength_coefficient_bounds=strength,
        forcing_bounds=forcing,
        forcing_lower_margin=lower_margin,
        forcing_width=width,
        forcing_width_margin=width_margin,
        numerical_enclosure_certified=arithmetic_ok,
        degree_remainder_coefficient=beta_degree,
        strength_remainder_coefficient=beta_strength,
        per_hypothesis_source_error_upper_bound=source_error,
        degree_nominal_response_bounds=degree_nominal,
        strength_nominal_response_bounds=strength_nominal,
        degree_actual_response_bounds=degree_actual,
        strength_actual_response_bounds=strength_actual,
        degree_recorded_response_bounds=degree_recorded,
        strength_recorded_response_bounds=strength_recorded,
        recorded_gap=gap,
        reverse_recorded_gap=reverse_gap,
        separation_margin=margin,
        disjoint_response_sets_certified=disjoint,
        target_separation_certified=target,
        contains_unit_profile=contains_unit,
        unit_profile_obstruction_certified=unit_obstruction,
        status=status,
        unresolved_conditions=reasons,
    )
