"""Necessary outer-winding inference from fifteen primitive collective readings.

ORGANIZED_OUTER_OBSERVATION.md owns the fixed state class, ten matched probes
and complete finite-time proof. The inverse never consumes a phase, source
report or assigned sector. Compatibility intervals do not prove that a raw
record is realizable by the declared full-state model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import (
    INTERVAL_BITS,
    I,
    arg,
    cos,
    pi_interval,
    sin,
)
from ._organized_outer_winding import _assess_organized_outer_winding
from ._repeated_organized_composition import _vector
from ._sine_preparation import _prepared_degree_four_probe_remainder

_FORM_BOUND = Q(1, 100)
_FINE_PHASE_BOUND = Q(1, 10**6)
_OUTER_ANGLE_BOUND = Q(7, 5)
_EPSILON = Q(1, 10)
_DELAY = Q(1, 100000)
_RAW_ERROR = Q(1, 10**15)
_WORK_BUDGET = Q(2)
_ANGLE_ARITHMETIC_ALLOWANCE = Q(1, 10**20)
_ORIGINAL_ALLOWANCE = Q(1, 10**10)


def _apply_outer_observation_probe(*, form, phase_offsets, marked_child, sign):
    """Apply one supplied child-uniform jump to admitted complete rows.

    This algebraic intervention selects no occurrence and certifies no state
    class. Both signed branches retain all fine coordinates and the global
    lifted-phase mean, with forms and every internal phase gap unchanged.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    if type(marked_child) is not int or marked_child not in range(5):
        raise ValueError("marked_child must be an integer in 0..4")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("sign must be the integer -1 or +1")
    return values, tuple(
        offset + sign * _EPSILON * (int(node // 9 == marked_child) - Q(1, 5))
        for node, offset in enumerate(offsets)
    )


@dataclass(frozen=True)
class _OrganizedOuterObservationProtocol:
    gamma_bounds: I
    cosine_inverse_factor_bounds: I
    sine_inverse_factor_bounds: I
    finite_remainder_upper_bound: Q
    odd_total_error_upper_bound: Q
    even_total_error_upper_bound: Q
    cosine_prior_lower_bound: Q
    cosine_width_upper_bound: Q
    sine_width_upper_bound: Q
    quadrature_interval_rounding_allowances: tuple[Q, Q]
    child_origin_angle_width_upper_bound: Q
    cycle_sum_width_upper_bound: Q
    winding_resolution_margin: Q
    continuous_edge_drift_upper_bound: Q
    internal_positive_gap_margin: Q
    internal_acute_gap_margin: Q
    outer_acute_gap_margin: Q
    signed_event_work_bounds: I
    absolute_event_work_upper_bound: Q
    work_margin: Q
    finite_comparison_domain_margin: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    finite_identity_certified: bool
    work_certified: bool
    uniform_identification_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    form_upper_bound: Q = _FORM_BOUND
    fine_phase_upper_bound: Q = _FINE_PHASE_BOUND
    child_origin_gap_upper_bound: Q = _OUTER_ANGLE_BOUND
    probe_amplitude: Q = _EPSILON
    elapsed_probe_scaled_time: Q = _DELAY
    elapsed_probe_original_time: Q = _DELAY * Q(1024, 1023)
    raw_readout_error_upper_bound: Q = _RAW_ERROR
    odd_readout_error_upper_bound: Q = 2 * _RAW_ERROR
    even_readout_error_upper_bound: Q = 4 * _RAW_ERROR
    probe_work_budget: Q = _WORK_BUDGET
    angle_arithmetic_allowance: Q = _ANGLE_ARITHMETIC_ALLOWANCE
    protocol_revision: str = "52126ac0"
    marked_child_order: tuple[int, ...] = tuple(range(5))
    receiver_child_order: tuple[int, ...] = (1, 2, 3, 4, 0)
    permitted_windings: tuple[int, ...] = (-1, 0, 1)
    scope: tuple[str, ...] = (
        "same_unknown_actual_source_for_ten_probes_and_one_baseline_branch",
        "complete_degree_four_quarter_rate_smooth_sine_law_with_held_capacity",
        "all_fine_forms_and_phases_retained_in_finite_nonlinear_remainder",
        "fifteen_arbitrary_bounded_raw_errors_with_exact_shared_baseline_cancellation",
        "unknown_origins_and_all_nine_fine_cycles_retained",
        "necessary_sector_compatibility_does_not_prove_record_realizability",
        "supplied_signed_probe_work_separate_for_every_branch",
        "identity_retained_only_through_declared_finite_observation_horizon",
        "no_trajectory_source_reset_operator_event_selection_or_physical_calibration",
    )


def _assess_organized_outer_observation_protocol():
    """Rebuild the fixed uniform information, work and finite-identity bounds."""
    pi = pi_interval()
    gamma = Q(1, 1023) / pi
    sine, one_minus_cosine = sin(I(_EPSILON)), 1 - cos(I(_EPSILON))
    remainder = _prepared_degree_four_probe_remainder(
        gamma_upper=gamma.hi,
        initial_form_upper_bound=_FORM_BOUND,
        phase_jump_upper_bound=_EPSILON,
        elapsed_time=_DELAY,
    )
    odd_error, even_error = (
        2 * remainder + 2 * _RAW_ERROR,
        2 * remainder + 4 * _RAW_ERROR,
    )
    kc = 2 / (gamma * _DELAY * sine)
    ks = 2 / (gamma * _DELAY * one_minus_cosine)
    fc = gamma.hi * _DELAY * sine.hi / 2
    fs = gamma.hi * _DELAY * one_minus_cosine.hi / 2
    grid = Q(1, 1 << INTERVAL_BITS)
    # Raw-band endpoint rounding is amplified by the inverse factor; the
    # final interval product adds at most one grid unit at either endpoint.
    rounding_c, rounding_s = (
        (2 * factor.hi + factor.width + 2) * grid for factor in (kc, ks)
    )
    wc = 2 * kc.hi * odd_error + kc.width * (fc + 2 * odd_error) + rounding_c
    ws = 2 * ks.hi * even_error + ks.width * (fs + 2 * even_error) + rounding_s
    c0 = cos(I(_OUTER_ANGLE_BOUND + 2 * _FINE_PHASE_BOUND)).lo
    if c0 <= 0:
        raise ArithmeticError("fixed observation class lost its positive cosine chart")
    angle_width = (
        ws / c0 + wc / c0**2 + 4 * _FINE_PHASE_BOUND + _ANGLE_ARITHMETIC_ALLOWANCE
    )
    cycle_width = 5 * angle_width
    resolution = 2 * pi.lo - cycle_width
    drift = 4 * gamma.hi * (_FORM_BOUND + gamma.hi * _DELAY) * _DELAY
    internal_positive = 2 * pi.lo / 9 - 2 * _FINE_PHASE_BOUND - drift
    internal_acute = pi.lo / 2 - 2 * pi.hi / 9 - 2 * _FINE_PHASE_BOUND - drift
    outer_acute = (
        pi.lo / 2 - _OUTER_ANGLE_BOUND - 2 * _FINE_PHASE_BOUND - _EPSILON - drift
    )
    work = 18 * _EPSILON
    finite_domain = Q(1, 2) - 2 * gamma.hi**2 * _DELAY**2
    identity = (
        ("internal_positive_gap", internal_positive),
        ("internal_acute_gap", internal_acute),
        ("outer_acute_gap", outer_acute),
    )
    margins = identity + (
        ("absolute_work", _WORK_BUDGET - work),
        ("finite_nonlinear_comparison_domain", finite_domain),
        ("positive_cosine_chart", c0),
        ("complete_cycle_winding_resolution", resolution),
    )
    identity_ok = all(value > 0 for _, value in identity)
    work_ok = work < _WORK_BUDGET
    information_ok = finite_domain > 0 and resolution > 0
    admitted = identity_ok and work_ok and information_ok
    return _OrganizedOuterObservationProtocol(
        gamma_bounds=gamma,
        cosine_inverse_factor_bounds=kc,
        sine_inverse_factor_bounds=ks,
        finite_remainder_upper_bound=remainder,
        odd_total_error_upper_bound=odd_error,
        even_total_error_upper_bound=even_error,
        cosine_prior_lower_bound=c0,
        cosine_width_upper_bound=wc,
        sine_width_upper_bound=ws,
        quadrature_interval_rounding_allowances=(rounding_c, rounding_s),
        child_origin_angle_width_upper_bound=angle_width,
        cycle_sum_width_upper_bound=cycle_width,
        winding_resolution_margin=resolution,
        continuous_edge_drift_upper_bound=drift,
        internal_positive_gap_margin=internal_positive,
        internal_acute_gap_margin=internal_acute,
        outer_acute_gap_margin=outer_acute,
        signed_event_work_bounds=I(-work, work),
        absolute_event_work_upper_bound=work,
        work_margin=_WORK_BUDGET - work,
        finite_comparison_domain_margin=finite_domain,
        sufficient_margins=margins,
        finite_identity_certified=identity_ok and finite_domain > 0,
        work_certified=work_ok,
        uniform_identification_certified=information_ok,
        admitted=admitted,
        status="certified_outer_observation_protocol" if admitted else "unavailable",
        unresolved_conditions=tuple(name for name, value in margins if value <= 0),
    )


def _intersection(interval, lower, upper):
    lo, hi = max(interval.lo, lower), min(interval.hi, upper)
    return I(lo, hi) if lo <= hi else None


@dataclass(frozen=True)
class _OrganizedOuterWindingInference:
    unprobed_means: tuple[Q, ...]
    positive_probe_means: tuple[Q, ...]
    negative_probe_means: tuple[Q, ...]
    protocol: _OrganizedOuterObservationProtocol
    odd_reading_differences: tuple[Q, ...]
    even_reading_combinations: tuple[Q, ...]
    raw_readings_compatible: tuple[bool, ...]
    cosine_bounds: tuple[I | None, ...]
    sine_bounds: tuple[I | None, ...]
    resultant_squared_modulus_bounds: tuple[I | None, ...]
    resultant_compatible: tuple[bool, ...]
    resultant_angle_bounds: tuple[I | None, ...]
    child_origin_angle_bounds: tuple[I | None, ...]
    cycle_sum_bounds: I | None
    candidate_windings: tuple[int, ...]
    identified_winding: int | None
    status: str
    unresolved_conditions: tuple[str, ...]
    candidate_meaning: str = "necessary_compatibility_not_joint_realizability"
    identification_scope: str = "conditional_on_the_declared_same_source_state_and_law"


def _infer_organized_outer_winding(
    *, unprobed_means, positive_probe_means, negative_probe_means
):
    """Infer necessary sectors using primitive readings and the fixed protocol.

    Every raw scalar is admitted before interval construction. Exact primitive
    subtraction preserves common-baseline cancellation even for large rational
    representations. A singleton identifies the actual sector only assuming
    the declared matched-source law and state premises; it is not their audit.
    """
    baseline = _vector(unprobed_means, "unprobed_means", 5)
    positive = _vector(positive_probe_means, "positive_probe_means", 5)
    negative = _vector(negative_probe_means, "negative_probe_means", 5)
    protocol = _assess_organized_outer_observation_protocol()
    odd = tuple(p - m for p, m in zip(positive, negative))
    even = tuple(p + m - 2 * b for p, m, b in zip(positive, negative, baseline))
    raw_bound = _FORM_BOUND + protocol.gamma_bounds.hi * _DELAY + _RAW_ERROR
    raw_compatible = tuple(abs(v) <= raw_bound for v in baseline + positive + negative)
    resultant_floor = cos(I(2 * _FINE_PHASE_BOUND)).lo ** 2
    cb, sb, moduli, radial_compatible, arguments, angles = [], [], [], [], [], []
    for dc, ds in zip(odd, even):
        bc, bs = (
            protocol.odd_total_error_upper_bound,
            protocol.even_total_error_upper_bound,
        )
        c = _intersection(
            protocol.cosine_inverse_factor_bounds * I(dc - bc, dc + bc),
            protocol.cosine_prior_lower_bound,
            Q(1),
        )
        s = _intersection(
            protocol.sine_inverse_factor_bounds * I(ds - bs, ds + bs), Q(-1), Q(1)
        )
        modulus = c**2 + s**2 if c is not None and s is not None else None
        radial_ok = (
            modulus is not None and modulus.lo <= 1 and modulus.hi >= resultant_floor
        )
        a = arg(c, s) if c is not None and s is not None else None
        angle = (
            _intersection(
                a + I(-2 * _FINE_PHASE_BOUND, 2 * _FINE_PHASE_BOUND),
                -_OUTER_ANGLE_BOUND,
                _OUTER_ANGLE_BOUND,
            )
            if a is not None
            else None
        )
        cb.append(c)
        sb.append(s)
        moduli.append(modulus)
        radial_compatible.append(radial_ok)
        arguments.append(a)
        angles.append(angle)
    total = sum(angles, I(0)) if all(a is not None for a in angles) else None
    pi = pi_interval()
    candidates = tuple(
        winding
        for winding in protocol.permitted_windings
        if total is not None
        and all(raw_compatible)
        and all(radial_compatible)
        and total.lo <= (2 * winding * pi).hi
        and (2 * winding * pi).lo <= total.hi
    )
    unique = len(candidates) == 1 and protocol.admitted
    if not candidates:
        status = "incompatible_records_or_premises"
    elif not protocol.admitted:
        status = "unavailable_protocol_certificate"
    elif unique:
        status = "conditionally_identified"
    else:
        status = "ambiguous"
    contradictions = tuple(
        f"raw_mean_outside_form_bound[{i}]"
        for i, ok in enumerate(raw_compatible)
        if not ok
    ) + tuple(
        f"interface_resultant_outside_fine_phase_annulus[{i}]"
        for i, ok in enumerate(radial_compatible)
        if not ok
    )
    conditions = (
        protocol.unresolved_conditions
        + contradictions
        + (("no_compatible_sector",) if not candidates else ())
        + (("multiple_compatible_sectors",) if len(candidates) > 1 else ())
    )
    return _OrganizedOuterWindingInference(
        unprobed_means=baseline,
        positive_probe_means=positive,
        negative_probe_means=negative,
        protocol=protocol,
        odd_reading_differences=odd,
        even_reading_combinations=even,
        raw_readings_compatible=raw_compatible,
        cosine_bounds=tuple(cb),
        sine_bounds=tuple(sb),
        resultant_squared_modulus_bounds=tuple(moduli),
        resultant_compatible=tuple(radial_compatible),
        resultant_angle_bounds=tuple(arguments),
        child_origin_angle_bounds=tuple(angles),
        cycle_sum_bounds=total,
        candidate_windings=candidates,
        identified_winding=candidates[0] if unique else None,
        status=status,
        unresolved_conditions=conditions,
    )


@dataclass(frozen=True)
class _OuterObservationSourceInclusion:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    outer_one_form_upper_bound: Q
    outer_zero_form_upper_bound: Q
    reprojected_fine_phase_upper_bound: Q
    outer_one_origin_gap_upper_bound: Q
    outer_zero_origin_gap_upper_bound: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    scope: str = (
        "fresh_actual_acquired_families_included_not_all_class_members_acquired"
    )


def _assess_outer_observation_source_inclusion(
    *, form_error_bounds, phase_error_bounds
):
    """Rebuild prior acquisitions independently of the public-only inverse.

    Actual reached fine errors can have nonzero child means. Absorb those
    means into the unknown origins; the remaining childwise-zero-mean fine
    phase error is bounded by twice the full source phase-error radius.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 5)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 5)
    if min(*rx, *rt) < 0 or max(*rx, *rt) > _ORIGINAL_ALLOWANCE:
        raise ValueError("original errors must belong to frozen [0, 1e-10]")
    acquisition = _assess_organized_outer_winding(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    r, z = (
        acquisition.full_form_error_upper_bound,
        acquisition.full_phase_error_upper_bound,
    )
    one_x = (
        max(box.abs_max for box in acquisition.primitive_entry.endpoint_form_bounds) + r
    )
    one_angle = (
        max(box.abs_max for box in acquisition.endpoint_oriented_outer_phase_gap_bounds)
        + 2 * z
    )
    zero_angle = 2 * z
    margins = (
        ("outer_one_form", _FORM_BOUND - one_x),
        ("outer_zero_form", _FORM_BOUND - r),
        ("reprojected_fine_phase", _FINE_PHASE_BOUND - 2 * z),
        ("outer_one_unknown_origin_gap", _OUTER_ANGLE_BOUND - one_angle),
        ("outer_zero_unknown_origin_gap", _OUTER_ANGLE_BOUND - zero_angle),
    )
    source_ok = acquisition.admitted and acquisition.control.capture_certified
    admitted = source_ok and all(value > 0 for _, value in margins)
    conditions = (
        () if source_ok else ("actual_acquired_source_not_certified",)
    ) + tuple(name for name, value in margins if value <= 0)
    return _OuterObservationSourceInclusion(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        outer_one_form_upper_bound=one_x,
        outer_zero_form_upper_bound=r,
        reprojected_fine_phase_upper_bound=2 * z,
        outer_one_origin_gap_upper_bound=one_angle,
        outer_zero_origin_gap_upper_bound=zero_angle,
        sufficient_margins=margins,
        admitted=admitted,
        status="certified_actual_source_inclusion" if admitted else "unavailable",
        unresolved_conditions=conditions,
    )
