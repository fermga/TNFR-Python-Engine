"""Necessary initial winding inference from one finite observation history.

ORGANIZED_SEQUENTIAL_OBSERVATION.md owns the frozen 15-event word and the
complete nonlinear bounds. The inverse consumes only 25 primitive collective
form readings. Events are supplied; neither their occurrence nor physical
calibration follows from this conditional observation theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_outer_observation import (
    _ANGLE_ARITHMETIC_ALLOWANCE,
    _DELAY,
    _EPSILON,
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
    _RAW_ERROR,
    _infer_outer_quadrature_cycle,
    _outer_observation_inverse_geometry,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered

_BLOCK_DURATION = 4 * _DELAY
_TOTAL_DURATION = 20 * _DELAY
_EVENT_MULTIPLIERS = (1, -2, 1)
_EVENT_WORK_BUDGET = Q(4)
_TOTAL_WORK_BUDGET = Q(40)


def _apply_sequential_observation_event(
    *, form, phase_offsets, marked_child, event_index
):
    """Apply one declared jump, retaining all actual evolved coordinates.

    Event indices 0, 1 and 2 respectively apply +epsilon, -2*epsilon and
    +epsilon times the centered marked-child pattern. The final command
    restores the offset only: it does not restore a previous evolved state.
    This detached map supplies no timing, occurrence or state-class proof.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    if type(marked_child) is not int or marked_child not in range(5):
        raise ValueError("marked_child must be an integer in 0..4")
    if type(event_index) is not int or event_index not in range(3):
        raise ValueError("event_index must be an integer in 0..2")
    jump = _EVENT_MULTIPLIERS[event_index] * _EPSILON
    return values, tuple(
        offset + jump * (int(node // 9 == marked_child) - Q(1, 5))
        for node, offset in enumerate(offsets)
    )


@dataclass(frozen=True)
class _OrganizedSequentialObservationProtocol:
    gamma_bounds: I
    cosine_inverse_factor_bounds: I
    sine_inverse_factor_bounds: I
    observation_form_upper_bound: Q
    baseline_third_derivative_upper_bound: Q
    odd_finite_remainder_upper_bound: Q
    even_finite_remainder_upper_bound: Q
    odd_initial_transport_error_upper_bound: Q
    even_initial_transport_error_upper_bound: Q
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
    initial_edge_drift_upper_bound: Q
    internal_positive_gap_margin: Q
    internal_acute_gap_margin: Q
    outer_acute_gap_margin: Q
    per_block_signed_event_work_bounds: tuple[I, ...]
    per_block_absolute_event_work_upper_bounds: tuple[Q, ...]
    total_absolute_event_work_upper_bound: Q
    event_work_margin: Q
    total_work_margin: Q
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
    elapsed_window_scaled_time: Q = _DELAY
    elapsed_window_original_time: Q = _DELAY * Q(1024, 1023)
    elapsed_block_scaled_time: Q = _BLOCK_DURATION
    total_elapsed_scaled_time: Q = _TOTAL_DURATION
    total_elapsed_original_time: Q = _TOTAL_DURATION * Q(1024, 1023)
    raw_readout_error_upper_bound: Q = _RAW_ERROR
    odd_readout_error_upper_bound: Q = Q(16, 3) * _RAW_ERROR
    even_readout_error_upper_bound: Q = 6 * _RAW_ERROR
    event_work_budget: Q = _EVENT_WORK_BUDGET
    total_work_budget: Q = _TOTAL_WORK_BUDGET
    angle_arithmetic_allowance: Q = _ANGLE_ARITHMETIC_ALLOWANCE
    protocol_revision: str = "8c0d4cca"
    marked_child_order: tuple[int, ...] = tuple(range(5))
    receiver_child_order: tuple[int, ...] = (1, 2, 3, 4, 0)
    event_multipliers: tuple[int, ...] = _EVENT_MULTIPLIERS
    window_command_multipliers: tuple[int, ...] = (0, 1, -1, 0)
    permitted_windings: tuple[int, ...] = (-1, 0, 1)
    scope: tuple[str, ...] = (
        "one_actual_history_without_source_copy_reset_or_measured_reference_branch",
        "complete_degree_four_quarter_rate_smooth_sine_law_with_held_capacity",
        "all_fine_forms_phases_background_motion_and_probe_backaction_retained",
        "twenty_five_primitive_readings_with_arbitrary_bounded_correlated_errors",
        "initial_quadratures_recovered_with_full_prior_history_transport_bound",
        "necessary_sector_compatibility_does_not_prove_record_realizability",
        "fifteen_supplied_signed_events_with_absolute_work_counted_separately",
        "identity_retained_only_through_declared_finite_observation_horizon",
        "no_trajectory_operator_descent_event_selection_or_physical_calibration",
    )


def _assess_organized_sequential_observation_protocol():
    """Rebuild the frozen full-history information, work and identity bounds."""
    pi = pi_interval()
    gamma = Q(1, 1023) / pi
    g, h, epsilon = gamma.hi, _DELAY, _EPSILON
    form_bound = _FORM_BOUND + g * _TOTAL_DURATION
    drift = 4 * g * form_bound * _TOTAL_DURATION
    m3 = (
        8 * form_bound
        + 4 * g
        + 16 * g**2 * form_bound
        + 4 * g**3
        + 16 * g**3 * form_bound**2
    )
    odd_finite = (
        Q(64, 3) * g * epsilon * (1 + g * form_bound) * h**2
        + Q(464, 9) * g**3 * epsilon * h**3
        + Q(2, 3) * m3 * h**3
    )
    even_finite = (
        32 * g * epsilon * (1 + g * form_bound) * h**2
        + Q(256, 3) * g**3 * epsilon * h**3
        + 2 * m3 * h**3
    )
    fc = g * h * sin(I(epsilon)).hi / 2
    fs = g * h * (1 - cos(I(epsilon))).hi / 2
    odd_transport, even_transport = fc * drift, fs * drift
    odd_error = odd_finite + Q(16, 3) * _RAW_ERROR + odd_transport
    even_error = even_finite + 6 * _RAW_ERROR + even_transport
    geometry = _outer_observation_inverse_geometry(
        gamma=gamma, odd_error=odd_error, even_error=even_error
    )
    internal_positive = 2 * pi.lo / 9 - 2 * _FINE_PHASE_BOUND - drift
    internal_acute = pi.lo / 2 - 2 * pi.hi / 9 - 2 * _FINE_PHASE_BOUND - drift
    outer_acute = (
        pi.lo / 2 - _OUTER_ANGLE_BOUND - 2 * _FINE_PHASE_BOUND - epsilon - drift
    )
    work_bounds = tuple(18 * abs(k) * epsilon for k in _EVENT_MULTIPLIERS)
    total_work = 5 * sum(work_bounds)
    event_margin = _EVENT_WORK_BUDGET - max(work_bounds)
    total_margin = _TOTAL_WORK_BUDGET - total_work
    finite_domain = Q(1, 2) - 2 * g**2 * _BLOCK_DURATION**2
    identity = (
        ("internal_positive_gap", internal_positive),
        ("internal_acute_gap", internal_acute),
        ("outer_acute_gap", outer_acute),
    )
    margins = identity + (
        ("absolute_event_work", event_margin),
        ("cumulative_absolute_event_work", total_margin),
        ("finite_nonlinear_comparison_domain", finite_domain),
        ("positive_cosine_chart", geometry.cosine_prior_lower_bound),
        ("complete_cycle_winding_resolution", geometry.winding_resolution_margin),
    )
    identity_ok = all(value > 0 for _, value in identity)
    work_ok = event_margin > 0 and total_margin > 0
    information_ok = finite_domain > 0 and geometry.winding_resolution_margin > 0
    admitted = identity_ok and work_ok and information_ok
    return _OrganizedSequentialObservationProtocol(
        gamma_bounds=gamma,
        **vars(geometry),
        observation_form_upper_bound=form_bound,
        baseline_third_derivative_upper_bound=m3,
        odd_finite_remainder_upper_bound=odd_finite,
        even_finite_remainder_upper_bound=even_finite,
        odd_initial_transport_error_upper_bound=odd_transport,
        even_initial_transport_error_upper_bound=even_transport,
        odd_total_error_upper_bound=odd_error,
        even_total_error_upper_bound=even_error,
        continuous_edge_drift_upper_bound=drift,
        initial_edge_drift_upper_bound=drift,
        internal_positive_gap_margin=internal_positive,
        internal_acute_gap_margin=internal_acute,
        outer_acute_gap_margin=outer_acute,
        per_block_signed_event_work_bounds=tuple(I(-v, v) for v in work_bounds),
        per_block_absolute_event_work_upper_bounds=work_bounds,
        total_absolute_event_work_upper_bound=total_work,
        event_work_margin=event_margin,
        total_work_margin=total_margin,
        finite_comparison_domain_margin=finite_domain,
        sufficient_margins=margins,
        finite_identity_certified=identity_ok and finite_domain > 0,
        work_certified=work_ok,
        uniform_identification_certified=information_ok,
        admitted=admitted,
        status=(
            "certified_sequential_observation_protocol" if admitted else "unavailable"
        ),
        unresolved_conditions=tuple(name for name, value in margins if value <= 0),
    )


@dataclass(frozen=True)
class _SequentialOuterWindingInference:
    readings: tuple[tuple[Q, ...], ...]
    protocol: _OrganizedSequentialObservationProtocol
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
    identification_scope: str = (
        "conditional_on_the_declared_single_history_state_law_and_observation"
    )


def _infer_sequential_outer_winding(*, readings):
    """Infer the initial sector from 25 primitive scalars and the fixed word.

    Readings are five ordered rows of five endpoints, one marked block per
    row. All values are admitted before protocol arithmetic. Exact shared
    endpoint combinations retain their correlated errors. A singleton is
    conditional on the complete-history premises, not a realizability proof.
    """
    rows = _ordered(readings, "readings", limit=6)
    if len(rows) != 5:
        raise ValueError("readings requires five ordered rows")
    values = tuple(_vector(row, f"readings[{i}]", 5) for i, row in enumerate(rows))
    protocol = _assess_organized_sequential_observation_protocol()
    odd = tuple(
        (row[0] - 4 * row[1] + 6 * row[2] - 4 * row[3] + row[4]) / 3 for row in values
    )
    even = tuple(row[0] - 2 * row[1] + 2 * row[3] - row[4] for row in values)
    raw_bound = protocol.observation_form_upper_bound + _RAW_ERROR
    raw_compatible = tuple(abs(value) <= raw_bound for row in values for value in row)
    inference = _infer_outer_quadrature_cycle(
        odd=odd, even=even, raw_compatible=raw_compatible, protocol=protocol
    )
    return _SequentialOuterWindingInference(
        readings=values, protocol=protocol, **vars(inference)
    )
