"""Collective observation under one carried finite-input word.

ORGANIZED_FINITE_OBSERVATION.md owns the complete-history proof. The prior
state class and algebraic quadrature inverse are shared with earlier owners;
their instantaneous event and response theorems are not consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_outer_observation import (
    _DELAY,
    _EPSILON,
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
    _infer_outer_quadrature_cycle,
    _outer_observation_inverse_geometry,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered

_PULSE_DURATION = Q(1)
_PLATEAU_LEVELS = (0, 1, -1)
_PULSE_MULTIPLIERS = (1, -2, 1)
_WORD_DURATION = 15 * (_DELAY + _PULSE_DURATION)
_INPUT_DURATION = 15 * _PULSE_DURATION
_PEAK_BUDGET = Q(5 * 10**8)
_LOSS_BUDGET = Q(2 * 10**9)
_WORK_BUDGET = Q(2 * 10**10)


@dataclass(frozen=True)
class _OrganizedFiniteObservationProtocol:
    gamma_bounds: I
    observation_form_upper_bound: Q
    phase_residual_upper_bound: Q
    plateau_second_derivative_upper_bound: Q
    plateau_transport_error_upper_bound: Q
    plateau_trapezoid_error_upper_bound: Q
    plateau_readout_error_upper_bound: Q
    plateau_total_error_upper_bound: Q
    odd_total_error_upper_bound: Q
    even_total_error_upper_bound: Q
    cosine_inverse_factor_bounds: I
    sine_inverse_factor_bounds: I
    cosine_prior_lower_bound: Q
    cosine_width_upper_bound: Q
    sine_width_upper_bound: Q
    quadrature_interval_rounding_allowances: tuple[Q, Q]
    child_origin_angle_width_upper_bound: Q
    cycle_sum_width_upper_bound: Q
    winding_resolution_margin: Q
    known_offset_peak_upper: Q
    live_form_peak_upper: Q
    full_gradient_peak_upper: Q
    input_peak_upper: Q
    peak_storage_upper: Q
    continuous_loss_upper: Q
    absolute_work_upper: Q
    continuous_loss_lower: Q
    peak_form_storage_lower: Q
    initial_storage_upper: Q
    positive_supplied_work_lower: Q
    internal_positive_gap_margin: Q
    internal_acute_gap_margin: Q
    collar_margin: Q
    outer_acute_gap_margin: Q
    raw_readout_error_upper_bound: Q
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
    plateau_duration: Q = _DELAY
    pulse_duration: Q = _PULSE_DURATION
    total_elapsed_scaled_time: Q = _WORD_DURATION
    total_elapsed_original_time: Q = Q(1024, 1023) * _WORD_DURATION
    total_input_scaled_time: Q = _INPUT_DURATION
    marked_child_order: tuple[int, ...] = tuple(range(5))
    receiver_child_order: tuple[int, ...] = (1, 2, 3, 4, 0)
    plateau_command_multipliers: tuple[int, ...] = _PLATEAU_LEVELS
    pulse_multipliers: tuple[int, ...] = _PULSE_MULTIPLIERS
    permitted_windings: tuple[int, ...] = (-1, 0, 1)
    node_order: tuple[int, ...] = tuple(range(45))
    held_capacities: tuple[Q, ...] = (Q(1),) * 45
    clock_factor: Q = Q(1, 4)
    phase_domain: str = "regular"
    peak_storage_budget: Q = _PEAK_BUDGET
    continuous_loss_budget: Q = _LOSS_BUDGET
    absolute_work_budget: Q = _WORK_BUDGET
    reading_shape: tuple[int, ...] = (5, 3, 2, 5)
    clock: str = "tau=(1023/1024)*t; all windows and pulses retain tau elapsed time"
    scope: tuple[str, ...] = (
        "one_actual_carried_five_child_history_without_reset_or_observed_phase",
        "fifteen_finite_neutral_uniform_child_form_pulses_and_fifteen_plateaus",
        "two_endpoints_of_all_five_child_means_per_plateau_150_primitive_scalars",
        "trapezoid_laplacian_correction_rebuilt_from_same_primitive_readings",
        "complete_live_sine_phase_and_form_rows_with_known_input_offset_removed",
        "same_initial_prior_and_quadrature_geometry_not_old_history_remainders",
        "arbitrary_bounded_correlated_errors_with_exact_endpoint_coefficients",
        "necessary_inverse_compatibility_does_not_authenticate_raw_records",
        "finite_identity_peak_storage_loss_and_cumulative_positive_work_retained",
        "released_at_actual_endpoint_without_form_or_phase_reset",
        "no_source_admission_or_physical_calibration_from_this_protocol_bound",
    )


def _bound_organized_finite_observation(*, raw_readout_error):
    """Bound the declared word for an independently specified reading allowance.

    The unchanged state prior is X=0.01, zeta=1e-6, A=1.4. Fixed finite
    ramps replace the old phase jumps, with explicit live-state transport.
    The shared inverse uses the same epsilon and plateau length; its old
    matched-branch or instantaneous-history remainder is never imported.
    """
    noise = _exact(raw_readout_error, "raw_readout_error")
    if noise < 0:
        raise ValueError("raw_readout_error must be nonnegative")
    pi = pi_interval()
    gamma = Q(1, 1023) / pi
    g, h, duration = gamma.hi, _DELAY, _WORD_DURATION
    form = _FORM_BOUND + g * duration
    phase = 2 * g * _FORM_BOUND + 2 * g**2 * duration
    second = 4 * form + 2 * g + 4 * g**2 * form
    transport = g * h * phase
    trapezoid = second * h**3 / 12
    reading = (2 + h / 2) * noise
    per_plateau = transport + trapezoid + reading
    odd_error, even_error = 2 * per_plateau, 4 * per_plateau
    inverse = _outer_observation_inverse_geometry(
        gamma=gamma, odd_error=odd_error, even_error=even_error
    )
    # The largest ramp is -2*epsilon. Its offset is zero at every reading.
    amplitude, pulse = 2 * _EPSILON, _PULSE_DURATION
    offset = 12 * amplitude / (5 * gamma.lo * pulse)
    gradient = 24 * amplitude / (5 * gamma.lo * pulse) + 8 * form
    forcing = 48 * amplitude / (5 * gamma.lo * pulse**2) + 6 * amplitude / (
        5 * gamma.lo * pulse
    )
    peak = 180 * (offset + form) ** 2 + 180
    loss = 45 * duration * gradient**2 / 4
    work = 45 * _INPUT_DURATION * gradient * forcing
    small_contrast = max(Q(0), _EPSILON - 4 * phase)
    large_contrast = max(Q(0), 2 * _EPSILON - 4 * phase)
    lower_loss = 90 * (2 * small_contrast**2 + large_contrast**2) / (g**2 * pulse)
    lower_peak = 12 * large_contrast**2 / (g**2 * pulse**2)
    initial_storage = 180 * _FORM_BOUND**2 + 180
    internal_positive = 2 * pi.lo / 9 - 2 * _FINE_PHASE_BOUND - 2 * phase
    internal_acute = pi.lo / 2 - 2 * pi.hi / 9 - 2 * _FINE_PHASE_BOUND - 2 * phase
    collar = pi.lo / 2 - 2 * _FINE_PHASE_BOUND - 4 * phase
    outer_acute = (
        pi.lo / 2 - _OUTER_ANGLE_BOUND - _EPSILON - 2 * _FINE_PHASE_BOUND - 2 * phase
    )
    identity = (
        ("internal_positive_gap", internal_positive),
        ("internal_acute_gap", internal_acute),
        ("circular_collar", collar),
        ("outer_acute_gap", outer_acute),
    )
    resources = (
        ("peak_storage", _PEAK_BUDGET - peak),
        ("continuous_loss", _LOSS_BUDGET - loss),
        ("cumulative_absolute_supplied_work", _WORK_BUDGET - work),
    )
    information = (
        ("positive_cosine_chart", inverse.cosine_prior_lower_bound),
        ("complete_cycle_winding_resolution", inverse.winding_resolution_margin),
    )
    margins = identity + resources + information
    identity_ok = all(value > 0 for _, value in identity)
    work_ok = all(value > 0 for _, value in resources)
    information_ok = all(value > 0 for _, value in information)
    admitted = identity_ok and work_ok and information_ok
    return _OrganizedFiniteObservationProtocol(
        gamma_bounds=gamma,
        observation_form_upper_bound=form,
        phase_residual_upper_bound=phase,
        plateau_second_derivative_upper_bound=second,
        plateau_transport_error_upper_bound=transport,
        plateau_trapezoid_error_upper_bound=trapezoid,
        plateau_readout_error_upper_bound=reading,
        plateau_total_error_upper_bound=per_plateau,
        odd_total_error_upper_bound=odd_error,
        even_total_error_upper_bound=even_error,
        **vars(inverse),
        known_offset_peak_upper=offset,
        live_form_peak_upper=offset + form,
        full_gradient_peak_upper=gradient,
        input_peak_upper=forcing,
        peak_storage_upper=peak,
        continuous_loss_upper=loss,
        absolute_work_upper=work,
        continuous_loss_lower=lower_loss,
        peak_form_storage_lower=lower_peak,
        initial_storage_upper=initial_storage,
        positive_supplied_work_lower=max(Q(0), lower_loss - initial_storage),
        internal_positive_gap_margin=internal_positive,
        internal_acute_gap_margin=internal_acute,
        collar_margin=collar,
        outer_acute_gap_margin=outer_acute,
        raw_readout_error_upper_bound=noise,
        sufficient_margins=margins,
        finite_identity_certified=identity_ok,
        work_certified=work_ok,
        uniform_identification_certified=information_ok,
        admitted=admitted,
        status="certified_finite_observation_protocol" if admitted else "unavailable",
        unresolved_conditions=tuple(name for name, value in margins if value <= 0),
    )


@dataclass(frozen=True)
class _OrganizedFiniteObservationWord:
    elapsed: Q
    stage: str
    block_index: int | None
    plateau_index: int | None
    marked_child: int | None
    receiver_child: int | None
    local_elapsed: Q
    phase_offset_bounds: tuple[I, ...]
    form_offset_bounds: tuple[I, ...]
    form_input_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    total_elapsed_scaled_time: Q = _WORD_DURATION
    scope: tuple[str, ...] = (
        "right_continuous_input_at_each_plateau_pulse_boundary",
        "terminal_input_zero_and_actual_state_released_without_reset",
        "returned_offsets_and_rates_are_not_complete_actual_state_or_rates",
        "waveform_depends_on_elapsed_clock_and_marked_schedule_only",
    )


def _evaluate_organized_finite_observation_word(*, elapsed):
    """Evaluate the fixed finite-input word with right-continuous switching.

    Every block has plateaus at command levels 0,+epsilon,-epsilon, each
    followed by a one-unit pulse of amplitude +epsilon,-2epsilon,+epsilon.
    The final input is zero. No phase or form state is read or installed.
    """
    time = _exact(elapsed, "elapsed")
    if not 0 <= time <= _WORD_DURATION:
        raise ValueError("elapsed requires 0 <= elapsed <= the complete word duration")
    zero = (I(0),) * 5
    if time == _WORD_DURATION:
        return _OrganizedFiniteObservationWord(
            elapsed=time,
            stage="terminal",
            block_index=None,
            plateau_index=None,
            marked_child=None,
            receiver_child=None,
            local_elapsed=Q(0),
            phase_offset_bounds=zero,
            form_offset_bounds=zero,
            form_input_bounds=zero,
            phase_tau_rate_bounds=zero,
            form_tau_rate_bounds=zero,
        )
    slot_duration = _DELAY + _PULSE_DURATION
    slot = time // slot_duration
    block, plateau = divmod(slot, 3)
    local = time - slot * slot_duration
    direction = tuple(Q(int(c == block)) - Q(1, 5) for c in range(5))
    base = tuple(I(_EPSILON * _PLATEAU_LEVELS[plateau] * k) for k in direction)
    if local < _DELAY:
        stage, local_elapsed = "plateau", local
        phase, form, forcing, phase_rate, form_rate = base, zero, zero, zero, zero
    else:
        # Import the admitted pulse mechanism only on this waveform path;
        # the raw inverse and protocol bounds require no acquisition owner.
        from ._organized_electrical_actuation import (
            _evaluate_organized_electrical_pulse,
        )

        stage, local_elapsed = "pulse", local - _DELAY
        pulse = _evaluate_organized_electrical_pulse(
            elapsed=local_elapsed,
            duration=_PULSE_DURATION,
            amplitude=_PULSE_MULTIPLIERS[plateau] * _EPSILON,
            marked_child=block,
        )
        phase = tuple(a + b for a, b in zip(base, pulse.phase_offset_bounds))
        form, forcing = pulse.form_offset_bounds, pulse.form_input_bounds
        phase_rate, form_rate = pulse.phase_tau_rate_bounds, pulse.form_tau_rate_bounds
    return _OrganizedFiniteObservationWord(
        elapsed=time,
        stage=stage,
        block_index=block,
        plateau_index=plateau,
        marked_child=block,
        receiver_child=(block + 1) % 5,
        local_elapsed=local_elapsed,
        phase_offset_bounds=phase,
        form_offset_bounds=form,
        form_input_bounds=forcing,
        phase_tau_rate_bounds=phase_rate,
        form_tau_rate_bounds=form_rate,
    )


@dataclass(frozen=True)
class _FiniteObservationWindingInference:
    readings: tuple
    protocol: _OrganizedFiniteObservationProtocol
    corrected_plateau_increments: tuple[tuple[Q, ...], ...]
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
        "conditional_on_declared_initial_prior_complete_finite_history_and_errors"
    )


def _fixed_rows(values, label, size):
    rows = _ordered(values, label, limit=size + 1)
    if len(rows) != size:
        raise ValueError(f"{label} requires exactly {size} ordered rows")
    return rows


def _infer_finite_observation_winding(*, readings, raw_readout_error):
    """Infer necessary initial windings from all 150 primitive mean readings.

    Each marked block has three plateaus in the fixed level order 0,+1,-1;
    each plateau has start/end vectors of all five ordered child means. The
    independent raw error budget is supplied before evaluating these records,
    not fitted to their result. No phase or source report is accepted.
    """
    blocks = _fixed_rows(readings, "readings", 5)
    normalized = []
    for c, block in enumerate(blocks):
        plateaus = _fixed_rows(block, f"readings[{c}]", 3)
        normalized.append(
            tuple(
                tuple(
                    _vector(row, f"readings[{c}][{j}][{e}]", 5)
                    for e, row in enumerate(
                        _fixed_rows(plateau, f"readings[{c}][{j}]", 2)
                    )
                )
                for j, plateau in enumerate(plateaus)
            )
        )
    values = tuple(normalized)
    protocol = _bound_organized_finite_observation(raw_readout_error=raw_readout_error)
    corrected = []
    for c, block in enumerate(values):
        receiver = (c + 1) % 5
        increments = []
        for start, end in block:
            total = tuple(a + b for a, b in zip(start, end))
            laplacian = (
                2 * total[receiver]
                - total[(receiver - 1) % 5]
                - total[(receiver + 1) % 5]
            )
            increments.append(end[receiver] - start[receiver] + _DELAY * laplacian / 8)
        corrected.append(tuple(increments))
    corrected = tuple(corrected)
    odd = tuple(row[1] - row[2] for row in corrected)
    even = tuple(row[1] + row[2] - 2 * row[0] for row in corrected)
    raw_bound = (
        protocol.observation_form_upper_bound + protocol.raw_readout_error_upper_bound
    )
    raw_compatible = tuple(
        abs(value) <= raw_bound
        for block in values
        for plateau in block
        for endpoint in plateau
        for value in endpoint
    )
    inference = _infer_outer_quadrature_cycle(
        odd=odd, even=even, raw_compatible=raw_compatible, protocol=protocol
    )
    return _FiniteObservationWindingInference(
        readings=values,
        protocol=protocol,
        corrected_plateau_increments=corrected,
        **vars(inference),
    )
