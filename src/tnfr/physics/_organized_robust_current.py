"""Conditional current separation under a carried finite-input word.

``ORGANIZED_ROBUST_CURRENT.md`` owns the perturbed complete laws and proof.
Primitive calibration and delivery allowances are supplied, not inferred.
This module computes no trajectory, source admission or physical experiment;
failure of a sufficient bound does not exclude either physical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._organized_electrical_identification import _positive_box, _positive_interval
from ._organized_finite_observation import _WORD_DURATION, _fixed_rows
from ._organized_outer_observation import (
    _DELAY,
    _EPSILON,
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector


@dataclass(frozen=True)
class _RobustCurrentModelBound:
    multiplier: int
    forcing_residual_integral_upper: Q
    observation_form_upper: Q
    phase_residual_upper: Q
    plateau_second_derivative_upper: Q
    sample_error_upper: Q
    corrected_increment_error_upper: Q
    cosine_quadrature_error_upper: Q
    sine_quadrature_error_upper: Q
    radial_error_upper: Q
    amplitude_lower: Q
    amplitude_upper: Q
    identity_margins: tuple[tuple[str, Q], ...]
    finite_identity_certified: bool


@dataclass(frozen=True)
class _OrganizedRobustCurrentBound:
    capacitance_ratio: I
    conductance_ratio: I
    josephson_ratio: I
    clock_ratio: I
    total_tracking: Q
    plateau_tracking: Q
    raw_readout_error: Q
    aperture: Q
    gamma_bounds: I
    lambda_bounds: I
    rho_bounds: I
    eta_bounds: I
    beta_bounds: I
    eta_rho_bounds: I
    damping_mismatch_upper: Q
    phase_command_mismatch_upper: Q
    offset_laplacian_integral_upper: Q
    model_bounds: tuple[_RobustCurrentModelBound, ...]
    separation_margin: Q
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    total_elapsed_scaled_time: Q = _WORD_DURATION
    plateau_duration: Q = _DELAY
    probe_amplitude: Q = _EPSILON
    reading_shape: tuple[int, ...] = (5, 3, 2, 5)
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "two_supplied_complete_laws_with_current_multipliers_one_and_two",
        "same_initial_prior_fixed_support_clock_word_and_150_form_readings",
        "all_calibration_ratios_and_delivery_error_budgets_are_independent_primitives",
        "finite_history_identity_and_current_envelopes_rebuilt_for_each_law",
        "one_nominal_current_scale_and_one_observation_statistic_for_both_models",
        "envelopes_in_nominal_current_units_not_refitted_to_each_candidate",
        "positive_separation_is_sufficient_conditional_on_every_declared_premise",
        "no_source_acquisition_physical_admission_or_exclusion_from_failed_bounds",
    )


def _bound_organized_robust_current(
    *,
    capacitance_ratio,
    conductance_ratio,
    josephson_ratio,
    clock_ratio,
    total_tracking,
    plateau_tracking,
    raw_readout_error,
    aperture,
):
    """Bound both laws without consuming a cached protocol or response.

    C, G, A and D are strictly positive scalar/interval calibration ratios.
    ``total_tracking`` bounds the complete history's integrated form-input
    error; ``plateau_tracking`` bounds its integral on each plateau and must
    not exceed the total. ``aperture`` bounds one-sided sampling displacement
    inside the declared plateau and lies in [0,h]. Raw form error is separate.
    Each error allowance is a nonnegative exact/represented real.

    The coefficient intervals enclose lambda=G/(C*D), rho=1/(C*D),
    eta=1/(A*D), beta=A*G**2/(C**2*D). Direct positive endpoint formulas
    retain their shared primitive dependence, including eta*rho. The
    subsequent majorants deliberately use sufficient independent extrema.
    """
    C, G, A, D = (
        _positive_box(value, label)
        for label, value in (
            ("capacitance_ratio", capacitance_ratio),
            ("conductance_ratio", conductance_ratio),
            ("josephson_ratio", josephson_ratio),
            ("clock_ratio", clock_ratio),
        )
    )
    F, P, noise, delta = (
        _exact(value, label)
        for label, value in (
            ("total_tracking", total_tracking),
            ("plateau_tracking", plateau_tracking),
            ("raw_readout_error", raw_readout_error),
            ("aperture", aperture),
        )
    )
    if min(F, P, noise, delta) < 0:
        raise ValueError(
            "tracking, raw readout and aperture bounds must be nonnegative"
        )
    if P > F:
        raise ValueError("plateau_tracking must not exceed total_tracking")
    if delta > _DELAY:
        raise ValueError("aperture must not exceed the declared plateau duration")
    lam = _positive_interval(G.lo / (C.hi * D.hi), G.hi / (C.lo * D.lo), "lambda")
    rho = _positive_interval(1 / (C.hi * D.hi), 1 / (C.lo * D.lo), "rho")
    eta = _positive_interval(1 / (A.hi * D.hi), 1 / (A.lo * D.lo), "eta")
    beta = _positive_interval(
        A.lo * G.lo**2 / (C.hi**2 * D.hi),
        A.hi * G.hi**2 / (C.lo**2 * D.lo),
        "beta",
    )
    eta_rho = _positive_interval(
        1 / (A.hi * C.hi * D.hi**2),
        1 / (A.lo * C.lo * D.lo**2),
        "eta_rho",
    )
    damping_mismatch = max(abs(1 - lam.lo), abs(1 - lam.hi))
    phase_mismatch = max(abs(eta_rho.lo - 1), abs(eta_rho.hi - 1))
    pi = pi_interval()
    gamma = _positive_interval(1 / (1023 * pi.hi), 1 / (1023 * pi.lo), "gamma")
    g, h = gamma.hi, _DELAY
    offset_integral = 32 / (5 * gamma.lo)
    forcing_residual = rho.hi * damping_mismatch * offset_integral / 4 + F
    sine = sin(I(_EPSILON))
    one_minus_cosine = 1 - cos(I(_EPSILON))
    carrier_modulus = cos(I(2 * _FINE_PHASE_BOUND))
    models = []
    unresolved = []
    for mu in (1, 2):
        form = _FORM_BOUND + mu * g * beta.hi * _WORD_DURATION + forcing_residual
        phase = 2 * g * eta.hi * form / lam.lo + phase_mismatch * 4 * _EPSILON / 5
        second = (
            4 * lam.hi**2 * form
            + 2 * lam.hi * mu * g * beta.hi
            + 4 * mu * g**2 * beta.hi * eta.hi * form
        )
        sample = noise + (lam.hi * form + mu * g * beta.hi / 2) * delta + P
        corrected = (
            mu * g * beta.hi * h * phase
            + damping_mismatch * h * form
            + second * h**3 / 12
            + P * (1 + h / 2 + lam.hi * h**2 / 4)
            + (2 + h / 2) * sample
        )
        cosine_error = 4 * corrected / (gamma.lo * h * sine.lo)
        sine_error = 8 * corrected / (gamma.lo * h * one_minus_cosine.lo)
        radial_error = sqrt(I(cosine_error**2 + sine_error**2)).hi
        margins = (
            (
                "internal_positive_gap",
                2 * pi.lo / 9 - 2 * _FINE_PHASE_BOUND - 2 * phase,
            ),
            (
                "internal_acute_gap",
                pi.lo / 2 - 2 * pi.hi / 9 - 2 * _FINE_PHASE_BOUND - 2 * phase,
            ),
            (
                "outer_acute_gap",
                pi.lo / 2
                - _OUTER_ANGLE_BOUND
                - _EPSILON
                - 2 * _FINE_PHASE_BOUND
                - 2 * phase,
            ),
            ("circular_collar", pi.lo / 2 - 2 * _FINE_PHASE_BOUND - 4 * phase),
        )
        identity_ok = all(value > 0 for _, value in margins)
        unresolved.extend(f"mu_{mu}_{name}" for name, value in margins if value <= 0)
        models.append(
            _RobustCurrentModelBound(
                multiplier=mu,
                forcing_residual_integral_upper=forcing_residual,
                observation_form_upper=form,
                phase_residual_upper=phase,
                plateau_second_derivative_upper=second,
                sample_error_upper=sample,
                corrected_increment_error_upper=corrected,
                cosine_quadrature_error_upper=cosine_error,
                sine_quadrature_error_upper=sine_error,
                radial_error_upper=radial_error,
                amplitude_lower=max(
                    Q(0), mu * beta.lo * carrier_modulus.lo - radial_error
                ),
                amplitude_upper=mu * beta.hi + radial_error,
                identity_margins=margins,
                finite_identity_certified=identity_ok,
            )
        )
    gap = models[1].amplitude_lower - models[0].amplitude_upper
    if gap <= 0:
        unresolved.append("radial_current_envelopes_not_strictly_separated")
    admitted = not unresolved
    return _OrganizedRobustCurrentBound(
        capacitance_ratio=C,
        conductance_ratio=G,
        josephson_ratio=A,
        clock_ratio=D,
        total_tracking=F,
        plateau_tracking=P,
        raw_readout_error=noise,
        aperture=delta,
        gamma_bounds=gamma,
        lambda_bounds=lam,
        rho_bounds=rho,
        eta_bounds=eta,
        beta_bounds=beta,
        eta_rho_bounds=eta_rho,
        damping_mismatch_upper=damping_mismatch,
        phase_command_mismatch_upper=phase_mismatch,
        offset_laplacian_integral_upper=offset_integral,
        model_bounds=tuple(models),
        separation_margin=gap,
        admitted=admitted,
        status=(
            "conditionally_separated_current_envelopes" if admitted else "unavailable"
        ),
        unresolved_conditions=tuple(unresolved),
    )


@dataclass(frozen=True)
class _OrganizedRawCurrentQuadratures:
    readings: tuple
    nominal_current_scale: I
    gamma_bounds: I
    corrected_plateau_increments: tuple[tuple[Q, ...], ...]
    odd_reading_differences: tuple[Q, ...]
    even_reading_combinations: tuple[Q, ...]
    cosine_current_bounds: tuple[I, ...]
    sine_current_bounds: tuple[I, ...]
    radial_current_bounds: tuple[I, ...]
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "single_fixed_statistic_for_both_laws_without_model_dependent_normalization",
        "nominal_current_scale_converts_units_and_is_not_fitted_to_records",
        "outward_arithmetic_bounds_do_not_include_sensor_or_history_error",
        "raw_statistics_do_not_authenticate_source_law_clock_or_realizable_history",
    )


def _organized_raw_current_quadratures(*, readings, nominal_current_scale):
    """Compute five raw physical-current quadratures from all 150 scalars.

    The input shape is (marked child, plateau, endpoint, child)=(5,3,2,5).
    Plateaus are in command order 0,+epsilon,-epsilon. All records are
    admitted before arithmetic. No candidate multiplier is accepted or used;
    no history/error envelope or physical decision is inferred here.
    """
    normalized = []
    for c, block in enumerate(_fixed_rows(readings, "readings", 5)):
        normalized.append(
            tuple(
                tuple(
                    _vector(row, f"readings[{c}][{j}][{e}]", 5)
                    for e, row in enumerate(
                        _fixed_rows(plateau, f"readings[{c}][{j}]", 2)
                    )
                )
                for j, plateau in enumerate(_fixed_rows(block, f"readings[{c}]", 3))
            )
        )
    values = tuple(normalized)
    scale = _positive_box(nominal_current_scale, "nominal_current_scale")
    pi = pi_interval()
    gamma = _positive_interval(1 / (1023 * pi.hi), 1 / (1023 * pi.lo), "gamma")
    corrected = []
    for c, block in enumerate(values):
        receiver = (c + 1) % 5
        row = []
        for start, end in block:
            total = tuple(a + b for a, b in zip(start, end))
            lap = (
                2 * total[receiver]
                - total[(receiver - 1) % 5]
                - total[(receiver + 1) % 5]
            )
            row.append(end[receiver] - start[receiver] + _DELAY * lap / 8)
        corrected.append(tuple(row))
    corrected = tuple(corrected)
    odd = tuple(row[1] - row[2] for row in corrected)
    even = tuple(row[1] + row[2] - 2 * row[0] for row in corrected)
    cosine_factor = 2 * scale / (gamma * _DELAY * sin(I(_EPSILON)))
    sine_factor = 2 * scale / (gamma * _DELAY * (1 - cos(I(_EPSILON))))
    cosine_current = tuple(cosine_factor * value for value in odd)
    sine_current = tuple(sine_factor * value for value in even)
    radius = tuple(sqrt(a**2 + b**2) for a, b in zip(cosine_current, sine_current))
    return _OrganizedRawCurrentQuadratures(
        readings=values,
        nominal_current_scale=scale,
        gamma_bounds=gamma,
        corrected_plateau_increments=corrected,
        odd_reading_differences=odd,
        even_reading_combinations=even,
        cosine_current_bounds=cosine_current,
        sine_current_bounds=sine_current,
        radial_current_bounds=radius,
    )
