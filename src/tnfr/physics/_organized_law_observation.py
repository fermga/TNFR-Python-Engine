"""Own-law analytical admission for observation on the fixed product support.

These bounds concern each supplied constitutive law and its own carrier.
They supply no acquired source, observed record, trajectory or inherited
cross-law identification verdict. The detached field installs no runtime law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any, Callable

from .._exact_time import exact_or_represented_real
from ..mathematics._interval_taylor import Jet, cos as jet_cos, sin as jet_sin
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_computable_prediction import _ForecastStage
from ._organized_electrical_source import (
    _HiddenDeviationBounds,
    _transport_hidden_deviations,
)
from ._organized_finite_observation import (
    _EPSILON,
    _FINE_PHASE_BOUND,
    _FORM_BOUND,
    _OUTER_ANGLE_BOUND,
)
from ._organized_outer_winding import _derive_organized_outer_winding
from ._sine_class_constitutive_comparison import _current_derivative_bound, _edge_law
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_form_gradient


_G = Q(1, 3000)
_DURATION = Q(16)
_INITIAL_HIDDEN_FORM = Q(31, 10**9)
_INITIAL_HIDDEN_PHASE = Q(601, 10**10)


def _observation_deformation(value: Any) -> Q:
    coefficient = exact_or_represented_real(value, "deformation")
    if not 0 <= coefficient <= 1:
        raise ValueError("observation law requires deformation in [0, 1]")
    return coefficient


def _frozen_edge_quadratures(angle: Any, *, deformation: Any) -> tuple[I, I]:
    """Ideal local contrasts at the fixed three commands, before time errors.

    This consumes a supplied phase gap for an analytical comparison. It is
    not a raw-record inverse or a complete-history compatibility test. The
    exact sine endpoint simplifies before interval evaluation; other members
    use the canonical actual current at all three commanded gaps.
    """
    coefficient = _observation_deformation(deformation)
    point = I.coerce(angle)
    if coefficient == 0:
        return cos(point), sin(point)
    negative, central, positive = tuple(
        _edge_law(cos(point + shift), sin(point + shift), coefficient)[1][0]
        for shift in (-_EPSILON, Q(0), _EPSILON)
    )
    return (
        (positive - negative) / (2 * sin(_EPSILON)),
        (2 * central - positive - negative) / (2 * (1 - cos(_EPSILON))),
    )


@dataclass(frozen=True)
class _OrganizedLawObservation:
    deformation: Q
    current_upper: Q
    current_lipschitz: Q
    current_second_derivative_upper: Q
    hidden: _HiddenDeviationBounds
    source_residual_upper: Q
    source_margin: Q
    source_form_error_upper: Q
    source_phase_error_upper: Q
    form_upper: Q
    fine_phase_drift_upper: Q
    identity_margins: tuple[tuple[str, Q], ...]
    carrier_log_norm_upper: Q
    carrier_growth_margin: Q
    admitted: bool
    initial_form_upper: Q = _FORM_BOUND
    initial_hidden_form_upper: Q = _INITIAL_HIDDEN_FORM
    initial_hidden_phase_upper: Q = _INITIAL_HIDDEN_PHASE
    initial_hidden_coordinate_upper: Q = _FINE_PHASE_BOUND
    initial_outer_gap_upper: Q = _OUTER_ANGLE_BOUND
    command_gap_upper: Q = _EPSILON
    gamma_upper: Q = _G
    duration: Q = _DURATION


def _bound_organized_observation_law(*, deformation: Any) -> _OrganizedLawObservation:
    """Rebuild own-law regularity, hidden error, identity and numerical growth."""
    coefficient = _observation_deformation(deformation)
    current, lipschitz, second = (
        _current_derivative_bound(coefficient, order) for order in range(3)
    )
    # A common 8*L bound covers both the graph Laplacian and the current
    # Jacobian. Enlarging the phase coupling too permits the existing kernel.
    hidden = _transport_hidden_deviations(
        initial_form=_INITIAL_HIDDEN_FORM,
        initial_phase=_INITIAL_HIDDEN_PHASE,
        gamma_upper=_G,
        laplacian_upper=8 * lipschitz,
        duration=_DURATION,
    )
    residual = _G * second * hidden.phase_upper**2 / 9
    source_margin = 1 - lipschitz * (_G * _DURATION) ** 2
    if source_margin <= 0:
        raise ArithmeticError("own-law source comparison denominator is nonpositive")
    source_error = _DURATION * residual / source_margin
    form = _FORM_BOUND + _G * current * _DURATION
    drift = 2 * _G * _FORM_BOUND + 2 * _G**2 * current * _DURATION
    pi = pi_interval()
    margins = (
        ("internal_positive_gap", 2 * pi.lo / 9 - 2 * _FINE_PHASE_BOUND - 2 * drift),
        (
            "internal_acute_gap",
            pi.lo / 2 - 2 * pi.hi / 9 - 2 * _FINE_PHASE_BOUND - 2 * drift,
        ),
        (
            "outer_acute_gap",
            pi.lo / 2
            - _OUTER_ANGLE_BOUND
            - _EPSILON
            - 2 * _FINE_PHASE_BOUND
            - 2 * drift,
        ),
        ("circular_collar", pi.lo / 2 - 2 * _FINE_PHASE_BOUND - 4 * drift),
    )
    # pi>3 implies gamma<1/3069. The absolute bound on L-K suffices;
    # the sine-specific positive-semidefinite ordering is not transferred.
    log_norm = (1 + lipschitz) / (2 * 3069)
    growth_margin = _G - log_norm
    admitted = growth_margin > 0 and all(value > 0 for _, value in margins)
    return _OrganizedLawObservation(
        coefficient,
        current,
        lipschitz,
        second,
        hidden,
        residual,
        source_margin,
        source_error,
        _G * _DURATION * source_error,
        form,
        drift,
        margins,
        log_norm,
        growth_margin,
        admitted,
    )


def _admit_carrier_stage(value: Any) -> _ForecastStage | None:
    """Rebuild consumed stage primitives, retaining unused metadata verbatim."""
    if value is None:
        return None
    if not isinstance(value, _ForecastStage):
        raise TypeError("carrier command requires an explicit forecast stage")
    start = exact_or_represented_real(value.start, "stage start")
    duration = exact_or_represented_real(value.duration, "stage duration")
    if duration <= 0:
        raise ValueError("stage duration must be strictly positive")
    child = value.child
    if child is not None and (type(child) is not int or not 0 <= child < 5):
        raise ValueError("stage child must be None or an ordinary integer in 0..4")
    base, increment = value.base, value.increment
    if child is not None:
        base = exact_or_represented_real(base, "stage base")
        increment = exact_or_represented_real(increment, "stage increment")
    return _ForecastStage(start, duration, value.name, child, base, increment)


def _organized_law_carrier_field(
    *, deformation: Any, stage: _ForecastStage | None = None
) -> Callable[[Any, Any], tuple[Any, ...]]:
    """Actual inherited degree-four carrier in intervals or matching jets.

    State is (Y,Psi) when a command stage is supplied, otherwise (U,Phi).
    The explicit stage clock is absolute structural tau. No input derivative
    is omitted: it cancels in both rows of the command-removal chart.
    """
    coefficient = _observation_deformation(deformation)
    stage = _admit_carrier_stage(stage)
    geometry = _derive_organized_outer_winding()
    edges = tuple((a // 9, b // 9) for a, b, _ in geometry.outer_edges if a % 9 == 0)
    neighbors = tuple(
        tuple(b if a == node else a for a, b in edges if node in (a, b))
        for node in range(5)
    )
    gamma = geometry.gamma_bounds

    def flow(time: Any, state: Any) -> tuple[Any, ...]:
        values = _ordered(state, "carrier state", limit=11)
        if len(values) != 10:
            raise ValueError("carrier requires five form and five phase coordinates")
        jets = tuple(isinstance(value, Jet) for value in values)
        if any(jets):
            if not all(jets) or len({value.order for value in values}) != 1:
                raise ValueError(
                    "carrier jets require one common order in all coordinates"
                )
            sine, cosine = jet_sin, jet_cos
        else:
            values = tuple(I.coerce(value) for value in values)
            sine, cosine = sin, cos
        zero = values[0] * 0
        if stage is None:
            command = (zero,) * 5
        else:
            if stage.child is not None:
                if all(jets):
                    if not isinstance(time, Jet) or time.order != values[0].order:
                        raise ValueError(
                            "command clock jet must match the state jet order"
                        )
                elif isinstance(time, Jet):
                    raise ValueError(
                        "interval carrier state requires an interval command clock"
                    )
            clock = time.coeffs[0] if isinstance(time, Jet) else I.coerce(time)
            if (
                clock.lo < I(stage.start).lo
                or clock.hi > I(stage.start + stage.duration).hi
            ):
                raise ValueError("carrier clock is outside the supplied command stage")
            command = (zero,) * 5 if stage.child is None else stage.command(time)
        phase = tuple(value + offset for value, offset in zip(values[5:], command))
        gradient = _sine_form_gradient(values[:5], neighbors)
        currents = [zero for _ in range(5)]
        for tail, head in edges:
            angle = phase[head] - phase[tail]
            current = _edge_law(cosine(angle), sine(angle), coefficient)[1][0]
            currents[tail] += current
            currents[head] -= current
        return tuple(
            (-q + current * gamma) / 4 for q, current in zip(gradient, currents)
        ) + tuple(q * gamma / 4 for q in gradient)

    return flow


@dataclass(frozen=True)
class _OrganizedLawRelaxationObstruction:
    harmonics: tuple[int, ...]
    uniform_edge_sine_coefficients: tuple[Q, ...]
    block_norms: tuple[Q, ...]
    second_moment: Q
    fourth_moment: Q
    sixth_absolute_moment: Q
    forcing_upper: Q
    dipole_heat_factor: Q
    sampled_response_upper: Q
    edge_drift_upper: Q
    continuous_sample_margin: Q
    remainder_block_factors: tuple[Q, ...]
    remainder_record_factor: Q
    elementary_first_block_factor: Q
    elementary_gap: Q
    block_feedback_gap: Q
    source_errors: tuple[Q, Q]
    historical_pair_error: Q
    obstruction_established: bool
    command_gap_upper: Q = _EPSILON
    reading_error: Q = Q(1, 10**14)
    scope: str = (
        "formal_global_and_block_bootstrap_obstruction_not_nonlinear_record_overlap"
    )


def _bound_organized_law_relaxation_obstruction() -> _OrganizedLawRelaxationObstruction:
    """Bound one near-circulation witness without computing a sampled matrix.

    Initial forms and all cosine coefficients are zero; each listed sine
    coefficient is repeated on all five edges. The witness cancels the second
    and fourth even moments. It obstructs the declared norm relaxation, not
    the constitutively constrained cross-law comparison.
    """
    sine = _bound_organized_observation_law(deformation=0)
    alternative = _bound_organized_observation_law(deformation=1)
    harmonics = (1, 9, 18)
    coefficients = (Q(1), -Q(323, 19683), Q(20, 19683))
    blocks = tuple(map(abs, coefficients))
    second, fourth = (
        sum((a * n**degree for n, a in zip(harmonics, coefficients)), Q(0))
        for degree in (2, 4)
    )
    sixth = sum((a * n**6 for n, a in zip(harmonics, blocks)), Q(0))
    forcing = sixth * _EPSILON**6 / 720
    # The zero-sum C5 heat spectral gap exceeds 1/3; the active forcing is
    # F(q)*(e_previous-e_next), with dipole Euclidean norm sqrt(2)<3/2.
    heat_factor = Q(9, 2)
    sampled = heat_factor * forcing
    drift = 4 * _G * _FORM_BOUND + 2 * _G**2 * alternative.current_upper * _DURATION
    margin = 1 - 2 * _G**2 * _DURATION
    factors = tuple(
        18 * n * drift + 36 * _G**2 * _DURATION * (1 + n * drift) / margin
        for n in harmonics
    )
    record_factor = 6 * _G**2 * _DURATION / margin
    elementary = 18 * drift
    elementary_gap = elementary - sampled
    block_gap = sum((a * b for a, b in zip(factors, blocks)), Q(0)) - sampled
    errors = (sine.source_form_error_upper, alternative.source_form_error_upper)
    pair_error = 2 * Q(1, 10**14) + sum(errors)
    established = (
        sine.admitted
        and alternative.admitted
        and second == fourth == 0
        and margin > 0
        and elementary_gap > 0
        and block_gap > 0
    )
    return _OrganizedLawRelaxationObstruction(
        harmonics,
        coefficients,
        blocks,
        second,
        fourth,
        sixth,
        forcing,
        heat_factor,
        sampled,
        drift,
        margin,
        factors,
        record_factor,
        elementary,
        elementary_gap,
        block_gap,
        errors,
        pair_error,
        established,
    )


@dataclass(frozen=True)
class _OrganizedPhaseCircleCertificate:
    odd_coefficients: tuple[I, I]
    even_coefficients: tuple[I, I]
    raw_error_bounds: tuple[Q, Q]
    ninth_radius: I
    normalized_ratios: tuple[I, I] | None
    ratio_upper: Q | None
    margin_lower: Q | None
    normalized_error_bounds: tuple[Q, Q] | None
    error_square_upper: Q | None
    available: bool
    exclusion_certified: bool
    reason: str
    scope: str = "conditional_shared_phase_contrast_exclusion_not_a_record_verdict"


def _certify_organized_phase_circle(
    *, odd_coefficients: Any, even_coefficients: Any, odd_error: Any, even_error: Any
) -> _OrganizedPhaseCircleCertificate:
    """Certify a supplied finite-contrast implication, without evaluating data.

    Premises are fundamental cancellation and two contrasts with the SAME
    phase theta: r9*(-a9*cos(theta)+a18*cos(2*theta)/2) and
    r9*(-b9*sin(theta)+b18*sin(2*theta)/2). Supplied absolute errors must
    include all measurement, source, numerical and nonlinear transport terms.
    The conclusion excludes simultaneous membership in the two supplied
    zero-centered error bands.
    This function checks coefficient/error arithmetic, not those premises.
    """

    def pair(raw: Any, label: str) -> tuple[I, I]:
        values = _ordered(raw, label, limit=3)
        if len(values) != 2:
            raise ValueError(f"{label} requires ninth and eighteenth coefficients")
        admitted = tuple(
            (
                I(value.lo, value.hi)
                if isinstance(value, I)
                else I(exact_or_represented_real(value, label))
            )
            for value in values
        )
        return admitted[0], admitted[1]

    odd = pair(odd_coefficients, "odd coefficients")
    even = pair(even_coefficients, "even coefficients")
    errors = (
        exact_or_represented_real(odd_error, "odd error"),
        exact_or_represented_real(even_error, "even error"),
    )
    if any(error < 0 for error in errors):
        raise ValueError("contrast error budgets must be nonnegative")
    radius = _derive_organized_outer_winding().gamma_bounds / (4 * 2187)
    if odd[1].contains(0) or even[1].contains(0):
        return _OrganizedPhaseCircleCertificate(
            odd,
            even,
            errors,
            radius,
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            "high_harmonic_denominator_contains_zero",
        )
    ratios = (2 * odd[0] / odd[1], 2 * even[0] / even[1])
    ratio_upper = max(ratio.abs_max for ratio in ratios)
    if ratio_upper >= 1:
        return _OrganizedPhaseCircleCertificate(
            odd,
            even,
            errors,
            radius,
            ratios,
            ratio_upper,
            None,
            None,
            None,
            False,
            False,
            "no_positive_uniform_circle_margin",
        )
    margin = 1 - ratio_upper
    normalized = tuple(
        2 * error / (radius.lo * min(abs(coefficient.lo), abs(coefficient.hi)))
        for error, coefficient in zip(errors, (odd[1], even[1]))
    )
    error_square = sum(error**2 for error in normalized)
    excluded = error_square < margin**2
    return _OrganizedPhaseCircleCertificate(
        odd,
        even,
        errors,
        radius,
        ratios,
        ratio_upper,
        margin,
        (normalized[0], normalized[1]),
        error_square,
        True,
        excluded,
        "strict_conditional_exclusion" if excluded else "error_budget_not_strict",
    )
