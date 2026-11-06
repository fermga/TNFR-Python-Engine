"""Necessary phase bounds for two laws sharing the fixed noisy form record.

The complete histories remain hypothetical. These analytical bounds use
their necessary common-reading condition, the existing three-level word,
and each law's rebuilt source error. No record, fit, trajectory, moment
kernel or sampled coefficient design is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import pi_interval
from ._organized_computable_prediction import _READING_ERROR
from ._organized_law_observation import _bound_organized_observation_law
from ._organized_law_phase_transport import _historical_cells


@dataclass(frozen=True)
class _OrganizedLawJointBounds:
    sample_difference_upper: Q
    plateau_duration: Q
    gamma_lower: Q
    gamma_upper: Q
    carrier_form_bounds: tuple[Q, Q]
    carrier_phase_drift_bounds: tuple[Q, Q]
    plateau_second_derivative_bounds: tuple[Q, Q]
    corrected_reading_force_error: Q
    trapezoid_force_error: Q
    initial_reference_current_error: Q
    sine_quadrature_difference_upper: Q
    alternative_sine_quadrature_error: Q
    initial_cosine_lower: Q
    initial_gap_difference_upper: Q
    same_winding_margin: Q
    initial_current_divergence_upper: Q
    fundamental_sine_radius_ratio: Q
    initial_fundamental_radius_ratio: Q
    reference_fundamental_radius_ratio: Q
    admitted: bool
    scope: str = "necessary_shared_record_phase_bounds_not_a_separation_verdict"


def _bound_organized_law_joint_phases() -> _OrganizedLawJointBounds:
    """Rebuild equal-winding and fundamental-phasor bounds from fixed premises.

    On a plateau D(P,Q)=Q-P+h*A(P+Q)/2. Each coordinate's endpoint
    coefficient norm is 2+h/2. Its departure from r*integral(S) is the
    trapezoid error of the diffusion integral, <=M2*h**3/12 per law.
    Dividing by h therefore gives h**2, not a first-order slope error.

    Three levels bound the sine quadrature difference. The acute initial
    gap domain then makes the difference of the two cycle sums less than
    2*pi, forcing equal winding. On the first plateau, the phase excursion
    is only <=2*g*M*h. Its initial force difference bounds B*f; C5 has
    diameter two. Equal winding supplies both signs of the fundamental
    sine difference, turning its range bound into an absolute bound.
    """
    laws = tuple(_bound_organized_observation_law(deformation=d) for d in (0, 1))
    if not all(law.admitted for law in laws):
        raise ArithmeticError("joint bounds require both own-law admissions")
    cells = _historical_cells()
    plateaus = tuple(cell for cell in cells if cell.name == "plateau")
    h = plateaus[0].duration
    if (
        cells[0].name != "plateau"
        or len(plateaus) != 15
        or any(cell.duration != h for cell in plateaus)
        or not 0 < h < 4
    ):
        raise ArithmeticError("joint bounds require the fixed fifteen paired plateaus")
    if any(
        (
            law.gamma_upper,
            law.duration,
            law.initial_form_upper,
            law.initial_outer_gap_upper,
            law.command_gap_upper,
        )
        != (
            laws[0].gamma_upper,
            laws[0].duration,
            laws[0].initial_form_upper,
            laws[0].initial_outer_gap_upper,
            laws[0].command_gap_upper,
        )
        for law in laws[1:]
    ):
        raise ArithmeticError("joint laws must retain the same clock and initial class")
    g, gamma_lower = laws[0].gamma_upper, Q(1, 3300)
    if gamma_lower >= 1 / (1023 * pi_interval().hi):
        raise ArithmeticError("the rational gamma lower bound is unavailable")
    delta = 2 * _READING_ERROR + sum(
        (law.source_form_error_upper for law in laws), Q(0)
    )
    forms = tuple(
        law.initial_form_upper + g * law.current_upper * law.duration / 2
        for law in laws
    )
    drifts = tuple(
        4 * g * law.initial_form_upper + 2 * g**2 * law.current_upper * law.duration
        for law in laws
    )
    seconds = tuple(
        form + g * law.current_upper / 2 + g**2 * law.current_lipschitz * form
        for law, form in zip(laws, forms)
    )
    reading = (2 + h / 2) * delta / h
    trapezoid = sum(seconds, Q(0)) * h**2 / 12
    global_drift = (
        g
        / 2
        * sum((law.current_lipschitz * drift for law, drift in zip(laws, drifts)), Q(0))
    )
    current_error = 4 * (reading + trapezoid + global_drift) / gamma_lower
    epsilon, angle = laws[0].command_gap_upper, laws[0].initial_outer_gap_upper
    if not 0 < 9 * epsilon < 1 or not 0 < angle < Q(3, 2):
        raise ArithmeticError("joint bounds require the declared Taylor domains")
    one_minus_cosine_lower = epsilon**2 / 2 - epsilon**4 / 24
    sine_quad_error = 2 * current_error / one_minus_cosine_lower
    # The 18th sine-quadrature coefficient is (1+cos(9*epsilon)) times
    # the ninth; its factor is <2 on this declared command domain.
    alternative_quad_error = (
        3 * (9 * epsilon) ** 2 / (2 * 2187 * one_minus_cosine_lower)
    )
    cosine_lower = 1 - angle**2 / 2 + angle**4 / 24 - angle**6 / 720
    gap_difference = (sine_quad_error + alternative_quad_error) / cosine_lower
    # pi>3 suffices; a common record implies an integer winding difference.
    winding_margin = 6 - 5 * gap_difference
    first_plateau_drift = (
        g**2
        * h
        * sum((law.current_lipschitz * form for law, form in zip(laws, forms)), Q(0))
    )
    initial_divergence = reading + trapezoid + first_plateau_drift
    ninth_radius_lower = gamma_lower / (4 * 2187)
    sine_ratio = 3 + 2 * initial_divergence / ninth_radius_lower
    initial_ratio = sine_ratio / cosine_lower
    reference_ratio = initial_ratio + 2187 * sum(drifts, Q(0))
    return _OrganizedLawJointBounds(
        delta,
        h,
        gamma_lower,
        g,
        (forms[0], forms[1]),
        (drifts[0], drifts[1]),
        (seconds[0], seconds[1]),
        reading,
        trapezoid,
        current_error,
        sine_quad_error,
        alternative_quad_error,
        cosine_lower,
        gap_difference,
        winding_margin,
        initial_divergence,
        sine_ratio,
        initial_ratio,
        reference_ratio,
        winding_margin > 0 and reference_ratio < 20,
    )
