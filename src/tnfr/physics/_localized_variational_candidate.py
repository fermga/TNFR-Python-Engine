"""Bounded binary64 proposals for carried-state cycle variational panels.

The finite Chebyshev integrated equation is a proposal mechanism. Neither
its projected feedback iteration nor its floating modal solves inherit the
exact Volterra contraction theorem. The independent interval panel owner
must validate the returned exact represented polynomial against admitted
source, generator, initial, and join evidence.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Set
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from time import perf_counter

import numpy as np
from scipy.fft import dct, fft, ifft
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from .._exact_time import exact_or_represented_real as _exact
from ._localized_variational_panels import _interval, _ordered_rows

_MAX_DEGREE = 96
_FEEDBACK_UPDATES = 20


def _finite(values, label):
    if not np.all(np.isfinite(values)):
        raise ArithmeticError(f"{label} produced nonfinite proposal arithmetic")
    return values


def _represented_value(center, label):
    try:
        result = float(center)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} is not representable in binary64") from exc
    if not math.isfinite(result) or (center and result == 0.0):
        raise ValueError(f"{label} is not representable in binary64")
    return result


def _represented_midpoint(value, label):
    return _represented_value(value.midpoint, f"{label} midpoint")


def _initial_row(values, label, size):
    """Keep an exact incoming point; an uncertain interval is not a point."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered finite sequence")
    try:
        count = len(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered finite sequence") from exc
    if count != size:
        raise ValueError(f"{label} must have exactly {size} coordinates")
    try:
        raw = tuple(values[index] for index in range(count))
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(f"{label} must be an indexed finite sequence") from exc
    return tuple(_exact(value, f"{label}[{index}]") for index, value in enumerate(raw))


def _center_table(rows, degree, label):
    result = np.zeros((len(rows), degree + 1))
    for node, row in enumerate(rows):
        for index, value in enumerate(row):
            result[node, index] = _represented_midpoint(
                value, f"{label}[{node},{index}]"
            )
    return result


def _integration_matrix(degree):
    """Integrate, drop degree d+1, then restore the exact zero-left convention.

    In exact arithmetic the discarded contribution is
    f_d*(T_(d+1)-(-1)^(d+1)T_0)/(2(d+1)). It remains part of the
    subsequently validated equation defect, not a free truncation.
    """
    matrix = np.zeros((degree + 1, degree + 1))
    matrix[1, 0] = 1.0
    if degree >= 2:
        matrix[1, 2] = -0.5
    for index in range(2, degree + 1):
        matrix[index, index - 1] = 1 / (2 * index)
        if index < degree:
            matrix[index, index + 1] = -1 / (2 * index)
    signs = (-1.0) ** np.arange(1, degree + 1)
    matrix[0] = -(signs @ matrix[1:])
    return matrix


def _lobatto_values(coefficients, sample_degree):
    """Evaluate ordinary coefficients at sample_degree+1 Lobatto nodes."""
    padded = np.zeros((len(coefficients), sample_degree + 1))
    padded[:, : coefficients.shape[1]] = coefficients
    padded[:, 1:sample_degree] *= 0.5
    return _finite(dct(padded, type=1, axis=1), "Lobatto evaluation")


def _lobatto_coefficients(values):
    degree = values.shape[1] - 1
    coefficients = dct(values, type=1, axis=1) / degree
    coefficients[:, (0, degree)] *= 0.5
    return _finite(coefficients, "Lobatto transform")


def _feedback_coefficients(phase, edge_values, degree):
    """Form the degree2d product before projecting the candidate feedback."""
    phase_values = _lobatto_values(phase, 2 * degree)
    flux = edge_values * (phase_values - np.roll(phase_values, -1, axis=0))
    feedback = (flux - np.roll(flux, 1, axis=0)) / 2
    _finite(feedback, "edge feedback")
    return _lobatto_coefficients(feedback)[:, : degree + 1]


def _initial_coefficients(values, initial):
    """Enforce p(-1)=initial exactly, without rounding the incoming point."""
    result = []
    for row, incoming in zip(_finite(values, "candidate reconstruction"), initial):
        tail = tuple(Q.from_float(float(value)) for value in row[1:])
        constant = incoming - sum(
            ((-1 if index % 2 else 1) * value for index, value in enumerate(tail, 1)),
            Q(0),
        )
        result.append((constant,) + tail)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _VariationalCandidate:
    form_coefficients: tuple[tuple[Q, ...], ...]
    phase_coefficients: tuple[tuple[Q, ...], ...]
    initial_form: tuple[Q, ...]
    initial_phase: tuple[Q, ...]
    node_count: int
    degree: int
    feedback_iterations: int
    factor_count: int
    sigma_center: Q
    duration_center: Q
    factorization_seconds: float
    feedback_seconds: float
    total_seconds: float
    method: str = "carried_cycle_chebyshev_modal_binary64_proposal_v1"
    scope: tuple[str, ...] = (
        "midpoint_inputs_are_binary64_proposal_choices_not_interval_enclosures",
        "full_complex_FFT_retains_both_real_rows_and_every_spatial_mode",
        "cached_small_temporal_LU_factors_no_dense_spatial_generator",
        "degree2d_feedback_formed_before_proposal_only_projection",
        "base_source_solve_followed_by_twenty_fixed_feedback_updates",
        "incoming_point_preserved_by_exact_rational_endpoint_recentring",
        "incoming_uncertainty_and_accumulated_defect_require_external_budget",
        "no_convergence_error_bound_canonical_admission_or_retention_verdict",
        "independent_interval_defect_initial_and_join_validation_required",
    )


def _construct_cycle_candidate(
    source_form_coefficients,
    source_phase_coefficients,
    edge_coefficient_polynomials,
    *,
    initial_form,
    initial_phase,
    sigma,
    duration,
    degree=96,
):
    """Propose a finite polynomial with fixed work, without certifying accuracy.

    All coefficient tables and scalar endpoints are re-admitted before float
    conversion. Each table has the same 3..1024 nodes, and nonempty ordinary
    Chebyshev rows of degree at most the declared ordinary integer 1..96.
    The whole sigma interval lies in [0,2), and duration in (0,16].
    Both incoming rows are exact or represented scalar points of the same
    size. Intervals are rejected: accumulated uncertainty belongs to the
    independent global defect and join budget, not a reset at this panel.

    Interval midpoints are explicitly used only to construct a candidate.
    Twenty feedback updates follow a base source solve; there is no adaptive
    stopping, conditioning certificate, or transferred contraction claim.
    The returned rational coefficients define the actual proposed polynomial
    and preserve the incoming point exactly at their left endpoint. Supplying
    the preceding exact polynomial endpoint therefore gives an exact join;
    externally rounded/replaced endpoints require their own mismatch bound.
    """
    started = perf_counter()
    if type(degree) is not int or not 1 <= degree <= _MAX_DEGREE:
        raise ValueError("candidate degree must be an ordinary integer in 1..96")
    source_form = _ordered_rows(source_form_coefficients, "candidate form source")
    size = len(source_form)
    source_phase = _ordered_rows(
        source_phase_coefficients, "candidate phase source", size
    )
    edge = _ordered_rows(
        edge_coefficient_polynomials, "candidate edge coefficients", size
    )
    incoming_form = _initial_row(initial_form, "candidate initial form", size)
    incoming_phase = _initial_row(initial_phase, "candidate initial phase", size)
    admitted_sigma = _interval(sigma, "candidate sigma")
    admitted_duration = _interval(duration, "candidate duration")
    if admitted_sigma.lo < 0 or admitted_sigma.hi >= 2:
        raise ValueError("candidate sigma must be wholly in [0,2)")
    if admitted_duration.lo <= 0 or admitted_duration.hi > 16:
        raise ValueError("candidate duration must be wholly in (0,16]")
    if any(
        len(row) > degree + 1
        for rows in (source_form, source_phase, edge)
        for row in rows
    ):
        raise ValueError(
            "input polynomial degree exceeds the declared candidate degree"
        )

    # Complete primitive admission above precedes every numerical proposal.
    loss = _represented_midpoint(admitted_sigma, "candidate sigma")
    panel_length = _represented_midpoint(admitted_duration, "candidate duration")
    if not 0 <= loss < 2 or not 0 < panel_length <= 16:
        raise ValueError("represented proposal scalars leave their numerical domain")
    form = _center_table(source_form, degree, "candidate form source")
    phase = _center_table(source_phase, degree, "candidate phase source")
    edge_centers = _center_table(edge, degree, "candidate edge coefficients")
    initial_x = np.array(
        [_represented_value(value, "candidate initial form") for value in incoming_form]
    )
    initial_theta = np.array(
        [
            _represented_value(value, "candidate initial phase")
            for value in incoming_phase
        ]
    )
    frequency = math.sqrt((1 - loss / 2) * (1 + loss / 2))
    alpha, mu = loss / 2 + 1j * frequency, -loss / 2 + 1j * frequency
    integration = _integration_matrix(degree)
    half_length = panel_length / 2

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        edge_values = _lobatto_values(edge_centers, 2 * degree)
        source = _finite(form + alpha * phase, "complex source encoding")
        source_modes = _finite(
            fft(source, axis=0, norm="ortho"), "source Fourier transform"
        )
        source_rhs = _finite(
            -half_length * (source_modes @ integration.T), "integrated source"
        )
        initial_modes = _finite(
            fft(initial_x + alpha * initial_theta, norm="ortho"),
            "initial Fourier transform",
        )
        source_rhs[:, 0] += initial_modes
        _finite(source_rhs, "initial and integrated source")
        factors = []
        factor_started = perf_counter()
        identity = np.eye(degree + 1, dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            for mode in range(1, size // 2 + 1):
                eigenvalue = 2 * math.sin(math.pi * mode / size) ** 2
                matrix = _finite(
                    identity - half_length * eigenvalue * mu * integration,
                    "modal temporal matrix",
                )
                try:
                    factor = lu_factor(matrix, check_finite=False)
                except LinAlgWarning as exc:
                    raise ArithmeticError(
                        "singular modal candidate factorization"
                    ) from exc
                _finite(factor[0], "modal factorization")
                group = (mode,) if mode == size - mode else (mode, size - mode)
                factors.append((group, factor))
        factor_seconds = perf_counter() - factor_started

        def solve_modes(right_hand_side):
            _finite(right_hand_side, "modal right hand side")
            solution = np.empty_like(right_hand_side)
            solution[0] = right_hand_side[0]  # The constant mode has lambda=0.
            for group, factor in factors:
                indices = list(group)
                solution[indices] = lu_solve(
                    factor, right_hand_side[indices].T, check_finite=False
                ).T
            return _finite(solution, "modal solve")

        modes = solve_modes(source_rhs)
        feedback_started = perf_counter()
        for _ in range(_FEEDBACK_UPDATES):
            encoded = _finite(
                ifft(modes, axis=0, norm="ortho"), "inverse Fourier transform"
            )
            phase_candidate = _finite(encoded.imag / frequency, "phase reconstruction")
            feedback = _feedback_coefficients(phase_candidate, edge_values, degree)
            feedback_modes = _finite(
                fft(feedback, axis=0, norm="ortho"), "feedback Fourier transform"
            )
            modes = solve_modes(
                source_rhs + half_length * (feedback_modes @ integration.T)
            )
        feedback_seconds = perf_counter() - feedback_started
        encoded = _finite(ifft(modes, axis=0, norm="ortho"), "final inverse transform")
        phase_candidate = _finite(
            encoded.imag / frequency, "final phase reconstruction"
        )
        form_candidate = _finite(
            encoded.real - (loss / 2) * phase_candidate, "final form reconstruction"
        )

    exact_form = _initial_coefficients(form_candidate, incoming_form)
    exact_phase = _initial_coefficients(phase_candidate, incoming_phase)
    return _VariationalCandidate(
        form_coefficients=exact_form,
        phase_coefficients=exact_phase,
        initial_form=incoming_form,
        initial_phase=incoming_phase,
        node_count=size,
        degree=degree,
        feedback_iterations=_FEEDBACK_UPDATES,
        factor_count=len(factors),
        sigma_center=Q.from_float(loss),
        duration_center=Q.from_float(panel_length),
        factorization_seconds=factor_seconds,
        feedback_seconds=feedback_seconds,
        total_seconds=perf_counter() - started,
    )


def _construct_zero_initial_cycle_candidate(
    source_form_coefficients,
    source_phase_coefficients,
    edge_coefficient_polynomials,
    *,
    sigma,
    duration,
    degree=96,
):
    """Compatibility entry point for the exactly zero incoming point."""
    started = perf_counter()
    form = _ordered_rows(source_form_coefficients, "candidate form source")
    result = _construct_cycle_candidate(
        form,
        source_phase_coefficients,
        edge_coefficient_polynomials,
        initial_form=(0,) * len(form),
        initial_phase=(0,) * len(form),
        sigma=sigma,
        duration=duration,
        degree=degree,
    )
    scope = tuple(
        (
            "zero_initial_endpoint_enforced_after_exact_represented_coefficient_admission"
            if item == "incoming_point_preserved_by_exact_rational_endpoint_recentring"
            else item
        )
        for item in result.scope
        if item != "incoming_uncertainty_and_accumulated_defect_require_external_budget"
    )
    return replace(
        result,
        method="zero_initial_cycle_chebyshev_modal_binary64_proposal_v1",
        scope=scope,
        total_seconds=perf_counter() - started,
    )
