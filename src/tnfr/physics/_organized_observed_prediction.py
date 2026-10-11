"""Static whole-record predictive sufficiency under unknown initialization.

The declared matrix contains forced heat responses to formal edge phasors,
not evaluated nonlinear source histories. Rational residuals, a global
nonlinear remainder and source transport certify a feasible-future diameter.
The dual weights alone are not a predictor of an individual raw record.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from typing import Any

from ..mathematics._interval_taylor import Jet, cos, sin
from ..mathematics._rational_interval import I
from ..mathematics._validated_linear_forcing import validated_linear_forcing_step
from ._organized_computable_prediction import _ForecastStage, _forecast_stages
from ._organized_electrical_source import _transport_hidden_deviations


_SIZE = 5
_ORDER = 16
_STEP = Q(1, 8)
_COEFFICIENT_BUDGET = Q(1, 10**12)
_PROPOSAL_SCALE = 1 << 40
_G = Q(1, 3000)
_T = Q(16)
_ETA = Q(1, 10**14)
_TARGET = Q(2, 10**13)
_GENERATOR = tuple(
    tuple(
        Q(-1, 2) if i == j else Q(1, 4) if (i - j) % 5 in (1, 4) else Q(0)
        for j in range(5)
    )
    for i in range(5)
)


@dataclass(frozen=True)
class _ObservationBudget:
    hidden_form_upper: Q
    hidden_phase_upper: Q
    source_error: Q
    historical_pair_error: Q
    edge_drift: Q
    heat_integral_upper: Q
    continuous_sample_margin: Q
    remainder_phasor_factor: Q
    remainder_record_factor: Q


def _observation_budget() -> _ObservationBudget:
    """Rebuild every source and nonlinear constant from declared primitives."""
    hidden = _transport_hidden_deviations(
        initial_form=Q(31, 10**9),
        initial_phase=Q(601, 10**10),
        gamma_upper=_G,
        laplacian_upper=Q(8),
        duration=_T,
    )
    source = _T * _G * hidden.phase_upper**2 / (9 * (1 - (_G * _T) ** 2))
    drift = 4 * _G * Q(1, 100) + 2 * _G**2 * _T
    heat = Q(6)
    a, beta = heat * 3 * drift, heat * _G**2 * _T
    margin = 1 - 2 * _G**2 * _T
    return _ObservationBudget(
        hidden.form_upper,
        hidden.phase_upper,
        source,
        2 * (_ETA + source),
        drift,
        heat,
        margin,
        a + beta * 6 * (1 + drift) / margin,
        beta / margin,
    )


@lru_cache(maxsize=4)
def _edge_forcing(stage: _ForecastStage, time: Jet) -> tuple[tuple[Jet, ...], ...]:
    """All ten columns of -B[cos(B'H), sin(B'H)] at one clock jet."""
    command = stage.command(time)
    zero = time * 0
    columns = []
    for function in (cos, sin):
        for edge in range(5):
            head = (edge + 1) % 5
            value = function(command[head] - command[edge])
            columns.append(
                tuple(
                    value if i == edge else -value if i == head else zero
                    for i in range(5)
                )
            )
    return tuple(columns)


@dataclass(frozen=True)
class _CoefficientStep:
    column: int
    time: Q
    duration: Q
    polynomial: tuple[I, ...]
    remainder: tuple[I, ...]
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q


@dataclass(frozen=True)
class _StaticDesign:
    observation_times: tuple[Q, ...]
    observation_matrix: tuple[tuple[I, ...], ...]
    future_matrix: tuple[tuple[I, ...], ...]
    steps_per_column: int
    maximum_coefficient_radius: Q
    maximum_local_error: Q
    admitted: bool
    reason: str | None
    steps: tuple[_CoefficientStep, ...] = ()


def _static_design() -> _StaticDesign:
    """The one fixed 15-column design; no physical initial state is supplied."""
    centers = [tuple(Q(i == column) for i in range(5)) for column in range(15)]
    radii = [Q(0)] * 15
    times: list[Q] = []
    rows: list[tuple[I, ...]] = []
    maximum_local = Q(0)
    steps = 0
    witnesses: list[_CoefficientStep] = []

    def snapshot() -> tuple[tuple[I, ...], ...]:
        return tuple(
            tuple(
                I(centers[j][i] - radii[j], centers[j][i] + radii[j]) for j in range(15)
            )
            for i in range(5)
        )

    for stage in _forecast_stages()[1:]:
        time = stage.start
        if stage.name == "plateau":
            times.append(time - 200)
            rows.extend(snapshot())
        end = stage.start + stage.duration
        while time < end:
            duration = min(_STEP, end - time)
            for column in range(15):

                def forcing(clock: Jet) -> tuple[Jet, ...]:
                    if column < 5:
                        return (clock * 0,) * 5
                    return _edge_forcing(stage, clock)[column - 5]

                step = validated_linear_forcing_step(
                    centers[column],
                    radii[column],
                    _GENERATOR,
                    duration,
                    forcing,
                    order=_ORDER,
                    time=time,
                )
                centers[column] = step.endpoint_center
                radii[column] = step.endpoint_radius
                maximum_local = max(maximum_local, step.local_error)
                witnesses.append(
                    _CoefficientStep(
                        column,
                        time,
                        duration,
                        step.polynomial,
                        step.remainder,
                        step.endpoint_center,
                        step.endpoint_radius,
                    )
                )
            steps += 1
            time += duration
        if stage.name == "plateau":
            times.append(time - 200)
            rows.extend(snapshot())
    radius = max(radii)
    admitted = radius <= _COEFFICIENT_BUDGET
    return _StaticDesign(
        tuple(times),
        tuple(rows),
        snapshot(),
        steps,
        radius,
        maximum_local,
        admitted,
        None if admitted else "static_coefficient_budget_exceeded",
        tuple(witnesses),
    )


def _rational_proposal(values: Any) -> tuple[tuple[Q, ...], ...]:
    """Finite binary64 proposals rounded once to the declared dyadic grid."""
    import math

    rows = []
    for row in values:
        normalized = []
        for value in row:
            number = float(value)
            if not math.isfinite(number):
                raise ArithmeticError("nonfinite matrix proposal")
            normalized.append(
                Q(round(Q.from_float(number) * _PROPOSAL_SCALE), _PROPOSAL_SCALE)
            )
        rows.append(tuple(normalized))
    return tuple(rows)


def _dual_proposal(design: _StaticDesign) -> tuple[Any, Any, str]:
    from contextlib import redirect_stdout
    from io import StringIO

    import numpy as np

    matrix = np.array(
        [[float(x.midpoint) for x in row] for row in design.observation_matrix]
    )
    target = np.array(
        [[float(x.midpoint) for x in row] for row in design.future_matrix]
    )
    inverse = np.linalg.solve(matrix.T @ matrix, matrix.T)
    configuration = StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    backend = f"NumPy {np.__version__}; binary64 proposals\n{configuration.getvalue()}"
    # Proposal generation has no evidential status; both arrays are checked below.
    return (
        _rational_proposal(inverse[5:]),
        _rational_proposal(target @ inverse),
        backend,
    )


@dataclass(frozen=True)
class _DualCertificate:
    phase_weights: tuple[tuple[Q, ...], ...]
    future_weights: tuple[tuple[Q, ...], ...]
    phase_weight_norm: Q
    future_weight_norm: Q
    inverse_initial_defect: Q
    inverse_phasor_defect: Q
    future_initial_defect: Q
    future_phasor_defect: Q
    bootstrap_margin: Q
    phasor_gain: Q | None
    carrier_future_gain: Q | None
    fine_future_diameter_upper: Q | None
    target_margin: Q | None
    admitted: bool
    reason: str | None


def _certify_dual(
    design: _StaticDesign, phase_weights: Any, future_weights: Any
) -> _DualCertificate:
    """Re-admit primitives and enclose residuals; trust no matrix verdict."""
    from ..mathematics._comparison_flow import _exact, _ordered

    def matrix(values: Any, n: int, m: int, intervals: bool = False) -> tuple:
        rows = tuple(
            tuple(_ordered(row, "matrix row")) for row in _ordered(values, "matrix")
        )
        if len(rows) != n or any(len(row) != m for row in rows):
            raise ValueError("matrix dimensions differ from the declared design")
        return tuple(
            tuple(I.coerce(x) if intervals else _exact(x) for x in row) for row in rows
        )

    observation = matrix(design.observation_matrix, 150, 15, True)
    future = matrix(design.future_matrix, 5, 15, True)
    k = matrix(phase_weights, 10, 150)
    w = matrix(future_weights, 5, 150)

    def defects(weights: tuple, target: tuple) -> tuple[Q, Q]:
        residual = tuple(
            tuple(
                target[i][j]
                - sum(
                    (I(weight) * observation[t][j] for t, weight in enumerate(row)),
                    I(0),
                )
                for j in range(15)
            )
            for i, row in enumerate(weights)
        )
        return tuple(
            max(sum(x.abs_max for x in row[lo:hi]) for row in residual)
            for lo, hi in ((0, 5), (5, 15))
        )

    identity = tuple(tuple(I(int(j == i + 5)) for j in range(15)) for i in range(10))
    e0, eb = defects(k, identity)
    n0, nb = defects(w, future)
    knorm = max(sum(map(abs, row)) for row in k)
    wnorm = max(sum(map(abs, row)) for row in w)
    budget = _observation_budget()
    a, b = budget.remainder_phasor_factor, budget.remainder_record_factor
    margin = 1 - eb - knorm * a
    gain = future_gain = diameter = target_margin = None
    if margin > 0:
        gain = (knorm * (1 + b) + e0) / margin
        future_gain = wnorm + n0 + nb * gain + (1 + wnorm) * (a * gain + b)
        diameter = budget.historical_pair_error * future_gain + 2 * budget.source_error
        target_margin = _TARGET - diameter
    admitted = target_margin is not None and target_margin >= 0
    return _DualCertificate(
        k,
        w,
        knorm,
        wnorm,
        e0,
        eb,
        n0,
        nb,
        margin,
        gain,
        future_gain,
        diameter,
        target_margin,
        admitted,
        (
            None
            if admitted
            else (
                "nonpositive_bootstrap_margin"
                if margin <= 0
                else "future_diameter_target_not_certified"
            )
        ),
    )


@dataclass(frozen=True)
class _ObservedPredictionAssessment:
    budget: _ObservationBudget
    design: _StaticDesign
    certificate: _DualCertificate | None
    proposal_backend: str | None
    admitted: bool
    reason: str | None


def _assess_organized_observed_prediction() -> _ObservedPredictionAssessment:
    """Execute once only after the complete declaration/source is committed."""
    design = _static_design()
    certificate = None
    backend = None
    reason = design.reason
    if design.admitted:
        try:
            k, w, backend = _dual_proposal(design)
        except (ArithmeticError, ValueError) as error:
            reason = f"matrix_proposal_unavailable:{type(error).__name__}"
        else:
            certificate = _certify_dual(design, k, w)
            reason = certificate.reason
    admitted = certificate is not None and certificate.admitted
    return _ObservedPredictionAssessment(
        _observation_budget(), design, certificate, backend, admitted, reason
    )
