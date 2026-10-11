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
