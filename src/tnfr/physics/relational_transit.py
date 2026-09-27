"""Validated continuous transit in the exact reflected two-ring subsystem.

This read-only proof computation encloses the ideal ODE from a supplied graph.
It does not advance engine state or reinterpret an Euler chord as an ODE tube.
Exact initial reflection, fixed unit capacity and the supplied relational law
are prerequisites. Numerical-policy failure is explicitly inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import comparison_flow_upper
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import atan_ratio as jet_atan_ratio
from ..mathematics._interval_taylor import cos as jet_cos
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._interval_taylor import sinc as jet_sinc
from ..mathematics._rational_interval import I, atan_ratio, cos, pi_interval, sin
from .relational_capture import (
    RelationalCaptureCertificate,
    _capture_rectangle_candidates,
    _capture_rectangle_kind,
    certify_relational_capture,
)

__all__ = ("RelationalTransitCertificate", "certify_relational_transit_capture")


@dataclass(frozen=True)
class TransitStep:
    """One admitted whole-time tube and its exact endpoint enclosure."""

    time: Q
    duration: Q
    tube: tuple[I, ...]
    endpoint: tuple[I, ...]
    picard_interior_margin: Q
    resultant_real_lower_bounds: tuple[Q, ...]
    propagated_initial_radii: tuple[Q, ...]
    local_remainder_bounds: tuple[I, ...]


@dataclass(frozen=True)
class RelationalTransitCertificate:
    """Detached proof evidence, including every accepted interval step.

    ``validated_horizon`` is relative to the supplied state, not a certified
    interpretation of the graph's stored clock. A selected protected-rectangle
    endpoint admits the existing continuous capture theorem on the entire box.
    It does not amend any earlier finite-executor prediction verdict.
    """

    initial: RelationalCaptureCertificate
    horizon: Q
    time_step: Q
    order: int
    initial_box: tuple[I, ...]
    steps: tuple[TransitStep, ...]
    validated_horizon: Q
    endpoint: tuple[I, ...]
    endpoint_storage: I
    positive_rectangle_margins: tuple[I, ...]
    initial_winding_zero: bool
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    failed_tube: tuple[I, ...] | None
    method: str = "centered_rational_Taylor_Picard_Metzler128_v1"
    scope: tuple[str, ...] = (
        "exact_copied_reflected_two_C5_rings_with_matching_adjacent_bridges",
        "unit_held_capacity_positive_relational_coefficients_no_forcing",
        "mathematical_pi_and_trigonometry_at_exact_represented_initial_state",
        "whole_time_Picard_tubes_including_central_resultant_admission",
        "normalized_Taylor_remainder_and_componentwise_flow_comparison",
        "strict_selected_rectangle_and_storage_below_7beta_on_whole_endpoint_box",
        "conditional_ideal_ODE_capture_not_future_finite_runtime_admission",
        "no_graph_writes_no_state_projection_no_physical_identification",
    )
    requested_sector: int | None = 1
    candidate_rectangle_margin_bounds: tuple[tuple[int, tuple[I, ...]], ...] = ()
    rectangle_kind: str | None = None
    rectangle_margin_bounds: tuple[I, ...] = ()

    @property
    def admitted(self):
        return self.status == "admitted"


def _exact_positive(value, label):
    if not isinstance(value, Q) and type(value) is not int:
        raise TypeError(f"{label} requires an exact Fraction or integer")
    value = Q(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _regular_bounds(box):
    """All five node rows, including the central row omitted by symmetry."""
    _, _, a, b = box
    return (
        (1 + cos(2 * a) + cos(a - b)).lo,
        (2 * cos(a / 2) * cos(a / 2 - b)).lo,
        (2 * cos(b)).lo,
    )


def _sinc(value):
    # The scalar enclosure is a zero-order use of the same derivative owner.
    return jet_sinc(Jet.constant(value, 0)).coeffs[0]


def _flow(state, e, w, beta):
    q, r, a, b = state
    is_jet = isinstance(a, Jet)
    cs, sn = (jet_cos, jet_sin) if is_jet else (cos, sin)
    ratio = jet_atan_ratio if is_jet else atan_ratio
    sc = jet_sinc if is_jet else _sinc
    c0 = 1 + cs(2 * a) + cs(a - b)
    s0 = -sn(2 * a) - sn(a - b)
    u = s0 / c0
    ar = ratio(u)
    pi = pi_interval()
    g0 = u * ar / pi
    d = a / 2 - b
    g4 = d / pi
    h0_inverse = ar / c0 / pi
    h4_inverse = 1 / (cs(a / 2) * sc(d) * (2 * pi))
    return (
        -e * q + (e / 2) * r + w * (3 * g0 - g4),
        (e / 3) * q - e * r + w * (2 * g4 - g0),
        q * h0_inverse * (w / beta),
        r * h4_inverse * (w / beta),
    )


def _flow_jets(box, order, e, w, beta):
    coefficients = [[value] for value in box]
    for index in range(1, order + 1):
        rows = _flow(tuple(Jet(tuple(row)) for row in coefficients), e, w, beta)
        for row, rate in zip(coefficients, rows):
            row.append(rate.coeffs[index - 1] / index)
    return tuple(tuple(row) for row in coefficients)


def _comparison_matrix(tube, e, w, beta):
    columns = []
    for column in range(4):
        variables = tuple(
            Jet((value, I(int(index == column)))) for index, value in enumerate(tube)
        )
        columns.append(tuple(row.coeffs[1] for row in _flow(variables, e, w, beta)))
    return tuple(
        tuple(columns[j][i].hi if i == j else columns[j][i].abs_max for j in range(4))
        for i in range(4)
    )


def _tube(box, duration, e, w, beta):
    """Attempt a strict first-exit enclosure; inflation alone never admits."""
    tube = box
    epsilon = Q(1, 1 << 90)
    for _ in range(16):
        try:
            lower = _regular_bounds(tube)
            if min(lower) <= 0:
                return None, tube, "whole_tube_resultant_not_positive"
            rate = _flow(tube, e, w, beta)
            image = tuple(x + f * I(0, duration) for x, f in zip(box, rate))
            margin = min(min(y.lo - b.lo, b.hi - y.hi) for y, b in zip(image, tube))
            if margin > 0:
                return (tube, margin, lower), None, None
            enlarged = tuple(x.hull(y) for x, y in zip(box, image))
            tube = tuple(
                I(
                    value.midpoint - value.radius * Q(5, 4) - epsilon,
                    value.midpoint + value.radius * Q(5, 4) + epsilon,
                )
                for value in enlarged
            )
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            return None, tube, f"tube_arithmetic_unavailable: {exc}"
    return None, tube, "strict_Picard_inclusion_not_resolved"


def _validated_step(box, duration, e, w, beta, order, time):
    admission, failed, reason = _tube(box, duration, e, w, beta)
    if admission is None:
        return None, failed, reason
    tube, margin, resultants = admission
    try:
        center = tuple(I(value.midpoint) for value in box)
        series = _flow_jets(center, order, e, w, beta)
        remainder = tuple(
            row[-1] * duration ** (order + 1)
            for row in _flow_jets(tube, order + 1, e, w, beta)
        )
        polynomial = []
        for row in series:
            value = row[-1]
            for coefficient in reversed(row[:-1]):
                value = value * duration + coefficient
            polynomial.append(value)
        propagated = comparison_flow_upper(
            _comparison_matrix(tube, e, w, beta),
            tuple(value.radius for value in box),
            duration,
        )
        endpoint = tuple(
            value + error + I(-radius, radius)
            for value, error, radius in zip(polynomial, remainder, propagated)
        )
        # Both are proven enclosures; intersection only removes overestimation.
        if any(max(x.lo, b.lo) > min(x.hi, b.hi) for x, b in zip(endpoint, tube)):
            raise ArithmeticError("disjoint endpoint and whole-time enclosures")
        endpoint = tuple(
            I(max(x.lo, b.lo), min(x.hi, b.hi)) for x, b in zip(endpoint, tube)
        )
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return None, tube, f"Taylor_comparison_unavailable: {exc}"
    return (
        TransitStep(
            time, duration, tube, endpoint, margin, resultants, propagated, remainder
        ),
        None,
        None,
    )


def _storage(box, beta):
    q, r, a, b = box
    return (
        (q + r / 2) ** 2 * Q(4, 5)
        + r**2
        + (10 - 2 * cos(2 * a) - 4 * cos(a - b) - 4 * cos(b)) * beta
    )


def _positive_margins(box):
    """Compatibility view of the positive member of the shared basin ledger."""
    return _rectangle_margins(box)[0][1]


def _rectangle_margins(box):
    _, _, a, b = box
    pi = pi_interval()
    return _capture_rectangle_candidates(
        a, b, lambda coordinate, coefficient: coordinate + pi * coefficient
    )


def certify_relational_transit_capture(
    graph,
    *,
    model,
    cycles,
    horizon,
    time_step,
    order=12,
    requested_sector: int | None = 1,
):
    """Enclose ideal continuous transit and admit selected protected capture.

    The caller declares an exact positive horizon and step. No adaptive retry,
    preparation change or time extension is performed. Order is a numerical
    policy (4..16), not a model parameter. The first unresolved tube is retained
    as unavailable. Every accepted interval accounts for analytic truncation,
    outward arithmetic, and propagated initial uncertainty. This is a proof
    calculation on the exact invariant reduction, not an engine time step.
    ``requested_sector`` is +1 by default for compatibility; 0 and -1 require
    the consensus and negative rectangles respectively. ``None`` accepts any
    of the three disjoint rectangles, with the actual limiting sector retained.
    """
    if requested_sector is not None and (
        type(requested_sector) is not int or requested_sector not in (-1, 0, 1)
    ):
        raise ValueError("requested_sector must be -1, 0, 1 or None")
    horizon = _exact_positive(horizon, "horizon")
    time_step = _exact_positive(time_step, "time_step")
    if type(order) is not int or not 4 <= order <= 16:
        raise ValueError("Taylor order must be an integer from 4 to 16")
    if horizon / time_step > 4096:
        raise ValueError("at most 4096 declared proof steps are supported")
    initial = certify_relational_capture(graph, model=model, cycles=cycles)
    A, B, a, b = initial.coordinates
    box = initial_box = tuple(I(value) for value in (3 * A - B, 2 * B - A, a, b))
    e, w = map(Q, initial.field.model.effective_weights)
    beta = Q(initial.field.model.storage_scale)
    reasons = []
    if not initial.exact_symmetry:
        reasons.append("exact_copy_reflection_required")
    if not initial.unit_capacity:
        reasons.append("unit_held_capacity_required")
    if not initial.positive_epi_weight:
        reasons.append("positive_epi_weight_required")
    if model.phase_domain != "positive_resultant":
        reasons.append("positive_resultant_model_required")
    # Certify the original lift has zero winding independently of telemetry.
    pi = pi_interval()
    initial_zero = initial.exact_symmetry and all(
        value.abs_max < pi.lo for value in (-2 * box[2], box[2] - box[3], box[3])
    )
    steps, time, failed_tube = [], Q(0), None
    if not reasons:
        while time < horizon:
            duration = min(time_step, horizon - time)
            step, failed_tube, failure = _validated_step(
                box, duration, e, w, beta, order, time
            )
            if step is None:
                reasons.append(failure)
                break
            steps.append(step)
            box = step.endpoint
            time += duration
    storage = _storage(box, beta)
    candidates = _rectangle_margins(box)
    margins = candidates[0][1]
    selected = tuple(
        (sector, bounds)
        for sector, bounds in candidates
        if all(value.lo > 0 for value in bounds)
        and (requested_sector is None or sector == requested_sector)
    )
    if len(selected) > 1:
        raise ArithmeticError("disjoint capture rectangles cannot both be admitted")
    rectangle_sector, rectangle_bounds = selected[0] if selected else (None, ())
    if time != horizon:
        reasons.append("requested_horizon_not_validated")
    if not selected:
        reasons.append(
            "whole_endpoint_positive_rectangle_not_certified"
            if requested_sector == 1
            else "whole_endpoint_requested_rectangle_not_certified"
        )
    if storage.hi >= 7 * beta:
        reasons.append("whole_endpoint_storage_below_7beta_not_certified")
    return RelationalTransitCertificate(
        initial=initial,
        horizon=horizon,
        time_step=time_step,
        order=order,
        initial_box=initial_box,
        steps=tuple(steps),
        validated_horizon=time,
        endpoint=box,
        endpoint_storage=storage,
        positive_rectangle_margins=margins,
        initial_winding_zero=initial_zero,
        target_sector=None if reasons else rectangle_sector,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=tuple(reasons),
        failed_tube=failed_tube,
        requested_sector=requested_sector,
        candidate_rectangle_margin_bounds=candidates,
        rectangle_kind=_capture_rectangle_kind(rectangle_sector),
        rectangle_margin_bounds=rectangle_bounds,
    )
