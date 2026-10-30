"""Conditional whole-record separation from retained primitive moment bounds.

The supplied C,D intervals must enclose the declared fixed integrals. This
consumer re-admits those primitives and rebuilds every consumed contrast and
history budget. It never evaluates a moment, response, fit or raw record;
the evidence reader owns association with the frozen numerical assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real, exp_unit_bounds
from ..mathematics._rational_interval import I, pi_interval, sqrt
from ._organized_law_contrast_kernels import (
    _MomentKernel,
    _RampContrast,
    _three_ramp_contrast,
)
from ._organized_law_joint_bounds import (
    _bound_organized_law_joint_phases,
    _OrganizedLawJointBounds,
)
from ._organized_law_observation import (
    _bound_organized_observation_law,
    _certify_organized_phase_circle,
    _OrganizedPhaseCircleCertificate,
)
from ._organized_law_phase_transport import _historical_cells
from ._organized_outer_winding import _derive_organized_outer_winding
from .relational_observations import _ordered

_HARMONICS = (1, 2, 9, 18, 36)
_LATE_PHASE_CAP = Q(1, 10**6)


@dataclass(frozen=True)
class _OrganizedLawLateBounds:
    interval_start: Q
    reference_time: Q
    interval_end: Q
    maximum_reference_distance: Q
    observation_times: tuple[Q, ...]
    ramp_endpoints: tuple[tuple[tuple[Q, Q], ...], ...]
    eigenvalues: tuple[I, I]
    gamma_upper: Q
    heat_phase_upper: Q
    forced_phase_upper: Q
    phase_motion_upper: Q
    phase_motion_cap: Q
    form_difference_radius_ratio: Q
    relative_phase_current_radius_ratio: Q
    current_remainder_radius_ratio: Q
    admitted: bool
    scope: str = "conditional_shared_record_late_history_bounds"


@dataclass(frozen=True)
class _OrganizedLawRecordSeparation:
    moments: tuple[_MomentKernel, ...]
    joint_bounds: _OrganizedLawJointBounds
    late_bounds: _OrganizedLawLateBounds
    contrast: _RampContrast | None
    current_remainder_radius_ratio: Q
    even_transport_multiplier: I | None
    odd_transport_multiplier: I | None
    even_error: Q | None
    odd_error: Q | None
    circle: _OrganizedPhaseCircleCertificate | None
    available: bool
    separation_certified: bool
    reason: str
    scope: str = "conditional_complete_noisy_record_sets_separation"


def _decay(mu: I, duration: Any) -> I:
    """Bound exp(-mu*t) by fixed subdivision of the exact-time owner."""
    if not isinstance(mu, I):
        raise TypeError("decay requires an explicit spectral interval")
    mu = I(mu.lo, mu.hi)
    duration = exact_or_represented_real(duration, "decay duration")
    if not 0 <= mu.lo <= mu.hi <= 1 or not 0 <= duration <= 16:
        raise ValueError("decay requires mu in [0,1] and duration in [0,16]")
    lower, _ = exp_unit_bounds(mu.lo * duration / 16)
    _, upper = exp_unit_bounds(mu.hi * duration / 16)
    return (1 / I(lower, upper)) ** 16


def _late_grid() -> tuple[Q, Q, Q, tuple[Q, ...], tuple[tuple[tuple[Q, Q], ...], ...]]:
    """Rebuild the two selected blocks, including every observed endpoint."""
    cells = _historical_cells()
    origin = cells[0].start
    observations = (Q(0),) + tuple(
        cell.start + cell.duration - origin for cell in cells
    )
    selected = tuple(cell for cell in cells if cell.child in (2, 3))
    blocks = tuple(tuple(cell for cell in selected if cell.child == c) for c in (2, 3))
    if any(
        len(block) != 6
        or tuple(cell.name for cell in block)
        != ("plateau", "ramp", "plateau", "ramp", "plateau", "ramp")
        for block in blocks
    ):
        raise ArithmeticError("late contrast requires the two complete child blocks")
    h = blocks[0][0].duration
    if h != Q(1, 100000):
        raise ArithmeticError("late contrast requires the fixed plateau duration")
    endpoints = []
    for child, block in zip((2, 3), blocks):
        if block[0].start - origin != 3 * child * (1 + h):
            raise ArithmeticError("late contrast requires the declared block times")
        for cell, base in zip(block[::2], (Q(0), Q(1, 10), Q(-1, 10))):
            if cell.duration != h or cell.increment != 0 or cell.base != base:
                raise ArithmeticError("late readings require constant short plateaus")
        ramps = block[1::2]
        if any(
            cell.duration != 1 or cell.base != base or cell.increment != increment
            for cell, (base, increment) in zip(
                ramps, ((Q(0), Q(1, 10)), (Q(1, 10), Q(-1, 5)), (Q(-1, 10), Q(1, 10)))
            )
        ):
            raise ArithmeticError("late moments require the declared cubic ramps")
        endpoints.append(
            tuple(
                (cell.start - origin, cell.start + cell.duration - origin)
                for cell in ramps
            )
        )
    if any(
        time not in observations
        for block in endpoints
        for pair in block
        for time in pair
    ):
        raise ArithmeticError("every consumed ramp endpoint must be observed")
    start = blocks[0][0].start - origin
    reference = blocks[1][0].start - origin
    end = blocks[1][-1].start + blocks[1][-1].duration - origin
    if reference - start != 3 * (1 + h) or end - reference != 3 * (1 + h):
        raise ArithmeticError("late reference must retain both complete blocks")
    return start, reference, end, observations, (endpoints[0], endpoints[1])


def _bound_late_history(joint: _OrganizedLawJointBounds) -> _OrganizedLawLateBounds:
    """Use a freshly rebuilt joint bound; no caller-supplied verdict is consumed."""
    start, reference, end, observations, endpoints = _late_grid()
    root = sqrt(5)
    modes = ((5 - root) / 8, (5 + root) / 8)
    rows = (1 + 1 / root, 1 + 3 / root)
    gamma = Q(1, 3200)
    if 1 / (1023 * pi_interval().lo) >= gamma:
        raise ArithmeticError(
            "late heat bound requires the justified gamma upper bound"
        )
    law = _bound_organized_observation_law(deformation=1)
    projector_norm = (2 + 2 * root) / 5
    if projector_norm.hi >= Q(13, 10):
        raise ArithmeticError("the C5 spectral-projector row bound is unavailable")
    left = sum(
        (
            row.hi * (_decay(mu, start).hi - _decay(mu, reference).lo)
            for row, mu in zip(rows, modes)
        ),
        Q(0),
    )
    right = sum(
        (
            row.hi * (_decay(mu, reference).hi - _decay(mu, end).lo)
            for row, mu in zip(rows, modes)
        ),
        Q(0),
    )
    distance = max(reference - start, end - reference)
    heat = gamma * law.initial_form_upper * max(left, right)
    forced = Q(13, 8) * gamma**2 * law.current_upper * distance
    radius_lower = _derive_organized_outer_winding().gamma_bounds.lo / (4 * 2187)
    # The all-time phasor bound gives forcing <=43*r9. A preceding sample
    # lies at most one unit earlier at every point in these historical cells.
    form_ratio = joint.sample_difference_upper / radius_lower + 43
    relative = 22 * gamma**2 * distance
    remainder = 38 * _LATE_PHASE_CAP + relative
    admitted = (
        joint.admitted
        and joint.reference_fundamental_radius_ratio < 20
        and law.admitted
        and heat + forced < _LATE_PHASE_CAP
        and form_ratio < 44
        and remainder < Q(46, 10**6)
    )
    return _OrganizedLawLateBounds(
        start,
        reference,
        end,
        distance,
        observations,
        endpoints,
        modes,
        gamma,
        heat,
        forced,
        heat + forced,
        _LATE_PHASE_CAP,
        form_ratio,
        relative,
        remainder,
        admitted,
    )


def _admit_moments(raw: Any, mu: I) -> tuple[_MomentKernel, ...]:
    values = _ordered(raw, "selected primitive moments", limit=6)
    if len(values) != 5:
        raise ValueError("mode two requires the five declared harmonic moments")
    rebuilt = []
    for item, harmonic in zip(values, _HARMONICS):
        if not isinstance(item, _MomentKernel):
            raise TypeError("primitive moment entries must be explicit moment records")
        if type(item.mode) is not int or item.mode != 2:
            raise ValueError("record separation uses only the declared second mode")
        if type(item.harmonic) is not int or item.harmonic != harmonic:
            raise ValueError("primitive moments require the declared harmonic order")
        if not isinstance(item.mu, I):
            raise TypeError("moment eigenvalue must be an explicit interval")
        observed_mu = I(item.mu.lo, item.mu.hi)
        if (observed_mu.lo, observed_mu.hi) != (mu.lo, mu.hi):
            raise ValueError("moment eigenvalue does not match the canonical C5 mode")
        primitives = (item.sine_hyperbolic, item.cosine_hyperbolic)
        if not all(isinstance(value, I) for value in primitives):
            raise TypeError("moment C,D primitives must be explicit intervals")
        sine, cosine = (I(value.lo, value.hi) for value in primitives)
        # The enclosures already include their producer's truncations. These
        # unused source diagnostics are not transported as checked premises.
        rebuilt.append(_MomentKernel(2, harmonic, mu, sine, cosine, Q(0), Q(0)))
    return tuple(rebuilt)


def _certify_organized_law_record_separation(
    *, moments: Any
) -> _OrganizedLawRecordSeparation:
    """Rebuild the complete conditional exclusion from primitive kernel bounds.

    Validity of the supplied integral enclosures is an explicit premise. The
    retained-evidence consumer must bind them to the declared assessment.
    Broad intervals and failed strict margins abstain; no overlap follows.
    """
    joint = _bound_organized_law_joint_phases()
    late = _bound_late_history(joint)
    mu = late.eigenvalues[1]
    primitive = _admit_moments(moments, mu)
    contrast = _three_ramp_contrast(2, mu, {item.harmonic: item for item in primitive})
    if not late.admitted or contrast is None:
        return _OrganizedLawRecordSeparation(
            primitive,
            joint,
            late,
            contrast,
            late.current_remainder_radius_ratio,
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            (
                "late_history_premise_unavailable"
                if not late.admitted
                else "fundamental_elimination_denominator_contains_zero"
            ),
        )
    alpha_e, alpha_o = contrast.alpha_even, contrast.alpha_odd
    even_weight = abs(1 + alpha_e) + 2 * abs(alpha_e) + abs(alpha_e - 1)
    odd_weight = abs(1 - alpha_o) + 2 * abs(alpha_o) + abs(1 + alpha_o)
    exp_lower, _ = exp_unit_bounds(mu.lo / 2)
    _, exp_upper = exp_unit_bounds(mu.hi / 2)
    common = I(exp_lower, exp_upper) * (1 - _decay(mu, 1)) / mu * (3 + sqrt(5)) / 4
    even_multiplier, odd_multiplier = common * even_weight, common * odd_weight
    radius = _derive_organized_outer_winding().gamma_bounds / (4 * 2187)
    current_error = radius.hi * late.current_remainder_radius_ratio
    even_error = (
        contrast.even_noise_multiplier.hi * joint.sample_difference_upper
        + even_multiplier.hi * current_error
    )
    odd_error = (
        contrast.odd_noise_multiplier.hi * joint.sample_difference_upper
        + odd_multiplier.hi * current_error
    )
    circle = _certify_organized_phase_circle(
        odd_coefficients=contrast.odd_coefficients,
        even_coefficients=contrast.even_coefficients,
        odd_error=odd_error,
        even_error=even_error,
    )
    return _OrganizedLawRecordSeparation(
        primitive,
        joint,
        late,
        contrast,
        late.current_remainder_radius_ratio,
        even_multiplier,
        odd_multiplier,
        even_error,
        odd_error,
        circle,
        circle.available,
        circle.exclusion_certified,
        (
            "strict_complete_record_separation"
            if circle.exclusion_certified
            else circle.reason
        ),
    )
