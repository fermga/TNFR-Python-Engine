"""Analytical transverse bounds for the supplied weighted-star collective family.

This is a fixed complete-law family, not an observer of an arbitrary graph.
The two distinct Laplacians and all three internal spatial modes are retained.
Scalar interval arithmetic bounds a rotation comparison over one collective
libration period; no trajectory or monodromy is numerically integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, exp_unit_bounds
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_weighted_star import (
    BARE_DEGREES,
    BLOCKS,
    CAPACITY_FAMILY,
    CONDUCTANCES,
    EDGES,
    LAW,
    NODES,
)
from .relational_observations import _ordered

__all__ = (
    "SineWeightedStarTransverseBand",
    "assess_sine_weighted_star_transverse_band",
)


@dataclass(frozen=True)
class SineWeightedStarTransverseMode:
    """One scalar Hill equation, with its independent spatial multiplicity.

    In normalized time, y'' + alpha*(lambda + cos(Delta))*y = 0.
    ``frequency_bounds`` encloses sqrt(alpha*(lambda+1)). Trace bounds refer
    to one full collective period, which need not be the minimal coefficient
    period. The unit determinant is derived from the zero-trace first-order
    generator, not estimated from a computed monodromy matrix.
    ``rotation_error_upper_bound`` bounds the operator 2-norm error in
    coordinates (frequency*y, y'), and therefore also the half-trace error.
    ``strict_trace_margin`` is 1 minus the half-trace absolute upper bound.
    """

    laplacian_eigenvalue: int
    multiplicity: int
    alpha_bounds: I
    frequency_bounds: I
    rotation_angle_bounds: I | None
    reference_half_trace_bounds: I | None
    perturbation_integral_upper_bound: Q | None
    rotation_error_upper_bound: Q | None
    half_trace_bounds: I | None
    strict_trace_margin: Q | None
    trace_certified: bool
    linear_stability_certified: bool
    unavailable_reasons: tuple[str, ...]
    monodromy_determinant: Q = Q(1)


@dataclass(frozen=True)
class SineWeightedStarTransverseBand:
    """A conditional band certificate, including a separate equilibrium case.

    ``amplitude_bound`` is k_max for every 0 <= k <= k_max, not a measured
    phase or a preparation inferred from stored graph state. Period bounds
    cover nonzero librations and their k -> 0 limit. At k=0 the reference is
    an equilibrium: bounded transverse motion follows from its positive
    constant oscillator coefficients, without assigning it a base period.
    Exports rebuild every derived field from the two primitive parameters.
    """

    capacity_scale_bounds: tuple[Q, Q]
    amplitude_bound: Q
    normalized_clock_rate_bounds: I
    normalized_period_bounds: I | None
    original_period_bounds: I | None
    modes: tuple[SineWeightedStarTransverseMode, ...]
    linear_transverse_stability_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    equilibrium_transverse_stability_certified: bool = True
    nodes: tuple[int, ...] = NODES
    blocks: tuple[tuple[int, ...], ...] = BLOCKS
    edges: tuple[tuple[int, int], ...] = EDGES
    conductances: tuple[int, ...] = CONDUCTANCES
    bare_degrees: tuple[int, ...] = BARE_DEGREES
    capacity_family: str = CAPACITY_FAMILY
    law: str = LAW
    transverse_dimension: int = 6
    collective_dimension: int = 4
    original_clock: str = "t"
    normalized_clock: str = "tau=sqrt(2)*(s+8)*t/pi"
    period_scope: str = "nonzero_libration_period_and_zero_amplitude_limit"
    arithmetic_method: str = INTERVAL_METHOD
    exponential_method: str = "shared_rational_exp_unit_bounds"
    scope: tuple[str, ...] = (
        "supplied_fixed_five_node_support_conductances_and_positive_held_capacity_family",
        "zero_internal_offsets_and_collective_libration_about_consensus",
        "six_transverse_dimensions_separate_from_four_collective_dimensions",
        "determinant_one_and_strict_trace_bound_certify_bounded_linear_transverse_motion",
        "zero_amplitude_uses_positive_constant_oscillators_not_a_measured_period",
        "unresolved_sufficient_bounds_do_not_prove_instability",
        "no_nonlinear_error_radius_retention_attraction_formation_or_physical_identification",
        "no_graph_state_admission_runtime_operator_event_or_numerical_trajectory",
    )

    def to_dict(self):
        if type(self) is not SineWeightedStarTransverseBand:
            raise TypeError("a SineWeightedStarTransverseBand is required")
        fresh = assess_sine_weighted_star_transverse_band(
            capacity_scale_bounds=self.capacity_scale_bounds,
            amplitude_bound=self.amplitude_bound,
        )
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-weighted-star-transverse-band.v1",
            "report": _project(fresh),
        }


def _mode_bound(eigenvalue, multiplicity, scale, amplitude, period):
    # s/(s+8) is strictly increasing: preserve that correlation before
    # outward interval materialization instead of dividing independent boxes.
    ratios = tuple(s / (s + 8) for s in scale)
    alpha = I(
        Q(3 * eigenvalue + 2, 2) * ratios[0] ** 2,
        Q(3 * eigenvalue + 2, 2) * ratios[1] ** 2,
    )
    frequency = sqrt(alpha * (eigenvalue + 1))
    angle = reference = half_trace = None
    budget = error = margin = None
    reasons = []
    if period is None:
        reasons.append("period_denominator_unresolved_at_interval_precision")
    elif frequency.lo <= 0:
        reasons.append("positive_rotation_frequency_unresolved_at_interval_precision")
    else:
        angle = frequency * period
        reference = cos(angle)
        budget = 2 * frequency.hi * amplitude**2 * period.hi / (eigenvalue + 1)
        if budget > 1:
            reasons.append("rotation_error_exceeds_unit_exponential_work_domain")
        else:
            error = exp_unit_bounds(budget)[1] - 1
            half_trace = reference + I(-error, error)
            margin = 1 - half_trace.abs_max
            if margin <= 0:
                reasons.append("strict_monodromy_trace_margin_unresolved")
    trace_certified = margin is not None and margin > 0
    # A constant positive oscillator is stable even when a deliberately broad
    # parameter box cannot resolve the trace of an arbitrary return duration.
    stable = amplitude == 0 or trace_certified
    return SineWeightedStarTransverseMode(
        laplacian_eigenvalue=eigenvalue,
        multiplicity=multiplicity,
        alpha_bounds=alpha,
        frequency_bounds=frequency,
        rotation_angle_bounds=angle,
        reference_half_trace_bounds=reference,
        perturbation_integral_upper_bound=budget,
        rotation_error_upper_bound=error,
        half_trace_bounds=half_trace,
        strict_trace_margin=margin,
        trace_certified=trace_certified,
        linear_stability_certified=stable,
        unavailable_reasons=tuple(reasons),
    )


def assess_sine_weighted_star_transverse_band(
    *, capacity_scale_bounds=(Q(23, 25), Q(1)), amplitude_bound=Q(1, 4)
) -> SineWeightedStarTransverseBand:
    """Bound all infinitesimal transverse modes on a declared family band.

    The ordered scale endpoints must be finite and strictly positive;
    0 <= k_max < 1 excludes the pendulum separatrix and rotations. Rational
    parameters retain their exact values; other real inputs use shared
    represented-real admission. Wider valid bands may be unavailable under
    this sufficient estimate and fixed outward precision. The default band
    is the prospectively specified control, not a parameter scan.
    """
    values = _ordered(capacity_scale_bounds, "capacity_scale_bounds", limit=3)
    if len(values) != 2:
        raise ValueError("capacity_scale_bounds requires two ordered endpoints")
    scale = tuple(
        exact_or_represented_real(value, f"capacity_scale_bounds[{i}]")
        for i, value in enumerate(values)
    )
    if scale[0] <= 0 or scale[1] < scale[0]:
        raise ValueError("capacity_scale_bounds must be ordered and strictly positive")
    amplitude = exact_or_represented_real(amplitude_bound, "amplitude_bound")
    if not 0 <= amplitude < 1:
        raise ValueError("amplitude_bound must satisfy 0 <= k_max < 1")
    pi = pi_interval()
    clock_rate = sqrt(2) * I(scale[0] + 8, scale[1] + 8) / pi
    denominator = sqrt(I(1 - amplitude**2))
    if denominator.lo > 0:
        normalized_period = I((2 * pi).lo, (2 * pi / denominator).hi)
        original_period = normalized_period / clock_rate
    else:
        normalized_period = original_period = None
    modes = tuple(
        _mode_bound(eigenvalue, multiplicity, scale, amplitude, normalized_period)
        for eigenvalue, multiplicity in ((1, 2), (4, 1))
    )
    stable = all(mode.linear_stability_certified for mode in modes)
    reasons = tuple(
        f"lambda_{mode.laplacian_eigenvalue}:{reason}"
        for mode in modes
        if not mode.linear_stability_certified
        for reason in mode.unavailable_reasons
    )
    return SineWeightedStarTransverseBand(
        capacity_scale_bounds=scale,
        amplitude_bound=amplitude,
        normalized_clock_rate_bounds=clock_rate,
        normalized_period_bounds=normalized_period,
        original_period_bounds=original_period,
        modes=modes,
        linear_transverse_stability_certified=stable,
        status="certified_linear_transverse_stability" if stable else "unavailable",
        unavailable_reasons=reasons,
    )
