"""Static full-support bounds for the fixed two-ramp receiver enclosure.

The sign-reflection reduction belongs only to the nominal reference. Actual
source errors and velocity defects retain all 26 mean-zero phase directions.
No helper in this module evolves a trajectory or supplies source admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._organized_network_capture import _derive_organized_triangle_capture
from ._organized_network_compatibility import _EQUAL_CONTROL, _derive_organized_triangle

_REFERENCE_REPRESENTATIVES = tuple(range(9)) + tuple(range(18, 22))
_REFERENCE_REFLECTION = tuple(
    9 * ((1 - node // 9) % 3) + 8 - node % 9 for node in range(27)
)
_OSL_COSINE_BITS = 20
_OSL_BISECTIONS = 8
_OSL_RATE_BOUNDS = (Q(-1), Q(2))


@lru_cache(maxsize=1)
def _receiver_capture_geometry():
    """Retain immutable static geometry, never a source or trajectory verdict."""
    capture = _derive_organized_triangle_capture(mixing=_EQUAL_CONTROL)
    if not capture.target_certified or not capture.barrier_certified:
        raise ArithmeticError("fixed equal-support capture geometry is unavailable")
    return capture


@lru_cache(maxsize=1)
def _receiver_weighted_edges():
    """Reuse the immutable fixed support, checking its exact reflection."""
    edges = _derive_organized_triangle(mixing=_EQUAL_CONTROL).weighted_edges
    ordinary = sorted((min(a, b), max(a, b), w) for a, b, w in edges)
    reflected = sorted(
        (
            min(_REFERENCE_REFLECTION[a], _REFERENCE_REFLECTION[b]),
            max(_REFERENCE_REFLECTION[a], _REFERENCE_REFLECTION[b]),
            w,
        )
        for a, b, w in edges
    )
    if ordinary != reflected:
        raise ValueError("fixed receiver support does not preserve the reflection")
    return edges


def _phase_box(values):
    values = tuple(I.coerce(value) for value in values)
    if len(values) != 27:
        raise ValueError("receiver observations require all 27 phase coordinates")
    return values


def _nonnegative(value, name):
    value = _exact(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _lift_receiver_reference(values):
    """Lift 13 independent coordinates by the exact signed reflection.

    The fixed node (2,4) has value zero. Each of the other 13 pairs has
    opposite values, so every represented reference has global mean zero.
    Interval and same-order Taylor-jet inputs retain their uncertainty.
    """
    values = tuple(values)
    if len(values) != 13:
        raise ValueError("the sign-reflected receiver reference has dimension 13")
    if any(isinstance(value, Jet) for value in values):
        if not all(isinstance(value, Jet) for value in values):
            raise TypeError("reference coordinates must be all intervals or all jets")
        if len({value.order for value in values}) != 1:
            raise ValueError("reference Taylor jets must have a common order")
        zero = Jet.constant(0, values[0].order)
    else:
        values = tuple(I.coerce(value) for value in values)
        zero = I(0)
    result = [zero] * 27
    for node, value in zip(_REFERENCE_REPRESENTATIVES, values):
        result[node] = value
        result[_REFERENCE_REFLECTION[node]] = -value
    return tuple(result)


def _receiver_reference_flow(values):
    """Return the exact restricted field F=S/4 with generic interval/jet sine."""
    full = _lift_receiver_reference(values)
    is_jet = isinstance(full[0], Jet)
    zero = Jet.constant(0, full[0].order) if is_jet else I(0)
    sine = jet_sin if is_jet else sin
    rates = [zero] * 27
    for a, b, weight in _receiver_weighted_edges():
        current = sine(full[b] - full[a]) * (weight / 4)
        rates[a] = rates[a] + current
        rates[b] = rates[b] - current
    return tuple(rates[node] for node in _REFERENCE_REPRESENTATIVES)


@dataclass(frozen=True)
class _ReceiverLocalOSL:
    """Full mean-zero logarithmic bound on a supplied convex phase box."""

    phase_box: tuple[I, ...]
    phase_radius: Q
    edge_cosine_lower_bounds: tuple[Q, ...]
    growth_rate_upper_bound: Q
    bisections_performed: int
    certified: bool
    cosine_rounding_bits: int = _OSL_COSINE_BITS
    search_rate_bounds: tuple[Q, Q] = _OSL_RATE_BOUNDS
    phase_mean_leaf_dimension: int = 26


def _receiver_local_osl(phase_box, *, phase_radius):
    """Certify a signed growth upper bound using all 27 fine coordinates.

    Lower edge cosines give a Loewner upper Jacobian J. Taking the smaller
    cosine bound on each reflected edge pair makes J commute with the
    reflection. Its antisymmetric and mean-zero symmetric blocks have
    respective Gram matrices 2I and 2I+4*11^T; both are checked exactly.
    The search floor is not a lower spectral bound. No 13-coordinate
    reference symmetry is imposed on the actual error directions.
    """
    phases = _phase_box(phase_box)
    radius = _nonnegative(phase_radius, "phase_radius")
    edge_error = sqrt(I(2)).hi * radius
    scale = 1 << _OSL_COSINE_BITS
    jacobian = [[Q(0) for _ in range(27)] for _ in range(27)]
    edges = _receiver_weighted_edges()
    bounds = {}
    for a, b, _ in edges:
        lower = cos(phases[b] - phases[a] + I(-edge_error, edge_error)).lo
        lower = Q((lower.numerator * scale) // lower.denominator, scale)
        bounds[min(a, b), max(a, b)] = lower
    lower_cosines = []
    for a, b, weight in edges:
        reflected_a, reflected_b = _REFERENCE_REFLECTION[a], _REFERENCE_REFLECTION[b]
        lower = min(
            bounds[min(a, b), max(a, b)],
            bounds[min(reflected_a, reflected_b), max(reflected_a, reflected_b)],
        )
        lower_cosines.append(lower)
        coefficient = weight * lower / 4
        jacobian[a][a] -= coefficient
        jacobian[b][b] -= coefficient
        jacobian[a][b] += coefficient
        jacobian[b][a] += coefficient
    anti = tuple(
        tuple(
            jacobian[a][b]
            - jacobian[a][_REFERENCE_REFLECTION[b]]
            - jacobian[_REFERENCE_REFLECTION[a]][b]
            + jacobian[_REFERENCE_REFLECTION[a]][_REFERENCE_REFLECTION[b]]
            for b in _REFERENCE_REPRESENTATIVES
        )
        for a in _REFERENCE_REPRESENTATIVES
    )
    fixed = 22
    symmetric = tuple(
        tuple(
            jacobian[a][b]
            + jacobian[a][_REFERENCE_REFLECTION[b]]
            + jacobian[_REFERENCE_REFLECTION[a]][b]
            + jacobian[_REFERENCE_REFLECTION[a]][_REFERENCE_REFLECTION[b]]
            - 2
            * (
                jacobian[a][fixed]
                + jacobian[_REFERENCE_REFLECTION[a]][fixed]
                + jacobian[fixed][b]
                + jacobian[fixed][_REFERENCE_REFLECTION[b]]
            )
            + 4 * jacobian[fixed][fixed]
            for b in _REFERENCE_REPRESENTATIVES
        )
        for a in _REFERENCE_REPRESENTATIVES
    )

    def passes(rate):
        return all(
            exact_symmetric_semidefinite(
                tuple(
                    tuple(
                        rate * (2 * int(i == j) + offset) - block[i][j]
                        for j in range(13)
                    )
                    for i in range(13)
                )
            )
            for block, offset in ((anti, 0), (symmetric, 4))
        )

    low, high = _OSL_RATE_BOUNDS
    if not passes(high):
        raise ArithmeticError("fixed global degree-four Jacobian bound failed")
    for _ in range(_OSL_BISECTIONS):
        middle = (low + high) / 2
        if passes(middle):
            high = middle
        else:
            low = middle
    return _ReceiverLocalOSL(
        phase_box=phases,
        phase_radius=radius,
        edge_cosine_lower_bounds=tuple(lower_cosines),
        growth_rate_upper_bound=high,
        bisections_performed=_OSL_BISECTIONS,
        certified=True,
    )


def _target_edge_integer(a, b):
    """Return fixed edge registration to alpha*(j-4) in all three regions."""
    return ((b % 9) - (a % 9)) // 9


def _potential_and_gradient(phases):
    potential, gradient = I(0), [I(0)] * 27
    for a, b, weight in _receiver_weighted_edges():
        difference = phases[b] - phases[a]
        potential = potential + weight * (1 - cos(difference))
        current = weight * sin(difference)
        gradient[a] = gradient[a] - current
        gradient[b] = gradient[b] + current
    gradient_norm = sqrt(I(sum(value.abs_max**2 for value in gradient))).hi
    return potential, gradient_norm


def _observed_winding(phases, region, edge_error):
    pi = pi_interval()
    integers = []
    for j in range(9):
        difference = phases[9 * region + (j + 1) % 9] - phases[9 * region + j]
        ratio = difference.midpoint / (2 * pi.midpoint) + Q(1, 2)
        integer = ratio.numerator // ratio.denominator
        principal = difference - 2 * integer * pi
        if principal.abs_max + edge_error >= pi.lo:
            return None
        integers.append(integer)
    return -sum(integers)


@dataclass(frozen=True)
class _ReceiverTubeObservation:
    """Static inequalities, without source, transit or capture provenance."""

    phase_box: tuple[I, ...]
    phase_error_bound: Q
    form_norm_upper_bound: Q
    donor_registered_branch_margins: tuple[Q, Q]
    windings: tuple[int | None, ...]
    complete_reference_potential_bounds: I
    reference_gradient_norm_upper_bound: Q
    target_potential_bounds: I
    target_registered_gap_upper_bound: Q
    capture_cell_margin: Q
    complete_capture_excess_upper_bound: Q
    capture_energy_margin: Q
    capture_inequalities_pass: bool


def _observe_receiver_tube(phase_box, *, phase_error_bound, form_norm_upper_bound):
    """Observe a full reference box plus a separate Euclidean error ball.

    Radii are primitive bounds, not evidence that a trajectory reaches this
    box. The caller must retain source, whole-interval and conserved-mean
    premises. Reference potential and its nonzero linear gradient term are
    both computed on the complete weighted support.
    """
    phases = _phase_box(phase_box)
    radius = _nonnegative(phase_error_bound, "phase_error_bound")
    form = _nonnegative(form_norm_upper_bound, "form_norm_upper_bound")
    pi = pi_interval()
    edge_error = sqrt(I(2)).hi * radius
    donor_margins = []
    for region in range(2):
        gaps = tuple(
            phases[9 * region + (j + 1) % 9]
            - phases[9 * region + j]
            + (2 * pi if j == 8 else 0)
            for j in range(9)
        )
        donor_margins.append(pi.lo - max(gap.abs_max for gap in gaps) - edge_error)
    windings = tuple(
        _observed_winding(phases, region, edge_error) for region in range(3)
    )
    potential, gradient_norm = _potential_and_gradient(phases)
    capture = _receiver_capture_geometry()
    target_potential = (
        capture.geometry.internal_storage_bounds
        + capture.geometry.aligned_attachment_work_bounds
    )
    max_gap = max(
        (phases[b] - phases[a] - 2 * _target_edge_integer(a, b) * pi).abs_max
        for a, b, _ in _receiver_weighted_edges()
    )
    cell_margin = capture.cell_edge_gap_bound - max_gap - edge_error
    excess = (
        potential.hi
        - target_potential.lo
        + gradient_norm * radius
        + 4 * radius**2
        + 4 * form**2
    )
    energy_margin = capture.boundary_energy_lower_bound - excess
    return _ReceiverTubeObservation(
        phase_box=phases,
        phase_error_bound=radius,
        form_norm_upper_bound=form,
        donor_registered_branch_margins=tuple(donor_margins),
        windings=windings,
        complete_reference_potential_bounds=potential,
        reference_gradient_norm_upper_bound=gradient_norm,
        target_potential_bounds=target_potential,
        target_registered_gap_upper_bound=max_gap,
        capture_cell_margin=cell_margin,
        complete_capture_excess_upper_bound=excess,
        capture_energy_margin=energy_margin,
        capture_inequalities_pass=cell_margin > 0 and energy_margin > 0,
    )
