"""Source-independent targets and full-state barriers for the two C9 triangles.

ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md owns this static theorem. A valid
barrier supplies no initial state, acquisition, entry or recovery verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._exact_linear_algebra import exact_matrix_inverse
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._organized_network_compatibility import _EQUAL_CONTROL as _EQUAL
from ._organized_network_compatibility import _UNEQUAL_CONTROL as _UNEQUAL
from ._organized_network_compatibility import (
    _derive_organized_triangle,
    _mixings,
    _OrganizedTriangle,
)

_TARGET_GAP = Q(22, 25)
_CELL_GAP = Q(2)
_BREGMAN = Q(3, 20)
_RESISTANCE = Q(3, 5)
_LAPLACIAN_GAP = Q(2, 5)


def _edge_resistances(geometry):
    """Exact grounded solve on the actual full support, without projection."""
    laplacian = [[Q(0) for _ in range(27)] for _ in range(27)]
    for a, b, weight in geometry.weighted_edges:
        laplacian[a][a] += weight
        laplacian[b][b] += weight
        laplacian[a][b] -= weight
        laplacian[b][a] -= weight
    inverse = exact_matrix_inverse(tuple(tuple(row[:26]) for row in laplacian[:26]))

    def entry(a, b):
        return inverse[a][b] if a < 26 and b < 26 else Q(0)

    return tuple(
        entry(a, a) + entry(b, b) - 2 * entry(a, b)
        for a, b, _ in geometry.weighted_edges
    )


def _target(geometry):
    """Certify the implicit target; bracket endpoints are never target values."""
    alpha = 2 * pi_interval() / 9
    if geometry.mixing == _EQUAL:
        return None, None, None, (I(0),) * 3, (I(0),) * 3, I(0), True
    high = geometry.contacts[0].amplitude_bounds
    low = geometry.contacts[1].amplitude_bounds
    phase = geometry.contacts[0].phase_origin_shift_bounds

    def current_difference(a):
        return high * sin(I(a)) - low * sin(pi_interval() / 3 - 2 * a)

    left, right = Q(347, 1000), Q(348, 1000)
    original = I(left, right)
    derivative = (
        high * cos(original) + 2 * low * cos(pi_interval() / 3 - 2 * original)
    ).lo
    bracket_ok = current_difference(left).hi < 0 < current_difference(right).lo
    if bracket_ok and derivative > 0:
        for _ in range(32):
            middle = (left + right) / 2
            value = current_difference(middle)
            if value.hi < 0:
                left = middle
            elif value.lo > 0:
                right = middle
            else:
                break  # Retain the proved bracket, never choose an unresolved sign.
    signs = (current_difference(left), current_difference(right))
    a = I(left, right)
    b = pi_interval() / 3 - 2 * a
    gaps = (a - phase, b - alpha / 2, a - alpha + phase)
    raw = (I(0), gaps[0], gaps[0] + gaps[1])
    mean = sum(raw, I(0)) / 3
    origins = tuple(value - mean for value in raw)
    excess = 9 * (2 * high * cos(a) + low * cos(b) - Q(3, 2) * (1 + cos(alpha)))
    certified = bracket_ok and derivative > 0 and signs[0].hi < 0 < signs[1].lo
    return a, signs, derivative, origins, gaps, excess, certified


@dataclass(frozen=True)
class _OrganizedTriangleCapture:
    """Conditional geometry shared by separately admitted source families."""

    geometry: _OrganizedTriangle
    target_root_bounds: I | None
    target_root_endpoint_sign_bounds: tuple[I, I] | None
    target_root_derivative_lower_bound: Q | None
    target_phase_origins_bounds: tuple[I, ...]
    target_contact_gaps_bounds: tuple[I, ...]
    target_fine_gap_bounds: tuple[I, ...]
    target_hessian_lower_bound: Q
    laplacian_gap_lower_bound: Q
    edge_effective_resistances: tuple[Q, ...]
    bregman_coefficient_bounds: I
    nominal_excess_bounds: I
    aligned_gradient_norm_bounds: I
    boundary_energy_lower_bound: Q
    joint_coercivity_lower_bound: Q
    cell_edge_gap_bound: Q
    target_edge_gap_bound: Q
    target_certified: bool
    barrier_certified: bool


def _derive_organized_triangle_capture(*, mixing):
    """Rebuild the complete weighted target and barrier from primitive weights."""
    weights = _mixings(mixing)
    if weights not in (_EQUAL, _UNEQUAL):
        raise ValueError("capture geometry requires one of the two fixed controls")
    geometry = _derive_organized_triangle(mixing=weights)
    root, signs, derivative, origins, gaps, excess, root_ok = _target(geometry)
    alpha = 2 * pi_interval() / 9
    gap_by_contact = dict(zip(geometry.contact_order, gaps))
    fine_gaps = tuple(
        (
            alpha
            if a // 9 == b // 9
            else (gap_by_contact[(a // 9, b // 9)] + ((b - a) % 9) * alpha)
        )
        for a, b, _ in geometry.weighted_edges
    )
    target_ok = root_ok and max(value.abs_max for value in fine_gaps) < _TARGET_GAP
    resistances = _edge_resistances(geometry)
    width = _CELL_GAP - _TARGET_GAP
    coefficient = (
        cos(I(_TARGET_GAP)) - cos(I(_CELL_GAP)) - width * sin(I(_TARGET_GAP))
    ) / width**2
    barrier = _BREGMAN * width**2 / _RESISTANCE
    barrier_ok = (
        coefficient.lo > _BREGMAN
        and 0 < min(resistances) <= max(resistances) < _RESISTANCE
        and _CELL_GAP + _TARGET_GAP < pi_interval().lo
    )
    divergence = [Q(0)] * 3
    for (a, b), weight in zip(geometry.contact_order, weights):
        divergence[a] += weight
        divergence[b] -= weight
    gradient = 3 * sin(alpha) * sqrt(I(sum(value**2 for value in divergence)))
    return _OrganizedTriangleCapture(
        geometry=geometry,
        target_root_bounds=root,
        target_root_endpoint_sign_bounds=signs,
        target_root_derivative_lower_bound=derivative,
        target_phase_origins_bounds=origins,
        target_contact_gaps_bounds=gaps,
        target_fine_gap_bounds=fine_gaps,
        target_hessian_lower_bound=cos(I(_TARGET_GAP)).lo * _LAPLACIAN_GAP,
        laplacian_gap_lower_bound=_LAPLACIAN_GAP,
        edge_effective_resistances=resistances,
        bregman_coefficient_bounds=coefficient,
        nominal_excess_bounds=excess,
        aligned_gradient_norm_bounds=gradient,
        boundary_energy_lower_bound=barrier,
        joint_coercivity_lower_bound=_BREGMAN * _LAPLACIAN_GAP,
        cell_edge_gap_bound=_CELL_GAP,
        target_edge_gap_bound=_TARGET_GAP,
        target_certified=target_ok,
        barrier_certified=barrier_ok,
    )
