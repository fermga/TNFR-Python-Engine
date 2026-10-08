"""Actual-family geometry, work and identity bounds on unit C9 contact graphs.

Private callers normalize primitives and freshly prove the unprobed handoff.
These bounds do not admit a supplied report, install a contact or establish a
trajectory approximation. Full sine and reduced tracking consumers share the
same original-family geometric premises.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sqrt


@dataclass(frozen=True)
class _JoinedPortBounds:
    supported: bool
    joined_gap_lower_bound: Q | None
    joined_cosine_bounds: I
    joined_barrier_lower_bound: Q | None
    joined_radius_squared_upper_bound: Q | None
    joined_excess_storage_upper_bound: Q | None
    joined_radius_margin_bounds: I | None
    joined_storage_margin_bounds: I | None
    contact_work_upper_bound: Q | None
    work_margin_bounds: I | None
    joined_form_mean_bounds: I | None
    joined_phase_mean_bounds: I | None
    identity_certified: bool
    work_within_allowance: bool


def _joined_port_bounds(
    *,
    geometry,
    phase_origins,
    endpoint_radius,
    radius,
    work_allowance,
    handoff_available,
) -> _JoinedPortBounds:
    n, edges = geometry.component_count, geometry.contacts
    eps, r, origins = endpoint_radius, radius, phase_origins
    cosine = cos(4 * pi_interval() / 9 + sqrt(I(2)) * r)
    supported = n >= 2 and geometry.connected
    gap = Q(4, 9 * n * (geometry.contact_diameter + 8)) if supported else None
    barrier = gap * cosine.lo * r**2 / 2 if gap is not None and cosine.lo > 0 else None
    z2 = energy = radius_margin = storage_margin = None
    work = work_margin = mean_x = mean_phase = None
    identity = allowed = False
    if handoff_available:
        origin_mean = sum(origins, Q(0)) / n
        spread = sum(((value - origin_mean) ** 2 for value in origins), Q(0))
        z2 = 2 * n * eps**2 + 9 * spread
        phase_cost = sum(
            ((abs(origins[j] - origins[i]) + 2 * eps) ** 2 / 2 for i, j in edges), Q(0)
        )
        energy = (4 + max(geometry.contact_degrees)) * n * eps**2 + phase_cost
        radius_margin = I(r**2 - z2)
        storage_margin = I(barrier - energy) if barrier is not None else None
        identity = bool(
            storage_margin is not None
            and radius_margin.lo > 0
            and storage_margin.lo > 0
        )
        work = 2 * len(edges) * eps**2 + phase_cost
        work_margin = I(work_allowance - work)
        allowed = work <= work_allowance
        mass = 18 * n + 2 * len(edges)
        mean_error = Q(2 * len(edges), mass) * eps
        mean_x = I(-mean_error, mean_error)
        center = (
            sum(
                (
                    (18 + degree) * origin
                    for degree, origin in zip(geometry.contact_degrees, origins)
                ),
                Q(0),
            )
            / mass
        )
        mean_phase = I(center - mean_error, center + mean_error)
    return _JoinedPortBounds(
        supported=supported,
        joined_gap_lower_bound=gap,
        joined_cosine_bounds=cosine,
        joined_barrier_lower_bound=barrier,
        joined_radius_squared_upper_bound=z2,
        joined_excess_storage_upper_bound=energy,
        joined_radius_margin_bounds=radius_margin,
        joined_storage_margin_bounds=storage_margin,
        contact_work_upper_bound=work,
        work_margin_bounds=work_margin,
        joined_form_mean_bounds=mean_x,
        joined_phase_mean_bounds=mean_phase,
        identity_certified=identity,
        work_within_allowance=allowed,
    )
