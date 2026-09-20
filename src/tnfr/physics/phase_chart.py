"""Certified common semicircle lifts of represented nodal phases.

The circle uses mathematical pi, with the existing rational enclosure as
its numerical owner. This detached observation neither computes a circular
mean nor certifies a phase update, gate, controller or runtime trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA
from ..mathematics._phase_midpoint import _affine_interval, _pi_bounds
from .phase_cycle_geometry import PhaseCycleGeometry, derive_phase_cycle_geometry

__all__ = ["CommonPhaseChartObservation", "observe_common_phase_chart"]

Interval = tuple[Fraction, Fraction]
AffinePi = tuple[Fraction, int]


@dataclass(frozen=True)
class CommonPhaseChartObservation:
    """An exact-real common lift or an explicit exclusion/abstention.

    ``phase`` contains exact rational values of materialized binary64 radians.
    Affine values mean ``rational + coefficient*pi`` with mathematical pi.
    Cyclic gaps follow ``sorted_indices``; an omitted gap is the unique gap
    proved wider than pi. Lift arrays instead follow ``nodes``. Every lift
    is ``phase[i]+2*pi*lift_turns[i]`` in one common interval narrower than pi.

    ``excluded`` proves that no open common semicircle contains these phases.
    ``unresolved`` means the finite pi enclosure could not decide that fact.
    Neither outcome alone determines graph winding: a winding-zero field may
    fail common-chart admission. On admission all fundamental cycle periods
    telescope exactly to zero, hence every support cycle has zero winding.
    """

    geometry: PhaseCycleGeometry
    nodes: tuple
    phase: tuple[Fraction, ...]
    sorted_indices: tuple[int, ...]
    circular_gap_affine: tuple[AffinePi, ...]
    circular_gap_enclosures: tuple[Interval, ...]
    status: str
    omitted_gap_index: int | None
    lift_turns: tuple[int, ...] | None
    lift_enclosures: tuple[Interval, ...] | None
    diameter_affine: AffinePi | None
    diameter_enclosure: Interval | None
    semicircle_margin_enclosure: Interval | None
    cycle_periods: tuple[int, ...] | None
    zero_cycle_winding: bool | None
    scope: tuple[str, ...] = (
        "exact_real_circle_for_captured_materialized_radian_values",
        "shared_rational_pi_enclosure_without_trigonometric_mean_or_tolerance",
        "shared_simple_connected_phase_support_geometry_and_evaluation_limits",
        "common_lift_implies_zero_winding_on_every_support_cycle",
        "no_U3_capacity_pressure_or_transport_weight_admission",
        "no_update_convex_hull_binary64_trajectory_or_autonomous_formation_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether a common open semicircle is proved for this observation."""
        return self.status == "admitted"


def observe_common_phase_chart(graph) -> CommonPhaseChartObservation:
    """Construct a common lift with diameter strictly below mathematical pi.

    Read explicit scalar phase aliases in the engine's canonical represented
    range ``[0,math.tau)``. Sorting their exact rational values gives circular
    empty gaps. A finite set lies in an open semicircle precisely when one
    empty gap is wider than pi; cut immediately after that gap and lift lower
    represented angles by true ``2*pi``. All comparisons use outward rational
    pi intervals, with no empirical tolerance or normalization of the input.

    Equality at a mathematical diameter of pi is excluded, not admitted by
    tolerance. Pure rational radian inputs cannot differ by exactly irrational
    pi: in particular ``(0,math.pi)`` is slightly narrower than a semicircle,
    not an exact antipodal pair. The finite enclosure may still abstain on an
    undecided comparison; availability is never replaced by a zero result.

    Topology validation and cycle bases reuse ``derive_phase_cycle_geometry``:
    connected simple undirected support, 2 to 32 nodes and at most 50 edges.
    Those resource limits are not restrictions of the common-chart theorem.
    Conductance, capacity, pressure and the configured U3 gate are irrelevant
    to this geometric observation and are not admitted or modified.
    """
    geometry = derive_phase_cycle_geometry(graph)
    nodes = geometry.nodes
    phases = []
    for node in nodes:
        raw = get_attr(
            graph.nodes[node],
            ALIAS_THETA,
            strict=True,
            conv=lambda value: value,
        )
        represented, exact = finite_represented_real(raw, "nodal phase")
        if not 0.0 <= represented < math.tau:
            raise ValueError("nodal phases must lie in canonical [0,math.tau)")
        phases.append(exact)
    phase = tuple(phases)
    size = len(nodes)
    ordered = tuple(sorted(range(size), key=lambda index: (phase[index], index)))
    gaps = tuple(
        (
            phase[ordered[(position + 1) % size]] - phase[index],
            2 if position == size - 1 else 0,
        )
        for position, index in enumerate(ordered)
    )
    pi_bounds = _pi_bounds()
    enclosures = tuple(_affine_interval(*gap, pi_bounds) for gap in gaps)
    margins = tuple(
        _affine_interval(rational, coefficient - 1, pi_bounds)
        for rational, coefficient in gaps
    )
    candidates = tuple(index for index, margin in enumerate(margins) if margin[0] > 0)
    if len(candidates) > 1:
        raise RuntimeError("two disjoint circular gaps cannot both exceed pi")
    omitted = candidates[0] if candidates else None
    status = (
        "admitted"
        if candidates
        else ("excluded" if all(margin[1] <= 0 for margin in margins) else "unresolved")
    )
    lifts = lift_enclosures = diameter = diameter_enclosure = margin_enclosure = None
    periods = zero_winding = None
    if omitted is not None:
        cut = phase[ordered[(omitted + 1) % size]]
        lifts = tuple(int(value < cut) for value in phase)
        lift_enclosures = tuple(
            _affine_interval(value, 2 * turn, pi_bounds)
            for value, turn in zip(phase, lifts)
        )
        rational, coefficient = gaps[omitted]
        diameter = (-rational, 2 - coefficient)
        diameter_enclosure = _affine_interval(*diameter, pi_bounds)
        margin_enclosure = margins[omitted]
        # Differences of one common lift are already strictly within (-pi,pi).
        # The shared integer fundamental cycles then telescope in both the
        # rational and pi coordinates; no rounded phase sum defines a period.
        edge_differences = tuple(
            (phase[right] - phase[left], 2 * (lifts[right] - lifts[left]))
            for left, right in geometry.edges
        )
        periods_list = []
        for row in geometry.cycle_rows:
            rational_period = sum(
                (sign * value[0] for sign, value in zip(row, edge_differences)),
                Fraction(0),
            )
            pi_period = sum(
                sign * value[1] for sign, value in zip(row, edge_differences)
            )
            if rational_period or pi_period:
                raise RuntimeError(
                    "a common lift lost its exact cycle telescoping identity"
                )
            periods_list.append(pi_period // 2)
        periods = tuple(periods_list)
        zero_winding = True
    return CommonPhaseChartObservation(
        geometry=geometry,
        nodes=nodes,
        phase=phase,
        sorted_indices=ordered,
        circular_gap_affine=gaps,
        circular_gap_enclosures=enclosures,
        status=status,
        omitted_gap_index=omitted,
        lift_turns=lifts,
        lift_enclosures=lift_enclosures,
        diameter_affine=diameter,
        diameter_enclosure=diameter_enclosure,
        semicircle_margin_enclosure=margin_enclosure,
        cycle_periods=periods,
        zero_cycle_winding=zero_winding,
    )
