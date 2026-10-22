"""Exact resultant-chart and winding observations on a supplied pure cycle.

Inputs are declared rational turns, meaning angle divided by mathematical
``2*pi``. They are not inferred from measured or materialized radians. The
skip-two cycles are an observational graph derived from the existing support;
they add no couplings and execute no phase law.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Rational
from typing import Any

from .._exact_time import exact_or_represented_real
from .phase_cycle_geometry import PhaseCycleGeometry, derive_phase_cycle_geometry

__all__ = ("CycleResultantSector", "derive_cycle_resultant_sector")

_HALF = Fraction(1, 2)
_QUARTER = Fraction(1, 4)


@dataclass(frozen=True)
class CycleResultantSector:
    """Detached exact geometry in the caller's oriented cycle coordinates.

    ``phase_turns`` lie in [0,1); ``edge_turns`` lie in [-1/2,1/2) and
    follow each node to its successor. An antipodal edge makes that cycle's
    winding unavailable, without automatically making a resultant singular.
    Auxiliary cycles follow every second node: one for odd size, two for even
    size. Their winding is unavailable precisely at a zero neighbor resultant.

    ``regular_phase_chart`` also excludes a nonzero negative-real resultant
    relative to the central phase. ``resultant_sector_available`` only asks
    whether all auxiliary windings are defined, so these properties differ.
    """

    geometry: PhaseCycleGeometry
    cycle_nodes: tuple[Any, ...]
    phase_turns: tuple[Fraction, ...]
    edge_turns: tuple[Fraction, ...]
    support_winding: int | None
    auxiliary_cycles: tuple[tuple[Any, ...], ...]
    auxiliary_winding: tuple[int | None, ...]
    zero_resultant_nodes: tuple[Any, ...]
    negative_real_resultant_nodes: tuple[Any, ...]
    strict_acute_edges: bool
    scope: tuple[str, ...] = (
        "supplied_connected_simple_undirected_pure_cycle_with_at_least_five_nodes",
        "shared_topology_owner_resource_budget_is_not_a_physical_restriction",
        "declared_exact_rational_turns_not_inferred_from_binary64_radians",
        "exact_zero_and_negative_real_relative_resultant_decisions_without_trigonometry",
        "skip_two_cycles_are_observations_not_added_support_or_coupling",
        "support_winding_branch_and_resultant_chart_boundaries_are_distinct",
        "no_graph_write_capacity_clock_evolution_energy_threshold_or_executor_extension",
        "no_trajectory_sector_generation_formation_or_physical_identity_certificate",
    )

    @property
    def regular_phase_chart(self) -> bool:
        """Whether every relative resultant is nonzero and off its Arg cut."""
        return not self.zero_resultant_nodes and not self.negative_real_resultant_nodes

    @property
    def resultant_sector_available(self) -> bool:
        """Whether every derived cycle has a well-defined exact winding."""
        return all(value is not None for value in self.auxiliary_winding)


def _wrap(turn: Fraction) -> Fraction:
    return (turn + _HALF) % 1 - _HALF


def _winding(turns: tuple[Fraction, ...]) -> int | None:
    edges = tuple(
        _wrap(right - left) for left, right in zip(turns, turns[1:] + turns[:1])
    )
    if -_HALF in edges:
        return None
    total = sum(edges, Fraction(0))
    # Every wrapped edge differs from its exact nodal difference by an integer.
    if total.denominator != 1:
        raise ArithmeticError("exact circular nodal differences must close integrally")
    return total.numerator


def _cycle_order(geometry: PhaseCycleGeometry, cycle_nodes) -> tuple[Any, ...]:
    if isinstance(cycle_nodes, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("cycle_nodes must be an ordered iterable")
    try:
        cycle = tuple(cycle_nodes)
    except TypeError as exc:
        raise TypeError("cycle_nodes must be an ordered iterable") from exc
    try:
        distinct = set(cycle)
    except TypeError as exc:
        raise ValueError("cycle_nodes must contain support node labels") from exc
    if len(cycle) != len(geometry.nodes) or distinct != set(geometry.nodes):
        raise ValueError("cycle_nodes must contain the full support exactly once")
    positions = {node: index for index, node in enumerate(geometry.nodes)}
    expected = {
        tuple(sorted((positions[left], positions[right])))
        for left, right in zip(cycle, cycle[1:] + cycle[:1])
    }
    if expected != set(geometry.edges):
        raise ValueError(
            "support must be exactly the supplied ordered cycle without chords"
        )
    return cycle


def derive_cycle_resultant_sector(
    graph, *, cycle_nodes, phase_turns: Mapping
) -> CycleResultantSector:
    """Classify a pure cycle's exact resultant chart and two winding read-outs.

    The mapping must cover exactly the support and contain ``Rational`` values
    excluding Booleans; floats are rejected even when numerically integral.
    Values are normalized modulo one without choosing a common phase origin.
    Graph attributes, including live phases and transport weights, are unused.

    For the two outgoing wrapped turns ``a,b`` at a node, the resultant is
    zero exactly when ``(a-b) mod 1 = 1/2``. Otherwise it is negative real
    relative to the node exactly when ``(a+b) mod 1 = 0`` and ``abs(a)>1/4``.
    These tests retain antipodal-edge and resultant boundaries separately;
    neither a regular result nor an available winding admits a runtime step.
    """
    geometry = derive_phase_cycle_geometry(graph)
    if len(geometry.nodes) < 5:
        raise ValueError("resultant-sector observation requires at least five nodes")
    cycle = _cycle_order(geometry, cycle_nodes)
    if not isinstance(phase_turns, Mapping):
        raise TypeError("phase_turns must map the exact full support to rational turns")
    if set(phase_turns) != set(geometry.nodes):
        raise ValueError("phase_turns must match the exact full support")
    turns = []
    for node in cycle:
        raw = phase_turns[node]
        if isinstance(raw, bool) or not isinstance(raw, Rational):
            raise TypeError(
                "phase turns must be exact Rational values excluding Booleans"
            )
        turns.append(exact_or_represented_real(raw, "phase turn") % 1)
    turns = tuple(turns)
    edge_turns = tuple(
        _wrap(right - left) for left, right in zip(turns, turns[1:] + turns[:1])
    )
    zero, negative = [], []
    for index, node in enumerate(cycle):
        left = _wrap(turns[index - 1] - turns[index])
        right = edge_turns[index]
        if (left - right) % 1 == _HALF:
            zero.append(node)
        elif (left + right) % 1 == 0 and abs(left) > _QUARTER:
            negative.append(node)
    size = len(cycle)
    starts = range(1 if size % 2 else 2)
    length = size if size % 2 else size // 2
    auxiliary_indices = tuple(
        tuple((start + 2 * offset) % size for offset in range(length))
        for start in starts
    )
    return CycleResultantSector(
        geometry=geometry,
        cycle_nodes=cycle,
        phase_turns=turns,
        edge_turns=edge_turns,
        support_winding=_winding(turns),
        auxiliary_cycles=tuple(
            tuple(cycle[index] for index in row) for row in auxiliary_indices
        ),
        auxiliary_winding=tuple(
            _winding(tuple(turns[index] for index in row)) for row in auxiliary_indices
        ),
        zero_resultant_nodes=tuple(zero),
        negative_real_resultant_nodes=tuple(negative),
        strict_acute_edges=all(abs(turn) < _QUARTER for turn in edge_turns),
    )
