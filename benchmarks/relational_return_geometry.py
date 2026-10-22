"""Static acute geometry for two C5 rings with a supplied return path.

The mathematical existence/uniqueness and recovery claims belong to
theory/nodal/RELATIONAL_PATTERN_MEMORY.md. This instrument encloses the
opposite-winding root, reconstructs rational witnesses and evaluates the
native field once per witness. It advances no time and executes no live event.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction as Q

import networkx as nx

from tnfr.dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import (
    I,
    cos,
    pi_interval,
    sin,
)
from tnfr.physics.phase_cycle_geometry import (
    PhaseChordExtension,
    PhaseCycleState,
)
from tnfr.physics.phase_cycle_geometry import PhaseRootBracket as RootBracket
from tnfr.physics.phase_cycle_geometry import (
    _enclose_decreasing_phase_root,
    _return_path_nodal_affine_coefficients,
    _return_path_storage_residual,
    derive_phase_chord_extension,
    derive_phase_cycle_geometry,
    reconstruct_phase_cycle_state,
)

MODEL = RelationalExchangeModel(1.0)
RING_EDGES = tuple(
    (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
)
MEDIATED_EDGES = RING_EDGES + ((0, 10), (10, 5))
RETURN_EDGE = (1, 6)
NAMED_CYCLES = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (0, 10, 5, 6, 1))
CONNECTING_EDGES = ((0, 10), (10, 5), (6, 1))


@dataclass(frozen=True)
class GeometryWitness:
    """Exact circular witness plus a separate rounded native observation."""

    state: PhaseCycleState
    named_periods: tuple[Q, ...]
    connecting_turns: tuple[Q, ...]
    minimum_acute_margin_turns: Q
    native_field: RelationalExchangeField
    status: str

    @property
    def maximum_native_rate(self):
        return max(map(abs, self.native_field.form_rate + self.native_field.phase_rate))


@dataclass(frozen=True)
class ReturnGeometryReport:
    extension: PhaseChordExtension
    opposite_root: RootBracket
    same: GeometryWitness
    opposite: GeometryWitness
    opposite_phase_storage_excess: I
    scope: str = "static_supplied_support_no_event_or_trajectory"


def root_residual(turns: Q | I) -> I:
    """Enclose F(t)=cos(t/4)-sin(t)-sin(2t/3), with t=2*pi*turns."""
    return _return_path_storage_residual(turns, Q(0))


def enclose_opposite_root(*, refinements=40) -> RootBracket:
    """Bisect with strict outward-interval signs, rejecting unresolved signs.

    The 1..64 refinement budget is numerical policy, not a model parameter.
    No rounded transcendental sign or residual tolerance selects an endpoint.
    """
    return _enclose_decreasing_phase_root(
        root_residual, lower=Q(1, 12), upper=Q(1, 8), refinements=refinements
    )


def opposite_nodal_turns(s: Q) -> tuple[Q, ...]:
    """Affine circular family; sine balance additionally requires F(2*pi*s)=0."""
    if not isinstance(s, Q) or not Q(1, 12) <= s <= Q(1, 8):
        raise ValueError("opposite parameter must be an exact turn in [1/12, 1/8]")
    return tuple(a + b * s for a, b in _return_path_nodal_affine_coefficients())


def _gap(turns, left, right):
    return (turns[right] - turns[left] + Q(1, 2)) % 1 - Q(1, 2)


def _witness(graph, geometry, turns, *, status):
    state = reconstruct_phase_cycle_state(
        geometry, edge_turns=tuple(_gap(turns, i, j) for i, j in geometry.edges)
    )
    periods = tuple(
        sum((_gap(turns, i, j) for i, j in zip(cycle, cycle[1:] + cycle[:1])), Q(0))
        for cycle in NAMED_CYCLES
    )
    # Native binary64 radians and measured residuals are separate evidence
    # from the exact-turn state and certified ideal root interval.
    native = graph.copy()
    for node in native:
        native.nodes[node].update(
            EPI=0.0, nu_f=1.0, theta=math.tau * float(state.nodal_turns[node])
        )
    return GeometryWitness(
        state=state,
        named_periods=periods,
        connecting_turns=tuple(_gap(turns, i, j) for i, j in CONNECTING_EDGES),
        minimum_acute_margin_turns=Q(1, 4) - max(map(abs, state.edge_turns)),
        native_field=evaluate_relational_exchange(native, model=MODEL),
        status=status,
    )


def analyze_return_geometry(*, refinements=40) -> ReturnGeometryReport:
    """Reconstruct both sectors on held unit support without a time step."""
    root = enclose_opposite_root(refinements=refinements)
    before = nx.Graph()
    before.add_nodes_from(range(11))
    before.add_edges_from(MEDIATED_EDGES, weight=1.0)
    after = before.copy()
    after.add_edge(*RETURN_EDGE, weight=1.0)
    old_geometry = derive_phase_cycle_geometry(before)
    geometry = derive_phase_cycle_geometry(after)
    extension = derive_phase_chord_extension(old_geometry, geometry)
    same_turns = tuple(Q(node % 5, 5) if node < 10 else Q(0) for node in range(11))
    angle = 2 * pi_interval() * I(root.lower, root.upper)
    # Ideal structural storage, not physical energy or an occurrence law.
    opposite_storage = (
        8 * (1 - sin(angle / 4)) + 2 * (1 - cos(angle)) + 3 * (1 - cos(2 * angle / 3))
    )
    same_storage = 10 * (1 - cos(2 * pi_interval() / 5))
    return ReturnGeometryReport(
        extension=extension,
        opposite_root=root,
        same=_witness(
            after, geometry, same_turns, status="exact_turn_sine_balance_by_oddness"
        ),
        opposite=_witness(
            after,
            geometry,
            opposite_nodal_turns(root.midpoint),
            status="rational_midpoint_not_an_exact_equilibrium",
        ),
        opposite_phase_storage_excess=opposite_storage - same_storage,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, default=40)
    report = analyze_return_geometry(refinements=parser.parse_args().refinements)
    root = report.opposite_root
    print(f"Certified root turns: [{root.lower}, {root.upper}]")
    print(f"Root midpoint radians (rounded): {math.tau * float(root.midpoint):.16g}")
    excess = report.opposite_phase_storage_excess
    print(f"Certified opposite phase-storage excess: [{excess.lo}, {excess.hi}]")
    for name, witness in (("same", report.same), ("opposite", report.opposite)):
        print(f"{name}: {witness.status}; periods={witness.named_periods}")
        print(f"  connecting turns: {witness.connecting_turns}")
        print(f"  native maximum rate: {witness.maximum_native_rate:.8g}")
        print(
            f"  native phase storage: {float(witness.native_field.phase_storage):.16g}"
        )


if __name__ == "__main__":
    main()
