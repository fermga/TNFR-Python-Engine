"""Full edge-branch admission of the hypothetical maintained-parent UM reset.

The primitive source is re-admitted by the reset owner on every call. This
module neither installs its proposed support nor runs an operator or flow.
An acute chamber is an initial geometric domain, not an invariant set or a
stationary-existence certificate. The conditional enclosure retains the
separately assumed per-output allowance of the preceding reset admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, pi_interval
from .maintained_parent_um import UmResetPrediction, predict_parent_um


@dataclass(frozen=True)
class PhaseChamber:
    """Every oriented edge has delta = phase[right]-phase[left]-2*pi*shift."""

    phases: tuple[I, ...]
    shifts: tuple[int, ...]
    differences: tuple[I, ...]
    acute_margins: tuple[I, ...]
    minimum_acute_margin: I


@dataclass(frozen=True)
class FundamentalCycle:
    """One chord followed by its closing tree path, in canonical orientation.

    The chord coefficient is +1. Its gap sum is exactly
    ``-2*pi*branch_sum``, so ``winding`` is the negative branch sum.
    A dense integer row retains every edge, including zero coefficients.
    """

    chord_edge_index: int
    edge_coefficients: tuple[int, ...]
    branch_sum: int
    winding: int


@dataclass(frozen=True)
class SuccessorChamber:
    """Immutable full support, lift branches and complete cycle inventory.

    All edge indices use ``reset.nodes``. Conductances retain the same
    canonical edge order. The conditional arrays assume the proven support,
    exact unchanged coordinates and at most epsilon error in changed outputs;
    they do not establish an implementation error bound or an actual reset.
    """

    reset: UmResetPrediction
    edges: tuple[tuple[int, int], ...]
    conductances: tuple[I, ...]
    conditional_conductances: tuple[I, ...]
    conditional_capacities: tuple[I, ...]
    tree_edge_indices: tuple[int, ...]
    cycles: tuple[FundamentalCycle, ...]
    ideal: PhaseChamber
    conditional: PhaseChamber
    numerical_allowance: Q

    @property
    def cycle_rank(self):
        return len(self.cycles)


def _admit_edges(count, edges):
    """Require canonical simple support; do not normalize malformed labels."""
    if type(count) is not int or count < 2:
        raise ValueError("the chamber requires at least two nodes")
    edges = tuple(edges)
    for edge in edges:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(index) is not int for index in edge)
            or not 0 <= edge[0] < edge[1] < count
        ):
            raise ValueError("edges require canonical distinct node-index pairs")
    if edges != tuple(sorted(set(edges))):
        raise ValueError("edges require unique lexicographic order")
    return edges


def _certify_chamber(phases, edges):
    """Prove a unique acute branch from whole rational intervals.

    Candidate integer bounds use exact rational rounding, never a midpoint
    or binary64 phase. A branch touching either acute cut remains unsupported.
    Arbitrary integer lifts are allowed when their intervals resolve the cut.
    """
    phases = tuple(I.coerce(value) for value in phases)
    edges = _admit_edges(len(phases), edges)
    pi = pi_interval()
    shifts, differences, margins = [], [], []
    for left, right in edges:
        raw = phases[right] - phases[left]
        turns = raw / (2 * pi)
        lower = turns.lo - Q(1, 4)
        upper = turns.hi + Q(1, 4)
        first = -((-lower.numerator) // lower.denominator)
        last = upper.numerator // upper.denominator
        if first != last:
            raise ValueError("edge has no uniquely certified acute branch")
        difference = raw - 2 * first * pi
        margin = pi / 2 - abs(difference)
        if margin.lo <= 0:
            raise ValueError("edge has unresolved or nonpositive acute margin")
        shifts.append(first)
        differences.append(difference)
        margins.append(margin)
    if not margins:
        raise ValueError("the chamber requires nonempty support")
    return PhaseChamber(
        phases,
        tuple(shifts),
        tuple(differences),
        tuple(margins),
        I(min(value.lo for value in margins), min(value.hi for value in margins)),
    )


def _cycle_inventory(count, edges, shifts):
    """Build a deterministic spanning tree and all fundamental integer rows."""
    edges = _admit_edges(count, edges)
    shifts = tuple(shifts)
    if len(shifts) != len(edges) or any(type(value) is not int for value in shifts):
        raise ValueError("one exact integer shift is required for every edge")
    components = list(range(count))

    def component(node):
        while node != components[node]:
            node = components[node]
        return node

    tree, chords = [], []
    adjacency = [[] for _ in range(count)]
    for index, (left, right) in enumerate(edges):
        left_root, right_root = component(left), component(right)
        if left_root == right_root:
            chords.append(index)
            continue
        components[right_root] = left_root
        tree.append(index)
        adjacency[left].append((right, index, 1))
        adjacency[right].append((left, index, -1))
    if len(tree) != count - 1:
        raise ValueError("successor support must be connected")
    cycles = []
    for chord in chords:
        left, right = edges[chord]
        # Follow the chord left -> right and the unique tree path right -> left.
        pending = [(right, ())]
        visited = set()
        while pending:
            node, path = pending.pop()
            if node == left:
                break
            visited.add(node)
            for neighbor, edge_index, sign in adjacency[node]:
                if neighbor not in visited:
                    pending.append((neighbor, path + ((edge_index, sign),)))
        coefficients = [0] * len(edges)
        coefficients[chord] = 1
        for edge_index, sign in path:
            coefficients[edge_index] = sign
        branch_sum = sum(
            coefficient * shift
            for coefficient, shift in zip(coefficients, shifts, strict=True)
        )
        cycles.append(
            FundamentalCycle(chord, tuple(coefficients), branch_sum, -branch_sum)
        )
    return tuple(tree), tuple(cycles)


def predict_successor_chamber(graph):
    """Rebuild the default UM map and certify every hypothetical edge branch.

    Source provenance and preceding THOL admission belong to
    ``load_thol_successor``. Cached chamber reports are never accepted here as
    primitive premises. No root, attraction, maintenance or runtime is inferred.
    """
    reset = predict_parent_um(graph)
    epsilon = reset.numerical_allowance
    old = dict(zip(reset.old_edges, reset.old_conductances, strict=True))
    new = dict(zip(reset.new_edges, reset.new_conductances, strict=True))
    if old.keys() & new.keys():
        raise ValueError("old and proposed edges must be disjoint")
    edges = _admit_edges(len(reset.nodes), tuple(sorted(old.keys() | new.keys())))
    conductances = tuple(
        I.coerce(old[edge] if edge in old else new[edge]) for edge in edges
    )
    conditional_conductances = tuple(
        value if edge in old else value + I(-epsilon, epsilon)
        for edge, value in zip(edges, conductances, strict=True)
    )
    if any(value.lo <= 0 for value in conditional_conductances):
        raise ValueError("successor requires strictly positive conductances")
    conditional_capacities = tuple(
        (
            value + I(-epsilon, epsilon)
            if index in reset.capacity_changed_indices
            else value
        )
        for index, value in enumerate(reset.capacity_after)
    )
    if any(value.lo <= 0 for value in conditional_capacities):
        raise ValueError("successor requires strictly positive capacities")
    ideal = _certify_chamber(reset.phase_after, edges)
    conditional_phases = tuple(
        value + I(-epsilon, epsilon) if index in reset.phase_changed_indices else value
        for index, value in enumerate(reset.phase_after)
    )
    conditional = _certify_chamber(conditional_phases, edges)
    if conditional.shifts != ideal.shifts:
        raise ValueError("conditional phase enclosure changes the edge branch vector")
    tree, cycles = _cycle_inventory(len(reset.nodes), edges, ideal.shifts)
    return SuccessorChamber(
        reset,
        edges,
        conductances,
        conditional_conductances,
        conditional_capacities,
        tree,
        cycles,
        ideal,
        conditional,
        epsilon,
    )
