"""Exact obstruction to a unique equivariant choice under a supplied action.

For a state X and deterministic equivariant selector s, every permutation g
fixing X must fix s(X): s(X) = s(gX) = g s(X). Thus a nonempty admissible
candidate set without a stabilizer-fixed vertex cannot supply such a choice.
Optional exact relation labels include declared support in that stabilizer.
This finite statement authenticates neither a live graph nor the complete
runtime state, histories, phase gauge, operator maps or birth support.

The graph symmetry helpers enumerate graph automorphisms with optional numeric
edge matching and a truncation cap. Here callers instead supply a finite group
of exact permutations; closure is checked before its state stabilizer is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = ["SelectorSymmetryObstruction", "derive_selector_symmetry"]


@dataclass(frozen=True)
class SelectorSymmetryObstruction:
    """Detached finite-action result, not a sealed runtime certificate.

    ``fixed_candidates`` enumerates choices not excluded by this necessary
    condition. One or several such vertices do not derive a selector or prove
    its uniqueness on any larger state space.
    """

    state_labels: tuple[object, ...]
    relation_labels: tuple[tuple[object, ...], ...] | None
    permutations: tuple[tuple[int, ...], ...]
    candidates: tuple[int, ...]
    stabilizer_permutations: tuple[tuple[int, ...], ...]
    candidate_orbits: tuple[tuple[int, ...], ...]
    fixed_candidates: tuple[int, ...]
    validation_work: int
    max_nodes: int
    max_permutations: int
    max_validation_work: int

    @property
    def unique_equivariant_selection_obstructed(self) -> bool:
        """Whether no candidate is fixed by the supplied state stabilizer."""
        return not self.fixed_candidates

    @property
    def scope(self) -> str:
        return "Supplied finite action and declared exact node/relation labels only"


class _Work:
    def __init__(self, maximum):
        self.maximum = maximum
        self.used = 0

    def charge(self, amount=1):
        if self.used + amount > self.maximum:
            raise ValueError("selector symmetry validation exceeds max_validation_work")
        self.used += amount


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_tuple(values, maximum, name, work):
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise ValueError(f"{name} must be an ordered finite iterable")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered finite iterable") from exc
    result = []
    for _ in range(maximum + 1):
        try:
            value = next(iterator)
        except StopIteration:
            return tuple(result)
        work.charge()
        result.append(value)
    raise ValueError(f"{name} exceeds its declared size limit")


def _label(value, work, depth=0):
    work.charge()
    if depth > 16:
        raise ValueError("state label nesting exceeds 16 levels")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is Fraction:
        numerator, denominator = value.numerator, value.denominator
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise ValueError("state labels require valid exact rational values")
        return Fraction(numerator, denominator)
    if type(value) is tuple:
        return tuple(_label(item, work, depth + 1) for item in value)
    raise ValueError("state labels require exact scalar labels or immutable tuples; no floats")


def derive_selector_symmetry(
    *, state_labels, permutations, candidates, relation_labels=None,
    max_nodes=64, max_permutations=256, max_validation_work=5_000_000,
) -> SelectorSymmetryObstruction:
    """Validate a finite action and derive its fixed-candidate obstruction.

    Vertices are indices ``0..len(state_labels)-1``. Each supplied permutation
    lists the image of each vertex; the nonempty list must contain identity,
    contain no duplicates and be closed under composition. Because the maps
    are permutations, finite composition closure also guarantees inverses.
    The supplied group need not be the full symmetry group of anything.

    Labels are exact ``None``, booleans, integers, strings, Fractions or nested
    tuples thereof (at most 16 nested levels); equality is exact numeric/value
    equality, so 1, True and Fraction(1) describe the same color. Use tagged
    tuples when their types should distinguish states. No float or tolerance
    is admitted. Fractions are copied and caller containers are detached.

    Optional ``relation_labels`` is an exact square label matrix, transformed
    in both indices. The stabilizer preserves these relations and the node
    labels together. Directed relations are allowed. For graph support use
    labels such as ``(edge_exists, exact_weight)`` so absence is distinct from
    a zero-weight edge. ``None`` leaves support unrepresented. Supplied labels
    are declarations, not observations authenticated against a live graph;
    omitted state, history and policy marks are outside the conclusion.

    The nonempty candidate set must be invariant under the label stabilizer.
    A noninvariant set carries additional asymmetric information, which must
    instead be declared in the state labels. Subsets of a larger true group
    can prove an obstruction, but absence of an obstruction in a subgroup
    says nothing about additional unrepresented symmetries.

    Size and work exhaustion raise ValueError without returning a partial
    result. Work counts consumed items, label nodes, permutation-coordinate
    checks, composition coordinates and stabilizer/candidate visits. It is
    not a bound on integer bit complexity or user-iterator execution time.
    """
    max_nodes = _positive_integer(max_nodes, "max_nodes")
    max_permutations = _positive_integer(max_permutations, "max_permutations")
    maximum = _positive_integer(max_validation_work, "max_validation_work")
    work = _Work(maximum)
    raw_labels = _bounded_tuple(state_labels, max_nodes, "state_labels", work)
    if not raw_labels:
        raise ValueError("state_labels must contain at least one vertex")
    labels = tuple(_label(value, work) for value in raw_labels)
    size = len(labels)
    identity = tuple(range(size))
    relations = None
    if relation_labels is not None:
        rows = _bounded_tuple(relation_labels, size, "relation_labels", work)
        if len(rows) != size:
            raise ValueError("relation_labels must be a square matrix matching the vertices")
        detached_rows = []
        for row in rows:
            entries = _bounded_tuple(row, size, "relation_labels row", work)
            if len(entries) != size:
                raise ValueError("relation_labels must be a square matrix matching the vertices")
            detached_rows.append(tuple(_label(value, work) for value in entries))
        relations = tuple(detached_rows)
    raw_group = _bounded_tuple(permutations, max_permutations, "permutations", work)
    group = []
    for raw in raw_group:
        permutation = _bounded_tuple(raw, size, "permutation", work)
        work.charge(size)
        if (
            len(permutation) != size
            or any(type(index) is not int for index in permutation)
            or set(permutation) != set(identity)
        ):
            raise ValueError("each permutation must be a bijection of the vertex indices")
        group.append(permutation)
    group_set = set(group)
    if not group or identity not in group_set:
        raise ValueError("permutations must contain identity")
    if len(group_set) != len(group):
        raise ValueError("duplicate permutations are not admitted")
    group = tuple(sorted(group_set))
    for left in group:
        for right in group:
            work.charge(size)
            if tuple(left[right[index]] for index in identity) not in group_set:
                raise ValueError("permutations must be closed under composition")
    stabilizer = []
    for permutation in group:
        work.charge(size)
        if not all(labels[index] == labels[permutation[index]] for index in identity):
            continue
        if relations is not None:
            work.charge(size * size)
            if any(
                relations[i][j] != relations[permutation[i]][permutation[j]]
                for i in identity for j in identity
            ):
                continue
        stabilizer.append(permutation)
    selected = _bounded_tuple(candidates, size, "candidates", work)
    if not selected or any(type(index) is not int or not 0 <= index < size for index in selected):
        raise ValueError("candidates must be nonempty valid vertex indices")
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("duplicate candidates are not admitted")
    selected = tuple(sorted(selected_set))
    for permutation in stabilizer:
        work.charge(len(selected))
        if any(permutation[index] not in selected_set for index in selected):
            raise ValueError("candidates must be invariant under the state stabilizer")
    pending = set(selected)
    orbits = []
    fixed = []
    while pending:
        vertex = min(pending)
        work.charge(len(stabilizer))
        orbit = tuple(sorted({permutation[vertex] for permutation in stabilizer}))
        orbits.append(orbit)
        pending.difference_update(orbit)
        if len(orbit) == 1:
            fixed.append(vertex)
    return SelectorSymmetryObstruction(
        labels, relations, group, selected, tuple(stabilizer), tuple(orbits), tuple(fixed),
        work.used, max_nodes, max_permutations, maximum,
    )
