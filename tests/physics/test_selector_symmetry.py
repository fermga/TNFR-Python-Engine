"""Finite selection consistency oracles for a supplied exact group action."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import permutations, product, repeat

import pytest

from tnfr.physics.selector_symmetry import derive_selector_symmetry


def _derive(labels=(0, 0), group=((0, 1), (1, 0)), candidates=(0, 1), **kwargs):
    return derive_selector_symmetry(
        state_labels=labels,
        permutations=group,
        candidates=candidates,
        **kwargs,
    )


def _push(permutation, labels):
    transformed = [None] * len(labels)
    for vertex, image in enumerate(permutation):
        transformed[image] = labels[vertex]
    return tuple(transformed)


def test_symmetric_orbit_cannot_supply_a_unique_equivariant_parent():
    rotations = tuple(
        tuple((i + shift) % 8 for i in range(8)) for shift in (0, 2, 4, 6)
    )
    result = _derive(("high", "low") * 4, rotations, (0, 2, 4, 6))
    assert result.candidate_orbits == ((0, 2, 4, 6),)
    assert result.fixed_candidates == ()
    assert result.unique_equivariant_selection_obstructed
    assert "Supplied" in result.scope


def test_marked_parent_can_remove_this_obstruction_without_deriving_a_selector():
    group = tuple(permutations(range(3)))
    result = _derive(("marked", "other", "other"), group, (0, 1, 2))
    assert result.candidate_orbits == ((0,), (1, 2))
    assert result.fixed_candidates == (0,)
    assert not result.unique_equivariant_selection_obstructed
    # The same declaration can restrict candidacy to its distinguished mark.
    assert _derive(("marked", "other", "other"), group, (0,)).fixed_candidates == (0,)


def test_two_fixed_candidates_do_not_imply_a_unique_selection():
    result = _derive(("first", "second"))
    assert result.fixed_candidates == (0, 1)
    assert not result.unique_equivariant_selection_obstructed


def test_fixed_candidates_equal_consistent_selector_extensions_on_every_small_state_orbit():
    """Independent oracle: assign s(gX)=g(v) and reject contradictory values."""
    group = tuple(permutations(range(3)))
    checked = 0
    for labels in product((0, 1), repeat=3):
        for membership in product((False, True), repeat=3):
            candidates = tuple(i for i, allowed in enumerate(membership) if allowed)
            if not candidates:
                continue
            # Candidacy is transported with the state. It must be independent
            # of which permutation represents the same state in its orbit.
            orbit_candidates = {}
            valid = True
            for permutation in group:
                state = _push(permutation, labels)
                images = frozenset(permutation[i] for i in candidates)
                if state in orbit_candidates and orbit_candidates[state] != images:
                    valid = False
                orbit_candidates[state] = images
            if not valid:
                with pytest.raises(ValueError, match="invariant"):
                    _derive(labels, group, candidates)
                continue
            admissible_choices = []
            for candidate in candidates:
                selection = {}
                consistent = True
                for permutation in group:
                    state = _push(permutation, labels)
                    choice = permutation[candidate]
                    if state in selection and selection[state] != choice:
                        consistent = False
                    selection[state] = choice
                if consistent:
                    admissible_choices.append(candidate)
            result = _derive(labels, group, candidates)
            assert result.fixed_candidates == tuple(admissible_choices)
            assert result.unique_equivariant_selection_obstructed == (
                not admissible_choices
            )
            checked += 1
    assert checked == 20


def test_subgroup_absence_of_obstruction_does_not_promote_to_full_group():
    labels = (0, 0, 0)
    subgroup = ((0, 1, 2), (1, 0, 2))
    assert _derive(labels, subgroup, (0, 1, 2)).fixed_candidates == (2,)
    assert (
        _derive(labels, tuple(permutations(range(3))), (0, 1, 2)).fixed_candidates == ()
    )


def test_exact_numeric_colors_and_tagged_type_distinctions():
    assert _derive((1, Fraction(1))).unique_equivariant_selection_obstructed
    assert _derive((True, 1)).unique_equivariant_selection_obstructed
    result = _derive((("integer", 1), ("boolean", True)))
    assert result.fixed_candidates == (0, 1)
    assert _derive(
        (Fraction(1, 3), Fraction(1, 3))
    ).unique_equivariant_selection_obstructed


def test_input_order_is_not_a_symmetry_breaking_selector():
    left = _derive()
    right = _derive(group=((1, 0), (0, 1)), candidates=(1, 0))
    assert left == right


def test_relabeling_transports_fixed_candidates_and_all_candidate_orbits():
    labels = ("marked", "other", "other")
    group = tuple(permutations(range(3)))
    relabel = (2, 0, 1)
    before = _derive(labels, group, (0, 1, 2))
    after = _derive(_push(relabel, labels), group, (0, 1, 2))
    assert after.fixed_candidates == tuple(
        sorted(relabel[i] for i in before.fixed_candidates)
    )
    expected = {
        frozenset(relabel[i] for i in orbit) for orbit in before.candidate_orbits
    }
    assert {frozenset(orbit) for orbit in after.candidate_orbits} == expected


def test_input_containers_are_detached_and_result_is_frozen():
    rational = Fraction(1, 3)
    labels = [("mark", rational), ("mark", rational)]
    group = [[0, 1], [1, 0]]
    candidates = [0, 1]
    result = _derive(labels, group, candidates)
    labels[0] = "changed"
    group[0][0] = 1
    candidates.clear()
    assert result.state_labels == (("mark", Fraction(1, 3)),) * 2
    assert result.state_labels[0][1] is not rational
    assert result.permutations == ((0, 1), (1, 0))
    assert result.candidates == (0, 1)
    with pytest.raises(FrozenInstanceError):
        result.candidates = (0,)


@pytest.mark.parametrize(
    "group, message",
    [
        ((), "identity"),
        (((1, 0),), "identity"),
        (((0, 1), (0, 1)), "duplicate"),
        (((0, 0),), "bijection"),
        (((0,),), "bijection"),
        (((0, 1, 2),), "size limit"),
        (((0, True),), "bijection"),
        (((0, 1.0),), "bijection"),
        (((0, -1),), "bijection"),
        (((0, []),), "bijection"),
    ],
)
def test_malformed_permutation_actions_fail_closed(group, message):
    with pytest.raises(ValueError, match=message):
        _derive(group=group)


def test_truncated_cycle_action_is_not_silently_used_as_a_group():
    with pytest.raises(ValueError, match="composition"):
        _derive((0, 0, 0), ((0, 1, 2), (1, 2, 0)), (0, 1, 2))


@pytest.mark.parametrize(
    "candidates, message",
    [
        ((), "nonempty"),
        ((0, 0), "duplicate"),
        ((0,), "invariant"),
        ((True,), "valid vertex"),
        ((1.0,), "valid vertex"),
        ((-1,), "valid vertex"),
        ((2,), "valid vertex"),
        ({0, 1}, "ordered"),
    ],
)
def test_invalid_or_asymmetrically_undeclared_candidates_fail(candidates, message):
    with pytest.raises(ValueError, match=message):
        _derive(candidates=candidates)


@pytest.mark.parametrize(
    "labels",
    [
        (),
        (0.0, 0.0),
        (float("nan"), 0),
        (float("inf"), 0),
        (object(), object()),
        ([1], [1]),
        ({"a": 1}, {"a": 1}),
        {0, 1},
        "ab",
    ],
)
def test_nonexact_mutable_or_empty_labels_fail(labels):
    with pytest.raises(ValueError):
        _derive(labels=labels)


def test_label_nesting_and_infinite_input_materialization_are_bounded():
    nested = 0
    for _ in range(18):
        nested = (nested,)
    with pytest.raises(ValueError, match="nesting"):
        _derive(labels=(nested, 0))
    with pytest.raises(ValueError, match="size limit"):
        _derive(labels=repeat(0), max_nodes=2)
    with pytest.raises(ValueError, match="size limit"):
        _derive(group=repeat((0, 1)), max_permutations=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_nodes": 0},
        {"max_nodes": True},
        {"max_nodes": 2.0},
        {"max_permutations": 0},
        {"max_permutations": False},
        {"max_validation_work": 0},
        {"max_validation_work": 1.0},
    ],
)
def test_resource_limits_are_strict_positive_integers(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        _derive(**kwargs)


def test_work_limit_is_exact_and_never_returns_a_partial_obstruction():
    reference = _derive()
    assert _derive(max_validation_work=reference.validation_work).fixed_candidates == ()
    for limit in (1, reference.validation_work - 1):
        with pytest.raises(ValueError, match="max_validation_work"):
            _derive(max_validation_work=limit)


def test_single_vertex_group_is_valid_without_manufacturing_nontrivial_symmetry():
    result = _derive((None,), ((0,),), (0,))
    assert result.fixed_candidates == (0,)
    assert result.candidate_orbits == ((0,),)
    assert not result.unique_equivariant_selection_obstructed


def test_directed_relation_break_is_detected_even_with_identical_node_labels():
    result = _derive(relation_labels=((False, True), (False, False)))
    assert result.stabilizer_permutations == ((0, 1),)
    assert result.fixed_candidates == (0, 1)
    assert result.relation_labels == ((False, True), (False, False))


def test_support_and_weight_marks_define_stabilizer_without_live_graph_claim():
    # Three-node path has reflection, while its centre is uniquely fixed.
    absent, present = (False, 0), (True, 1)
    relations = (
        (absent, present, absent),
        (present, absent, present),
        (absent, present, absent),
    )
    result = _derive(
        (0, 0, 0), tuple(permutations(range(3))), (0, 1, 2), relation_labels=relations
    )
    assert result.stabilizer_permutations == ((0, 1, 2), (2, 1, 0))
    assert result.fixed_candidates == (1,)
    assert result.candidate_orbits == ((0, 2), (1,))


def test_absent_and_zero_weight_edges_are_distinct_when_declared():
    absent, zero_edge = (False, 0), (True, 0)
    result = _derive(relation_labels=((absent, zero_edge), (absent, absent)))
    assert result.fixed_candidates == (0, 1)
    assert _derive(
        relation_labels=((0, 0), (0, 0))
    ).unique_equivariant_selection_obstructed


def test_relations_are_deeply_detached_and_share_the_work_budget():
    relations = [[0, 1], [1, 0]]
    result = _derive(relation_labels=relations)
    relations[0][1] = 2
    assert result.relation_labels == ((0, 1), (1, 0))
    assert result.unique_equivariant_selection_obstructed
    with pytest.raises(ValueError, match="max_validation_work"):
        _derive(
            relation_labels=result.relation_labels,
            max_validation_work=result.validation_work - 1,
        )


@pytest.mark.parametrize(
    "relations",
    [
        (),
        ((0, 1),),
        ((0,), (0, 1)),
        ((0, 1, 2), (0, 1)),
        ((0.0, 1), (1, 0)),
        "abcd",
        ((object(), 1), (1, 0)),
    ],
)
def test_incomplete_or_inexact_relation_matrices_fail_closed(relations):
    with pytest.raises(ValueError):
        _derive(relation_labels=relations)
