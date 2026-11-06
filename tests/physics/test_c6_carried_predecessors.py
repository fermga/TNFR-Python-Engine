"""Exact predecessor frontiers checked against an independent 4096-state map."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_viability as owner
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

DELTA = F(1, 2**54)
QUANTUM = DELTA / 2
LOW = float(F(1, 2) - 4 * DELTA)
HIGH = float(F(LOW) + DELTA)
ROWS = tuple(product((LOW, HIGH), repeat=6))
TARGET = (2, 1, 2, 0, 2, 0)
ZERO = (0,) * 6


def _image(point):
    bits = tuple(int(value == 2) for value in point)
    return tuple(
        point[i] + bits[(i - 1) % 6] + bits[(i + 1) % 6] - 2 * bits[i] for i in range(6)
    )


def _state(point=ZERO, band=(0.375, 0.625)):
    exact = tuple(F(LOW) + QUANTUM * index for index in point)
    visible = tuple(map(float, exact))
    return NodalRemainderState(
        visible,
        tuple(value - F(x) for value, x in zip(exact, visible, strict=True)),
        *band,
    )


def _point(state):
    indices = tuple((value - F(LOW)) / QUANTUM for value in state.exact_epi)
    assert all(index.denominator == 1 for index in indices)
    return tuple(map(int, indices))


def _derive(reference, target=TARGET, *, state=None, **kwargs):
    return owner.derive_c6_carried_predecessors(
        reference,
        state=_state() if state is None else state,
        target=_state(target),
        epi_states=ROWS,
        timestep=1.0,
        max_depth=kwargs.pop("max_depth", 6),
        **kwargs,
    )


def _zone(point):
    extended = tuple(point) + (0,)
    matrix = tuple(tuple(extended[i] - extended[j] for j in range(7)) for i in range(7))
    return owner.C6CarriedForwardZone(_state(point).epi, matrix)


def _replay(reference, path):
    for first, second in zip(path, path[1:]):
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, first.epi).pressure
        step = advance_nodal_remainder(
            first, timestep=1.0, capacity=(1.0,) * 6, pressure=pressure
        )
        assert step.after == second


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


@pytest.fixture(scope="module")
def inverse(reference):
    # Bind the independent integer formula to every canonical source row.
    for bits, row in zip(product((0, 1), repeat=6), ROWS, strict=True):
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, row).pressure
        expected = tuple(
            QUANTUM * (bits[(i - 1) % 6] + bits[(i + 1) % 6] - 2 * bits[i])
            for i in range(6)
        )
        assert tuple(map(F, pressure)) == expected
    predecessors = defaultdict(set)
    for point in product(range(-1, 3), repeat=6):
        predecessors[_image(point)].add(point)
    return predecessors


@pytest.fixture(scope="module")
def complete(reference):
    return _derive(reference)


def test_complete_predecessor_frontiers_match_all_4096_physical_states(
    inverse, complete
):
    frontier = {TARGET}
    expected = []
    while frontier:
        expected.append(frontier)
        frontier = set().union(*(inverse[point] for point in frontier))
    expected.append(set())
    observed = tuple(
        {_point(state) for state in layer.states} for layer in complete.layers
    )
    assert observed == tuple(expected)
    assert complete.frontier_counts == (1, 1, 3, 6, 6, 2, 0)
    assert complete.status == "past_excluded"
    assert complete.completed_depth == complete.past_exclusion_depth == 6
    assert complete.maximum_compatible_past_depth == 5
    assert complete.row_checks == 19 * 64 == 1216
    assert complete.grid_quantum == QUANTUM
    assert complete.coordinate_spacings == (QUANTUM,) * 6
    assert complete.affine_origin == _state().exact_epi


def test_every_predecessor_binds_its_exact_successor_and_retained_carry(
    reference, complete
):
    for depth, layer in enumerate(complete.layers[1:], 1):
        assert layer.depth == depth and len(layer.states) == len(
            layer.successor_indices
        )
        previous = complete.layers[depth - 1].states
        assert len(set(layer.states)) == len(layer.states)
        for state, target_index in zip(
            layer.states, layer.successor_indices, strict=True
        ):
            target = previous[target_index]
            assert _image(_point(state)) == _point(target)
            _replay(reference, (state, target))
    assert len(complete.witness_path) == 6
    assert complete.witness_path[-1] == complete.target
    assert complete.target.remainder != (F(0),) * 6
    _replay(reference, complete.witness_path)


def test_pointwise_past_exclusion_never_promotes_to_a_domain_or_actual_origin_theorem(
    complete,
):
    assert complete.origin_reachability_depths == ()
    assert not complete.finite_origin_reachability_certified
    assert complete.origin_path == ()
    assert complete.origin_path_within_domain_excluded
    assert not complete.conditional_invariance_certified
    assert not complete.conditional_boundedness_certified
    assert not complete.whole_domain_exclusion_certified
    assert not complete.future_runtime_certified
    assert not complete.asymptotic_convergence_certified
    assert complete.state == _state() and complete.target == _state(TARGET)


def test_no_predecessor_is_an_exact_one_layer_absence(reference, inverse):
    target = (2, -1, 2, -1, 2, 1)
    assert inverse[target] == set()
    result = _derive(reference, target)
    assert result.status == "past_excluded"
    assert result.frontier_counts == (1, 0)
    assert (
        result.past_exclusion_depth == 1 and result.maximum_compatible_past_depth == 0
    )
    assert result.row_checks == 64
    assert result.witness_path == (result.target,)


def test_incomplete_frontier_is_discarded_even_if_its_last_candidate_is_missing(
    reference, complete
):
    result = _derive(reference, max_row_checks=complete.row_checks - 1)
    assert result.status == "resource_limit"
    assert result.completed_depth == 5 and result.frontier_counts == (1, 1, 3, 6, 6, 2)
    assert result.layers == complete.layers[:-1]
    assert (
        result.past_exclusion_depth is None
        and result.maximum_compatible_past_depth is None
    )
    assert result.row_checks <= result.max_row_checks == 1215
    assert not result.origin_path_within_domain_excluded
    assert not result.finite_origin_reachability_certified


def test_exact_row_budget_can_certify_the_empty_frontier(reference, complete):
    result = _derive(reference, max_row_checks=complete.row_checks)
    assert result.status == "past_excluded" and result.layers == complete.layers
    assert result.row_checks == result.max_row_checks == 1216


def test_resource_guard_before_first_complete_layer_keeps_only_the_target(reference):
    result = _derive(reference, max_row_checks=63)
    assert result.status == "resource_limit" and result.completed_depth == 0
    assert result.frontier_counts == (1,) and result.layers[0].states == (
        result.target,
    )
    assert result.past_exclusion_depth is None


def test_depth_query_does_not_turn_a_nonempty_frontier_into_exclusion(
    reference, complete
):
    result = _derive(reference, max_depth=5)
    assert result.status == "depth_limit" and result.layers == complete.layers[:-1]
    assert (
        result.past_exclusion_depth is None
        and result.maximum_compatible_past_depth is None
    )
    assert result.row_checks == 17 * 64


def test_depth_zero_is_a_valid_empty_work_query(reference):
    result = _derive(reference, max_depth=0)
    assert result.status == "depth_limit" and result.frontier_counts == (1,)
    assert result.row_checks == 0 and result.completed_depth == 0
    assert result.witness_path == (result.target,)


def test_exact_two_cycle_stays_a_finite_depth_query_and_preserves_origin_path(
    reference,
):
    origin = _state((0, 2) * 3)
    target = (2, 0) * 3
    result = _derive(reference, target, state=origin, max_depth=4)
    assert result.status == "depth_limit" and result.completed_depth == 4
    assert result.past_exclusion_depth is None
    assert result.finite_origin_reachability_certified
    assert (
        1 in result.origin_reachability_depths
        and 3 in result.origin_reachability_depths
    )
    assert result.origin_path[0] == origin and result.origin_path[-1] == result.target
    assert len(result.origin_path) == 2
    _replay(reference, result.origin_path)
    assert (
        not result.conditional_invariance_certified
        and not result.conditional_boundedness_certified
    )


def test_target_equal_to_origin_certifies_only_the_finite_zero_step_match(reference):
    result = _derive(reference, ZERO, max_depth=0)
    assert result.origin_reachability_depths == (0,)
    assert result.finite_origin_reachability_certified
    assert result.origin_path == (result.state,)
    assert not result.conditional_boundedness_certified


def test_explicit_subcell_domain_removes_predecessors_without_changing_pressure(
    reference, complete
):
    domains = (_zone(ZERO), _zone(TARGET))
    result = _derive(reference, domain_zones=domains)
    assert result.status == "past_excluded" and result.frontier_counts == (1, 0)
    assert result.domain_zones == domains
    assert result.pressures == complete.pressures
    assert result.coordinate_spacings == complete.coordinate_spacings
    assert result.target == complete.target


def test_rn_tie_ownership_matches_the_integer_oracle(inverse, complete):
    all_states = tuple(state for layer in complete.layers for state in layer.states)
    assert any(QUANTUM in state.remainder for state in all_states)
    for state in all_states:
        point = _point(state)
        assert state.epi == tuple(HIGH if x == 2 else LOW for x in point)
        assert all(x != 3 for x in point)
    for layer, previous in zip(complete.layers[1:], complete.layers):
        expected = set().union(*(inverse[_point(target)] for target in previous.states))
        assert {_point(state) for state in layer.states} == expected


def test_coordinate_cosets_are_stricter_than_the_global_common_grid(reference):
    isolated = (LOW, HIGH, LOW, LOW, LOW, LOW)
    rows = ((LOW,) * 6, isolated)
    origin = _state()
    target = _state((0, 1, 0, 0, 0, 0))
    assert (
        target.epi == origin.epi
        and target.exact_epi[1] - origin.exact_epi[1] == QUANTUM
    )
    with pytest.raises(ValueError, match="coset|lattice|spacing"):
        owner.derive_c6_carried_predecessors(
            reference,
            state=origin,
            target=target,
            epi_states=rows,
            timestep=1.0,
            max_depth=1,
        )


def test_tiny_carry_change_cannot_silently_change_the_exact_target_coset(reference):
    target = _state(TARGET)
    changed = replace(
        target, remainder=(target.remainder[0] + F(1, 2**3222),) + target.remainder[1:]
    )
    assert changed.epi == target.epi
    with pytest.raises(ValueError, match="coset|lattice|spacing"):
        owner.derive_c6_carried_predecessors(
            reference,
            state=_state(),
            target=changed,
            epi_states=ROWS,
            timestep=1.0,
            max_depth=1,
        )


@pytest.mark.parametrize("max_depth", (-1, True, 1.5))
def test_invalid_depth_queries_fail_closed(reference, max_depth):
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, max_depth=max_depth)


@pytest.mark.parametrize(
    "field,value", (("max_row_checks", 0), ("max_row_checks", True), ("max_cells", 0))
)
def test_invalid_resource_guards_fail_closed(reference, field, value):
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, **{field: value})


def test_domain_must_contain_both_supplied_origin_and_exact_target(reference):
    with pytest.raises(ValueError, match="origin|state|contain"):
        _derive(reference, domain_zones=(_zone(TARGET),))
    with pytest.raises(ValueError, match="target|contain"):
        _derive(reference, domain_zones=(_zone(ZERO),))


def test_duplicate_domain_rows_are_rejected(reference):
    domains = (_zone(ZERO), _zone(TARGET), _zone(TARGET))
    with pytest.raises(ValueError, match="duplicate|distinct"):
        _derive(reference, domain_zones=domains)


def test_noncanonical_dbm_caches_are_rejected_instead_of_assumed_closed(reference):
    zone = _zone(TARGET)
    bounds = [list(row) for row in zone.bounds]
    bounds[0][1] += 1
    invalid = replace(zone, bounds=tuple(map(tuple, bounds)))
    with pytest.raises(ValueError, match="closed|canonical|triangle"):
        _derive(reference, domain_zones=(_zone(ZERO), invalid))


def test_boolean_difference_bounds_are_not_exact_integer_certificates(reference):
    zone = _zone(TARGET)
    bounds = [list(row) for row in zone.bounds]
    bounds[0][0] = False
    invalid = replace(zone, bounds=tuple(map(tuple, bounds)))
    with pytest.raises((TypeError, ValueError), match="integer|bounds"):
        _derive(reference, domain_zones=(_zone(ZERO), invalid))


def test_domain_cannot_expand_beyond_its_nearest_even_source_cell(reference):
    low, high = (-1,) * 6 + (0,), (2,) * 6 + (0,)
    bounds = tuple(
        tuple(0 if i == j else high[i] - low[j] for j in range(7)) for i in range(7)
    )
    invalid = owner.C6CarriedForwardZone((LOW,) * 6, bounds)
    with pytest.raises(ValueError, match="RN|rounding|cell"):
        _derive(reference, domain_zones=(invalid, _zone(TARGET)))


def test_target_band_must_match_the_source_state_band(reference):
    target = replace(_state(TARGET), epi_lower=0.4)
    with pytest.raises(ValueError, match="band"):
        owner.derive_c6_carried_predecessors(
            reference,
            state=_state(),
            target=target,
            epi_states=ROWS,
            timestep=1.0,
            max_depth=1,
        )


def test_reference_caches_are_rebuilt_and_selected_path_uses_fresh_canonical_pressure(
    reference, complete, monkeypatch
):
    altered = replace(
        reference,
        sources=(F(99),) * 6,
        epi_quantum=F(3),
        gradient_quantum=F(7),
        rows=(),
    )
    calls = []
    canonical = owner._observe_rebuilt_c6_pressure_lattice

    def observe(ref, row):
        calls.append((ref, row))
        return canonical(ref, row)

    monkeypatch.setattr(owner, "_observe_rebuilt_c6_pressure_lattice", observe)
    result = _derive(altered)
    assert result.reference == reference and result.layers == complete.layers
    assert result.pressures == complete.pressures
    assert calls and all(ref.sources == reference.sources for ref, _row in calls)
    assert set(row for _ref, row in calls) == set(ROWS)
    _replay(reference, result.witness_path)


def test_exhausted_past_still_records_a_real_origin_path_at_depth_one(
    reference, complete
):
    origin = complete.layers[1].states[0]
    result = _derive(reference, state=origin)
    assert result.status == "past_excluded" and result.past_exclusion_depth == 6
    assert result.finite_origin_reachability_certified
    assert 1 in result.origin_reachability_depths
    assert not result.origin_path_within_domain_excluded
    assert result.origin_path == (origin, result.target)
    _replay(reference, result.origin_path)


def test_origin_found_only_in_an_unfinished_frontier_is_not_certified(
    reference, complete
):
    origin = complete.layers[1].states[0]
    assert ROWS.index(origin.epi) < 63
    result = _derive(reference, state=origin, max_row_checks=63)
    assert result.status == "resource_limit" and result.completed_depth == 0
    assert result.layers[0].states == (result.target,)
    assert result.origin_reachability_depths == () and result.origin_path == ()
    assert not result.finite_origin_reachability_certified
    assert not result.origin_path_within_domain_excluded
