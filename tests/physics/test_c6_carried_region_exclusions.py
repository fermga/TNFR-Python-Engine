"""Whole-region predecessor bounds checked against a finite nodal oracle."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_viability as owner
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, _observe_rebuilt_c6_pressure_lattice,
)


DELTA = F(1, 2**54)
QUANTUM = DELTA / 2
ENCODING_GRID = F(1, 2**3222)
LOW = float(F(1, 2) - 4 * DELTA)
HIGH = float(F(LOW) + DELTA)
ROWS = tuple(product((LOW, HIGH), repeat=6))
ZERO = (0,) * 6
POINT = (2, 1, 2, 0, 2, 0)
CYCLE_A, CYCLE_B = (0, 2) * 3, (2, 0) * 3
DOMAIN = frozenset(product(range(-1, 3), repeat=6))


def _row(point):
    return tuple(HIGH if x == 2 else LOW for x in point)


def _image(point):
    bits = tuple(int(x == 2) for x in point)
    return tuple(point[i] + bits[(i - 1) % 6] + bits[(i + 1) % 6] - 2 * bits[i] for i in range(6))


def _state(point=ZERO):
    exact = tuple(F(LOW) + QUANTUM * x for x in point)
    row = tuple(map(float, exact))
    return NodalRemainderState(row, tuple(x - F(y) for x, y in zip(exact, row, strict=True)), .375, .625)


def _zone(points, *, origin=ZERO, quantum=QUANTUM):
    points = tuple(points)
    row = _row(points[0])
    assert all(_row(point) == row for point in points)
    indices = tuple(tuple(QUANTUM * (x - base) / quantum for x, base in zip(point, origin, strict=True)) + (F(0),)
                    for point in points)
    assert all(value.denominator == 1 for point in indices for value in point)
    bounds = tuple(tuple(max(int(point[i] - point[j]) for point in indices) for j in range(7)) for i in range(7))
    return owner.C6CarriedForwardZone(row, bounds)


def _derive(reference, targets, **kwargs):
    return owner.derive_c6_carried_region_exclusions(
        reference, state=kwargs.pop('state', _state()), epi_states=kwargs.pop('epi_states', ROWS),
        timestep=1., target_regions=targets, **kwargs,
    )


def _represented(result, zones):
    assert result.grid_quantum == QUANTUM
    base = tuple((x - F(LOW)) / QUANTUM for x in result.affine_origin)
    assert all(x.denominator == 1 for x in base)
    matrices = {zone.epi: zone.bounds for zone in zones}
    points = set()
    for point in DOMAIN:
        matrix = matrices.get(_row(point))
        if matrix is None:
            continue
        shifted = tuple(x - int(y) for x, y in zip(point, base, strict=True)) + (0,)
        if all(shifted[i] - shifted[j] <= matrix[i][j] for i in range(7) for j in range(7)):
            points.add(point)
    return points


@pytest.fixture(scope='module')
def reference():
    return derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)


@pytest.fixture(scope='module')
def oracle(reference):
    cells, inverse = defaultdict(set), defaultdict(set)
    for point in DOMAIN:
        cells[_row(point)].add(point)
        inverse[_image(point)].add(point)
    for bits, row in zip(product((0, 1), repeat=6), ROWS, strict=True):
        fresh = _observe_rebuilt_c6_pressure_lattice(reference, row).pressure
        expected = tuple(QUANTUM * (bits[(i - 1) % 6] + bits[(i + 1) % 6] - 2 * bits[i]) for i in range(6))
        assert tuple(map(F, fresh)) == expected

    def hull(points):
        groups = defaultdict(list)
        for point in points:
            groups[_row(point)].append(point + (0,))
        represented = set()
        for row, group in groups.items():
            bounds = tuple(tuple(max(p[i] - p[j] for p in group) for j in range(7)) for i in range(7))
            represented.update(point for point in cells[row] if all(
                (point + (0,))[i] - (point + (0,))[j] <= bounds[i][j] for i in range(7) for j in range(7)
            ))
        return represented

    def layers(initial, origin=ZERO):
        result = [set(initial)]
        while result[-1] and origin not in result[-1]:
            previous = result[-1]
            following = hull(set().union(*(inverse[point] for point in previous)))
            result.append(following)
            if following == previous:
                break
        return result

    return cells, inverse, layers


@pytest.fixture(scope='module')
def slab():
    row = (LOW, HIGH, LOW, LOW, LOW, HIGH)
    points = frozenset(point for point in DOMAIN if _row(point) == row and point[0] == 1)
    assert len(points) == 27
    return points, _zone(points)


def test_an_entire_27_point_slab_is_excluded_by_complete_region_layers(reference, oracle, slab):
    points, zone = slab
    result = _derive(reference, ((zone,),))
    query = result.queries[0]
    expected = oracle[2](points)
    assert tuple(map(len, expected)) == (27, 15, 0)
    assert tuple(item.zone_count for item in query.iterations) == (1, 5, 0)
    assert tuple(item.origin_present for item in query.iterations) == (False, False, False)
    assert _represented(result, query.target_regions) == set(points)
    assert query.status == 'empty_complete_layer' and query.retained_zones == ()
    assert query.completed_depth == query.past_exclusion_depth == 2
    assert query.origin_path_within_domain_excluded
    assert not query.actual_origin_reachability_certified
    assert all(_image(point) not in DOMAIN for point in points)


def test_nonnested_region_sizes_are_not_treated_as_a_descending_sequence(reference, oracle):
    result = _derive(reference, ((_zone((POINT,)),),))
    query = result.queries[0]
    expected = oracle[2]({POINT})
    assert tuple(map(len, expected)) == (1, 1, 3, 6, 6, 2, 0)
    assert tuple(item.zone_count for item in query.iterations) == (1, 1, 3, 6, 6, 2, 0)
    assert not expected[2] <= expected[1] and not expected[1] <= expected[2]
    assert query.status == 'empty_complete_layer' and query.past_exclusion_depth == 6
    assert query.origin_path_within_domain_excluded


def test_incomplete_region_layer_retains_the_previous_complete_geometry(reference, oracle):
    result = _derive(reference, ((_zone((POINT,)),),), max_intersections=17)
    query = result.queries[0]
    expected = oracle[2]({POINT})
    assert result.intersections == query.intersections == 17
    assert query.status == 'resource_limit' and query.completed_depth == 4
    assert _represented(result, query.retained_zones) == expected[4]
    assert tuple(item.zone_count for item in query.iterations) == (1, 1, 3, 6, 6)
    assert not query.origin_path_within_domain_excluded and query.past_exclusion_depth is None


def test_origin_in_a_new_hull_does_not_mean_the_actual_origin_reaches_a_target(reference, oracle):
    origin = (0, 2, 0, 0, 0, 0)
    first, second = (0, 0, 1, 0, 0, 0), (2, 0, 1, 0, 0, 0)
    zones = (_zone((first,), origin=origin), _zone((second,), origin=origin))
    result = _derive(reference, (zones,), state=_state(origin))
    query = result.queries[0]
    expected = oracle[2]({first, second}, origin=origin)
    assert origin not in {first, second} and origin in expected[1]
    assert _image(origin) not in {first, second}
    assert _image(_image(origin)) == _image(origin)
    assert query.status == 'origin_not_excluded' and query.completed_depth == 1
    assert query.iterations[-1].origin_present
    assert _represented(result, query.retained_zones) == expected[1]
    assert not query.origin_path_within_domain_excluded and not query.actual_origin_reachability_certified
    current = result.state
    for expected_point in (_image(origin), _image(origin)):
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, current.epi).pressure
        current = advance_nodal_remainder(current, timestep=1., capacity=(1.,) * 6, pressure=pressure).after
        assert current == _state(expected_point)
    assert current not in (_state(first), _state(second))


def test_initial_origin_membership_is_inconclusive_without_starting_a_search(reference):
    result = _derive(reference, ((_zone((ZERO,)),),))
    query = result.queries[0]
    assert query.status == 'origin_not_excluded' and query.completed_depth == 0
    assert result.intersections == query.intersections == 0
    assert not query.origin_path_within_domain_excluded and not query.actual_origin_reachability_certified


def test_exact_stationary_region_with_absent_origin_certifies_path_exclusion(reference):
    other = (2,) * 6
    rows = ((LOW,) * 6, (HIGH,) * 6)
    origin_zone = _zone((ZERO,), quantum=ENCODING_GRID)
    target_zone = _zone((other,), quantum=ENCODING_GRID)
    result = _derive(reference, ((target_zone,),), epi_states=rows, domain_zones=(origin_zone, target_zone))
    query = result.queries[0]
    assert result.grid_quantum == ENCODING_GRID
    assert query.status == 'stationary_complete_layer' and query.completed_depth == 1
    assert query.retained_zones == query.target_regions
    assert query.origin_path_within_domain_excluded and query.past_exclusion_depth is None
    assert tuple(item.origin_present for item in query.iterations) == (False, False)
    assert not query.actual_origin_reachability_certified


def test_equal_cardinality_two_cycle_is_not_misread_as_a_stationary_region(reference):
    domain = tuple(_zone((point,)) for point in (ZERO, CYCLE_A, CYCLE_B))
    result = _derive(reference, ((_zone((CYCLE_A,)),),), domain_zones=domain, max_intersections=3)
    query = result.queries[0]
    assert query.status == 'resource_limit' and query.completed_depth == 3
    assert tuple(item.zone_count for item in query.iterations) == (1, 1, 1, 1)
    assert _represented(result, query.retained_zones) == {CYCLE_B}
    assert query.retained_zones != query.target_regions
    assert not query.origin_path_within_domain_excluded


def test_shared_budget_advances_labels_round_robin_without_promoting_a_partial_query(reference):
    domain = tuple(_zone((point,)) for point in (ZERO, CYCLE_A, CYCLE_B))
    targets = ((_zone((CYCLE_A,)),), (_zone((CYCLE_B,)),))
    result = _derive(reference, targets, domain_zones=domain, max_intersections=3)
    first, second = result.queries
    assert (first.completed_depth, second.completed_depth) == (2, 1)
    assert (first.intersections, second.intersections) == (2, 1)
    assert result.intersections == sum(query.intersections for query in result.queries) == 3
    assert all(query.status == 'resource_limit' for query in result.queries)
    assert _represented(result, first.retained_zones) == {CYCLE_A}
    assert _represented(result, second.retained_zones) == {CYCLE_A}
    assert all(not query.origin_path_within_domain_excluded for query in result.queries)


def test_excluded_slabs_do_not_certify_the_entire_domain_or_a_live_graph(reference, slab):
    result = _derive(reference, ((slab[1],),))
    assert result.queries[0].origin_path_within_domain_excluded
    assert result.common_grid_relaxes_coordinate_cosets
    assert not result.conditional_invariance_certified and not result.conditional_boundedness_certified
    assert not result.future_runtime_certified and not result.asymptotic_convergence_certified
    assert result.state == _state() and result.affine_origin == result.state.exact_epi


def test_custom_domain_must_contain_the_original_state(reference):
    with pytest.raises(ValueError, match='origin'):
        _derive(reference, ((_zone((POINT,)),),), domain_zones=(_zone((POINT,)),))


def test_every_target_must_be_a_subset_of_its_declared_source_domain(reference):
    domain = (_zone((ZERO,)),)
    with pytest.raises(ValueError, match='inside the declared domain'):
        _derive(reference, ((_zone((POINT,)),),), domain_zones=domain)


def test_nonclosed_target_bounds_are_rejected(reference):
    zone = _zone((POINT,))
    matrix = [list(row) for row in zone.bounds]
    matrix[0][1] += 1
    with pytest.raises(ValueError, match='closed'):
        _derive(reference, ((replace(zone, bounds=tuple(map(tuple, matrix))),),))


def test_target_rn_ties_cannot_be_extended_into_the_wrong_visible_cell(reference):
    points = ((2, 0, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0))
    lower, upper = points[0] + (0,), points[1] + (0,)
    bounds = tuple(tuple(0 if i == j else upper[i] - lower[j] for j in range(7)) for i in range(7))
    invalid = owner.C6CarriedForwardZone(_row(points[0]), bounds)
    with pytest.raises(ValueError, match='RN cell'):
        _derive(reference, ((invalid,),))


def test_duplicate_visible_rows_inside_one_target_query_are_rejected(reference):
    zone = _zone((POINT,))
    with pytest.raises(ValueError, match='distinct'):
        _derive(reference, ((zone, zone),))


@pytest.mark.parametrize('targets', ((), [], (None,), ((),)))
def test_empty_or_implicit_target_queries_fail_closed(reference, targets):
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, targets)


@pytest.mark.parametrize('keyword,value', (('max_intersections', 0), ('max_intersections', True), ('max_cells', 0)))
def test_invalid_computational_budgets_are_rejected(reference, keyword, value):
    with pytest.raises(ValueError):
        _derive(reference, ((_zone((POINT,)),),), **{keyword: value})


def test_reference_derived_caches_are_rebuilt_before_any_region_pressure_is_used(reference, slab, monkeypatch):
    forged = replace(reference, sources=(F(99),) * 6, rows=(), epi_quantum=F(3), gradient_quantum=F(7))
    calls = []
    canonical = owner._observe_rebuilt_c6_pressure_lattice

    def observe(ref, row):
        calls.append((ref, row))
        return canonical(ref, row)

    monkeypatch.setattr(owner, '_observe_rebuilt_c6_pressure_lattice', observe)
    result = _derive(forged, ((slab[1],),))
    assert result.reference == reference
    assert result.queries[0].status == 'empty_complete_layer'
    assert {row for _ref, row in calls} == set(ROWS)
    assert all(ref.sources == reference.sources for ref, _row in calls)
