"""Independent finite-map oracles for exact carried C6 short returns."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


DELTA = F(1, 2**54)
VALUES = tuple(float(F(1, 2) + (i - 4) * DELTA) for i in range(3))
DOMAIN = set(product(range(3), repeat=6))
ROWS = tuple(product(VALUES, repeat=6))
ORIGIN = (0, 0, 0, 1, 2, 1)
NO_RETURN_ORIGIN = (0, 0, 0, 0, 1, 1)
TRANSIENT = {(1, 0, 1, 1, 0, 1), (1, 0, 0, 1, 0, 0)}


def _row(point):
    return tuple(VALUES[i] for i in point)


def _state(point=ORIGIN):
    return NodalRemainderState(_row(point), (F(0),) * 6, .375, .625)


def _image(point):
    return tuple(point[i - 1] + point[(i + 1) % 6] - point[i] for i in range(6))


def _return(point):
    first = _image(point)
    if first not in DOMAIN:
        return None
    return _image(first) if first in TRANSIENT and _image(first) in DOMAIN else (
        None if first in TRANSIENT else first
    )


def _layers(origin):
    layers = [(DOMAIN - TRANSIENT, TRANSIENT)]
    while True:
        low = {origin} | {_return(p) for p in layers[-1][0] if _return(p) is not None}
        high = {_image(p) for p in low if _image(p) in TRANSIENT}
        layers.append((low, high))
        if low == layers[-2][0]:
            return layers


def _point(result, guard):
    low, high = tuple(-guard[6][i] for i in range(6)), tuple(guard[i][6] for i in range(6))
    assert low == high
    exact = tuple(a + result.grid_quantum * k for a, k in zip(result.affine_origin, low, strict=True))
    point = tuple((x - F(VALUES[0])) / DELTA for x in exact)
    assert all(x.denominator == 1 for x in point)
    return tuple(map(int, point))


def _points(result, zones):
    return {_point(result, zone.bounds) for zone in zones}


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)


def _derive(reference, **changes):
    args = dict(state=_state(), epi_states=ROWS, timestep=2.,
                transient_epi_states=tuple(_row(p) for p in sorted(TRANSIENT)),
                max_intersections=1_000_000)
    args.update(changes)
    return owner.derive_c6_carried_return_envelope(reference, **args)


@pytest.fixture(scope="module")
def result(reference):
    return _derive(reference)


def test_exact_return_relation_matches_an_independent_729_state_map(result):
    actual = {(_point(result, edge.source_guard), _point(result, edge.target_guard))
              for edge in result.return_relation}
    expected = {(p, _return(p)) for p in DOMAIN - TRANSIENT if _return(p) is not None}
    assert actual == expected and len(actual) == len(result.return_relation) == 168
    assert result.ordinary_transition_count == sum(_image(p) in DOMAIN for p in DOMAIN)
    assert result.grid_quantum == DELTA and result.affine_origin == _state().exact_epi
    assert result.relation_complete and result.transient_isolation_certified


def test_guards_retain_every_complete_intermediate_constraint_and_translation(result):
    two_step = 0
    for edge in result.return_relation:
        source, target = _point(result, edge.source_guard), _point(result, edge.target_guard)
        assert edge.source_epi == _row(source) and edge.target_epi == _row(target)
        assert edge.shift == tuple(b - a for a, b in zip(source, target, strict=True))
        if edge.intermediate_epi is not None:
            two_step += 1
            middle = _point(result, edge.intermediate_guard)
            assert middle in TRANSIENT and middle == _image(source) and target == _image(middle)
            assert edge.intermediate_epi == _row(middle)
        else:
            assert edge.intermediate_guard is None and target == _image(source)
    assert two_step == 4


def test_complete_return_layers_match_the_finite_oracle_and_cover_the_origin_path(result):
    layers = _layers(ORIGIN)
    counts = tuple((len(low), len(high)) for low, high in layers)
    assert counts == ((727, 2), (44, 1), (10, 1), (5, 1), (5, 1))
    assert tuple((layer.base_zone_count, layer.transient_zone_count) for layer in result.iterations) == counts
    assert tuple(layer.ordinal for layer in result.iterations) == tuple(range(len(layers)))
    assert all(layer.origin_retained for layer in result.iterations)
    assert _points(result, result.retained_zones) == set.union(*layers[-1])
    point, visited = ORIGIN, set()
    while point in DOMAIN and point not in visited:
        assert all(point in low | high for low, high in layers)
        visited.add(point)
        point = _image(point)
    assert result.status == "fixed_point" and result.domain_confined_origin_paths_covered
    assert result.clipped_return_inclusion_certified


def test_nonreturning_transient_images_are_kept_even_when_the_next_step_exits(reference):
    result = _derive(reference, state=_state(NO_RETURN_ORIGIN))
    first = _image(NO_RETURN_ORIGIN)
    assert first in TRANSIENT and _image(first) not in DOMAIN
    retained = _points(result, result.retained_zones)
    assert NO_RETURN_ORIGIN in retained and first in retained
    assert any(_point(result, e.source_guard) == NO_RETURN_ORIGIN for e in result.intermediate_transitions)
    assert all(_point(result, e.source_guard) != NO_RETURN_ORIGIN for e in result.return_relation)
    pressure = result.pressures[ROWS.index(result.state.epi)]
    step = advance_nodal_remainder(result.state, timestep=2., capacity=(1.,) * 6, pressure=pressure)
    assert step.after == _state(first) and not any(step.nodal_balance_residual)
    assert result.status == "fixed_point"
    assert not result.clipped_forward_inclusion_certified
    assert not result.conditional_invariance_certified and not result.conditional_boundedness_certified
    assert not result.future_runtime_certified and not result.asymptotic_convergence_certified


def test_interrupted_ordinary_construction_retains_the_entire_validated_domain(reference):
    limited = _derive(reference, max_intersections=1)
    assert limited.status == "construction_resource_limit" and not limited.relation_complete
    assert not limited.transient_isolation_certified and not limited.clipped_return_inclusion_certified
    assert limited.retained_zones == limited.domain_zones
    assert limited.return_relation == limited.intermediate_transitions == ()
    assert len(limited.iterations) == 1 and limited.construction_intersections == 1
    assert limited.closure_intersections == 0 and limited.domain_confined_origin_paths_covered


def test_interrupted_two_step_construction_never_publishes_an_incomplete_relation(reference):
    limited = _derive(reference, max_intersections=len(ROWS) ** 2)
    assert limited.status == "construction_resource_limit" and limited.transient_isolation_certified
    assert not limited.relation_complete and limited.return_relation == ()
    assert limited.retained_zones == limited.domain_zones and len(limited.iterations) == 1
    assert limited.intersections == limited.max_intersections and limited.closure_intersections == 0


def test_partial_intermediate_reconstruction_discards_the_entire_layer(reference, result):
    first_layer_work = len(result.return_relation) + len(result.intermediate_transitions)
    limited = _derive(reference, max_intersections=result.construction_intersections + first_layer_work - 1)
    assert limited.relation_complete and limited.status == "resource_limit"
    assert limited.retained_zones == limited.domain_zones and len(limited.iterations) == 1
    assert limited.closure_intersections == first_layer_work - 1
    assert limited.intersections == limited.max_intersections


def test_fixed_point_requires_a_complete_layer_including_all_intermediates(reference, result):
    limited = _derive(reference, max_intersections=result.intersections - 1)
    assert limited.status == "resource_limit" and len(limited.iterations) == len(result.iterations) - 1
    assert limited.retained_zones == result.retained_zones
    complete = _derive(reference, max_intersections=result.intersections)
    assert complete.status == "fixed_point" and complete.iterations == result.iterations
    assert complete.intersections == complete.max_intersections


def test_stationary_transient_row_is_rejected_once_its_self_edge_is_discovered(reference):
    with pytest.raises(ValueError, match="transient-to-transient"):
        _derive(reference, transient_epi_states=(_row((0,) * 6),))


def test_empty_transient_selection_reduces_to_an_ordinary_conditional_envelope(reference):
    state = NodalRemainderState((.5,) * 6, tuple(F(i, 2**3222) for i in range(6)), .375, .625)
    result = _derive(reference, state=state, epi_states=(state.epi,), timestep=1., transient_epi_states=())
    assert result.status == "fixed_point" and result.return_relation[0].shift == (0,) * 6
    assert result.grid_quantum == F(1, 2**3222) and len(set(result.affine_origin)) == 6
    assert result.retained_zones == result.domain_zones and result.intermediate_transitions == ()


def test_forged_reference_caches_are_rebuilt_from_the_canonical_source(reference, result):
    forged = replace(reference, sources=(17.,) * 6, rows=(), epi_quantum=F(7), gradient_quantum=F(19))
    rebuilt = _derive(forged)
    assert rebuilt.reference == reference and rebuilt.pressures == result.pressures
    assert rebuilt.return_relation == result.return_relation and rebuilt.retained_zones == result.retained_zones


@pytest.mark.parametrize("tamper", ("origin", "duplicate", "unknown", "container"))
def test_invalid_transient_selection_is_rejected(reference, tamper):
    transient = (_state().epi,)
    if tamper == "duplicate":
        transient = (_row((0,) * 6),) * 2
    elif tamper == "unknown":
        transient = ((.5,) * 6,)
    elif tamper == "container":
        transient = []
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, transient_epi_states=transient)


@pytest.mark.parametrize("tamper", ("missing_origin", "unclosed", "outside_rn", "carry", "phase"))
def test_invalid_domain_or_nodal_primitive_is_rejected(reference, result, tamper):
    args = dict(domain_zones=result.domain_zones)
    if tamper == "missing_origin":
        args["domain_zones"] = tuple(z for z in result.domain_zones if z.epi != result.state.epi)
    elif tamper in ("unclosed", "outside_rn"):
        zone = result.domain_zones[0]
        bounds = [list(row) for row in zone.bounds]
        if tamper == "unclosed":
            bounds[0][1] += 1
        else:
            bounds = [[100 if i != j else 0 for j in range(7)] for i in range(7)]
        args["domain_zones"] = (C6CarriedForwardZone(zone.epi, tuple(map(tuple, bounds))),) + result.domain_zones[1:]
    elif tamper == "carry":
        args["state"] = replace(_state(), remainder=(DELTA / 3,) + (F(0),) * 5)
    else:
        reference = replace(reference, source=replace(reference.source, phase=(float("nan"),) * 6))
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, **args)


@pytest.mark.parametrize("guard,value", (("max_cells", True), ("max_cells", 1),
                                         ("max_intersections", 0), ("max_intersections", 1.5)))
def test_computational_guards_fail_closed(reference, guard, value):
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, **{guard: value})
