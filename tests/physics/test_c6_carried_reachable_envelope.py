"""Finite-map oracles for origin-preserving, domain-clipped C6 envelopes."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_viability as owner
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


DELTA = F(1, 2**54)
VALUES = tuple(float(F(1, 2) + (i - 4) * DELTA) for i in range(3))
ROWS = tuple(product(VALUES, repeat=6))
ORIGIN = (1, 2, 2, 1, 0, 0)
ESCAPING = (0, 2, 0, 0, 0, 0)


def _state(point=ORIGIN):
    return NodalRemainderState(tuple(VALUES[i] for i in point), (F(0),) * 6, .375, .625)


def _image(point):
    return tuple(point[i - 1] + point[(i + 1) % 6] - point[i] for i in range(6))


def _layers(origin):
    domain = set(product(range(3), repeat=6))
    result = [domain]
    while True:
        following = {origin} | ({_image(point) for point in result[-1]} & domain)
        result.append(following)
        if following == result[-2]:
            return result
        assert following < result[-2]


def _points(result, zones):
    points = set()
    for zone in zones:
        lower = tuple(-zone.bounds[6][i] for i in range(6))
        upper = tuple(zone.bounds[i][6] for i in range(6))
        assert lower == upper
        exact = tuple(a + result.grid_quantum * n for a, n in zip(result.affine_origin, lower))
        assert tuple(map(float, exact)) == zone.epi
        point = tuple((x - F(VALUES[0])) / DELTA for x in exact)
        assert all(x.denominator == 1 for x in point)
        points.add(tuple(map(int, point)))
    return points


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)


@pytest.fixture(scope="module")
def result(reference):
    return owner.derive_c6_carried_reachable_envelope(
        reference, state=_state(), epi_states=ROWS, timestep=2.,
    )


def test_complete_origin_injected_layers_match_an_independent_729_state_oracle(result):
    layers = _layers(ORIGIN)
    assert tuple(map(len, layers)) == (729, 46, 10, 4, 4)
    assert result.status == "fixed_point" and result.grid_quantum == DELTA
    assert tuple(item.zone_count for item in result.iterations) == tuple(map(len, layers))
    assert tuple(item.ordinal for item in result.iterations) == tuple(range(5))
    assert all(item.origin_retained for item in result.iterations)
    assert _points(result, result.domain_zones) == layers[0]
    assert _points(result, result.retained_zones) == layers[-1]
    assert ORIGIN in layers[-1]
    assert result.domain_confined_paths_covered and result.clipped_forward_inclusion_certified


def test_every_saved_pressure_is_the_exact_integer_diffusion_map(result):
    for point, pressure in zip(product(range(3), repeat=6), result.pressures, strict=True):
        expected = tuple(DELTA * (a - b) for a, b in zip(_image(point), point))
        assert tuple(2 * F(p) for p in pressure) == expected
    first = advance_nodal_remainder(result.state, timestep=2., capacity=(1.,) * 6,
                                    pressure=result.pressures[ROWS.index(result.state.epi)])
    assert first.after == _state((1,) * 6)
    assert result.state == _state() and result.affine_origin == result.state.exact_epi


def test_stationary_clipped_envelope_can_contain_an_origin_that_actually_exits(reference):
    result = owner.derive_c6_carried_reachable_envelope(
        reference, state=_state(ESCAPING), epi_states=ROWS, timestep=2.,
    )
    retained = _points(result, result.retained_zones)
    assert result.status == "fixed_point" and retained == _layers(ESCAPING)[-1]
    assert len(retained) == 4 and ESCAPING in retained
    assert _image(ESCAPING) == (2, -2, 2, 0, 0, 0)
    assert _image(ESCAPING) not in _points(result, result.domain_zones)
    first = advance_nodal_remainder(result.state, timestep=2., capacity=(1.,) * 6,
                                    pressure=result.pressures[ROWS.index(result.state.epi)])
    assert first.after.epi not in ROWS and not any(first.nodal_balance_residual)
    assert result.clipped_forward_inclusion_certified
    assert not result.conditional_invariance_certified and not result.conditional_boundedness_certified
    assert not result.future_runtime_certified and not result.asymptotic_convergence_certified


def test_partial_first_image_keeps_all_validated_source_cells(reference):
    limited = owner.derive_c6_carried_reachable_envelope(
        reference, state=_state(), epi_states=ROWS, timestep=2., max_intersections=1,
    )
    assert limited.status == "resource_limit" and len(limited.iterations) == 1
    assert limited.retained_zones == limited.domain_zones
    assert limited.intersections == 1 and len(limited.retained_zones) == 729
    assert limited.domain_confined_paths_covered and limited.clipped_forward_inclusion_certified


def test_final_equality_requires_a_complete_image_and_never_uses_a_partial_layer(reference, result):
    limited = owner.derive_c6_carried_reachable_envelope(
        reference, state=_state(), epi_states=ROWS, timestep=2., max_intersections=result.intersections - 1,
    )
    assert limited.status == "resource_limit"
    assert tuple(item.zone_count for item in limited.iterations) == (729, 46, 10, 4)
    assert limited.retained_zones == result.retained_zones
    assert limited.intersections == limited.max_intersections
    complete = owner.derive_c6_carried_reachable_envelope(
        reference, state=_state(), epi_states=ROWS, timestep=2., max_intersections=result.intersections,
    )
    assert complete.status == "fixed_point" and complete.iterations == result.iterations


def test_actual_six_coordinate_carry_origin_is_preserved_in_zero_pressure_domain(reference):
    quantum = F(1, 2**3222)
    state = NodalRemainderState((.5,) * 6, tuple(i * quantum for i in range(6)), .375, .625)
    result = owner.derive_c6_carried_reachable_envelope(
        reference, state=state, epi_states=(state.epi,), timestep=1.,
    )
    assert result.status == "fixed_point" and result.intersections == 1
    assert result.grid_quantum == quantum and result.affine_origin == state.exact_epi
    assert len(set(result.affine_origin)) == 6 and result.state is state
    assert result.retained_zones == result.domain_zones and result.pressures == ((0.,) * 6,)


def test_explicit_singleton_domain_keeps_only_the_declared_conditional_path(reference):
    state = _state()
    point = owner.C6CarriedForwardZone(state.epi, ((0,) * 7,) * 7)
    result = owner.derive_c6_carried_reachable_envelope(
        reference, state=state, epi_states=ROWS, timestep=2., domain_zones=(point,),
    )
    assert result.status == "fixed_point" and result.retained_zones == (point,)
    assert result.clipped_forward_inclusion_certified and not result.conditional_boundedness_certified
    assert _image(ORIGIN) != ORIGIN


def test_forged_reference_caches_cannot_change_the_reachable_geometry(reference, result):
    forged = replace(reference, sources=(19.,) * 6, rows=(), epi_quantum=F(5), gradient_quantum=F(11))
    rebuilt = owner.derive_c6_carried_reachable_envelope(
        forged, state=_state(), epi_states=ROWS, timestep=2.,
    )
    assert rebuilt.reference == reference and rebuilt.pressures == result.pressures
    assert rebuilt.retained_zones == result.retained_zones and rebuilt.iterations == result.iterations


@pytest.mark.parametrize("tamper", ("origin_absent", "nonclosed", "outside_rn", "phase", "carry"))
def test_invalid_domain_or_primitive_is_rejected(reference, result, tamper):
    args = dict(state=_state(), epi_states=ROWS, timestep=2., domain_zones=result.domain_zones)
    if tamper == "origin_absent":
        args["domain_zones"] = tuple(z for z in result.domain_zones if z.epi != args["state"].epi)
    elif tamper in ("nonclosed", "outside_rn"):
        zone = result.domain_zones[0]
        b = [list(row) for row in zone.bounds]
        if tamper == "nonclosed":
            b[0][1] += 1
        else:
            b = owner._dbm_box(tuple(-10 for _ in range(6)), tuple(10 for _ in range(6)))
        args["domain_zones"] = (replace(zone, bounds=tuple(map(tuple, b))),) + result.domain_zones[1:]
    elif tamper == "phase":
        reference = replace(reference, source=replace(reference.source, phase=(float("nan"),) * 6))
    else:
        args["state"] = replace(args["state"], remainder=(DELTA / 3,) + (F(0),) * 5)
    with pytest.raises((TypeError, ValueError)):
        owner.derive_c6_carried_reachable_envelope(reference, **args)


@pytest.mark.parametrize("guard,value", (("max_cells", True), ("max_cells", 1),
                                         ("max_intersections", 0), ("max_intersections", 1.5)))
def test_computational_guards_fail_closed(reference, guard, value):
    with pytest.raises((TypeError, ValueError)):
        owner.derive_c6_carried_reachable_envelope(
            reference, state=_state(), epi_states=ROWS, timestep=2., **{guard: value},
        )
