"""Exact finite oracles and scoped exclusions for carried C6 excursions."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics.c6_carried_excursion import (
    derive_c6_carried_excursion_exclusion,
    derive_c6_carried_mode_excursion_exclusion,
)
from tnfr.physics.c6_carried_viability import (
    C6CarriedForwardZone,
    _dbm_box,
    _dbm_close,
    _prepare_carried_family,
    derive_c6_carried_forward_envelope,
)
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

DELTA = F(1, 2**54)
Q = DELTA / 4
LOW, HIGH = float(F(1, 2) - 4 * DELTA), float(F(1, 2) - 3 * DELTA)
ROWS = tuple(product((LOW, HIGH), repeat=6))
A, B, C = (LOW, HIGH) * 3, (HIGH, LOW) * 3, (LOW,) * 6
U0, UT = (0, 5, 0, 5, 0, 5), (0, 4, 1, 5, 0, 5)
W = (1, -1, 0, 0, 0, 0)
PHASE = tuple(
    map(
        float.fromhex,
        (
            "0x1.0b8fb3e3956cbp-55",
            "0x1.0c152382d7365p+0",
            "0x1.0c152382d7365p+1",
            "0x1.921fb54442d18p+1",
            "0x1.0c152382d7365p+2",
            "0x1.4f1a6c638d03fp+2",
        ),
    )
)


def _state_u(point):
    exact = tuple(F(LOW) + Q * x for x in point)
    epi = tuple(map(float, exact))
    return NodalRemainderState(
        epi, tuple(x - F(y) for x, y in zip(exact, epi)), 0.375, 0.625
    )


def _zone_u(row, lower, upper, origin):
    return C6CarriedForwardZone(
        row,
        _dbm_box(
            tuple(x - y for x, y in zip(lower, origin)),
            tuple(x - y for x, y in zip(upper, origin)),
        ),
    )


def _tiny_inputs(origin=U0, target=UT, domain_rows=(A, B), weights=W):
    state = _state_u(origin)
    zones = tuple(
        _zone_u(
            row,
            tuple(-2 if x == LOW else 3 for x in row),
            tuple(2 if x == LOW else 5 for x in row),
            origin,
        )
        for row in domain_rows
    )
    return dict(
        state=state,
        epi_states=ROWS,
        timestep=0.5,
        active_epi_states=(A,),
        weights=weights,
        domain_zones=zones,
        target_regions=(_zone_u(_state_u(target).epi, target, target, origin),),
    )


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


@pytest.fixture(scope="module")
def tiny(reference):
    return derive_c6_carried_excursion_exclusion(reference, **_tiny_inputs())


def _assert_extremum(item):
    point = item.point + (0,)
    assert all(
        point[i] - point[j] <= item.zone.bounds[i][j]
        for i in range(7)
        for j in range(7)
    )
    assert sum(w * x for w, x in zip(item.weights, item.point)) == item.value
    sign = 1 if item.sense == "upper" else -1
    expected = tuple(sign * w for w in item.weights + (-sum(item.weights),))
    balance, cost = [0] * 7, 0
    for i, j, amount in item.dual_flow:
        assert type(amount) is int and amount > 0
        balance[i] += amount
        balance[j] -= amount
        cost += amount * item.zone.bounds[i][j]
    assert tuple(balance) == expected and sign * cost == item.value


def test_canonical_finite_oracle_verifies_all_ingresses_and_exact_extrema(
    reference, tiny
):
    for row in ROWS:
        bit = tuple(int(x == HIGH) for x in row)
        expected = tuple(
            DELTA / 2 * (bit[(i - 1) % 6] + bit[(i + 1) % 6] - 2 * bit[i])
            for i in range(6)
        )
        assert (
            tuple(map(F, _observe_rebuilt_c6_pressure_lattice(reference, row).pressure))
            == expected
        )
    active = tuple(product(*(range(-2, 3) if x == LOW else range(3, 6) for x in A)))
    inactive = tuple(product(*(range(-2, 3) if x == LOW else range(3, 6) for x in B)))
    assert len(active) == len(inactive) == 3375
    active_set = set(active)
    ingress = {
        tuple(x + (-2 if i % 2 == 0 else 2) for i, x in enumerate(point))
        for point in inactive
    }
    ingress &= active_set
    assert len(ingress) == 64

    def potential(point):
        return sum(w * (x - y) for w, x, y in zip(W, point, U0))

    assert (min(map(potential, active)), max(map(potential, active))) == (-2, 4)
    assert (min(map(potential, ingress)), max(map(potential, ingress))) == (2, 4)
    assert tiny.grid_quantum == Q and tiny.minimum_drift == 4
    assert tiny.target_upper == 1 and tiny.ingress_lower == 2
    assert len(tiny.ingress) == 1
    lower, upper = tiny.domain_extrema[0]
    assert (lower.value, upper.value) == (-2, 4)
    for item in (lower, upper, tiny.target_extrema[0], tiny.ingress[0].lower):
        _assert_extremum(item)


def test_one_shared_step_clears_the_derived_initial_window_without_reset(tiny):
    assert tiny.status == "cleared_initial_budget" and tiny.prefix_deadline == 1
    assert tiny.observed_prefix_steps == 1
    assert tiny.prefix_steps[0].before is tiny.state
    assert tiny.endpoint == _state_u((2, 3, 2, 3, 2, 3))
    assert tiny.endpoint_potential == 4 > tiny.target_upper
    assert tiny.exact_total_area == tuple(
        Q * (2 if i % 2 == 0 else -2) for i in range(6)
    )
    assert tiny.nodal_balance_residual == (F(0),) * 6
    assert tiny.origin_path_within_domain_excluded and not tiny.actual_target_reached
    assert tiny.state.remainder == (F(0), Q) * 3


def test_a_touching_ingress_bound_is_inconclusive_and_does_not_execute(
    reference, monkeypatch
):
    import tnfr.physics.c6_carried_excursion as owner

    def no_step(*args, **kwargs):
        raise AssertionError("a failed strict ingress premise must not run a prefix")

    monkeypatch.setattr(owner, "advance_nodal_remainder", no_step)
    result = derive_c6_carried_excursion_exclusion(
        reference, **_tiny_inputs(target=(1,) + UT[1:])
    )
    assert result.ingress_lower == result.target_upper == 2
    assert result.status == "ingress_gap_not_strict" and result.prefix_steps == ()
    assert (
        not result.origin_path_within_domain_excluded
        and not result.actual_target_reached
    )


@pytest.mark.parametrize("weights", ((0,) * 6, tuple(-w for w in W)))
def test_nonpositive_drift_never_becomes_an_exclusion(reference, weights):
    result = derive_c6_carried_excursion_exclusion(
        reference, **_tiny_inputs(weights=weights)
    )
    assert result.minimum_drift <= 0 and result.status == "nonpositive_drift"
    assert result.prefix_deadline is None and result.prefix_steps == ()
    assert not result.origin_path_within_domain_excluded


def test_nonzero_sum_weights_use_the_fixed_zero_coordinate_in_the_exact_dual(reference):
    result = derive_c6_carried_excursion_exclusion(
        reference, **_tiny_inputs(weights=(1, 0, 0, 0, 0, 0))
    )
    assert (
        result.minimum_drift == 2
        and result.target_upper == 0
        and result.ingress_lower == 1
    )
    assert result.status == "cleared_initial_budget"
    for item in (
        result.domain_extrema[0]
        + result.target_extrema
        + tuple(x.lower for x in result.ingress)
    ):
        _assert_extremum(item)
        assert any(6 in (i, j) for i, j, _ in item.dual_flow)


@pytest.mark.parametrize(
    ("origin", "status"),
    (
        ((4, 0, 4, 0, 4, 0), "initially_inactive"),
        ((2, 5, 0, 5, 0, 5), "initially_above_targets"),
    ),
)
def test_no_prefix_is_needed_for_an_inactive_or_already_cleared_origin(
    reference, origin, status
):
    result = derive_c6_carried_excursion_exclusion(
        reference, **_tiny_inputs(origin=origin)
    )
    assert result.status == status and result.prefix_steps == ()
    assert result.endpoint is result.state and result.origin_path_within_domain_excluded


def test_target_membership_is_reported_instead_of_being_hidden_by_proof_gates(
    reference,
):
    result = derive_c6_carried_excursion_exclusion(reference, **_tiny_inputs(origin=UT))
    assert result.status == "target_reached" and result.actual_target_reached
    assert result.prefix_steps == () and not result.origin_path_within_domain_excluded


@pytest.mark.parametrize(
    ("domain_rows", "status"),
    (
        ((A, B), "prefix_left_domain"),
        ((A, B, C), "initial_visit_ended"),
    ),
)
def test_leaving_the_initial_visit_preserves_the_exact_scope(
    reference, domain_rows, status
):
    result = derive_c6_carried_excursion_exclusion(
        reference,
        **_tiny_inputs(origin=(0, 4, 0, 4, 0, 4), domain_rows=domain_rows),
    )
    assert result.status == status and result.observed_prefix_steps == 1
    assert result.endpoint == _state_u((2,) * 6)
    assert result.origin_path_within_domain_excluded
    assert (
        not result.conditional_invariance_certified
        and not result.conditional_boundedness_certified
    )
    assert (
        not result.future_runtime_certified
        and not result.asymptotic_convergence_certified
    )


def test_exact_band_departure_keeps_the_unexecuted_candidate_separate(reference):
    target_epi = (LOW, HIGH, LOW, LOW, LOW, HIGH)
    origin_bits, target_bits = (0, 1) * 3, (0, 1, 0, 0, 0, 1)
    state = NodalRemainderState(A, (F(0),) * 6, LOW, HIGH)
    point = tuple(x - y for x, y in zip(target_bits, origin_bits))
    origin_zone = C6CarriedForwardZone(A, _dbm_box((0,) * 6, (0,) * 6))
    target_zone = C6CarriedForwardZone(target_epi, _dbm_box(point, point))
    result = derive_c6_carried_excursion_exclusion(
        reference,
        state=state,
        epi_states=ROWS,
        timestep=2.0,
        active_epi_states=(A, target_epi),
        weights=W,
        domain_zones=(origin_zone, target_zone),
        target_regions=(target_zone,),
    )
    assert result.status == "prefix_band_exit" and result.prefix_steps == ()
    assert result.endpoint is state and result.exact_total_area == (F(0),) * 6
    assert result.terminal_exact_candidate[0] == F(LOW) + 2 * DELTA > F(HIGH)
    assert (
        result.origin_path_within_domain_excluded and not result.actual_target_reached
    )


def test_stale_pressure_caches_are_rebuilt_and_the_source_carry_is_preserved(
    reference, tiny
):
    forged = replace(
        reference,
        sources=(99.0,) * 6,
        rows=(),
        epi_quantum=F(7),
        gradient_quantum=F(11),
    )
    result = derive_c6_carried_excursion_exclusion(forged, **_tiny_inputs())
    assert result.reference == reference and result.pressures == tiny.pressures
    assert (
        result.endpoint == tiny.endpoint
        and result.state.remainder == tiny.state.remainder
    )


@pytest.mark.parametrize("tamper", ("phase", "carry", "domain", "target", "active"))
def test_invalid_primitive_or_domain_inputs_fail_closed(reference, tamper):
    inputs = _tiny_inputs()
    if tamper == "phase":
        reference = replace(
            reference, source=replace(reference.source, phase=(float("nan"),) * 6)
        )
    elif tamper == "carry":
        inputs["state"] = replace(
            inputs["state"], remainder=(Q / 3,) + inputs["state"].remainder[1:]
        )
    elif tamper == "domain":
        zone = inputs["domain_zones"][0]
        bounds = [list(row) for row in zone.bounds]
        bounds[0][0] = -1
        inputs["domain_zones"] = (
            replace(zone, bounds=tuple(map(tuple, bounds))),
        ) + inputs["domain_zones"][1:]
    elif tamper == "target":
        inputs["target_regions"] = (replace(inputs["target_regions"][0], epi=B),)
    else:
        inputs["active_epi_states"] = (A, A)
    with pytest.raises((ValueError, TypeError)):
        derive_c6_carried_excursion_exclusion(reference, **inputs)


@pytest.fixture(scope="module")
def actual_domain():
    # Captured B47 primitives only; reconstruct the complete RN/pair envelope
    # from canonical pressure instead of reading any ignored research artifact.
    terms = (
        (-142714449118892709, 112),
        (286627028123761141, 113),
        (113137032337055403, 112),
        (-17740104902540123, 109),
        (-18010438065504949, 108),
        (14060687418323491, 111),
    )
    epi = tuple(float(F(1, 2) + k * DELTA) for k in (-3, -1, 2, 0, 8, -5))
    state = NodalRemainderState(epi, tuple(F(n, 2**e) for n, e in terms), 0.375, 0.625)
    reference = derive_c6_pressure_lattice(
        phase=PHASE,
        epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    low, high = (-4, -1, 2, -1, 6, -6), (-3, 0, 4, 0, 8, -5)
    rows = tuple(
        tuple(
            float(F(1, 2) + (high[i] if mask & (1 << i) else low[i]) * DELTA)
            for i in range(6)
        )
        for mask in range(64)
    )
    envelope = derive_c6_carried_forward_envelope(
        reference,
        state=state,
        epi_states=rows,
        timestep=0.0625,
        max_intersections=250000,
    )
    assert (
        envelope.completed_image_layers == 259
        and envelope.iterations[-1].origin_retained
    )
    _, origin, _, _, _, areas, cells = _prepare_carried_family(
        reference,
        state=state,
        epi_states=rows,
        timestep=0.0625,
        row_limit=4096,
        row_limit_label="max_cells",
    )
    grid = envelope.grid_quantum
    lower = tuple(
        min(((cell.lower[i] - origin[i]) / grid).__ceil__() for cell in cells)
        for i in range(6)
    )
    upper = tuple(
        max(((cell.upper[i] - origin[i]) / grid).__floor__() for cell in cells)
        for i in range(6)
    )
    return reference, envelope, areas, lower, upper


def _actual_inputs(actual_domain, node=2):
    _reference, envelope, areas, lower, upper = actual_domain
    targets = []
    for zone in envelope.retained_zones:
        index = envelope.epi_states.index(zone.epi)
        shift = areas[index][node] / envelope.grid_quantum
        assert shift.denominator == 1
        bounds = [list(row) for row in zone.bounds]
        if node == 2:
            bounds[node][6] = min(bounds[node][6], lower[node] - int(shift) - 1)
        else:
            bounds[6][node] = min(bounds[6][node], -(upper[node] - int(shift) + 1))
        closed = _dbm_close(bounds)
        if closed is not None:
            targets.append(C6CarriedForwardZone(zone.epi, closed))
    assert len(targets) == 8
    return dict(
        state=envelope.state,
        epi_states=envelope.epi_states,
        timestep=0.0625,
        domain_zones=envelope.retained_zones,
        target_regions=tuple(targets),
        active_epi_states=tuple(
            row
            for mask, row in enumerate(envelope.epi_states)
            if bool(mask & (1 << node)) == (node == 3)
        ),
        weights=(-2, 0, 2, 1, 0, -1) if node == 2 else (1, -1, -3, -5, 5, 3),
    )


@pytest.fixture(scope="module")
def actual_excursion(actual_domain):
    return derive_c6_carried_excursion_exclusion(
        actual_domain[0], **_actual_inputs(actual_domain)
    )


def test_actual_node_two_excursion_closes_at_first_strict_budget_crossing(
    actual_domain, actual_excursion
):
    result = actual_excursion
    assert result.minimum_drift == 1418638309884336
    assert (
        result.target_upper
        == 237475340900157211
        < result.ingress_lower
        == 449419196811482809
    )
    assert result.prefix_deadline == 168 and result.observed_prefix_steps == 56
    assert (
        result.status == "cleared_initial_budget"
        and result.origin_path_within_domain_excluded
    )
    assert result.endpoint_potential == 247713108641769776
    previous = result.state
    for step in result.prefix_steps:
        assert step.before == previous
        pressure = _observe_rebuilt_c6_pressure_lattice(
            actual_domain[0], step.before.epi
        ).pressure
        assert step.exact_increment == tuple(F(0.0625) * F(p) for p in pressure)
        assert step == advance_nodal_remainder(
            step.before, timestep=0.0625, capacity=(1.0,) * 6, pressure=pressure
        )
        coordinates = tuple(
            (x - start) / result.grid_quantum
            for x, start in zip(step.before.exact_epi, result.affine_origin)
        )
        assert (
            sum(w * x for w, x in zip(result.weights, coordinates))
            <= result.target_upper
        )
        previous = step.after
    assert previous == result.endpoint and result.nodal_balance_residual == (F(0),) * 6
    assert (
        not result.conditional_boundedness_certified
        and not result.future_runtime_certified
    )


def test_insufficient_prefix_resource_does_not_inherit_a_later_positive_result(
    actual_domain,
):
    result = derive_c6_carried_excursion_exclusion(
        actual_domain[0],
        **_actual_inputs(actual_domain),
        max_prefix_steps=55,
    )
    assert (
        result.status == "prefix_resource_limit" and result.observed_prefix_steps == 55
    )
    assert (
        result.endpoint_potential <= result.target_upper
        and result.prefix_deadline == 168
    )
    assert (
        not result.origin_path_within_domain_excluded
        and not result.actual_target_reached
    )


def test_actual_node_three_positive_drift_fails_the_necessary_ingress_gate(
    actual_domain,
):
    result = derive_c6_carried_excursion_exclusion(
        actual_domain[0], **_actual_inputs(actual_domain, node=3)
    )
    assert result.minimum_drift == 17393571931258 > 0
    assert (
        result.ingress_lower
        == -4243436431404132151
        < result.target_upper
        == 1133385369716752920
    )
    assert result.status == "ingress_gap_not_strict" and result.prefix_steps == ()
    assert not result.origin_path_within_domain_excluded


def test_a_genuinely_reached_target_during_the_actual_prefix_is_reported(actual_domain):
    reference, envelope, _areas, _lower, _upper = actual_domain
    pressure = _observe_rebuilt_c6_pressure_lattice(
        reference, envelope.state.epi
    ).pressure
    first = advance_nodal_remainder(
        envelope.state, timestep=0.0625, capacity=(1.0,) * 6, pressure=pressure
    )
    point = tuple(
        int((x - y) / envelope.grid_quantum)
        for x, y in zip(first.after.exact_epi, envelope.affine_origin)
    )
    inputs = _actual_inputs(actual_domain)
    inputs["target_regions"] = (
        C6CarriedForwardZone(first.after.epi, _dbm_box(point, point)),
    )
    result = derive_c6_carried_excursion_exclusion(reference, **inputs)
    assert result.status == "target_reached" and result.actual_target_reached
    assert result.prefix_steps == (first,) and result.endpoint == first.after
    assert not result.origin_path_within_domain_excluded


MODE_POINTS = ((0, 0, 0, 0, 1, 1), (1, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1))
MODE_ROWS = tuple(tuple(float(F(LOW) + DELTA * x) for x in p) for p in MODE_POINTS)


def _mode_inputs(offsets=(0, 2, 1), active=(0, 1, 2), domain=(0, 1, 2), target=2):
    origin = MODE_POINTS[0]
    points = tuple(tuple(x - y for x, y in zip(point, origin)) for point in MODE_POINTS)
    zones = tuple(
        C6CarriedForwardZone(row, _dbm_box(point, point))
        for row, point in zip(MODE_ROWS, points)
    )
    return dict(
        state=NodalRemainderState(MODE_ROWS[0], (F(0),) * 6, 0.375, 0.625),
        epi_states=MODE_ROWS,
        timestep=2.0,
        weights=(0,) * 6,
        active_epi_states=tuple(MODE_ROWS[i] for i in active),
        cell_offsets=tuple((MODE_ROWS[i], offsets[i]) for i in active),
        domain_zones=tuple(zones[i] for i in domain),
        target_regions=(zones[target],),
    )


def test_mode_offsets_certify_a_canonical_dag_with_no_positive_common_gradient(
    reference,
):
    # The exact h=2 pure-diffusion integer map has A->B and B,C->outside.
    # These three cells are singleton on the derived increment lattice.
    result = derive_c6_carried_mode_excursion_exclusion(reference, **_mode_inputs())
    expected = tuple(
        tuple(p[i - 1] + p[(i + 1) % 6] - p[i] for i in range(6)) for p in MODE_POINTS
    )
    assert expected == (MODE_POINTS[1], (-1, 1, 1, -1, 1, 1), (1, 0, 0, 0, 1, -1))
    for before, after, pressure in zip(
        MODE_POINTS, expected, result.pressures, strict=True
    ):
        assert tuple(2 * F(p) for p in pressure) == tuple(
            DELTA * (a - b) for a, b in zip(after, before)
        )
    assert result.grid_quantum == DELTA and len(result.active_transitions) == 1
    (edge,) = result.active_transitions
    assert (edge.source_epi, edge.target_epi, edge.potential_increment) == (
        MODE_ROWS[0],
        MODE_ROWS[1],
        2,
    )
    assert edge.intersection == result.domain_zones[1]
    assert (
        result.minimum_drift == 2
        and result.initial_potential == 0
        and result.target_upper == 1
    )
    assert result.ingress == () and result.ingress_lower is None
    assert result.prefix_deadline == 1 and result.observed_prefix_steps == 1
    assert result.status == "cleared_initial_budget" and result.endpoint_potential == 2
    assert (
        result.endpoint.epi == MODE_ROWS[1] and result.endpoint.remainder == (F(0),) * 6
    )
    assert result.prefix_steps[0].before is result.state and not any(
        result.nodal_balance_residual
    )
    assert (
        result.origin_path_within_domain_excluded
        and not result.conditional_boundedness_certified
    )


def test_constant_shift_of_all_offsets_preserves_the_deadline_and_actual_prefix(
    reference,
):
    base = derive_c6_carried_mode_excursion_exclusion(reference, **_mode_inputs())
    shifted = derive_c6_carried_mode_excursion_exclusion(
        reference, **_mode_inputs(offsets=(9, 11, 10))
    )
    assert (
        shifted.initial_potential == 9
        and shifted.target_upper == 10
        and shifted.endpoint_potential == 11
    )
    assert shifted.prefix_deadline == base.prefix_deadline == 1
    assert (
        shifted.prefix_steps == base.prefix_steps
        and shifted.minimum_drift == base.minimum_drift
    )


@pytest.mark.parametrize("offset", (0, -2))
def test_mode_edge_with_zero_or_negative_drift_is_not_certified(reference, offset):
    result = derive_c6_carried_mode_excursion_exclusion(
        reference, **_mode_inputs(offsets=(0, offset, 1))
    )
    assert result.minimum_drift == offset and result.status == "nonpositive_drift"
    assert result.prefix_steps == () and not result.origin_path_within_domain_excluded


@pytest.mark.parametrize(
    ("domain", "status"),
    (((0, 1, 2), "initial_visit_ended"), ((0, 2), "prefix_left_domain")),
)
def test_no_internal_edge_has_a_one_step_visit_bound_without_an_invented_drift(
    reference, domain, status
):
    result = derive_c6_carried_mode_excursion_exclusion(
        reference,
        **_mode_inputs(active=(0, 2), domain=domain),
    )
    assert result.minimum_drift is None and result.active_transitions == ()
    assert result.prefix_deadline == 1 and result.observed_prefix_steps == 1
    assert result.endpoint.epi == MODE_ROWS[1] and result.endpoint_potential is None
    assert result.status == status and result.origin_path_within_domain_excluded
    assert (
        not result.conditional_invariance_certified
        and not result.future_runtime_certified
    )


def test_ingress_uses_the_destination_offset_and_strict_comparison(reference):
    excluded = derive_c6_carried_mode_excursion_exclusion(
        reference, **_mode_inputs(active=(1, 2))
    )
    assert (
        excluded.initial_potential is None and excluded.status == "initially_inactive"
    )
    assert (
        excluded.ingress_lower == 2
        and excluded.target_upper == 1
        and len(excluded.ingress) == 1
    )
    assert excluded.ingress[0].lower.value == 0
    assert excluded.ingress[0].target_epi == MODE_ROWS[1]
    assert excluded.prefix_steps == () and excluded.origin_path_within_domain_excluded
    touching = derive_c6_carried_mode_excursion_exclusion(
        reference,
        **_mode_inputs(active=(1, 2), offsets=(0, 2, 2)),
    )
    assert touching.ingress_lower == touching.target_upper == 2
    assert (
        touching.status == "ingress_gap_not_strict"
        and not touching.origin_path_within_domain_excluded
    )


def test_mode_resource_limit_keeps_the_unfinished_initial_visit_inconclusive(reference):
    result = derive_c6_carried_mode_excursion_exclusion(
        reference,
        **_mode_inputs(offsets=(0, 2, 3)),
        max_prefix_steps=1,
    )
    assert result.minimum_drift == 2 and result.prefix_deadline == 2
    assert (
        result.status == "prefix_resource_limit" and result.observed_prefix_steps == 1
    )
    assert result.endpoint_potential == 2 <= result.target_upper == 3
    assert not result.origin_path_within_domain_excluded


def test_mode_initial_offset_above_the_target_requires_no_prefix(reference):
    result = derive_c6_carried_mode_excursion_exclusion(
        reference, **_mode_inputs(offsets=(3, 5, 1))
    )
    assert result.status == "initially_above_targets" and result.initial_potential == 3
    assert (
        result.prefix_deadline == 0
        and result.prefix_steps == ()
        and result.endpoint is result.state
    )
    assert result.origin_path_within_domain_excluded


def test_a_reached_mode_target_remains_actual_evidence_not_an_exclusion(reference):
    result = derive_c6_carried_mode_excursion_exclusion(
        reference, **_mode_inputs(target=1)
    )
    assert result.status == "target_reached" and result.actual_target_reached
    assert result.observed_prefix_steps == 1 and result.endpoint.epi == MODE_ROWS[1]
    assert not result.origin_path_within_domain_excluded


def test_zero_offsets_reuse_the_old_positive_oracle_and_preserve_nonzero_carry(
    reference, tiny
):
    result = derive_c6_carried_mode_excursion_exclusion(reference, **_tiny_inputs())
    assert result.cell_offsets == ((A, 0),) and result.initial_potential == 0
    assert result.minimum_drift == tiny.minimum_drift and result.status == tiny.status
    assert result.prefix_steps == tiny.prefix_steps and result.endpoint == tiny.endpoint
    assert result.state.remainder == tiny.state.remainder != (F(0),) * 6


@pytest.mark.parametrize(
    "tamper", ("missing", "duplicate", "inactive", "bool", "float", "fraction", "list")
)
def test_mode_offsets_fail_closed_on_incomplete_or_nonexact_input(reference, tamper):
    args = _mode_inputs()
    values = args["cell_offsets"]
    if tamper == "missing":
        args["cell_offsets"] = values[:-1]
    elif tamper == "duplicate":
        args["cell_offsets"] = (values[0], values[0], values[2])
    elif tamper == "inactive":
        args["cell_offsets"] = (((LOW,) * 6, 0),) + values[1:]
    elif tamper == "list":
        args["cell_offsets"] = list(values)
    else:
        value = {"bool": True, "float": 0.0, "fraction": F(0)}[tamper]
        args["cell_offsets"] = ((values[0][0], value),) + values[1:]
    with pytest.raises((ValueError, TypeError)):
        derive_c6_carried_mode_excursion_exclusion(reference, **args)


def test_mode_pressure_and_grid_caches_are_rederived_before_the_edge_proof(reference):
    forged = replace(
        reference,
        sources=(100.0,) * 6,
        rows=(),
        epi_quantum=F(7),
        gradient_quantum=F(13),
    )
    good = derive_c6_carried_mode_excursion_exclusion(reference, **_mode_inputs())
    result = derive_c6_carried_mode_excursion_exclusion(forged, **_mode_inputs())
    assert result.reference == reference and result.grid_quantum == good.grid_quantum
    assert (
        result.active_transitions == good.active_transitions
        and result.prefix_steps == good.prefix_steps
    )
