"""Independent counted child-swap chart and ideal hybrid-algebra controls.

No flow, operator execution or first-hit search is performed. The phase
calculations check the ideal circle at supplied states, not binary64 runtime
equivalence or a numerical trajectory through an event.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_child_quotient as owner

EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * left + slot, 9 * right + slot)
    for left, right in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
ORBITS = (
    tuple((j,) for j in range(9))
    + tuple((9 + j, 18 + j) for j in range(9))
    + tuple((j,) for j in range(27, 45))
)
NODE_ORBIT = tuple(
    next(a for a, orbit in enumerate(ORBITS) if i in orbit) for i in range(45)
)
MASSES = tuple(Q(len(orbit)) for orbit in ORBITS)
REPRESENTATIVES = tuple(orbit[0] for orbit in ORBITS)
ZERO = (Q(0),) * 36
UNIT = (Q(1),) * 36
FORM = tuple(Q((5 * i) % 17 - 8, 4096) for i in range(36))
OFFSETS = tuple(Q((7 * i) % 11 - 5, 4096) for i in range(36))
CAPACITIES = tuple(Q(4 + i % 5, 4) for i in range(36))


def _lift(values):
    return tuple(values[a] for a in NODE_ORBIT)


def _incidence(values):
    gradient = [Q(0)] * 45
    for left, right in EDGES:
        gradient[left] += values[left] - values[right]
        gradient[right] += values[right] - values[left]
    return tuple(gradient)


def _coarse_edges():
    weights = {}
    for left, right in EDGES:
        pair = tuple(sorted((NODE_ORBIT[left], NODE_ORBIT[right])))
        weights[pair] = weights.get(pair, Q(0)) + 1
    return tuple(
        (left, right, weight) for (left, right), weight in sorted(weights.items())
    )


COARSE_EDGES = _coarse_edges()


def _counts():
    result = []
    for representative in REPRESENTATIVES:
        row = [0] * 36
        for left, right in EDGES:
            if representative in (left, right):
                neighbor = right if left == representative else left
                row[NODE_ORBIT[neighbor]] += 1
        result.append(tuple(row))
    return tuple(result)


COUNTS = _counts()


@pytest.fixture(scope="module", autouse=True)
def no_event_or_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child quotient controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def factors():
    defaults = canonical_glyph_factor_defaults()
    return (
        Q(defaults["EN_mix"]),
        Q(DEFAULT_PHASE_LOCKING_COEFFICIENT),
        Q(defaults["SHA_vf_factor"]),
    )


def _mp(context, value):
    if isinstance(value, (Q, int)):
        return context.mpf(value.numerator) / int(value.denominator)
    return value


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _fields(context, form, offsets, capacities, *, coarse):
    """Independent full-law sums; coarse energy uses masses, rows per member."""
    size = 36 if coarse else 45
    edges = COARSE_EDGES if coarse else tuple((a, b, Q(1)) for a, b in EDGES)
    masses = MASSES if coarse else (Q(1),) * 45
    alpha = 2 * context.pi / 9
    gamma = 1 / (1023 * context.pi)
    phases = tuple(
        alpha * (i % 9 - 4) + _mp(context, value) for i, value in enumerate(offsets)
    )
    q = [context.mpf(0)] * size
    current = [context.mpf(0)] * size
    energy = context.mpf(0)
    for left, right, weight in edges:
        weight = _mp(context, weight)
        difference = _mp(context, form[left]) - _mp(context, form[right])
        gap = phases[right] - phases[left]
        q[left] += weight * difference
        q[right] -= weight * difference
        current[left] += weight * context.sin(gap)
        current[right] -= weight * context.sin(gap)
        energy += weight * (difference**2 / 2 + 1 - context.cos(gap))
    q = tuple(value / _mp(context, mass) for value, mass in zip(q, masses))
    current = tuple(value / _mp(context, mass) for value, mass in zip(current, masses))
    x_rate = tuple(
        _mp(context, nu) * (-value + gamma * sine) / 4
        for nu, value, sine in zip(capacities, q, current)
    )
    phase_rate = tuple(
        _mp(context, nu) * gamma * value / 4 for nu, value in zip(capacities, q)
    )
    loss = context.fsum(
        _mp(context, mass) * _mp(context, nu) * value**2 / 4
        for mass, nu, value in zip(masses, capacities, q)
    )
    return q, current, x_rate, phase_rate, energy, loss


def _ideal_word(context, form, offsets, capacities, counts, factors):
    """Same-snapshot primary resets, with counted neighbors and real phase lifts."""
    beta, coefficient, kappa = (_mp(context, value) for value in factors)
    forms = tuple(_mp(context, value) for value in form)
    phases = tuple(_mp(context, value) for value in offsets)
    capacity = list(_mp(context, value) for value in capacities)
    after_form, after_phase = list(forms), list(phases)
    alpha = 2 * context.pi / 9
    for target in (0, 1):
        degree = sum(counts[target])
        neighbor_mean = (
            context.fsum(n * forms[j] for j, n in enumerate(counts[target])) / degree
        )
        after_form[target] = (1 - beta) * forms[target] + beta * neighbor_mean
        relative_resultant = context.fsum(
            n
            * context.exp(
                context.j
                * (alpha * ((j % 9) - (target % 9)) + phases[j] - phases[target])
            )
            for j, n in enumerate(counts[target])
            if n
        )
        after_phase[target] += coefficient * context.arg(relative_resultant)
        capacity[target] *= kappa
    return tuple(after_form), tuple(after_phase), tuple(capacity)


def _fine_counts():
    rows = [[0] * 45 for _ in range(45)]
    for left, right in EDGES:
        rows[left][right] += 1
        rows[right][left] += 1
    return tuple(tuple(row) for row in rows)


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_source_child_quotient()


def _state(**overrides):
    return owner._lift_organized_source_child_quotient(
        **{
            "form_coordinates": FORM,
            "phase_offset_coordinates": OFFSETS,
            "capacity_coordinates": UNIT,
            **overrides,
        }
    )


def test_geometry_retains_per_member_counts_and_aggregate_energy_weights(geometry):
    assert geometry.blocks == ORBITS
    assert geometry.representatives == REPRESENTATIVES
    assert geometry.node_blocks == NODE_ORBIT
    assert geometry.orbit_masses == MASSES
    assert len(geometry.fine_geometry.weighted_edges) == len(EDGES) == 90
    fine_edges = {
        (min(a, b), max(a, b), w) for a, b, w in geometry.fine_geometry.weighted_edges
    }
    assert fine_edges == {(min(a, b), max(a, b), Q(1)) for a, b in EDGES}
    assert geometry.multiplicity == COUNTS
    assert geometry.support_degrees == tuple(sum(row) for row in COUNTS)
    coarse_edges = {(min(a, b), max(a, b), w) for a, b, w in geometry.weighted_edges}
    assert coarse_edges == set(COARSE_EDGES)
    assert len(coarse_edges) == 63
    assert sum((w for _, _, w in coarse_edges), Q(0)) == 90
    for a in range(36):
        assert geometry.weighted_degrees[a] == MASSES[a] * sum(COUNTS[a])
        for b in range(36):
            assert MASSES[a] * COUNTS[a][b] == MASSES[b] * COUNTS[b][a]
    assert sum(n > 0 for n in COUNTS[0]) == 4
    assert geometry.support_degrees[:2] == (5, 5)


def test_lift_and_strict_restriction_are_inverse_without_averaging():
    result = _state(
        form_coordinates=(value for value in FORM),
        phase_offset_coordinates=list(OFFSETS),
        capacity_coordinates=(1,) * 36,
    )
    assert result.form == _lift(FORM)
    assert result.phase_offsets == _lift(OFFSETS)
    assert result.capacities == (Q(1),) * 45
    for values in (
        result.form_coordinates,
        result.phase_offset_coordinates,
        result.capacity_coordinates,
    ):
        assert all(type(value) is Q for value in values)
    restricted = owner._restrict_organized_source_child_quotient(
        form=result.form,
        phase_offsets=result.phase_offsets,
        capacities=result.capacities,
    )
    assert restricted.form_coordinates == FORM
    assert restricted.phase_offset_coordinates == OFFSETS
    assert restricted.capacity_coordinates == UNIT
    assert restricted.form == result.form
    assert restricted.phase_offsets == result.phase_offsets


@pytest.mark.parametrize("capacity_case", ("unit", "nonuniform", "post_silence"))
def test_full_nonlinear_rows_energy_loss_and_weighted_means(mp, factors, capacity_case):
    capacities = {
        "unit": UNIT,
        "nonuniform": CAPACITIES,
        "post_silence": (factors[2],) * 2 + UNIT[2:],
    }[capacity_case]
    result = owner._observe_organized_source_child_quotient(
        form_coordinates=FORM,
        phase_offset_coordinates=OFFSETS,
        capacity_coordinates=capacities,
    )
    coarse = _fields(mp, FORM, OFFSETS, capacities, coarse=True)
    fine = _fields(mp, _lift(FORM), _lift(OFFSETS), _lift(capacities), coarse=False)
    q = _incidence(_lift(FORM))
    assert result.form_gradient == tuple(q[i] for i in REPRESENTATIVES)
    gamma = 1 / (1023 * mp.pi)
    for a in range(36):
        _contains(mp, result.phase_current_bounds[a], coarse[1][a])
        pressure = (-coarse[0][a] + gamma * coarse[1][a]) / 4
        _contains(mp, result.pressure_tau_bounds[a], pressure)
        for name, row in (("form", coarse[2]), ("phase", coarse[3])):
            for clock, factor in (
                ("tau", mp.mpf(1)),
                ("original", mp.mpf(1023) / 1024),
                ("slow", 2 / gamma**2),
            ):
                _contains(
                    mp,
                    getattr(result, f"{name}_{clock}_rate_bounds")[a],
                    row[a] * factor,
                )
        for i in ORBITS[a]:
            for channel in range(4):
                assert abs(fine[channel][i] - coarse[channel][a]) < mp.mpf("1e-85")
    _contains(mp, result.joint_storage_bounds, fine[4])
    assert abs(coarse[4] - fine[4]) < mp.mpf("1e-85")
    loss = sum((nu * value**2 / 4 for nu, value in zip(_lift(capacities), q)), Q(0))
    assert result.dissipation_tau == loss
    assert abs(coarse[5] - _mp(mp, loss)) < mp.mpf("1e-85")
    _contains(mp, result.storage_tau_rate_bounds, -_mp(mp, loss))
    weights = tuple(mass / nu for mass, nu in zip(MASSES, capacities))
    total_weight = sum(weights, Q(0))
    form_total = sum((weight * value for weight, value in zip(weights, FORM)), Q(0))
    phase_total = sum((weight * value for weight, value in zip(weights, OFFSETS)), Q(0))
    assert result.inverse_capacity_weight_total == total_weight
    assert result.inverse_capacity_weighted_form_total == form_total
    assert result.inverse_capacity_weighted_phase_offset_total == phase_total
    assert result.inverse_capacity_weighted_form_mean == form_total / total_weight
    assert (
        result.inverse_capacity_weighted_phase_offset_mean == phase_total / total_weight
    )
    assert result.fine.inverse_capacity_weighted_form_total == form_total
    assert result.fine.inverse_capacity_weighted_phase_offset_total == phase_total


@pytest.mark.parametrize(
    "field,values",
    (
        ("form_coordinates", (True,) + FORM[1:]),
        ("phase_offset_coordinates", (float("nan"),) + OFFSETS[1:]),
        ("capacity_coordinates", (True,) + UNIT[1:]),
        ("capacity_coordinates", (Q(0),) + UNIT[1:]),
        ("capacity_coordinates", (-Q(1),) + UNIT[1:]),
        ("form_coordinates", FORM[:35]),
        ("phase_offset_coordinates", OFFSETS + (Q(0),)),
    ),
)
def test_invalid_primitives_reject_before_geometry(monkeypatch, field, values):
    def forbidden():
        pytest.fail("Invalid primitive coordinates reached geometry construction")

    monkeypatch.setattr(owner, "_derive_organized_source_child_quotient", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _state(**{field: values})


@pytest.mark.parametrize(
    "field,delta",
    (
        ("form", Q(1, 2**600)),
        ("phase_offsets", Q(1, 2**600)),
        ("capacities", Q(1, 2**600)),
        ("phase_offsets", Q(6.283185307179586)),
    ),
)
def test_restriction_rejects_off_sector_values_instead_of_wrapping_or_averaging(
    field, delta
):
    data = {
        "form": _lift(FORM),
        "phase_offsets": _lift(OFFSETS),
        "capacities": _lift(UNIT),
    }
    changed = list(data[field])
    changed[18] += delta
    data[field] = tuple(changed)
    with pytest.raises(ValueError):
        owner._restrict_organized_source_child_quotient(**data)


def test_chart_retains_both_states_hidden_by_the_recent_guard_summary():
    states, summaries, accelerations = [], [], []
    h = Q(1, 128)
    for negative_slot in (6, 7):
        profile = tuple(
            h if j == 3 else -h if j == negative_slot else Q(0) for j in range(9)
        )
        state = _state(form_coordinates=profile * 4, phase_offset_coordinates=ZERO)
        assert state.form == profile * 5
        q = _incidence(state.form)
        energy = sum(((state.form[a] - state.form[b]) ** 2 / 2 for a, b in EDGES), Q(0))
        loss = sum((value**2 / 4 for value in q), Q(0))
        means = tuple(
            sum(state.form[start : start + 9], Q(0)) / 9 for start in range(0, 45, 9)
        )
        assert energy == 10 * h**2
        assert loss == 15 * h**2
        summaries.append(
            (means, energy, loss, tuple((state.form[i], q[i]) for i in (0, 1)))
        )
        accelerations.append(
            sum((value * rate / 8 for value, rate in zip(q, _incidence(q))), Q(0))
        )
        states.append(state)
    assert summaries[0] == summaries[1]
    assert states[0].form_coordinates != states[1].form_coordinates
    assert states[0].form != states[1].form
    assert accelerations == [Q(105, 4) * h**2, 25 * h**2]


def test_counted_ideal_word_and_changed_capacity_continuation_commute_with_lift(
    mp, factors, geometry
):
    state = _state()
    coarse_after = _ideal_word(mp, FORM, OFFSETS, UNIT, geometry.multiplicity, factors)
    fine_after = _ideal_word(
        mp, state.form, state.phase_offsets, state.capacities, _fine_counts(), factors
    )
    for coarse_row, fine_row in zip(coarse_after, fine_after):
        for lifted, actual in zip(_lift(coarse_row), fine_row):
            assert abs(lifted - actual) < mp.mpf("1e-85")
    assert abs(coarse_after[0][0] - _mp(mp, FORM[0])) > mp.mpf("1e-8")
    assert abs(coarse_after[1][0] - _mp(mp, OFFSETS[0])) > mp.mpf("1e-8")
    assert coarse_after[2][:2] == (_mp(mp, factors[2]),) * 2
    assert coarse_after[2][2:] == (mp.mpf(1),) * 34

    # The same fixed threshold uses the full inherited energy. This compares
    # guard functions only; it does not locate or forecast their first root.
    threshold = mp.mpf(1) / 100000
    carrier_storage = 45 * (1 - mp.cos(2 * mp.pi / 9))
    before = _fields(mp, FORM, OFFSETS, UNIT, coarse=True)
    assert before[4] - carrier_storage > threshold
    for coarse_state, fine_state in (
        ((FORM, OFFSETS, UNIT), (state.form, state.phase_offsets, state.capacities)),
        (coarse_after, fine_after),
    ):
        coarse = _fields(mp, *coarse_state, coarse=True)
        fine = _fields(mp, *fine_state, coarse=False)
        for channel in (2, 3):
            for reduced, complete in zip(_lift(coarse[channel]), fine[channel]):
                assert abs(reduced - complete) < mp.mpf("1e-85")
        assert abs(coarse[4] - fine[4]) < mp.mpf("1e-85")
        assert abs(coarse[5] - fine[5]) < mp.mpf("1e-85")
        fine_guard = fine[4] - carrier_storage - threshold
        coarse_guard = coarse[4] - carrier_storage - threshold
        assert abs(fine_guard - coarse_guard) < mp.mpf("1e-85")


def test_naive_simple_neighbor_reception_loses_a_required_copy(geometry, factors):
    a = Q(1, 100)
    values = ZERO[:9] + (a,) + ZERO[10:]
    counts = geometry.multiplicity[0]
    correct = sum((n * value for n, value in zip(counts, values)), Q(0)) / sum(counts)
    naive = sum((value for n, value in zip(counts, values) if n), Q(0)) / sum(
        bool(n) for n in counts
    )
    assert correct == 2 * a / 5
    assert naive == a / 4
    assert factors[0] * (correct - naive) == factors[0] * 3 * a / 20 > 0


def test_naive_simple_phasor_control_needs_an_off_carrier_state(mp, geometry):
    alpha = 2 * mp.pi / 9
    r = 1 + 2 * mp.cos(alpha)
    for epsilon in (mp.mpf(0), mp.mpf(1) / 100):
        phases = tuple(
            alpha * (i % 9 - 4) + (epsilon if i == 9 else 0) for i in range(36)
        )
        actual = mp.fsum(
            n * mp.exp(mp.j * (phases[j] - phases[0]))
            for j, n in enumerate(geometry.multiplicity[0])
            if n
        )
        naive = mp.fsum(
            mp.exp(mp.j * (phases[j] - phases[0]))
            for j, n in enumerate(geometry.multiplicity[0])
            if n
        )
        assert abs(actual - (r + 2 * mp.exp(mp.j * epsilon))) < mp.mpf("1e-85")
        assert abs(naive - (r + mp.exp(mp.j * epsilon))) < mp.mpf("1e-85")
        if epsilon:
            assert mp.arg(actual) > mp.arg(naive) > 0
        else:
            assert abs(mp.arg(actual) - mp.arg(naive)) < mp.mpf("1e-85")
