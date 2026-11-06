"""Independent exact child-exchange information and nonlinear rate controls.

Only supplied states and static derivatives are evaluated. No trajectory,
event, guard search or runtime symmetry probe supplies these identities.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_child_orbit as owner
from tnfr.physics._organized_source_child_quotient import (
    _observe_organized_source_child_quotient,
    _restrict_organized_source_child_quotient,
)

EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * a + slot, 9 * b + slot)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
ORBITS = (
    tuple((j,) for j in range(9))
    + tuple((9 + j, 18 + j) for j in range(9))
    + tuple((j,) for j in range(27, 45))
)
NODE_ORBIT = tuple(
    next(a for a, block in enumerate(ORBITS) if i in block) for i in range(45)
)
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))
NEIGHBORS = tuple(
    tuple(b if a == i else a for a, b in EDGES if i in (a, b)) for i in range(45)
)
ZERO = (Q(0),) * 36
UNIT = (Q(1),) * 36
H = Q(1, 128)
FORM = tuple(Q((5 * i) % 17 - 8, 4096) for i in range(36))
OFFSETS = tuple(Q((7 * i) % 11 - 5, 4096) for i in range(36))
CAPACITIES = tuple(Q(4 + i % 5, 4) for i in range(36))
HIDDEN = tuple(Q((11 * i) % 19 - 9, 8192) for i in range(18))


def _lift(values):
    return tuple(values[a] for a in NODE_ORBIT)


def _project(values):
    return tuple(sum((values[i] for i in block), Q(0)) / len(block) for block in ORBITS)


def _exchange(values):
    return tuple(values[j] for j in SWAP)


def _fine_state(form=FORM, offsets=OFFSETS, hidden=HIDDEN, capacities=CAPACITIES):
    x, delta = list(_lift(form)), list(_lift(offsets))
    for slot in range(9):
        x[9 + slot] += hidden[slot]
        x[18 + slot] -= hidden[slot]
        delta[9 + slot] += hidden[9 + slot]
        delta[18 + slot] -= hidden[9 + slot]
    return tuple(x), tuple(delta), _lift(capacities)


def _gram(vector):
    return tuple(tuple(a * b for b in vector) for a in vector)


def _gradient(values):
    result = [Q(0)] * 45
    for a, b in EDGES:
        result[a] += values[a] - values[b]
        result[b] += values[b] - values[a]
    return tuple(result)


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child orbit controls cannot execute events or trajectories")

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


def _mp(context, value):
    if isinstance(value, (Q, int)):
        return context.mpf(value.numerator) / int(value.denominator)
    return value


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _fields(context, form, offsets, capacities):
    """Independent two-law edge rows, energy/loss and form acceleration."""
    gamma, alpha = 1 / (1023 * context.pi), 2 * context.pi / 9
    phases = tuple(
        alpha * (i % 9 - 4) + _mp(context, value) for i, value in enumerate(offsets)
    )
    q = _gradient(form)
    sine = [context.mpf(0)] * 45
    energy = context.mpf(0)
    for a, b in EDGES:
        gap = phases[b] - phases[a]
        current = context.sin(gap)
        sine[a] += current
        sine[b] -= current
        energy += _mp(context, form[b] - form[a]) ** 2 / 2 + 1 - context.cos(gap)
    dx = tuple(
        _mp(context, nu) * (-_mp(context, value) + gamma * current) / 4
        for nu, value, current in zip(capacities, q, sine)
    )
    dtheta = tuple(
        gamma * _mp(context, nu) * _mp(context, value) / 4
        for nu, value in zip(capacities, q)
    )
    qdot = _gradient(dx)
    sine_dot = [context.mpf(0)] * 45
    for a, b in EDGES:
        value = context.cos(phases[b] - phases[a]) * (dtheta[b] - dtheta[a])
        sine_dot[a] += value
        sine_dot[b] -= value
    ddx = tuple(
        _mp(context, nu) * (-value + gamma * current) / 4
        for nu, value, current in zip(capacities, qdot, sine_dot)
    )
    loss = sum((nu * value**2 / 4 for nu, value in zip(capacities, q)), Q(0))
    return dx, dtheta, energy, loss, ddx


def _ideal_word(context, form, offsets, capacities):
    defaults = canonical_glyph_factor_defaults()
    beta = _mp(context, Q(defaults["EN_mix"]))
    coefficient = _mp(context, Q(DEFAULT_PHASE_LOCKING_COEFFICIENT))
    kappa = Q(defaults["SHA_vf_factor"])
    x = tuple(_mp(context, value) for value in form)
    delta = tuple(_mp(context, value) for value in offsets)
    after_x, after_delta, after_capacity = list(x), list(delta), list(capacities)
    alpha = 2 * context.pi / 9
    for target in (0, 1):
        peers = NEIGHBORS[target]
        after_x[target] = (1 - beta) * x[target] + beta * context.fsum(
            x[j] for j in peers
        ) / len(peers)
        resultant = context.fsum(
            context.exp(
                context.j * (alpha * ((j % 9) - target) + delta[j] - delta[target])
            )
            for j in peers
        )
        after_delta[target] += coefficient * context.arg(resultant)
        after_capacity[target] *= kappa
    return tuple(after_x), tuple(after_delta), tuple(after_capacity)


def _encode(state=None):
    x, phase, capacity = _fine_state() if state is None else state
    return owner._encode_organized_source_child_orbit(
        form=x, phase_offsets=phase, capacities=capacity
    )


def _arguments(state):
    return {
        "form_coordinates": state.form_coordinates,
        "phase_offset_coordinates": state.phase_offset_coordinates,
        "capacity_coordinates": state.capacity_coordinates,
        "hidden_gram": state.hidden_gram,
    }


def _observe(state):
    return owner._observe_organized_source_child_orbit(**_arguments(state))


@pytest.fixture(scope="module")
def state():
    return _encode()


@pytest.fixture(scope="module")
def observed(state):
    return _observe(state)


def test_roundtrip_selects_one_global_sign_and_retains_all_joint_products(state):
    original = _fine_state()
    assert state.geometry.blocks == ORBITS
    assert state.form_coordinates == FORM
    assert state.phase_offset_coordinates == OFFSETS
    assert state.capacity_coordinates == CAPACITIES
    assert state.hidden_gram == _gram(HIDDEN)
    assert state.hidden_coordinates == tuple(-x for x in HIDDEN)
    assert next(x for x in state.hidden_coordinates if x) > 0
    assert (state.form, state.phase_offsets, state.capacities) == tuple(
        _exchange(row) for row in original
    )
    assert owner._decode_organized_source_child_orbit(**_arguments(state)) == state
    assert _encode(tuple(_exchange(row) for row in original)) == state
    assert _encode((state.form, state.phase_offsets, state.capacities)) == state
    assert sum(len(row) for row in state.hidden_gram) == 18**2


@pytest.mark.parametrize("pivot", (0, 8, 9, 17))
def test_zero_prefix_and_tiny_rational_pivots_decode_without_rounding(pivot):
    hidden = [Q(0)] * 18
    hidden[pivot] = -Q(1, 2**600)
    state = _encode(_fine_state(hidden=tuple(hidden)))
    assert state.hidden_coordinates[pivot] == Q(1, 2**600)
    assert state.hidden_gram[pivot][pivot] == Q(1, 2**1200)
    assert owner._decode_organized_source_child_orbit(**_arguments(state)) == state


def test_raw_lifts_and_represented_scalar_values_are_not_wrapped_or_replaced():
    form = list(FORM)
    form[0] = Q(0.1)
    offsets = tuple(value + Q(25, 7) for value in OFFSETS)
    hidden = [Q(0)] * 18
    hidden[9] = Q(2 * math.pi)
    x, phase, capacities = _fine_state(tuple(form), offsets, tuple(hidden))
    x = list(x)
    x[0] = 0.1
    state = _encode((x, phase, capacities))
    assert state.form_coordinates[0] == Q(0.1) != Q(1, 10)
    assert state.phase_offset_coordinates == offsets
    assert state.hidden_coordinates[9] == Q(2 * math.pi)
    assert state.phase_offsets[9] - state.phase_offsets[18] == 2 * Q(2 * math.pi)


def test_inputs_are_detached_and_no_reader_mutates_primitives(state):
    source = [list(row) for row in _fine_state()]
    before = deepcopy(source)
    encoded = _encode(source)
    assert source == before
    arguments = {
        key: [list(row) for row in value] if key == "hidden_gram" else list(value)
        for key, value in _arguments(state).items()
    }
    saved = deepcopy(arguments)
    decoded = owner._decode_organized_source_child_orbit(**arguments)
    owner._observe_organized_source_child_orbit(**arguments)
    assert arguments == saved
    arguments["hidden_gram"][0][0] = 999
    source[0][0] = 999
    assert encoded == decoded == state


def test_rank_zero_agrees_with_the_existing_exact_sector_observer(mp):
    state = _encode(_fine_state(hidden=(Q(0),) * 18))
    report = _observe(state)
    old = _observe_organized_source_child_quotient(
        form_coordinates=FORM,
        phase_offset_coordinates=OFFSETS,
        capacity_coordinates=CAPACITIES,
    )
    assert state.hidden_coordinates == (Q(0),) * 18
    assert state.hidden_gram == _gram((Q(0),) * 18)
    assert state.form == old.state.form
    assert state.phase_offsets == old.state.phase_offsets
    assert state.capacities == old.state.capacities
    assert report.form_gradient == old.form_gradient
    assert report.dissipation_tau == old.dissipation_tau
    assert report.joint_storage_bounds == old.joint_storage_bounds
    for name in ("form_tau_rate_bounds", "phase_tau_rate_bounds"):
        for left, right in zip(getattr(report, name), getattr(old, name)):
            assert max(left.lo, right.lo) <= min(left.hi, right.hi)
    for row in report.hidden_gram_tau_rate_bounds:
        for value in row:
            _contains(mp, value, 0)
    _restrict_organized_source_child_quotient(
        form=state.form, phase_offsets=state.phase_offsets, capacities=state.capacities
    )


@pytest.mark.parametrize("capacities", (UNIT, CAPACITIES))
def test_complete_projected_rows_storage_and_loss_match_independent_fine_law(
    capacities, mp
):
    state = _encode(_fine_state(capacities=capacities))
    report = _observe(state)
    dx, dtheta, energy, loss, _ = _fields(
        mp, state.form, state.phase_offsets, state.capacities
    )
    assert report.form_gradient == _project(_gradient(state.form))
    _contains(mp, report.joint_storage_bounds, energy)
    assert report.dissipation_tau == loss
    _contains(mp, report.storage_tau_rate_bounds, -_mp(mp, loss))
    gamma = 1 / (1023 * mp.pi)
    for name, values, factor in (
        ("form_tau_rate_bounds", dx, mp.mpf(1)),
        ("phase_tau_rate_bounds", dtheta, mp.mpf(1)),
        ("form_slow_rate_bounds", dx, 2 / gamma**2),
        ("phase_slow_rate_bounds", dtheta, 2 / gamma**2),
        ("form_original_rate_bounds", dx, mp.mpf(1023) / 1024),
        ("phase_original_rate_bounds", dtheta, mp.mpf(1023) / 1024),
    ):
        for interval, value in zip(getattr(report, name), _project(values)):
            _contains(mp, interval, factor * value)
    weights = tuple(1 / nu for nu in state.capacities)
    total = sum(weights, Q(0))
    form_total = sum((w * x for w, x in zip(weights, state.form)), Q(0))
    phase_total = sum((w * x for w, x in zip(weights, state.phase_offsets)), Q(0))
    assert report.inverse_capacity_weight_total == total
    assert report.inverse_capacity_weighted_form_total == form_total
    assert report.inverse_capacity_weighted_phase_offset_total == phase_total
    assert report.inverse_capacity_weighted_form_mean == form_total / total
    assert report.inverse_capacity_weighted_phase_offset_mean == phase_total / total


def test_every_gram_rate_is_the_independent_joint_chain_rule(state, observed, mp):
    dx, dtheta, _, _, _ = _fields(mp, state.form, state.phase_offsets, state.capacities)
    hidden_rate = tuple(
        (row[9 + j] - row[18 + j]) / 2 for row in (dx, dtheta) for j in range(9)
    )
    hidden = tuple(_mp(mp, value) for value in state.hidden_coordinates)
    for bounds, value in zip(observed.hidden_tau_rate_bounds, hidden_rate):
        _contains(mp, bounds, value)
    for i in range(18):
        for j in range(18):
            expected = hidden_rate[i] * hidden[j] + hidden[i] * hidden_rate[j]
            _contains(mp, observed.hidden_gram_tau_rate_bounds[i][j], expected)
            # Every 2x2 cone minor has vanishing derivative along vv^T.
            pivot = 0
            gij = hidden_rate[i] * hidden[j] + hidden[i] * hidden_rate[j]
            gpp = 2 * hidden[pivot] * hidden_rate[pivot]
            gip = hidden_rate[i] * hidden[pivot] + hidden[i] * hidden_rate[pivot]
            gpj = hidden_rate[pivot] * hidden[j] + hidden[pivot] * hidden_rate[j]
            residual = (
                gij * hidden[pivot] ** 2
                + hidden[i] * hidden[j] * gpp
                - gip * hidden[pivot] * hidden[j]
                - hidden[i] * hidden[pivot] * gpj
            )
            assert abs(residual) < mp.mpf("1e-95")


def test_nonuniform_capacity_hidden_rows_use_the_actual_cycle_and_phase_matrix(
    state, observed, mp
):
    gamma, alpha = 1 / (1023 * mp.pi), 2 * mp.pi / 9
    a = tuple(_mp(mp, value) for value in state.hidden_coordinates[:9])
    b = tuple(_mp(mp, value) for value in state.hidden_coordinates[9:])
    phases = tuple(
        alpha * (j % 9 - 4) + _mp(mp, value)
        for j, value in enumerate(state.phase_offset_coordinates)
    )

    def sinc(value):
        return mp.sin(value) / value if value else mp.mpf(1)

    h_matrix = [[Q(0)] * 9 for _ in range(9)]
    k_matrix = [[mp.mpf(0)] * 9 for _ in range(9)]
    for j in range(9):
        h_matrix[j][j] = Q(4)
        for k in ((j - 1) % 9, (j + 1) % 9):
            h_matrix[j][k] = -Q(1)
            weight = mp.cos(phases[9 + k] - phases[9 + j]) * sinc(b[k] - b[j])
            k_matrix[j][j] += weight
            k_matrix[j][k] -= weight
        k_matrix[j][j] += (
            mp.cos(phases[j] - phases[9 + j]) + mp.cos(phases[18 + j] - phases[9 + j])
        ) * sinc(b[j])
    ha = tuple(
        mp.fsum(_mp(mp, value) * a[k] for k, value in enumerate(row))
        for row in h_matrix
    )
    kb = tuple(mp.fsum(value * b[k] for k, value in enumerate(row)) for row in k_matrix)
    capacities = tuple(_mp(mp, value) for value in state.capacity_coordinates[9:18])
    assert len(set(capacities)) > 1
    closed = tuple(
        -nu * (value + gamma * phase) / 4
        for nu, value, phase in zip(capacities, ha, kb)
    ) + tuple(gamma * nu * value / 4 for nu, value in zip(capacities, ha))
    dx, dtheta, _, _, _ = _fields(mp, state.form, state.phase_offsets, state.capacities)
    direct = tuple(
        (row[9 + j] - row[18 + j]) / 2 for row in (dx, dtheta) for j in range(9)
    )
    assert max(abs(left - right) for left, right in zip(closed, direct)) < mp.mpf(
        "1e-95"
    )
    for interval, value in zip(observed.hidden_tau_rate_bounds, closed):
        _contains(mp, interval, value)


def _local_control(negative_slot):
    projected = (H, -H) + ZERO[2:]
    b = tuple(-H if j == negative_slot else H for j in range(9))
    return _fine_state(projected, ZERO, (Q(0),) * 9 + b, UNIT)


def _local_grams(gram):
    return tuple((gram[j][j], gram[j][9 + j], gram[9 + j][9 + j]) for j in range(9))


def test_frozen_local_grams_lose_cross_slot_signs_and_projected_form_rate(mp):
    first = _encode(_local_control(2))
    second = _encode(_local_control(4))
    left, right = _observe(first), _observe(second)
    assert _local_grams(first.hidden_gram) == _local_grams(second.hidden_gram)
    assert first.hidden_gram != second.hidden_gram
    assert first.form_coordinates == second.form_coordinates
    assert first.phase_offset_coordinates == second.phase_offset_coordinates == ZERO
    for field in ("form", "phase_offsets"):
        first_means = tuple(
            sum(getattr(first, field)[9 * c : 9 * c + 9], Q(0)) / 9 for c in range(5)
        )
        second_means = tuple(
            sum(getattr(second, field)[9 * c : 9 * c + 9], Q(0)) / 9 for c in range(5)
        )
        assert first_means == second_means
    f1 = _fields(mp, first.form, first.phase_offsets, first.capacities)
    f2 = _fields(mp, second.form, second.phase_offsets, second.capacities)
    assert abs(f1[2] - f2[2]) < mp.mpf("1e-95")
    assert left.dissipation_tau == right.dissipation_tau == 20 * H**2
    alpha, gamma, h = 2 * mp.pi / 9, 1 / (1023 * mp.pi), _mp(mp, H)
    baseline = 45 * (1 - mp.cos(alpha))
    excess = 6 * h**2 + 4 * mp.cos(alpha) * (1 - mp.cos(2 * h)) + 36 * (1 - mp.cos(h))
    assert 0 < excess < mp.mpf(1) / 100
    for report, fields in ((left, f1), (right, f2)):
        assert abs(fields[2] - baseline - excess) < mp.mpf("1e-95")
        _contains(mp, report.joint_storage_bounds, baseline + excess)
    for row in (0, 1):
        assert abs(f1[0][row] - f2[0][row]) < mp.mpf("1e-95")
        assert abs(f1[1][row] - f2[1][row]) < mp.mpf("1e-95")
        assert tuple(first.phase_offsets[j] for j in NEIGHBORS[row]) == tuple(
            second.phase_offsets[j] for j in NEIGHBORS[row]
        )
    expected = gamma * mp.sin(alpha) * (-2 * mp.sin(h) ** 2) / 4
    actual = _project(f1[0])[10] - _project(f2[0])[10]
    assert expected < 0
    assert abs(actual - expected) < mp.mpf("1e-95")
    bounds = left.form_tau_rate_bounds[10] - right.form_tau_rate_bounds[10]
    _contains(mp, bounds, expected)
    assert bounds.hi < 0


def _mixed_control(sign):
    a = (H,) + (Q(0),) * 8
    b = (Q(0), sign * H) + (Q(0),) * 7
    return _fine_state(ZERO, ZERO, a + b, UNIT)


def test_frozen_separate_channel_grams_lose_mixed_rate_and_projected_curvature(mp):
    first, second = _encode(_mixed_control(1)), _encode(_mixed_control(-1))
    left, right = _observe(first), _observe(second)
    assert _local_grams(first.hidden_gram) == _local_grams(second.hidden_gram)
    for block in (range(9), range(9, 18)):
        assert tuple(
            tuple(first.hidden_gram[i][j] for j in block) for i in block
        ) == tuple(tuple(second.hidden_gram[i][j] for j in block) for i in block)
    assert first.hidden_gram[0][10] == H**2
    assert second.hidden_gram[0][10] == -(H**2)
    gamma, h = 1 / (1023 * mp.pi), _mp(mp, H)
    _contains(mp, left.hidden_gram_tau_rate_bounds[10][10], -gamma * h**2 / 2)
    _contains(mp, right.hidden_gram_tau_rate_bounds[10][10], gamma * h**2 / 2)
    assert left.hidden_gram_tau_rate_bounds[10][10].hi < 0
    assert right.hidden_gram_tau_rate_bounds[10][10].lo > 0
    f1 = _fields(mp, first.form, first.phase_offsets, first.capacities)
    f2 = _fields(mp, second.form, second.phase_offsets, second.capacities)
    for channel in (0, 1):
        assert max(
            abs(a - b) for a, b in zip(_project(f1[channel]), _project(f2[channel]))
        ) < mp.mpf("1e-95")
    assert left.dissipation_tau == right.dissipation_tau == 9 * H**2
    alpha = 2 * mp.pi / 9
    baseline = 45 * (1 - mp.cos(alpha))
    excess = 4 * h**2 + 4 * (1 + mp.cos(alpha)) * (1 - mp.cos(h))
    assert 0 < excess < mp.mpf(1) / 100
    for report, fields in ((left, f1), (right, f2)):
        assert abs(fields[2] - baseline - excess) < mp.mpf("1e-95")
        _contains(mp, report.joint_storage_bounds, baseline + excess)
    expected_curvature_difference = (
        5 * gamma**2 * h * mp.sin(2 * mp.pi / 9) * mp.sin(h) / 8
    )
    actual_curvature_difference = _project(f1[4])[9] - _project(f2[4])[9]
    assert expected_curvature_difference > 0
    assert abs(actual_curvature_difference - expected_curvature_difference) < mp.mpf(
        "1e-95"
    )


def test_marked_ideal_word_descends_and_retains_joint_hidden_state(mp):
    original = _fine_state(capacities=UNIT)
    state = _encode(original)
    canonical = (state.form, state.phase_offsets, state.capacities)
    fine_word = _ideal_word(mp, *original)
    decoded_word = _ideal_word(mp, *canonical)
    defaults = canonical_glyph_factor_defaults()
    beta = _mp(mp, Q(defaults["EN_mix"]))
    coefficient = _mp(mp, Q(DEFAULT_PHASE_LOCKING_COEFFICIENT))
    projected_forms = tuple(_mp(mp, value) for value in state.form_coordinates)
    projected_phases = tuple(
        2 * mp.pi * (j % 9 - 4) / 9 + _mp(mp, value)
        for j, value in enumerate(state.phase_offset_coordinates)
    )
    for target in (0, 1):
        left, right = (target - 1) % 9, (target + 1) % 9
        merged, donor = 9 + target, 27 + target
        counted_mean = (
            projected_forms[left]
            + projected_forms[right]
            + 2 * projected_forms[merged]
            + projected_forms[donor]
        ) / 5
        expected_form = (1 - beta) * projected_forms[target] + beta * counted_mean
        hidden_magnitude = mp.sqrt(_mp(mp, state.hidden_gram[9 + target][9 + target]))
        resultant = mp.fsum(
            mp.exp(mp.j * (projected_phases[j] - projected_phases[target]))
            for j in (left, right, donor)
        ) + 2 * mp.exp(
            mp.j * (projected_phases[merged] - projected_phases[target])
        ) * mp.cos(
            hidden_magnitude
        )
        assert resultant.real > 4
        expected_offset = _mp(
            mp, state.phase_offset_coordinates[target]
        ) + coefficient * mp.arg(resultant)
        assert abs(decoded_word[0][target] - expected_form) < mp.mpf("1e-95")
        assert abs(decoded_word[1][target] - expected_offset) < mp.mpf("1e-95")
    for index in (0, 1):
        assert max(
            abs(a - b)
            for a, b in zip(_project(fine_word[index]), _project(decoded_word[index]))
        ) < mp.mpf("1e-95")
    assert fine_word[2] == decoded_word[2]
    after_hidden = tuple(
        (row[9 + j] - row[18 + j]) / 2 for row in decoded_word[:2] for j in range(9)
    )
    for i in range(18):
        assert abs(after_hidden[i] - _mp(mp, state.hidden_coordinates[i])) < mp.mpf(
            "1e-95"
        )
        for j in range(18):
            assert abs(
                after_hidden[i] * after_hidden[j] - _mp(mp, state.hidden_gram[i][j])
            ) < mp.mpf("1e-95")


@pytest.mark.parametrize(
    "kind",
    (
        "asymmetric",
        "negative",
        "rank_two",
        "zero_diagonal_cross",
        "irrational_lift",
        "tiny_rank_two",
        "boolean",
        "wrong_shape",
    ),
)
def test_decoder_rejects_noncone_data_or_an_inexact_rational_lift(kind):
    gram = [[Q(0)] * 18 for _ in range(18)]
    if kind == "asymmetric":
        gram[0][0], gram[0][1] = Q(1), Q(1)
    elif kind == "negative":
        gram[0][0] = -Q(1)
    elif kind == "rank_two":
        gram[0][0] = gram[1][1] = Q(1)
    elif kind == "zero_diagonal_cross":
        gram[0][1] = gram[1][0] = Q(1)
    elif kind == "irrational_lift":
        gram[0][0] = Q(2)
    elif kind == "tiny_rank_two":
        gram[0][0], gram[1][1] = Q(1), Q(1, 2**1200)
    elif kind == "boolean":
        gram[0][0] = True
    else:
        gram.pop()
    with pytest.raises(
        (TypeError, ValueError), match="rational" if kind == "irrational_lift" else None
    ):
        owner._decode_organized_source_child_orbit(
            form_coordinates=FORM,
            phase_offset_coordinates=OFFSETS,
            capacity_coordinates=CAPACITIES,
            hidden_gram=gram,
        )


@pytest.mark.parametrize(
    "field,bad",
    (
        ("form", True),
        ("phase_offsets", float("nan")),
        ("capacities", Q(0)),
        ("capacities", -Q(1)),
        ("capacities", True),
    ),
)
def test_encoder_rejects_invalid_primitives(field, bad):
    arguments = dict(zip(("form", "phase_offsets", "capacities"), _fine_state()))
    changed = list(arguments[field])
    changed[0] = bad
    arguments[field] = changed
    with pytest.raises((TypeError, ValueError)):
        owner._encode_organized_source_child_orbit(**arguments)


def test_asymmetric_held_capacity_and_wrong_fine_length_are_rejected():
    x, phase, capacities = _fine_state()
    changed = list(capacities)
    changed[9] += Q(1, 2**600)
    with pytest.raises(ValueError):
        _encode((x, phase, changed))
    with pytest.raises(ValueError):
        _encode((x[:-1], phase, capacities))
