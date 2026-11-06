"""Actual-law resource controls for two finite reacting mature UM objects.

The fields are evaluated only at declared constructed points. Neither those
points nor a generic admitted coefficient box substitutes for an acquired
endpoint. All-time claims use exact balances and a first-exit argument.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_mature_partner_obstruction as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
MATURE = Q(1, 2**28)
PREFIX = 3 * MATURE / 2
DELTA = Q(1, 1024)
AMPLITUDE = Q(1, 8192)
TARGET_ERROR = Q(1, 10**8)
WEIGHT_ERROR = Q(2, 10**60)
RADIUS = Q(1, 32)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    pairs = {
        (i, j)
        for i in range(18)
        for j in range(i + 1, 18)
        if min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
        or (i // 9 != j // 9 and bins[i] == bins[j])
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights(side):
    return tuple(Q(3, 4) + Q((i + 2 * side) % 5, 16) for i in range(63))


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, weights, strict=True)
    )


def _laplacian(edges):
    matrix = [[Q(0)] * 18 for _ in range(18)]
    for a, b, w in edges:
        matrix[a][a] += w
        matrix[b][b] += w
        matrix[a][b] -= w
        matrix[b][a] -= w
    return tuple(tuple(row) for row in matrix)


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _product(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(18)) for j in range(18))
        for i in range(18)
    )


def _state(forms, phases):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + value
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )


def _oracle(mp, forms, phases, weights=None, connected=True):
    weights = (_weights(0), _weights(1)) if weights is None else weights
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )
    q, sine = [mp.mpf(0)] * 36, [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            pressure = _mp(mp, weight) * (x[a] - x[b])
            current = mp.sin(phase[b] - phase[a])
            q[a] += pressure
            q[b] -= pressure
            sine[a] += current
            sine[b] -= current
    inputs = (
        tuple(x[18 + j] - x[j] for j in range(18)) if connected else (mp.mpf(0),) * 18
    )
    inputs += tuple(-value for value in inputs)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-hi + gamma * si) / 9 + ui for hi, si, ui in zip(q, sine, inputs))
    dy = tuple(gamma * hi / 9 for hi in q)
    return x, tuple(q), tuple(sine), inputs, dx, dy


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(ERROR, ERROR), phase_error_bounds=(ERROR, ERROR)
    )
    arguments.update(changes)
    return owner._assess_mixed_mature_partner_obstruction(**arguments)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("finite partner controls cannot advance or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    mutual = field_owner._mixed_two_channel_field(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
        law="full",
    )
    cut = tuple(
        source_owner._coupling_successor_field(
            completion="common_sense",
            new_edge_conductances=_weights(side),
            loss_normalization="bare_degree",
        )
        for side in (0, 1)
    )
    return mutual, cut


def test_complete_72_rows_keep_both_reactions_phase_and_structural_clock(mp, fields):
    forms = tuple(Q((7 * i) % 17 - 8, 8192) for i in range(36))
    phases = tuple(Q((i % 7) - 3, 16384) for i in range(36))
    _, q, sine, inputs, dx, dy = _oracle(mp, forms, phases)
    rates = fields[0](_state(forms, phases))
    for interval, expected in zip(rates, dx + dy):
        _contains(mp, interval, expected)
    for j in range(18):
        assert inputs[j] == -inputs[18 + j]
    assert max(abs(v) for v in inputs[:9]) > 0
    assert max(abs(v) for v in inputs[9:18]) > 0
    assert max(abs(v) for v in dy) > 0
    clock = _mp(mp, Q(1023, 1024))
    original_x = tuple(
        -clock * hi / 9 + si / (1024 * mp.pi * 9) + clock * ui
        for hi, si, ui in zip(q, sine, inputs)
    )
    original_y = tuple(hi / (1024 * mp.pi * 9) for hi in q)
    assert max(
        abs(a / clock - b) for a, b in zip(original_x + original_y, dx + dy)
    ) < mp.mpf("1e-190")
    state = _state(forms, phases)
    disconnected = []
    for side, field in enumerate(fields[1]):
        local = (
            state[18 * side : 18 * (side + 1)] + state[36 + 18 * side : 54 + 18 * side]
        )
        disconnected.append(field(local))
    cut_rates = (
        disconnected[0][:18]
        + disconnected[1][:18]
        + disconnected[0][18:]
        + disconnected[1][18:]
    )
    cut = _oracle(mp, forms, phases, connected=False)
    for interval, expected in zip(cut_rates, cut[-2] + cut[-1]):
        _contains(mp, interval, expected)
    assert max(abs(a - b) for a, b in zip(dx, cut[-2])) > 0
    assert dy == cut[-1]


def test_common_form_resources_decouple_without_resetting_means_or_gauges(mp, fields):
    centered = tuple(
        _center(tuple(Q((i % 7) - 3, 16384) for i in range(18 * side, 18 * (side + 1))))
        for side in (0, 1)
    )
    phases = tuple(Q((i % 5) - 2, 32768) for i in range(36))
    base = centered[0] + centered[1]
    means = (Q(3, 8), Q(-5, 16))
    forms = tuple(value + means[i // 18] for i, value in enumerate(base))
    _, _, _, _, dx, dy = _oracle(mp, forms, phases)
    _, _, _, _, base_dx, base_dy = _oracle(mp, base, phases)
    for side in (0, 1):
        rate = _mp(mp, means[1 - side] - means[side])
        assert abs(sum(dx[18 * side : 18 * (side + 1)]) / 18 - rate) < mp.mpf("1e-190")
        assert abs(sum(dy[18 * side : 18 * (side + 1)])) < mp.mpf("1e-190")
        for i in range(18 * side, 18 * (side + 1)):
            assert abs(dx[i] - rate - base_dx[i]) < mp.mpf("1e-190")
            assert abs(dy[i] - base_dy[i]) < mp.mpf("1e-190")
    # Uniform form offsets may be arbitrarily large for this law identity;
    # they are not declared members of the tightly bounded actual source.
    common_derivative = 18 * (
        means[0] * (means[1] - means[0]) + means[1] * (means[0] - means[1])
    )
    assert common_derivative == -18 * (means[0] - means[1]) ** 2 < 0
    assert (means[1] - means[0]) - (means[0] - means[1]) == -2 * (means[0] - means[1])
    gauge = tuple(Q(5, 16) if i < 18 else Q(-7, 32) for i in range(36))
    shifted = tuple(a + b for a, b in zip(phases, gauge))
    for interval, expected in zip(fields[0](_state(forms, shifted)), dx + dy):
        _contains(mp, interval, expected)
    uniform = (Q(2),) * 18 + (Q(-3),) * 18
    oracle = _oracle(mp, uniform, (Q(0),) * 36)
    assert oracle[1] == (mp.mpf(0),) * 36
    assert oracle[-1] == (mp.mpf(0),) * 36
    assert max(abs(v) for v in oracle[-2]) > 0


def test_actual_storage_derivative_uses_weighted_forms_and_bare_potential(mp):
    forms = tuple(
        Q((5 * i) % 19 - 9, 16384) + Q(1 if i < 18 else -2, 32) for i in range(36)
    )
    phases = tuple(Q((i % 7) - 3, 32768) for i in range(36))
    x, q, sine, inputs, dx, dy = _oracle(mp, forms, phases)
    derivative = _dot(q, dx) - _dot(sine, dy)
    loss = -_dot(q, q) / 9
    work = _dot(q, inputs)
    differences = tuple(x[j] - x[18 + j] for j in range(18))
    gradient_differences = tuple(q[j] - q[18 + j] for j in range(18))
    assert abs(work + _dot(differences, gradient_differences)) < mp.mpf("1e-190")
    assert abs(derivative - loss - work) < mp.mpf("1e-190")
    cut = _oracle(mp, forms, phases, connected=False)
    assert abs(_dot(cut[1], cut[-2]) - _dot(cut[2], cut[-1]) - loss) < mp.mpf("1e-190")
    centered = _center(forms[:18]) + _center(forms[18:])
    centered_oracle = _oracle(mp, centered, phases)
    centered_power = _dot(centered_oracle[1], centered_oracle[3])
    assert abs(centered_power - work) < mp.mpf("1e-190")
    assert loss < 0
    # Changing either denominator to the old quarter law changes this loss.
    assert abs(loss + _dot(q, q) / 4) > mp.mpf("1e-6")


def test_contact_work_can_be_positive_at_arbitrarily_small_weight_mismatch(mp):
    old, new = _support()
    edge = next(
        i
        for i, (a, b) in enumerate(new)
        if a < 9 <= b and (9 * (_turns()[b] - _turns()[a])) % 9 in (1, 8)
    )
    a, b = new[edge]
    vector = tuple(Q(int(i == a) - int(i == b)) for i in range(18))
    scale = PREFIX / 8
    for delta in (Q(1, 8), WEIGHT_ERROR):
        second = (Q(1),) * 63
        first = tuple(Q(1) - delta if i == edge else Q(1) for i in range(63))
        h1, h2 = _laplacian(_edges(first)), _laplacian(_edges(second))
        x1 = tuple(scale * v for v in vector)
        x2 = tuple(scale * (1 - delta / 10) * v for v in vector)
        q1, q2 = _matvec(h1, x1), _matvec(h2, x2)
        work = -_dot(
            tuple(a - b for a, b in zip(q1, q2)), tuple(a - b for a, b in zip(x1, x2))
        )
        base = _dot(vector, _matvec(h2, vector))
        assert base == 20
        assert (
            work
            == scale**2 * delta**2 * (Q(2, 5) - base / 100)
            == scale**2 * delta**2 / 5
            > 0
        )
        loss = -(_dot(q1, q1) + _dot(q2, q2)) / 9
        assert loss + work < 0
        assert _product(h1, h2) != _product(h2, h1)
        # The tiny positive sign is exact rational evidence. A dyadic128
        # displayed interval need not resolve a power of order delta_w².
        if delta == Q(1, 8):
            forms = x1 + x2
            phases = (Q(0),) * 36
            oracle = _oracle(mp, forms, phases, weights=(first, second))
            assert abs(_dot(oracle[1], oracle[3]) - _mp(mp, work)) < mp.mpf("1e-190")
            field = field_owner._mixed_two_channel_field(
                first_new_edge_conductances=first,
                second_new_edge_conductances=second,
                law="full",
            )
            for interval, expected in zip(
                field(_state(forms, phases)), oracle[-2] + oracle[-1]
            ):
                _contains(mp, interval, expected)
    # Both examples are constructed coefficient/state points. No claim is
    # made that either is the endpoint of a historical acquired source.
    assert len(old) == 18


def test_symmetric_split_identity_bounds_contact_work_without_commutation():
    new = _support()[1]
    base = (Q(7, 8),) * 63
    first = tuple(w + ((-1) ** i) * WEIGHT_ERROR for i, w in enumerate(base))
    second = tuple(w - ((-1) ** i) * WEIGHT_ERROR for i, w in enumerate(base))
    h1, h2 = _laplacian(_edges(first)), _laplacian(_edges(second))
    average = tuple(
        tuple((a + b) / 2 for a, b in zip(r1, r2)) for r1, r2 in zip(h1, h2)
    )
    defect = tuple(tuple((a - b) / 2 for a, b in zip(r1, r2)) for r1, r2 in zip(h1, h2))
    assert max(sum(abs(v) for v in row) for row in defect) <= 14 * WEIGHT_ERROR
    assert all(sum(i in edge for edge in new) == 7 for i in range(18))
    x1 = _center(tuple(Q((i % 7) - 3, 8192) for i in range(18)))
    x2 = _center(tuple(Q((i % 5) - 2, 16384) for i in range(18)))
    plus = tuple(a + b for a, b in zip(x1, x2))
    minus = tuple(a - b for a, b in zip(x1, x2))
    q1, q2 = _matvec(h1, x1), _matvec(h2, x2)
    work = -_dot(tuple(a - b for a, b in zip(q1, q2)), minus)
    # Unnormalized plus/minus avoid an unnecessary numerical square root.
    assert work == -_dot(minus, _matvec(average, minus)) - _dot(
        minus, _matvec(defect, plus)
    )
    norm2 = _dot(x1, x1) + _dot(x2, x2)
    assert (_dot(plus, plus) + _dot(minus, minus)) / 2 == norm2
    assert work <= 14 * WEIGHT_ERROR * norm2
    assert (_dot(q1, q1) + _dot(q2, q2)) / 9 >= norm2
    assert (
        -(_dot(q1, q1) + _dot(q2, q2)) / 9 + work
        <= -(1 - 14 * WEIGHT_ERROR) * norm2
        < 0
    )
    same_work = -_dot(minus, _matvec(average, minus))
    assert same_work <= 0


def test_excess_energy_bounds_do_not_spend_the_original_winding_storage(mp):
    forms = _center(tuple(Q((i % 7) - 3, 16384) for i in range(18)))
    forms += _center(tuple(Q((i % 5) - 2, 16384) for i in range(18)))
    phases = _center(tuple(Q((i % 5) - 2, 32768) for i in range(18)))
    phases += _center(tuple(Q((i % 7) - 3, 32768) for i in range(18)))
    x, q, _, _, _, _ = _oracle(mp, forms, phases)
    target = tuple(2 * mp.pi * _mp(mp, t) for t in _turns() * 2)
    y = tuple(_mp(mp, v) for v in phases)
    potential, baseline = mp.mpf(0), mp.mpf(0)
    for side in (0, 1):
        for a, b, _ in _edges(_weights(side)):
            a, b = a + 18 * side, b + 18 * side
            angle = target[b] - target[a]
            baseline += 1 - mp.cos(angle)
            potential += mp.cos(angle) - mp.cos(angle + y[b] - y[a])
            # The actual bare Hessian, not weighted-sine storage, controls
            # every segment in the declared acute neighborhood.
            assert mp.cos(angle + y[b] - y[a]) > mp.mpf(1) / 10
    norm2 = _dot(x, x) + _dot(y, y)
    excess = _dot(x, q) / 2 + potential
    assert _mp(mp, Q(1, 5)) * norm2 <= excess <= 9 * norm2
    assert baseline > 1 and excess < mp.mpf("1e-4")
    assert potential > 0


def test_resource_and_mean_bounds_close_the_chart_for_every_duration(
    report, record_property
):
    assert report.source_association_certified
    assert report.prefix_norm_upper_bound == PREFIX
    assert report.common_form_norm_upper_bound == Q(1, 32) * PREFIX
    assert report.new_edge_incident_counts == (7,) * 18
    assert report.transport_difference_upper_bound == 14 * WEIGHT_ERROR
    assert report.dissipation_margin == 1 - 14 * WEIGHT_ERROR > 0
    assert report.storage_balance_certified
    assert report.weighted_gap_lower_bound**2 / 9 == 1
    assert report.phase_cosine_lower_bound * report.bare_gap_lower_bound / 2 == Q(1, 5)
    assert (
        report.transport_norm_upper_bound / 2
        == report.bare_hessian_norm_upper_bound / 2
        == 9
    )
    assert report.excess_storage_coercivity == Q(1, 5)
    assert report.excess_storage_upper_coefficient == 9
    assert report.initial_excess_storage_upper_bound == 9 * PREFIX**2
    assert report.full_norm_squared_upper_bound == 45 * PREFIX**2
    # E controls only centered forms. Add the independently contractive
    # common means; do not claim their entire value belongs to E.
    for common_square in (Q(0), PREFIX**2 / 4, PREFIX**2):
        centered_initial = PREFIX**2 - common_square
        assert 45 * centered_initial + common_square <= 45 * PREFIX**2
    assert report.full_norm_upper_bound == 7 * PREFIX
    assert report.norm_squared_margin == (49 - 45) * PREFIX**2 > 0
    assert report.acute_chart_margin == RADIUS - 7 * PREFIX > 0
    assert report.marked_gap_upper_bound == 10 * PREFIX
    assert report.marked_gap_squared_margin == (100 - 90) * PREFIX**2 > 0
    assert report.registration_margin == DELTA / 4 - 10 * PREFIX > 0
    assert report.all_time_identity_certified
    assert report.all_contact_durations_covered
    assert not report.contact_work_nonpositive_claimed
    assert not report.zero_interaction_claimed and not report.common_form_reset
    # These strict margins close a first exit, not a collection of finite
    # horizon samples. The disconnected energy loses only the contact term.
    assert (
        report.connected_target_unreachable and report.disconnected_target_unreachable
    )
    record_property(
        "available_excess_storage_upper_bound",
        str(report.initial_excess_storage_upper_bound),
    )
    record_property("all_time_full_norm_bound", str(report.full_norm_upper_bound))
    record_property("all_time_marked_gap_bound", str(report.marked_gap_upper_bound))
    record_property("contact_dissipation_margin", str(report.dissipation_margin))


def test_independent_target_phase_and_gap_floors_exclude_the_full_domain(
    report, record_property
):
    smallest = AMPLITUDE - TARGET_ERROR
    assert report.target_domain_certified
    assert (
        report.target_radius_upper_bound
        == 3 * (AMPLITUDE + TARGET_ERROR) + PREFIX
        < RADIUS
    )
    target_gap = smallest - Q(3, 2) * PREFIX
    assert report.target_marked_gap_lower_bound == target_gap > 0
    assert report.target_gap_separation_margin == smallest - Q(23, 2) * PREFIX > 0
    assert report.marked_gap_upper_bound < target_gap
    phase_lower = 3 * smallest - PREFIX
    assert report.target_phase_norm_lower_bound == phase_lower > 0
    assert report.target_excess_storage_lower_bound == phase_lower**2 / 5
    assert (
        report.resource_shortfall_lower_bound == phase_lower**2 / 5 - 9 * PREFIX**2 > 0
    )
    # Independently attain the triangle lower bound within the algebraic
    # target domain. This is not a dynamically reached state.
    offset = (-smallest / 2,) * 9 + (smallest / 2,) * 9
    offset += (smallest / 2,) * 9 + (-smallest / 2,) * 9
    assert _dot(offset, offset) == (3 * smallest) ** 2
    base = tuple(-PREFIX * v / (3 * smallest) for v in offset)
    assert _dot(base, base) == PREFIX**2
    target = tuple(a + b for a, b in zip(offset, base))
    assert _dot(target, target) == phase_lower**2
    assert sum(target[:18]) == sum(target[18:]) == 0
    partners = tuple(9 + (5 * (j - 4) + 4) % 9 for j in range(9))
    for j, partner in enumerate(partners):
        assert target[partner] - target[j] >= target_gap
        assert target[18 + partner] - target[18 + j] <= -target_gap
    assert (
        report.geometric_obstruction_certified and report.resource_obstruction_certified
    )
    assert report.certified and not report.phase_jump_performed
    # The target reader is a domain description, not an observed response.
    assert report.target_reference.phase_jump_amplitude == AMPLITUDE
    assert report.target_reference.phase_jump_error_bound == TARGET_ERROR
    record_property("target_marked_gap_lower_bound", str(target_gap))
    record_property(
        "geometric_separation_margin", str(report.target_gap_separation_margin)
    )
    record_property(
        "necessary_target_excess_storage", str(report.target_excess_storage_lower_bound)
    )
    record_property(
        "excess_resource_shortfall", str(report.resource_shortfall_lower_bound)
    )


def test_fresh_readiness_and_target_premises_not_old_success_or_jump_work(
    monkeypatch, report
):
    readiness = report.readiness
    target = readiness.circulation
    ignored_target = replace(
        target,
        certified=False,
        work_certified=False,
        phase_jump_work_upper_bound=Q(100),
        jump_work_margin=Q(-1),
        independent_hypotheses_separated=False,
        record_separation_margin=Q(-1),
    )
    ignored = replace(
        readiness,
        certified=False,
        composed_response_certified=False,
        circulation=ignored_target,
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: ignored
    )
    independent = _assess()
    assert independent.certified
    assert (
        independent.initial_excess_storage_upper_bound
        == report.initial_excess_storage_upper_bound
    )
    assert (
        independent.resource_shortfall_lower_bound
        == report.resource_shortfall_lower_bound
    )
    for name, value in (
        ("source_certified", (True, False)),
        ("all_time_source_identity_certified", (False, True)),
        ("readiness_certified", (True, False)),
        ("handoff_norm_upper_bounds", (MATURE, 2 * MATURE)),
        ("derived_mature_norm_upper_bound", 2 * MATURE),
        ("waiting_duration", Q(10**10 - 1)),
        (
            "returned_norm_squared_upper_bounds",
            (MATURE**2, readiness.returned_norm_squared_upper_bounds[1]),
        ),
    ):
        changed = replace(readiness, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    for side, name, value in (
        (0, "edge_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
        (1, "form_error_bound", 2 * ERROR),
        (1, "phase_radius", 2 * RADIUS),
        (0, "relative_phase_gap", 2 * DELTA),
        (1, "source_certified", False),
    ):
        source = replace(readiness.sources[side], **{name: value})
        changed = replace(
            readiness,
            sources=tuple(
                source if i == side else readiness.sources[i] for i in range(2)
            ),
        )
        assert changed.certified
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    source = readiness.sources[1]
    wrong_origins = replace(
        source.precontact_reference,
        supplied_phase_origins=(-DELTA, DELTA),
    )
    altered_source = replace(source, precontact_reference=wrong_origins)
    changed = replace(readiness, sources=(readiness.sources[0], altered_source))
    assert changed.certified and altered_source.certified
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: changed
    )
    result = _assess()
    assert not result.source_association_certified and not result.certified
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: readiness
    )
    geometry = readiness.source_geometry
    for name, value in (
        ("weighted_gap_lower_bound", Q(2)),
        ("bare_gap_lower_bound", Q(3)),
        ("new_edge_pairs", geometry.new_edge_pairs[:-1]),
    ):
        changed = replace(geometry, **{name: value})
        monkeypatch.setattr(
            owner, "_derive_coupling_successor_geometry", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified


def test_prefix_gauges_and_target_associations_are_rechecked_independently(
    monkeypatch, report
):
    readiness, target = report.readiness, report.target_reference
    for name, value in (
        ("charge_amplitude", Q(1, 100)),
        ("prefix_norm_upper_bound", 2 * PREFIX),
        ("prefix_leak_duration", Q(1, 16)),
    ):
        altered = replace(readiness, circulation=replace(target, **{name: value}))
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    for name, value in (
        ("phase_jump_amplitude", 2 * AMPLITUDE),
        ("phase_jump_error_bound", 2 * TARGET_ERROR),
        ("contact_pairs", target.contact_pairs[:-1]),
    ):
        altered = replace(readiness, circulation=replace(target, **{name: value}))
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert result.source_association_certified and result.storage_balance_certified
        assert not result.target_domain_certified and not result.certified
        assert not result.connected_target_unreachable
    for side, name in (
        (0, "edge_weight_deviation_upper_bound"),
        (1, "gauge_phase_deviation_upper_bound"),
    ):
        source = target.sources[side]
        changed = replace(source, **{name: 2 * getattr(source, name)})
        downstream = replace(
            target,
            sources=tuple(
                changed if i == side else target.sources[i] for i in range(2)
            ),
        )
        altered = replace(readiness, circulation=downstream)
        assert altered.certified and downstream.certified
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified


def test_nonnegative_outside_source_caps_is_unavailable_not_an_obstruction():
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            result = _assess(**{name: values})
            assert not result.source_association_certified
            assert not result.certified and result.status == "unavailable"
            assert not result.connected_target_unreachable
            assert not result.disconnected_target_unreachable


def test_invalid_pairs_precede_readiness_and_geometry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid source allowance reached readiness or geometry")

    monkeypatch.setattr(owner, "_assess_mixed_circulation_readiness", forbidden)
    monkeypatch.setattr(owner, "_derive_coupling_successor_geometry", forbidden)
    for name in ("form_error_bounds", "phase_error_bounds"):
        for value in (
            True,
            np.bool_(False),
            "0",
            None,
            float("nan"),
            float("inf"),
            complex(0, 1),
            -Q(1, 10**100),
        ):
            for side in (0, 1):
                pair = tuple(value if i == side else Q(0) for i in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        for values in ((0,), (0, 0, 0), repeat(Q(0)), {0, 1}, {0: 0, 1: 0}):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: values})


def test_one_shot_fixed_width_and_no_horizon_or_resource_fabrication(report):
    def fixed(value):
        value = Q(value)
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
    )
    assert normalized == report
    assert report.status == "conditional_all_time_target_obstruction"
    assert (
        report.physical_capacities == (Q(1),) * 36 and report.full_state_dimension == 72
    )
    assert (
        report.exchange_strength == 1 and report.common_form_difference_decay_rate == 2
    )
    assert report.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert not report.arbitrary_common_offsets_admitted_as_sources
    assert (
        not report.source_trajectory_evaluated
        and not report.runtime_source_authenticated
    )
    with pytest.raises(FrozenInstanceError):
        report.zero_interaction_claimed = True
    for name in (
        "readiness",
        "source",
        "target_reference",
        "initial_state",
        "mature_norm_upper_bounds",
        "resource",
        "partner_mean",
        "contact_duration",
        "elapsed_time",
        "phase_jump_error_bound",
        "readout_error_bound",
        "observed_transfer",
        "phase_target",
        "exchange_strength",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
