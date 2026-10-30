"""Independent controls for an actual registered source with centered resource.

Static points check laws and inequalities, not acquired endpoints. The first
section is a mathematical preparation criterion, with no runtime detector,
trajectory, new input or successful mutual transfer asserted here.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_registered_resource as owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
AMPLITUDE = DELTA / 8
SECTION = 3 * DELTA / 16
REGISTRATION = DELTA / 4
WEIGHT_ERROR = Q(2, 10**60)
WAIT = Q(10**10)
MATURE = Q(1, 2**28)
PREFIX = 3 * MATURE / 2
SCALE = Q(6, 5)
GAMMA_LO, GAMMA_HI = Q(1, 3300), Q(1, 3000)
COMPARISON = Q(3, 2) * (14 * ETA + 2 * WAIT * WEIGHT_ERROR)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


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
    bins = tuple(int((9 * value) % 9) for value in _turns())
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


def _weights():
    return tuple(Q(3, 4) + Q(i % 5, 16) for i in range(63))


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, weights, strict=True)
    )


def _nominal_weights(mp):
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if i < 9 else 1) * _mp(mp, DELTA) / 2
        for i, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((phase[b] - phase[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _laplacian(edges):
    rows = [[0] * 18 for _ in range(18)]
    for a, b, weight in edges:
        rows[a][a] += weight
        rows[b][b] += weight
        rows[a][b] -= weight
        rows[b][a] -= weight
    return tuple(tuple(row) for row in rows)


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(18)) for j in range(18))
        for i in range(18)
    )


def _offset(gap):
    return (-gap / 2,) * 9 + (gap / 2,) * 9


def _state(forms, errors, gauge=Q(0)):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + error + gauge
        for turn, error in zip(_turns(), errors)
    )


def _oracle(mp, forms, errors, weights):
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns(), errors)
    )
    gradient, sine = [mp.mpf(0)] * 18, [mp.mpf(0)] * 18
    for a, b, weight in _edges(weights):
        flow = _mp(mp, weight) * (x[a] - x[b])
        current = mp.sin(phase[b] - phase[a])
        gradient[a] += flow
        gradient[b] -= flow
        sine[a] += current
        sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-q + gamma * s) / 9 for q, s in zip(gradient, sine))
    dy = tuple(gamma * q / 9 for q in gradient)
    return x, tuple(gradient), tuple(sine), dx, dy


def _assess(**changes):
    values = dict(form_error_bound=ERROR, phase_error_bound=ERROR)
    values.update(changes)
    return owner._assess_mixed_registered_resource(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("registered resource controls cannot advance or replay a source")

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
def field():
    return source_owner._coupling_successor_field(
        completion="common_sense",
        new_edge_conductances=_weights(),
        loss_normalization="bare_degree",
    )


def test_complete_unforced_rows_keep_clock_signed_form_and_actual_gauge(mp, field):
    forms = _center(tuple(Q((7 * i) % 17 - 8, 8192) for i in range(18)))
    errors = tuple(a + Q(i % 5 - 2, 32768) for i, a in enumerate(_offset(SECTION)))
    _, q, sine, dx, dy = _oracle(mp, forms, errors, _weights())
    for actual, expected in zip(field(_state(forms, errors)), dx + dy):
        _contains(mp, actual, expected)
    assert min(forms) < 0 < max(forms) and max(abs(v) for v in dy) > 0
    assert abs(sum(dx)) < mp.mpf("1e-190")
    assert abs(sum(dy)) < mp.mpf("1e-190")
    clock = _mp(mp, Q(1023, 1024))
    original = tuple(
        -clock * qi / 9 + si / (1024 * mp.pi * 9) for qi, si in zip(q, sine)
    )
    original += tuple(qi / (1024 * mp.pi * 9) for qi in q)
    assert max(abs(clock * a - b) for a, b in zip(dx + dy, original)) < mp.mpf("1e-190")
    for actual, expected in zip(field(_state(forms, errors, Q(7, 13))), dx + dy):
        _contains(mp, actual, expected)


def test_nominal_full_rows_give_unforced_negative_velocity_wedge(mp):
    weights = _nominal_weights(mp)
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    assert _mp(mp, Q(4, 9)) < k < mp.mpf(1) / 2 and 2 < c < 3
    assert 2 * k / 3 - gamma**2 > 0
    for value in (SECTION / 2, SECTION, DELTA):
        q = _mp(mp, value)
        for speed in (mp.mpf(0), -(gamma**2) * q):
            v = speed / (4 * gamma * k)
            forms = (-v,) * 9 + (v,) * 9
            _, _, _, dx, dy = _oracle(mp, forms, _offset(q), weights)
            expected_v = -2 * k * v - gamma * c * mp.sin(q) / 9
            assert max(abs(value + expected_v) for value in dx[:9]) < mp.mpf("1e-190")
            assert max(abs(value - expected_v) for value in dx[9:]) < mp.mpf("1e-190")
            assert abs(dy[9] - dy[0] - speed) < mp.mpf("1e-190")
            acceleration = 4 * gamma * k * expected_v
            if speed == 0:
                assert acceleration < 0
            else:
                lower = gamma**2 * (2 * k / 3 - gamma**2) * q
                assert acceleration + gamma**2 * speed > lower > 0
            assert mp.sqrt(_dot(forms, forms)) < 3 * gamma * _mp(mp, DELTA)
    # The wedge gives q(t)>=Delta*exp(-gamma^2*t)>0, not finite collapse.
    assert 3 * GAMMA_HI * DELTA < Q(1, 1024)


def test_gauge_projection_and_exact_gap_covectors_reject_an_average_only_claim():
    errors = tuple((-1) ** i * 3 * ETA for i in range(18))
    centered = _center(errors)
    forms = _center(tuple(Q(i % 3 - 1) * ETA / 3 for i in range(18)))
    assert _dot(forms, forms) <= 2 * ETA**2
    assert _dot(centered, centered) <= _dot(errors, errors) <= 162 * ETA**2
    assert (
        _dot(forms, forms) + _dot(centered, centered) <= 164 * ETA**2 < (14 * ETA) ** 2
    )
    jrow = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9
    assert _dot(jrow, jrow) == Q(2, 9) < Q(1, 4)
    pair_rows = []
    for j in range(9):
        partner = 9 + (5 * (j - 4) + 4) % 9
        row = tuple(Q(int(i == partner) - int(i == j)) for i in range(18))
        residual = tuple(a - b for a, b in zip(row, jrow))
        assert _dot(residual, residual) == Q(16, 9)
        pair_rows.append(row)
    assert tuple(sum(row[i] for row in pair_rows) / 9 for i in range(18)) == jrow
    # This phase point shares J=b but has an unregistered individual pair.
    # It is a law-level counterexample, not an acquired first-section state.
    phases = list(_offset(SECTION))
    phases[9] += REGISTRATION
    phases[10] -= REGISTRATION
    assert sum(phases) == 0 and _dot(jrow, phases) == SECTION
    assert max(_dot(row, phases) for row in pair_rows) > REGISTRATION
    assert SECTION + Q(4, 3) * COMPARISON < REGISTRATION


def test_actual_weight_defect_breaks_carrier_and_forces_both_error_rows(mp):
    nominal = _nominal_weights(mp)
    increments = tuple((-1) ** i * WEIGHT_ERROR for i in range(63))
    actual = tuple(w + _mp(mp, d) for w, d in zip(nominal, increments))
    xr = (Q(1, 65536),) * 9 + (-Q(1, 65536),) * 9
    yr = _offset(SECTION)
    ex = _center(tuple(Q(i % 7 - 3, 10**9) for i in range(18)))
    ey = _center(tuple(Q(i % 5 - 2, 10**9) for i in range(18)))
    xa, ya = tuple(a + b for a, b in zip(xr, ex)), tuple(a + b for a, b in zip(yr, ey))
    ref, result = _oracle(mp, xr, yr, nominal), _oracle(mp, xa, ya, actual)
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges(actual)))
    delta = _laplacian(tuple((a, b, w) for (a, b), w in zip(_support()[1], increments)))
    dhxr = _matvec(delta, xr)
    hex_ = _matvec(h, tuple(_mp(mp, value) for value in ex))
    gamma = 1 / (1023 * mp.pi)
    assert max(abs(value) for value in dhxr) > 0
    assert max(sum(abs(v) for v in row) for row in delta) <= 14 * WEIGHT_ERROR
    for i in range(18):
        force = _mp(mp, dhxr[i])
        expected_x = (-hex_[i] + gamma * (result[2][i] - ref[2][i]) - force) / 9
        expected_y = gamma * (hex_[i] + force) / 9
        assert abs(result[-2][i] - ref[-2][i] - expected_x) < mp.mpf("1e-190")
        assert abs(result[-1][i] - ref[-1][i] - expected_y) < mp.mpf("1e-190")
    # A single tiny legal coefficient perturbation already defeats exact
    # regional invariance. Its point is not asserted to be an actual image.
    edge = next(i for i, (a, b) in enumerate(_support()[1]) if a < 9 <= b)
    one = tuple(
        w + (_mp(mp, WEIGHT_ERROR) if i == edge else 0) for i, w in enumerate(nominal)
    )
    altered = _oracle(mp, xr, yr, one)
    a, _ = _support()[1][edge]
    other = next(i for i in range(9) if i != a)
    defect = _mp(mp, WEIGHT_ERROR * 2 * xr[0])
    assert abs(altered[-2][a] - altered[-2][other] + defect / 9) < mp.mpf("1e-190")
    assert abs(altered[-1][a] - altered[-1][other] - gamma * defect / 9) < mp.mpf(
        "1e-190"
    )
    assert abs(altered[-2][a] - altered[-2][other]) > mp.mpf("1e-70")
    assert 14 * (1 + GAMMA_HI) * Q(1, 1024) / 9 < 2


def test_incremental_metric_keeps_noncommuting_stiffness_and_mixed_terms(mp):
    reference_phase = _offset(SECTION)
    y = _center(tuple(Q((i % 9) ** 2, 65536) for i in range(18)))
    target = tuple(
        2 * mp.pi * _mp(mp, t) + _mp(mp, e) for t, e in zip(_turns(), reference_phase)
    )
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges(_weights())))
    secants = []
    for a, b, _ in _edges(_weights()):
        base, displacement = target[b] - target[a], _mp(mp, y[b] - y[a])
        coefficient = (
            (mp.sin(base + displacement) - mp.sin(base)) / displacement
            if displacement
            else mp.cos(base)
        )
        secants.append((a, b, coefficient))
    k = _laplacian(secants)
    hk, kh = _product(h, k), _product(k, h)
    assert max(abs(a - b) for r, s in zip(hk, kh) for a, b in zip(r, s)) > mp.mpf(
        "1e-6"
    )
    gamma, scale = 1 / (1023 * mp.pi), _mp(mp, SCALE)
    yy = tuple(_mp(mp, value) for value in y)
    ky = _matvec(k, yy)
    reference = _oracle(mp, (Q(0),) * 18, reference_phase, _weights())
    for sign in (-1, 1):
        x = tuple(sign * value / 2 for value in yy)
        hx = _matvec(h, x)
        dx = tuple((-a - gamma * b) / 9 for a, b in zip(hx, ky))
        dy = tuple(gamma * a / 9 for a in hx)
        actual = _oracle(
            mp, x, tuple(a + b for a, b in zip(reference_phase, y)), _weights()
        )
        assert max(
            abs(a - b - c) for a, b, c in zip(actual[-2], reference[-2], dx)
        ) < mp.mpf("1e-190")
        assert max(
            abs(a - b - c) for a, b, c in zip(actual[-1], reference[-1], dy)
        ) < mp.mpf("1e-190")
        assert sign * _dot(x, yy) > 0
        metric = (
            mp.mpf(9)
            / 4
            * (
                _dot(x, x)
                + scale**2 * _dot(yy, yy)
                + 2 * scale**2 * gamma * _dot(x, yy)
            )
        )
        derivative = (
            mp.mpf(9)
            / 2
            * (
                _dot(x, dx)
                + scale**2 * _dot(yy, dy)
                + scale**2 * gamma * (_dot(dx, yy) + _dot(x, dy))
            )
        )
        exact = (
            -(1 - scale**2 * gamma**2) * _dot(x, hx) / 2
            - gamma * _dot(x, ky) / 2
            - scale**2 * gamma**2 * _dot(yy, ky) / 2
        )
        assert abs(derivative - exact) < mp.mpf("1e-190")
        assert derivative < 0 < metric
    lower, upper = Q(9, 4) * (1 - Q(1, 2500)), Q(9, 4) * (1 + Q(1, 2500))
    assert SCALE**2 * upper / lower < Q(9, 4)


def test_section_resource_lower_bound_is_actual_and_not_baseline_storage(mp):
    # Cauchy's bound is exact for all states with the given mean section,
    # independently of the comparison trajectory used to certify registration.
    for fine in (Q(0), Q(1, 10**9)):
        deviation = tuple(fine * (i % 3 - 1) for i in range(18))
        y = tuple(a + b for a, b in zip(_offset(SECTION), deviation))
        assert sum(y[:9]) / 9 == -SECTION / 2
        assert sum(y[9:]) / 9 == SECTION / 2
        assert _dot(y, y) == Q(9, 2) * SECTION**2 + _dot(deviation, deviation)
    lower = Q(9, 10) * SECTION**2
    assert lower == Q(81, 40) * AMPLITUDE**2
    assert (
        lower
        > Q(9, 5) * AMPLITUDE**2
        > (3 * (AMPLITUDE - Q(1, 10**8)) - PREFIX) ** 2 / 5
    )
    phase = tuple(2 * mp.pi * _mp(mp, t) for t in _turns())
    baseline = sum(1 - mp.cos(phase[b] - phase[a]) for a, b, _ in _edges(_weights()))
    y = _offset(SECTION)
    excess = sum(
        mp.cos(phase[b] - phase[a]) - mp.cos(phase[b] - phase[a] + _mp(mp, y[b] - y[a]))
        for a, b, _ in _edges(_weights())
    )
    assert baseline > 1
    assert _mp(mp, lower) < excess < 1
    # This nominal phase point is only an energy oracle, not the section.
    assert Q(9, 5) * AMPLITUDE**2 != lower


def test_signed_section_form_and_rate_use_actual_rows_not_error_derivatives(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    # Variation of constants drops only earlier nonnegative integrands.
    # This is a closed unit-interval kernel, not a computed source history.
    unit_kernel = (1 - mp.exp(-2 * k)) / (2 * k)
    assert unit_kernel > mp.mpf(1) / 3
    for q in (SECTION / 2, SECTION, DELTA):
        assert mp.sin(_mp(mp, q)) > _mp(mp, 3 * q / 4)
    assert (
        gamma * c * _mp(mp, 3 * SECTION / 8) * unit_kernel / 9
        > gamma * _mp(mp, SECTION) / 36
    )
    form_upper = -GAMMA_LO * SECTION / 18 + COMPARISON
    rate_error = 2 * GAMMA_HI * (COMPARISON + WEIGHT_ERROR)
    rate_upper = -4 * GAMMA_LO**2 * SECTION / 81 + rate_error
    assert form_upper < -GAMMA_LO * SECTION / 36 < 0
    assert rate_upper < 0
    assert report.regional_form_contrast_upper_bound == form_upper
    assert report.form_orientation_margin == GAMMA_LO * SECTION / 36 - COMPARISON
    assert report.section_mean_gap_rate_upper_bound == rate_upper
    assert report.rate_orientation_margin == -rate_upper
    assert report.section_orientation_certified
    # The rate error uses theta'=gamma*Hx/9 and a covector of norm<1/2.
    assert GAMMA_HI / 18 * (18 * COMPARISON + 14 * WEIGHT_ERROR / 1024) <= rate_error
    # A constructed signed mode is transmitted with the opposite reaction.
    # Its finite effect on a recipient is deliberately not inferred.
    donor = (-form_upper / 2,) * 9 + (form_upper / 2,) * 9
    receiver = (Q(0),) * 18
    receiver_input = tuple(a - b for a, b in zip(donor, receiver))
    donor_input = tuple(-value for value in receiver_input)
    assert sum(receiver_input) == sum(donor_input) == 0
    assert sum(receiver_input[9:]) / 9 - sum(receiver_input[:9]) / 9 == form_upper
    assert all(a + b == 0 for a, b in zip(receiver_input, donor_input))


def test_finite_section_and_later_identity_keep_resource_at_the_section_only(
    report, record_property
):
    lower, upper = Q(9, 4) * (1 - Q(1, 2500)), Q(9, 4) * (1 + Q(1, 2500))
    rate = Q(5) / (216 * 2750**2 * (1 + Q(1, 2500)))
    assert report.metric_lower_bound == lower and report.metric_upper_bound == upper
    assert report.metric_decay_rate == rate
    assert report.metric_growth_squared_margin == Q(9, 4) - SCALE**2 * upper / lower > 0
    assert report.initial_comparison_norm_squared_upper_bound == 164 * ETA**2
    assert report.initial_comparison_norm_upper_bound == 14 * ETA
    assert report.comparison_forcing_upper_bound == 2 * WEIGHT_ERROR
    assert report.comparison_error_upper_bound == COMPARISON < Q(1, 10**40)
    assert report.initial_mean_gap_lower_bound == DELTA - 6 * ETA > SECTION
    assert report.deadline_mean_gap_upper_bound == MATURE / 2 < SECTION
    assert (
        report.first_unit_mean_gap_lower_bound
        == DELTA * (1 - GAMMA_HI**2) - COMPARISON
        > SECTION
    )
    assert report.section_time_bounds == (Q(1), WAIT)
    assert report.section_exists_certified and not report.section_age_computed
    assert report.section_marked_gap_bounds == (
        SECTION - Q(4, 3) * COMPARISON,
        SECTION + Q(4, 3) * COMPARISON,
    )
    assert 0 < report.section_marked_gap_bounds[0]
    assert report.predeadline_registered_gap_upper_bound == SECTION + 3 * COMPARISON
    assert report.postdeadline_registered_gap_upper_bound == 3 * MATURE / 2
    assert (
        report.registered_gap_upper_bound
        == max(SECTION + 3 * COMPARISON, 3 * MATURE / 2)
        < REGISTRATION
    )
    assert report.registration_retained_after_section_certified
    assert report.section_phase_norm_squared_lower_bound == Q(9, 2) * SECTION**2
    assert report.section_excess_storage_lower_bound == Q(9, 10) * SECTION**2
    assert report.comparison_phase_floor_upper_bound == Q(9, 5) * AMPLITUDE**2
    assert report.resource_floor_margin == Q(9, 40) * AMPLITUDE**2 > 0
    assert (
        report.excess_storage_upper_bound
        == Q(27, 2) * DELTA**2 + 486 * ETA + 18 * ETA**2
    )
    assert report.section_excess_storage_lower_bound < report.excess_storage_upper_bound
    assert report.source_association_certified and report.deadline_certified
    assert report.nominal_reference_certified and report.comparison_certified
    assert report.section_resource_certified and report.certified
    assert report.resource_lower_bound_at_section_only
    assert not report.resource_lower_bound_for_arbitrary_later_age
    for name, value in (
        ("section_mean_gap", SECTION),
        ("comparison_error_upper_bound", COMPARISON),
        ("registered_gap_upper_bound", report.registered_gap_upper_bound),
        ("section_resource_lower_bound", report.section_excess_storage_lower_bound),
        ("necessary_floor_margin", report.resource_floor_margin),
        ("regional_form_upper_bound", report.regional_form_contrast_upper_bound),
        ("mean_gap_rate_upper_bound", report.section_mean_gap_rate_upper_bound),
    ):
        record_property(name, str(value))
    record_property("first_section_structural_time_interval", "(1,10000000000)")
    record_property("actual_section_time_evaluated", "false")
    record_property("mutual_transfer_certified", "not part of this contract")


def test_uniform_one_source_reference_ignores_second_copy_and_old_response(
    monkeypatch, report
):
    ready = report.readiness_reference
    irrelevant_second = replace(
        ready.sources[1], source_certified=False, certified=False
    )
    irrelevant_first = replace(
        ready.sources[0],
        certified=False,
        normalization_common_form_separation_certified=False,
    )
    replaced = replace(
        ready,
        sources=(irrelevant_first, irrelevant_second),
        source_certified=(True, False),
        initial_inclusion_certified=(True, False),
        all_time_source_identity_certified=(True, False),
        readiness_certified=(True, False),
        returned_norm_squared_upper_bounds=(
            ready.returned_norm_squared_upper_bounds[0],
            Q(100),
        ),
        circulation=None,
        handoff_norm_upper_bounds=None,
        handoff_association_certified=False,
        composed_response_certified=False,
        certified=False,
    )
    observed = []

    def rebuilt(**kwargs):
        observed.append(kwargs)
        return replaced

    monkeypatch.setattr(owner, "_assess_mixed_circulation_readiness", rebuilt)
    result = _assess()
    assert result.certified and result.source == irrelevant_first
    assert result.physical_source_count == 1
    assert observed[0]["form_error_bounds"] == (ERROR, ERROR)
    assert observed[0]["phase_error_bounds"] == (ERROR, ERROR)
    assert (
        result.section_excess_storage_lower_bound
        == report.section_excess_storage_lower_bound
    )
    assert result.section_time_bounds == report.section_time_bounds
    assert not result.charge_or_leak_prefix_applied
    assert (
        not result.new_external_input_supplied and not result.new_phase_jump_performed
    )


def test_relevant_actual_source_and_deadline_are_readmitted_despite_stale_success(
    monkeypatch, report
):
    ready, source = report.readiness_reference, report.source
    for name, value in (
        ("source_certified", False),
        ("relative_phase_gap", 2 * DELTA),
        ("phase_reset_error_bound", Q(0)),
        ("gauge_phase_deviation_upper_bound", 7 * ETA),
        ("edge_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
        ("gamma_range", (Q(1, 6600), GAMMA_HI)),
    ):
        changed = replace(source, **{name: value})
        altered = replace(ready, sources=(changed, ready.sources[1]))
        assert altered.certified and changed.certified
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    for name, value in (
        ("supplied_phase_origins", (-DELTA, DELTA)),
        ("initial_form_means", (Q(1), Q(0))),
    ):
        base = replace(source.precontact_reference, **{name: value})
        changed = replace(source, precontact_reference=base)
        altered = replace(ready, sources=(changed, ready.sources[1]))
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        assert not _assess().certified
    common = source.completion_reports[0]
    changed = replace(
        source,
        completion_reports=(
            replace(common, initial_energy_upper_bound=0),
            source.completion_reports[1],
        ),
    )
    altered = replace(ready, sources=(changed, ready.sources[1]))
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
    )
    assert not _assess().source_association_certified
    for name, value in (
        ("waiting_duration", WAIT / 2),
        ("derived_mature_norm_upper_bound", 2 * MATURE),
        ("decay_upper_bound", Q(1, 2**38)),
        (
            "returned_norm_squared_upper_bounds",
            (MATURE**2, ready.returned_norm_squared_upper_bounds[1]),
        ),
    ):
        altered = replace(ready, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert result.source_association_certified
        assert not result.deadline_certified and not result.section_exists_certified
        assert not result.certified


def test_geometry_and_nonvacuous_comparison_cannot_be_inferred_from_flags(
    monkeypatch, report
):
    registered = owner._derive_registered_phase_probe_geometry()
    for changed in (
        replace(registered, cross_symmetry_certified=False),
        replace(
            registered,
            successor=replace(
                registered.successor,
                new_edge_pairs=registered.successor.new_edge_pairs[:-1],
            ),
        ),
    ):
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: changed
        )
        assert not _assess().source_association_certified
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: registered
    )
    monkeypatch.setattr(owner, "cos", lambda value: I(-1, 1))
    result = _assess()
    assert result.source_association_certified and result.deadline_certified
    assert not result.nominal_reference_certified and not result.comparison_certified
    assert not result.section_exists_certified and not result.section_resource_certified


def test_malformed_primitives_precede_work_and_excess_source_caps_are_unavailable(
    monkeypatch,
):
    for name in ("form_error_bound", "phase_error_bound"):
        result = _assess(**{name: ERROR + Q(1, 10**100)})
        assert result.status == "unavailable" and not result.certified
        assert not result.source_association_certified
        assert (
            not result.section_exists_certified
            and not result.section_resource_certified
        )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid original allowance reached numerical or source work")

    for name in (
        "_assess_mixed_circulation_readiness",
        "_derive_registered_phase_probe_geometry",
        "_sine_common_quadratic_coefficients",
        "pi_interval",
        "cos",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for name in ("form_error_bound", "phase_error_bound"):
        for value in (
            True,
            np.bool_(False),
            None,
            "0",
            float("nan"),
            float("inf"),
            complex(1, 2),
            -Q(1, 10**100),
            (Q(0),),
            {"value": 0},
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: value})


def test_fixed_width_scalars_and_no_supplied_endpoint_level_age_or_resource(report):
    value = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    assert _assess(form_error_bound=value, phase_error_bound=value) == report
    assert report.status == "conditional_registered_resource_section"
    assert report.physical_capacities == (Q(1),) * 18
    assert report.full_state_dimension == 36 and report.physical_source_count == 1
    assert report.original_supplied_phase_origins == (-DELTA / 2, DELTA / 2)
    assert report.mean_gap_covector_norm_upper_bound == Q(1, 2)
    assert report.gap_minus_mean_covector_norm == Q(4, 3)
    assert not report.runtime_source_authenticated
    assert not report.runtime_stopping_rule_executed
    assert not report.section_age_computed and not report.source_trajectory_evaluated
    with pytest.raises(FrozenInstanceError):
        report.section_mean_gap = 0
    for name in (
        "source",
        "readiness",
        "initial_state",
        "section_mean_gap",
        "section_level",
        "elapsed_time",
        "section_age",
        "mature_norm_upper_bound",
        "partner",
        "input",
        "phase_target",
        "resource",
        "readout_error_bound",
        "observed_transfer",
        "gamma",
        "new_edge_conductances",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
