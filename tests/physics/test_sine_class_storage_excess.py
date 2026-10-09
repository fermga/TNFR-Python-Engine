"""Independent storage algebra and static finite bounds; no response execution."""

from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import pytest

from tests.physics.test_sine_class_neighbor_nonadditivity import _dot, _geometry, _mv
from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_class_storage_excess as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _arguments():
    return dict(
        mediator_class=2,
        amplitude=Q(7, 10000),
        horizon=Q(1, 8),
        endpoint_radius=Q(1, 10**32),
    )


@pytest.fixture(scope="module")
def bound():
    return owner._bound_storage_excess(**_arguments())


@pytest.mark.parametrize("mediator_class", (1, 2))
def test_onset_channels_from_independent_mixed_quartic_edge_polynomial(mediator_class):
    onset = owner._derive_storage_excess_onset(mediator_class)
    _, edges, degrees, _, normalized = _geometry()
    u, v = tuple(row[4] for row in normalized), tuple(row[22] for row in normalized)
    r, s = _mv(normalized, u), _mv(normalized, v)
    expected = (
        (Q(9, 16), Q(0), Q(0)),
        (Q(5, 64), Q(15, 128), Q(5, 64)),
        (Q(0), Q(0), Q(9, 16)),
        (Q(341, 96), Q(145, 64), Q(341, 96)),
    )
    assert onset.mixed_channel_factors == expected
    assert onset.geometry.degrees == degrees
    assert (onset.time_degree, onset.amplitude_degree) == (5, 4)
    assert onset.common_rational_factor == Q(-1, 60)
    # Polarize the actual edge polynomial, independently of the implemented
    # mixed-monomial expansion and including opposite signed impulses.
    for a, b in ((Q(2, 5), Q(3, 7)), (Q(-4, 3), Q(2, 9)), (Q(1), Q(-1))):
        for channel in range(4):
            selected = edges[9 * channel : 9 * (channel + 1)]

            def polynomial(left, right):
                first = tuple(left * x + right * y for x, y in zip(u, v))
                second = tuple(left * x + right * y for x, y in zip(r, s))
                return sum(
                    (second[j] - second[i]) * (first[j] - first[i]) ** 3
                    for i, j in selected
                )

            mixed = polynomial(a, b) - polynomial(a, 0) - polynomial(0, b)
            c31, c22, c13 = onset.mixed_channel_factors[channel]
            assert mixed == c31 * a**3 * b + c22 * a**2 * b**2 + c13 * a * b**3


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _mul(left, right):
    corners = tuple(a * b for a in left for b in right)
    return min(corners), max(corners)


def _scale(value, pair):
    return _mul((value, value), pair)


def _power(pair, degree):
    endpoints = pair[0] ** degree, pair[1] ** degree
    return (0 if degree % 2 == 0 and pair[0] <= 0 <= pair[1] else min(endpoints)), max(
        endpoints
    )


def test_static_cone_rebuilds_all_spatial_powers_and_twenty_nine_edges(bound):
    _, edges, _, _, matrix = _geometry()
    h, g = bound.horizon, Q(1, 3000)
    d = 1 - 2 * g**2 * h**2
    phase_error = Q(4, 3) * h**2 * (1 + g**2 / d)
    independent = []
    for port, stored in (
        (4, bound.direct_term.donor_spatial_columns),
        (22, bound.direct_term.receiver_spatial_columns),
    ):
        vector = tuple(Q(int(i == port)) for i in range(27))
        columns = []
        for _ in range(5):
            vector = _mv(matrix, vector)
            columns.append(vector)
        assert stored == tuple(columns)
        error = Q(1, 3) * (2 * h) ** 3 * max(map(abs, columns[4]))
        phase, adjoint = [], []
        for i, j in edges:
            first, second, third, fourth = (
                column[j] - column[i] for column in columns[:4]
            )
            slope = -h * second / 2
            phase.append(
                (
                    first + min(0, slope) - phase_error,
                    first + max(0, slope) + phase_error,
                )
            )
            adjoint.append(
                _add(
                    _add((second - error, second + error), _scale(-third, (0, 2 * h))),
                    _scale(fourth / 2, (0, (2 * h) ** 2)),
                )
            )
        independent.append((phase, adjoint))
    total = (Q(0), Q(0))
    for index, ((u, rd), (v, rr)) in enumerate(
        zip(zip(*independent[0]), zip(*independent[1]))
    ):
        u2v = _mul(_power(u, 2), v)
        uv2 = _mul(u, _power(v, 2))
        common = _scale(3, _add(u2v, uv2))
        edge = _mul(
            (Q(1, 6), Q(1)) if index < 27 else (Q(1), Q(1)),
            _add(
                _mul(rd, _add(common, _power(v, 3))),
                _mul(rr, _add(_power(u, 3), common)),
            ),
        )
        stored = bound.direct_term.mixed_edge_bounds[index]
        assert stored.lo <= edge[0] <= edge[1] <= stored.hi
        total = _add(total, edge)
    stored = bound.direct_term.mixed_edge_sum_bounds
    assert stored.lo <= total[0] <= total[1] <= stored.hi
    assert Q(1, 2) < stored.lo < stored.hi < 17
    assert len(bound.direct_term.mixed_edge_bounds) == 29


def test_complete_storage_bound_charges_distinct_tail_and_source_orders(bound):
    g, h, m, eps = Q(1, 3000), bound.horizon, bound.amplitude, bound.endpoint_radius
    d, ell = 1 - 2 * g**2 * h**2, 1 - 2 * g * h
    expected_tail = 232 * (4 * h**2 / d**2 + Q(64, 3) * g**2 * h**3 / d)
    for amplitude, row in zip((0, m, m, 2 * m), bound.per_history_errors):
        ratio = (2 * g * amplitude) ** 2
        assert row.total_input_variation == amplitude
        assert (
            row.higher_amplitude_storage_error_upper_bound
            == expected_tail * ratio**3 / (1 - ratio)
        )
        source_defect = 4 * g * h * eps / (ell * d) * (g * amplitude / d + eps / ell)
        assert row.nonlinear_initialization_error_upper_bound == source_defect
        assert (
            row.nominal_form_nonlinearity_upper_bound
            == 2 * g**3 * amplitude**2 * h / d**3
        )
    # No-input nonlinear source defect is retained even though its nominal
    # amplitude tail vanishes; it cannot be dropped from a nonzero word.
    zero = bound.per_history_errors[0]
    assert zero.higher_amplitude_storage_error_upper_bound == 0
    assert zero.nonlinear_initialization_error_upper_bound > 0
    assert zero.storage_source_error_upper_bound > 0
    assert bound.higher_amplitude_error_upper_bound == sum(
        row.higher_amplitude_storage_error_upper_bound
        for row in bound.per_history_errors
    )
    assert bound.source_error_upper_bound == sum(
        row.storage_source_error_upper_bound for row in bound.per_history_errors
    )
    assert 0 < bound.quartic_correction_upper_bound < Q(26, 10**39)
    assert 0 < bound.higher_amplitude_error_upper_bound < Q(99, 10**37)
    assert 0 < bound.source_error_upper_bound < Q(95, 10**45)
    direct = bound.direct_term.direct_quartic_storage_bounds
    error = (
        bound.quartic_correction_upper_bound + bound.higher_amplitude_error_upper_bound
    )
    assert bound.nominal_excess_storage_bounds == (direct[0] - error, direct[1] + error)
    assert bound.full_excess_storage_bounds == (
        direct[0] - error - bound.source_error_upper_bound,
        direct[1] + error + bound.source_error_upper_bound,
    )
    lower, upper = bound.full_excess_storage_bounds
    assert -Q(2565, 10**35) < lower < upper < -Q(5049, 10**37)
    assert bound.negative_storage_certified
    assert bound.integrated_excess_loss_bounds == (-upper, -lower)
    assert bound.negative_storage_margin == -upper


@pytest.mark.parametrize("changes", ({"amplitude": 0}, {"horizon": 0}))
def test_exact_zero_histories_override_source_defects(changes):
    result = owner._bound_storage_excess(
        **(_arguments() | {"endpoint_radius": Q(1, 100)} | changes)
    )
    assert result.exact_mixed_zero
    assert result.direct_term is None
    assert (
        result.full_excess_storage_bounds
        == result.integrated_excess_loss_bounds
        == (0, 0)
    )
    assert result.source_error_upper_bound == 0
    assert not result.negative_storage_certified
    assert len(result.per_history_errors) == 4


def test_source_uncertainty_can_remove_sufficient_sign_without_claiming_null():
    result = owner._bound_storage_excess(
        **(_arguments() | {"endpoint_radius": Q(1, 10**8)})
    )
    assert (
        result.full_excess_storage_bounds[0] < 0 < result.full_excess_storage_bounds[1]
    )
    assert not result.negative_storage_certified
    assert not result.exact_mixed_zero


def test_exact_scale_survives_below_interval_grid_and_represented_inputs(bound):
    tiny = owner._bound_storage_excess(
        **(_arguments() | {"amplitude": Q(1, 10**50), "endpoint_radius": 0})
    )
    assert tiny.negative_storage_certified
    assert 0 < tiny.negative_storage_margin < Q(1, 2**128)
    normalized = owner._bound_storage_excess(**(_arguments() | {"horizon": 0.125}))
    assert normalized == bound


@pytest.mark.parametrize(
    "key,value",
    (
        ("mediator_class", True),
        ("mediator_class", 1.0),
        ("mediator_class", 3),
        ("amplitude", True),
        ("amplitude", Q(-1, 100)),
        ("amplitude", Q(8, 10000)),
        ("horizon", Decimal("NaN")),
        ("horizon", Q(1, 7)),
        ("horizon", -1),
        ("endpoint_radius", Decimal("Infinity")),
        ("endpoint_radius", -1),
    ),
)
def test_invalid_primitives_reject_before_structural_coefficients(
    key, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached structural coefficient construction")

    monkeypatch.setattr(owner, "_derive_storage_excess_onset", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._bound_storage_excess(**(_arguments() | {key: value}))


def test_complete_full_and_tangent_storage_balance_keeps_both_rows():
    _, edges, degrees, laplacian, _ = _geometry()
    x = tuple(Q((7 * i) % 19 - 9, 37) for i in range(27))
    y = tuple(Q((11 * i) % 17 - 8, 43) for i in range(27))
    lx = _mv(laplacian, x)
    tangent = [[Q(0)] * 27 for _ in range(27)]
    full_gradient = [Q(0)] * 27
    for index, (i, j) in enumerate(edges):
        # Rational edge-gradient witnesses give an exact chain-rule check;
        # the balance is independent of which smooth edge potential supplied them.
        cosine, current = Q(1, 5) + Q(index, 100), Q((index * 5) % 13 - 6, 31)
        tangent[i][i] += cosine
        tangent[j][j] += cosine
        tangent[i][j] -= cosine
        tangent[j][i] -= cosine
        full_gradient[i] -= current
        full_gradient[j] += current
    tangent_gradient = _mv(tangent, y)
    assert _mv(laplacian, tangent_gradient) != _mv(tangent, _mv(laplacian, y))
    gamma = Q(1, 3100)
    for gradient in (full_gradient, tangent_gradient):
        form_rate = tuple(
            -(q + gamma * u) / degree for q, u, degree in zip(lx, gradient, degrees)
        )
        phase_rate = tuple(gamma * q / degree for q, degree in zip(lx, degrees))
        loss = sum(q**2 / degree for q, degree in zip(lx, degrees))
        assert _dot(lx, form_rate) + _dot(gradient, phase_rate) == -loss
        assert _dot(lx, form_rate) != -loss
        assert sum(degree * rate for degree, rate in zip(degrees, form_rate)) == 0
        assert sum(degree * rate for degree, rate in zip(degrees, phase_rate)) == 0


def test_simultaneous_work_has_no_mixed_term_but_tangent_cross_energy_can():
    _, _, _, laplacian, normalized = _geometry()
    x = tuple(Q(i - 13, 101) for i in range(27))
    a, b = Q(3, 19), Q(-5, 23)

    def form_energy(vector):
        return _dot(vector, _mv(laplacian, vector)) / 2

    energies = []
    for left, right in ((0, 0), (a, 0), (0, b), (a, b)):
        state = tuple(
            value + left * (i == 4) + right * (i == 22) for i, value in enumerate(x)
        )
        energy = form_energy(state)
        expected_work = (
            left * _mv(laplacian, x)[4]
            + right * _mv(laplacian, x)[22]
            + Q(3, 2) * (left**2 + right**2)
        )
        assert energy - form_energy(x) == expected_work
        energies.append(energy)
    assert energies[3] - energies[1] - energies[2] + energies[0] == 0
    u, v = tuple(row[4] for row in normalized), tuple(row[22] for row in normalized)
    mixed = (
        form_energy(tuple(a * p + b * q for p, q in zip(u, v)))
        - form_energy(tuple(a * p for p in u))
        - form_energy(tuple(b * q for q in v))
    )
    assert mixed == a * b * _dot(u, _mv(laplacian, v))
    assert mixed != 0


def test_identical_mediator_reading_and_charges_do_not_observe_storage_or_loss():
    _, _, degrees, laplacian, _ = _geometry()
    eps = Q(1, 10**32)
    s = eps / 2
    form = (s, -s) + (Q(0),) * 25
    gradient = _mv(laplacian, form)
    assert form[13] == 0
    assert sum(form[:9]) == sum(form[9:18]) == sum(form[18:]) == 0
    assert sum(d * value for d, value in zip(degrees, form)) == 0
    assert sum(value**2 for value in form) <= eps**2
    assert _dot(form, gradient) / 2 == 3 * s**2
    assert (
        sum(value**2 / degree for value, degree in zip(gradient, degrees)) == 10 * s**2
    )
    # Holding the wound target phases also preserves its winding. This is
    # geometric ambiguity of one readout, not a claim of acquired realizations.


@pytest.mark.parametrize("mediator_class", (1, 2))
def test_coordinate_observation_allowance_covers_both_potentials_and_gauges(
    mediator_class,
):
    mp = mpmath.mp.clone()
    mp.dps = 80
    _, edges, _, _, _ = _geometry()
    x, y, dx, dy = Q(1, 11), Q(1, 7), Q(1, 101), Q(1, 113)
    allowance = owner._bound_storage_observation_error(
        form_bound=x, phase_bound=y, form_error_bound=dx, phase_error_bound=dy
    )
    assert allowance == 8 * (116 * (x * dx + y * dy) + 58 * (dx**2 + dy**2))

    def number(value):
        return mp.mpf(value.numerator) / value.denominator

    classes = (1, mediator_class, 1)
    target = tuple(2 * mp.pi * classes[i // 9] * (i % 9 - 4) / 9 for i in range(27))

    def storage(forms, phases, full):
        result = mp.mpf(0)
        for i, j in edges:
            gap = target[j] - target[i]
            deviation = phases[j] - phases[i]
            result += (forms[j] - forms[i]) ** 2 / 2
            result += (
                mp.cos(gap) - mp.cos(gap + deviation)
                if full
                else mp.cos(gap) * deviation**2 / 2
            )
        return result

    state_x = tuple(number(x) * ((i * 7) % 11 - 5) / 5 for i in range(27))
    state_y = tuple(number(y) * ((i * 3) % 13 - 6) / 6 for i in range(27))
    # Check the actual trigonometric full and target-Hessian tangent balances,
    # in addition to the exact algebraic gradient identity above.
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    gradient_x = [mp.mpf(0)] * 27
    full_y = [mp.mpf(0)] * 27
    tangent_y = [mp.mpf(0)] * 27
    for i, j in edges:
        phase_gap = target[j] - target[i]
        for gradient, edge_value in (
            (gradient_x, state_x[j] - state_x[i]),
            (full_y, mp.sin(phase_gap + state_y[j] - state_y[i])),
            (tangent_y, mp.cos(phase_gap) * (state_y[j] - state_y[i])),
        ):
            gradient[i] -= edge_value
            gradient[j] += edge_value
    gamma = 1 / (1023 * mp.pi)
    loss = sum(value**2 / degree for value, degree in zip(gradient_x, degrees))
    for gradient_y in (full_y, tangent_y):
        derivative = sum(
            (-gx * (gx + gamma * gy) + gy * gamma * gx) / degree
            for gx, gy, degree in zip(gradient_x, gradient_y, degrees)
        )
        assert abs(derivative + loss) < mp.mpf("1e-70")
    initial_gap = storage(state_x, state_y, True) - storage(state_x, state_y, False)
    assert abs(initial_gap) > mp.mpf("1e-8")
    for first, second in ((0, 0), (Q(1, 100), 0), (0, Q(1, 70)), (Q(1, 100), Q(1, 70))):
        post = tuple(
            value + number(Q(first)) * (i == 4) + number(Q(second)) * (i == 22)
            for i, value in enumerate(state_x)
        )
        post_gap = storage(post, state_y, True) - storage(post, state_y, False)
        assert abs(post_gap - initial_gap) < mp.mpf("1e-70")
    for sign_x, sign_y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        measured_x = tuple(
            value + sign_x * (-1) ** i * number(dx) for i, value in enumerate(state_x)
        )
        measured_y = tuple(
            value + sign_y * (-1) ** (i // 2) * number(dy)
            for i, value in enumerate(state_y)
        )
        for full in (False, True):
            initial = storage(state_x, state_y, full)
            measured = storage(measured_x, measured_y, full)
            assert abs(measured - initial) < number(allowance / 8)
            shifted = storage(
                tuple(value + 2 for value in state_x),
                tuple(value - 3 for value in state_y),
                full,
            )
            assert abs(shifted - initial) < mp.mpf("1e-70")


@pytest.mark.parametrize(
    "key", ("form_bound", "phase_bound", "form_error_bound", "phase_error_bound")
)
@pytest.mark.parametrize("value", (True, Q(-1), Decimal("NaN")))
def test_observation_allowance_rejects_invalid_primitives(key, value):
    arguments = dict(
        form_bound=1, phase_bound=1, form_error_bound=0, phase_error_bound=0
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bound_storage_observation_error(**(arguments | {key: value}))


def test_observation_precision_is_separately_supplied_and_not_a_form_delta(bound):
    x, y = Q(1, 500), Q(1, 10**6)
    coarse = owner._bound_storage_observation_error(
        form_bound=x,
        phase_bound=y,
        form_error_bound=Q(1, 10**30),
        phase_error_bound=Q(1, 10**30),
    )
    finer = owner._bound_storage_observation_error(
        form_bound=x,
        phase_bound=y,
        form_error_bound=Q(1, 10**40),
        phase_error_bound=Q(1, 10**40),
    )
    assert finer < bound.negative_storage_margin / 2 < coarse
    assert (
        owner._bound_storage_observation_error(
            form_bound=x, phase_bound=y, form_error_bound=0, phase_error_bound=0
        )
        == 0
    )
