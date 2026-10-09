"""Independent static controls for the matched-target constitutive family."""

from decimal import Decimal
from fractions import Fraction as Q
from math import factorial

import mpmath as mp
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._interval_taylor import Jet, cos, sin
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._rational_interval import cos as interval_cos
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics import _sine_class_constitutive_comparison as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _geometry():
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    matrix = [[Q(0)] * 27 for _ in range(27)]
    for i, j in edges:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return edges, degrees, matrix


def _source():
    parameters = tuple(Q((7 * i) % 17 - 8, 13) for i in range(27))
    return dict(
        form=tuple(Q((11 * i) % 19 - 9, 31) for i in range(27)),
        phase_phasors=tuple(
            ((1 - t**2) / (1 + t**2), 2 * t / (1 + t**2)) for t in parameters
        ),
        deformation=Q(3, 5),
    )


def _multiply(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _harmonic(phasor, order):
    result = (Q(1), Q(0))
    for _ in range(order):
        result = _multiply(result, phasor)
    return result


def _convolve(left, right):
    return tuple(
        sum((left[j] * right[n - j] for j in range(n + 1)), Q(0))
        for n in range(len(left))
    )


def _potential_angle_coefficients(phasor, deformation):
    """Formal angle derivatives from complex powers, not time evolution."""
    real, imaginary = phasor
    cycle = (
        (real, imaginary),
        (-imaginary, real),
        (-real, -imaginary),
        (imaginary, -real),
    )
    phase = tuple(
        tuple(cycle[n % 4][j] / factorial(n) for n in range(5)) for j in range(2)
    )
    power = ((Q(1),) + (Q(0),) * 4, (Q(0),) * 5)
    for _ in range(9):
        ac, bd = _convolve(power[0], phase[0]), _convolve(power[1], phase[1])
        ad, bc = _convolve(power[0], phase[1]), _convolve(power[1], phase[0])
        power = tuple(a - b for a, b in zip(ac, bd)), tuple(
            a + b for a, b in zip(ad, bc)
        )
    gap = (1 - power[0][0],) + tuple(-x for x in power[0][1:])
    square = _convolve(gap, gap)
    return tuple(
        Q(n == 0) - phase[0][n] + deformation * square[n] / 39366 for n in range(5)
    )


@pytest.fixture(scope="module")
def state():
    return owner._observe_constitutive_state(**_source())


def test_every_edge_current_is_the_derivative_of_its_own_potential(state):
    edges, degrees, _ = _geometry()
    assert state.edges == edges and state.degrees == degrees
    for edge, phasor, potential, derivatives in zip(
        edges,
        state.relative_phasors,
        state.edge_potential_values,
        state.edge_current_derivatives,
    ):
        i, j = edge
        ci, si = state.phase_phasors[i]
        assert phasor == _multiply((ci, -si), state.phase_phasors[j])
        expected = _potential_angle_coefficients(phasor, state.deformation)
        assert potential == expected[0]
        assert derivatives == tuple(
            expected[n + 1] * factorial(n + 1) for n in range(4)
        )
        assert potential >= 0
        reversed_potential, reversed_jet = owner._edge_law(
            phasor[0], -phasor[1], state.deformation
        )
        assert reversed_potential == potential
        assert reversed_jet == tuple(
            (-1) ** (n + 1) * value for n, value in enumerate(derivatives)
        )


def test_full_rows_balance_in_actual_degree_metric_and_tau_clock(state):
    edges, degrees, matrix = _geometry()
    gradient = tuple(
        sum((a * b for a, b in zip(row, state.form)), Q(0)) for row in matrix
    )
    currents = [Q(0)] * 27
    potential = Q(0)
    for i, j in edges:
        ci, si = state.phase_phasors[i]
        c, s = _multiply((ci, -si), state.phase_phasors[j])
        c9, s9 = _harmonic((c, s), 9)
        current = s + state.deformation * s9 * (1 - c9) / 2187
        currents[i] += current
        currents[j] -= current
        potential += 1 - c + state.deformation * (1 - c9) ** 2 / 39366
    assert state.form_gradient == gradient
    assert state.phase_current_sums == tuple(currents)
    assert state.phase_storage == potential
    assert (
        state.form_storage
        == sum((x * q for x, q in zip(state.form, gradient)), Q(0)) / 2
    )
    assert state.storage == state.form_storage + potential
    expected_loss = sum((q**2 / d for q, d in zip(gradient, degrees)), Q(0))
    assert expected_loss > 0 and state.loss == expected_loss
    assert state.storage_rate == -expected_loss
    assert state.storage_rate_from_rows_bounds.contains(-expected_loss)
    # Exact algebraic gamma values test both reciprocal rows independently of
    # any approximate physical coefficient. The actual fixed gamma is enclosed.
    for gamma in (state.gamma_bounds.lo, state.gamma_bounds.hi):
        dx = tuple((-q + gamma * s) / d for q, s, d in zip(gradient, currents, degrees))
        dy = tuple(gamma * q / d for q, d in zip(gradient, degrees))
        assert all(band.contains(x) for band, x in zip(state.form_rate_bounds, dx))
        assert all(band.contains(y) for band, y in zip(state.phase_rate_bounds, dy))
        assert (
            sum((q * x - s * y for q, s, x, y in zip(gradient, currents, dx, dy)), Q(0))
            == -expected_loss
        )
        assert sum((d * x for d, x in zip(degrees, dx)), Q(0)) == 0
        assert sum((d * y for d, y in zip(degrees, dy)), Q(0)) == 0
        assert sum((q * x for q, x in zip(gradient, dx)), Q(0)) != -expected_loss
    assert all(value.lo == value.hi == 0 for value in state.weighted_charge_rate_bounds)
    assert state.clock == "tau=e*t; e=1023/1024"


def test_independent_trigonometric_full_field_and_both_symmetries(state):
    def mpq(value):
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(90):
        angles = tuple(mp.atan2(mpq(s), mpq(c)) for c, s in state.phase_phasors)
        gamma = 1 / (1023 * mp.pi)
        for i in range(27):
            neighbors = tuple(j if i == k else k for k, j in state.edges if i in (k, j))
            pressure = sum(
                mp.sin(angles[j] - angles[i])
                + mpq(state.deformation)
                * mp.sin(9 * (angles[j] - angles[i]))
                * (1 - mp.cos(9 * (angles[j] - angles[i])))
                / 2187
                for j in neighbors
            )
            form_rate = (-mpq(state.form_gradient[i]) + gamma * pressure) / len(
                neighbors
            )
            phase_rate = gamma * mpq(state.form_gradient[i]) / len(neighbors)
            assert (
                mpq(state.form_rate_bounds[i].lo)
                <= form_rate
                <= mpq(state.form_rate_bounds[i].hi)
            )
            assert (
                mpq(state.phase_rate_bounds[i].lo)
                <= phase_rate
                <= mpq(state.phase_rate_bounds[i].hi)
            )
    rotation = (Q(3, 5), Q(4, 5))
    rotated = owner._observe_constitutive_state(
        **(
            _source()
            | {
                "phase_phasors": tuple(
                    _multiply(rotation, p) for p in state.phase_phasors
                ),
                "form": tuple(x + 17 for x in state.form),
            }
        )
    )
    assert rotated.relative_phasors == state.relative_phasors
    assert rotated.form_rate_bounds == state.form_rate_bounds
    assert rotated.phase_rate_bounds == state.phase_rate_bounds
    assert rotated.storage == state.storage
    assert rotated.weighted_form_charge - state.weighted_form_charge == 58 * 17


@pytest.mark.parametrize("turn", (0, 1, 2, 7, 8))
def test_actual_potential_jets_match_at_all_target_gap_types(turn):
    angle = Jet((2 * Q(turn, 9) * pi_interval(), I(1), I(0), I(0), I(0)))
    base, base_current = owner._edge_law(cos(angle), sin(angle), Q(0))
    alternative, current = owner._edge_law(cos(angle), sin(angle), Q(1))
    difference = alternative - base
    for order, value in enumerate(difference.coeffs):
        expected = Q(1, 24) if order == 4 else Q(0)
        assert value.contains(expected)
        assert value.width < Q(1, 10**28)
    for order in range(4):
        difference = (current[order] - base_current[order]).coeffs[0]
        assert difference.contains(Q(order == 3))
        assert difference.width < Q(1, 10**28)


@pytest.mark.parametrize("mediator_class", (1, 2))
@pytest.mark.parametrize("deformation", (Q(0), Q(1), Q(7, 3)))
def test_mixed_onset_from_independent_full_edge_cubic_expansion(
    mediator_class, deformation
):
    report = owner._derive_constitutive_onset(
        mediator_class=mediator_class,
        deformation=deformation,
        donor_amplitude=Q(2, 11),
        receiver_amplitude=Q(3, 17),
    )
    edges, degrees, matrix = _geometry()
    u = tuple(row[4] / d for row, d in zip(matrix, degrees))
    v = tuple(row[22] / d for row, d in zip(matrix, degrees))
    weights = [[Q(0), Q(0)] for _ in range(4)]
    for index, (i, j) in enumerate(edges):
        du, dv = u[j] - u[i], v[j] - v[i]
        incidence = Q(int(i == 13) - int(j == 13), degrees[13])
        # Derivative/6, mixed multiplicity 3 and integration of t^3.
        weights[min(index // 9, 3)][0] += incidence * du**2 * dv / 8
        weights[min(index // 9, 3)][1] += incidence * du * dv**2 / 8
    assert weights == [[0, 0], [Q(1, 1024)] * 2, [0, 0], [Q(15, 1024)] * 2]
    cosine = report.geometry.channel_cosine_bounds[1]
    expected = (
        (16 * deformation - 15 - cosine.hi) / 1024,
        (16 * deformation - 15 - cosine.lo) / 1024,
    )
    assert report.normalized_mixed_coefficient_bounds == expected
    amplitude = Q(2, 11) * Q(3, 17) * (Q(2, 11) + Q(3, 17))
    corners = tuple(
        g**4 * amplitude * c
        for g in (report.geometry.gamma_bounds.lo, report.geometry.gamma_bounds.hi)
        for c in expected
    )
    assert report.mixed_time_four_coefficient_bounds == (min(corners), max(corners))
    assert report.target_potential_derivative_differences == (0, 0, 0, 0, deformation)
    if deformation == 0:
        assert report.mixed_time_four_coefficient_bounds[1] < 0
    if deformation == 1:
        assert report.mixed_time_four_coefficient_bounds[0] > 0
    # The public result is a local coefficient, never a horizon decision.
    assert not hasattr(report, "response_bound_available")


def _full_local_coefficients(mediator_class, deformation, donor, receiver):
    """Four static derivatives of the actual full field, with both rows.

    This independent local jet expansion supplies no time interval, numerical
    integration, trajectory enclosure or sine-specific amplitude recurrence.
    """
    edges, degrees, _ = _geometry()
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    classes = (1, mediator_class, 1)
    initial = tuple(
        I(donor * (i == 4) + receiver * (i == 22)) for i in range(27)
    ) + tuple(2 * Q(classes[c] * (j - 4), 9) * pi for c in range(3) for j in range(9))
    coefficients = [initial]
    for order in range(4):
        state = tuple(Jet(tuple(row[i] for row in coefficients)) for i in range(54))
        rates = [Jet.constant(0, order) for _ in range(54)]
        for i, j in edges:
            gap = state[27 + j] - state[27 + i]
            phase_current = (
                sin(gap) + deformation * sin(9 * gap) * (1 - cos(9 * gap)) / 2187
            )
            difference = state[j] - state[i]
            form_current = difference + phase_current * gamma
            phase_rate = difference * (-gamma)
            rates[i] += form_current / degrees[i]
            rates[j] -= form_current / degrees[j]
            rates[27 + i] += phase_rate / degrees[i]
            rates[27 + j] -= phase_rate / degrees[j]
        coefficients.append(tuple(rate.coeffs[order] / (order + 1) for rate in rates))
    return tuple(coefficients)


@pytest.mark.parametrize("mediator_class", (1, 2))
@pytest.mark.parametrize("deformation", (Q(0), Q(1)))
def test_full54_local_jets_exclude_earlier_mixed_terms_and_reproduce_onset(
    mediator_class, deformation
):
    donor, receiver = Q(1, 13), Q(2, 17)
    histories = tuple(
        _full_local_coefficients(mediator_class, deformation, a, b)
        for a, b in ((0, 0), (donor, 0), (0, receiver), (donor, receiver))
    )
    mixed = tuple(
        histories[3][n][13]
        - histories[1][n][13]
        - histories[2][n][13]
        + histories[0][n][13]
        for n in range(5)
    )
    assert all(value.contains(0) for value in mixed[:4])
    assert all(value.width < Q(1, 10**28) for value in mixed)
    report = owner._derive_constitutive_onset(
        mediator_class=mediator_class,
        deformation=deformation,
        donor_amplitude=donor,
        receiver_amplitude=receiver,
    )
    lower, upper = report.mixed_time_four_coefficient_bounds
    assert mixed[4].lo <= lower <= upper <= mixed[4].hi
    if deformation:
        assert mixed[4].lo > 0
    else:
        assert mixed[4].hi < 0
    # Every positive-degree coefficient keeps both actual weighted charges.
    _, degrees, _ = _geometry()
    for history in histories:
        for coefficient in history[1:]:
            for offset in (0, 27):
                charge = sum(
                    (d * coefficient[offset + i] for i, d in enumerate(degrees)), I(0)
                )
                assert charge.contains(0) and charge.width < Q(1, 10**28)


def test_zero_onset_and_signed_subgrid_products_are_not_finite_null_claims():
    arguments = dict(
        mediator_class=2,
        deformation=1,
        donor_amplitude=Q(1, 10**20),
        receiver_amplitude=Q(2, 10**20),
    )
    positive = owner._derive_constitutive_onset(**arguments)
    negative = owner._derive_constitutive_onset(
        **(
            arguments
            | {
                "donor_amplitude": -arguments["donor_amplitude"],
                "receiver_amplitude": -arguments["receiver_amplitude"],
            }
        )
    )
    assert (
        0
        < positive.mixed_time_four_coefficient_bounds[0]
        < positive.mixed_time_four_coefficient_bounds[1]
        < Q(1, 2**128)
    )
    assert negative.mixed_time_four_coefficient_bounds == tuple(
        -v for v in reversed(positive.mixed_time_four_coefficient_bounds)
    )
    null = owner._derive_constitutive_onset(
        **(arguments | {"receiver_amplitude": -arguments["donor_amplitude"]})
    )
    assert null.mixed_time_four_coefficient_bounds == (0, 0)
    assert not hasattr(null, "exact_mixed_zero")


def test_bridge_cubic_removal_class_threshold_and_preserved_class_contrast():
    reports = {
        (k, coefficient): owner._derive_constitutive_onset(
            mediator_class=k,
            deformation=coefficient,
            donor_amplitude=Q(1, 13),
            receiver_amplitude=Q(2, 17),
        )
        for k in (1, 2)
        for coefficient in (Q(0), Q(31, 32), Q(1))
    }
    assert reports[1, Q(31, 32)].mixed_time_four_coefficient_bounds[1] < 0
    assert reports[2, Q(31, 32)].mixed_time_four_coefficient_bounds[0] > 0
    for k in (1, 2):
        bridge = reports[k, Q(1)].target_current_third_derivative_bounds[3]
        assert bridge.lo == bridge.hi == 0
        assert reports[k, Q(1)].target_current_third_derivative_bounds[1].lo > 0
    # Compare like endpoints before subtraction, preserving the common lambda.
    # This family shifts both class coefficients equally; it does not remove
    # every other nonlinear-law freedom or transfer a finite class response.
    differences = {
        coefficient: tuple(
            a - b
            for a, b in zip(
                reports[1, coefficient].normalized_mixed_coefficient_bounds,
                reports[2, coefficient].normalized_mixed_coefficient_bounds,
            )
        )
        for coefficient in (Q(0), Q(31, 32), Q(1))
    }
    assert differences[Q(0)] == differences[Q(31, 32)] == differences[Q(1)]
    assert differences[Q(0)][1] < 0


def test_nominal_identity_budget_has_its_own_graph_and_potential_bounds():
    edges, _, laplacian = _geometry()
    largest_distance = 0
    for start in range(27):
        distances, frontier = {start: 0}, [start]
        while frontier:
            node = frontier.pop(0)
            for i, j in edges:
                if node not in (i, j):
                    continue
                neighbor = j if node == i else i
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
        assert len(distances) == 27
        largest_distance = max(largest_distance, max(distances.values()))
    assert largest_distance == 10
    gap = Q(2, 135)
    slack = tuple(
        tuple(value - gap * (Q(i == j) - Q(1, 27)) for j, value in enumerate(row))
        for i, row in enumerate(laplacian)
    )
    assert exact_symmetric_semidefinite(slack)
    radius = Q(1, 12)
    assert Q(17, 12) ** 2 > 2
    sine_curvature = interval_cos(4 * pi_interval() / 9 + Q(17, 12) * radius)
    assert sine_curvature.lo > Q(1, 20)
    # W''=(cos(9d)-cos(18d))/243 >= -2/243. This covers
    # every lambda in [0,1] on the entire Euclidean phase-error ball.
    assert sine_curvature.lo - Q(2, 243) > Q(1, 25)
    coercivity = gap / 50
    assert coercivity == Q(1, 3375)
    impulse = Q(7, 10000)
    source = tuple(impulse * (i in (4, 22)) for i in range(27))
    mean = sum(source, Q(0)) / 27
    squared_radius = sum(((value - mean) ** 2 for value in source), Q(0))
    work = (
        sum(
            (
                source[i] * sum((value * x for value, x in zip(row, source)), Q(0))
                for i, row in enumerate(laplacian)
            ),
            Q(0),
        )
        / 2
    )
    assert squared_radius == Q(50, 27) * impulse**2 < radius**2
    assert work == 3 * impulse**2 < coercivity * radius**2 == Q(1, 486000)


def test_deformation_zero_recovers_sine_but_does_not_relabel_its_certificates(state):
    sine = owner._observe_constitutive_state(**(_source() | {"deformation": 0}))
    assert sine.edge_potential_values == tuple(1 - c for c, _ in sine.relative_phasors)
    assert tuple(row[0] for row in sine.edge_current_derivatives) == tuple(
        s for _, s in sine.relative_phasors
    )
    assert sine.phase_rate_bounds == state.phase_rate_bounds
    assert sine.form_rate_bounds != state.form_rate_bounds
    assert sine.form_storage == state.form_storage and sine.loss == state.loss
    assert not hasattr(state, "acquisition_certified")


def test_first_resultant_does_not_determine_the_new_current():
    plain = [(Q(1), Q(0))] * 27
    rotated = plain.copy()
    plain[12], plain[14] = (Q(1), Q(0)), (Q(-1), Q(0))
    rotated[12], rotated[14] = (Q(3, 5), Q(4, 5)), (Q(-3, 5), Q(-4, 5))
    neighbors = (4, 12, 14, 22)
    assert tuple(sum(plain[i][c] for i in neighbors) for c in range(2)) == (2, 0)
    assert tuple(sum(rotated[i][c] for i in neighbors) for c in range(2)) == (2, 0)
    reports = tuple(
        owner._observe_constitutive_state(
            form=(Q(0),) * 27, phase_phasors=phases, deformation=1
        )
        for phases in (plain, rotated)
    )
    assert reports[0].phase_current_sums[13] == 0
    assert reports[1].phase_current_sums[13] != 0
    assert (
        reports[1].form_rate_bounds[13].lo > 0 or reports[1].form_rate_bounds[13].hi < 0
    )


def test_global_current_derivative_bounds_come_from_independent_fourier_amplitudes():
    coefficient = Q(5, 7)
    # j=sin(delta)+lambda*(sin(9delta)-sin(18delta)/2)/2187.
    for order in range(17):
        independent = 1 + coefficient * (9**order + Q(18**order, 2)) / 2187
        assert owner._current_derivative_bound(coefficient, order) == independent
    assert owner._current_derivative_bound(0, 16) == 1


@pytest.mark.parametrize(
    "bad", (True, False, float("nan"), float("inf"), Decimal("NaN"), "1", None)
)
@pytest.mark.parametrize("field", ("deformation", "form", "phase_phasors"))
def test_state_scalars_are_admitted_before_parameter_materialization(
    bad, field, monkeypatch
):
    arguments = _source()
    if field == "form":
        arguments[field] = (bad,) + arguments[field][1:]
    elif field == "phase_phasors":
        arguments[field] = ((bad, 0),) + arguments[field][1:]
    else:
        arguments[field] = bad
    monkeypatch.setattr(owner, "_parameters", lambda *_: pytest.fail("late admission"))
    with pytest.raises((TypeError, ValueError)):
        owner._observe_constitutive_state(**arguments)


@pytest.mark.parametrize("field", ("form", "phase_phasors"))
@pytest.mark.parametrize("length", (0, 26, 28))
def test_exact_state_shape_is_required(field, length):
    arguments = _source()
    value = Q(0) if field == "form" else (Q(1), Q(0))
    arguments[field] = (value,) * length
    with pytest.raises(ValueError):
        owner._observe_constitutive_state(**arguments)


def test_iterators_are_bounded_and_unit_phasors_are_not_normalized():
    def too_long():
        for _ in range(28):
            yield Q(0)
        pytest.fail("state admission read beyond its fixed shape budget")

    with pytest.raises(ValueError):
        owner._observe_constitutive_state(**(_source() | {"form": too_long()}))
    for phasor in ((Q(1), Q(1)), (Q(1),), (Q(1), Q(0), Q(0))):
        with pytest.raises(ValueError):
            owner._observe_constitutive_state(
                **(_source() | {"phase_phasors": (phasor,) * 27})
            )
    with pytest.raises(ValueError):
        owner._observe_constitutive_state(**(_source() | {"deformation": -1}))


@pytest.mark.parametrize(
    "field,bad",
    (
        ("deformation", True),
        ("deformation", -1),
        ("donor_amplitude", float("inf")),
        ("receiver_amplitude", Decimal("NaN")),
    ),
)
def test_onset_scalar_admission_precedes_geometry(field, bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive_neighbor_onset", lambda *_: pytest.fail("late admission")
    )
    arguments = dict(
        mediator_class=2,
        deformation=1,
        donor_amplitude=Q(1, 3),
        receiver_amplitude=Q(1, 4),
    ) | {field: bad}
    with pytest.raises((TypeError, ValueError)):
        owner._derive_constitutive_onset(**arguments)


@pytest.mark.parametrize("bad", (True, Q(1), 1.0, 0, 3))
def test_onset_requires_an_ordinary_admitted_class(bad):
    with pytest.raises(ValueError):
        owner._derive_constitutive_onset(
            mediator_class=bad, deformation=1, donor_amplitude=1, receiver_amplitude=2
        )


@pytest.mark.parametrize("bad", (True, Q(1), 1.0, -1, 17))
def test_derivative_order_policy_is_not_scalar_coercion(bad):
    with pytest.raises(ValueError):
        owner._current_derivative_bound(1, bad)
