"""Coefficient freedom and identification in the conditional relational law."""

import math
from copy import deepcopy
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.mathematics.linear_observation import (
    derive_coordinate_memory,
    derive_linear_observation,
)
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
)
from tnfr.physics.relational_observations import bound_relational_coefficient_from_jet
from tnfr.physics.support_transport import observe_support_transport


def _path(forms=(0.0, 0.0, 0.0), phases=(0.0, 0.0, 0.0), capacities=(1, 0.5, 2)):
    graph = nx.path_graph(len(forms))
    graph.graph["GAMMA"] = {"type": "none"}
    for node, form, phase, capacity in zip(
        graph, forms, phases, capacities, strict=True
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=99.0)
    return graph


def _difference_generator(model, capacity=1.0):
    graph = _path((0, 0), (0, 0), (capacity, capacity))
    tangent = evaluate_relational_uniform_tangent(graph, model=model)
    lift = np.array(((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)))
    observe = np.array(((1, -1, 0, 0), (0, 0, 1, -1)))
    full = np.array(tangent.generator)
    reduced = observe @ full @ lift
    np.testing.assert_allclose(full @ lift, lift @ reduced, rtol=0, atol=2e-15)
    return reduced


def test_balance_and_capacity_additivity_leave_different_damping_ratios():
    graph = _path((0.5, 0.0, -0.25), (-0.125, 0.25, 0.125))
    saved = deepcopy(graph)
    fields, discriminants = [], []
    for beta in (1.0, 0.25):
        model = RelationalExchangeModel(beta)
        field = evaluate_relational_exchange(graph, model=model)
        fields.append(field)
        assert field.continuous_loss > 0
        assert any(field.work.exchange)
        assert abs(field.balance_residual) < Q(1, 10**14)
        parts = []
        for capacities in ((1, 0, 2), (0, 0.5, 0)):
            part = _path(field.epi, field.phase, capacities)
            reading = evaluate_relational_exchange(part, model=model)
            parts.append(np.array((*reading.form_rate, *reading.phase_rate)))
            for i, capacity in enumerate(capacities):
                if capacity == 0:
                    assert reading.form_rate[i] == reading.phase_rate[i] == 0
        np.testing.assert_allclose(
            parts[0] + parts[1],
            (*field.form_rate, *field.phase_rate),
            rtol=0,
            atol=2e-15,
        )
        generator = _difference_generator(model)
        discriminants.append(np.trace(generator) ** 2 - 4 * np.linalg.det(generator))
    assert fields[0].pressure == fields[1].pressure
    assert fields[0].form_rate == fields[1].form_rate
    assert fields[0].continuous_loss == fields[1].continuous_loss
    assert fields[1].phase_rate == pytest.approx(
        tuple(4 * rate for rate in fields[0].phase_rate), rel=0, abs=2e-15
    )
    assert discriminants[0] > 0 > discriminants[1]
    assert nx.utils.graphs_equal(graph, saved)


def test_normalized_form_clock_conversion_preserves_both_rows_and_invariant():
    graph = _path((0.5, 0.0, -0.25), (-0.125, 0.25, 0.125))
    model = RelationalExchangeModel(0.75)
    before = evaluate_relational_exchange(graph, model=model)
    amplitude, clock = 3, 2
    e, w = model.effective_weights
    normalization = e + amplitude * w
    converted = RelationalExchangeModel(
        amplitude**2 * model.storage_scale, epi_weight=e, phase_weight=amplitude * w
    )
    after_graph = deepcopy(graph)
    for node in after_graph:
        after_graph.nodes[node]["EPI"] = amplitude * graph.nodes[node]["EPI"] - 0.25
        after_graph.nodes[node]["nu_f"] *= normalization / clock
    after = evaluate_relational_exchange(after_graph, model=converted)
    assert after.form_rate == pytest.approx(
        tuple(amplitude * rate / clock for rate in before.form_rate), abs=2e-15
    )
    assert after.phase_rate == pytest.approx(
        tuple(rate / clock for rate in before.phase_rate), abs=2e-15
    )
    assert after.storage == amplitude**2 * before.storage
    original = model.storage_scale * (e / w) ** 2
    observed = (
        converted.storage_scale * (converted.epi_weight / converted.phase_weight) ** 2
    )
    assert observed == pytest.approx(original, rel=0, abs=2e-15)
    for node in after_graph:
        after_graph.nodes[node]["nu_f"] /= normalization
    wrong = evaluate_relational_exchange(after_graph, model=converted)
    assert np.linalg.norm(np.array(wrong.phase_rate) - after.phase_rate) > 1e-3


def test_phase_source_normalization_moves_the_numeric_boundary_not_the_law():
    s = pytest.importorskip("sympy")
    e, w, beta, scale, nu, q, degree, h = s.symbols(
        "e w beta scale nu q degree h", positive=True
    )
    g = s.Symbol("g", real=True)
    original = s.Matrix((nu * (-e * q / degree + w * g), w * nu * q / (beta * h)))
    rescaled = s.Matrix(
        (
            nu * (-e * q / degree + (w / scale) * (scale * g)),
            (w / scale) * nu * q / (beta * h / scale),
        )
    )
    assert (rescaled - original).applyfunc(s.simplify) == s.zeros(2, 1)
    chi = beta * (e / w) ** 2
    changed_chi = beta * (e / (w / scale)) ** 2
    assert s.simplify(changed_chi / (4 * scale**2 / s.pi**2) - chi / (4 / s.pi**2)) == 0


def test_positive_exchange_scale_cannot_remove_the_p2_jacobi_defect():
    s = pytest.importorskip("sympy")
    x0, x1, t0, t1, delta = s.symbols("x0 x1 t0 t1 delta", real=True)
    w, beta, n0, n1 = s.symbols("w beta n0 n1", positive=True)
    coordinates = (x0, x1, t0, t1)
    h = s.pi * s.sin(t1 - t0) / (t1 - t0)
    mobility = s.diag(n0 / h, n1 / h)
    tensor = (w / beta) * s.zeros(2).row_join(-mobility).col_join(
        mobility.row_join(s.zeros(2))
    )
    triple = (1, 0, 2)
    jacobiator = sum(
        tensor[i, k] * s.diff(tensor[j, l], coordinate)
        for i, j, l in (triple, (0, 2, 1), (2, 1, 0))
        for k, coordinate in enumerate(coordinates)
    )
    h_delta = s.pi * s.sin(delta) / delta
    expected = -((w / beta) ** 2) * n0 * n1 * s.diff(h_delta, delta) / h_delta**3
    assert s.simplify(jacobiator.subs({t0: 0, t1: delta}) - expected) == 0
    positive = expected.subs({delta: s.pi / 3, w: 1, beta: 1, n0: 1, n1: 1})
    assert positive.is_positive


def test_state_dependent_capacity_slopes_drop_out_only_at_joint_equilibrium():
    graph = _path()
    model = RelationalExchangeModel(1.0)
    base_capacity = np.array([graph.nodes[i]["nu_f"] for i in graph])
    reference = np.array(
        evaluate_relational_uniform_tangent(graph, model=model).generator
    )
    h = 1e-5

    def rates(vector):
        capacities = base_capacity * np.exp(2 * vector[:3] - 3 * vector[3:])
        sample = _path(vector[:3], vector[3:], capacities)
        field = evaluate_relational_exchange(sample, model=model)
        return np.array((*field.form_rate, *field.phase_rate))

    columns = []
    for axis in np.eye(6):
        columns.append((rates(h * axis) - rates(-h * axis)) / (2 * h))
    np.testing.assert_allclose(np.array(columns).T, reference, rtol=0, atol=3e-9)
    phase_state = np.array((0, 0, 0, -0.125, 0.0, 0.25))
    shift = np.array((1, 1, 1, 0, 0, 0))
    change = (rates(phase_state + h * shift) - rates(phase_state - h * shift)) / (2 * h)
    np.testing.assert_allclose(change, 2 * rates(phase_state), rtol=0, atol=3e-9)
    assert np.linalg.norm(change) > 1e-3


@pytest.fixture(scope="module")
def exact_consensus_memory():
    """Exact rational reference using represented pi, not mathematical pi."""
    pi = Q(math.pi)
    e = w = Q(1, 2)
    beta = Q(1)
    laplacian = np.array(((1, -1, 0), (-1, 2, -1), (0, -1, 1)), dtype=object)
    transport = np.diag((Q(1), Q(1, 4), Q(2))) @ laplacian
    zero = np.zeros((3, 3), dtype=object)
    generator = np.block(
        [[-e * transport, -w / pi * transport], [w / (beta * pi) * transport, zero]]
    )
    memory = derive_coordinate_memory(generator.tolist(), (0, 1, 2))
    return transport, generator, memory, pi, beta * (e / w) ** 2


def test_hidden_phase_memory_carries_the_same_coefficient_invariant(
    exact_consensus_memory,
):
    transport, generator, memory, pi, chi = exact_consensus_memory
    instantaneous = np.array(memory.visible_generator, dtype=object)
    kernel = np.array(memory.kernel_at_zero, dtype=object)
    np.testing.assert_array_equal(memory.hidden_generator, np.zeros((3, 3)))
    np.testing.assert_array_equal(kernel, -(transport @ transport) / (4 * pi**2))
    np.testing.assert_array_equal(
        instantaneous @ instantaneous + pi**2 * chi * kernel, 0
    )
    native = evaluate_relational_uniform_tangent(
        _path(), model=RelationalExchangeModel(1.0)
    )
    np.testing.assert_allclose(
        native.generator, np.array(generator, dtype=float), rtol=0, atol=2e-15
    )


def test_hidden_initial_phase_is_not_removed_by_observing_all_forms(
    exact_consensus_memory,
):
    _, generator, memory, _, _ = exact_consensus_memory
    observe = np.concatenate(
        (np.eye(3, dtype=int), np.zeros((3, 3), dtype=int)), axis=1
    )
    realization = derive_linear_observation(generator.tolist(), observe.tolist())
    assert realization.dimension == 5
    source = np.array(memory.hidden_to_visible, dtype=object)
    assert any(source @ np.array((1, -2, 1), dtype=object))
    np.testing.assert_array_equal(source @ np.ones(3, dtype=object), 0)


@pytest.mark.parametrize("beta,capacity", ((1.0, 1.0), (0.25, 3.0), (4.0, 0.5)))
def test_two_same_mode_poles_identify_chi_without_a_clock_scale(beta, capacity):
    model = RelationalExchangeModel(beta)
    poles = np.linalg.eigvals(_difference_generator(model, capacity))
    identified = poles.sum() ** 2 / (math.pi**2 * poles.prod())
    assert identified == pytest.approx(beta, rel=2e-14, abs=2e-15)


def test_observation_rank_does_not_guarantee_two_excited_poles():
    generator = ((-3, -2), (1, 0))
    observed = derive_linear_observation(generator, ((1, 0),))
    assert observed.dimension == 2
    eigenline = np.array((-1, 1))
    np.testing.assert_array_equal(np.array(generator) @ eigenline, -eigenline)
    filtered = derive_linear_observation(generator, ((1, 2),))
    assert filtered.dimension == 1
    assert filtered.reduced_generator == ((Q(-1),),)


def test_pairing_poles_from_different_spatial_modes_gives_a_false_ratio():
    generator = ((-3, -2, 0, 0), (1, 0, 0, 0), (0, 0, -9, -6), (0, 0, 3, 0))
    observed = derive_linear_observation(generator, ((1, 0, 1, 0),))
    assert observed.dimension == 4
    first_mode = Q((-1 - 2) ** 2, (-1) * (-2))
    second_mode = Q((-3 - 6) ** 2, (-3) * (-6))
    mixed = Q((-1 - 3) ** 2, (-1) * (-3))
    assert first_mode == second_mode == Q(9, 2)
    assert mixed == Q(16, 3) != first_mode


def test_real_poles_allow_form_overshoot_in_the_consensus_tangent():
    generator = _difference_generator(RelationalExchangeModel(1.0))
    poles = np.linalg.eigvals(generator)
    assert np.isreal(poles).all() and (poles < 0).all()
    slow, fast = sorted(poles, reverse=True)
    initial = np.array((1.0, 0.0))
    assert (generator @ initial)[0] == -1.0
    ratio_at_four = (slow * math.exp(4 * slow) - fast * math.exp(4 * fast)) / (
        slow - fast
    )
    assert -0.061 < ratio_at_four < -0.060


def test_same_coefficients_have_different_pole_classes_at_a_winding_equilibrium():
    model = RelationalExchangeModel(1.0)
    classes = []
    for winding in (0, 1):
        graph = nx.cycle_graph(5)
        for node in graph:
            graph.nodes[node].update(
                EPI=0.0, theta=math.tau * winding * node / 5, nu_f=1.0
            )
        tangent = evaluate_relational_uniform_tangent(graph, model=model)
        poles = np.linalg.eigvals(tangent.generator)
        classes.append(int(np.count_nonzero(np.abs(poles.imag) > 1e-10)))
        assert max(map(abs, tangent.field.form_rate)) < 1e-14
    assert classes == [0, 8]


def test_moving_boundary_generates_hidden_excess_despite_total_loss():
    graph = _path((1.0, 0.5, 0.0), capacities=(1, 3, 2))
    field = evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    reduced_storage = (Q(field.epi[0]) - Q(field.epi[2])) ** 2 / 4
    assert field.storage - reduced_storage == 0
    assert field.storage_rate == -Q(3, 8)
    hidden_form_rate = (
        field.form_rate[1] - (field.form_rate[0] + field.form_rate[2]) / 2
    )
    hidden_phase_rate = (
        field.phase_rate[1] - (field.phase_rate[0] + field.phase_rate[2]) / 2
    )
    assert hidden_form_rate == -Q(1, 8)
    assert hidden_phase_rate == pytest.approx(1 / (8 * math.pi), rel=0, abs=2e-16)
    excess_acceleration = 2 * (hidden_form_rate**2 + hidden_phase_rate**2)
    assert excess_acceleration == pytest.approx((1 + 1 / math.pi**2) / 32, abs=2e-16)
    assert excess_acceleration > 0


def _prepared_mode_jet(mode, model, *, capacity=1.0, amplitude=Q(1, 8)):
    size = len(mode)
    graph = _path(
        tuple(float(Q(1, 2) + amplitude * value) for value in mode),
        (0,) * size,
        (capacity,) * size,
    )
    field = evaluate_relational_exchange(graph, model=model)
    detached = deepcopy(graph)
    for node, pressure in zip(field.nodes, field.pressure, strict=True):
        detached.nodes[node]["delta_nfr"] = pressure
    snapshot = observe_support_transport(detached)
    reference = derive_phase_response(
        cosine_gram=((1,) * size,) * size,
        mean_neighbors=snapshot.support_neighbors,
        receiver_sources=tuple((i,) for i in range(size)),
        phase_factor=0,
    )
    response = derive_joint_nodal_response(
        snapshot,
        reference,
        epi_weight=Q(model.epi_weight),
        phase_weight=Q(model.phase_weight),
        capacity_weight=0,
        phase_rate_over_pi=tuple(Q(value) / Q(math.pi) for value in field.phase_rate),
        capacity_rate=(0,) * size,
    )
    degree = tuple(len(row) for row in snapshot.support_neighbors)
    norm = sum(d * value**2 for d, value in zip(degree, mode, strict=True))
    weights = tuple(Q(d * value, norm) for d, value in zip(degree, mode, strict=True))

    def project(values):
        return sum(
            weight * Q(value) for weight, value in zip(weights, values, strict=True)
        )

    return (
        field,
        response,
        weights,
        tuple(
            project(values)
            for values in (field.epi, field.form_rate, response.epi_acceleration)
        ),
    )


@pytest.mark.parametrize(
    "mode,eigenvalue,raw_weights,beta,capacity,amplitude",
    (
        ((1, -1), 2, (1, 1), 1, 1, Q(1, 8)),
        ((1, 0, -1), 1, (1, 3), Q(1, 4), 2, Q(-1, 4)),
        ((1, -1, 1), 2, (1, 2), 2, Q(1, 2), Q(1, 8)),
    ),
)
def test_consensus_preparation_identifies_chi_from_native_initial_jets(
    mode, eigenvalue, raw_weights, beta, capacity, amplitude
):
    model = RelationalExchangeModel(
        beta, epi_weight=raw_weights[0], phase_weight=raw_weights[1]
    )
    field, _, weights, (m0, m1, m2) = _prepared_mode_jet(
        mode, model, capacity=capacity, amplitude=amplitude
    )
    assert sum(weights) == 0
    assert sum(weight * value for weight, value in zip(weights, mode, strict=True)) == 1
    assert field.phase_source == (0,) * len(mode)
    e, w = model.effective_weights
    mu = float(capacity * eigenvalue)
    assert m0 == amplitude
    assert float(m1) == pytest.approx(-e * mu * float(amplitude), abs=2e-15)
    assert float(m2) == pytest.approx(
        mu**2 * (e**2 - w**2 / (float(beta) * math.pi**2)) * float(amplitude),
        abs=2e-15,
    )
    recovered = float(m1**2 / (m1**2 - m0 * m2)) / math.pi**2
    assert recovered == pytest.approx(float(beta) * (e / w) ** 2, rel=2e-13, abs=2e-15)


def test_dual_mode_observation_removes_the_offset_on_an_irregular_graph():
    mode = (1, -1, 1)
    field, response, weights, jets = _prepared_mode_jet(
        mode, RelationalExchangeModel(1)
    )
    assert weights == (Q(1, 4), -Q(1, 2), Q(1, 4))
    ordinary = tuple(Q(value, 3) for value in mode)
    wrong = tuple(
        sum(weight * Q(value) for weight, value in zip(ordinary, values, strict=True))
        for values in (field.epi, field.form_rate, response.epi_acceleration)
    )
    assert sum(ordinary) != 0
    assert wrong[1:] == jets[1:]
    assert wrong[1] ** 2 - wrong[0] * wrong[2] < 0
    assert jets[1] ** 2 - jets[0] * jets[2] > 0


def test_form_only_preparation_excites_both_poles_including_the_critical_case():
    s = pytest.importorskip("sympy")
    a = s.Symbol("a", nonnegative=True)
    b, c = s.symbols("b c", positive=True)
    amplitude = s.Symbol("amplitude", nonzero=True)
    generator = s.Matrix(((-a, -b), (c, 0)))
    initial = s.Matrix((amplitude, 0))
    observe = s.Matrix(((1, 0),))
    assert initial.row_join(generator * initial).det() == c * amplitude**2
    assert observe.col_join(observe * generator).det() == -b
    critical = ((-2, -1), (1, 0))
    report = derive_linear_observation(critical, ((1, 0),))
    assert report.dimension == 2
    assert s.Matrix(critical).eigenvals() == {-1: 2}
    assert (s.Matrix(critical) + s.eye(2)) != s.zeros(2)
    assert (s.Matrix(critical) + s.eye(2)) ** 2 == s.zeros(2)


def test_omitted_initial_phase_changes_the_inferred_coefficient():
    generator = np.array(((-3, -2), (1, 0)))

    def jets(state):
        rate = generator @ state
        acceleration = generator @ rate
        return tuple(map(Q, (state[0], rate[0], acceleration[0])))

    first, hidden = jets((1, 0)), jets((1, 1))
    assert first == (1, -3, 7)
    assert hidden == (1, -5, 13)
    baseline = first[1] ** 2 / (first[1] ** 2 - first[0] * first[2])
    contaminated = hidden[1] ** 2 / (hidden[1] ** 2 - hidden[0] * hidden[2])
    assert baseline == Q(9, 2)
    assert contaminated == Q(25, 12) != baseline


def test_local_mixture_of_diffusion_modes_can_mimic_a_finite_jet_ratio():
    s = pytest.importorskip("sympy")
    t = s.Symbol("t", real=True)
    signal = 3 * s.exp(-t) - s.exp(-2 * t)
    jets = tuple(s.diff(signal, t, order).subs(t, 0) for order in range(3))
    assert jets == (2, -1, -1)
    assert jets[1] ** 2 - jets[0] * jets[2] == 3
    assert jets[1] ** 2 / (jets[1] ** 2 - jets[0] * jets[2]) == s.Rational(1, 3)


def test_initial_jet_ratio_is_not_invariant_under_a_non_affine_clock():
    m0, m1, m2 = Q(1), -Q(3), Q(7)
    clock_rate, clock_acceleration = Q(2), Q(1)
    rate = m1 / clock_rate
    acceleration = m2 / clock_rate**2 - m1 * clock_acceleration / clock_rate**3
    original = m1**2 / (m1**2 - m0 * m2)
    apparent = rate**2 / (rate**2 - m0 * acceleration)
    assert original == Q(9, 2)
    assert apparent == 18 != original


def test_three_sample_error_bounds_have_the_stated_exact_constants():
    s = pytest.importorskip("sympy")
    u, h = s.symbols("u h", positive=True)
    rate_left = (4 * (h - u) ** 2 - (2 * h - u) ** 2) / (4 * h)
    rate_right = -((2 * h - u) ** 2) / (4 * h)
    acceleration_left = ((2 * h - u) ** 2 - 2 * (h - u) ** 2) / (2 * h**2)
    acceleration_right = (2 * h - u) ** 2 / (2 * h**2)
    rate_bound = -s.integrate(rate_left, (u, 0, h)) - s.integrate(
        rate_right, (u, h, 2 * h)
    )
    acceleration_bound = s.integrate(acceleration_left, (u, 0, h)) + s.integrate(
        acceleration_right, (u, h, 2 * h)
    )
    assert s.simplify(rate_bound) == h**2 / 3
    assert s.simplify(acceleration_bound) == h


def test_sampling_error_budget_bounds_chi_or_abstains_when_noise_dominates():
    error, third_bound = Q(1, 2**30), Q(6)
    reports = []
    for step in (Q(1, 128), Q(1, 65536)):
        values = tuple(
            1 - 3 * t + Q(7, 2) * t**2 + t**3 + sign * error
            for t, sign in ((0, 1), (step, -1), (2 * step, 1))
        )
        rate = (-3 * values[0] + 4 * values[1] - values[2]) / (2 * step)
        acceleration = (values[2] - 2 * values[1] + values[0]) / step**2
        rate_error = 4 * error / step + third_bound * step**2 / 3
        acceleration_error = 4 * error / step**2 + third_bound * step
        assert abs(rate + 3) == rate_error
        assert abs(acceleration - 7) == acceleration_error
        reports.append(
            bound_relational_coefficient_from_jet(
                form_bounds=(values[0] - error, values[0] + error),
                rate_bounds=(rate - rate_error, rate + rate_error),
                acceleration_bounds=(
                    acceleration - acceleration_error,
                    acceleration + acceleration_error,
                ),
            )
        )
    lower, upper = reports[0].coefficient_bounds
    assert upper - lower < Q(1, 10)
    with mp.workdps(80):
        exact = 9 / (2 * mp.pi**2)
        assert mp.mpf(lower.numerator) / lower.denominator <= exact
        assert exact <= mp.mpf(upper.numerator) / upper.denominator
    assert reports[1].coefficient_bounds is None
    assert "unresolved_restoring_gap" in reports[1].unavailable_reasons
