"""Conditional cotangent closure, not an installed TNFR evolution law.

The additional premise is that pi_i=H_i(theta)*x_i is canonical momentum.
Its genuine coordinate pullback forces a form-pressure correction. Exact
identities, existing represented pressure and linear predictions have distinct
scopes; no trajectory or physical identification is asserted here.
"""

from itertools import combinations

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import (
    _metric_differential,
    _nonrepeated_phase_geometry,
)
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.forcing_realization import capture_non_epi_forcing


def _connection(s, metric, differential, form):
    return s.Matrix(
        len(form),
        len(form),
        lambda i, j: (form[j] * differential[j, i] - form[i] * differential[i, j])
        / (metric[i, i] * metric[j, j]),
    )


def _p2_field(s):
    """Full conditional P2 field before imposing mean or consensus constraints."""
    x0, x1, delta = s.symbols("x0 x1 delta", real=True)
    e = s.Symbol("e", nonnegative=True)
    weight = s.Symbol("w", positive=True)
    form = s.Matrix((x0, x1))
    h = s.pi * s.sin(delta) / delta
    metric = h * s.eye(2)
    # delta=theta1-theta0, including the derivative at the other endpoint.
    differential = s.Matrix(((-s.diff(h, delta), s.diff(h, delta)),) * 2)
    connection = _connection(s, metric, differential, form)
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    form_rate = -e * laplacian * form + weight * s.Matrix((delta, -delta)) / s.pi
    form_rate += connection * form
    phase_rate = metric.inv() * form
    field = form_rate.col_join(s.Matrix((phase_rate[1] - phase_rate[0],)))
    return (x0, x1, delta), e, weight, h, phase_rate, field


def test_canonical_coordinate_pullback_forces_connection_and_satisfies_jacobi():
    s = pytest.importorskip("sympy")
    theta = s.symbols("theta0:2", real=True)
    momentum = s.symbols("p0:2", real=True)
    form = s.Matrix(s.symbols("x0:2", real=True))
    diagonal = tuple(s.Function(f"h{i}")(*theta) for i in range(2))
    metric = s.diag(*diagonal)
    differential = s.Matrix(diagonal).jacobian(theta)
    # Independent construction: push the canonical tensor through x=p/H.
    canonical = s.zeros(2).row_join(s.eye(2)).col_join((-s.eye(2)).row_join(s.zeros(2)))
    coordinates = (*theta, *momentum)
    observation = s.Matrix([momentum[i] / diagonal[i] for i in range(2)] + list(theta))
    tangent = observation.jacobian(coordinates)
    pulled = (tangent * canonical * tangent.T).subs(
        {momentum[i]: diagonal[i] * form[i] for i in range(2)}
    )
    connection = _connection(s, metric, differential, form)
    expected = connection.row_join(-metric.inv()).col_join(
        metric.inv().row_join(s.zeros(2))
    )
    assert (pulled - expected).applyfunc(s.simplify) == s.zeros(4)
    assert (pulled + pulled.T).applyfunc(s.simplify) == s.zeros(4)
    # Check every independent Jacobi triple, with arbitrary smooth H_i.
    new_coordinates = (*form, *theta)
    for i, j, k in combinations(range(4), 3):
        jacobi = sum(
            pulled[i, ell] * s.diff(pulled[j, k], coordinate)
            + pulled[j, ell] * s.diff(pulled[k, i], coordinate)
            + pulled[k, ell] * s.diff(pulled[i, j], coordinate)
            for ell, coordinate in enumerate(new_coordinates)
        )
        assert s.simplify(jacobi) == 0


def test_existing_prism_metric_requires_nonzero_pressure_correction_and_full_balance():
    s, graph, rows, phases = _nonrepeated_phase_geometry()
    metric, differential, source, _ = _metric_differential(s, rows, phases)
    reflected_metric, reflected_differential, reflected_source, _ = (
        _metric_differential(s, rows, tuple(-phase for phase in phases))
    )
    # Independent evaluations check the local metric identities used by the
    # joint field's odd-state symmetry and small-amplitude response expansion.
    assert (reflected_metric - metric).applyfunc(s.simplify) == s.zeros(6)
    assert (reflected_differential + differential).applyfunc(s.simplify) == s.zeros(6)
    assert (reflected_source + source).applyfunc(s.simplify) == s.zeros(6, 1)
    form = s.eye(6)[:, 0]
    connection = _connection(s, metric, differential, form)
    correction = connection * form
    expected = (3 - 9 * s.sqrt(3) / s.pi) / (9 * (1 + s.sqrt(3)) ** 2)
    assert s.simplify(correction[1] - expected) == 0
    assert expected < 0
    assert (connection + connection.T).applyfunc(s.simplify) == s.zeros(6)
    assert s.simplify(form.dot(correction)) == 0
    assert (differential * s.ones(6, 1)).applyfunc(s.simplify) == s.zeros(6, 1)
    laplacian = s.eye(6) - s.Matrix(6, 6, lambda i, j: s.Rational(int(j in rows[i]), 3))
    e, weight = s.symbols("e w", positive=True)
    phase_rate = metric.inv() * form
    form_rate = -e * laplacian * form + weight * source + correction
    energy_rate = form.dot(form_rate) - weight * (metric * source).dot(phase_rate)
    assert s.simplify(energy_rate + e * form.dot(laplacian * form)) == 0
    momentum_rate = (
        s.ones(1, 6) * metric * form_rate + form.T * differential * phase_rate
    )
    assert s.simplify(momentum_rate[0] + e * sum(metric * laplacian * form)) == 0
    assert s.simplify(sum(form_rate) - weight * sum(source) - sum(correction)) == 0

    # The pressure owner supplies the OLD source; it has not acquired K*x.
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0, "topo": 0}
    for i, node in enumerate(graph):
        graph.nodes[node].update(EPI=float(form[i]), theta=float(phases[i]))
    captured = capture_non_epi_forcing(graph)
    old_pressure = -(laplacian * form) / 2 + source / 2
    assert tuple(map(float, captured.phase_gradient)) == pytest.approx(
        tuple(map(float, source)), abs=3e-15
    )
    assert tuple(map(float, captured.full_kernel_pressure)) == pytest.approx(
        tuple(map(float, old_pressure)), abs=3e-15
    )
    assert float(correction[1]) < -0.01


def test_same_phase_different_form_has_distinct_phase_rate_and_acceleration():
    s = pytest.importorskip("sympy")
    coordinates, e, _, _, phase_velocity, field = _p2_field(s)
    x0, x1, delta = coordinates
    # Differentiate the full field before taking the removable consensus
    # limit. In particular the phase acceleration includes DH and K*x.
    acceleration = phase_velocity.jacobian(coordinates) * field
    consensus_velocity = phase_velocity.applyfunc(
        lambda value: s.limit(value, delta, 0)
    )
    consensus_acceleration = acceleration.applyfunc(
        lambda value: s.limit(s.simplify(value), delta, 0)
    )
    preparations = ({x0: 0, x1: 0}, {x0: s.Rational(1, 4), x1: -s.Rational(1, 4)})
    phase_rates = tuple(consensus_velocity.subs(point) for point in preparations)
    phase_accelerations = tuple(
        consensus_acceleration.subs(point) for point in preparations
    )
    assert phase_rates[1] - phase_rates[0] == s.Matrix((1, -1)) / (4 * s.pi)
    assert phase_accelerations[1] - phase_accelerations[0] == -e * s.Matrix((1, -1)) / (
        2 * s.pi
    )
    # An x-independent phase row cannot have the first response. This does
    # not assert that the cotangent premise is selected by the nodal identity.


def test_consensus_linearization_predicts_geometry_dependent_damped_modes():
    s = pytest.importorskip("sympy")
    rate, e, weight, degree, eigenvalue = s.symbols("s e w d ell", positive=True)
    mode = s.Matrix(
        ((-e * eigenvalue, -weight * eigenvalue / s.pi), (1 / (s.pi * degree), 0))
    )
    expected = (
        rate**2 + e * eigenvalue * rate + weight * eigenvalue / (s.pi**2 * degree)
    )
    assert s.simplify((rate * s.eye(2) - mode).det() - expected) == 0
    # Same e=w=1/2, degree two, with exact eigenvectors of the actual cycle
    # support. Geometry changes the least positive eigenvalue, not the gains.
    values = {}
    for size in (6, 8):
        support = s.Matrix(
            size,
            size,
            lambda i, j: int((j - i) % size in (1, size - 1)),
        )
        laplacian = s.eye(size) - support / 2
        vector = s.Matrix([s.cos(2 * s.pi * i / size) for i in range(size)])
        ell = s.simplify(1 - s.cos(2 * s.pi / size))
        assert (laplacian * vector - ell * vector).applyfunc(s.simplify) == s.zeros(
            size, 1
        )
        values[size] = ell
    assert values[6] == s.Rational(1, 2)
    assert values[8] == 1 - s.sqrt(2) / 2
    assert 9 < s.pi**2 < 10
    assert values[8] < s.Rational(2, 5) < 4 / s.pi**2
    assert 4 / s.pi**2 < s.Rational(4, 9) < values[6]
    # Thus C8's first mode is underdamped and C6's is not in THIS closure.


def test_p2_zero_momentum_reduction_has_exact_energy_and_admission_barriers():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, h, _, full_field = _p2_field(s)
    x0, x1, delta = coordinates
    a = s.Symbol("a", real=True)
    restricted = full_field.subs({x0: a, x1: -a}).applyfunc(s.simplify)
    assert s.simplify(restricted[0] + restricted[1]) == 0
    form_rate, phase_gap_rate = restricted[0], restricted[2]
    assert form_rate == -2 * e * a + weight * delta / s.pi
    assert s.simplify(phase_gap_rate + 2 * a / h) == 0
    energy = a**2 + weight * (1 - s.cos(delta))
    energy_rate = s.diff(energy, a) * form_rate + s.diff(energy, delta) * phase_gap_rate
    assert s.simplify(energy_rate) == -4 * e * a**2
    assert s.limit(h, delta, 0) == s.pi
    assert energy.subs({a: 0, delta: s.pi}) == 2 * weight
    assert energy.subs({a: 0, delta: s.pi / 2}) == weight
    # On a=0, invariance of E'=0 for e>0 forces delta=0. Sublevels E<2w
    # are compact inside the regular chart; E<w also stays in strict U3.
    assert form_rate.subs(a, 0) == weight * delta / s.pi
    jacobian = (
        s.Matrix((form_rate, phase_gap_rate))
        .jacobian((a, delta))
        .applyfunc(lambda value: s.limit(s.simplify(value), delta, 0).subs(a, 0))
    )
    assert s.simplify(jacobian.det()) == 2 * weight / s.pi**2
    discriminant = s.simplify(s.trace(jacobian) ** 2 - 4 * jacobian.det())
    assert discriminant == 4 * e**2 - 8 * weight / s.pi**2
    # The retained half/half coefficients do not ring on P2.
    assert discriminant.subs({e: s.Rational(1, 2), weight: s.Rational(1, 2)}) > 0


def test_p2_nonzero_momentum_reduction_inherits_full_rows_and_restoring_stiffness():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, h, phase_velocity, field = _p2_field(s)
    x0, x1, delta = coordinates
    a, momentum = s.symbols("a P", real=True)
    invariant = h * (x0 + x1)
    # Differentiate the full pullback field before reducing. Diffusion also
    # cancels here because BOTH P2 metric entries are the same h(delta).
    invariant_rate = s.Matrix([invariant]).jacobian(coordinates) * field
    assert s.simplify(invariant_rate[0]) == 0
    substitution = {x0: momentum / (2 * h) + a, x1: momentum / (2 * h) - a}
    reduced_a = s.simplify(((field[0] - field[1]) / 2).subs(substitution))
    reduced_delta = s.simplify(field[2].subs(substitution))
    common_phase_rate = s.simplify((sum(phase_velocity) / 2).subs(substitution))
    original_storage = (x0**2 + x1**2) / 2 + weight * (1 - s.cos(delta))
    reduced_storage = s.simplify(original_storage.subs(substitution))
    potential = momentum**2 / (4 * h**2) + weight * (1 - s.cos(delta))
    assert s.simplify(reduced_storage - a**2 - potential) == 0
    assert s.simplify(reduced_a + 2 * e * a - s.diff(potential, delta) / h) == 0
    assert s.simplify(reduced_delta + 2 * a / h) == 0
    assert s.simplify(common_phase_rate - momentum / (2 * h**2)) == 0
    storage_rate = (
        s.diff(reduced_storage, a) * reduced_a
        + s.diff(reduced_storage, delta) * reduced_delta
    )
    assert s.simplify(storage_rate + 4 * e * a**2) == 0

    # Derive the Hessian and polynomial from the FULL reduced vector field,
    # including the geometric-pressure term absent from the old pressure.
    curvature = s.limit(s.diff(potential, delta, 2), delta, 0)
    assert s.simplify(curvature - weight - momentum**2 / (6 * s.pi**2)) == 0
    jacobian = (
        s.Matrix((reduced_a, reduced_delta))
        .jacobian((a, delta))
        .applyfunc(lambda value: s.limit(s.simplify(value), delta, 0).subs(a, 0))
    )
    rate = s.Symbol("lambda")
    polynomial = s.expand((rate * s.eye(2) - jacobian).det())
    expected = (
        rate**2 + 2 * e * rate + 2 * weight / s.pi**2 + momentum**2 / (3 * s.pi**4)
    )
    assert s.simplify(polynomial - expected) == 0
    assert s.limit(common_phase_rate, delta, 0) == momentum / (2 * s.pi**2)
    # A nonzero P puts an infinite barrier at the phase-chart boundary;
    # strict acuteness instead uses the finite pi/2 potential threshold.
    positive_momentum = s.Symbol("P_positive", positive=True)
    assert (
        s.limit(potential.subs(momentum, positive_momentum), delta, s.pi, dir="-")
        == s.oo
    )
    assert s.simplify(potential.subs(delta, s.pi / 2)) == momentum**2 / 16 + weight


def test_p2_nonzero_momentum_leaks_without_connection_despite_same_energy_balance():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, h, _, full_field = _p2_field(s)
    x0, x1, delta = coordinates
    # This explicit comparison keeps the SAME reciprocal phase row and removes
    # only K*x. Its energy decay cannot distinguish it from the Poisson law.
    naive = s.Matrix(
        (
            -e * (x0 - x1) + weight * delta / s.pi,
            e * (x0 - x1) - weight * delta / s.pi,
            full_field[2],
        )
    )
    storage = (x0**2 + x1**2) / 2 + weight * (1 - s.cos(delta))
    storage_rate = (s.Matrix([storage]).jacobian(coordinates) * naive)[0]
    assert s.simplify(storage_rate + e * (x0 - x1) ** 2) == 0
    invariant = h * (x0 + x1)
    naive_momentum_rate = (s.Matrix([invariant]).jacobian(coordinates) * naive)[0]
    assert s.simplify(naive_momentum_rate - s.diff(h, delta) * (x1**2 - x0**2) / h) == 0
    # A regular nonzero-momentum state exposes the leak. Neither the prior
    # zero-P P2 sector nor the antiperiodic C8 preparation can supply this test.
    preparation = {x0: s.Rational(1, 4), x1: 0, delta: s.pi / 3}
    leak = s.simplify(naive_momentum_rate.subs(preparation))
    assert s.simplify(leak - (3 / s.pi - 1 / s.sqrt(3)) / 16) == 0
    assert s.pi < 3 * s.sqrt(3)
    assert leak > 0
    corrected_rate = (s.Matrix([invariant]).jacobian(coordinates) * full_field)[0]
    assert s.simplify(corrected_rate.subs(preparation)) == 0


def test_cotangent_scale_family_preserves_pressure_and_balance_but_not_response():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, h, _, _ = _p2_field(s)
    x0, x1, delta = coordinates
    form = s.Matrix((x0, x1))
    scale = s.Symbol("c", positive=True)
    theta = s.symbols("theta0:2", real=True)
    momentum = s.symbols("p0:2", real=True)
    lifted_h = h.subs(delta, theta[1] - theta[0])
    observation = s.Matrix(
        [momentum[i] / (scale * lifted_h) for i in range(2)] + list(theta)
    )
    canonical = s.zeros(2).row_join(s.eye(2)).col_join((-s.eye(2)).row_join(s.zeros(2)))
    tangent = observation.jacobian((*theta, *momentum))
    pulled = (tangent * canonical * tangent.T).subs(
        {momentum[i]: scale * lifted_h * form[i] for i in range(2)}
    )
    pulled = pulled.subs({theta[0]: 0, theta[1]: delta})
    metric = h * s.eye(2)
    differential = s.Matrix(((-s.diff(h, delta), s.diff(h, delta)),) * 2)
    connection = _connection(s, metric, differential, form)
    expected = (
        connection.row_join(-metric.inv()).col_join(metric.inv().row_join(s.zeros(2)))
        / scale
    )
    assert (pulled - expected).applyfunc(s.simplify) == s.zeros(4)

    # Scaling BOTH momentum geometry and the phase storage preserves the
    # original phase-pressure source. It does not select the value c=1.
    gradient = s.Matrix(
        (x0, x1, -scale * weight * s.sin(delta), scale * weight * s.sin(delta))
    )
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    field = pulled * gradient + (-e * laplacian * form).col_join(s.zeros(2, 1))
    expected_form = -e * laplacian * form + weight * s.Matrix((delta, -delta)) / s.pi
    expected_form += connection * form / scale
    assert (field[:2, :] - expected_form).applyfunc(s.simplify) == s.zeros(2, 1)
    assert (field[2:, :] - metric.inv() * form / scale).applyfunc(
        s.simplify
    ) == s.zeros(2, 1)
    assert s.simplify(gradient.dot(field) + e * (x0 - x1) ** 2) == 0

    # An exact finite-amplitude discriminator needs no trajectory, tuning or
    # linearization: start in the invariant zero-momentum P2 sector.
    amplitude = s.Symbol("a", real=True)
    point = {x0: amplitude, x1: -amplitude}
    amplitude_rate = s.simplify(field[0].subs(point))
    gap_rate = s.simplify((field[3] - field[2]).subs(point))
    assert amplitude_rate == -2 * e * amplitude + weight * delta / s.pi
    assert s.simplify(gap_rate + 2 * amplitude / (scale * h)) == 0
    acceleration = s.diff(amplitude_rate, amplitude) * amplitude_rate
    acceleration += s.diff(amplitude_rate, delta) * gap_rate
    at_consensus = s.limit(acceleration, delta, 0)
    assert (
        s.simplify(
            at_consensus
            - 4 * e**2 * amplitude
            + 2 * weight * amplitude / (scale * s.pi**2)
        )
        == 0
    )
    assert s.simplify(s.diff(at_consensus, scale)) == 2 * weight * amplitude / (
        scale**2 * s.pi**2
    )


def _consensus_path_pressure(form, capacity):
    """Observe the actual pressure owner, with no trajectory or phase forcing."""
    graph = nx.path_graph(len(form))
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0, "topo": 0}
    for node, value, rate in zip(graph, form, capacity, strict=True):
        attributes = graph.nodes[node]
        set_attr(attributes, ALIAS_EPI, value)
        set_attr(attributes, ALIAS_VF, rate)
        set_attr(attributes, ALIAS_THETA, 0.0)
    default_compute_delta_nfr(graph)
    return tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph)


def test_irregular_support_does_not_inherit_unweighted_cotangent_storage_decay():
    s = pytest.importorskip("sympy")
    form = s.Matrix((1, s.Rational(3, 2), 1))
    laplacian = s.Matrix(
        ((1, -1, 0), (-s.Rational(1, 2), 1, -s.Rational(1, 2)), (0, -1, 1))
    )
    pressure = s.Matrix(_consensus_path_pressure((1.0, 1.5, 1.0), (1.0,) * 3))
    assert pressure == s.Matrix((s.Float(0.25), s.Float(-0.25), s.Float(0.25)))
    assert form.dot(laplacian * form) == -s.Rational(1, 4)
    assert form.dot(pressure) == s.Float(0.125)
    # The reversible degree-weighted diffusion storage is a different
    # Hamiltonian choice; it cannot silently replace the unweighted storage.
    degree = s.diag(1, 2, 1)
    assert form.dot(degree * laplacian * form) == s.Rational(1, 2)
    assert form.dot(degree * pressure) == s.Float(-0.25)
    assert degree * form != form


def test_heterogeneous_capacity_does_not_inherit_unit_capacity_storage_decay():
    s = pytest.importorskip("sympy")
    form = s.Matrix((2, 1))
    capacity = s.diag(1, 4)
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    pressure = s.Matrix(_consensus_path_pressure((2.0, 1.0), (1.0, 4.0)))
    assert pressure == s.Matrix((s.Float(-0.5), s.Float(0.5)))
    assert form.dot(capacity * laplacian * form) == -2
    assert form.dot(capacity * pressure) == s.Float(1)
    # This is the nodal product at an admitted snapshot, not a violation of a
    # pressure-owner contract or evidence against its diffusion theorem.


def test_underdamped_predictor_solves_both_rows_and_reverses_form_before_a_full_period():
    s = pytest.importorskip("sympy")
    time = s.Symbol("t", real=True)
    alpha, omega, amplitude, degree, eigenvalue = s.symbols(
        "alpha omega epsilon d ell", positive=True
    )
    # This is an algebraic parameterization of the underdamped polynomial,
    # not a choice of production coefficients or a fit to a trajectory.
    e = 2 * alpha / eigenvalue
    weight = s.pi**2 * degree * (omega**2 + alpha**2) / eigenvalue
    assert e * eigenvalue / 2 == alpha
    assert s.simplify(weight * eigenvalue / (s.pi**2 * degree) - alpha**2) == omega**2
    form = (
        amplitude
        * s.exp(-alpha * time)
        * (s.cos(omega * time) - alpha / omega * s.sin(omega * time))
    )
    phase = (
        amplitude * s.exp(-alpha * time) * s.sin(omega * time) / (s.pi * degree * omega)
    )
    assert (
        s.simplify(
            s.diff(form, time)
            + e * eigenvalue * form
            + weight * eigenvalue * phase / s.pi
        )
        == 0
    )
    assert s.simplify(s.diff(phase, time) - form / (s.pi * degree)) == 0
    assert form.subs(time, 0) == amplitude
    assert phase.subs(time, 0) == 0
    observation_time = s.pi / (2 * omega)
    assert (
        s.simplify(form.subs(time, observation_time))
        == -amplitude * alpha * s.exp(-alpha * observation_time) / omega
    )
    assert form.subs(time, observation_time).is_negative
    old_form = amplitude * s.exp(-e * eigenvalue * time)
    assert old_form.subs(time, observation_time).is_positive


def test_equal_replica_blowup_inherits_exchange_scale_instead_of_unit_scale():
    s = pytest.importorskip("sympy")
    # A nonuniform phase preparation on C3 exercises both the resultant
    # magnitude derivative and the geometric correction. Two replicas per
    # vertex, with every base edge replaced by K2,2, give a regular K2,2,2.
    rows = ((1, 2), (0, 2), (0, 1))
    phases = (-s.pi / 6, 0, s.pi / 6)
    form = s.Matrix((1, 2, -1))
    replicas, size = 2, len(rows)
    fine_size = replicas * size
    lift = s.Matrix(fine_size, size, lambda i, j: int(i // replicas == j))
    fine_rows = tuple(
        tuple(replicas * j + b for j in rows[i // replicas] for b in range(replicas))
        for i in range(fine_size)
    )
    fine_phases = tuple(phases[i // replicas] for i in range(fine_size))
    fine_form = lift * form
    metric, differential, source, _ = _metric_differential(s, rows, phases)
    fine_metric, fine_differential, fine_source, _ = _metric_differential(
        s, fine_rows, fine_phases
    )
    assert (fine_metric * lift - replicas * lift * metric).applyfunc(
        s.simplify
    ) == s.zeros(fine_size, size)
    assert (fine_differential * lift - replicas * lift * differential).applyfunc(
        s.simplify
    ) == s.zeros(fine_size, size)
    assert (fine_source - lift * source).applyfunc(s.simplify) == s.zeros(fine_size, 1)
    connection = _connection(s, metric, differential, form)
    # Construct the full fine tensor before restricting it. Differentiating
    # an already restricted metric would lose individual replica derivatives.
    fine_connection = _connection(s, fine_metric, fine_differential, fine_form)
    assert (connection * form).applyfunc(s.simplify) != s.zeros(size, 1)
    assert (fine_connection * lift - lift * connection / replicas).applyfunc(
        s.simplify
    ) == s.zeros(fine_size, size)
    laplacian = s.eye(size) - s.Matrix(
        size, size, lambda i, j: s.Rational(int(j in rows[i]), len(rows[i]))
    )
    fine_laplacian = s.eye(fine_size) - s.Matrix(
        fine_size,
        fine_size,
        lambda i, j: s.Rational(int(j in fine_rows[i]), len(fine_rows[i])),
    )
    assert fine_laplacian * lift == lift * laplacian
    e, weight = s.symbols("e w", positive=True)
    expected_form_rate = -e * laplacian * form + weight * source
    expected_form_rate += connection * form / replicas
    fine_form_rate = -e * fine_laplacian * fine_form + weight * fine_source
    fine_form_rate += fine_connection * fine_form
    assert (fine_form_rate - lift * expected_form_rate).applyfunc(
        s.simplify
    ) == s.zeros(fine_size, 1)
    assert (
        fine_metric.inv() * fine_form - lift * metric.inv() * form / replicas
    ).applyfunc(s.simplify) == s.zeros(fine_size, 1)
    potential = sum(
        1 - s.cos(phases[j] - phases[i])
        for i, neighbors in enumerate(rows)
        for j in neighbors
        if i < j
    )
    fine_potential = sum(
        1 - s.cos(fine_phases[j] - fine_phases[i])
        for i, neighbors in enumerate(fine_rows)
        for j in neighbors
        if i < j
    )
    fine_energy = fine_form.dot(fine_form) / 2 + weight * fine_potential
    expected_energy = form.dot(form) / 2 + replicas * weight * potential
    assert s.simplify(fine_energy / replicas - expected_energy) == 0
    # Pull back and normalize the canonical one-form by the same replica
    # factor as storage. The effective momentum is m*H*x, not H*x.
    assert (
        lift.T * fine_metric * fine_form / replicas - replicas * metric * form
    ).applyfunc(s.simplify) == s.zeros(size, 1)
