"""Exact full-law exchange, memory and phase-only closure discriminators."""

from fractions import Fraction as Q
from functools import partial

import networkx as nx
import numpy as np
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import cos as jet_cos
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_forecast import _sine_flow

MODEL = RelationalExchangeModel(
    2, epi_weight=Q(3, 4), phase_weight=Q(1, 4), phase_domain="regular"
)


def _source(*, flat=False):
    graph = nx.Graph(((0, 1), (1, 2), (2, 3), (0, 2)))
    graph.graph["GAMMA"] = {"type": "none"}
    for i, (x, theta, nu) in enumerate(
        zip(
            (Q(1, 2), Q(-3, 4), Q(5, 4), Q(-1)),
            (Q(1, 8), Q(-1, 4), Q(1, 2), Q(-3, 8)),
            (Q(1), Q(3, 2), Q(2), Q(5, 4)),
        )
    ):
        graph.nodes[i].update(EPI=x, theta=0 if flat else theta, nu_f=nu)
    return graph, bound_relational_sine_exchange(graph, reference_model=MODEL)


def _jets(graph, source):
    return flow_jets(
        tuple(I(v) for v in source.epi + source.phase + (source.capacity[-1],)),
        2,
        partial(
            _sine_flow,
            neighbors=tuple(tuple(graph[i]) for i in source.nodes),
            visible_capacity=source.capacity[:-1],
            model=MODEL,
        ),
    )


def _zero(bound):
    assert bound.lo <= 0 <= bound.hi
    assert bound.abs_max < Q(1, 10**25)


def test_joint_coordinate_and_second_order_rows_match_shared_complete_flow():
    graph, source = _source()
    jets = _jets(graph, source)
    size = len(source.nodes)
    e, w = map(Q, MODEL.effective_weights)
    a, b = w / pi_interval(), (w / Q(MODEL.storage_scale)) / pi_interval()
    k = tuple(nu / d for nu, d in zip(source.capacity, source.degrees))
    currents = tuple(
        sum((sin(I(source.phase[j] - source.phase[i])) for j in graph[i]), I(0))
        for i in source.nodes
    )
    for i in source.nodes:
        xdot, theta_dot = jets[i][1], jets[size + i][1]
        # The sum cancels loss exactly in scaled time, without dropping form.
        _zero((theta_dot + (b / e) * xdot) / e - (a * b / e**2) * k[i] * currents[i])
        acceleration = 2 * jets[size + i][2]
        damping = (
            e * k[i] * sum((theta_dot - jets[size + j][1] for j in graph[i]), I(0))
        )
        drive = (
            a
            * b
            * k[i]
            * sum((k[i] * currents[i] - k[j] * currents[j] for j in graph[i]), I(0))
        )
        _zero(acceleration + damping - drive)


def test_phase_potential_initially_rises_while_full_storage_falls():
    graph, source = _source(flat=True)
    jets = tuple(Jet(row) for row in _jets(graph, source))
    n = len(source.nodes)
    potential = sum(
        (1 - jet_cos(jets[n + j] - jets[n + i]) for i, j in graph.edges),
        Jet.constant(0, 2),
    )
    form = sum(
        ((jets[j] - jets[i]) ** 2 / 2 for i, j in graph.edges), Jet.constant(0, 2)
    )
    energy = form + Q(MODEL.storage_scale) * potential
    velocity_numerators = source.phase_rate_numerators()
    # V'' = theta_dot^T L theta_dot at flat phase; shared rate numerators
    # retain exact rational coefficients before the mathematical-pi divisor.
    expected = (
        sum(
            (
                (velocity_numerators[j] - velocity_numerators[i]) ** 2
                for i, j in graph.edges
            ),
            Q(0),
        )
        / pi_interval() ** 2
    )
    _zero(potential.coeffs[1])
    _zero(2 * potential.coeffs[2] - expected)
    assert potential.coeffs[2].lo > 0
    assert energy.coeffs[1].hi < 0
    _zero(energy.coeffs[1] + source.continuous_loss)


def test_identical_joint_angle_alone_does_not_determine_its_rate():
    e, w = map(Q, MODEL.effective_weights)
    beta = Q(MODEL.storage_scale)
    alpha = (w / (beta * e)) / pi_interval()
    inverse_alpha = (beta * e / w) * pi_interval()
    u = Q(1, 8)
    # Ideal interval preparation: theta=(u,-u), z=(-u,u), hence y=0.
    # Keeping the nonperiodic difference z is essential to this chart.
    states = (
        (I(0),) * 4 + (I(1),),
        (-u * inverse_alpha, u * inverse_alpha, I(u), I(-u), I(1)),
    )
    rates = [
        _sine_flow(state, neighbors=((1,), (0,)), visible_capacity=(Q(1),), model=MODEL)
        for state in states
    ]
    for i in range(2):
        _zero(states[1][i + 2] + alpha * states[1][i])
        _zero(rates[0][i + 2] + alpha * rates[0][i])
    assert (rates[1][2] + alpha * rates[1][0]).hi < 0
    assert (rates[1][3] + alpha * rates[1][1]).lo > 0


def test_quotient_velocity_storage_and_loss_equal_original_full_storage():
    graph, source = _source()
    jets = _jets(graph, source)
    size = len(source.nodes)
    laplacian = nx.laplacian_matrix(graph, nodelist=source.nodes).toarray()
    k = np.array([float(nu / d) for nu, d in zip(source.capacity, source.degrees)])
    roots = np.diag(np.sqrt(k))
    symmetric = roots @ laplacian @ roots
    inverse_on_quotient = np.linalg.pinv(symmetric, hermitian=True)
    null_vector = 1 / np.sqrt(k)
    projector = np.eye(size) - np.outer(null_vector, null_vector) / (
        null_vector @ null_vector
    )
    np.testing.assert_allclose(symmetric @ inverse_on_quotient, projector, atol=2e-15)
    velocity = np.array(
        [float(jets[size + i][1].midpoint) for i in range(size)]
    ) / np.sqrt(k)
    acceleration = np.array(
        [float((2 * jets[size + i][2]).midpoint) for i in range(size)]
    ) / np.sqrt(k)
    np.testing.assert_allclose(null_vector @ velocity, 0, atol=2e-15)
    form, phase = np.array(source.epi, dtype=float), np.array(source.phase, dtype=float)
    potential = sum(1 - np.cos(phase[j] - phase[i]) for i, j in graph.edges)
    currents = np.array(
        [sum(np.sin(phase[j] - phase[i]) for j in graph[i]) for i in source.nodes]
    )
    e, w = MODEL.effective_weights
    a, b = w / np.pi, w / (MODEL.storage_scale * np.pi)
    original = form @ laplacian @ form / 2 + MODEL.storage_scale * potential
    transformed = velocity @ inverse_on_quotient @ velocity / 2 + a * b * potential
    derivative = (
        velocity @ inverse_on_quotient @ acceleration
        - a * b * (roots @ currents) @ velocity
    )
    np.testing.assert_allclose(transformed, b**2 * original, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(derivative, -e * (velocity @ velocity), atol=2e-15)
    np.testing.assert_allclose(
        derivative, -(b**2) * float(source.continuous_loss), atol=2e-15
    )


def test_memory_operator_order_symmetry_and_static_gain_on_heterogeneous_support():
    graph, source = _source()
    laplacian = (
        nx.laplacian_matrix(graph, nodelist=source.nodes).toarray().astype(float)
    )
    k = np.diag(
        [float(nu / degree) for nu, degree in zip(source.capacity, source.degrees)]
    )
    roots = np.sqrt(k)
    symmetric = roots @ laplacian @ roots
    generator = k @ laplacian
    e = MODEL.effective_weights[0]
    assert not np.allclose(k @ laplacian, laplacian @ k)
    response = generator @ expm(-e * generator / 3) @ k
    independent = roots @ symmetric @ expm(-e * symmetric / 3) @ roots
    np.testing.assert_allclose(response, independent, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(response, response.T, atol=2e-15)
    assert np.linalg.eigvalsh(response).min() > -1e-14
    assert np.min(response) < 0  # PSD does not mean entrywise nonnegative.
    metric = 1 / k.diagonal()
    horizon = 40.0
    integral, _error = quad_vec(
        lambda t: generator @ expm(-e * generator * t) @ k,
        0,
        horizon,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    tail = (expm(-e * generator * horizon) @ k - np.ones_like(k) / metric.sum()) / e
    expected_gain = (k - np.ones_like(k) / metric.sum()) / e
    np.testing.assert_allclose(integral + tail, expected_gain, atol=2e-13)


def test_exact_nonlinear_memory_retains_initial_form_and_moving_phase_history():
    graph, source = _source()
    adjacency = nx.to_numpy_array(graph, nodelist=source.nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    k = np.diag([float(nu / d) for nu, d in zip(source.capacity, source.degrees)])
    generator = k @ laplacian
    e, w = MODEL.effective_weights
    a, b = w / np.pi, w / (MODEL.storage_scale * np.pi)
    x0, theta0 = np.array(source.epi, dtype=float), np.array(source.phase, dtype=float)

    def current(theta):
        return (adjacency * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)

    def field(_time, state):
        x, theta = state[:4], state[4:]
        return np.concatenate(
            (-e * generator @ x + a * k @ current(theta), b * generator @ x)
        )

    end = 0.75
    response = solve_ivp(
        field,
        (0, end),
        np.concatenate((x0, theta0)),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
    )
    assert response.success
    identity = np.eye(4)
    initial_term = b / e * (identity - expm(-e * generator * end)) @ x0
    history, _error = quad_vec(
        lambda s: (identity - expm(-e * generator * (end - s)))
        @ k
        @ current(response.sol(s)[4:]),
        0,
        end,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    # Reconstructing memory along an independently computed full trajectory
    # checks equivalence; it is not a prospective prediction of that response.
    reconstructed = theta0 + initial_term + a * b / e * history
    np.testing.assert_allclose(
        reconstructed, response.y[4:, -1], atol=2e-12, rtol=2e-12
    )
    assert np.linalg.norm(initial_term) > 1e-2
    assert np.linalg.norm(a * b / e * history) > 1e-5
