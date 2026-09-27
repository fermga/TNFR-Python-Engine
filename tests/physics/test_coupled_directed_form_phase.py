"""Derived regional form angles on two explicitly supplied directed triangles.

The active response is outgoing EPI diffusion, with fixed positive capacities. The
regional complex coordinates encode real scalar form; they are not written
into primitive nodal phase. No new law, clock, event or engine path is added.
"""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy.linalg import expm

from tests.physics._internal_mode_fixture import NODES, Q_MODE, P, _apply
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.directed_diffusion import (
    directed_cayley_adjacency,
    directed_rw_laplacian,
)
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.structural_diffusion import structural_diffusion_operator


@pytest.fixture(scope="module")
def model():
    s = pytest.importorskip("sympy")
    shift_array = directed_cayley_adjacency(3, {1})
    shift = s.Matrix(shift_array.astype(int))
    adjacency = s.BlockMatrix([[shift, s.eye(3)], [s.eye(3), shift]]).as_explicit()
    generator = adjacency / 2 - s.eye(6)
    basis = s.Matrix.hstack(s.Matrix(P) / s.sqrt(2), s.Matrix(Q_MODE) / s.sqrt(6))
    lift = s.diag(basis, basis)
    reduced = s.simplify(lift.T * generator * lift)
    return s, adjacency, generator, basis, lift, reduced


def _gram_observables(s, state):
    z0, z1 = state[:2, 0], state[2:, 0]
    return s.Matrix(
        [z0.dot(z0), z1.dot(z1), z0.dot(z1), s.det(s.Matrix.hstack(z0, z1))]
    )


def test_declared_graph_and_both_production_readers_bind_the_exact_reduction(model):
    s, adjacency, generator, basis, lift, reduced = model
    graph = nx.DiGraph()
    graph.add_nodes_from(NODES)
    for i, node in enumerate(NODES):
        graph.nodes[node].update(EPI=0.5, theta=0.0, nu_f=1.0)
        for j, other in enumerate(NODES):
            if adjacency[i, j]:
                graph.add_edge(node, other, weight=1.0)
    assert graph.number_of_edges() == 12
    assert tuple(dict(graph.out_degree()).values()) == (2,) * 6
    nodes, laplacian = structural_diffusion_operator(graph)
    assert tuple(nodes) == NODES
    outgoing = directed_rw_laplacian(np.asarray(adjacency, dtype=float))
    np.testing.assert_array_equal(laplacian, outgoing)
    # All matrix coefficients are dyadic here; only the modal basis has surds.
    assert -s.Matrix(laplacian).applyfunc(s.Rational) == generator
    assert basis.T * basis == s.eye(2)
    assert s.simplify(generator * lift - lift * reduced) == s.zeros(6, 4)
    assert s.simplify(lift.T * generator - reduced * lift.T) == s.zeros(4, 6)
    means = s.diag(s.ones(3, 1), s.ones(3, 1))
    mean_generator = s.Matrix([[-1, 1], [1, -1]]) / 2
    assert generator * means == means * mean_generator
    assert s.simplify(lift.T * means) == s.zeros(4, 2)
    assert means.row_join(lift).rank() == 6


def test_joint_cartesian_law_and_complete_loss_follow_from_support(model):
    s, _, _, _, _, reduced = model
    nu = s.symbols("nu", positive=True)
    u0, v0, u1, v1 = s.symbols("u0 v0 u1 v1", real=True)
    state = s.Matrix([u0, v0, u1, v1])
    rate = nu * reduced * state
    diagonal = s.Matrix([[-5, s.sqrt(3)], [-s.sqrt(3), -5]]) / 4
    assert (
        reduced
        == s.BlockMatrix(
            [[diagonal, s.eye(2) / 2], [s.eye(2) / 2, diagonal]]
        ).as_explicit()
    )
    total = state.dot(state)
    gap = (u0 - u1) ** 2 + (v0 - v1) ** 2
    assert s.expand(2 * state.dot(rate) + 3 * nu * total / 2 + nu * gap) == 0
    # A rotating, coupled form still loses contrast. No amplitude maintenance
    # is inferred from the skew part, phase locking, or an observable period.


def test_polar_rate_is_derived_with_an_amplitude_ratio_not_a_fitted_phase_gain(model):
    s, _, _, _, _, reduced = model
    nu, r0, r1 = s.symbols("nu r0 r1", positive=True)
    phi0, phi1 = s.symbols("phi0 phi1", real=True)
    state = s.Matrix(
        [r0 * s.cos(phi0), r0 * s.sin(phi0), r1 * s.cos(phi1), r1 * s.sin(phi1)]
    )
    rate = nu * reduced * state
    expected_radial = (
        -5 * nu * r0 / 4 + nu * r1 * s.cos(phi1 - phi0) / 2,
        -5 * nu * r1 / 4 + nu * r0 * s.cos(phi1 - phi0) / 2,
    )
    expected_angular = (
        -s.sqrt(3) * nu / 4 + nu * r1 * s.sin(phi1 - phi0) / (2 * r0),
        -s.sqrt(3) * nu / 4 + nu * r0 * s.sin(phi0 - phi1) / (2 * r1),
    )
    for a, radius in enumerate((r0, r1)):
        offset = 2 * a
        form = state[offset : offset + 2, 0]
        velocity = rate[offset : offset + 2, 0]
        radial = form.dot(velocity) / radius
        angular = s.det(s.Matrix.hstack(form, velocity)) / radius**2
        assert s.trigsimp(radial - expected_radial[a]) == 0
        assert s.trigsimp(angular - expected_angular[a]) == 0
    delta_rate = expected_angular[1] - expected_angular[0]
    assert (
        s.trigsimp(delta_rate + nu * (r0 / r1 + r1 / r0) * s.sin(phi1 - phi0) / 2) == 0
    )
    # Equal amplitudes are an invariant special family, not a general closure.
    assert s.simplify((expected_radial[0] - expected_radial[1]).subs(r1, r0)) == 0
    assert s.trigsimp(delta_rate.subs(r1, r0) + nu * s.sin(phi1 - phi0)) == 0


def test_equal_observed_angles_can_have_different_future_phase_rates(model):
    s, _, _, _, _, reduced = model
    # These rows have equal regional phases (0, pi/2) and equal regional means.
    # Arbitrarily small common scaling places the fine EPI in its allowed band.
    first = s.Matrix([1, 0, 0, 1]) / 16
    second = s.Matrix([1, 0, 0, 2]) / 16

    def angular_at_first(state):
        velocity = reduced * state
        return s.det(s.Matrix.hstack(state[:2, 0], velocity[:2, 0])) / state[:2, 0].dot(
            state[:2, 0]
        )

    assert s.simplify(angular_at_first(first) - (s.Rational(1, 2) - s.sqrt(3) / 4)) == 0
    assert s.simplify(angular_at_first(second) - (1 - s.sqrt(3) / 4)) == 0
    assert s.simplify(angular_at_first(second) - angular_at_first(first)) == s.Rational(
        1, 2
    )
    # Hence the two angles alone cannot replace the complete regional form.
    # The finite Cartesian law remains defined if an amplitude reaches zero.
    zero = s.Matrix([0, 0, 1, 0])
    assert reduced * zero == s.Matrix(
        [s.Rational(1, 2), 0, -s.Rational(5, 4), -s.sqrt(3) / 4]
    )


def test_amplitude_ratio_closes_direction_rates_without_common_scale(model):
    s, _, _, _, _, reduced = model
    nu, radius, q = s.symbols("nu radius q", positive=True)
    phi0, phi1 = s.symbols("phi0 phi1", real=True)
    state = radius * s.Matrix(
        [s.cos(phi0), s.sin(phi0), q * s.cos(phi1), q * s.sin(phi1)]
    )
    rate = nu * reduced * state
    radial = []
    angular = []
    for a, r in enumerate((radius, radius * q)):
        form = state[2 * a : 2 * a + 2, 0]
        velocity = rate[2 * a : 2 * a + 2, 0]
        radial.append(s.trigsimp(form.dot(velocity) / r))
        angular.append(s.trigsimp(s.det(s.Matrix.hstack(form, velocity)) / r**2))
    ratio_rate = s.trigsimp((radial[1] - q * radial[0]) / radius)
    delta = phi1 - phi0
    assert s.trigsimp(ratio_rate - nu * (1 - q**2) * s.cos(delta) / 2) == 0
    assert s.trigsimp(angular[0] + s.sqrt(3) * nu / 4 - nu * q * s.sin(delta) / 2) == 0
    assert (
        s.trigsimp(angular[1] + s.sqrt(3) * nu / 4 + nu * s.sin(delta) / (2 * q)) == 0
    )
    assert all(
        radius not in expression.free_symbols for expression in (*angular, ratio_rate)
    )


def test_quadratic_observables_close_and_preserve_the_gram_constraint(model):
    s, _, _, _, _, reduced = model
    nu = s.symbols("nu", positive=True)
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    gram = _gram_observables(s, state)
    # Differentiate the observables of the fine-derived Cartesian generator.
    observed_rate = gram.jacobian(state) * (nu * reduced * state)
    closure = (
        nu * s.Matrix([[-5, 0, 2, 0], [0, -5, 2, 0], [1, 1, -5, 0], [0, 0, 0, -5]]) / 2
    )
    assert s.expand(observed_rate - closure * gram) == s.zeros(4, 1)
    assert s.expand(gram[2] ** 2 + gram[3] ** 2 - gram[0] * gram[1]) == 0

    coordinates = s.Matrix(s.symbols("I0 I1 c signed_area", real=True))
    i0, i1, c, signed_area = coordinates
    constraint = c**2 + signed_area**2 - i0 * i1
    constraint_rate = (
        s.Matrix([constraint]).jacobian(coordinates) * closure * coordinates
    )[0]
    # The off-constraint defect decays; the realizable zero set is invariant.
    assert s.expand(constraint_rate + 5 * nu * constraint) == 0


def test_quadratic_modes_match_one_fine_matrix_exponential(model):
    s, _, generator, _, lift, reduced = model
    nu = s.symbols("nu", positive=True)
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    z0, z1 = state[:2, 0], state[2:, 0]
    total = state.dot(state)
    correlation = z0.dot(z1)
    modes = s.Matrix(
        [
            total + 2 * correlation,
            total - 2 * correlation,
            z0.dot(z0) - z1.dot(z1),
            s.det(s.Matrix.hstack(z0, z1)),
        ]
    )
    rates = -nu * s.diag(3, 7, 5, 5) / 2
    assert s.expand(
        modes.jacobian(state) * nu * reduced * state - rates * modes
    ) == s.zeros(4, 1)

    # A single deterministic binary64 matrix-exponential comparison checks the
    # actual six-node generator, including its uniform mean, without a solver.
    initial = s.Matrix([1, 2, -3, 4]) / 64
    initial_fine = s.ones(6, 1) / 2 + lift * initial
    time, capacity = s.Rational(1, 3), s.Rational(3, 2)
    fine_endpoint = expm(
        np.asarray(time * capacity * generator, dtype=float)
    ) @ np.asarray(initial_fine, dtype=float)
    endpoint = np.asarray(lift.T, dtype=float) @ (fine_endpoint - 0.5)
    initial_modes = modes.subs(dict(zip(state, initial, strict=True)))
    expected = (
        s.diag(*(s.exp(value * capacity * time / nu) for value in rates.diagonal()))
        * initial_modes
    )
    observed = modes.subs(dict(zip(state, endpoint[:, 0], strict=True)))
    np.testing.assert_allclose(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=2e-13,
        atol=2e-17,
    )


def test_direction_and_total_intensity_discard_distinct_source_work_information(model):
    s, _, _, _, _, reduced = model
    nu, scale = s.symbols("nu scale", positive=True)
    state = s.Matrix([1, 0, s.Rational(3, 5), s.Rational(4, 5)]) / 16
    scaled = scale * state
    # Positive common scaling preserves both angles and q, but the source
    # contribution to dI0/dt (nu*c) and the complete quadratic rate scale as k^2.
    for a in range(2):
        z = state[2 * a : 2 * a + 2, 0]
        kz = scaled[2 * a : 2 * a + 2, 0]
        assert s.det(s.Matrix.hstack(z, kz)) == 0
        assert z.dot(kz).is_positive
    assert (
        s.simplify(scaled[2:, 0].dot(scaled[2:, 0]) / scaled[:2, 0].dot(scaled[:2, 0]))
        == 1
    )
    velocity, scaled_velocity = nu * reduced * state, nu * reduced * scaled
    source_work = (
        2 * state[:2, 0].dot(velocity[:2, 0])
        + 5 * nu * state[:2, 0].dot(state[:2, 0]) / 2
    )
    scaled_work = (
        2 * scaled[:2, 0].dot(scaled_velocity[:2, 0])
        + 5 * nu * scaled[:2, 0].dot(scaled[:2, 0]) / 2
    )
    assert s.simplify(source_work - nu * state[:2, 0].dot(state[2:, 0])) == 0
    assert source_work.is_positive
    assert s.simplify(scaled_work - scale**2 * source_work) == 0
    assert (
        s.simplify(2 * scaled.dot(scaled_velocity) - scale**2 * 2 * state.dot(velocity))
        == 0
    )

    # Even equal angles AND equal total intensity do not determine its rate.
    # Positive aligned amplitudes stay aligned in the co-rotating Metzler flow.
    first = s.Matrix([5, 0, 5, 0]) / 128
    second = s.Matrix([1, 0, 7, 0]) / 128
    assert first.dot(first) == second.dot(second) == s.Rational(25, 8192)
    assert first[:2, 0].dot(first[2:, 0]) == s.Rational(25, 16384)
    assert second[:2, 0].dot(second[2:, 0]) == s.Rational(7, 16384)
    assert s.simplify(2 * first.dot(nu * reduced * first)) == -75 * nu / 16384
    assert s.simplify(2 * second.dot(nu * reduced * second)) == -111 * nu / 16384


def test_zero_amplitude_has_a_nonzero_second_intensity_derivative(model):
    s, _, _, _, _, reduced = model
    nu = s.symbols("nu", positive=True)
    state = s.Matrix([0, 0, 1, 0]) / 16
    velocity = nu * reduced * state
    acceleration = nu * reduced * velocity
    z0, z1 = state[:2, 0], state[2:, 0]
    assert z0.dot(z0) == 0
    assert velocity[:2, 0] == s.Matrix([nu / 32, 0])
    assert 2 * z0.dot(velocity[:2, 0]) == 0
    second_derivative = 2 * (
        velocity[:2, 0].dot(velocity[:2, 0]) + z0.dot(acceleration[:2, 0])
    )
    assert s.simplify(second_derivative - nu**2 * z1.dot(z1) / 2) == 0
    assert second_derivative.is_positive
    # The polynomial/Cartesian continuation requires no angle at zero radius.


def test_common_rotation_preserves_gram_but_changes_fine_form_pressure_and_potential(
    model,
):
    s, adjacency, generator, _, lift, reduced = model
    rotation = s.diag(s.Matrix([[0, -1], [1, 0]]), s.Matrix([[0, -1], [1, 0]]))
    assert s.simplify(rotation * reduced - reduced * rotation) == s.zeros(4, 4)
    state = s.Matrix([1, 0, 1, 1]) / 16
    rotated = rotation * state
    original_columns = s.Matrix.hstack(state[:2, 0], state[2:, 0])
    rotated_columns = s.Matrix.hstack(rotated[:2, 0], rotated[2:, 0])
    assert original_columns.T * original_columns == rotated_columns.T * rotated_columns
    assert original_columns.det() == rotated_columns.det()
    assert original_columns.det() != 0
    fine = s.ones(6, 1) / 2 + lift * state
    rotated_fine = s.ones(6, 1) / 2 + lift * rotated
    assert fine != rotated_fine
    pressure, rotated_pressure = generator * fine, generator * rotated_fine
    assert pressure != rotated_pressure

    potentials = []
    for form, source in ((fine, pressure), (rotated_fine, rotated_pressure)):
        graph = nx.from_numpy_array(
            np.asarray(adjacency, dtype=float), create_using=nx.DiGraph
        )
        for i in graph:
            graph.nodes[i].update(
                EPI=float(form[i]), delta_nfr=float(source[i]), theta=0.0, nu_f=1.0
            )
        potentials.append(tuple(compute_structural_potential(graph).values()))
    assert not np.allclose(potentials[0], potentials[1], rtol=0.0, atol=1e-12)
    # An active common contrast rotation at fixed frames preserves the Gram
    # observation while changing fine form and potential. Primitive phases
    # stay fixed, and the stored pressures are the pure-EPI generator values.


def test_autonomous_gram_for_all_means_requires_regionally_constant_capacity(model):
    s, _, generator, basis, lift, _ = model
    capacities = s.Matrix(s.symbols("nu0:6", positive=True))
    mean_lift = s.diag(s.ones(3, 1), s.ones(3, 1))
    mean_leak = s.simplify(lift.T * s.diag(*capacities) * generator * mean_lift)
    b0, b1 = basis.T * capacities[:3, 0], basis.T * capacities[3:, 0]
    assert s.simplify(
        mean_leak - s.Matrix.hstack((-b0).col_join(b1), b0.col_join(-b1)) / 2
    ) == s.zeros(4, 2)

    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    gram = _gram_observables(s, state)
    mean_sensitivity = s.expand(gram.jacobian(state) * mean_leak)
    assert s.expand(
        mean_sensitivity[:2, 0]
        - s.Matrix([-state[:2, 0].dot(b0), state[2:, 0].dot(b1)])
    ) == s.zeros(2, 1)
    assert mean_sensitivity[:, 1] == -mean_sensitivity[:, 0]
    # Independent hidden means must not change either intensity derivative
    # for any contrast. Their polynomial coefficients require b0=b1=0, also
    # on an arbitrarily small open neighborhood of an admitted uniform form.
    coefficients = mean_sensitivity[:2, 0].jacobian(state)
    necessary = s.Matrix(
        [coefficients[0, 0], coefficients[0, 1], coefficients[1, 2], coefficients[1, 3]]
    )
    equations, rhs = s.linear_eq_to_matrix(necessary, tuple(capacities))
    assert rhs == s.zeros(4, 1)
    assert equations.rank() == 4
    assert s.Matrix.hstack(*equations.nullspace()) == mean_lift
    # The kernel is exactly (nu0,nu0,nu0,nu1,nu1,nu1); equality between
    # triangles is not forced. This is necessity, not just one failed profile.


def test_regional_capacities_close_gram_and_the_weighted_full_form_budget(model):
    s, _, generator, _, lift, reduced = model
    nu0, nu1 = s.symbols("nu0 nu1", positive=True)
    means = s.Matrix(s.symbols("mu0 mu1", real=True))
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    mean_lift = s.diag(s.ones(3, 1), s.ones(3, 1))
    capacities = s.diag(*([nu0] * 3 + [nu1] * 3))
    fine = mean_lift * means + lift * state
    fine_rate = capacities * generator * fine
    velocity = s.simplify(lift.T * fine_rate)
    assert s.simplify(
        velocity - s.diag(nu0, nu0, nu1, nu1) * reduced * state
    ) == s.zeros(4, 1)
    mean_rate = s.simplify(mean_lift.T * fine_rate / 3)
    assert (
        mean_rate
        == s.Matrix([nu0 * (means[1] - means[0]), nu1 * (means[0] - means[1])]) / 2
    )
    z0, z1 = state[0] + s.I * state[1], state[2] + s.I * state[3]
    for a, (nu, own, other) in enumerate(((nu0, z0, z1), (nu1, z1, z0))):
        complex_rate = velocity[2 * a] + s.I * velocity[2 * a + 1]
        assert (
            s.expand(
                complex_rate - nu * (-5 - s.I * s.sqrt(3)) * own / 4 - nu * other / 2
            )
            == 0
        )

    gram = _gram_observables(s, state)
    gram_rate = s.expand(gram.jacobian(state) * velocity)
    i0, i1, correlation, signed_area = gram
    assert s.expand(
        gram_rate[:2, 0]
        - s.Matrix(
            [
                -5 * nu0 * i0 / 2 + nu0 * correlation,
                -5 * nu1 * i1 / 2 + nu1 * correlation,
            ]
        )
    ) == s.zeros(2, 1)
    cross_rate = (-5 * (nu0 + nu1) + s.I * s.sqrt(3) * (nu0 - nu1)) * (
        correlation + s.I * signed_area
    ) / 4 + (nu0 * i1 + nu1 * i0) / 2
    assert s.expand(gram_rate[2] + s.I * gram_rate[3] - cross_rate) == 0
    constraint_rate = 2 * correlation * gram_rate[2] + 2 * signed_area * gram_rate[3]
    constraint_rate -= i1 * gram_rate[0] + i0 * gram_rate[1]
    assert s.expand(constraint_rate) == 0

    gap = (state[:2, 0] - state[2:, 0]).dot(state[:2, 0] - state[2:, 0])
    loss = 3 * (i0 + i1) / 2 + gap
    assert s.expand(gram_rate[0] / nu0 + gram_rate[1] / nu1 + loss) == 0
    full_budget = (fine.T * capacities.inv() * fine)[0]
    assert (
        s.simplify(
            full_budget
            - (3 * means[0] ** 2 + i0) / nu0
            - (3 * means[1] ** 2 + i1) / nu1
        )
        == 0
    )
    full_rate = 2 * (fine.T * capacities.inv() * fine_rate)[0]
    assert s.simplify(full_rate + 3 * (means[0] - means[1]) ** 2 + loss) == 0
    # These positive weighted budgets lose contrast even when regional
    # capacities differ. They do not introduce a law evolving capacity.


def test_nonconstant_capacity_breaks_gram_even_with_equal_observed_means(model):
    s, _, generator, _, lift, _ = model
    epsilon, radius = s.symbols("epsilon radius", positive=True)
    capacities = s.diag(1 + epsilon, 1, 1, 1, 1, 1)
    first = s.Matrix([radius, 0, radius, 0])
    rotated = s.Matrix([0, radius, 0, radius])
    assert _gram_observables(s, first) == _gram_observables(s, rotated)
    rates = []
    for state in (first, rotated):
        fine = s.ones(6, 1) / 2 + lift * state
        assert [s.simplify(sum(fine[a : a + 3, 0]) / 3) for a in (0, 3)] == [
            s.Rational(1, 2)
        ] * 2
        velocity = lift.T * capacities * generator * fine
        rates.append(s.simplify(2 * state.dot(velocity)))
    assert rates == [-(3 + epsilon) * radius**2, -3 * radius**2]
    assert s.simplify(
        (rates[0] - rates[1]).subs({epsilon: 1, radius: s.Rational(1, 16)})
    ) == -s.Rational(1, 256)
    # Even retaining both instantaneous means cannot repair this Gram loss:
    # common contrast orientation now affects the unweighted intensity rate.


def test_unequal_regional_capacity_rates_bind_fresh_engine_pressure_exactly(model):
    s, adjacency, generator, _, lift, reduced = model
    form = s.Matrix([10, 6, 8, 13, 9, 8]) / 16
    nu0, nu1 = s.Rational(3, 2), s.Rational(5, 4)
    capacities = s.diag(*([nu0] * 3 + [nu1] * 3))
    graph = nx.from_numpy_array(
        np.asarray(adjacency, dtype=float), create_using=nx.DiGraph
    )
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
    )
    for i in graph:
        graph.nodes[i].update(
            EPI=float(form[i]), nu_f=float(capacities[i, i]), theta=0.0
        )
    default_compute_delta_nfr(graph)
    # Binary64 inputs, neighbor means, pressure and products are dyadic here;
    # convert actual outputs to exact rationals before the surd projection.
    pressure = s.Matrix(
        [s.Rational(get_attr(graph.nodes[i], ALIAS_DNFR, None)) for i in graph]
    )
    rate = s.Matrix(
        [
            s.Rational(
                get_attr(graph.nodes[i], ALIAS_VF, None)
                * get_attr(graph.nodes[i], ALIAS_DNFR, None)
            )
            for i in graph
        ]
    )
    assert pressure == generator * form
    assert rate == capacities * generator * form
    state = s.simplify(lift.T * form)
    velocity = s.simplify(lift.T * rate)
    assert s.simplify(
        velocity - s.diag(nu0, nu0, nu1, nu1) * reduced * state
    ) == s.zeros(4, 1)
    assert s.simplify(
        2
        * (
            state[:2, 0].dot(velocity[:2, 0]) / nu0
            + state[2:, 0].dot(velocity[2:, 0]) / nu1
        )
    ) == -s.Rational(39, 256)
    assert s.simplify(2 * (form.T * capacities.inv() * rate)[0]) == -s.Rational(51, 256)
    assert all(
        get_attr(graph.nodes[i], ALIAS_EPI, None) == float(form[i]) for i in graph
    )
    assert all(get_attr(graph.nodes[i], ALIAS_THETA, None) == 0 for i in graph)
    assert tuple(get_attr(graph.nodes[i], ALIAS_VF, None) for i in graph) == tuple(
        map(float, capacities.diagonal())
    )

    # A held-pressure Euler step is an exact dyadic arithmetic control here;
    # it is not asserted to equal the continuous state-dependent flow.
    dt = s.Rational(1, 32)
    update_epi_via_nodal_equation(graph, dt=float(dt), t=0.0, method="euler")
    endpoint = s.Matrix(
        [s.Rational(get_attr(graph.nodes[i], ALIAS_EPI, None)) for i in graph]
    )
    assert endpoint == form + dt * rate
    assert tuple(get_attr(graph.nodes[i], ALIAS_VF, None) for i in graph) == tuple(
        map(float, capacities.diagonal())
    )
    assert all(get_attr(graph.nodes[i], ALIAS_THETA, None) == 0 for i in graph)


def test_capacity_detuning_predicts_phase_curvature_missing_from_frozen_ratio(model):
    s, _, generator, _, lift, _ = model
    nu0, nu1 = s.symbols("nu0 nu1", positive=True)
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    fine_generator = s.diag(*([nu0] * 3 + [nu1] * 3)) * generator
    velocity = s.simplify(lift.T * fine_generator * lift * state)
    angular = []
    for offset in (0, 2):
        form = state[offset : offset + 2, 0]
        rate = velocity[offset : offset + 2, 0]
        angular.append(s.det(s.Matrix.hstack(form, rate)) / form.dot(form))
    relative_rate = angular[1] - angular[0]
    relative_acceleration = (s.Matrix([relative_rate]).jacobian(state) * velocity)[0]
    initial = dict(zip(state, (1, 0, 0, 1), strict=True))
    difference, total = nu0 - nu1, nu0 + nu1
    assert (
        s.simplify(relative_rate.subs(initial) - s.sqrt(3) * difference / 4 + total / 2)
        == 0
    )
    assert s.simplify(relative_acceleration.subs(initial) + 5 * difference**2 / 8) == 0

    delta = s.symbols("delta", real=True)
    comparator = s.sqrt(3) * difference / 4 - total * s.sin(delta) / 2
    assert (
        s.simplify(comparator.subs(delta, s.pi / 2) - relative_rate.subs(initial)) == 0
    )
    assert (s.diff(comparator, delta) * comparator).subs(delta, s.pi / 2) == 0
    # Equal initial phase and phase rate cannot identify the later response.
    # This is a prospective Taylor coefficient, not a fit to simulated phases.
    assert relative_acceleration.subs(initial).subs(nu1, nu0).simplify() == 0


def test_capacity_eigenpairs_and_nonvanishing_discriminant_follow_from_fine_law(model):
    s, _, generator, _, lift, _ = model
    nu0, nu1 = s.symbols("nu0 nu1", positive=True)
    reduced = s.simplify(lift.T * s.diag(*([nu0] * 3 + [nu1] * 3)) * generator * lift)
    complex_generator = s.Matrix(
        [
            [reduced[0, 0] + s.I * reduced[1, 0], reduced[0, 2]],
            [reduced[2, 0], reduced[2, 2] + s.I * reduced[3, 2]],
        ]
    )
    c = -(5 + s.I * s.sqrt(3)) / 4
    difference = nu0 - nu1
    discriminant = nu0 * nu1 + c**2 * difference**2
    assert s.expand(s.re(discriminant) - nu0 * nu1 - 11 * difference**2 / 8) == 0
    assert s.expand(s.im(discriminant) - 5 * s.sqrt(3) * difference**2 / 8) == 0
    # Its real part is strictly positive for positive capacities. The
    # principal square root therefore has positive real part and no collision.
    root = s.sqrt(discriminant)
    ratios = []
    for sign in (1, -1):
        eigenvalue = (c * (nu0 + nu1) + sign * root) / 2
        ratio = (-c * difference + sign * root) / nu0
        eigenvector = s.Matrix([1, ratio])
        assert s.simplify(
            complex_generator * eigenvector - eigenvalue * eigenvector
        ) == s.zeros(2, 1)
        ratios.append(ratio)
        assert (
            s.simplify(eigenvalue.subs(nu1, nu0) - nu0 * (c + s.Rational(sign, 2))) == 0
        )
        assert s.simplify(ratio.subs(nu1, nu0) - sign) == 0
    assert s.simplify(ratios[0] * ratios[1] + nu1 / nu0) == 0


@pytest.mark.parametrize("capacities", ((1.0, 2.0), (2.0, 1.0), (1.0, 1.0)))
@pytest.mark.parametrize("preparation", ("generic", "fast_eigenline"))
def test_capacity_modes_predict_fine_six_node_evolution(model, capacities, preparation):
    _, _, generator, _, lift, _ = model
    nu0, nu1 = capacities
    c = -(5 + 1j * np.sqrt(3)) / 4
    root = np.sqrt(nu0 * nu1 + c**2 * (nu0 - nu1) ** 2)
    eigenvalues = np.array([(c * (nu0 + nu1) + sign * root) / 2 for sign in (1, -1)])
    ratios = np.array([(-c * (nu0 - nu1) + sign * root) / nu0 for sign in (1, -1)])
    vectors = np.vstack((np.ones(2), ratios))
    initial = (
        np.array([1 + 2j, -3 + 1j]) / 64
        if preparation == "generic"
        else vectors[:, 1] / 64
    )
    coefficients = np.linalg.solve(vectors, initial)
    if preparation == "fast_eigenline":
        assert abs(coefficients[0]) < 1e-17
    else:
        assert abs(coefficients[0]) > 1e-3
    initial_cartesian = np.array(
        [initial[0].real, initial[0].imag, initial[1].real, initial[1].imag]
    )
    means = np.array([3 / 8, 5 / 8])
    real_lift = np.asarray(lift, dtype=float)
    initial_fine = np.repeat(means, 3) + real_lift @ initial_cartesian
    fine_generator = np.diag(np.repeat(capacities, 3)) @ np.asarray(
        generator, dtype=float
    )
    weighted_mean = (nu1 * means[0] + nu0 * means[1]) / (nu0 + nu1)

    for time in (0.0, 0.125, 0.75):
        # The reference acts on real six-node form, never on a complex array
        # silently cast to the real scalar domain of the shared engine.
        observed_fine = expm(time * fine_generator) @ initial_fine
        expected = vectors @ (np.exp(time * eigenvalues) * coefficients)
        contrast = np.array(
            [expected[0].real, expected[0].imag, expected[1].real, expected[1].imag]
        )
        mean_gap = (means[0] - means[1]) * np.exp(-(nu0 + nu1) * time / 2)
        expected_means = weighted_mean + np.array([nu0, -nu1]) * mean_gap / (nu0 + nu1)
        expected_fine = np.repeat(expected_means, 3) + real_lift @ contrast
        np.testing.assert_allclose(observed_fine, expected_fine, rtol=2e-14, atol=2e-16)
        if preparation == "fast_eigenline":
            np.testing.assert_allclose(
                expected[1], ratios[1] * expected[0], rtol=2e-14, atol=1e-18
            )
    # Finite controls supplement the algebraic theorem, not a capacity sweep
    # or a claim of sustained amplitudes. The equal-capacity branch is included.
    assert root.real > 0
    assert np.all(eigenvalues.real < 0)
    assert abs(np.angle(ratios[0])) < np.arctan(np.sqrt(3) / 5)


def test_capacity_detuning_continues_cartesian_form_through_zero_region(model):
    s, _, generator, _, lift, _ = model
    nu0, nu1 = s.symbols("nu0 nu1", positive=True)
    fine_generator = s.diag(*([nu0] * 3 + [nu1] * 3)) * generator
    reduced = s.simplify(lift.T * fine_generator * lift)
    at_zero = s.Matrix([0, 0, s.Rational(1, 32), 0])
    velocity = reduced * at_zero
    assert velocity[:2, 0] == s.Matrix([nu0 / 64, 0])
    assert 2 * at_zero[:2, 0].dot(velocity[:2, 0]) == 0
    second_intensity_rate = 2 * velocity[:2, 0].dot(velocity[:2, 0])
    assert s.simplify(second_intensity_rate - nu0**2 / 2048) == 0

    # Prepare a future coordinate zero independently by reversing a finite
    # reference exponential. The tested forward path remains ordinary form
    # diffusion; no angle is assigned at the zero or used to cross it.
    numerical_generator = np.asarray(fine_generator.subs({nu0: 1, nu1: 2}), dtype=float)
    real_lift = np.asarray(lift, dtype=float)
    zero_fine = 0.5 + real_lift @ np.asarray(at_zero, dtype=float)[:, 0]
    crossing_time = 0.25
    initial = expm(-crossing_time * numerical_generator) @ zero_fine
    assert 0 < min(initial) <= max(initial) < 1
    observed = expm(crossing_time * numerical_generator) @ initial
    np.testing.assert_allclose(observed, zero_fine, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        real_lift.T @ observed,
        np.asarray(at_zero, dtype=float)[:, 0],
        rtol=0,
        atol=2e-16,
    )
    before = real_lift.T @ (
        expm((crossing_time - 1e-5) * numerical_generator) @ initial
    )
    after = real_lift.T @ (expm((crossing_time + 1e-5) * numerical_generator) @ initial)
    assert before[0] < 0 < after[0]
    assert np.all(np.isfinite(before)) and np.all(np.isfinite(after))


def test_initially_uniform_recipient_has_an_exact_transient_form_episode(model):
    s, _, generator, _, lift, reduced = model
    rho, radius = s.symbols("rho radius", positive=True)
    time = s.symbols("t", nonnegative=True)
    rotation = s.Matrix(
        [s.cos(s.sqrt(3) * rho * time / 4), -s.sin(s.sqrt(3) * rho * time / 4)]
    )
    slow, fast = s.exp(-3 * rho * time / 4), s.exp(-7 * rho * time / 4)
    recipient = radius * (slow - fast) * rotation / 2
    donor = radius * (slow + fast) * rotation / 2
    state = recipient.col_join(donor)
    assert s.simplify(state.diff(time) - rho * reduced * state) == s.zeros(4, 1)
    assert state.subs(time, 0) == s.Matrix([0, 0, radius, 0])
    fine = s.ones(6, 1) / 2 + lift * state
    assert s.simplify(fine.diff(time) - rho * generator * fine) == s.zeros(6, 1)
    gram = s.simplify(_gram_observables(s, state))
    intensity = (
        radius**2 * s.exp(-3 * rho * time / 2) * (1 - s.exp(-rho * time)) ** 2 / 4
    )
    assert s.simplify(gram[0] - intensity) == 0
    assert s.simplify(gram[2] ** 2 - gram[0] * gram[1]) == gram[3] == 0
    assert s.limit(intensity, time, s.oo) == 0
    # Contrast is transferred into a uniform recipient and later tends to zero.
    # It is positive at every finite t>0: loss of visibility is not extinction.


def test_episode_peak_discriminates_predeclared_observation_thresholds(model):
    s, _, _, _, _, _ = model
    u = s.symbols("u", positive=True)
    profile = u ** s.Rational(3, 2) * (1 - u) ** 2 / 4
    assert s.simplify(s.diff(profile, u) - s.sqrt(u) * (1 - u) * (3 - 7 * u) / 8) == 0
    # u=exp(-rho*t) decreases from one to zero. Thus t*=log(7/3)/rho
    # is the unique interior maximum. A cut below it has two crossings, an
    # equal cut has one tangency, and a greater cut has no crossing.
    peak = profile.subs(u, s.Rational(3, 7))
    assert (
        s.simplify(peak - s.Rational(4, 49) * s.Rational(3, 7) ** s.Rational(3, 2)) == 0
    )
    cut = s.Rational(1, 8192)
    assert s.simplify(peak / 128 - cut).is_positive
    assert s.simplify(cut - peak / 512).is_positive
    assert profile.subs(u, 1) == s.limit(profile, u, 0, dir="+") == 0


@pytest.mark.parametrize("scenario", ("strong", "weak", "disconnected"))
def test_reserved_episode_predictions_use_fresh_pressure_and_shared_steps(
    model, scenario
):
    s, connected_adjacency, connected_generator, _, _, _ = model
    amplitude = Fraction(1, 32 if scenario == "weak" else 16)
    dt, epi_weight, cut = Fraction(1, 32), Fraction(1, 2), Fraction(1, 8192)
    initial = (Fraction(1, 2),) * 3 + tuple(Fraction(1, 2) + amplitude * p for p in P)
    adjacency = s.Matrix(connected_adjacency)
    generator = connected_generator
    if scenario == "disconnected":
        adjacency[:3, 3:] = s.zeros(3)
        adjacency[3:, :3] = s.zeros(3)
        generator = adjacency - s.eye(6)

    # Freeze the law, preparations, supports, clock, threshold, exact Euler
    # reference and continuous reserved predictions before executing the engine.
    pressure_matrix = tuple(
        tuple(Fraction(value) for value in row)
        for row in (epi_weight * generator).tolist()
    )
    euler = tuple(
        tuple(Fraction(i == j) + dt * value for j, value in enumerate(row))
        for i, row in enumerate(pressure_matrix)
    )
    assert all(min(row) >= 0 and sum(row) == 1 for row in euler)
    reference = [initial]
    for _ in range(160):
        reference.append(_apply(euler, reference[-1]))

    def regional_intensity(values):
        mean = sum(values[:3]) / 3
        return sum((value - mean) ** 2 for value in values[:3])

    reserved = (0, 16, 32, 64, 160)
    expected_visible = (
        (False, False, True, True, False) if scenario == "strong" else (False,) * 5
    )
    continuous = {}
    for index, visible in zip(reserved, expected_visible, strict=True):
        time = float(index * dt)
        continuous[index] = expm(
            float(epi_weight) * time * np.asarray(generator, dtype=float)
        ) @ np.asarray(initial, dtype=float)
        predicted_intensity = (
            0.0
            if scenario == "disconnected"
            else float(2 * amplitude**2)
            * np.exp(-3 * time / 4)
            * (1 - np.exp(-time / 2)) ** 2
            / 4
        )
        assert (predicted_intensity > float(cut)) == visible
        assert (regional_intensity(reference[index]) > cut) == visible
        np.testing.assert_allclose(
            regional_intensity(continuous[index]),
            predicted_intensity,
            rtol=1e-12,
            atol=1e-30,
        )

    graph = nx.from_numpy_array(
        np.asarray(adjacency, dtype=float), create_using=nx.DiGraph
    )
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 0.25, "epi": 0.5, "vf": 0.125, "topo": 0.125},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        EPI_MIN=0.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
    )
    for node, value in zip(graph, initial, strict=True):
        graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0)

    def captured(attribute):
        return tuple(
            Fraction(get_attr(graph.nodes[node], attribute, None)) for node in graph
        )

    defect_bound = Fraction(0)
    retained = {0: initial}
    pressure_defects, integration_defects = [], []
    for index in range(160):
        before = captured(ALIAS_EPI)
        default_compute_delta_nfr(graph)
        pressure = captured(ALIAS_DNFR)
        pressure_defect = tuple(
            actual - expected
            for actual, expected in zip(
                pressure, _apply(pressure_matrix, before), strict=True
            )
        )
        update_epi_via_nodal_equation(
            graph, dt=float(dt), t=float(index * dt), method="euler"
        )
        after = captured(ALIAS_EPI)
        integration_defect = tuple(
            new - old - dt * p
            for new, old, p in zip(after, before, pressure, strict=True)
        )
        local_defect = tuple(
            dt * p + e for p, e in zip(pressure_defect, integration_defect, strict=True)
        )
        pressure_defects.append(pressure_defect)
        integration_defects.append(integration_defect)
        defect_bound += max(map(abs, local_defect))
        # Stochastic M propagates any signed local defects with sup-norm gain
        # at most one. This bounds runtime arithmetic, not Euler truncation.
        assert (
            max(
                abs(actual - expected)
                for actual, expected in zip(after, reference[index + 1], strict=True)
            )
            <= defect_bound
        )
        assert min(initial) <= min(after) <= max(after) <= max(initial)
        assert captured(ALIAS_VF) == (1,) * 6
        assert captured(ALIAS_THETA) == (0,) * 6
        assert all(
            abs(sum(after[start : start + 3]) / 3 - Fraction(1, 2)) <= defect_bound
            for start in (0, 3)
        )
        if scenario == "disconnected":
            assert after[:3] == (Fraction(1, 2),) * 3
        if index + 1 in reserved:
            retained[index + 1] = after

    assert len(pressure_defects) == len(integration_defects) == 160
    assert defect_bound < Fraction(1, 10**12)
    for index, visible in zip(reserved, expected_visible, strict=True):
        assert (regional_intensity(retained[index]) > cut) == visible
        # Both finite references were reserved. Their difference isolates the
        # finite Euler error from the much smaller accumulated runtime defect.
        discretization_error = np.max(
            np.abs(np.asarray(reference[index], dtype=float) - continuous[index])
        )
        observed_error = np.max(
            np.abs(np.asarray(retained[index], dtype=float) - continuous[index])
        )
        assert observed_error <= discretization_error + float(defect_bound) + 2e-16


@pytest.fixture(scope="module")
def interacting_components(model):
    s, adjacency, generator, basis, _, _ = model
    # Reciprocal ports join corresponding vertices with conductance two. Each
    # full row has strength four; attaching a component changes its normalization.
    joint_adjacency = s.BlockMatrix(
        [[adjacency, 2 * s.eye(6)], [2 * s.eye(6), adjacency]]
    ).as_explicit()
    joint_generator = joint_adjacency / 4 - s.eye(12)
    diagonal = generator / 2 - s.eye(6) / 2
    assert (
        joint_generator
        == s.BlockMatrix(
            [[diagonal, s.eye(6) / 2], [s.eye(6) / 2, diagonal]]
        ).as_explicit()
    )
    lift = s.diag(basis, basis, basis, basis)
    mean_lift = s.diag(*(s.ones(3, 1) for _ in range(4)))
    contrast = s.Matrix(tuple(P) * 4) / 16
    aligned = s.ones(12, 1) / 2 + contrast
    reversed_component = s.diag(*([-1] * 6 + [1] * 6)) * contrast
    opposed = s.ones(12, 1) / 2 + reversed_component
    return s, joint_adjacency, joint_generator, lift, mean_lift, (aligned, opposed)


def test_corresponding_ports_preserve_joint_complex_and_gram_closure(
    interacting_components,
):
    s, _, generator, lift, mean_lift, _ = interacting_components
    mean_generator = (
        s.Matrix([[-3, 1, 2, 0], [1, -3, 0, 2], [2, 0, -3, 1], [0, 2, 1, -3]]) / 4
    )
    mean_projection = mean_lift.T / 3
    assert generator * mean_lift == mean_lift * mean_generator
    assert mean_projection * generator == mean_generator * mean_projection
    assert s.simplify(mean_projection * generator * lift) == s.zeros(4, 8)
    assert s.simplify(lift.T * generator * mean_lift) == s.zeros(8, 4)

    # Four ordered (u,v) pairs describe the four triangular contrasts. These
    # coefficients are specified independently of the fine projection below.
    complex_projection = s.diag(*(s.Matrix([[1, s.I]]) for _ in range(4)))
    damping = s.Matrix([[-9, 2, 4, 0], [2, -9, 0, 4], [4, 0, -9, 2], [0, 4, 2, -9]]) / 8
    complex_generator = damping - s.I * s.sqrt(3) * s.eye(4) / 8
    projection = complex_projection * lift.T
    assert s.simplify(
        projection * generator - complex_generator * projection
    ) == s.zeros(4, 12)
    assert mean_lift.row_join(lift).rank() == 12

    cartesian = s.Matrix(s.symbols("u0 v0 u1 v1 u2 v2 u3 v3", real=True))
    fine = lift * cartesian
    z = complex_projection * cartesian
    gram = z * z.conjugate().T
    # Differentiate the actual fine-derived observables, not a caller-supplied
    # Gram flow. The common imaginary drift cancels from every Gram entry.
    observed_rate = s.Matrix(gram).reshape(16, 1).jacobian(cartesian) * (
        lift.T * generator * fine
    )
    expected_rate = damping * gram + gram * damping
    assert s.simplify(observed_rate - expected_rate.reshape(16, 1)) == s.zeros(16, 1)
    assert (
        s.expand(
            expected_rate[0, 0]
            + s.Rational(9, 4) * gram[0, 0]
            - (gram[0, 1] + gram[1, 0]) / 4
            - (gram[0, 2] + gram[2, 0]) / 2
        )
        == 0
    )


def test_equal_local_grams_do_not_determine_boundary_contrast_response(
    interacting_components,
):
    s, _, generator, lift, mean_lift, preparations = interacting_components
    projection = s.diag(*(s.Matrix([[1, s.I]]) for _ in range(4))) * lift.T
    grams = []
    contrast_rates = []
    interface_rates = []
    for fine in preparations:
        assert mean_lift.T * fine / 3 == s.ones(4, 1) / 2
        z = s.simplify(projection * fine)
        gram = s.simplify(z * z.conjugate().T)
        grams.append(gram)
        contrasts = fine - s.ones(12, 1) / 2
        local = contrasts[:6, 0]
        velocity = (generator * fine)[:6, 0]
        contrast_rates.append(2 * local.dot(velocity))
        interface_rates.append(local.dot(contrasts[6:, 0] - local))
        assert gram.rank() == 1
        # Dropping the cross block gives rank two and is not a joint Gram
        # realizable by one simultaneous four-region scalar-form preparation.
        assert s.diag(gram[:2, :2], gram[2:, 2:]).rank() == 2

    assert preparations[0][6:, 0] == preparations[1][6:, 0]
    assert grams[0][:2, :2] == grams[1][:2, :2] == s.ones(2) / 128
    assert grams[0][2:, 2:] == grams[1][2:, 2:] == s.ones(2) / 128
    assert grams[0][:2, 2:] == s.ones(2) / 128
    assert grams[1][:2, 2:] == -s.ones(2) / 128
    assert contrast_rates == [-s.Rational(3, 256), -s.Rational(11, 256)]
    assert interface_rates == [0, -s.Rational(1, 32)]
    # These are component contrast-square rates, not per-vertex port fluxes.
    assert contrast_rates[1] - contrast_rates[0] == interface_rates[1]


@pytest.fixture(scope="module")
def component_interaction_execution(interacting_components):
    s, adjacency, _, _, _, preparations = interacting_components
    retained = []
    for initial in preparations:
        graph = nx.from_numpy_array(
            np.asarray(adjacency, dtype=float), create_using=nx.DiGraph
        )
        graph.graph.update(
            DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
            GAMMA={"type": "none"},
            use_extended_dynamics=False,
            DT_MIN=0.0,
            EPI_MIN=0.0,
            EPI_MAX=1.0,
            CLIP_MODE="hard",
        )
        for node in graph:
            graph.nodes[node].update(EPI=float(initial[node]), nu_f=1.0, theta=0.0)

        def captured(attribute):
            return s.Matrix(
                [
                    s.Rational(get_attr(graph.nodes[node], attribute, None))
                    for node in graph
                ]
            )

        nodes, laplacian = structural_diffusion_operator(graph)
        default_compute_delta_nfr(graph)
        pressure = captured(ALIAS_DNFR)
        unchanged_form = captured(ALIAS_EPI)
        update_epi_via_nodal_equation(graph, dt=1 / 32, t=0.0, method="euler")
        retained.append(
            (
                nodes,
                laplacian,
                pressure,
                unchanged_form,
                captured(ALIAS_EPI),
                captured(ALIAS_VF),
                captured(ALIAS_THETA),
            )
        )
    return retained


def test_interacting_component_rates_and_shared_step_bind_the_full_support(
    interacting_components, component_interaction_execution
):
    s, adjacency, generator, _, _, preparations = interacting_components
    assert tuple(sum(adjacency.row(i)) for i in range(12)) == (4,) * 12
    outgoing = directed_rw_laplacian(np.asarray(adjacency, dtype=float))
    for initial, evidence in zip(
        preparations, component_interaction_execution, strict=True
    ):
        nodes, laplacian, pressure, before, endpoint, capacity, phase = evidence
        assert tuple(nodes) == tuple(range(12))
        np.testing.assert_array_equal(laplacian, outgoing)
        assert -s.Matrix(laplacian).applyfunc(s.Rational) == generator
        assert before == initial
        assert pressure == generator * initial
        assert endpoint == initial + generator * initial / 32
        assert capacity == s.ones(12, 1)
        assert phase == s.zeros(12, 1)
        assert min(initial) <= min(endpoint) <= max(endpoint) <= max(initial)
        # Equality here binds a dyadic held-pressure Euler step. It is not an
        # assertion that Euler equals the continuous matrix exponential.


def test_joint_gram_realizability_relations_follow_from_fine_form(model):
    s, _, _, basis, _, _ = model
    # Three components retain two triangular contrasts each. This checks the
    # observation's state domain, with no new support law or evolution solver.
    lift = s.diag(*(basis for _ in range(6)))
    mean_lift = s.diag(*(s.ones(3, 1) for _ in range(6)))
    cartesian = s.Matrix([0, 0, 1, 2, 2, -1, 1, 0, -1, 1, 0, 3]) / 32
    fine = mean_lift * (s.ones(6, 1) / 2) + lift * cartesian
    projection = s.diag(*(s.Matrix([[1, s.I]]) for _ in range(6))) * lift.T
    z = s.simplify(projection * fine)
    gram = s.simplify(z * z.conjugate().T)
    assert mean_lift.T * fine / 3 == s.ones(6, 1) / 2
    assert z[:2, 0] == s.Matrix([0, (1 + 2 * s.I) / 32])

    local_a, local_b = gram[:2, :2], gram[2:4, 2:4]
    relation_ab, relation_bc, relation_ac = (
        gram[:2, 2:4],
        gram[2:4, 4:6],
        gram[:2, 4:6],
    )
    intensity_b = s.trace(local_b)
    assert intensity_b == s.Rational(3, 512)
    assert s.simplify(relation_ab * relation_bc - intensity_b * relation_ac) == s.zeros(
        2
    )
    assert s.simplify(
        relation_ab * relation_ab.conjugate().T - intensity_b * local_a
    ) == s.zeros(2)
    # A zero regional contrast has no orientation or nonzero relation row.
    assert gram[0, :] == s.zeros(1, 6)
    assert gram[:, 0] == s.zeros(6, 1)

    # A nonzero diagonal anchor reconstructs one representative without
    # assigning an angle to the zero contrast or recovering a common angle.
    anchor = 1
    representative = s.simplify(gram[:, anchor] / s.sqrt(gram[anchor, anchor]))
    assert representative[anchor] == s.sqrt(5) / 32
    assert s.simplify(representative * representative.conjugate().T - gram) == s.zeros(
        6
    )


def test_pairwise_realizable_relations_can_fail_joint_realizability(model):
    s, _, _, _, _, _ = model
    correlation = s.Matrix([[1, 1, -1], [1, 1, 1], [-1, 1, 1]])
    # Every pair alone can come from two unit scalar contrasts. These are
    # compatibility checks of proposed observations, not dynamical evidence.
    for pair in ((0, 1), (1, 2), (0, 2)):
        local_pair = correlation.extract(pair, pair)
        assert local_pair.eigenvals() == {s.Integer(0): 1, s.Integer(2): 1}
    witness = s.Matrix([1, -1, 1])
    assert (witness.T * correlation * witness)[0] == -3
    # The negative quadratic form rules out a joint Gram. The three proposed
    # relations also violate the deterministic cycle-composition identity.
    cycle = correlation[0, 1] * correlation[1, 2] * correlation[2, 0]
    assert cycle == -1
    assert correlation[0, 0] * correlation[1, 1] * correlation[2, 2] == 1


@pytest.fixture(scope="module")
def imperfect_component_model(interacting_components):
    s, adjacency, base, lift, mean_lift, _ = interacting_components
    # Frozen prospective protocol: one port perturbation, one amplitude, one
    # horizon and one step size. Both preparations reverse ALL four contrasts.
    epsilon, amplitude = s.Rational(1, 4), s.Rational(1, 16)
    eta = epsilon / (4 + epsilon)
    adjacency = s.Matrix(adjacency)
    adjacency[0, 6] += epsilon
    strength = s.diag(*(sum(adjacency.row(i)) for i in range(12)))
    generator = strength.inv() * adjacency - s.eye(12)
    contrast = s.Matrix(tuple(P) * 4)
    preparations = tuple(
        s.ones(12, 1) / 2 + sign * amplitude * contrast for sign in (1, -1)
    )
    dt, steps = Fraction(1, 1024), 32
    time = steps * dt
    # Reserve continuous endpoints independently before running the engine.
    continuous = tuple(
        expm(float(time) * np.asarray(generator, dtype=float))
        @ np.asarray(initial, dtype=float)[:, 0]
        for initial in preparations
    )
    baseline = tuple(
        expm(float(time) * np.asarray(base, dtype=float))
        @ np.asarray(initial, dtype=float)[:, 0]
        for initial in preparations
    )
    return {
        "sympy": s,
        "adjacency": adjacency,
        "base": base,
        "generator": generator,
        "lift": lift,
        "mean_lift": mean_lift,
        "preparations": preparations,
        "eta": eta,
        "amplitude": amplitude,
        "dt": dt,
        "steps": steps,
        "continuous": continuous,
        "baseline": baseline,
    }


def test_single_port_breaks_gram_projectability_with_a_known_modal_source(
    imperfect_component_model,
):
    model = imperfect_component_model
    s, base, generator = model["sympy"], model["base"], model["generator"]
    epsilon, time = s.symbols("epsilon time", nonnegative=True)
    eta = epsilon / (4 + epsilon)
    adjacency = model["adjacency"].copy()
    adjacency[0, 6] = 2 + epsilon
    normalized = s.diag(
        *(1 / sum(adjacency.row(i)) for i in range(12))
    ) * adjacency - s.eye(12)
    port, row = s.eye(12)[:, 0], s.zeros(1, 12)
    row[6], row[1], row[3] = s.Rational(1, 2), -s.Rational(1, 4), -s.Rational(1, 4)
    assert s.simplify(normalized - base - eta * port * row) == s.zeros(12)
    assert s.simplify(normalized * s.ones(12, 1)) == s.zeros(12, 1)
    assert generator == normalized.subs(epsilon, s.Rational(1, 4))

    projection = s.diag(*(s.Matrix([[1, s.I]]) for _ in range(4))) * model["lift"].T
    means = model["mean_lift"].T / 3
    grams, initial_mean_rates = [], []
    for fine in model["preparations"]:
        assert means * fine == s.ones(4, 1) / 2
        z = s.simplify(projection * fine)
        grams.append(s.simplify(z * z.conjugate().T))
        initial_mean_rates.append((means * generator * fine)[0])
    assert grams[0] == grams[1] == s.ones(4) / 128
    assert initial_mean_rates == [
        model["eta"] * model["amplitude"] / 6,
        -model["eta"] * model["amplitude"] / 6,
    ]

    # The uniformly phased baseline mode is a two-dimensional real invariant
    # subspace. Check its ODE and initial condition instead of exp(12 x 12).
    cosine = s.Matrix(tuple(P) * 4)
    sine = s.Matrix(tuple(Q_MODE) * 4)
    angle = s.sqrt(3) * time / 8
    mode = s.exp(-3 * time / 8) * (
        cosine * s.cos(angle) - sine * s.sin(angle) / s.sqrt(3)
    )
    assert mode.subs(time, 0) == cosine
    assert s.simplify(s.diff(mode, time) - base * mode) == s.zeros(12, 1)
    assert s.simplify((row * mode)[0] - s.exp(-3 * time / 8) * s.cos(angle) / 2) == 0
    # A uniform zero-contrast state remains exactly uniform for either support;
    # no phase needs to be assigned and no additional trajectory is necessary.
    assert generator * (s.ones(12, 1) / 2) == s.zeros(12, 1)


@pytest.fixture(scope="module")
def imperfect_component_execution(imperfect_component_model):
    from tnfr.dynamics import integrators

    model = imperfect_component_model
    dt, steps = model["dt"], model["steps"]
    pressure_matrix = tuple(
        tuple(Fraction(int(value.p), int(value.q)) for value in row)
        for row in model["generator"].tolist()
    )
    euler = tuple(
        tuple(Fraction(i == j) + dt * value for j, value in enumerate(row))
        for i, row in enumerate(pressure_matrix)
    )
    retained = []
    with pytest.MonkeyPatch.context() as patch:
        # Bind the scalar owners explicitly; built-in no-Gamma otherwise admits
        # the integrator's NumPy path even for this small graph.
        patch.setattr(integrators, "np", None)
        for supplied in model["preparations"]:
            initial = tuple(Fraction(int(value.p), int(value.q)) for value in supplied)
            reference = [initial]
            for _ in range(steps):
                reference.append(_apply(euler, reference[-1]))
            graph = nx.from_numpy_array(
                np.asarray(model["adjacency"], dtype=float), create_using=nx.DiGraph
            )
            graph.graph.update(
                DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
                GAMMA={"type": "none"},
                use_extended_dynamics=False,
                vectorized_dnfr=False,
                DT_MIN=0.0,
                EPI_MIN=0.0,
                EPI_MAX=1.0,
                CLIP_MODE="hard",
            )
            for node, value in zip(graph, initial, strict=True):
                graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0)

            def captured(attribute):
                return tuple(
                    Fraction(get_attr(graph.nodes[node], attribute, None))
                    for node in graph
                )

            trace, defects, bounds = [initial], [], [Fraction(0)]
            for index in range(steps):
                before = captured(ALIAS_EPI)
                default_compute_delta_nfr(graph)
                pressure = captured(ALIAS_DNFR)
                pressure_defect = tuple(
                    actual - expected
                    for actual, expected in zip(
                        pressure, _apply(pressure_matrix, before), strict=True
                    )
                )
                update_epi_via_nodal_equation(
                    graph, dt=float(dt), t=float(index * dt), method="euler", n_jobs=1
                )
                after = captured(ALIAS_EPI)
                integration_defect = tuple(
                    new - old - dt * rate
                    for new, old, rate in zip(after, before, pressure, strict=True)
                )
                local_defect = tuple(
                    dt * p + step
                    for p, step in zip(pressure_defect, integration_defect, strict=True)
                )
                bounds.append(bounds[-1] + max(map(abs, local_defect)))
                defects.append((pressure_defect, integration_defect))
                trace.append(after)
            retained.append(
                {
                    "trace": tuple(trace),
                    "reference": tuple(reference),
                    "defects": tuple(defects),
                    "bounds": tuple(bounds),
                    "capacity": captured(ALIAS_VF),
                    "phase": captured(ALIAS_THETA),
                }
            )
    return euler, tuple(retained)


def test_imperfect_port_scalar_execution_retains_exact_arithmetic_defects(
    imperfect_component_model,
    imperfect_component_execution,
):
    model = imperfect_component_model
    euler, evidence = imperfect_component_execution
    assert all(min(row) >= 0 and sum(row) == 1 for row in euler)
    for run in evidence:
        initial = run["trace"][0]
        assert len(run["trace"]) == len(run["reference"]) == model["steps"] + 1
        assert len(run["defects"]) == model["steps"]
        for actual, expected, bound in zip(
            run["trace"], run["reference"], run["bounds"], strict=True
        ):
            # Exact stochastic propagation of retained local arithmetic defects
            # bounds the actual binary64 state against rational Euler execution.
            assert (
                max(abs(x - y) for x, y in zip(actual, expected, strict=True)) <= bound
            )
            assert min(initial) <= min(actual) <= max(actual) <= max(initial)
        assert run["bounds"][-1] < Fraction(1, 10**12)
        assert run["capacity"] == (1,) * 12
        assert run["phase"] == (0,) * 12


def test_reserved_imperfect_port_response_exceeds_solver_uncertainty(
    imperfect_component_model,
    imperfect_component_execution,
    record_property,
):
    model = imperfect_component_model
    _, evidence = imperfect_component_execution
    time = model["steps"] * model["dt"]
    amplitude = Fraction(int(model["amplitude"].p), int(model["amplitude"].q))
    eta = float(model["eta"])
    orbit_bound = 2 * float(amplitude) / np.sqrt(3)
    mean_model_bound = eta * float(time) * orbit_bound
    gram_model_bound = 24 * eta * float(time) * orbit_bound**2
    # G is a stochastic-generator contraction, with ||G||_inf <= 2. Thus the
    # uniform-shift-free Euler truncation bound is 2*a*T*h, per trajectory.
    truncation_bound = 2 * amplitude * time * model["dt"]
    assert time == Fraction(1, 32)
    assert truncation_bound == Fraction(1, 262144)
    lower_separation = (
        eta
        * float(amplitude)
        * float(time)
        * np.exp(-float(time))
        * np.cos(np.sqrt(3) * float(time) / 8)
        / 3
    )
    projection = np.asarray(model["lift"].T, dtype=float)

    def observations(fine):
        means = np.mean(fine.reshape(4, 3), axis=1)
        cartesian = projection @ fine
        z = cartesian[::2] + 1j * cartesian[1::2]
        return means, np.outer(z, z.conjugate())

    runtime_means = []
    for index, run in enumerate(evidence):
        endpoint = np.asarray(run["trace"][-1], dtype=float)
        continuous = model["continuous"][index]
        baseline = model["baseline"][index]
        arithmetic_bound = float(run["bounds"][-1])
        solver_bound = float(truncation_bound) + arithmetic_bound
        means, gram = observations(continuous)
        base_means, base_gram = observations(baseline)
        actual_means, actual_gram = observations(endpoint)
        runtime_means.append(sum(run["trace"][-1][:3]) / 3)
        # The matrix exponential is an independent numerical reference. These
        # finite comparisons are separate from the exact symbolic identities.
        assert np.max(np.abs(endpoint - continuous)) <= solver_bound
        assert np.max(np.abs(means - base_means)) <= mean_model_bound
        assert np.linalg.norm(gram - base_gram, "fro") <= gram_model_bound
        assert (
            np.max(np.abs(actual_means - base_means)) <= mean_model_bound + solver_bound
        )
        assert np.linalg.norm(actual_gram - base_gram, "fro") <= (
            gram_model_bound + 24 * orbit_bound * solver_bound
        )
        record_property(f"preparation_{index}_mean", str(runtime_means[-1]))
        record_property(f"preparation_{index}_arithmetic_bound", arithmetic_bound)
        record_property(
            f"preparation_{index}_max_pressure_defect",
            float(
                max(abs(value) for pressure, _ in run["defects"] for value in pressure)
            ),
        )
        record_property(
            f"preparation_{index}_max_integration_defect",
            float(
                max(
                    abs(value)
                    for _, integration in run["defects"]
                    for value in integration
                )
            ),
        )
        record_property(
            f"preparation_{index}_euler_discretization_error",
            float(
                np.max(
                    np.abs(np.asarray(run["reference"][-1], dtype=float) - continuous)
                )
            ),
        )
        record_property(
            f"preparation_{index}_continuous_mean_error",
            float(np.max(np.abs(means - base_means))),
        )
        record_property(
            f"preparation_{index}_continuous_gram_error",
            float(np.linalg.norm(gram - base_gram, "fro")),
        )
    pair_numerical_bound = 2 * float(truncation_bound) + sum(
        float(run["bounds"][-1]) for run in evidence
    )
    gap = runtime_means[0] - runtime_means[1]
    assert lower_separation > 2 * pair_numerical_bound
    assert float(gap) >= lower_separation - pair_numerical_bound
    assert float(gap) > pair_numerical_bound
    record_property("runtime_mean_gap", float(gap))
    # NumPy scalars are valid calculations but not xdist report payloads.
    record_property("continuous_gap_lower_bound", float(lower_separation))
    record_property("pair_numerical_bound", pair_numerical_bound)
    record_property("mean_model_bound", float(mean_model_bound))
    record_property("gram_model_bound", float(gram_model_bound))
