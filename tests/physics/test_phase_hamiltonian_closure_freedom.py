"""A genuine selected symplectic chart still leaves a constitutive choice.

The existing pressure-coordinate pullback is closed on its regular centered
chart. Choosing that form restricts Hamiltonian accelerations, but does not
select a Hamiltonian or derive a phase clock. The two graph-derived potentials
below are counterchoices, not new TNFR laws. Exact local tangent controls retain
the original form, fixed support/capacity and the complete mean response; no
trajectory, persistence candidate, numerical derivative or engine update is run.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    Q_MODE,
    P,
    _prepared_pressure_coordinates,
)


@pytest.fixture(scope="module")
def chart():
    s, graph, rows, phases, center, _, lap, project, scaled_source, inverse = (
        _prepared_pressure_coordinates()
    )
    source = s.Matrix([(center - phase) / s.pi for phase in phases])
    return s, graph, rows, phases, lap, project, scaled_source, inverse, source


def test_selected_canonical_form_fixes_only_the_velocity_part_of_the_hamiltonian():
    s = pytest.importorskip("sympy")
    # Five orthonormal centered coordinates and their five pressure velocities.
    # This is the existing degree-three form, not the ambient tetrad substrate.
    q = s.Matrix(s.symbols("q0:5", real=True))
    v = s.Matrix(s.symbols("v0:5", real=True))
    coordinates = tuple(q) + tuple(v)
    arbitrary_remainder = s.Function("C")(*coordinates)
    kinetic = 3 * v.dot(v) / 2
    hamiltonian = kinetic + arbitrary_remainder
    form = 3 * s.zeros(5).row_join(s.eye(5)).col_join((-s.eye(5)).row_join(s.zeros(5)))
    gradient = s.Matrix([s.diff(hamiltonian, item) for item in coordinates])
    velocity = -form.inv() * gradient
    assert form + form.T == s.zeros(10) and form.rank() == 10
    assert all(s.diff(entry, item) == 0 for entry in form for item in coordinates)
    # Constant coefficients give d(form)=0, not merely pointwise skewness.
    # Requiring q_dot=v is precisely dC/dv=0. On connected velocity fibers,
    # the smooth remainder is therefore an arbitrary U(q), not a fixed U.
    assert velocity[:5, 0] - v == s.Matrix(
        [s.diff(arbitrary_remainder, item) / 3 for item in v]
    )
    potential = s.Function("U")(*q)
    selected = kinetic + potential
    selected_gradient = s.Matrix([s.diff(selected, item) for item in coordinates])
    selected_velocity = -form.inv() * selected_gradient
    assert selected_velocity[:5, 0] == v
    assert selected_velocity[5:, 0] == -s.Matrix(
        [s.diff(potential, item) / 3 for item in q]
    )
    assert s.expand(selected_gradient.dot(selected_velocity)) == 0


def test_two_graph_potentials_agree_on_original_form_but_not_as_local_laws(chart):
    s, graph, rows, _, lap, project, _, _, _ = chart
    stiffnesses = (lap, lap**2)
    original = s.Matrix(P * 2) / 4
    inter_triangle = s.Matrix((1, 1, 1, -1, -1, -1))
    assert project * original == original
    assert lap * original == lap**2 * original == original
    assert lap * inter_triangle == s.Rational(2, 3) * inter_triangle
    assert (lap**2 - lap) * inter_triangle == -s.Rational(2, 9) * inter_triangle
    # This is the original stored EPI, not a new prepared candidate.
    actual_form = s.Matrix([data["EPI"] for _, data in graph.nodes(data=True)])
    assert actual_form == s.ones(6, 1) / 2 + original
    assert stiffnesses[0] != stiffnesses[1]
    for stiffness in stiffnesses:
        assert stiffness == stiffness.T
        assert stiffness * s.ones(6, 1) == s.zeros(6, 1)
        assert sum(stiffness.eigenvals().values()) == 6
        assert all(eigenvalue >= 0 for eigenvalue in stiffness.eigenvals())
        assert stiffness.rank() == 5
        assert project * stiffness == stiffness * project == stiffness

    # Both use existing graph algebra, but L^2 extends dependence to two hops.
    # A separately required one-hop law could exclude it; closedness does not.
    assert 4 not in rows[0] and lap[0, 4] == 0 and (lap**2)[0, 4] != 0
    permutation = s.eye(6)[:, (3, 0, 5, 1, 4, 2)]
    relabeled = permutation.T * lap * permutation
    assert relabeled**2 == permutation.T * lap**2 * permutation
    assert lap**2 * original == lap * original
    assert (-(lap**2) + lap) * inter_triangle != s.zeros(6, 1)


def test_both_local_lifts_keep_nodal_form_rate_and_the_forced_mean_response(chart):
    s, _, _, phases, lap, project, scaled, inverse, source = chart
    assert max(phases) - min(phases) == s.pi / 3 < s.pi / 2
    assert (project * scaled * inverse - project).applyfunc(s.simplify) == s.zeros(6)
    e, w = s.symbols("e w", positive=True)
    a, b, c = s.symbols("a b c", real=True)
    mode_p = s.Matrix(P * 2)
    mode_q = s.Matrix(Q_MODE * 2)
    inter_triangle = s.Matrix((1, 1, 1, -1, -1, -1))
    # a=1/4,b=c=0 is the original form. b and c expose local derivatives;
    # no graph is prepared with either tangent displacement.
    y = a * mode_p + b * mode_q + c * inter_triangle
    original_point = {a: s.Rational(1, 4), b: 0, c: 0}
    pressure = -e * lap * y + w * source
    velocity = project * pressure
    assert s.simplify(sum(pressure)) == 0  # Instantaneously, not an invariant.
    phase_velocities = []
    accelerations = []
    for stiffness in (lap, lap**2):
        target = -stiffness * y
        phase_velocity = (s.pi * inverse * (target + e * lap * velocity) / w).applyfunc(
            s.simplify
        )
        acceleration = (
            -e * lap * velocity + w * scaled * phase_velocity / s.pi
        ).applyfunc(s.simplify)
        assert (project * acceleration - target).applyfunc(s.simplify) == s.zeros(6, 1)
        mean_acceleration = sum(acceleration) / 6
        expected_mean = b * (1 + e**2) * (s.sqrt(3) - 2) / (1 + s.sqrt(3))
        assert s.simplify(mean_acceleration - expected_mean) == 0
        assert s.simplify(s.diff(mean_acceleration, b)) != 0
        assert s.simplify(mean_acceleration.subs(original_point)) == 0
        # H=3||v||^2/2+3*y^T*stiffness*y/2 conserves its own value under
        # this selected acceleration, not the earlier passive storage.
        energy_rate = 3 * velocity.dot(project * acceleration + stiffness * y)
        assert s.simplify(energy_rate) == 0
        assert s.simplify(sum(phase_velocity)) == 0  # Selected gauge only.
        phase_velocities.append(phase_velocity)
        accelerations.append(acceleration)

    phase_difference = (phase_velocities[1] - phase_velocities[0]).applyfunc(s.simplify)
    assert phase_difference.subs(original_point) == s.zeros(6, 1)
    phase_tangent_difference = phase_difference.diff(c)
    assert phase_tangent_difference != s.zeros(6, 1)
    assert (w * project * scaled * phase_tangent_difference / s.pi).applyfunc(
        s.simplify
    ) == s.Rational(2, 9) * inter_triangle
    acceleration_difference = (accelerations[1] - accelerations[0]).applyfunc(
        s.simplify
    )
    assert acceleration_difference.subs(original_point) == s.zeros(6, 1)
    assert acceleration_difference.diff(c) == s.Rational(2, 9) * inter_triangle
    # A common phase rotation is a free lift direction and changes neither
    # source acceleration nor the required EPI mean. It is not a derived clock.
    assert scaled * s.ones(6, 1) == s.zeros(6, 1)
