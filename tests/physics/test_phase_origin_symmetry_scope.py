"""Exact prism reflection constraints, without selecting a phase evolution law.

The tests check finite group actions and vector identities. Uniqueness and
invariant-axis consequences for smooth autonomous ODEs are analytic hypotheses,
not assertions inferred by running a trajectory. A second retained vector can
break the instantaneous form stabilizer without breaking covariance of a law.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    LIFT,
    NODES,
    PROJECTION,
    _exact_generator,
    _graph,
)
from tnfr.physics.support_transport import observe_support_transport
from tnfr.physics.symmetry_sectors import (
    automorphism_permutations,
    permutation_matrix,
    reynolds_projector,
)

s = pytest.importorskip("sympy")


def _exact_dyadic_matrix(array):
    """Retain represented entries exactly; used only for 0/1 and half entries."""
    return s.Matrix([[s.Rational(float(value)) for value in row] for row in array])


@pytest.fixture(scope="module")
def representation():
    graph = _graph()
    group = automorphism_permutations(graph, weight="weight", cap=12)
    assert len(group) == 12
    rotation = {(a, i): (a, (i + 1) % 3) for a, i in NODES}
    reflection = {(a, i): (a, 1 - i if i < 2 else i) for a, i in NODES}
    assert rotation in group and reflection in group
    fine = tuple(
        _exact_dyadic_matrix(permutation_matrix(g, NODES))
        for g in (rotation, reflection)
    )
    lift, projection = s.Matrix(LIFT), s.Matrix(PROJECTION)
    normalize = s.diag(s.sqrt(2), s.sqrt(6))
    repeat = s.Matrix([[1, 0], [0, 1], [1, 0], [0, 1]])
    normalized_lift = lift * repeat * normalize.inv()
    normalized_projection = normalize * projection[:2, :]
    actions = tuple(
        s.simplify(normalized_projection * p * normalized_lift) for p in fine
    )
    return (
        graph,
        rotation,
        reflection,
        fine,
        actions,
        normalized_lift,
        normalized_projection,
    )


def test_actual_fine_automorphisms_induce_the_same_action_on_form_and_phase(
    representation,
):
    graph, _, _, fine, (rotation, reflection), lift, projection = representation
    assert rotation == s.Matrix(
        [[-s.Rational(1, 2), -s.sqrt(3) / 2], [s.sqrt(3) / 2, -s.Rational(1, 2)]]
    )
    assert reflection == s.diag(-1, 1)
    assert rotation**3 == s.eye(2) and reflection**2 == s.eye(2)
    assert s.simplify(reflection * rotation * reflection - rotation.inv()) == s.zeros(2)
    generator = s.Matrix(_exact_generator(observe_support_transport(graph)))
    assert s.simplify(projection * lift) == s.eye(2)
    assert s.simplify(generator * lift + lift) == s.zeros(6, 2)
    a, b, common = s.symbols("a b common", real=True)
    coordinates = s.Matrix([a, b])
    eta = lift * coordinates
    # Each actual prism neighborhood sees one member of each internal index.
    for node in NODES:
        assert sorted(i for _, i in graph.neighbors(node)) == [0, 1, 2]
    phase_source = common * s.ones(6, 1) - eta / s.pi
    assert s.simplify(projection * phase_source + coordinates / s.pi) == s.zeros(2, 1)
    for permutation, action in zip(fine, (rotation, reflection), strict=True):
        assert permutation * generator == generator * permutation
        assert s.simplify(permutation * lift - lift * action) == s.zeros(6, 2)
        assert s.simplify(
            projection * permutation * phase_source - action * projection * phase_source
        ) == s.zeros(2, 1)


def test_reflection_axes_and_tangent_projectors_differ_from_zero_mean_source_lines(
    representation,
):
    graph, rotation_map, reflection_map, _, (rotation, reflection), lift, projection = (
        representation
    )
    identity = {node: node for node in NODES}
    # Conjugate each fine reflection by the actual simultaneous triangle cycle.
    power_map = identity
    for k, angle in enumerate((s.pi / 2, s.pi / 6, 5 * s.pi / 6)):
        inverse = {target: source for source, target in power_map.items()}
        mapping = {node: power_map[reflection_map[inverse[node]]] for node in NODES}
        reflected = s.simplify(rotation**k * reflection * rotation ** (-k))
        direction = s.Matrix([s.cos(angle), s.sin(angle)])
        assert s.simplify(reflected * direction - direction) == s.zeros(2, 1)
        fine_projector = _exact_dyadic_matrix(
            reynolds_projector(graph, nodes=NODES, permutations=[mapping])
        )
        fixed = s.simplify(projection * fine_projector * lift)
        assert s.simplify(fixed - (s.eye(2) + reflected) / 2) == s.zeros(2)
        assert s.simplify(fixed - direction * direction.T) == s.zeros(2)
        assert s.simplify((s.eye(2) - fixed) * direction) == s.zeros(2, 1)
        # Equivariance at a fixed point imposes (I-S)F=2(I-projector)F=0.
        assert s.simplify(s.eye(2) - reflected - 2 * (s.eye(2) - fixed)) == s.zeros(2)
        assert all(s.simplify(value) != 0 for value in (lift * direction)[:3])
        power_map = {node: rotation_map[power_map[node]] for node in NODES}
    # In the regular repeated-phase chart, c=0 exactly when an eta_i is zero.
    for angle in (0, s.pi / 3, 2 * s.pi / 3):
        eta = lift * s.Matrix([s.cos(angle), s.sin(angle)])
        assert s.simplify(s.prod(eta[:3])) == 0


def test_linear_form_only_equivariance_excludes_a_selected_quarter_turn(representation):
    _, _, _, _, (rotation, reflection), _, _ = representation
    a, b, c, d = s.symbols("a b c d", real=True)
    matrix = s.Matrix([[a, b], [c, d]])
    equations = list(matrix * rotation - rotation * matrix)
    equations += list(matrix * reflection - reflection * matrix)
    assert s.linsolve(equations, (a, b, c, d)) == s.FiniteSet((d, 0, 0, d))
    quarter_turn = s.Matrix([[0, -1], [1, 0]])
    assert rotation * quarter_turn == quarter_turn * rotation
    assert reflection * quarter_turn * reflection == -quarter_turn
    assert reflection * quarter_turn != quarter_turn * reflection


def test_equivariant_shape_only_angular_control_turns_but_vanishes_on_mirrors(
    representation,
):
    _, _, _, _, (rotation, reflection), _, _ = representation
    x, y = s.symbols("x y", real=True)
    state = s.Matrix([x, y])
    quarter_turn = s.Matrix([[0, -1], [1, 0]])

    def pseudoscalar(value):
        return value[0] * (value[0] ** 2 - 3 * value[1] ** 2)  # Re(z^3)

    def field(value):
        return pseudoscalar(value) * quarter_turn * value

    for k in range(3):
        for action in (rotation**k, rotation**k * reflection):
            assert (
                s.simplify(
                    pseudoscalar(action * state) - action.det() * pseudoscalar(state)
                )
                == 0
            )
            assert s.simplify(field(action * state) - action * field(state)) == s.zeros(
                2, 1
            )
    for angle in (s.pi / 6, s.pi / 2, 5 * s.pi / 6):
        assert s.simplify(field(s.Matrix([s.cos(angle), s.sin(angle)]))) == s.zeros(
            2, 1
        )
    assert field(s.Matrix([1, 0])) == s.Matrix([0, 1])
    # This polynomial is an algebraic control, not a selected TNFR phase law.


def test_two_vector_signed_area_can_break_the_form_axis_without_breaking_covariance(
    representation,
):
    _, _, _, _, (rotation, reflection), _, _ = representation
    x, y, a, b = s.symbols("x y a b", real=True)
    form, phase_contrast = s.Matrix([x, y]), s.Matrix([a, b])

    def area(left, right):
        return s.det(s.Matrix.hstack(left, right))

    # Declared row coefficients e=1, k=1/pi; not a normalized runtime mix.
    # No evolution law for phase_contrast is supplied by this form equation.
    def form_rate(left, right):
        return -left - right / s.pi

    for k in range(3):
        for action in (rotation**k, rotation**k * reflection):
            assert (
                s.simplify(
                    area(action * form, action * phase_contrast)
                    - action.det() * area(form, phase_contrast)
                )
                == 0
            )
            assert s.simplify(
                form_rate(action * form, action * phase_contrast)
                - action * form_rate(form, phase_contrast)
            ) == s.zeros(2, 1)
    radius = s.symbols("r", positive=True)
    on_axis, transverse = s.Matrix([0, radius]), s.Matrix([radius, 0])
    rate = form_rate(on_axis, transverse)
    assert reflection * on_axis == on_axis
    assert reflection * transverse == -transverse
    assert area(on_axis, transverse) == -(radius**2)
    assert rate == s.Matrix([-radius / s.pi, -radius])
    assert s.simplify(area(on_axis, rate) / radius**2) == 1 / s.pi
    # The full two-vector state is not reflection-fixed. The nonzero transverse
    # instantaneous rate is not a trajectory or an autonomous closure witness.
