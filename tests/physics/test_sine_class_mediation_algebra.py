"""Independent fine-graph algebra for class-dependent C9 mediation.

Only static matrices, derivatives and work identities are evaluated. These
controls execute no formed-state assessor, response trajectory or frozen driver.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.physics._sine_port_geometry import (
    _central_port_geometry,
    _component_interior_matrix,
)

DONOR, MEDIATOR, RECEIVER = 4, 13, 22
CONTACTS = ((DONOR, MEDIATOR), (MEDIATOR, RECEIVER))


def _mv(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _add(first, second, scale=Q(1)):
    return tuple(a + scale * b for a, b in zip(first, second))


def _basis(index, size=27):
    return tuple(Q(i == index) for i in range(size))


def _normalized_laplacian(edges, degrees):
    matrix = [[Q(0) for _ in degrees] for _ in degrees]
    for left, right in edges:
        for node, neighbor in ((left, right), (right, left)):
            matrix[node][node] += Q(1, degrees[node])
            matrix[node][neighbor] -= Q(1, degrees[node])
    return tuple(map(tuple, matrix))


@pytest.fixture(scope="module")
def graph():
    cycles = tuple(
        tuple((9 * part + j, 9 * part + (j + 1) % 9) for j in range(9))
        for part in range(3)
    )
    edges = sum(cycles, ()) + CONTACTS
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))
    return dict(
        cycles=cycles,
        edges=edges,
        degrees=degrees,
        A=_normalized_laplacian(edges, degrees),
        internal=tuple(_normalized_laplacian(part, degrees) for part in cycles),
        bridge=_normalized_laplacian(CONTACTS, degrees),
    )


def _phase_matrix(graph, mediator_cosine, outer_cosine=Q(3, 4)):
    donor, middle, receiver = graph["internal"]
    return tuple(
        tuple(
            outer_cosine * (donor[i][j] + receiver[i][j])
            + mediator_cosine * middle[i][j]
            + graph["bridge"][i][j]
            for j in range(27)
        )
        for i in range(27)
    )


def _tangent_action(graph, cosine, gamma, state):
    form, phase = state[:27], state[27:]
    ax = _mv(graph["A"], form)
    cy = _mv(_phase_matrix(graph, cosine), phase)
    return tuple(-a - gamma * c for a, c in zip(ax, cy)) + tuple(gamma * a for a in ax)


def _lift(vector):
    return tuple(vector[5 * part + abs(j - 4)] for part in range(3) for j in range(9))


def test_fine_support_actual_degrees_and_exact_even_lift(graph):
    assert len(graph["edges"]) == 29
    assert sum(graph["degrees"]) == 58
    assert tuple(graph["degrees"][i] for i in (DONOR, MEDIATOR, RECEIVER)) == (3, 4, 3)
    reduced = _central_port_geometry(3, ((0, 1), (1, 2)))
    for index in range(15):
        vector = _basis(index, 15)
        assert _mv(graph["A"], _lift(vector)) == _lift(
            _mv(reduced.normalized_form_matrix, vector)
        )
        for part in range(3):
            assert _mv(graph["internal"][part], _lift(vector)) == _lift(
                _mv(_component_interior_matrix(reduced, part), vector)
            )
    # Reusing degree three at the middle port changes its actual charge balance.
    wrong_degrees = tuple(
        3 if i == MEDIATOR else d for i, d in enumerate(graph["degrees"])
    )
    wrong = _normalized_laplacian(graph["edges"], wrong_degrees)
    rate = tuple(-x for x in _mv(wrong, _basis(MEDIATOR)))
    assert sum(d * value for d, value in zip(graph["degrees"], rate)) == Q(-4, 3)


def test_complete_walk_coefficient_and_earlier_class_blind_terms(graph):
    a = graph["A"]
    middle = graph["internal"][1]
    donor = _basis(DONOR)
    ad = _mv(a, donor)
    mad = _mv(middle, ad)
    assert ad[MEDIATOR] == Q(-1, 4)
    assert all(ad[j] == 0 for j in (MEDIATOR - 1, MEDIATOR + 1))
    assert mad[MEDIATOR] == Q(-1, 8)
    assert all(value == 0 for value in middle[RECEIVER])
    assert _mv(a, mad)[RECEIVER] == Q(1, 24)
    assert _mv(a, ad)[RECEIVER] == Q(1, 12)
    reduced = _central_port_geometry(3, ((0, 1), (1, 2)))
    reduced_ad = _mv(reduced.normalized_form_matrix, _basis(0, 15))
    reduced_mad = _mv(_component_interior_matrix(reduced, 1), reduced_ad)
    assert _mv(reduced.normalized_form_matrix, reduced_mad)[10] == Q(1, 24)


@pytest.mark.parametrize("gamma,amplitude", [(Q(1, 7), Q(2, 11)), (Q(1, 13), Q(3, 17))])
def test_formal_full54_tangent_first_contrast_is_third_derivative(
    graph, gamma, amplitude
):
    cosines = (Q(3, 4), Q(1, 4))
    states = [tuple(amplitude * x for x in _basis(DONOR)) + (Q(0),) * 27] * 2
    for derivative in range(4):
        difference = states[0][RECEIVER] - states[1][RECEIVER]
        expected = amplitude * gamma**2 * (cosines[0] - cosines[1]) / 24
        assert difference == (expected if derivative == 3 else 0)
        if derivative == 2:
            assert (
                states[0][RECEIVER]
                == states[1][RECEIVER]
                == amplitude * (1 - gamma**2) / 12
            )
        states = [
            _tangent_action(graph, cosine, gamma, state)
            for cosine, state in zip(cosines, states)
        ]


def test_scaled_generator_telescoping_and_tail_coefficients(graph):
    eta, first, second = Q(1, 49), Q(3, 4), Q(1, 4)
    a = graph["A"]

    def matrix(cosine):
        c = _phase_matrix(graph, cosine)
        return tuple(
            tuple(-v for v in row) + tuple(-eta * v for v in crow)
            for row, crow in zip(a, c)
        ) + tuple(row + (Q(0),) * 27 for row in a)

    def norm(matrix):
        return max(sum(abs(v) for v in row) for row in matrix)

    m1, m2 = matrix(first), matrix(second)
    difference = tuple(tuple(x - y for x, y in zip(r1, r2)) for r1, r2 in zip(m1, m2))
    assert max(norm(m1), norm(m2)) <= 3
    assert norm(difference) <= 2 * eta * (first - second)
    initial = _basis(DONOR) + (Q(0),) * 27
    powers1, powers2 = [initial], [initial]
    for order in range(1, 8):
        powers1.append(_mv(m1, powers1[-1]))
        powers2.append(_mv(m2, powers2[-1]))
        telescoped = [Q(0)] * 54
        for right_power in range(order):
            term = _mv(difference, powers2[right_power])
            for _ in range(order - right_power - 1):
                term = _mv(m1, term)
            telescoped = _add(telescoped, term)
        actual = _add(powers1[order], powers2[order], Q(-1))
        assert tuple(telescoped) == actual
        assert max(map(abs, actual)) <= 2 * eta * (first - second) * order * 3 ** (
            order - 1
        )
    # The n=4 exponential-series term and all subsequent ratio bounds are
    # algebraic in h: no finite response or chosen certificate is evaluated.
    assert Q(2 * 4 * 3**3, 24) == 9
    for order in range(4, 12):
        assert Q(3, order) <= Q(3, 4)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 75
    return context


def _mp_fraction(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _edge_sine_jet(graph, phase, direction, second_direction, mp):
    field, first, second = ([mp.mpf(0) for _ in range(27)] for _ in range(3))
    for left, right in graph["edges"]:
        gap = phase[right] - phase[left]
        speed = direction[right] - direction[left]
        acceleration = second_direction[right] - second_direction[left]
        values = (
            mp.sin(gap),
            mp.cos(gap) * speed,
            mp.cos(gap) * acceleration - mp.sin(gap) * speed**2,
        )
        for row, value in zip((field, first, second), values):
            row[left] += value / graph["degrees"][left]
            row[right] -= value / graph["degrees"][right]
    return tuple(field), tuple(first), tuple(second)


def test_complete_nonlinear_edge_derivatives_retain_the_same_leading_contrast(
    graph, mp
):
    # These are off-protocol formal amplitude/rate values, and instantaneous
    # derivatives only. No finite trajectory or selected horizon is evaluated.
    gamma, amplitude = mp.mpf(1) / 13, mp.mpf(2) / 17
    a = tuple(tuple(_mp_fraction(mp, x) for x in row) for row in graph["A"])
    x = tuple(amplitude * int(i == DONOR) for i in range(27))
    phase_velocity = tuple(gamma * value for value in _mv(a, x))
    third = []
    for mediator_class in (1, 2):
        classes = (1, mediator_class, 1)
        phase = tuple(
            2 * mp.pi * classes[part] * (j - 4) / 9
            for part in range(3)
            for j in range(9)
        )
        zero = (mp.mpf(0),) * 27
        f0, df0, _ = _edge_sine_jet(graph, phase, phase_velocity, zero, mp)
        assert max(map(abs, f0)) < mp.mpf("1e-70")
        x1 = _add(tuple(-value for value in _mv(a, x)), f0, gamma)
        phase_acceleration = tuple(gamma * value for value in _mv(a, x1))
        _, _, ddf0 = _edge_sine_jet(
            graph, phase, phase_velocity, phase_acceleration, mp
        )
        x2 = _add(tuple(-value for value in _mv(a, x1)), df0, gamma)
        x3 = _add(tuple(-value for value in _mv(a, x2)), ddf0, gamma)
        assert abs(x2[RECEIVER] - amplitude * (1 - gamma**2) / 12) < mp.mpf("1e-70")
        third.append(x3[RECEIVER])
        # The genuinely nonlinear Hessian term vanishes at the receiver:
        # its interior is initially motionless and its bridge gap is zero.
        _, _, hessian = _edge_sine_jet(graph, phase, phase_velocity, zero, mp)
        assert abs(hessian[RECEIVER]) < mp.mpf("1e-70")
    expected = (
        amplitude * gamma**2 * (mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9)) / 24
    )
    assert abs(third[0] - third[1] - expected) < mp.mpf("1e-70")
    assert expected > 0


def test_probe_work_quotient_and_actual_weighted_mean(graph):
    eps, amplitude = Q(1, 101), Q(1, 37)
    base = (2, -1, 0, 1, 3, -2, -1, -1, -1)
    source = tuple(
        eps * Q(base[(j + part) % 9], 5) for part in range(3) for j in range(9)
    )
    assert all(sum(source[9 * part : 9 * part + 9]) == 0 for part in range(3))
    assert all(
        sum(x * x for x in source[9 * part : 9 * part + 9]) <= eps**2
        for part in range(3)
    )

    def kinetic(edges, state):
        return sum((state[i] - state[j]) ** 2 / 2 for i, j in edges)

    joined, disconnected = kinetic(graph["edges"], source), kinetic(
        sum(graph["cycles"], ()), source
    )
    assert joined - disconnected == kinetic(CONTACTS, source)
    assert 0 <= joined - disconnected <= 4 * eps**2
    after = _add(source, _basis(DONOR), amplitude)
    lx_d = graph["degrees"][DONOR] * _mv(graph["A"], source)[DONOR]
    work = kinetic(graph["edges"], after) - joined
    assert work == amplitude * lx_d + Q(3, 2) * amplitude**2
    assert abs(work - Q(3, 2) * amplitude**2) <= 6 * amplitude * eps
    before_mean = sum(d * x for d, x in zip(graph["degrees"], source)) / 58
    after_mean = sum(d * x for d, x in zip(graph["degrees"], after)) / 58
    assert before_mean == (source[DONOR] + 2 * source[MEDIATOR] + source[RECEIVER]) / 58
    assert abs(before_mean) <= 4 * eps / 58
    assert after_mean - before_mean == 3 * amplitude / 58
    quotient = tuple(value - sum(after) / 27 for value in after)
    assert (
        sum(x * x for x in quotient)
        <= 3 * eps**2 + 2 * amplitude * eps + Q(26, 27) * amplitude**2
    )
    assert Q(1, 2) * Q(2, 135) * Q(1, 20) == Q(1, 2700)


def test_mediator_reflection_preserves_ports_and_complete_edge_law(graph, mp):
    reflection = tuple(9 + 8 - (i - 9) if 9 <= i < 18 else i for i in range(27))
    assert reflection[MEDIATOR] == MEDIATOR
    assert reflection[DONOR] == DONOR and reflection[RECEIVER] == RECEIVER
    assert {
        tuple(sorted((reflection[i], reflection[j]))) for i, j in graph["edges"]
    } == {tuple(sorted(edge)) for edge in graph["edges"]}
    # An arbitrary perturbed full state is reflected too; merely changing the
    # nominal winding while keeping an unrelated residual would not suffice.
    phase = tuple(mp.mpf((i * i + 3) % 17) / 11 for i in range(27))
    zero = (mp.mpf(0),) * 27
    field = _edge_sine_jet(graph, phase, zero, zero, mp)[0]
    reflected = _edge_sine_jet(
        graph, tuple(phase[i] for i in reflection), zero, zero, mp
    )[0]
    assert max(abs(reflected[i] - field[reflection[i]]) for i in range(27)) < mp.mpf(
        "1e-70"
    )
    assert _mv(graph["A"], _basis(DONOR)) == tuple(
        _mv(graph["A"], _basis(DONOR))[i] for i in reflection
    )


def test_heat_and_zero_probe_have_no_class_dependent_derivative(graph):
    # In the complete phase-blind alternative, form evolves under the SAME A.
    # Paired baselines cancel even when the two actual source states differ.
    baselines = [
        tuple(Q((i + 1) ** 2, 101) for i in range(27)),
        tuple(Q((-1) ** i * (i + 3), 89) for i in range(27)),
    ]
    kick = tuple(Q(2, 19) * x for x in _basis(DONOR))
    probed = [_add(source, kick) for source in baselines]
    response = kick
    zero = (Q(0),) * 54
    for _ in range(6):
        assert _tangent_action(graph, Q(1, 4), Q(1, 7), zero) == zero
        for source, with_probe in zip(baselines, probed):
            assert _add(with_probe, source, Q(-1)) == response
        baselines = [tuple(-x for x in _mv(graph["A"], source)) for source in baselines]
        probed = [tuple(-x for x in _mv(graph["A"], source)) for source in probed]
        response = tuple(-x for x in _mv(graph["A"], response))
        assert all(isinstance(x, Q) for x in response)
