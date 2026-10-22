"""Independent full-state controls for the acquired mediator's memory law.

Static graph algebra, finite jets and response-free error inequalities are used.
No formation assessment, retained producer or reserved response is executed.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.mathematics.linear_observation import derive_coordinate_memory
from tnfr.physics.relational_sine_class_memory import derive_sine_class_mediated_memory

VISIBLE_NODES = tuple(range(9)) + tuple(range(18, 27))
HIDDEN_NODES = tuple(range(9, 18))
VISIBLE = VISIBLE_NODES + tuple(i + 27 for i in VISIBLE_NODES)
HIDDEN = HIDDEN_NODES + tuple(i + 27 for i in HIDDEN_NODES)


def _mv(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _mm(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _plus(first, second, factor=Q(1)):
    return tuple(a + factor * b for a, b in zip(first, second))


def _block(matrix, rows, columns):
    return tuple(tuple(matrix[i][j] for j in columns) for i in rows)


def _norm(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


@pytest.fixture(scope="module")
def geometry():
    cycles = tuple(
        tuple((9 * part + j, 9 * part + (j + 1) % 9) for j in range(9))
        for part in range(3)
    )
    contacts = ((4, 13), (13, 22))
    edges = sum(cycles, ()) + contacts
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))

    def laplacian(edges):
        result = [[Q(0) for _ in range(27)] for _ in range(27)]
        for left, right in edges:
            for node, neighbor in ((left, right), (right, left)):
                result[node][node] += Q(1, degrees[node])
                result[node][neighbor] -= Q(1, degrees[node])
        return tuple(map(tuple, result))

    return dict(
        edges=edges,
        degrees=degrees,
        A=laplacian(edges),
        internal=tuple(laplacian(part) for part in cycles),
        contact=laplacian(contacts),
    )


def _generator(geometry, gamma, outer, middle):
    a = geometry["A"]
    donor, mediator, receiver = geometry["internal"]
    c = tuple(
        tuple(
            outer * (donor[i][j] + receiver[i][j])
            + middle * mediator[i][j]
            + geometry["contact"][i][j]
            for j in range(27)
        )
        for i in range(27)
    )
    return tuple(
        tuple(-value for value in row) + tuple(-gamma * value for value in crow)
        for row, crow in zip(a, c)
    ) + tuple(tuple(gamma * value for value in row) + (Q(0),) * 27 for row in a)


@pytest.fixture(scope="module")
def structure():
    return derive_sine_class_mediated_memory()


def test_original54_partition_matches_independent_fine_edges(geometry, structure):
    assert structure.visible_indices == VISIBLE
    assert structure.hidden_indices == HIDDEN
    assert len(VISIBLE) == 36 and len(HIDDEN) == 18
    assert set(VISIBLE).isdisjoint(HIDDEN)
    assert set(VISIBLE) | set(HIDDEN) == set(range(54))
    a = geometry["A"]
    for actual, rows, columns in (
        (structure.normalized_laplacian_vv, VISIBLE_NODES, VISIBLE_NODES),
        (structure.normalized_laplacian_vh, VISIBLE_NODES, HIDDEN_NODES),
        (structure.normalized_laplacian_hv, HIDDEN_NODES, VISIBLE_NODES),
        (structure.normalized_laplacian_hh, HIDDEN_NODES, HIDDEN_NODES),
    ):
        assert actual == _block(a, rows, columns)
    assert structure.mediator_phase_interior_hh == _block(
        geometry["internal"][1], HIDDEN_NODES, HIDDEN_NODES
    )
    assert structure.visible_phase_contact_vv == _block(
        geometry["contact"], VISIBLE_NODES, VISIBLE_NODES
    )
    assert structure.hidden_phase_contact_hh == _block(
        geometry["contact"], HIDDEN_NODES, HIDDEN_NODES
    )
    assert structure.degrees == geometry["degrees"]


@pytest.mark.parametrize(
    "gamma,outer,first,second",
    [(Q(1, 7), Q(3, 4), Q(2, 3), Q(1, 5)), (Q(1, 11), Q(4, 5), Q(3, 5), Q(1, 4))],
)
def test_full_channel_kernel_jets_and_exact_static_midpoint(
    geometry, structure, gamma, outer, first, second
):
    memories = [
        derive_coordinate_memory(_generator(geometry, gamma, outer, c), VISIBLE)
        for c in (first, second)
    ]
    initial, changed = memories
    assert initial.visible_generator == changed.visible_generator
    assert initial.hidden_to_visible == changed.hidden_to_visible
    assert initial.visible_to_hidden == changed.visible_to_hidden
    assert initial.kernel_at_zero == changed.kernel_at_zero
    product = _mm(
        _block(geometry["A"], VISIBLE_NODES, HIDDEN_NODES),
        _block(geometry["A"], HIDDEN_NODES, VISIBLE_NODES),
    )
    assert product == structure.kernel_at_zero_spatial_matrix
    factors = ((1 - gamma**2, gamma), (-gamma, -(gamma**2)))
    expected_zero = tuple(
        tuple(factors[i // 18][j // 18] * product[i % 18][j % 18] for j in range(36))
        for i in range(36)
    )
    assert initial.kernel_at_zero == expected_zero
    derivatives = [
        _mm(_mm(m.hidden_to_visible, m.hidden_generator), m.visible_to_hidden)
        for m in memories
    ]
    delta = tuple(_plus(a, b, Q(-1)) for a, b in zip(*derivatives))
    spatial = structure.kernel_derivative_difference_spatial_matrix
    assert spatial[13][4] == Q(1, 24)
    scalar = (
        (gamma**2 * (first - second), Q(0)),
        (-(gamma**3) * (first - second), Q(0)),
    )
    assert delta == tuple(
        tuple(scalar[i // 18][j // 18] * spatial[i % 18][j % 18] for j in range(36))
        for i in range(36)
    )
    midpoint = tuple(
        tuple(
            Q(1, 2) if (i < 9 and j in (4, 13)) or (i >= 9 and j in (22, 31)) else Q(0)
            for j in range(36)
        )
        for i in range(18)
    )
    static = []
    for memory in memories:
        assert exact_rank(memory.hidden_generator) == 18
        residual = tuple(
            _plus(row, other)
            for row, other in zip(
                _mm(memory.hidden_generator, midpoint), memory.visible_to_hidden
            )
        )
        assert not any(value for row in residual for value in row)
        static.append(
            tuple(
                _plus(row, other)
                for row, other in zip(
                    memory.visible_generator, _mm(memory.hidden_to_visible, midpoint)
                )
            )
        )
    assert static[0] == static[1]
    assert structure.hidden_midpoint_map == tuple(row[:18] for row in midpoint[:9])
    expected_form = tuple(
        _plus(row, other)
        for row, other in zip(
            structure.normalized_laplacian_vv,
            _mm(structure.normalized_laplacian_vh, structure.hidden_midpoint_map),
        )
    )
    assert expected_form == structure.quasistatic_form_matrix
    # The retained port mobility is 1/3: a half-strength effective contact
    # therefore contributes -1/6, not a re-normalized degree 1/(2+1/2).
    assert expected_form[13][4] == Q(-1, 6)


def test_memory_derivative_recurrence_keeps_hidden_source_and_paired_cancellation(
    geometry,
):
    generator = _generator(geometry, Q(1, 9), Q(3, 4), Q(2, 5))
    memory = derive_coordinate_memory(generator, VISIBLE)
    source = tuple(Q(((i + 2) ** 2 % 19) - 9, 137) for i in range(54))
    kick = tuple(Q(2, 23) if i == 4 else Q(0) for i in range(54))

    def visible_jets(initial):
        full, visible, hidden_source = (
            initial,
            [tuple(initial[i] for i in VISIBLE)],
            tuple(initial[i] for i in HIDDEN),
        )
        kernel = memory.visible_to_hidden
        kernels = []
        for order in range(6):
            expected = _plus(
                _mv(memory.visible_generator, visible[-1]),
                _mv(memory.hidden_to_visible, hidden_source),
            )
            for j in range(order):
                expected = _plus(expected, _mv(kernels[j], visible[order - 1 - j]))
            full = _mv(generator, full)
            assert expected == tuple(full[i] for i in VISIBLE)
            visible.append(expected)
            kernels.append(_mm(memory.hidden_to_visible, kernel))
            kernel = _mm(memory.hidden_generator, kernel)
            hidden_source = _mv(memory.hidden_generator, hidden_source)
        return visible

    baseline, probed, ideal = (
        visible_jets(value) for value in (source, _plus(source, kick), kick)
    )
    assert any(_mv(memory.hidden_to_visible, tuple(source[i] for i in HIDDEN)))
    assert all(_plus(p, b, Q(-1)) == i for p, b, i in zip(probed, baseline, ideal))


def test_odd_hidden_modes_do_not_justify_a_minimal18_claim(geometry):
    memory = derive_coordinate_memory(
        _generator(geometry, Q(1, 8), Q(3, 4), Q(1, 5)), VISIBLE
    )
    reflection = tuple(8 - i if i < 9 else 26 - i for i in range(18))
    for column in range(18):
        reflected_column = tuple(
            memory.hidden_generator[i][reflection[column]] for i in range(18)
        )
        assert reflected_column == tuple(
            memory.hidden_generator[reflection[i]][column] for i in range(18)
        )
    assert all(
        tuple(row[i] for i in reflection) == row for row in memory.hidden_to_visible
    )
    for channel in range(2):
        for local in range(4):
            odd = tuple(
                Q(i == 9 * channel + local) - Q(i == 9 * channel + 8 - local)
                for i in range(18)
            )
            assert not any(_mv(memory.hidden_to_visible, odd))
            derivative = _mv(memory.hidden_generator, odd)
            assert tuple(derivative[i] for i in reflection) == tuple(
                -x for x in derivative
            )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 70
    return context


def _full_rows(geometry, x, phase, gamma, mp):
    ax = tuple(
        sum(
            mp.mpf(value.numerator) / value.denominator * xj
            for value, xj in zip(row, x)
        )
        for row in geometry["A"]
    )
    sine = [mp.mpf(0)] * 27
    for i, j in geometry["edges"]:
        current = mp.sin(phase[j] - phase[i])
        sine[i] += current / geometry["degrees"][i]
        sine[j] -= current / geometry["degrees"][j]
    return tuple(-v + gamma * f for v, f in zip(ax, sine)), tuple(gamma * v for v in ax)


def _second_rows(geometry, x, phase, gamma, mp):
    dx, dy = _full_rows(geometry, x, phase, gamma, mp)
    adx = tuple(
        sum(
            mp.mpf(value.numerator) / value.denominator * xj
            for value, xj in zip(row, dx)
        )
        for row in geometry["A"]
    )
    phase_field_rate = [mp.mpf(0)] * 27
    for i, j in geometry["edges"]:
        current = mp.cos(phase[j] - phase[i]) * (dy[j] - dy[i])
        phase_field_rate[i] += current / geometry["degrees"][i]
        phase_field_rate[j] -= current / geometry["degrees"][j]
    return tuple(-v + gamma * f for v, f in zip(adx, phase_field_rate)), tuple(
        gamma * v for v in adx
    )


def test_nonlinear_midpoint_is_stationary_classblind_and_not_an_invariant_branch(
    geometry, mp
):
    gamma = mp.mpf(1) / 17
    visible_x = tuple(mp.mpf((i + 1) % 7) / 100 for i in VISIBLE_NODES)
    visible_y = tuple(mp.mpf((2 * i + 3) % 11) / 1000 for i in VISIBLE_NODES)
    form_midpoint = (visible_x[4] + visible_x[13]) / 2
    phase_midpoint = (visible_y[4] + visible_y[13]) / 2
    visible_rows = []
    for k in (1, 2):
        x, phase = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
        for position, node in enumerate(VISIBLE_NODES):
            x[node] = visible_x[position]
            phase[node] = 2 * mp.pi * (node % 9 - 4) / 9 + visible_y[position]
        for node in HIDDEN_NODES:
            x[node] = form_midpoint
            phase[node] = 2 * mp.pi * k * (node % 9 - 4) / 9 + phase_midpoint
        dx, dy = _full_rows(geometry, x, phase, gamma, mp)
        assert max(abs(row[i]) for row in (dx, dy) for i in HIDDEN_NODES) < mp.mpf(
            "1e-65"
        )
        visible_rows.append(
            tuple(dx[i] for i in VISIBLE_NODES) + tuple(dy[i] for i in VISIBLE_NODES)
        )
        assert max(abs((row[4] + row[22]) / 2) for row in (dx, dy)) > mp.mpf("1e-7")
    assert max(abs(a - b) for a, b in zip(*visible_rows)) < mp.mpf("1e-65")


def test_same_charge_hidden_perturbation_changes_visible_acceleration(geometry, mp):
    gamma, perturbation = mp.mpf(1) / 13, mp.mpf(1) / 1000
    x = tuple(mp.mpf((i * 3 + 2) % 11) / 200 for i in range(27))
    phase = tuple(
        2 * mp.pi * (2 if 9 <= i < 18 else 1) * (i % 9 - 4) / 9
        + mp.mpf((i * 2) % 7) / 1000
        for i in range(27)
    )
    changed = list(x)
    changed[12] += perturbation
    changed[11] -= perturbation
    assert geometry["degrees"][11] == geometry["degrees"][12] == 2
    assert changed[13] == x[13]
    first, second = _full_rows(geometry, x, phase, gamma, mp), _full_rows(
        geometry, changed, phase, gamma, mp
    )
    assert max(
        abs(b[i] - a[i]) for a, b in zip(first, second) for i in VISIBLE_NODES
    ) < mp.mpf("1e-65")
    initial_second, changed_second = _second_rows(
        geometry, x, phase, gamma, mp
    ), _second_rows(geometry, changed, phase, gamma, mp)
    form_gap = changed_second[0][22] - initial_second[0][22]
    phase_gap = changed_second[1][22] - initial_second[1][22]
    assert abs(
        form_gap - perturbation * (1 - gamma**2 * mp.cos(phase[13] - phase[22])) / 12
    ) < mp.mpf("1e-65")
    assert abs(phase_gap + gamma * perturbation / 12) < mp.mpf("1e-65")
    assert form_gap > 0


def test_response_free_transfer_bound_preserves_the_predeclared_contrast():
    # These are analytic upper bounds only, not an evaluated source/response.
    g, a, h, eps, delta = (
        Q(1, 3000),
        Q(1, 1000),
        Q(1, 10000),
        Q(1, 10**32),
        Q(1, 10**30),
    )
    denominator = 1 - 2 * g**2 * h**2

    def error(amplitude):
        q = (amplitude + eps + 2 * g * h * eps) / denominator
        value = (
            2 * g * eps**2 * h
            + 4 * g**2 * eps * q * h**2
            + Q(8, 3) * g**3 * q**2 * h**3
        ) / (1 - 3 * h)
        return q, value

    qa, ea = error(a)
    q0, e0 = error(Q(0))
    assert denominator > Q(999, 1000)
    assert qa < Q(101, 100) * a and q0 < 2 * eps
    assert 2 * (ea + e0) < Q(21, 10**29)
    p_lower = Q(3, 10**25)
    assert 2 * (ea + e0) < p_lower / 1000
    assert 1296 * h / (1 - 3 * h / 4) < Q(13, 100)
    recorded_lower = Q(869, 1000) * p_lower - 4 * delta
    static_upper = 4 * eps / (1 - 3 * h) + 4 * delta
    assert recorded_lower > Q(26, 10**26) > Q(1, 10**25)
    assert static_upper < Q(5, 10**30) < recorded_lower
