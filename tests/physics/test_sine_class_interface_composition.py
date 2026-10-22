"""Independent component rows, feedback defects and boundary balances.

Only static edge algebra and exact scalar comparison inequalities are used.
No kernel exponential, coefficient generation or scientific response is run.
"""

from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_class_interface_composition as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _graph():
    rings = tuple(
        tuple((9 * r + j, 9 * r + (j + 1) % 9) for j in range(9)) for r in range(3)
    )
    bridges = ((4, 13), (13, 22))
    edges = sum(rings, ()) + bridges
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    laplacian = [[Q(0)] * 27 for _ in range(27)]
    for i, j in edges:
        laplacian[i][i] += 1
        laplacian[j][j] += 1
        laplacian[i][j] -= 1
        laplacian[j][i] -= 1
    return rings, bridges, degrees, tuple(map(tuple, laplacian))


def _mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Q(0))


@pytest.fixture(scope="module", params=(1, 2))
def composition(request):
    return owner._derive_interface_composition(request.param)


def test_internal_partitions_use_joined_degrees_and_recover_full_spatial_rows(
    composition,
):
    rings, _, degrees, laplacian = _graph()
    rebuilt = [[Q(0)] * 27 for _ in range(27)]
    assert composition.central_degrees == (3, 4, 3)
    for r, component in enumerate(composition.components):
        nodes = component.node_indices
        assert nodes == tuple(range(9 * r, 9 * r + 9))
        assert component.joined_degrees == tuple(degrees[i] for i in nodes)
        local = [[Q(0)] * 9 for _ in range(9)]
        for i, j in rings[r]:
            for target, other in ((i, j), (j, i)):
                local[target % 9][target % 9] += Q(1, degrees[target])
                local[target % 9][other % 9] -= Q(1, degrees[target])
        assert component.spatial_partition.generator == tuple(map(tuple, local))
        assert component.spatial_partition.visible_generator == (
            (Q(2, degrees[nodes[4]]),),
        )
        assert len(component.hidden_state_indices) == 16
        assert set(component.full_state_indices) == set(
            nodes + tuple(i + 27 for i in nodes)
        )
        for i, gi in enumerate(nodes):
            for j, gj in enumerate(nodes):
                rebuilt[gi][gj] += local[i][j]
    for i, gi in enumerate((4, 13, 22)):
        for j, gj in enumerate((4, 13, 22)):
            rebuilt[gi][gj] += composition.normalized_contact_laplacian[i][j]
    assert tuple(map(tuple, rebuilt)) == tuple(
        tuple(value / degrees[i] for value in row) for i, row in enumerate(laplacian)
    )
    assert composition.contact_laplacian == ((1, -1, 0), (-1, 2, -1), (0, -1, 1))
    assert sum(degrees) == 58


def test_two_channel_visible_blocks_enclose_independent_full_generator(composition):
    mp = mpmath.mp.clone()
    mp.dps = 70
    gamma = 1 / (1023 * mp.pi)
    _, _, degrees, laplacian = _graph()
    ports = (4, 13, 22)
    classes = (1, composition.collective.mediator_class, 1)
    for i in range(6):
        for j in range(6):
            row, col = ports[i % 3], ports[j % 3]
            if j >= 3 and i >= 3:
                exact = mp.mpf(0)
            elif j < 3:
                spatial = mp.mpf(int(laplacian[row][col])) / degrees[row]
                exact = -spatial if i < 3 else gamma * spatial
            else:
                if row == col:
                    weighted = 2 * mp.cos(2 * classes[i] * mp.pi / 9) + degrees[row] - 2
                else:
                    weighted = int(laplacian[row][col])
                exact = -gamma * weighted / degrees[row]
            band = composition.recomposed_visible_generator_bounds[i][j]
            assert mp.mpf(band.lo.numerator) / band.lo.denominator <= exact
            assert exact <= mp.mpf(band.hi.numerator) / band.hi.denominator


def test_nonlinear_component_and_bridge_rows_recover_full_edges_and_storage(
    composition,
):
    mp = mpmath.mp.clone()
    mp.dps = 70
    gamma = 1 / (1023 * mp.pi)
    rings, bridges, degrees, laplacian = _graph()
    x = tuple(mp.mpf((i * 7) % 17 - 8) / 19 for i in range(27))
    theta = tuple(mp.mpf((i * 11) % 23 - 11) / 13 for i in range(27))
    direct_x, direct_y = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    for i, j in sum(rings, ()) + bridges:
        current = x[j] - x[i] + gamma * mp.sin(theta[j] - theta[i])
        phase_current = gamma * (x[i] - x[j])
        for row, value in ((direct_x, current), (direct_y, phase_current)):
            row[i] += value / degrees[i]
            row[j] -= value / degrees[j]

    ports = composition.collective.port_nodes
    port_x, port_theta = tuple(x[i] for i in ports), tuple(theta[i] for i in ports)
    incidence = composition.contact_incidence
    gaps = _mv(tuple(zip(*incidence)), port_theta)
    jx = tuple(-value for value in _mv(composition.contact_laplacian, port_x))
    js = tuple(
        -value for value in _mv(incidence, tuple(mp.sin(value) for value in gaps))
    )
    composed_x, composed_y = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    internal_storage_rate = mp.mpf(0)
    for r, component in enumerate(composition.components):
        nodes = component.node_indices
        local_x = tuple(x[i] for i in nodes)
        ax = _mv(component.spatial_partition.generator, local_x)
        force = [mp.mpf(0)] * 9
        for i, j in rings[r]:
            value = mp.sin(theta[j] - theta[i])
            force[i % 9] += value
            force[j % 9] -= value
        a = tuple(ax[i] * degrees[node] for i, node in enumerate(nodes))
        b = tuple(-value for value in force)
        for i, node in enumerate(nodes):
            composed_x[node] = -ax[i] + gamma * force[i] / degrees[node]
            composed_y[node] = gamma * ax[i]
        p, degree = nodes[4], degrees[nodes[4]]
        composed_x[p] += (jx[r] + gamma * js[r]) / degree
        composed_y[p] -= gamma * jx[r] / degree
        local_charge_x = sum(degrees[node] * composed_x[node] for node in nodes)
        local_charge_y = sum(degrees[node] * composed_y[node] for node in nodes)
        assert abs(local_charge_x - (jx[r] + gamma * js[r])) < mp.mpf("1e-65")
        assert abs(local_charge_y + gamma * jx[r]) < mp.mpf("1e-65")
        supply = ((a[4] - gamma * b[4]) * jx[r] + gamma * a[4] * js[r]) / degree
        local_loss = sum(a[i] ** 2 / degrees[node] for i, node in enumerate(nodes))
        rate = sum(
            a[i] * composed_x[node] + b[i] * composed_y[node]
            for i, node in enumerate(nodes)
        )
        assert abs(rate - (supply - local_loss)) < mp.mpf("1e-65")
        internal_storage_rate += rate
    assert max(
        abs(a - b) for a, b in zip(composed_x + composed_y, direct_x + direct_y)
    ) < mp.mpf("1e-65")
    assert abs(sum(d * v for d, v in zip(degrees, direct_x))) < mp.mpf("1e-65")
    assert abs(sum(d * v for d, v in zip(degrees, direct_y))) < mp.mpf("1e-65")
    bridge_rate = sum(
        (x[i] - x[j]) * (direct_x[i] - direct_x[j])
        + mp.sin(theta[i] - theta[j]) * (direct_y[i] - direct_y[j])
        for i, j in bridges
    )
    loss = sum(value**2 / d for value, d in zip(_mv(laplacian, x), degrees))
    assert abs(internal_storage_rate + bridge_rate + loss) < mp.mpf("1e-65")
    assert abs(internal_storage_rate + loss) > mp.mpf("1e-3")


def test_degree_two_and_form_only_boundary_substitutions_fail(composition):
    # Piecewise constant forms make every internal row zero. All remaining
    # phase evolution is caused by the existing bridges, not a phase event.
    forms = (Q(1), Q(0), Q(-1))
    jx = tuple(-value for value in _mv(composition.contact_laplacian, forms))
    gamma = Q(1, 7)
    phase = tuple(-gamma * value / d for value, d in zip(jx, (3, 4, 3)))
    assert phase == (Q(1, 21), Q(0), Q(-1, 21))
    assert phase != (Q(0),) * 3
    # Using isolated degree two in a component changes its joined row and
    # fails the original weighted mean balance for a different port vector.
    jx = tuple(
        -value for value in _mv(composition.contact_laplacian, (Q(1), Q(0), Q(0)))
    )
    assert sum(jx) == 0
    assert sum(d * current / 2 for d, current in zip((3, 4, 3), jx)) == Q(1, 2)
    for component in composition.components:
        assert component.spatial_partition.visible_generator != ((Q(1),),)


def test_hidden_initialization_is_needed_even_with_identical_ports_and_means(
    composition,
):
    eps = Q(1, 10**32)
    component = composition.components[1]
    state = tuple(eps / 2 * (int(i == 3) - int(i == 2)) for i in range(9))
    assert state[4] == 0 and _dot(component.joined_degrees, state) == 0
    assert (
        sum(d * value**2 for d, value in zip(component.joined_degrees, state)) == eps**2
    )
    internal_rate = -_mv(component.spatial_partition.generator, state)[4]
    assert internal_rate == eps / 8
    assert len(component.hidden_state_indices) == 16


def test_bridge_cubic_force_and_simultaneous_event_work_are_not_separate_self_terms(
    composition,
):
    _, bridges, degrees, laplacian = _graph()
    gamma = Q(1, 7)
    port_phase = (Q(1, 5), Q(-1, 4), Q(2, 7))
    incidence = composition.contact_incidence
    differences = _mv(tuple(zip(*incidence)), port_phase)
    cubic = tuple(
        gamma * value / (6 * d)
        for value, d in zip(_mv(incidence, tuple(v**3 for v in differences)), (3, 4, 3))
    )
    direct = [Q(0)] * 3
    for i, j in ((0, 1), (1, 2)):
        value = -gamma * (port_phase[j] - port_phase[i]) ** 3 / 6
        direct[i] += value / (3, 4, 3)[i]
        direct[j] -= value / (3, 4, 3)[j]
    assert tuple(direct) == cubic and any(cubic)
    assert _dot((3, 4, 3), cubic) == 0
    x = tuple(Q((i * 3) % 13 - 6, 17) for i in range(27))
    q = (Q(1, 10), Q(-1, 15), Q(1, 20))
    ports = (4, 13, 22)
    after = tuple(
        value + (q[ports.index(i)] if i in ports else 0) for i, value in enumerate(x)
    )
    energy_jump = (_dot(after, _mv(laplacian, after)) - _dot(x, _mv(laplacian, x))) / 2
    pressure_work = _dot(q, tuple(_mv(laplacian, x)[i] for i in ports))
    quadratic = composition.collective.port_jump_quadratic_matrix
    assert energy_jump == pressure_work + _dot(q, _mv(quadratic, q)) / 2
    assert (
        _dot(q, _mv(quadratic, q)) / 2
        == sum(d * value**2 / 2 for d, value in zip((3, 4, 3), q))
        - q[0] * q[1]
        - q[1] * q[2]
    )
    assert (_dot(degrees, after) - _dot(degrees, x)) / 58 == _dot((3, 4, 3), q) / 58
    assert len(bridges) == 2


def _error_arguments():
    return dict(
        horizon=Q(2),
        form_initial_error=Q(1, 11),
        phase_initial_error=Q(1, 13),
        form_residual_bound=Q(1, 17),
        phase_residual_bound=Q(1, 19),
    )


@pytest.mark.parametrize("key", tuple(_error_arguments()))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_scalar_error_inputs_are_admitted_before_arithmetic(key, bad):
    with pytest.raises((ValueError, TypeError)):
        owner._bound_interface_composition(**(_error_arguments() | {key: bad}))


@pytest.mark.parametrize("key", tuple(_error_arguments()))
def test_negative_budgets_are_rejected(key):
    with pytest.raises(ValueError):
        owner._bound_interface_composition(**(_error_arguments() | {key: Q(-1)}))


def test_feedback_bounds_solve_coupled_comparison_with_both_residuals():
    result = owner._bound_interface_composition(**_error_arguments())
    ax = Q(1, 11) + 2 * Q(1, 17)
    ay = Q(1, 13) + 2 * Q(1, 19)
    alpha = Q(4, 3000)
    x, y = result.form_error_upper_bound, result.phase_error_upper_bound
    assert x == ax + alpha * y and y == ay + alpha * x
    assert x > ax and y > ay
    assert result.feedback_denominator == 1 - alpha**2 > 0
    assert result.integrated_form_residual_bound == Q(2, 17)
    assert result.integrated_phase_residual_bound == Q(2, 19)
    assert result.port_pressure_error_upper_bounds == (6 * x, 8 * x, 6 * x)
    # Independent positive fixed-point iteration is bounded at every stage.
    previous_x, previous_y = ax, ay
    for _ in range(12):
        previous_x, previous_y = ax + alpha * previous_y, ay + alpha * previous_x
        assert previous_x <= x and previous_y <= y


def test_zero_window_exact_error_and_subgrid_budgets_remain_meaningful():
    tiny = Q(1, 2**700)
    result = owner._bound_interface_composition(
        **(
            _error_arguments()
            | {"horizon": 0, "form_initial_error": tiny, "phase_initial_error": 0}
        )
    )
    assert result.form_error_upper_bound == tiny
    assert result.phase_error_upper_bound == 0
    assert (
        result.integrated_form_residual_bound
        == result.integrated_phase_residual_bound
        == 0
    )
    with pytest.raises(ValueError, match="horizon"):
        owner._bound_interface_composition(
            **(_error_arguments() | {"horizon": Q(2) + tiny})
        )


def test_cubic_witness_residual_and_uniform_resolution_budget():
    a, h, eps, g = Q(7, 5000), Q(2), Q(1, 10**32), Q(1, 3000)
    result = owner._bound_cubic_interface_composition(
        total_input_variation=a, horizon=h, endpoint_radius=eps
    )
    d, ell = 1 - 2 * g**2 * h**2, 1 - 2 * g * h
    y1 = g * a / d
    x2 = 2 * g * h * y1**2 / d
    y2 = 2 * g * h * x2
    x3 = h * (4 * g * y1 * y2 + Q(4, 3) * g * y1**3) / d
    y3, ys = 2 * g * h * x3, eps / ell
    w = y2 + y3 + ys
    # Expand the cubic cross terms independently as a difference of cubes.
    cubic = Q(4, 3) * g * ((y1 + w) ** 3 - y1**3)
    quadratic = 2 * g * ((y1 + w) ** 2 - y1**2 - 2 * y1 * y2)
    tail = Q(2, 3) * g * (y1 + w) ** 4
    rho = quadratic + cubic + tail
    assert result.first_phase_upper_bound == y1
    assert (
        result.second_form_upper_bound == x2 and result.second_phase_upper_bound == y2
    )
    assert result.third_form_upper_bound == x3 and result.third_phase_upper_bound == y3
    assert (
        result.linear_source_phase_upper_bound == ys
        and result.higher_phase_upper_bound == w
    )
    assert result.quadratic_residual_upper_bound == quadratic > 0
    assert result.cubic_residual_upper_bound == cubic > 0
    assert result.sine_tail_residual_upper_bound == tail > 0
    error = result.composition_error
    assert error.form_residual_bound == rho < Q(11, 10**30)
    assert (
        error.phase_residual_bound
        == error.form_initial_error
        == error.phase_initial_error
        == 0
    )
    assert (
        error.form_error_upper_bound == h * rho / (1 - (2 * g * h) ** 2) < Q(22, 10**30)
    )
    assert error.phase_error_upper_bound == 2 * g * h * error.form_error_upper_bound


def test_zero_input_does_not_erase_nonlinear_initialization_residual():
    eps = Q(1, 10**32)
    result = owner._bound_cubic_interface_composition(
        total_input_variation=0, horizon=1, endpoint_radius=eps
    )
    assert (
        result.first_phase_upper_bound
        == result.second_form_upper_bound
        == result.third_form_upper_bound
        == 0
    )
    assert result.linear_source_phase_upper_bound > 0
    assert result.composition_error.form_residual_bound > 0
    empty = owner._bound_cubic_interface_composition(
        total_input_variation=0, horizon=1, endpoint_radius=0
    )
    assert empty.composition_error.form_error_upper_bound == 0
    zero_time = owner._bound_cubic_interface_composition(
        total_input_variation=Q(1, 1000), horizon=0, endpoint_radius=eps
    )
    assert zero_time.composition_error.form_error_upper_bound == 0


def test_explicit_cubic_residual_bounds_direct_nonlinear_internal_and_bridge_edges():
    # These are unrelated static phase vectors inside the level majorants,
    # not generated coefficients or a claimed reached trajectory.
    result = owner._bound_cubic_interface_composition(
        total_input_variation=Q(7, 6000),
        horizon=Q(3, 2),
        endpoint_radius=Q(1, 10**10),
    )
    mp = mpmath.mp.clone()
    mp.dps = 80

    def real(value):
        return mp.mpf(value.numerator) / value.denominator

    sizes = (
        result.first_phase_upper_bound,
        result.second_phase_upper_bound,
        result.third_phase_upper_bound,
        result.linear_source_phase_upper_bound,
    )
    levels = tuple(
        tuple(real(size) * ((i * multiplier) % 11 - 5) / 5 for i in range(27))
        for size, multiplier in zip(sizes, (3, 5, 7, 9))
    )
    rings, bridges, degrees, _ = _graph()
    theta = tuple(2 * mp.pi * (1, 2, 1)[i // 9] * (i % 9 - 4) / 9 for i in range(27))
    gamma = 1 / (1023 * mp.pi)
    residual = [mp.mpf(0)] * 27
    contact_residuals = []
    for i, j in sum(rings, ()) + bridges:
        gap = theta[j] - theta[i]
        first, second, third, source = (row[j] - row[i] for row in levels)
        total = first + second + third + source
        true = mp.sin(gap + total) - mp.sin(gap)
        approximation = (
            mp.cos(gap) * total
            - mp.sin(gap) * (first**2 / 2 + first * second)
            - mp.cos(gap) * first**3 / 6
        )
        defect = gamma * (true - approximation)
        residual[i] += defect / degrees[i]
        residual[j] -= defect / degrees[j]
        if (i, j) in bridges:
            contact_residuals.append(defect)
    assert all(value != 0 for value in contact_residuals)
    assert max(map(abs, residual)) < real(result.composition_error.form_residual_bound)
    # Antisymmetry is retained by the defect itself, not just by the full law.
    assert abs(sum(d * value for d, value in zip(degrees, residual))) < mp.mpf("1e-75")


@pytest.mark.parametrize(
    "change",
    (
        {"total_input_variation": Q(7, 5000) + Q(1, 2**500)},
        {"horizon": 3},
        {"endpoint_radius": -1},
        {"total_input_variation": True},
    ),
)
def test_cubic_domain_admission_precedes_residual_comparison(change, monkeypatch):
    monkeypatch.setattr(
        owner, "_bound_interface_composition", lambda **_: pytest.fail("late admission")
    )
    with pytest.raises((ValueError, TypeError)):
        owner._bound_cubic_interface_composition(
            **(
                dict(total_input_variation=Q(1, 1000), horizon=1, endpoint_radius=0)
                | change
            )
        )


@pytest.mark.parametrize("bad", (True, Q(1), np.int64(1), 0, 3))
def test_mediator_class_has_ordinary_integer_admission(bad):
    with pytest.raises(ValueError):
        owner._derive_interface_composition(bad)
