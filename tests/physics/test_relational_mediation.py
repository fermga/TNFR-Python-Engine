"""Static and two-step mediation controls under the joint form/phase law.

The fine path 0--10--5 is supplied. Ideal tangent formulas are compared with
the native field, while rational matrices test their declared coefficient
family separately. The two-step regression does not evaluate the reserved
response horizon. Midpoint controls check the inherited fast-limit field;
they imply no primitive edge birth, exact finite-capacity visible-only
closure or positive diffusion-memory kernel.
"""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
    step_relational_exchange,
)
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.mathematics.linear_observation import (
    derive_coordinate_memory,
    derive_linear_observation,
)

MODEL = RelationalExchangeModel(storage_scale=1.0)
CAPACITY = 1.0
AMPLITUDE = 1 / 64
RINGS = tuple(tuple(range(offset, offset + 5)) for offset in (0, 5))
RING_EDGES = tuple(
    (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
)
MEDIATED_EDGES = RING_EDGES + ((0, 10), (10, 5))
VISIBLE = tuple(range(10)) + tuple(range(11, 21))
PORT_ROWS = ((0, 10), (5, 15))


def _graph(*, direct=False, donor=0.0, mediator=0.0, mediator_capacity=CAPACITY):
    graph = nx.Graph()
    graph.add_nodes_from(range(10 if direct else 11))
    graph.add_edges_from(
        RING_EDGES + ((0, 5),) if direct else MEDIATED_EDGES, weight=1.0
    )
    graph.graph.update(_t=3.0, retained={"history": ["unchanged"]})
    for node in graph:
        graph.nodes[node].update(
            EPI=donor if node == 0 else mediator if node == 10 else 0.0,
            theta=math.tau * (node % 5) / 5 if node < 10 else 0.0,
            nu_f=mediator_capacity if node == 10 else CAPACITY,
            delta_nfr=99.0,
        )
    return graph


def _state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _nonuniform_visible_graph(*, mediator_capacity=CAPACITY):
    graph = _graph(mediator=1 / 40, mediator_capacity=mediator_capacity)
    forms = (
        1 / 64,
        -1 / 128,
        1 / 256,
        -1 / 192,
        1 / 96,
        -1 / 80,
        1 / 128,
        -1 / 256,
        1 / 160,
        -1 / 96,
    )
    phase_offsets = (
        0.03,
        -0.006,
        0.004,
        -0.012,
        0.008,
        -0.02,
        0.007,
        -0.009,
        0.006,
        -0.004,
    )
    for node, form, offset in zip(range(10), forms, phase_offsets, strict=True):
        graph.nodes[node]["EPI"] = form
        graph.nodes[node]["theta"] += offset
    graph.nodes[10]["theta"] = 0.011
    return graph


def _two_mediator_graph(graph=None, *, fractions=(Q(1, 3), Q(2, 3))):
    graph = _graph() if graph is None else graph.copy()
    graph.remove_edge(10, 5)
    graph.add_node(11, nu_f=CAPACITY, delta_nfr=99.0)
    graph.add_edges_from(((10, 11), (11, 5)), weight=1.0)
    for node, fraction in zip((10, 11), fractions, strict=True):
        for attribute in ("EPI", "theta"):
            left, right = graph.nodes[0][attribute], graph.nodes[5][attribute]
            graph.nodes[node][attribute] = left + float(fraction) * (right - left)
    return graph


def _star_graph(*, offsets=(0.0, 0.0, 0.0), nonuniform=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(16))
    for ring in range(3):
        offset = 5 * ring
        graph.add_edges_from(
            ((offset + i, offset + (i + 1) % 5) for i in range(5)), weight=1.0
        )
        graph.add_edge(offset, 15, weight=1.0)
        for i in range(5):
            graph.nodes[offset + i].update(
                EPI=AMPLITUDE * (-1) ** (offset + i) / (i + 1) if nonuniform else 0.0,
                theta=math.tau * i / 5
                + offsets[ring]
                + ((i - 2) / 512 if nonuniform else 0.0),
                nu_f=CAPACITY,
                delta_nfr=99.0,
            )
    ports = tuple(graph.nodes[node] for node in (0, 5, 10))
    graph.nodes[15].update(
        EPI=sum(port["EPI"] for port in ports) / 3,
        theta=math.atan2(
            sum(math.sin(port["theta"]) for port in ports),
            sum(math.cos(port["theta"]) for port in ports),
        ),
        nu_f=CAPACITY,
        delta_nfr=99.0,
    )
    return graph


def _assert_induced_port_rows(field, *, segments):
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    for port, other, internal in ((0, 5, (1, 4)), (5, 0, (6, 9))):
        gradient = sum(field.epi[port] - field.epi[j] for j in internal)
        gradient += (field.epi[port] - field.epi[other]) / segments
        gaps = [field.phase[j] - field.phase[port] for j in internal]
        gaps.append((field.phase[other] - field.phase[port]) / segments)
        real = sum(math.cos(gap) for gap in gaps)
        imaginary = sum(math.sin(gap) for gap in gaps)
        angle = math.atan2(imaginary, real)
        sinc = math.sin(angle) / angle if angle else 1.0
        metric = math.pi * math.hypot(real, imaginary) * sinc
        assert field.form_rate[port] == pytest.approx(
            -e * gradient / 3 + w * angle / math.pi, rel=0, abs=2e-15
        )
        assert field.phase_rate[port] == pytest.approx(
            w * gradient / (beta * metric), rel=0, abs=2e-15
        )


def _generator(cosine, inverse_pi):
    """Assemble the proved equilibrium Jacobian from individual fine edges."""
    size = 11
    laplacian = [[0] * size for _ in range(size)]
    hessian = [[0] * size for _ in range(size)]
    degrees, strengths = [0] * size, [0] * size
    for left, right in MEDIATED_EDGES:
        edge_cosine = 1 if 10 in (left, right) else cosine
        for matrix, weight in ((laplacian, 1), (hessian, edge_cosine)):
            matrix[left][left] += weight
            matrix[right][right] += weight
            matrix[left][right] -= weight
            matrix[right][left] -= weight
        for node in (left, right):
            degrees[node] += 1
            strengths[node] += edge_cosine
    e, w, beta, nu = map(
        Q, (MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale, CAPACITY)
    )
    return tuple(
        tuple(-e * nu * value / degrees[i] for value in laplacian[i])
        + tuple(-w * nu * inverse_pi * value / strengths[i] for value in hessian[i])
        for i in range(size)
    ) + tuple(
        tuple(
            w * nu * inverse_pi * value / (beta * strengths[i])
            for value in laplacian[i]
        )
        + (0,) * size
        for i in range(size)
    )


@pytest.fixture(scope="module")
def tangent():
    """Differentiate 22 field coordinates once, without advancing time."""
    graph = _graph()
    before = _state(graph)
    base = evaluate_relational_exchange(graph, model=MODEL)
    probe = 2.0**-18
    observed = np.empty((22, 22))
    for column in range(22):
        node = column % 11
        attribute = "EPI" if column < 11 else "theta"
        rates = []
        for sign in (-1, 1):
            sample = graph.copy()
            sample.nodes[node][attribute] += sign * probe
            field = evaluate_relational_exchange(sample, model=MODEL)
            rates.append(np.array(field.form_rate + field.phase_rate))
        observed[:, column] = (rates[1] - rates[0]) / (2 * probe)
    expected = _generator(math.cos(math.tau / 5), 1 / math.pi)
    assert _state(graph) == before
    return base, expected, observed


def test_native_mediated_field_has_the_full_joint_equilibrium_jacobian(tangent):
    field, generator, observed = tangent
    cosine = math.cos(math.tau / 5)
    expected_metric = tuple(
        math.pi
        * (2 if node == 10 else 1 + 2 * cosine if node in (0, 5) else 2 * cosine)
        for node in range(11)
    )
    assert field.phase_metric == pytest.approx(expected_metric, rel=0, abs=2e-15)
    assert field.form_rate == pytest.approx((0.0,) * 11, rel=0, abs=1e-15)
    assert field.phase_rate == (0.0,) * 11
    assert observed == pytest.approx(np.array(generator, dtype=float), rel=0, abs=1e-9)


def test_static_mediated_phase_minimum_does_not_install_instantaneous_relaxation():
    graph = nx.path_graph(3)
    gap, offset = 0.6, 0.125
    for node, phase in enumerate((0.0, gap / 2, gap)):
        graph.nodes[node].update(EPI=0.0, theta=phase, nu_f=1.0)
    midpoint = evaluate_relational_exchange(graph, model=MODEL)
    minimum = 2 * MODEL.storage_scale * (1 - math.cos(gap / 2))
    assert float(midpoint.storage) == pytest.approx(minimum, rel=0, abs=2e-15)
    assert midpoint.phase_gradient == pytest.approx(
        (-math.sin(gap / 2), 0.0, math.sin(gap / 2)), rel=0, abs=2e-15
    )

    graph.nodes[1]["theta"] += offset
    displaced = evaluate_relational_exchange(graph, model=MODEL)
    excess = 2 * MODEL.storage_scale * math.cos(gap / 2) * (1 - math.cos(offset))
    assert float(displaced.storage - midpoint.storage) == pytest.approx(
        excess, rel=0, abs=2e-15
    )
    assert displaced.phase_gradient[1] == pytest.approx(
        2 * math.cos(gap / 2) * math.sin(offset), rel=0, abs=2e-15
    )
    # The restoring phase gradient first produces form; it is not a phase jump.
    assert displaced.form_rate[1] < 0
    assert displaced.phase_rate == (0.0,) * 3
    graph.nodes[1]["nu_f"] = 0.0
    frozen = evaluate_relational_exchange(graph, model=MODEL)
    assert frozen.storage == displaced.storage
    assert frozen.pressure[1] == displaced.pressure[1] < 0
    assert frozen.form_rate[1] == frozen.phase_rate[1] == 0.0
    assert frozen.phase[1] == gap / 2 + offset


def test_shared_memory_retains_both_mediator_coordinates_and_signed_kernel(tangent):
    _, generator, _ = tangent
    memory = derive_coordinate_memory(generator, VISIBLE)
    e, w, beta, nu = (
        MODEL.epi_weight,
        MODEL.phase_weight,
        MODEL.storage_scale,
        CAPACITY,
    )
    h = 1 + 2 * math.cos(math.tau / 5)
    hidden = nu * np.array([[-e, -w / math.pi], [w / (beta * math.pi), 0]])
    outward = nu * np.array(
        [[e / 3, w / (math.pi * h)], [-w / (beta * math.pi * h), 0]]
    )
    inward = nu * np.array([[e / 2, w / (2 * math.pi)], [-w / (2 * beta * math.pi), 0]])
    cross = nu**2 * np.array(
        [
            [e**2 / 6 - w**2 / (2 * beta * math.pi**2 * h), e * w / (6 * math.pi)],
            [-e * w / (2 * beta * math.pi * h), -(w**2) / (2 * beta * math.pi**2 * h)],
        ]
    )
    assert memory.visible_indices == VISIBLE
    assert memory.hidden_indices == (10, 21)
    assert np.array(memory.hidden_generator, dtype=float) == pytest.approx(
        hidden, rel=0, abs=1e-15
    )
    b = np.array(memory.hidden_to_visible, dtype=float)
    c = np.array(memory.visible_to_hidden, dtype=float)
    kernel = np.array(memory.kernel_at_zero, dtype=float)
    for rows in PORT_ROWS:
        assert b[list(rows)] == pytest.approx(outward, rel=0, abs=1e-15)
        assert c[:, list(rows)] == pytest.approx(inward, rel=0, abs=1e-15)
    assert kernel[np.ix_(PORT_ROWS[1], PORT_ROWS[0])] == pytest.approx(
        cross, rel=0, abs=1e-15
    )
    interior = [i for i in range(20) if all(i not in rows for rows in PORT_ROWS)]
    assert np.count_nonzero(b[interior]) == np.count_nonzero(c[:, interior]) == 0
    assert (
        np.count_nonzero(kernel[interior]) == np.count_nonzero(kernel[:, interior]) == 0
    )
    # Joint exchange has signed cross channels; a positive diffusion-kernel
    # conclusion cannot be inherited just because the support is reciprocal.
    assert cross[0, 0] > 0 and cross[0, 1] > 0
    assert cross[1, 0] < 0 and cross[1, 1] < 0


def test_hidden_initial_form_changes_visible_rates_at_identical_ring_states(tangent):
    _, generator, _ = tangent
    memory = derive_coordinate_memory(generator, VISIBLE)
    base = evaluate_relational_exchange(_graph(), model=MODEL)
    changed = evaluate_relational_exchange(_graph(mediator=AMPLITUDE), model=MODEL)
    base_state = np.array(base.epi + base.phase)[list(VISIBLE)]
    changed_state = np.array(changed.epi + changed.phase)[list(VISIBLE)]
    assert np.array_equal(base_state, changed_state)
    rate_change = (
        np.array(changed.form_rate + changed.phase_rate)
        - np.array(base.form_rate + base.phase_rate)
    )[list(VISIBLE)]
    expected = np.array(memory.hidden_to_visible, dtype=float) @ np.array(
        [AMPLITUDE, 0.0]
    )
    assert rate_change == pytest.approx(expected, rel=0, abs=1e-15)
    assert np.count_nonzero(expected) == 4


@pytest.mark.parametrize("mediator_capacity", (1.5, 0.0))
def test_nonlinear_pressure_chart_retains_both_moving_endpoint_means(mediator_capacity):
    """Static chart checks with full ring state; no two-port closure or step."""
    graph = _nonuniform_visible_graph(mediator_capacity=mediator_capacity)
    before = _state(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    mean_form = (field.epi[0] + field.epi[5]) / 2
    mean_phase = (field.phase[0] + field.phase[5]) / 2
    delta = field.phase[5] - field.phase[0]
    u, v = field.epi[10] - mean_form, field.phase[10] - mean_phase
    sinc_v = math.sin(v) / v
    metric = 2 * math.pi * math.cos(delta / 2) * sinc_v
    expected_pressure = -e * u - w * v / math.pi
    assert field.phase_metric[10] == pytest.approx(metric, rel=3e-15)
    assert field.form_gradient[10] == pytest.approx(2 * u, rel=3e-15)
    assert field.phase_source[10] == pytest.approx(-v / math.pi, rel=3e-15)
    assert field.pressure[10] == pytest.approx(expected_pressure, rel=3e-15, abs=1e-17)
    assert -math.pi * (field.pressure[10] + e * u) / w == pytest.approx(v, abs=1e-16)

    mean_form_rate = (field.form_rate[0] + field.form_rate[5]) / 2
    mean_phase_rate = (field.phase_rate[0] + field.phase_rate[5]) / 2
    assert abs(mean_form_rate) > 1e-4 and abs(mean_phase_rate) > 1e-4
    chart_u_rate = mediator_capacity * field.pressure[10] - mean_form_rate
    chart_v_rate = 2 * mediator_capacity * w * u / (beta * metric) - mean_phase_rate
    chart_pressure_rate = -e * chart_u_rate - w * chart_v_rate / math.pi
    assert chart_u_rate == pytest.approx(
        field.form_rate[10] - mean_form_rate, abs=1e-16
    )
    assert chart_v_rate == pytest.approx(
        field.phase_rate[10] - mean_phase_rate, abs=1e-16
    )
    assert chart_u_rate + mean_form_rate == pytest.approx(
        field.form_rate[10], abs=1e-16
    )
    reconstructed_phase_rate = (
        -math.pi * (chart_pressure_rate + e * chart_u_rate) / w + mean_phase_rate
    )
    assert reconstructed_phase_rate == pytest.approx(field.phase_rate[10], abs=1e-16)

    # The native pressure map is evaluated independently along the captured
    # full-field direction. These probes neither evolve nor freeze the ports.
    increment = 2.0**-15
    pressures = []
    for sign in (-1, 1):
        probe = graph.copy()
        for node in field.nodes:
            probe.nodes[node]["EPI"] += sign * increment * field.form_rate[node]
            probe.nodes[node]["theta"] += sign * increment * field.phase_rate[node]
        pressures.append(evaluate_relational_exchange(probe, model=MODEL).pressure[10])
    observed_pressure_rate = (pressures[1] - pressures[0]) / (2 * increment)
    assert observed_pressure_rate == pytest.approx(
        chart_pressure_rate, rel=0, abs=2e-12
    )
    if mediator_capacity == 0.0:
        assert field.form_rate[10] == field.phase_rate[10] == 0.0
        assert chart_u_rate == -mean_form_rate != 0.0
        assert chart_v_rate == -mean_phase_rate != 0.0
        assert abs(chart_pressure_rate) > 1e-4
    assert _state(graph) == before


def test_midpoint_lift_induces_native_port_rows_and_condensed_storage_balance():
    graph = _nonuniform_visible_graph()
    unlifted = evaluate_relational_exchange(graph, model=MODEL)
    for attribute in ("EPI", "theta"):
        graph.nodes[10][attribute] = (
            graph.nodes[0][attribute] + graph.nodes[5][attribute]
        ) / 2
    before = _state(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    e, beta = MODEL.epi_weight, MODEL.storage_scale
    _assert_induced_port_rows(field, segments=2)

    form_cost = sum((field.epi[i] - field.epi[j]) ** 2 / 2 for i, j in RING_EDGES)
    form_cost += (field.epi[0] - field.epi[5]) ** 2 / 4
    phase_cost = sum(
        1 - math.cos(field.phase[j] - field.phase[i]) for i, j in RING_EDGES
    )
    phase_cost += 2 * (1 - math.cos((field.phase[5] - field.phase[0]) / 2))
    assert float(field.storage) == pytest.approx(
        form_cost + beta * phase_cost, abs=4e-15
    )
    u = unlifted.epi[10] - (unlifted.epi[0] + unlifted.epi[5]) / 2
    v = unlifted.phase[10] - (unlifted.phase[0] + unlifted.phase[5]) / 2
    delta = unlifted.phase[5] - unlifted.phase[0]
    hidden_excess = u**2 + 2 * beta * math.cos(delta / 2) * (1 - math.cos(v))
    assert unlifted.epi[:10] == field.epi[:10]
    assert unlifted.phase[:10] == field.phase[:10]
    assert float(unlifted.storage - field.storage) == pytest.approx(
        hidden_excess, rel=0, abs=4e-15
    )
    assert hidden_excess > 0
    # Fixed hidden initial storage is retained even when a fast-regime theorem
    # controls the visible state difference on a small 1/mu scale.
    expected_loss = e * sum(
        sum(field.epi[i] - field.epi[j] for j in graph[i]) ** 2 / graph.degree[i]
        for i in range(10)
    )
    assert float(field.continuous_loss) == pytest.approx(expected_loss, abs=2e-17)
    assert float(field.storage_rate) == pytest.approx(-expected_loss, abs=2e-17)
    assert abs(field.form_rate[10]) < 1e-16 and abs(field.phase_rate[10]) < 1e-16
    assert _state(graph) == before


def test_midpoint_induced_phase_pressure_differs_from_a_direct_unit_edge():
    delta = 1 / 128
    midpoint, direct = _graph(), _graph(direct=True)
    for graph in (midpoint, direct):
        for node in RINGS[1]:
            graph.nodes[node]["theta"] += delta
    midpoint.nodes[10]["theta"] = delta / 2
    lifted = evaluate_relational_exchange(midpoint, model=MODEL)
    replaced = evaluate_relational_exchange(direct, model=MODEL)
    cosine = math.cos(math.tau / 5)
    expected_lift = (
        MODEL.phase_weight
        / math.pi
        * math.atan2(math.sin(delta / 2), 2 * cosine + math.cos(delta / 2))
    )
    expected_direct = (
        MODEL.phase_weight
        / math.pi
        * math.atan2(math.sin(delta), 2 * cosine + math.cos(delta))
    )
    assert lifted.form_rate[0] == pytest.approx(expected_lift, abs=2e-16)
    assert replaced.form_rate[0] == pytest.approx(expected_direct, abs=2e-16)
    assert 0 < lifted.form_rate[0] < replaced.form_rate[0]
    assert lifted.phase_rate == (0.0,) * 11
    assert replaced.phase_rate == (0.0,) * 10


def test_finite_capacity_midpoint_is_not_invariant_when_internal_ring_state_changes():
    graph = _graph(mediator_capacity=7.0)
    graph.nodes[1]["EPI"] = AMPLITUDE
    field = evaluate_relational_exchange(graph, model=MODEL)
    assert field.epi[10] == (field.epi[0] + field.epi[5]) / 2 == 0.0
    assert field.phase[10] == (field.phase[0] + field.phase[5]) / 2 == 0.0
    assert field.form_rate[10] == field.phase_rate[10] == 0.0
    mean_form_rate = (field.form_rate[0] + field.form_rate[5]) / 2
    mean_phase_rate = (field.phase_rate[0] + field.phase_rate[5]) / 2
    r = 1 + 2 * math.cos(math.tau / 5)
    assert mean_form_rate == pytest.approx(MODEL.epi_weight * AMPLITUDE / 6, abs=2e-16)
    assert mean_phase_rate == pytest.approx(
        -MODEL.phase_weight * AMPLITUDE / (2 * MODEL.storage_scale * math.pi * r),
        abs=2e-16,
    )
    assert mean_form_rate > 0 > mean_phase_rate
    # Thus the two relative midpoint coordinates have nonzero derivatives
    # -mean_form_rate and -mean_phase_rate at every such finite capacity.


def test_fast_mediator_common_quadratic_identity_and_box_remainder_use_native_rows():
    # Exact algebra with a formal positive rational parameter, not a claim
    # that 22/7 equals mathematical pi or its binary64 materialization.
    parameter = Q(22, 7)
    p = ((Q(2), parameter), (parameter, 2 + parameter**2))
    a0 = ((-Q(1, 2), -1 / (2 * parameter)), (1 / (2 * parameter), Q(0)))
    transpose = tuple(zip(*a0))
    left, right = product(transpose, p), product(p, a0)
    assert tuple(tuple(x + y for x, y in zip(a, b)) for a, b in zip(left, right)) == (
        (Q(-1), Q(0)),
        (Q(0), Q(-1)),
    )
    tangent = evaluate_relational_uniform_tangent(_graph(), model=MODEL)
    native = np.array(tangent.generator)[np.ix_((10, 21), (10, 21))]
    p_native = np.array([[2, math.pi], [math.pi, 2 + math.pi**2]])
    np.testing.assert_allclose(
        native.T @ p_native + p_native @ native, -np.eye(2), rtol=0, atol=2e-15
    )

    pi = pi_interval()
    # P-I has determinant1; 14I-P has determinant144-13*pi².
    assert pi.lo > 3 and (144 - 13 * pi**2).lo > 0
    quarter = Q(1, 4)
    cosine_floor, sinc_floor = 1 - quarter**2 / 2, 1 - quarter**2 / 6
    assert cos(I(quarter)).lo > cosine_floor
    assert (sin(I(quarter)) / quarter).lo > sinc_floor
    remainder_bound = (1 / (cosine_floor * sinc_floor) - 1) / 6
    assert 2 * 14 * remainder_bound < Q(1, 2)

    graph = _graph(mediator=1 / 100)
    for node in RINGS[1]:
        graph.nodes[node]["theta"] += 0.5
    graph.nodes[10]["theta"] = 0.5  # delta/2+v, at delta=.5 and v=.25.
    field = evaluate_relational_exchange(graph, model=MODEL)
    linear = np.array([[-0.5, -0.5 / math.pi], [0.5 / math.pi, 0.0]]) @ [0.01, 0.25]
    residual = np.array([field.form_rate[10], field.phase_rate[10]]) - linear
    assert abs(residual[0]) < 1e-16
    assert 0 < residual[1] < float(remainder_bound) * 0.01
    # This checks the frozen fast field and uniform proof coefficients, not
    # finite-mu trajectory accuracy, a basin radius or an executed projection.


def test_two_mediator_joint_and_sequential_elimination_retain_the_native_interface():
    graph = _two_mediator_graph(_nonuniform_visible_graph())
    before = _state(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    _assert_induced_port_rows(field, segments=3)
    assert field.form_rate[10:] == pytest.approx((0.0, 0.0), rel=0, abs=2e-16)
    assert field.phase_rate[10:] == pytest.approx((0.0, 0.0), rel=0, abs=2e-16)
    difference = field.epi[5] - field.epi[0]
    delta = field.phase[5] - field.phase[0]
    ring_storage = sum(
        (field.epi[i] - field.epi[j]) ** 2 / 2
        + MODEL.storage_scale * (1 - math.cos(field.phase[j] - field.phase[i]))
        for i, j in RING_EDGES
    )
    assert float(field.storage) == pytest.approx(
        ring_storage
        + difference**2 / 6
        + 3 * MODEL.storage_scale * (1 - math.cos(delta / 3)),
        rel=0,
        abs=4e-15,
    )
    for eliminated, survivor, left, right in ((10, 11, 0, 5), (11, 10, 5, 0)):
        # The inherited length-two message gives the same stationary thirds
        # in either elimination order. No bare unit edge replaces that message.
        sequential = graph.copy()
        for attribute in ("EPI", "theta"):
            left_value = graph.nodes[left][attribute]
            right_value = graph.nodes[right][attribute]
            value = (left_value / 2 + right_value) / 1.5
            assert value == pytest.approx(graph.nodes[survivor][attribute], abs=1e-17)
            sequential.nodes[survivor][attribute] = value
            sequential.nodes[eliminated][attribute] = (left_value + value) / 2

        # Probe a nonstationary survivor while retaining the first midpoint.
        # This independently exercises its original degree-two row and the
        # half-angle incoming phasor, which would be lost by bare-edge reuse.
        sequential.nodes[survivor]["EPI"] += 1 / 128
        sequential.nodes[survivor]["theta"] += 1 / 256
        for attribute in ("EPI", "theta"):
            sequential.nodes[eliminated][attribute] = (
                sequential.nodes[left][attribute]
                + sequential.nodes[survivor][attribute]
            ) / 2
        observed = evaluate_relational_exchange(sequential, model=MODEL)
        q = (observed.epi[survivor] - observed.epi[left]) / 2
        q += observed.epi[survivor] - observed.epi[right]
        gaps = (
            (observed.phase[left] - observed.phase[survivor]) / 2,
            observed.phase[right] - observed.phase[survivor],
        )
        real, imaginary = (
            sum(function(gap) for gap in gaps) for function in (math.cos, math.sin)
        )
        angle = math.atan2(imaginary, real)
        metric = math.pi * math.hypot(real, imaginary) * math.sin(angle) / angle
        assert abs(observed.pressure[eliminated]) < 2e-16
        assert abs(observed.phase_rate[eliminated]) < 2e-16
        assert observed.pressure[survivor] == pytest.approx(
            -MODEL.epi_weight * q / 2 + MODEL.phase_weight * angle / math.pi,
            rel=0,
            abs=2e-16,
        )
        assert observed.phase_rate[survivor] == pytest.approx(
            MODEL.phase_weight * q / (MODEL.storage_scale * metric),
            rel=0,
            abs=2e-16,
        )
        assert abs(observed.pressure[survivor]) > 1e-3
    assert _state(graph) == before


def test_iterated_bare_edge_midpoints_do_not_stationarize_two_mediators():
    visible = _nonuniform_visible_graph()
    joint = evaluate_relational_exchange(_two_mediator_graph(visible), model=MODEL)
    wrong = evaluate_relational_exchange(
        _two_mediator_graph(visible, fractions=(Q(1, 4), Q(1, 2))), model=MODEL
    )
    difference = wrong.epi[5] - wrong.epi[0]
    delta = wrong.phase[5] - wrong.phase[0]
    assert wrong.epi[:10] == joint.epi[:10]
    assert wrong.phase[:10] == joint.phase[:10]
    assert abs(wrong.pressure[10]) < 2e-16
    assert abs(wrong.phase_rate[10]) < 2e-16
    assert wrong.form_gradient[11] == pytest.approx(-difference / 4, abs=2e-17)
    assert wrong.phase_gradient[11] == pytest.approx(
        math.sin(delta / 4) - math.sin(delta / 2), rel=0, abs=2e-16
    )
    assert abs(wrong.phase_rate[11]) > 1e-4
    assert float(wrong.storage) > float(joint.storage)
    wrong_form_cost = sum(
        (wrong.epi[i] - wrong.epi[j]) ** 2 / 2 for i, j in ((0, 10), (10, 11), (11, 5))
    )
    assert wrong_form_cost - difference**2 / 6 == pytest.approx(
        difference**2 / 48, rel=0, abs=2e-19
    )


def test_two_mediator_fast_quadratic_bound_matches_joint_native_hidden_rows():
    laplacian = np.array([[2.0, -1.0], [-1.0, 2.0]])
    zero, identity = np.zeros((2, 2)), np.eye(2)
    linear = np.block(
        [
            [-laplacian / 4, -laplacian / (4 * math.pi)],
            [laplacian / (4 * math.pi), zero],
        ]
    )
    metric = np.block(
        [
            [2 * identity, math.pi * identity],
            [math.pi * identity, (2 + math.pi**2) * identity],
        ]
    )
    tangent = evaluate_relational_uniform_tangent(_two_mediator_graph(), model=MODEL)
    native = np.array(tangent.generator)[np.ix_((10, 11, 22, 23), (10, 11, 22, 23))]
    np.testing.assert_allclose(native, linear, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        native.T @ metric + metric @ native,
        -np.block([[laplacian, zero], [zero, laplacian]]) / 2,
        rtol=0,
        atol=2e-15,
    )
    # Uniform proof coefficients on |delta|<=1/2 and |v_i|<=1/8.
    a_max, b_max = Q(11, 48), Q(3, 16)
    cosine_floor, sinc_floor = 1 - a_max**2 / 2, 1 - b_max**2 / 6
    assert cos(I(a_max)).lo > cosine_floor
    assert (sin(I(b_max)) / b_max).lo > sinc_floor
    eta = 1 / (cosine_floor * sinc_floor) - 1
    assert pi_interval().lo > 3 and eta < Q(1, 28)
    assert 2 * 14 * eta / 4 < Q(1, 4)

    graph = _graph()
    delta = 0.5
    for node in RINGS[1]:
        graph.nodes[node]["theta"] += delta
    graph = _two_mediator_graph(graph)
    u, v = np.array([0.01, -0.015]), np.array([-0.125, 0.125])
    for index, node in enumerate((10, 11)):
        graph.nodes[node]["EPI"] += u[index]
        graph.nodes[node]["theta"] += v[index]
    field = evaluate_relational_exchange(graph, model=MODEL)
    a = (delta / 3 + v[1] / 2, delta / 3 - v[0] / 2)
    b = (v[0] - v[1] / 2, v[1] - v[0] / 2)
    phase_metric = np.array(
        [
            2 * math.pi * math.cos(angle) * math.sin(offset) / offset
            for angle, offset in zip(a, b, strict=True)
        ]
    )
    expected = np.concatenate(
        (
            -laplacian @ u / 4 - laplacian @ v / (4 * math.pi),
            (laplacian @ u) / (2 * phase_metric),
        )
    )
    observed = np.array(field.form_rate[10:] + field.phase_rate[10:])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-16)
    residual = observed - linear @ np.concatenate((u, v))
    assert np.linalg.norm(residual[:2]) < 2e-16
    assert 0 < np.linalg.norm(residual[2:]) < float(eta / 4) * np.linalg.norm(u)
    # This is a static fast-field check, not a finite-capacity trajectory bound.


def test_series_phase_lift_is_retained_beyond_endpoint_circular_data():
    graph = _graph()
    for node in RINGS[1]:
        graph.nodes[node]["theta"] += 3 * math.pi / 4
    graph = _two_mediator_graph(graph)
    fields = []
    for lifted_gap in (3 * math.pi / 4, -5 * math.pi / 4):
        branch = graph.copy()
        branch.nodes[10]["theta"] = lifted_gap / 3
        branch.nodes[11]["theta"] = 2 * lifted_gap / 3
        gaps = [
            math.remainder(
                branch.nodes[j]["theta"] - branch.nodes[i]["theta"], math.tau
            )
            for i, j in ((0, 10), (10, 11), (11, 5))
        ]
        assert gaps == pytest.approx((lifted_gap / 3,) * 3, rel=0, abs=2e-15)
        assert max(map(abs, gaps)) < math.pi / 2
        field = evaluate_relational_exchange(branch, model=MODEL)
        assert field.form_rate[10:] == pytest.approx((0.0, 0.0), rel=0, abs=3e-16)
        assert field.phase_rate == (0.0,) * 12
        fields.append(field)
    assert fields[0].epi[:10] == fields[1].epi[:10]
    assert fields[0].phase[:10] == fields[1].phase[:10]
    assert fields[0].pressure[0] > 0 > fields[1].pressure[0]
    # Both branches lie outside the small-delta fast-bound box. Their static
    # distinction establishes no shared basin or autonomous branch selection.


def test_three_port_star_lift_preserves_all_native_visible_rows_and_storage():
    graph = _star_graph(offsets=(0.06, -0.025, 0.04), nonuniform=True)
    before = _state(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    ports = (0, 5, 10)
    real = sum(math.cos(field.phase[i]) for i in ports)
    imaginary = sum(math.sin(field.phase[i]) for i in ports)
    center = math.atan2(imaginary, real)
    assert max(abs(field.phase[i] - center) for i in ports) < math.pi / 2
    for node in range(15):
        ring, local = divmod(node, 5)
        internal = tuple(5 * ring + (local + step) % 5 for step in (-1, 1))
        q = sum(field.epi[node] - field.epi[j] for j in internal)
        gaps = [field.phase[j] - field.phase[node] for j in internal]
        if node in ports:
            q += sum(field.epi[node] - field.epi[j] for j in ports) / 3
            gaps.append(center - field.phase[node])
        r, s = (sum(function(gap) for gap in gaps) for function in (math.cos, math.sin))
        angle = math.atan2(s, r)
        sinc = math.sin(angle) / angle if angle else 1.0
        metric = math.pi * math.hypot(r, s) * sinc
        assert field.form_rate[node] == pytest.approx(
            -MODEL.epi_weight * q / len(gaps) + MODEL.phase_weight * angle / math.pi,
            rel=0,
            abs=2e-15,
        )
        assert field.phase_rate[node] == pytest.approx(
            MODEL.phase_weight * q / (MODEL.storage_scale * metric),
            rel=0,
            abs=2e-15,
        )
    ring_storage = sum(
        (field.epi[i] - field.epi[j]) ** 2 / 2
        + MODEL.storage_scale * (1 - math.cos(field.phase[j] - field.phase[i]))
        for i, j in graph.edges
        if 15 not in (i, j)
    )
    star_form_storage = sum(
        (field.epi[i] - field.epi[j]) ** 2 / 6 for i, j in ((0, 5), (0, 10), (5, 10))
    )
    assert float(field.storage) == pytest.approx(
        ring_storage
        + star_form_storage
        + MODEL.storage_scale * (3 - math.hypot(real, imaginary)),
        rel=0,
        abs=7e-15,
    )
    assert abs(field.form_rate[15]) < 2e-16
    assert abs(field.phase_rate[15]) < 2e-16
    assert _state(graph) == before


def test_three_port_native_gradient_and_form_rate_have_mixed_rectangles():
    # An additive pair phase potential gives a zero rectangle in its port-0
    # gradient when the other two phases vary independently. This is a native
    # static counterexample, not an autonomous law or a numerical derivative.
    gradients = {}
    for b, c in ((0.0, 0.0), (0.4, 0.0), (0.0, 0.2), (0.4, 0.2)):
        field = evaluate_relational_exchange(
            _star_graph(offsets=(0.0, b, c)), model=MODEL
        )
        resultant = math.sqrt(3 + 2 * (math.cos(b) + math.cos(c) + math.cos(b - c)))
        expected = -(math.sin(b) + math.sin(c)) / resultant
        assert field.phase_gradient[0] == pytest.approx(expected, rel=0, abs=1e-15)
        gradients[b, c] = field.phase_gradient[0]
    rectangle = (
        gradients[0.4, 0.2]
        - gradients[0.4, 0.0]
        - gradients[0.0, 0.2]
        + gradients[0.0, 0.0]
    )
    assert -7e-6 < rectangle < -5e-6

    # The storage gradient and the form row are distinct. Hold ring A fixed
    # and independently vary B and C to test the actual inherited response.
    rates = {}
    for b, c in ((0.25, 0.25), (0.35, 0.25), (0.25, 0.35), (0.35, 0.35)):
        field = evaluate_relational_exchange(
            _star_graph(offsets=(0.0, b, c)), model=MODEL
        )
        center = math.atan2(math.sin(b) + math.sin(c), 1 + math.cos(b) + math.cos(c))
        expected = (
            MODEL.phase_weight
            / math.pi
            * math.atan2(
                math.sin(center), 2 * math.cos(math.tau / 5) + math.cos(center)
            )
        )
        assert field.form_rate[0] == pytest.approx(expected, rel=0, abs=2e-16)
        rates[b, c] = field.form_rate[0]
    rate_rectangle = (
        rates[0.35, 0.35] - rates[0.35, 0.25] - rates[0.25, 0.35] + rates[0.25, 0.25]
    )
    assert rate_rectangle > 1e-5
    # A sum of independent pair contributions to this form row would have
    # zero mixed rectangle; the mediator transmits a collective phase message.


def test_star_cosine_pair_surrogate_matches_quadratic_order_only():
    baseline = evaluate_relational_exchange(_star_graph(), model=MODEL)
    errors = []
    for gap in (0.25, 0.125):
        field = evaluate_relational_exchange(
            _star_graph(offsets=(0.0, gap, 0.0)), model=MODEL
        )
        native_cost = float(field.phase_storage - baseline.phase_storage)
        resultant = math.sqrt(5 + 4 * math.cos(gap))
        pair_cost = 2 * (1 - math.cos(gap)) / 3
        assert native_cost == pytest.approx(3 - resultant, rel=0, abs=3e-15)
        assert native_cost / gap**2 == pytest.approx(1 / 3, abs=7e-4)
        assert pair_cost / gap**2 == pytest.approx(1 / 3, abs=2e-3)
        error = native_cost - pair_cost
        assert error == pytest.approx((3 - resultant) ** 2 / 6, rel=0, abs=3e-15)
        assert error > 0
        errors.append(error)
    assert 15.8 < errors[0] / errors[1] < 16.1
    # This rejects the standard cosine pair surrogate beyond quadratic order.
    # A differently chosen pair potential can match through fourth order;
    # the separate mixed-rectangle test rules out exact pair additivity.


def _acceleration_probe(graph, field):
    """Static D F(z)[F(z)] comparison, not two trajectory endpoints."""
    probe = 2.0**-10
    rates = []
    for sign in (-1, 1):
        sample = graph.copy()
        for node, form, phase in zip(
            field.nodes, field.form_rate, field.phase_rate, strict=True
        ):
            sample.nodes[node]["EPI"] += sign * probe * form
            sample.nodes[node]["theta"] += sign * probe * phase
        result = evaluate_relational_exchange(sample, model=MODEL)
        rates.append(np.array(result.form_rate + result.phase_rate))
    return (rates[1] - rates[0]) / (2 * probe)


@pytest.mark.parametrize("mediator_capacity", (0.0, 1.0, 2.0))
def test_mediator_capacity_scales_the_initial_receiver_acceleration(mediator_capacity):
    graph = _graph(donor=AMPLITUDE, mediator_capacity=mediator_capacity)
    before = _state(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    observed = _acceleration_probe(graph, field)
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    h = 1 + 2 * math.cos(math.tau / 5)
    form_acceleration = (
        AMPLITUDE
        * CAPACITY
        * mediator_capacity
        * (e**2 / 6 - w**2 / (2 * beta * math.pi**2 * h))
    )
    phase_acceleration = (
        -AMPLITUDE * e * w * CAPACITY * mediator_capacity / (2 * beta * math.pi * h)
    )
    assert field.form_rate[5:10] == pytest.approx((0.0,) * 5, rel=0, abs=1e-15)
    assert field.phase_rate[5:10] == (0.0,) * 5
    assert observed[5] == pytest.approx(form_acceleration, rel=0, abs=1e-10)
    assert observed[16] == pytest.approx(phase_acceleration, rel=0, abs=1e-10)
    if mediator_capacity:
        assert form_acceleration > 0 and phase_acceleration < 0
    else:
        assert field.form_rate[10] == field.phase_rate[10] == 0.0
        assert form_acceleration == phase_acceleration == 0.0
        # A frozen mediator supplies no initial transfer; positive-capacity
        # local recovery cannot be asserted for this boundary preparation.
    assert _state(graph) == before


@pytest.mark.parametrize("mediator_capacity", (0.0, 1.0, 2.0))
def test_two_simultaneous_steps_match_the_closed_receiver_response(mediator_capacity):
    graph = _graph(donor=AMPLITUDE, mediator_capacity=mediator_capacity)
    dt = 1 / 32
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    cosine = math.cos(math.tau / 5)
    strength = 1 + 2 * cosine
    phase_displacement = w * mediator_capacity * AMPLITUDE * dt / (2 * beta * math.pi)
    angle = math.atan2(
        math.sin(phase_displacement), 2 * cosine + math.cos(phase_displacement)
    )
    angle_over_sine = (
        angle / math.sin(phase_displacement) if phase_displacement else 1 / strength
    )
    expected_form = (
        e**2 * AMPLITUDE * mediator_capacity * CAPACITY * dt**2 / 6
        - w * CAPACITY * dt * angle / math.pi
    )
    expected_phase = (
        -e
        * w
        * AMPLITUDE
        * mediator_capacity
        * CAPACITY
        * dt**2
        * angle_over_sine
        / (2 * beta * math.pi)
    )

    # These ideal closed formulas include the second step's nonlinear phase
    # metric. The absolute tolerance allows represented reference/libm rounding;
    # it is not a continuous-trajectory error bound or the reserved F4 criterion.
    assert graph.nodes[5]["theta"] == 0.0
    first = step_relational_exchange(graph, model=MODEL, dt=dt)
    assert (first.after.epi[5], first.after.phase[5]) == pytest.approx(
        (0.0, 0.0), rel=0, abs=1e-14
    )
    assert (first.after.epi[10], first.after.phase[10]) == pytest.approx(
        (e * mediator_capacity * AMPLITUDE * dt / 2, -phase_displacement),
        rel=0,
        abs=1e-14,
    )
    second = step_relational_exchange(graph, model=MODEL, dt=dt)
    assert (second.after.epi[5], second.after.phase[5]) == pytest.approx(
        (expected_form, expected_phase), rel=0, abs=1e-14
    )
    assert graph.graph["_t"] == 3.0 + 2 * dt


def test_direct_bridge_and_disconnected_receiver_distinguish_the_mediator():
    direct = evaluate_relational_exchange(
        _graph(direct=True, donor=AMPLITUDE), model=MODEL
    )
    h = 1 + 2 * math.cos(math.tau / 5)
    assert direct.form_rate[5] == pytest.approx(
        MODEL.epi_weight * CAPACITY * AMPLITUDE / 3, rel=0, abs=1e-15
    )
    assert direct.phase_rate[5] == pytest.approx(
        -MODEL.phase_weight
        * CAPACITY
        * AMPLITUDE
        / (MODEL.storage_scale * math.pi * h),
        rel=0,
        abs=1e-15,
    )
    # Removing the receiver path leaves a separately admitted C5. Do not
    # execute a disconnected graph or manufacture a law for an isolated mediator.
    receiver = _graph(donor=AMPLITUDE).subgraph(RINGS[1]).copy()
    before = _state(receiver)
    field = evaluate_relational_exchange(receiver, model=MODEL)
    observed = _acceleration_probe(receiver, field)
    assert field.form_rate == pytest.approx((0.0,) * 5, rel=0, abs=1e-15)
    assert field.phase_rate == (0.0,) * 5
    assert observed == pytest.approx(np.zeros(10), rel=0, abs=1e-12)
    assert _state(receiver) == before


def test_full_donor_reflection_permutes_rates_and_preserves_external_response():
    graph = _graph(donor=AMPLITUDE)
    # Nonuniform internal form and phase deviations prevent this from being
    # merely a port-only control or a phase-sign inversion at equilibrium.
    for node, form, phase_offset in (
        (0, AMPLITUDE, 1 / 128),
        (1, AMPLITUDE / 2, 1 / 256),
        (2, -AMPLITUDE / 3, -1 / 128),
        (3, AMPLITUDE / 4, 1 / 192),
        (4, -AMPLITUDE / 5, -1 / 256),
    ):
        graph.nodes[node]["EPI"] = form
        graph.nodes[node]["theta"] += phase_offset
    permutation = {node: node for node in graph}
    permutation.update({1: 4, 4: 1, 2: 3, 3: 2})
    edges = {frozenset(edge) for edge in graph.edges}
    assert {
        frozenset((permutation[left], permutation[right]))
        for left, right in graph.edges
    } == edges

    # Copy all nodal coordinates simultaneously under the support automorphism;
    # applying only a phase reversal would describe a different operation.
    states = {node: dict(graph.nodes[node]) for node in graph}
    reflected = graph.copy()
    for node in reflected:
        reflected.nodes[node].clear()
        reflected.nodes[node].update(states[permutation[node]])
    before = (_state(graph), _state(reflected))
    original_field = evaluate_relational_exchange(graph, model=MODEL)
    reflected_field = evaluate_relational_exchange(reflected, model=MODEL)
    assert original_field.nodes == reflected_field.nodes == tuple(range(11))
    assert reflected_field.epi != original_field.epi
    assert reflected_field.phase != original_field.phase
    assert reflected_field.capacity == original_field.capacity == (CAPACITY,) * 11
    assert reflected_field.form_rate == pytest.approx(
        tuple(original_field.form_rate[permutation[node]] for node in graph),
        rel=0,
        abs=2e-14,
    )
    assert reflected_field.phase_rate == pytest.approx(
        tuple(original_field.phase_rate[permutation[node]] for node in graph),
        rel=0,
        abs=2e-14,
    )
    for name in ("form_rate", "phase_rate"):
        assert getattr(reflected_field, name)[5:] == pytest.approx(
            getattr(original_field, name)[5:], rel=0, abs=2e-14
        )
    assert (_state(graph), _state(reflected)) == before
    # This static check supports full-state reflection equivariance. It is
    # neither a generic phase-only symmetry nor evidence of magnetic polarity.


def test_rational_coefficient_family_needs_two_hidden_initial_coordinates():
    # These declared rational coefficients are not exact cos(2*pi/5) or 1/pi.
    # The positive determinant of the analytic port block proves the same
    # two-coordinate obstruction at the ideal geometry; no floating rank is used.
    generator = _generator(Q(1, 3), Q(2, 7))
    output = tuple(tuple(Q(i == j) for j in range(22)) for i in VISIBLE)
    realization = derive_linear_observation(generator, output)
    assert realization.output_rank == 20
    assert realization.dimension == 22
    assert realization.rank_progression == (20, 22, 22)
    assert realization.extra_coordinates == 2
