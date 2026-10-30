"""Exact algebra controls for conditional pressure-completion identities.

Small graphs supply independent witnesses; the final check only reads the
retained contact response. No event or trajectory is executed. These controls
neither select a pressure law nor prove a universal theorem by sampling. The
theory owner supplies the quantified arguments. Quarter-turn phase lifts give
exact rational sines. Rate pairs retain coefficients of 1 and 1/pi separately,
without changing the clock.
"""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import (
    exact_square_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), Q(0))


def _multiply(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _scale(coefficient, vector):
    return tuple(coefficient * value for value in vector)


def _orthogonal_projector(normal):
    square = _dot(normal, normal)
    return tuple(
        tuple(Q(i == j) - left * right / square for j, right in enumerate(normal))
        for i, left in enumerate(normal)
    )


def _laplacian(size, edges):
    matrix = [[Q(0) for _ in range(size)] for _ in range(size)]
    for left, right, weight in edges:
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


def _sine_current(phases, edges):
    """Bare-edge sine current for integer multiples of pi/2, including lifts."""
    sine_quarter_turn = (Q(0), Q(1), Q(0), Q(-1))
    current = [Q(0)] * len(phases)
    for left, right, _weight in edges:
        value = sine_quarter_turn[(phases[right] - phases[left]) % 4]
        current[left] += value
        current[right] -= value
    return tuple(current)


@pytest.fixture(
    params=(
        ((0, 1, Q(2, 3)), (1, 2, Q(5, 4)), (2, 3, Q(7, 6))),
        (
            (0, 1, Q(3, 2)),
            (1, 2, Q(2, 5)),
            (2, 0, Q(4, 3)),
            (2, 3, Q(5, 7)),
        ),
        (
            (0, 1, Q(2, 3)),
            (1, 2, Q(5, 4)),
            (2, 3, Q(7, 6)),
            (0, 2, Q(0)),
        ),
    ),
    ids=("weighted_path", "weighted_triangle_tail", "zero_conductance_bare_edge"),
)
def state(request):
    edges = request.param
    size = 4
    degree = tuple(Q(sum(i in (a, b) for a, b, _ in edges)) for i in range(size))
    capacity = (Q(1, 2), Q(2), Q(3, 4), Q(5, 3))
    rho = tuple(d / nu for d, nu in zip(degree, capacity, strict=True))
    return {
        "edges": edges,
        "degree": degree,
        "capacity": capacity,
        "rho": rho,
        "laplacian": _laplacian(size, edges),
        "x": (Q(1, 3), Q(-2, 5), Q(1), Q(7, 8)),
        "phase": (0, 1, 2, 1),
    }


def _rows(state, deviation):
    """Return xdot=(constant, inverse-pi coefficient), theta's coefficient."""
    current = _sine_current(state["phase"], state["edges"])
    gradient = _multiply(state["laplacian"], state["x"])
    mobility = tuple(
        nu / d for nu, d in zip(state["capacity"], state["degree"], strict=True)
    )
    form_constant = tuple(
        nu * value for nu, value in zip(state["capacity"], deviation, strict=True)
    )
    form_inverse_pi = tuple(m * s for m, s in zip(mobility, current, strict=True))
    phase_inverse_pi = tuple(m * q for m, q in zip(mobility, gradient, strict=True))
    return form_constant, form_inverse_pi, phase_inverse_pi


def _edge_storage_rate(state, deviation):
    """Differentiate each quadratic/cosine edge, independently of q dot xdot."""
    form_constant, form_inverse_pi, phase_inverse_pi = _rows(state, deviation)
    constant, inverse_pi = Q(0), Q(0)
    sine_quarter_turn = (Q(0), Q(1), Q(0), Q(-1))
    for left, right, weight in state["edges"]:
        form_gap = state["x"][right] - state["x"][left]
        phase_gap = state["phase"][right] - state["phase"][left]
        constant += weight * form_gap * (form_constant[right] - form_constant[left])
        inverse_pi += (
            weight * form_gap * (form_inverse_pi[right] - form_inverse_pi[left])
        )
        inverse_pi += sine_quarter_turn[phase_gap % 4] * (
            phase_inverse_pi[right] - phase_inverse_pi[left]
        )
    return constant, inverse_pi


def _projected_power_direction(state):
    gradient = _multiply(state["laplacian"], state["x"])
    power_gradient = tuple(
        nu * q for nu, q in zip(state["capacity"], gradient, strict=True)
    )
    projector = _orthogonal_projector(state["degree"])
    return _multiply(projector, power_gradient)


def test_complete_rows_have_exact_storage_and_weighted_mean_balances(state):
    deviation = (Q(2, 7), Q(-1, 3), Q(4, 5), Q(-2, 9))
    form_constant, form_inverse_pi, phase_inverse_pi = _rows(state, deviation)
    gradient = _multiply(state["laplacian"], state["x"])
    constant, inverse_pi = _edge_storage_rate(state, deviation)

    assert constant == _dot(gradient, form_constant)
    assert constant != 0
    assert inverse_pi == 0
    assert _dot(state["rho"], form_inverse_pi) == 0
    assert _dot(state["rho"], phase_inverse_pi) == 0
    assert _dot(state["rho"], form_constant) / sum(state["rho"]) == _dot(
        state["degree"], deviation
    ) / sum(state["rho"])
    # A zero-conductance edge is still present in the phase and degree rows.
    if any(weight == 0 for _, _, weight in state["edges"]):
        positive = tuple(edge for edge in state["edges"] if edge[2] > 0)
        assert state["degree"] != tuple(
            Q(sum(i in (a, b) for a, b, _ in positive)) for i in range(4)
        )


def test_mean_preserving_direction_has_strict_power_on_connected_support(state):
    laplacian = state["laplacian"]
    # Positive grounded minor independently checks connected conductance support.
    grounded = tuple(tuple(row[1:]) for row in laplacian[1:])
    assert exact_symmetric_semidefinite(grounded, strict=True)
    assert not exact_symmetric_semidefinite(laplacian, strict=True)
    projector = _orthogonal_projector(state["degree"])
    assert exact_square_matrix_product(projector, projector) == projector
    assert _multiply(projector, state["degree"]) == (Q(0),) * 4

    direction = _projected_power_direction(state)
    assert _dot(state["degree"], direction) == 0
    assert _dot(direction, direction) > 0
    for sign in (-1, 1):
        deviation = _scale(Q(sign, 13), direction)
        constant, inverse_pi = _edge_storage_rate(state, deviation)
        assert inverse_pi == 0
        assert constant == Q(sign, 13) * _dot(direction, direction)
        assert sign * constant > 0


def test_relaxation_preserves_zero_deviation_but_does_not_make_storage_passive(state):
    tau, kappa = Q(7, 5), Q(3, 2)
    direction = _projected_power_direction(state)
    deviation = _scale(tau / (4 * kappa), direction)
    deviation_rate = _scale(-1 / tau, deviation)
    base_rate, inverse_pi = _edge_storage_rate(state, deviation)
    added_storage_rate = kappa * _dot(deviation, deviation_rate)

    assert _scale(-1 / tau, (Q(0),) * 4) == (Q(0),) * 4
    assert _dot(state["degree"], deviation_rate) == 0
    assert inverse_pi == 0
    assert base_rate > 0
    assert added_storage_rate < 0
    assert base_rate + added_storage_rate == Q(3, 4) * base_rate > 0


@pytest.mark.parametrize("gamma", (Q(0), Q(2, 3)))
def test_reciprocal_exchange_is_passive_but_does_not_preserve_zero_deviation(
    state, gamma
):
    kappa = Q(3, 2)
    direction = _projected_power_direction(state)
    for multiplier in (Q(-1, 5), Q(0), Q(2, 7)):
        deviation = _scale(multiplier, direction)
        deviation_rate = tuple(
            -v / kappa - gamma * r for v, r in zip(direction, deviation, strict=True)
        )
        base_rate, inverse_pi = _edge_storage_rate(state, deviation)
        total_rate = base_rate + kappa * _dot(deviation, deviation_rate)

        assert inverse_pi == 0
        assert _dot(state["degree"], deviation_rate) == 0
        assert total_rate == -kappa * gamma * _dot(deviation, deviation) <= 0
        if multiplier == 0:
            assert any(value != 0 for value in deviation_rate)
        elif gamma > 0:
            assert total_rate < 0


def test_zero_form_gradient_is_only_an_instantaneous_power_exception(state):
    constant_form = {**state, "x": (Q(2, 5),) * 4}
    assert _projected_power_direction(constant_form) == (Q(0),) * 4
    deviation = (Q(1, 7), Q(-1, 3), Q(2, 5), Q(4, 9))
    assert _edge_storage_rate(constant_form, deviation) == (0, 0)

    # At r=0, nonzero sine current can immediately leave the q=0 set.
    _, form_inverse_pi, phase_inverse_pi = _rows(constant_form, (Q(0),) * 4)
    assert phase_inverse_pi == (Q(0),) * 4
    assert any(_multiply(state["laplacian"], form_inverse_pi))


def test_disconnected_support_admits_nonconstant_form_with_zero_power_direction():
    edges = ((0, 1, Q(2, 3)), (2, 3, Q(5, 7)))
    state = {
        "laplacian": _laplacian(4, edges),
        "degree": (Q(1),) * 4,
        "capacity": (Q(1, 2), Q(2), Q(3, 4), Q(5, 3)),
        "x": (Q(1), Q(1), Q(-2), Q(-2)),
    }
    assert len(set(state["x"])) > 1
    assert _multiply(state["laplacian"], state["x"]) == (Q(0),) * 4
    assert _projected_power_direction(state) == (Q(0),) * 4


def test_instantaneous_work_neutrality_at_equilibrium_need_not_persist(state):
    equilibrium = {**state, "x": (Q(2, 5),) * 4, "phase": (0,) * 4}
    deviation = _multiply(
        _orthogonal_projector(state["degree"]), (Q(1), Q(0), Q(0), Q(0))
    )
    velocity, form_inverse_pi, phase_inverse_pi = _rows(equilibrium, deviation)
    assert _dot(state["degree"], deviation) == 0
    assert form_inverse_pi == phase_inverse_pi == (Q(0),) * 4
    assert _edge_storage_rate(equilibrium, deviation) == (0, 0)

    # At q=S=0, gradients multiplying the unspecified second rates vanish.
    # The remaining Hessian term is independent of any finite smooth rdot.
    edge_second_rate = sum(
        (weight * (velocity[b] - velocity[a]) ** 2 for a, b, weight in state["edges"]),
        Q(0),
    )
    assert edge_second_rate == _dot(velocity, _multiply(state["laplacian"], velocity))
    assert edge_second_rate > 0


def test_mean_preserving_phase_counterterm_has_a_zero_current_obstruction(state):
    deviation = _scale(Q(1, 11), _projected_power_direction(state))
    base_rate, inverse_pi = _edge_storage_rate(state, deviation)
    current = _sine_current(state["phase"], state["edges"])
    projection = _multiply(_orthogonal_projector(state["rho"]), current)
    denominator = _dot(projection, projection)
    assert denominator > 0
    counterterm = _scale(base_rate / denominator, projection)

    assert inverse_pi == 0
    assert base_rate > 0
    assert _dot(state["rho"], counterterm) == 0
    assert base_rate - _dot(current, counterterm) == 0
    assert any(counterterm)

    # With aligned phases, the entire phase gradient vanishes. No finite
    # phase-only row can then cancel the strictly positive form power.
    aligned = {**state, "phase": (0,) * 4}
    zero_current = _sine_current(aligned["phase"], aligned["edges"])
    assert zero_current == (Q(0),) * 4
    assert _edge_storage_rate(aligned, deviation) == (base_rate, 0)
    assert _dot(zero_current, zero_current) == 0


def test_retained_raw_oz_pressure_would_drive_the_original_weighted_form_mean(
    monkeypatch,
):
    from tests.research import test_collective_pressure_handoff_evidence as evidence
    from tnfr.dynamics import integrators, relational

    def forbidden_flow(*args, **kwargs):
        raise AssertionError("retained pressure analysis must not advance a flow")

    monkeypatch.setattr(integrators.DefaultIntegrator, "integrate", forbidden_flow)
    monkeypatch.setattr(integrators, "update_epi_via_nodal_equation", forbidden_flow)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden_flow)
    paths = tuple(evidence.EVIDENCE / name for name in ("response.json", "report.json"))
    receipts = tuple(evidence.file_receipt(path) for path in paths)
    response = evidence._read("response.json")
    report = evidence._read("report.json")
    assert response["status"] == "completed"
    assert report["all_checks_passed"] is False
    assert len(report["checks"]) == 9
    assert sum(row["passed"] is True for row in report["checks"]) == 8

    captures = response["captures"]
    before = evidence._rebuild(captures["IL_completed"])
    raw = evidence._rebuild(captures["OZ_raw"])
    assert before == raw
    old_pressure = tuple(
        map(evidence.exact_record, captures["IL_completed"]["stored_pressure"])
    )
    raw_pressure = tuple(
        map(evidence.exact_record, captures["OZ_raw"]["stored_pressure"])
    )
    assert raw_pressure != old_pressure
    rho = tuple(Q(d) / nu for d, nu in zip(raw.degrees, raw.capacity, strict=True))
    total_weight = sum(rho)

    # This is the derivative of the supplied extension xdot=nu*p_raw, not
    # a measured continuation or a reconstruction of pressure from its response.
    supplied_form_rate = tuple(
        nu * p for nu, p in zip(raw.capacity, raw_pressure, strict=True)
    )
    direct_mean_rate = _dot(rho, supplied_form_rate) / total_weight
    residual = tuple(
        I(p) - phi for p, phi in zip(raw_pressure, raw.pressure, strict=True)
    )
    residual_mean_rate = (
        sum((d * r for d, r in zip(raw.degrees, residual, strict=True)), I(0))
        / total_weight
    )
    constitutive_mean_rate = (
        sum((d * phi for d, phi in zip(raw.degrees, raw.pressure, strict=True)), I(0))
        / total_weight
    )
    assert constitutive_mean_rate.contains(0)
    assert residual_mean_rate.contains(direct_mean_rate)
    assert residual_mean_rate.lo > 0
    assert direct_mean_rate == _dot(raw.degrees, raw_pressure) / total_weight
    stored_jump = tuple(a - b for a, b in zip(raw_pressure, old_pressure, strict=True))
    assert _dot(raw.degrees, stored_jump) > 0

    # Preparation names do not determine the actual post-contact form gradient.
    node_index = {node: index for index, node in enumerate(raw.nodes)}
    edges = tuple(
        (node_index[a], node_index[b], weight)
        for (a, b), weight in zip(raw.edges, raw.conductances, strict=True)
    )
    independently_rebuilt_gradient = _multiply(
        _laplacian(len(raw.nodes), edges), raw.epi
    )
    assert independently_rebuilt_gradient == raw.form_gradient
    assert any(independently_rebuilt_gradient)
    assert receipts == tuple(evidence.file_receipt(path) for path in paths)
