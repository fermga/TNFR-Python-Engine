"""Prepared-prism harmonic lift obstruction and conditional passive exchange.

Read-only field extraction binds the specified auxiliary direction to its
actual K/J coordinates. Exact ideal derivatives use the prepared surd Gram,
without rounding it into the rational phase-response API. No phase law,
engine trajectory or new Hamiltonian coupling is implemented.
"""

import numpy as np
import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    _prepared_phase_geometry,
    _prepared_phase_metric,
)
from tnfr.physics.fields import compute_phase_current, compute_phase_curvature
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    hamiltonian_vector_field,
)


def test_actual_prism_readouts_bind_the_auxiliary_harmonic_geometric_direction():
    s, graph, rows, phases, center, _, _, _ = _prepared_phase_geometry()
    resultant_mean = (1 + s.sqrt(3)) / 3
    expected_k = s.Matrix([phase - center for phase in phases])
    expected_j = expected_k.applyfunc(lambda value: -resultant_mean * s.sin(value))
    for node, phase in zip(NODES, phases, strict=True):
        graph.nodes[node]["theta"] = float(phase)
    assert (
        max(abs(phases[i] - phases[j]) for i, row in enumerate(rows) for j in row)
        == s.pi / 3
    )
    actual_k, actual_j = compute_phase_curvature(graph), compute_phase_current(graph)
    np.testing.assert_allclose(
        [actual_k[node] for node in NODES],
        list(map(float, expected_k)),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        [actual_j[node] for node in NODES],
        list(map(float, expected_j)),
        atol=1e-14,
        rtol=0,
    )
    point = extract_phase_space_point(graph)
    assert point.nodes == NODES
    np.testing.assert_array_equal(point.k_phi, [actual_k[node] for node in NODES])
    np.testing.assert_array_equal(point.j_phi, [actual_j[node] for node in NODES])
    velocity = hamiltonian_vector_field(point).reshape(6, 4)
    np.testing.assert_array_equal(velocity[:, 0], point.j_phi)
    np.testing.assert_array_equal(velocity[:, 1], -point.k_phi)
    # This auxiliary velocity has not been applied to the graph or identified
    # with primitive theta_dot. The next test checks whether such a lift exists.


def test_no_primitive_phase_velocity_lifts_both_harmonic_components():
    s, _, rows, phases, center, _, gram, mean_response = _prepared_phase_geometry()
    _, _, _, _, source, source_jacobian = _prepared_phase_metric()
    curvature = s.Matrix([phase - center for phase in phases])
    current = s.Matrix(
        [
            sum(s.sin(phases[j] - phases[i]) for j in neighbors) / len(neighbors)
            for i, neighbors in enumerate(rows)
        ]
    ).applyfunc(s.simplify)
    assert curvature == -s.pi * source
    dk = s.eye(6) - mean_response
    assert s.simplify(dk + s.pi * source_jacobian) == s.zeros(6)
    assert dk.rank() == 5 and dk.nullspace() == [s.ones(6, 1)]
    # Differentiate the production unique-neighbor mean-sine expression.
    dj = s.Matrix(
        6,
        6,
        lambda i, j: (
            (gram[i, j] if j in rows[i] else 0)
            - int(i == j) * sum(gram[i, other] for other in rows[i])
        )
        / len(rows[i]),
    )
    assert s.simplify(dj * s.ones(6, 1)) == s.zeros(6, 1)
    assert s.simplify(dk * current - current) == s.zeros(6, 1)
    common_rotation = s.symbols("common_rotation", real=True)
    required_phase_rate = current + common_rotation * s.ones(6, 1)
    # Rank five makes these ALL solutions of DK*theta_dot=J, including
    # nonrepeated proposals; common rotation cannot change the current rate.
    actual_current_rate = (dj * required_phase_rate).applyfunc(s.simplify)
    magnitude = (5 + 3 * s.sqrt(3)) / 36
    assert actual_current_rate == s.Matrix([-magnitude, magnitude, 0] * 2)
    residual = (actual_current_rate + curvature).applyfunc(s.simplify)
    assert s.simplify(residual[0] + s.pi / 6 + magnitude) == 0
    assert residual[0].is_negative and residual[1].is_positive
    stacked = dk.col_join(dj)
    required = current.col_join(-curvature)
    assert stacked.rank() == 5
    assert stacked.row_join(required).rank() == 6
    # Thus no phase velocity can realize this harmonic direction at this
    # regular prepared state; other field flows and supports remain open.


def test_passive_reciprocal_exchange_has_loss_without_being_a_pure_gradient():
    s = pytest.importorskip("sympy")
    a, e, k = s.symbols("a e k", positive=True)
    damping = s.symbols("damping", nonnegative=True)  # b=-damping
    x, y, u, v = s.symbols("x y u v", real=True)
    state = s.Matrix([x, y, u, v])
    velocity = s.Matrix(
        [-e * x - k * u, -e * y - k * v, a * x - damping * u, a * y - damping * v]
    )
    storage = a * (x * x + y * y) + k * (u * u + v * v)
    rate = (s.Matrix([storage]).jacobian(state) * velocity)[0]
    assert (
        s.expand(rate + 2 * a * e * (x * x + y * y) + 2 * k * damping * (u * u + v * v))
        == 0
    )
    transform = s.diag(s.sqrt(a), s.sqrt(a), s.sqrt(k), s.sqrt(k))
    generator = velocity.jacobian(state)
    normalized = s.simplify(transform * generator * transform.inv())
    loss = s.diag(e, e, damping, damping)
    exchange = normalized + loss
    assert exchange + exchange.T == s.zeros(4)
    assert exchange[:2, 2:] == -s.sqrt(a * k) * s.eye(2)
    assert exchange[2:, :2] == s.sqrt(a * k) * s.eye(2)
    preparation = {x: 0, y: 0, u: 1, v: 0, damping: 0}
    assert rate.subs(preparation) == 0
    assert velocity.subs(preparation) == s.Matrix([-k, 0, 0, 0])
    # For b=0, zero instantaneous loss does NOT imply zero velocity in this
    # skew-plus-loss model. Its zero-loss set z=0 is invariant only at zeta=0.
    assert s.solve((-k * u, -k * v), (u, v)) == {u: 0, v: 0}
    assert generator.trace() == -2 * (e + damping)
    # The neutral b=e>0 condition lies outside this passive b<=0 family.
    # This is a conditional storage identity, not the engine's joint energy.
