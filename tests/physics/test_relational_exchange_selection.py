"""Conditional exchange freedom under total, rather than nodewise, balance.

This comparison is not an engine law. On the fixed triangle every primitive
state is at most one hop away. Its graph-family expression can read two-hop
state on other graphs, so it does not defeat universal one-hop requirements.
"""

import pytest

from tests.physics._internal_mode_fixture import _metric_differential


def _exchange(s, adjacency, form, phase, capacity, *, weight=1, storage=1):
    size = len(form)
    degrees = tuple(sum(adjacency[i, j] for j in range(size)) for i in range(size))
    degree = s.diag(*degrees)
    normalized_form = (s.eye(size) - degree.inv() * adjacency) * form
    gradient = s.Matrix(
        [
            sum(adjacency[i, j] * s.sin(phase[i] - phase[j]) for j in range(size))
            for i in range(size)
        ]
    )

    def edge_entry(i, j):
        if not adjacency[i, j] or capacity[i] + capacity[j] == 0:
            return s.S.Zero
        mobility = capacity[i] * capacity[j] / (capacity[i] + capacity[j])
        return (
            weight
            * adjacency[i, j]
            * mobility
            * (normalized_form[i] + normalized_form[j])
            * s.sin(phase[j] - phase[i])
            / (storage * degrees[i] * degrees[j])
        )

    skew = s.Matrix(size, size, edge_entry)
    return skew, gradient, (skew * gradient).applyfunc(s.simplify)


@pytest.fixture(scope="module")
def triangle_reference():
    s = pytest.importorskip("sympy")
    adjacency = s.ones(3) - s.eye(3)
    form = s.Matrix((s.Rational(1, 3), 0, -s.Rational(1, 3)))
    phase = s.Matrix((0, s.pi / 6, -s.pi / 6))
    capacity = s.ones(3, 1)
    rows = ((1, 2), (0, 2), (0, 1))
    metric, _, source, jacobian = _metric_differential(s, rows, tuple(phase))
    return s, adjacency, form, phase, capacity, metric, source, jacobian


def test_total_balance_and_zero_capacity_do_not_force_capacity_additivity(
    triangle_reference,
):
    s, adjacency, form, phase, _, metric, source, _ = triangle_reference
    capacities = s.Matrix(s.symbols("nu0:3", positive=True))
    w, beta = s.symbols("w beta", positive=True)
    skew, gradient, extra = _exchange(
        s, adjacency, form, phase, capacities, weight=w, storage=beta
    )
    assert (skew + skew.T).applyfunc(s.simplify) == s.zeros(3)
    assert s.simplify(gradient.dot(extra)) == 0
    assert (gradient + metric * source).applyfunc(s.simplify) == s.zeros(3, 1)
    laplacian = 2 * s.eye(3) - adjacency
    capacity = s.diag(*capacities)
    base = (w / beta) * metric.inv() * capacity * laplacian * form
    complete_work = w * (laplacian * form).dot(capacity * source)
    complete_work += beta * gradient.dot(base + extra)
    assert s.simplify(complete_work) == 0
    partial = _exchange(s, adjacency, form, phase, s.Matrix((1, 1, 0)))[2]
    assert partial[2] == 0
    assert partial != s.zeros(3, 1)
    single = [
        _exchange(s, adjacency, form, phase, s.eye(3)[:, node])[2] for node in range(3)
    ]
    assert all(value == s.zeros(3, 1) for value in single)
    assert partial != single[0] + single[1]
    assert _exchange(s, adjacency, form, phase, s.zeros(3, 1))[2] == s.zeros(3, 1)
    # Harmonic pair mobility is smooth on positive-capacity strata and in
    # evolving form/phase for each held capacity, including frozen zero rows.
    # Its zero extension is continuous, not differentiable, at a joint zero.


def test_triangle_counterpart_changes_native_source_and_nodewise_work(
    triangle_reference,
):
    s, adjacency, form, phase, capacity, metric, _, jacobian = triangle_reference
    _, gradient, extra = _exchange(s, adjacency, form, phase, capacity)
    expected = s.Matrix((1 + s.sqrt(3), -3 - s.sqrt(3), -3 - s.sqrt(3))) / 64
    assert extra == expected
    assert extra[0] != extra[1]
    source_change = (jacobian * extra).applyfunc(s.simplify)
    expected_change = s.Matrix(
        (-2 - s.sqrt(3), 1 + s.sqrt(3) / 2, 1 + s.sqrt(3) / 2)
    ) / (32 * s.pi)
    assert (source_change - expected_change).applyfunc(s.simplify) == s.zeros(3, 1)
    local_work = s.matrix_multiply_elementwise(gradient, extra)
    assert local_work != s.zeros(3, 1)
    assert s.simplify(sum(local_work)) == 0
    # A held-capacity intervention targets separability directly, without
    # inferring a law from the same trajectory it is asked to predict.
    w, beta = s.symbols("w beta", positive=True)
    reduced_capacity = s.Matrix((1, s.Rational(1, 2), 1))
    q = (2 * s.eye(3) - adjacency) * form
    baseline = (w / beta) * metric.inv() * s.diag(*capacity) * q
    reduced_baseline = (w / beta) * metric.inv() * s.diag(*reduced_capacity) * q
    reduced_extra = _exchange(
        s,
        adjacency,
        form,
        phase,
        reduced_capacity,
        weight=w,
        storage=beta,
    )[2]
    assert s.simplify(reduced_baseline[0] - baseline[0]) == 0
    change = reduced_baseline[0] + reduced_extra[0] - baseline[0] - w * extra[0] / beta
    assert s.simplify(change + w * (1 + s.sqrt(3)) / (192 * beta)) == 0
    assert s.simplify(reduced_extra[0] - (2 * w / (3 * beta)) * extra[0]) == 0
    # All other triangle vertices are neighbors: this witness is genuinely
    # one-hop in primitive state on this support, not a hidden local telemetry
    # assumption. No phase trajectory or universal selection is asserted.


def test_counterpart_preserves_symmetries_units_and_equal_replica_reduction(
    triangle_reference,
):
    s, adjacency, form, phase, capacity, _, _, _ = triangle_reference
    _, _, extra = _exchange(s, adjacency, form, phase, capacity)
    shift, rotation = s.symbols("shift rotation", real=True)
    scale, clock = s.symbols("scale clock", positive=True)
    transformed = _exchange(
        s,
        adjacency,
        scale * form + shift * s.ones(3, 1),
        phase + rotation * s.ones(3, 1),
        capacity / clock,
        weight=scale,
        storage=scale**2,
    )[2]
    assert (transformed - extra / clock).applyfunc(s.simplify) == s.zeros(3, 1)
    reflected = _exchange(s, adjacency, -form, -phase, capacity)[2]
    assert reflected == -extra
    permutation = s.Matrix(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    relabeled = _exchange(
        s,
        permutation * adjacency * permutation.T,
        permutation * form,
        permutation * phase,
        permutation * capacity,
    )[2]
    assert relabeled == permutation * extra
    replicas = 2
    lift = s.kronecker_product(s.eye(3), s.ones(replicas, 1))
    fine_adjacency = s.kronecker_product(adjacency, s.ones(replicas))
    _, fine_gradient, fine_extra = _exchange(
        s, fine_adjacency, lift * form, lift * phase, lift * capacity
    )
    _, gradient, _ = _exchange(s, adjacency, form, phase, capacity)
    assert fine_gradient == replicas * lift * gradient
    assert fine_extra == lift * extra
    # Primitive locality on K3 does not extend to this larger graph merely
    # because the prepared synchronized reduction is exact.
