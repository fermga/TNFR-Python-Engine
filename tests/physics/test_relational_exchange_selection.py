"""Conditional exchange freedom under total, rather than nodewise, balance.

This comparison is not an engine law. On the fixed triangle every primitive
state is at most one hop away. Its graph-family expression can read two-hop
state on other graphs, so it does not defeat universal one-hop requirements.
"""

from fractions import Fraction as Q
from itertools import product

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
    # Relative phase eliminates any common phase reference. Both baseline
    # endpoint capacities stay fixed; the added exchange reads the neighbor.
    baseline_relative_change = (
        reduced_baseline[0] - reduced_baseline[2] - baseline[0] + baseline[2]
    )
    assert s.simplify(baseline_relative_change) == 0
    relative_change = (
        baseline_relative_change
        + reduced_extra[0]
        - reduced_extra[2]
        - (w / beta) * (extra[0] - extra[2])
    )
    assert s.simplify(relative_change + w * (2 + s.sqrt(3)) / (96 * beta)) == 0
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


def test_relative_capacity_intervention_cancels_scale_for_the_two_named_laws(
    triangle_reference,
):
    s, adjacency, form, phase, capacity, metric, _, _ = triangle_reference
    root = s.sqrt(3)
    assert (metric - s.diag(s.pi * root, 2 * (1 + root), 2 * (1 + root))).applyfunc(
        s.simplify
    ) == s.zeros(3)
    gradient = (2 * s.eye(3) - adjacency) * form
    assert gradient == s.Matrix((1, 0, -1))
    k = s.Symbol("k", positive=True)
    relative = {}
    for name, capacities in (
        ("before", capacity),
        ("after", s.Matrix((1, s.Rational(1, 2), 1))),
    ):
        baseline = k * metric.inv() * s.diag(*capacities) * gradient
        extra = _exchange(s, adjacency, form, phase, capacities, weight=k)[2]
        relative[name] = (
            s.simplify(baseline[0] - baseline[2]),
            s.simplify(baseline[0] - baseline[2] + extra[0] - extra[2]),
        )

    alpha = 1 / (s.pi * root) + 1 / (2 * (1 + root))
    correction = (2 + root) / 32
    expected_before = (k * alpha, k * (alpha + correction))
    expected_after = (k * alpha, k * (alpha + 2 * correction / 3))
    assert all(
        s.simplify(actual - expected) == 0
        for values, references in (
            (relative["before"], expected_before),
            (relative["after"], expected_after),
        )
        for actual, expected in zip(values, references, strict=True)
    )
    assert alpha.is_positive and correction.is_positive
    deficit = correction / (3 * (alpha + correction))
    assert deficit.is_positive
    assert s.simplify(s.Rational(1, 3) - deficit).is_positive
    changes = tuple(
        s.simplify(after / before - 1)
        for before, after in zip(relative["before"], relative["after"], strict=True)
    )
    assert changes[0] == 0
    assert s.simplify(changes[1] + deficit) == 0

    # Shared gain, clock, form amplitude and capacity scale cancel only when
    # preserved across the two matched preparations. No trajectory is sampled.
    scale = s.Symbol("scale", nonzero=True)
    for before, after, expected in zip(
        relative["before"], relative["after"], changes, strict=True
    ):
        assert s.simplify(scale * after / (scale * before) - 1 - expected) == 0
    # A changed readout/clock scale can make the separable law mimic the
    # mediated normalized response, despite unchanged underlying relative rates.
    assert (
        s.simplify(
            (1 - deficit) * relative["after"][0] / relative["before"][0] - 1 + deficit
        )
        == 0
    )

    # The test separates correction amplitudes zero and one. It cannot exclude
    # every mediated completion: arbitrarily weak added exchange approaches zero.
    amplitude = s.Symbol("amplitude", nonnegative=True)
    scaled_change = s.factor(
        (alpha + 2 * amplitude * correction / 3) / (alpha + amplitude * correction) - 1
    )
    assert (
        s.simplify(
            scaled_change
            + amplitude * correction / (3 * (alpha + amplitude * correction))
        )
        == 0
    )
    assert s.limit(scaled_change, amplitude, 0, dir="+") == 0
    assert s.simplify(scaled_change.subs(amplitude, 1) + deficit) == 0


def test_normalized_intervention_error_budget_separates_all_static_noise_corners():
    from tnfr.mathematics._rational_interval import I, pi_interval
    from tnfr.physics.relational_observations import bound_relational_rate_contrast

    # Independent rational sqrt(3) bracket, certified by squaring positive ends.
    # The shared pi enclosure supplies mathematical pi, not a binary64 proxy.
    root_bounds = (Q(17320508075688772935, 10**19), Q(17320508075688772936, 10**19))
    assert 0 < root_bounds[0] and root_bounds[0] ** 2 < 3 < root_bounds[1] ** 2
    root = I(*root_bounds)
    alpha = 1 / (pi_interval() * root) + 1 / (2 * (1 + root))
    correction = (2 + root) / 32
    deficit = correction / (3 * (alpha + correction))
    assert Q(1, 24) < deficit.lo < deficit.hi < Q(1, 3)
    rounding_allowance, final_radius = Q(1, 1584), Q(1, 48)
    assert Q(2, 99) + rounding_allowance == final_radius
    assert deficit.lo > 2 * final_radius

    # Illustrative k=1/2 only sets signal size. These independently enclosed
    # ideal rates and imposed errors are static controls, not an acquisition.
    cases = (
        (alpha / 2, alpha / 2, I(0)),
        ((alpha + correction) / 2, (alpha + 2 * correction / 3) / 2, -deficit),
    )
    reports = []
    for before, after, ideal_change in cases:
        error = before.lo / 202
        assert max(before.radius, after.radius) < error
        candidate_reports = []
        for before_sign, after_sign in product((-1, 1), repeat=2):
            # Extreme admissible errors around each enclosing rate midpoint.
            # Subtract its radius so every ideal rate remains inside the input.
            measured_before = before.midpoint + before_sign * (error - before.radius)
            measured_after = after.midpoint + after_sign * (error - after.radius)
            assert 0 < error <= (measured_before - error) / 200
            before_bounds = (measured_before - error, measured_before + error)
            after_bounds = (measured_after - error, measured_after + error)
            assert before_bounds[0] <= before.lo <= before.hi <= before_bounds[1]
            assert after_bounds[0] <= after.lo <= after.hi <= after_bounds[1]

            # Exact rational positive-denominator corners are an independent
            # reference. Dividing after/before avoids repeating the baseline.
            exact_lower = after_bounds[0] / before_bounds[1] - 1
            exact_upper = after_bounds[1] / before_bounds[0] - 1
            assert exact_lower >= ideal_change.lo - Q(2, 99)
            assert exact_upper <= ideal_change.hi + Q(2, 99)
            report = bound_relational_rate_contrast(
                before_bounds=before_bounds, after_bounds=after_bounds
            )
            assert report.normalized_change_bounds is not None
            lower, upper = report.normalized_change_bounds
            assert lower <= exact_lower <= exact_upper <= upper
            assert max(exact_lower - lower, upper - exact_upper) <= rounding_allowance
            assert lower >= ideal_change.lo - final_radius
            assert upper <= ideal_change.hi + final_radius
            candidate_reports.append((lower, upper))
        reports.append(candidate_reports)
    assert max(upper for _, upper in reports[1]) < min(lower for lower, _ in reports[0])
