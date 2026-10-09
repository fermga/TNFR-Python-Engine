"""One-hop form acceleration need not admit a one-hop primitive phase law.

These exact local derivatives keep the retained reciprocal prism and regular
repeated phase preparation. The selected Hamiltonian acceleration -kappa*L*y
is a comparison, not a derived TNFR law. Gauge freedom is allowed before
testing locality; a dense centered representative alone would not suffice.
No trajectory, alternative preparation, coefficient choice or engine update
is introduced, and the complete form-mean response remains in the calculation.
"""

import pytest

from tests.physics._internal_mode_fixture import Q_MODE, _prepared_pressure_coordinates


@pytest.fixture(scope="module")
def chart():
    s, _, rows, phases, center, _, lap, project, scaled, inverse = (
        _prepared_pressure_coordinates()
    )
    kappa = s.Symbol("kappa", nonnegative=True)
    e, w = s.symbols("e w", positive=True)
    source = s.Matrix([(center - phase) / s.pi for phase in phases])
    stiffness = kappa * lap + e**2 * lap**2
    # This omits the common positive pi/w factor in D_x omega. Any other
    # representative adds a columnwise common-phase derivative, 1*c^T.
    slope = (-inverse * stiffness).applyfunc(lambda value: s.expand(s.radsimp(value)))
    return s, rows, lap, project, scaled, inverse, source, kappa, e, w, slope


def test_phase_lift_derivative_keeps_the_full_forced_form_mean(chart):
    s, _, lap, project, scaled, inverse, source, kappa, e, w, slope = chart
    assert (project * scaled * inverse - project).applyfunc(s.simplify) == s.zeros(6)
    assert scaled * s.ones(6, 1) == s.zeros(6, 1)
    assert (project * scaled).rank() == 5
    x = s.Matrix(s.symbols("x0:6", real=True))
    pressure = -e * lap * x + w * source
    velocity = project * pressure
    # The fixture inverse already has rationalized Q(sqrt(3)) coefficients;
    # expansion keeps this linear symbolic control inexpensive and exact.
    omega = (s.pi * inverse * (-kappa * lap * x + e * lap * velocity) / w).applyfunc(
        s.expand
    )
    derivative = omega.jacobian(x)
    assert (derivative - s.pi * slope / w).applyfunc(s.expand) == s.zeros(6)

    # The centered target does not authorize dropping the uniform source row.
    full_acceleration = (-e * lap * pressure + w * scaled * omega / s.pi).applyfunc(
        s.expand
    )
    assert (project * full_acceleration + kappa * lap * x).applyfunc(
        s.expand
    ) == s.zeros(6, 1)
    full_derivative = full_acceleration.jacobian(x)
    expected_derivative = e**2 * lap**2 + scaled * slope
    assert (full_derivative - expected_derivative).applyfunc(s.expand) == s.zeros(6)
    mean_response = sum(full_derivative * s.Matrix(Q_MODE * 2)) / 6
    expected_mean = (kappa + e**2) * (5 - 3 * s.sqrt(3)) / 2
    assert s.simplify(mean_response - expected_mean) == 0
    assert expected_mean.is_negative
    # Q_MODE is an existing tangent direction; no perturbed graph is run.
    assert s.simplify(sum(pressure) / 6 - w * sum(source) / 6) == 0


def test_retained_phase_has_a_positive_gauge_invariant_remote_derivative(chart):
    s, rows, _, _, _, _, _, kappa, e, w, slope = chart
    assert 0 not in rows[4] and 0 not in rows[5] and 0 not in (4, 5)
    coefficient_kappa = (193 - 111 * s.sqrt(3)) / 1716
    coefficient_diffusion = (37 - 19 * s.sqrt(3)) / 396
    # Exact integer comparisons establish strict positivity, without a
    # floating threshold, tuned coefficient or small-spread approximation.
    assert 193**2 - 3 * 111**2 == 286 > 0
    assert 37**2 - 3 * 19**2 == 286 > 0
    assert coefficient_kappa.is_positive and coefficient_diffusion.is_positive
    obstruction = kappa * coefficient_kappa + e**2 * coefficient_diffusion
    assert s.simplify(slope[4, 0] - slope[5, 0] - obstruction) == 0
    assert obstruction.is_positive
    common_column = s.Symbol("common_column", real=True)
    left = s.pi * slope[4, 0] / w + common_column
    right = s.pi * slope[5, 0] / w + common_column
    assert s.simplify(left - right - s.pi * obstruction / w) == 0
    # Both remote partial derivatives would have to vanish for a C1 law
    # depending only on each node and its immediate neighbors. Their
    # difference is positive in every common-phase gauge, including kappa=0.
    assert s.simplify(obstruction.subs(kappa, 0) - e**2 * coefficient_diffusion) == 0


def test_obstruction_survives_relabeling_and_arbitrary_common_column_shifts(chart):
    s, rows, lap, project, scaled, inverse, _, kappa, e, _, slope = chart
    order = (3, 0, 5, 1, 4, 2)
    permutation = s.eye(6)[:, order]
    relabeled_lap = permutation.T * lap * permutation
    relabeled_scaled = permutation.T * scaled * permutation
    relabeled_inverse = permutation.T * inverse * permutation
    assert (project * relabeled_scaled * relabeled_inverse - project).applyfunc(
        s.simplify
    ) == s.zeros(6)
    relabeled_slope = permutation.T * slope * permutation
    assert (
        relabeled_slope
        + relabeled_inverse * (kappa * relabeled_lap + e**2 * relabeled_lap**2)
    ).applyfunc(s.simplify) == s.zeros(6)
    gauges = s.Matrix(1, 6, s.symbols("c0:6", real=True))
    candidate = relabeled_slope + s.ones(6, 1) * gauges
    column, first, second = (order.index(old) for old in (0, 4, 5))
    assert order[column] not in rows[order[first]]
    assert order[column] not in rows[order[second]]
    assert (
        s.simplify(
            candidate[first, column]
            - candidate[second, column]
            - (slope[4, 0] - slope[5, 0])
        )
        == 0
    )


def test_consensus_has_a_sparse_lift_so_the_obstruction_is_not_gauge_centering(chart):
    s, rows, lap, project, _, _, _, kappa, e, w, _ = chart
    # Consensus is an analytic boundary control of the same response formula,
    # not an additional persistence candidate or a prepared runtime graph.
    scaled_consensus = -lap
    sparse_derivative = s.pi * (kappa * s.eye(6) + e**2 * lap) / w
    target = -(kappa * lap + e**2 * lap**2)
    assert (
        w * project * scaled_consensus * sparse_derivative / s.pi - target
    ).applyfunc(s.simplify) == s.zeros(6)
    for i in range(6):
        for j in range(6):
            if i != j and j not in rows[i]:
                assert sparse_derivative[i, j] == 0
    assert (
        sparse_derivative * s.ones(6, 1) - s.pi * kappa * s.ones(6, 1) / w
    ).applyfunc(s.simplify) == s.zeros(6, 1)
    # A common EPI shift induces only a common phase velocity. Requiring the
    # centered representative itself to be sparse would wrongly reject it.
    x = s.Matrix(s.symbols("x0:6", real=True))
    velocity = -e * lap * x
    omega = s.pi * (kappa * x - e * velocity) / w
    assert (omega.jacobian(x) - sparse_derivative).applyfunc(s.simplify) == s.zeros(6)
    acceleration = -e * lap * velocity + w * scaled_consensus * omega / s.pi
    assert (acceleration + kappa * lap * x).applyfunc(s.simplify) == s.zeros(6, 1)
    assert s.simplify(sum(acceleration)) == 0
