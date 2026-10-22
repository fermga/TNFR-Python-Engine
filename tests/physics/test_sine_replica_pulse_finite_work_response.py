"""Full-flow finite probes, exact response symmetries and evidence boundaries."""

from fractions import Fraction as Q

import mpmath as mp
import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_replica_pulse as pulse
from tnfr.sdk import relational_report_to_dict

EPSILON = Q(1, 2**20)


@pytest.fixture(scope="module")
def witness():
    captured = {}
    original = pulse.validated_taylor_step

    def capture(initial, duration, field, domain, **kwargs):
        if len(initial) == 20:
            captured["field"] = field
        return original(initial, duration, field, domain, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pulse, "validated_taylor_step", capture)
        report = pulse.assess_sine_replica_pulse_finite_work_response(
            probe_amplitude=EPSILON, order=10
        )
    return report, captured["field"]


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def test_complete_finite_response_falls_inside_prior_nonlinear_prediction(witness):
    report, _ = witness
    assert report.status == "certified_finite_response"
    assert len(report.steps) == 4
    assert all(step is not None and len(step.endpoint) == 20 for step in report.steps)
    assert all(step.picard_interior_margin > 0 for step in report.steps)
    assert report.failure_reasons == (None,) * 4
    assert report.unavailable_reasons == ()
    assert report.numerical_directional_response_certified
    actual = report.antisymmetric_response_bounds
    predicted = report.predicted_antisymmetric_response_bounds
    assert predicted.lo <= actual.lo <= actual.hi <= predicted.hi
    assert actual.lo > report.analytic_contrast_lower_bound > Q(6, 10**10)
    # The raw four-reading signal and the normalized contrast have different sizes.
    assert 0 < 2 * EPSILON * actual.hi < Q(2, 10**15)


def test_error_budget_comes_from_global_flow_variations_and_declared_ports(witness):
    report, _ = witness
    h = Q(1, 16)
    E = 1 / (1 - 2 * h)
    third = 4 * E * (E - 1) * (2 * E - 1)
    error = Q(80, 6) * third * EPSILON**2
    volterra_tail = 2 * Q(3, 8) ** 3 * h**6 / (720 * (1 - Q(3, 8) * h**2 / 56))
    lower = 20 * (Q(33, 800000) * h**5 - volterra_tail)
    assert report.flow_third_derivative_upper_bound == third
    assert report.finite_probe_error_upper_bound == error
    assert report.analytic_contrast_lower_bound == lower - error
    assert report.fine_edge_margin_lower_bound == Q(13, 60) - Q(16, 7) * EPSILON
    assert report.fine_edge_margin_lower_bound > Q(1, 5)
    # Check the majorant ODE analytically, independently of its closed formula.
    for E in (Q(1), Q(8, 7), Q(3, 2)):
        v, z = E, 2 * E * (E - 1)
        w = 8 * E**3 - 12 * E**2 + 4 * E
        derivative = (24 * E**2 - 24 * E + 4) * 2 * E
        assert derivative == 2 * w + 12 * v * z + 8 * v**3


def test_all_fine_rows_use_shared_law_and_correct_scaled_clock(witness):
    report, field = witness
    # Build the doubled C5 independently of the report and production field.
    neighbors = tuple(
        tuple(j for j in range(10) if (j // 2 - i // 2) % 5 in (1, 4))
        for i in range(10)
    )
    assert report.neighbors == neighbors
    state = tuple(I(Q(i - 5, 11)) for i in range(10)) + tuple(
        I(Q(i * i - 20, 17)) for i in range(10)
    )
    actual = field(state)
    with mp.workdps(80):
        x = [_mp(value.lo) for value in state[:10]]
        phase = [_mp(value.lo) for value in state[10:]]
        expected = [
            sum(mp.sin(phase[j] - phase[i]) for j in row) / 4
            for i, row in enumerate(neighbors)
        ] + [sum(x[i] - x[j] for j in row) / 4 for i, row in enumerate(neighbors)]
        for bound, value in zip(actual, expected):
            assert _mp(bound.lo) <= value <= _mp(bound.hi)


def test_full_field_time_reversal_and_member_swap_are_distinct_symmetries(witness):
    report, field = witness
    state = tuple(I(Q(i - 3, 9)) for i in range(20))
    rates = field(state)
    reversed_rates = field(tuple(-v for v in state[:10]) + state[10:])
    expected_reversed = rates[:10] + tuple(-v for v in rates[10:])
    permutation = tuple(i ^ 1 for i in range(10))
    full_permutation = permutation + tuple(i + 10 for i in permutation)
    swapped_rates = field(tuple(state[i] for i in full_permutation))
    for observed, expected in (
        (reversed_rates, expected_reversed),
        (swapped_rates, tuple(rates[i] for i in full_permutation)),
    ):
        assert all(
            max(a.lo, b.lo) <= min(a.hi, b.hi) for a, b in zip(observed, expected)
        )
    collective, internal = report.port_form_directions
    for i, j in enumerate(permutation):
        assert collective[i] == collective[j]
        assert internal[i] == -internal[j]


def test_reversed_history_swaps_full_propagator_blocks_not_only_port_labels():
    identity, zero = np.eye(2), np.zeros((2, 2))
    D = np.diag([0.5, 1.0])
    C0, C1 = np.array([[2, 1], [1, 3]]), np.array([[3, -1], [-1, 2]])
    drift = np.block([[identity, zero], [D, identity]])
    kick0 = np.block([[identity, -C0], [zero, identity]])
    kick1 = np.block([[identity, -C1], [zero, identity]])
    M = kick1 @ drift @ kick0 @ drift
    J = np.block([[zero, -identity], [identity, zero]])
    R = np.block([[-identity, zero], [zero, identity]])
    np.testing.assert_array_equal(M.T @ J @ M, J)
    reversed_history = R @ np.linalg.inv(M) @ R
    np.testing.assert_allclose(reversed_history[:2, :2], M[2:, 2:].T, atol=1e-13)
    assert not np.allclose(reversed_history[:2, :2], M[:2, :2].T)


def test_parametric_coordinate_change_preserves_the_actual_work_ports(witness):
    report, _ = witness
    raw = (
        np.array(
            [
                [float(value.midpoint) for value in row]
                for row in report.tangent.reference.mode_blocks[1]
            ]
        )
        * np.pi
    )
    order = [0, 2, 1, 3]
    generator = raw[np.ix_(order, order)]
    D = np.diag([1 - np.cos(2 * np.pi / 5), 1.0])
    root = np.sqrt(D)
    zero, identity = np.zeros((2, 2)), np.eye(2)
    transform = np.block([[root, zero], [zero, np.linalg.inv(root)]])
    actual = transform @ generator @ np.linalg.inv(transform)
    stiffness = -root @ generator[:2, 2:] @ root
    expected = np.block([[zero, -stiffness], [identity, zero]])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-14)
    # H=20 D P becomes 20 sqrt(D) times velocity, with initial velocity sqrt(D).
    output = np.column_stack([20 * D, zero]) @ np.linalg.inv(transform)
    np.testing.assert_allclose(output, np.column_stack([20 * root, zero]), atol=1e-14)


def test_partial_numerical_failure_keeps_analytic_prediction_and_all_trials(
    witness, monkeypatch
):
    report, _ = witness
    trials = []

    def partial(initial, duration, field, domain, **kwargs):
        if len(initial) == 10:
            return report.tangent.step, None, None
        index = len(trials)
        trials.append(initial)
        assert initial == report.initial_boxes[index]
        if index == 1:
            return None, initial, "instrumented_unavailable"
        return report.steps[index], None, None

    monkeypatch.setattr(pulse, "validated_taylor_step", partial)
    failed = pulse.assess_sine_replica_pulse_finite_work_response(
        probe_amplitude=EPSILON
    )
    assert len(trials) == 4
    assert failed.status == "unavailable"
    assert failed.response_bounds is None
    assert failed.response_sign is None
    assert not failed.numerical_directional_response_certified
    assert failed.analytic_directional_response_certified
    assert failed.predicted_antisymmetric_response_bounds is not None
    assert failed.unavailable_reasons == (
        "collective_negative: instrumented_unavailable",
    )


@pytest.mark.parametrize(
    "epsilon", [0, -1, True, float("nan"), float("inf"), Q(1, 2**19)]
)
def test_invalid_amplitudes_reject_before_any_solver(epsilon, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid amplitude reached propagation")

    monkeypatch.setattr(pulse, "validated_taylor_step", forbidden)
    with pytest.raises((ValueError, TypeError)):
        pulse.assess_sine_replica_pulse_finite_work_response(probe_amplitude=epsilon)


def test_sdk_retains_finite_preparation_and_both_evidence_types(witness):
    report, _ = witness
    data = relational_report_to_dict(report)
    assert data["schema"] == "tnfr.relational-report.v1"
    assert data["report_type"] == "SineReplicaPulseFiniteWorkResponse"
    assert data["report"] == report.to_dict()["report"]
    assert data["report"]["probe_amplitude"] == {"numerator": 1, "denominator": 2**20}
    assert len(data["report"]["steps"]) == 4
    assert data["report"]["analytic_directional_response_certified"] is True
    assert data["report"]["numerical_directional_response_certified"] is True
