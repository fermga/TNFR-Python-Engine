"""Independent full-law controls for acquired unlike C9 composition.

The interface and phase registration are supplied. Capture, rather than the
existence of its exact carrier, is the new claim. These controls evaluate
finite algebraic certificates and static vector fields, never trajectories.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_mixed_capture as owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**6)
EPSILON = Q(1, 10**70)
ETA = 20 * EPSILON
DELTA = Q(1, 1024)
HORIZON = Q(2 * 10**10)


def _assess(**changes):
    values = dict(
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_sine_mixed_capture(**values)


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _contains(context, bounds, value, width="1e-30"):
    lower, upper = (bounds.lo, bounds.hi) if hasattr(bounds, "lo") else bounds
    assert _mp(context, lower) <= value <= _mp(context, upper)
    assert _mp(context, upper - lower) < context.mpf(width)


def _geometry(joined=True, adapted=True):
    # Solve the phase congruence independently, instead of copying p's formula.
    permutation = tuple(
        next(k for k in range(9) if (2 * (k - 4) - (j - 4)) % 9 == 0) for j in range(9)
    )
    internal = tuple((9 * r + j, 9 * r + (j + 1) % 9) for r in (0, 1) for j in range(9))
    cross = tuple((j, 9 + (permutation[j] if adapted else j)) for j in range(9))
    edges = internal + (cross if joined else ())
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    return permutation, internal, cross, edges, degrees


def _laplacian(edges):
    rows = [[Q(0) for _ in range(18)] for _ in range(18)]
    for i, j in edges:
        rows[i][i] += 1
        rows[j][j] += 1
        rows[i][j] -= 1
        rows[j][i] -= 1
    return tuple(tuple(row) for row in rows)


def _product(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _state(context, forms=(Q(0), Q(0)), phase_gap=DELTA, xerror=None, yerror=None):
    xerror = (Q(0),) * 18 if xerror is None else xerror
    yerror = (Q(0),) * 18 if yerror is None else yerror
    alpha = 2 * context.pi / 9
    form = tuple(_mp(context, forms[i // 9] + xerror[i]) for i in range(18))
    phase = tuple(
        (1 + i // 9) * alpha * (i % 9 - 4)
        + (-1 if i < 9 else 1) * _mp(context, phase_gap) / 2
        + _mp(context, yerror[i])
        for i in range(18)
    )
    return form, phase


def _rows(context, form, phase, joined=True, adapted=True):
    _, _, cross, edges, degrees = _geometry(joined, adapted)
    gradient = [context.mpf(0) for _ in range(18)]
    sine = [context.mpf(0) for _ in range(18)]
    storage = work = context.mpf(0)
    for i, j in edges:
        difference = form[i] - form[j]
        gradient[i] += difference
        gradient[j] -= difference
        current = context.sin(phase[j] - phase[i])
        sine[i] += current
        sine[j] -= current
        energy = difference**2 / 2 + 1 - context.cos(phase[j] - phase[i])
        storage += energy
        if (i, j) in cross:
            work += energy
    gamma = 1 / (1023 * context.pi)
    dx = tuple((-g + gamma * s) / d for g, s, d in zip(gradient, sine, degrees))
    dt = tuple(gamma * g / d for g, d in zip(gradient, degrees))
    return dx, dt, tuple(gradient), tuple(sine), storage, work


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_operator_or_retained_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("mixed-capture controls cannot evolve or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def capture():
    return _assess()


def test_phase_congruence_rebuilds_mixed_support_and_branch_association(capture):
    geometry = capture.geometry
    permutation, internal, cross, edges, degrees = _geometry()
    assert geometry.nodes == tuple(range(18))
    assert geometry.matching_permutation == permutation
    assert sorted(permutation) == list(range(9))
    assert geometry.internal_edges == tuple((i, j, Q(1)) for i, j in internal)
    assert geometry.contact_edges == tuple((i, j, Q(1)) for i, j in cross)
    assert geometry.weighted_edges == tuple((i, j, Q(1)) for i, j in edges)
    assert geometry.degrees == degrees == (3,) * 18
    assert geometry.class_order == (1, 2)
    assert geometry.held_capacities == (Q(1),) * 18
    assert geometry.degree == 3 and geometry.clock_factor == Q(1, 3)
    assert geometry.support_certified and geometry.phase_alignment_certified
    target = tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))
    turns = tuple((2 * (permutation[j] - 4) - (j - 4)) // 9 for j in range(9))
    assert geometry.target_phase_turns == target
    assert geometry.contact_branch_turns == turns
    assert all(target[9 + permutation[j]] - target[j] == turns[j] for j in range(9))
    assert any(turn != 0 for turn in turns)
    # The matching is not a C9 automorphism; no Cartesian spectrum transfers.
    assert any(
        (permutation[(j + 1) % 9] - permutation[j]) % 9 not in (1, 8) for j in range(9)
    )


def test_same_slot_failure_is_only_a_carrier_obstruction(capture, mp):
    geometry = capture.geometry
    expected = tuple(Q(j - 4, 9) for j in range(9))
    assert geometry.same_slot_offset_turns == expected
    assert not geometry.same_slot_carrier_compatible
    form, phase = _state(mp)
    dx, dt, *_ = _rows(mp, form, phase, adapted=False)
    assert max(dx[:9]) - min(dx[:9]) > mp.mpf("1e-4")
    assert all(value == 0 for value in dt)
    roots = tuple(mp.exp(2j * mp.pi * _mp(mp, value)) for value in expected)
    assert abs(sum(roots)) < mp.mpf("1e-190")
    assert any(abs(value - roots[0]) > 1 for value in roots[1:])
    # This rejects the all-state four-coordinate ansatz, not future capture.


def test_all_36_fine_rows_restrict_to_the_live_four_coordinate_law(capture, mp):
    gamma = 1 / (1023 * mp.pi)
    forms = (Q(-2, 7), Q(3, 11))
    form, phase = _state(mp, forms=forms)
    dx, dt, gradient, sine, storage, work = _rows(mp, form, phase)
    difference = _mp(mp, forms[0] - forms[1])
    expected_x = (-difference + gamma * mp.sin(_mp(mp, DELTA))) / 3
    expected_phase = gamma * difference / 3
    for i in range(18):
        sign = 1 if i < 9 else -1
        assert abs(dx[i] - sign * expected_x) < mp.mpf("1e-190")
        assert abs(dt[i] - sign * expected_phase) < mp.mpf("1e-190")
    alpha = 2 * mp.pi / 9
    baseline = 9 * (1 - mp.cos(alpha)) + 9 * (1 - mp.cos(2 * alpha))
    expected_work = 9 * (difference**2 / 2 + 1 - mp.cos(_mp(mp, DELTA)))
    assert abs(work - expected_work) < mp.mpf("1e-190")
    assert abs(storage - baseline - expected_work) < mp.mpf("1e-190")
    _contains(mp, capture.geometry.internal_storage_bounds, baseline)
    loss = sum(g * a - s * b for g, s, a, b in zip(gradient, sine, dx, dt))
    assert abs(loss + sum(g**2 for g in gradient) / 3) < mp.mpf("1e-190")
    # Original time rescales both rows, not only EPI.
    original_x = mp.mpf(1023) * expected_x / 1024
    original_phase = mp.mpf(1023) * expected_phase / 1024
    assert abs(1024 * original_x / 1023 - expected_x) < mp.mpf("1e-190")
    assert abs(1024 * original_phase / 1023 - expected_phase) < mp.mpf("1e-190")


def test_gap_proof_uses_exact_mean_residual_decomposition_not_product_spectrum(mp):
    _, internal, cross, edges, _ = _geometry()
    laplacian = _laplacian(edges)
    cycle = _laplacian(internal)
    matching = _laplacian(cross)
    means = tuple(
        tuple(Q(1, 9) if i // 9 == j // 9 else Q(0) for j in range(18))
        for i in range(18)
    )
    relative_mean = tuple(
        tuple(means[i][j] - Q(1, 18) for j in range(18)) for i in range(18)
    )
    residual = tuple(
        tuple(Q(i == j) - means[i][j] for j in range(18)) for i in range(18)
    )
    assert _product(laplacian, relative_mean) == tuple(
        tuple(2 * value for value in row) for row in relative_mean
    )
    assert _product(cycle, means) == ((Q(0),) * 18,) * 18
    assert _product(matching, means) == _product(means, matching)
    residual_laplacian = _product(_product(residual, laplacian), residual)
    assert laplacian == tuple(
        tuple(2 * relative_mean[i][j] + residual_laplacian[i][j] for j in range(18))
        for i in range(18)
    )
    # Every nonconstant cycle Fourier direction is bounded independently.
    assert all(2 - 2 * mp.cos(2 * mp.pi * k / 9) > mp.mpf(2) / 5 for k in range(1, 9))
    assert all(sum(abs(value) for value in row) == 6 for row in laplacian)


def test_common_metric_dissipates_under_nonuniform_full_rows_without_commutation(
    capture, mp
):
    xerror = tuple(Q((i % 9) - 4, 10000) * (1 if i < 9 else -1) for i in range(18))
    yerror = tuple(
        Q(((i % 9) ** 2) - Q(68, 3), 100000) * (1 if i < 9 else -1) for i in range(18)
    )
    assert sum(xerror) == sum(yerror) == 0
    assert sum(a * b for a, b in zip(xerror, yerror)) != 0
    form, phase = _state(mp, phase_gap=Q(0), xerror=xerror, yerror=yerror)
    _, target = _state(mp, phase_gap=Q(0))
    y = tuple(a - b for a, b in zip(phase, target))
    dx, dt, gradient, sine, *_ = _rows(mp, form, phase)
    gamma = 1 / (1023 * mp.pi)
    metric = (
        mp.mpf(3)
        / 4
        * (
            sum(a * a + b * b for a, b in zip(form, y))
            + 2 * gamma * sum(a * b for a, b in zip(form, y))
        )
    )
    derivative = (
        mp.mpf(3)
        / 2
        * sum(
            a * da + b * db + gamma * (da * b + a * db)
            for a, b, da, db in zip(form, y, dx, dt)
        )
    )
    assert metric > 0
    assert derivative < -_mp(mp, capture.common_metric_decay_rate) * metric
    assert abs(sum(dx)) < mp.mpf("1e-190")
    assert abs(sum(dt)) < mp.mpf("1e-190")
    assert abs(
        sum(g * a - s * b for g, s, a, b in zip(gradient, sine, dx, dt))
        + sum(g**2 for g in gradient) / 3
    ) < mp.mpf("1e-190")
    # Actual secant Hessian weights vary; K and L need not commute.
    _, _, _, edges, _ = _geometry()
    stiffness = [[mp.mpf(0) for _ in range(18)] for _ in range(18)]
    for i, j in edges:
        base, displacement = target[j] - target[i], y[j] - y[i]
        weight = (
            (mp.sin(base + displacement) - mp.sin(base)) / displacement
            if displacement
            else mp.cos(base)
        )
        assert mp.mpf(1) / 20 < weight <= 1 + mp.mpf("1e-190")
        stiffness[i][i] += weight
        stiffness[j][j] += weight
        stiffness[i][j] -= weight
        stiffness[j][i] -= weight
    laplacian = tuple(tuple(_mp(mp, q) for q in row) for row in _laplacian(edges))
    ky = tuple(sum(a * b for a, b in zip(row, y)) for row in stiffness)
    x_l_x = sum(a * b for a, b in zip(form, gradient))
    x_k_y = sum(a * b for a, b in zip(form, ky))
    y_k_y = sum(a * b for a, b in zip(y, ky))
    expected_derivative = (
        -(1 - gamma**2) * x_l_x / 2 - gamma * x_k_y / 2 - gamma**2 * y_k_y / 2
    )
    assert abs(derivative - expected_derivative) < mp.mpf("1e-190")
    assert all(abs(a + b) < mp.mpf("1e-190") for a, b in zip(ky, sine))
    left, right = _product(laplacian, stiffness), _product(stiffness, laplacian)
    assert max(
        abs(a - b) for row, other in zip(left, right) for a, b in zip(row, other)
    ) > mp.mpf("1e-3")


def test_actual_aggregate_rows_keep_fine_correlation_remainder(mp):
    # This law-level point checks actual full rows; it is not an evolved source.
    xerror = tuple(Q((i % 9) - 4, 100000) for i in range(18))
    yerror = tuple(Q((i % 9) - 4, 100000) * (1 if i < 9 else -1) for i in range(18))
    form, phase = _state(
        mp, forms=(Q(-1, 1000), Q(1, 1000)), xerror=xerror, yerror=yerror
    )
    dx, dt, *_ = _rows(mp, form, phase)
    permutation, *_ = _geometry()
    gaps = tuple(
        phase[9 + permutation[j]]
        - phase[j]
        - 2 * mp.pi * ((2 * (permutation[j] - 4) - (j - 4)) // 9)
        for j in range(9)
    )
    mean_gap = sum(gaps) / 9
    u = sum(form[9:]) / 9 - sum(form[:9]) / 9
    du = sum(dx[9:]) / 9 - sum(dx[:9]) / 9
    dj = sum(dt[9:]) / 9 - sum(dt[:9]) / 9
    gamma = 1 / (1023 * mp.pi)
    assert abs(dj - 2 * gamma * u / 3) < mp.mpf("1e-190")
    defect = du + 2 * u / 3 + 2 * gamma * mp.sin(mean_gap) / 3
    centered = tuple(value - mean_gap for value in gaps)
    assert abs(sum(centered)) < mp.mpf("1e-190")
    assert abs(defect) <= gamma * sum(value**2 for value in centered) / 27
    _, target = _state(mp, phase_gap=Q(0))
    full_y2 = sum((a - b) ** 2 for a, b in zip(phase, target))
    assert abs(defect) <= 2 * gamma * full_y2 / 27
    assert abs(defect) > mp.mpf("1e-20")


def test_actual_attachment_work_retains_residuals_and_external_supply(capture, mp):
    xerror = tuple(ETA * Q((i % 9) - 4, 20) for i in range(18))
    yerror = tuple(ETA * Q((i % 9) - 4, 20) * (1 if i < 9 else -1) for i in range(18))
    for error in (xerror, yerror):
        assert sum(error[:9]) == sum(error[9:]) == 0
        assert sum(value**2 for value in error[:9]) < ETA**2
        assert sum(value**2 for value in error[9:]) < ETA**2
    form, phase = _state(mp, xerror=xerror, yerror=yerror)
    joined = _rows(mp, form, phase)
    cut = _rows(mp, form, phase, joined=False)
    nominal = 9 * (1 - mp.cos(_mp(mp, DELTA)))
    _contains(mp, capture.nominal_attachment_work_bounds, nominal)
    assert nominal > 0
    correction = joined[-1] - nominal
    assert 0 < correction < _mp(mp, 4 * ETA**2)
    assert abs(joined[-2] - cut[-2] - joined[-1]) < mp.mpf("1e-190")
    assert capture.attachment_work_correction_upper_bound == 4 * ETA**2


def test_original_sources_wait_without_new_phase_reset_or_tighter_preparation(capture):
    assert capture.source_allowances_within_protocol and capture.source_certified
    assert capture.isolated_waiting_certified
    assert tuple(row.target_winding for row in capture.recoveries) == (1, 2)
    for row in capture.recoveries:
        assert row.deformation == 0
        assert row.form_error_bound == row.phase_error_bound == ERROR
        assert row.formation_time == 100
        assert row.relaxation_duration == 10**13
        assert row.endpoint_radius == EPSILON
        assert row.recovery_certified
    assert capture.recovered_radius == EPSILON
    assert capture.waiting_radius == ETA
    assert capture.isolated_excess_upper_bound == 4 * EPSILON**2
    assert capture.isolated_waiting_storage_margin > 0
    assert 4 * EPSILON**2 / Q(1, 100) == ETA**2


def test_full_state_acute_barrier_retains_both_internal_windings(capture, mp):
    radius = Q(1, 12)
    angle = 4 * mp.pi / 9 + mp.sqrt(2) * _mp(mp, radius)
    assert mp.cos(angle) > mp.mpf(1) / 20
    for ring, winding in enumerate((1, 2)):
        turns = capture.geometry.target_phase_turns[9 * ring : 9 * (ring + 1)]
        differences = tuple(turns[(j + 1) % 9] - turns[j] for j in range(9))
        # The ordinary lift telescopes to zero; the marked circular winding
        # instead uses the same exact principal branch on every cycle edge.
        wrapped = tuple(value - (value + Q(1, 2)) // 1 for value in differences)
        assert sum(differences) == 0
        assert sum(wrapped) == winding
    norm = Q(9, 2) * DELTA**2 + 4 * ETA**2
    assert capture.initial_joint_norm_squared_upper_bound == norm
    assert capture.post_attachment_excess_upper_bound == 3 * norm
    assert capture.phase_hessian_lower_bound == Q(1, 20)
    assert capture.full_graph_gap_lower_bound == Q(2, 5)
    assert capture.storage_coercivity_lower_bound == Q(1, 100)
    assert capture.storage_barrier == Q(1, 14400)
    assert capture.radius_margin == radius**2 - norm > 0
    assert capture.storage_margin == capture.storage_barrier - 3 * norm > 0
    assert capture.constituent_identity_certified


def test_independent_common_metric_deadline_and_all_future_identity(
    capture, record_property
):
    low, high = Q(1, 3300), Q(1, 3000)
    mobility = Q(4, 3)
    lower, upper = (1 - high) / mobility, (1 + high) / mobility
    rate = low**2 / (150 * (1 + high))
    assert capture.common_metric_gamma_bounds == (low, high)
    assert capture.common_metric_clock_matrix_factor == mobility
    assert capture.common_metric_lower_bound == lower
    assert capture.common_metric_upper_bound == upper
    assert upper / lower == Q(3001, 2999) < Q(1001, 1000)
    assert capture.common_metric_decay_rate == rate > Q(6, 10**10)
    assert capture.recovery_duration == HORIZON
    assert capture.recovery_exponent_lower_bound == rate * HORIZON > 12
    assert capture.decay_power == 12
    assert capture.decay_upper_bound == Q(1, 2**12)
    assert (
        capture.initial_metric_upper_bound
        == upper * capture.initial_joint_norm_squared_upper_bound
    )
    returned = upper / lower * capture.initial_joint_norm_squared_upper_bound / 2**12
    assert capture.returned_joint_norm_squared_upper_bound == returned
    assert returned < DELTA**2 / 256
    assert (
        capture.returned_registered_gap_squared_upper_bound
        == 2 * returned
        < DELTA**2 / 64
    )
    assert capture.returned_mean_registration_squared_upper_bound == Q(2, 9) * returned
    assert (
        capture.returned_relative_form_row_defect_upper_bound
        == Q(2, 27) * high * returned
    )
    assert capture.original_recovery_duration == HORIZON * Q(1024, 1023)
    assert capture.joint_return_margin > 0 and capture.registration_margin > 0
    assert capture.joint_identity_certified
    record_property(
        "returned_joint_norm_squared_over_delta_squared", str(returned / DELTA**2)
    )
    record_property("recovery_exponent_lower_bound", str(rate * HORIZON))
    record_property("finite_deadline_tau", str(HORIZON))
    # Monotonic W after the deadline extends this bound to every later time.


def test_initial_identity_absence_and_cut_mean_are_exact_on_same_actual_state(
    capture, mp
):
    assert capture.initial_identity_absent
    assert capture.initial_registered_gap_bounds == (DELTA - 2 * ETA, DELTA + 2 * ETA)
    assert DELTA - 2 * ETA > DELTA / 2
    assert capture.registration_target == DELTA / 2
    assert capture.disconnected_registration_bounds == (DELTA, DELTA)
    xerror = tuple(ETA * Q((i % 9) - 4, 20) for i in range(18))
    yerror = tuple(-value for value in xerror)
    form, phase = _state(mp, xerror=xerror, yerror=yerror)
    dx, dt, *_ = _rows(mp, form, phase, joined=False)
    assert _geometry(joined=False)[-1] == (2,) * 18
    assert all(
        abs(sum(row[start : start + 9])) < mp.mpf("1e-190")
        for row in (dx, dt)
        for start in (0, 9)
    )
    permutation = capture.geometry.matching_permutation
    gaps = tuple(
        phase[9 + permutation[j]]
        - phase[j]
        - 2 * mp.pi * capture.geometry.contact_branch_turns[j]
        for j in range(9)
    )
    assert abs(sum(gaps) / 9 - _mp(mp, DELTA)) < mp.mpf("1e-190")
    assert min(gaps) > _mp(mp, DELTA / 2)


def test_recorded_contrast_has_predeclared_margin_with_correlated_errors(
    capture, record_property
):
    assert capture.connected_registration_bounds == (-DELTA / 8, DELTA / 8)
    assert capture.actual_contrast_bounds == (7 * DELTA / 8, 9 * DELTA / 8)
    expected = (7 * DELTA / 8 - 2 * NOISE, 9 * DELTA / 8 + 2 * NOISE)
    assert capture.recorded_contrast_bounds == expected
    assert capture.response_target == DELTA / 2
    assert capture.contrast_margin == expected[0] - DELTA / 2 > 0
    assert capture.observation_allowance_within_protocol
    assert capture.response_certified and capture.certified
    assert not capture.unresolved_conditions
    record_property("recorded_contrast_lower_bound", str(expected[0]))
    record_property("response_target", str(DELTA / 2))
    record_property("connected_abs_registration_upper_bound", str(DELTA / 8))
    # Opposite errors attain the worst allowance; no independence is assumed.
    assert 7 * DELTA / 8 + (-NOISE) - NOISE == expected[0]


def test_each_call_rebuilds_and_consumes_both_actual_source_classes(monkeypatch):
    actual = owner._assess_constitutive_recovery
    for failing in (1, 2):
        calls = []

        def fresh(**kwargs):
            calls.append(kwargs)
            result = actual(**kwargs)
            return (
                replace(result, recovery_certified=False)
                if kwargs["target_winding"] == failing
                else result
            )

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_assess_constitutive_recovery", fresh)
            report = _assess()
        assert [row["target_winding"] for row in calls] == [1, 2]
        assert all(
            row["deformation"] == 0
            and row["form_error_bound"] == ERROR
            and row["phase_error_bound"] == ERROR
            for row in calls
        )
        assert not report.source_certified
        assert not report.isolated_waiting_certified
        assert not report.joint_identity_certified
        assert not report.response_certified
        assert report.status == "unavailable" and report.unresolved_conditions
    with monkeypatch.context() as patch:
        patch.setattr(
            owner,
            "_assess_constitutive_recovery",
            lambda **kwargs: replace(
                actual(**kwargs), target_winding=3 - kwargs["target_winding"]
            ),
        )
        wrong_association = _assess()
        assert not wrong_association.isolated_waiting_certified
        assert not wrong_association.certified


def test_failed_geometry_or_return_margin_cannot_certify(monkeypatch):
    derive = owner._derive_mixed_constitutive_pair
    metric = owner._sine_common_quadratic_coefficients
    with monkeypatch.context() as patch:
        patch.setattr(
            owner,
            "_derive_mixed_constitutive_pair",
            lambda: replace(derive(), phase_alignment_certified=False),
        )
        report = _assess()
        assert not report.certified and report.unresolved_conditions
    with monkeypatch.context() as patch:

        def no_decay(**kwargs):
            lower, upper, _ = metric(**kwargs)
            return lower, upper, Q(0)

        patch.setattr(owner, "_sine_common_quadratic_coefficients", no_decay)
        report = _assess()
        assert not report.joint_identity_certified
        assert not report.response_certified
        assert report.status == "unavailable" and report.unresolved_conditions


def test_valid_outside_protocol_allowances_are_unavailable_not_obstructions():
    for name, value in (
        ("form_error_bound", ERROR + Q(1, 10**100)),
        ("phase_error_bound", ERROR + Q(1, 10**100)),
        ("observation_error_bound", NOISE + Q(1, 10**100)),
    ):
        report = _assess(**{name: value})
        assert getattr(report, name) == value
        assert not report.certified
        assert report.status == "unavailable" and report.unresolved_conditions
        assert report.recorded_contrast_bounds[0] > DELTA / 2


def test_raw_primitive_rejection_precedes_geometry_source_or_metric(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached scientific certificate work")

    for name in (
        "_derive_mixed_constitutive_pair",
        "_assess_constitutive_recovery",
        "_sine_common_quadratic_coefficients",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for field in ("form_error_bound", "phase_error_bound", "observation_error_bound"):
        for value in (
            True,
            np.bool_(False),
            Q(-1, 10**100),
            float("nan"),
            float("inf"),
            1j,
            "0.0",
            None,
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{field: value})


def test_fixed_width_rational_payloads_normalize_and_tiny_positive_noise_survives(
    capture,
):
    report = _assess(
        form_error_bound=Q(np.int64(1), np.int64(10**10)),
        phase_error_bound=Q(np.int64(1), np.int64(10**10)),
        observation_error_bound=Q(np.int64(1), np.int64(10**6)),
    )
    assert report == capture
    tiny = Q(1, 10**100)
    exact = _assess(observation_error_bound=Q(0))
    noisy = _assess(observation_error_bound=tiny)
    assert noisy.observation_error_bound == tiny
    assert noisy.recorded_contrast_bounds == (
        exact.actual_contrast_bounds[0] - 2 * tiny,
        exact.actual_contrast_bounds[1] + 2 * tiny,
    )


def test_frozen_certificate_accepts_no_cached_source_or_protocol_override(capture):
    with pytest.raises(FrozenInstanceError):
        capture.certified = False
    for extra in (
        dict(source=capture),
        dict(recoveries=capture.recoveries),
        dict(recovery_duration=HORIZON),
        dict(relative_phase_gap=DELTA),
        dict(matching_permutation=tuple(range(9))),
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
