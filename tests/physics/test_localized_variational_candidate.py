"""Independent finite polynomial controls for an uncertified modal proposal."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import numpy as np
import pytest
from scipy.linalg import expm

import tnfr.physics._localized_variational_candidate as owner
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I
from tnfr.physics._localized_variational_panels import _bound_cycle_variational_panel


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response():
    with forbid_organized_response_evaluation():
        yield


def _integral_matrix(degree):
    """Integrate each basis element explicitly, then restore its left zero."""
    result = np.zeros((degree + 1, degree + 1))
    for column in range(degree + 1):
        if column == 0:
            result[1, column] = 1
        elif column == 1:
            if degree >= 2:
                result[2, column] = Q(1, 4)
        else:
            result[column - 1, column] = -Q(1, 2 * (column - 1))
            if column < degree:
                result[column + 1, column] = Q(1, 2 * (column + 1))
        result[0, column] = -sum(
            (-1) ** index * result[index, column] for index in range(1, degree + 1)
        )
    return result


def _direct_product(left, right):
    result = np.zeros(len(left) + len(right) - 1)
    for j, first in enumerate(left):
        for k, second in enumerate(right):
            result[j + k] += first * second / 2
            result[abs(j - k)] += first * second / 2
    return result


def _dense_integrated_solution(
    source_form,
    source_phase,
    edges,
    sigma,
    duration,
    initial_form=None,
    initial_phase=None,
):
    """Assemble both real rows directly; no Fourier/modal proposal code is used."""
    size, count = source_form.shape
    integration = _integral_matrix(count - 1)
    laplacian = np.eye(size)
    for index in range(size):
        laplacian[index, (index - 1) % size] -= 0.5
        laplacian[index, (index + 1) % size] -= 0.5

    def integrated_action(vector):
        rows = vector.reshape(2 * size, count)
        form, phase = rows[:size], rows[size:]
        gap = phase - np.roll(phase, -1, axis=0)
        flux = np.array(
            [_direct_product(edge, value)[:count] for edge, value in zip(edges, gap)]
        )
        feedback = (flux - np.roll(flux, 1, axis=0)) / 2
        rate = np.vstack(
            (-sigma * laplacian @ form - laplacian @ phase + feedback, laplacian @ form)
        )
        return (duration / 2 * rate @ integration.T).ravel()

    dimension = 2 * size * count
    identity = np.eye(dimension)
    matrix = identity - np.column_stack(
        [integrated_action(identity[:, column]) for column in range(dimension)]
    )
    right = -duration / 2 * np.vstack((source_form, source_phase)) @ integration.T
    if initial_form is not None:
        right[:size, 0] += initial_form
        right[size:, 0] += initial_phase
    return np.linalg.solve(matrix, right.ravel()).reshape(2 * size, count)


def _rows(report):
    return np.array(report.form_coefficients + report.phase_coefficients, dtype=float)


def test_integral_convention_keeps_zero_left_endpoint_and_dropped_top_term():
    for degree in (1, 2, 5, 12):
        matrix = owner._integration_matrix(degree)
        np.testing.assert_allclose(matrix, _integral_matrix(degree), atol=1e-16)
        np.testing.assert_allclose(
            (-1.0) ** np.arange(degree + 1) @ matrix, 0, atol=1e-16
        )
    # Degree2 integration of T2 drops T3 and re-centers the constant term.
    np.testing.assert_array_equal(owner._integration_matrix(2)[:, 2], (-0.5, -0.5, 0))


def test_constant_mean_forcing_has_the_independent_linear_solution():
    size, duration = 8, Q(1, 2)
    result = owner._construct_zero_initial_cycle_candidate(
        ((2,),) * size,
        ((-3,),) * size,
        tuple((Q(index + 1, 100), Q(-1, 200)) for index in range(size)),
        sigma=Q(1, 8),
        duration=duration,
        degree=8,
    )
    expected = np.zeros((2 * size, 9))
    expected[:size, :2] = -float(duration)
    expected[size:, :2] = 3 * float(duration) / 2
    np.testing.assert_allclose(_rows(result), expected, rtol=0, atol=2e-15)
    assert result.feedback_iterations == 20
    assert result.factor_count == size // 2
    for row in result.form_coefficients + result.phase_coefficients:
        assert (
            sum(((-1) ** index * value for index, value in enumerate(row)), Q(0)) == 0
        )
        assert all(isinstance(value, Q) for value in row)


@pytest.mark.parametrize("size,sigma", [(5, 0.0), (6, 0.2)])
def test_both_real_rows_and_all_modes_match_independent_dense_equation(size, sigma):
    degree = 6
    rng = np.random.default_rng(5102 + size)
    source_form = rng.integers(-5, 6, size=(size, degree + 1)).astype(float) / 64
    source_phase = rng.integers(-7, 8, size=(size, degree + 1)).astype(float) / 64
    edges = np.zeros((size, degree + 1))
    edges[:, :3] = rng.integers(-2, 3, size=(size, 3)) / 128
    duration = 0.25
    result = owner._construct_zero_initial_cycle_candidate(
        source_form, source_phase, edges, sigma=sigma, duration=duration, degree=degree
    )
    expected = _dense_integrated_solution(
        source_form, source_phase, edges, sigma, duration
    )
    np.testing.assert_allclose(_rows(result), expected, rtol=2e-12, atol=2e-14)
    assert result.factor_count == size // 2


def test_oversampled_edge_product_retains_high_modes_before_proposal_projection():
    degree, size = 4, 5
    phase = np.zeros((size, degree + 1))
    edge = np.zeros_like(phase)
    phase[:, degree] = (1, -2, 3, -1, 2)
    edge[:, degree] = (2, -1, 1, 3, -2)
    values = owner._lobatto_values(edge, 2 * degree)
    actual = owner._feedback_coefficients(phase, values, degree)
    gap = phase - np.roll(phase, -1, axis=0)
    full_flux = np.array([_direct_product(a, b) for a, b in zip(edge, gap)])
    full = (full_flux - np.roll(full_flux, 1, axis=0)) / 2
    np.testing.assert_allclose(actual, full[:, : degree + 1], atol=3e-15)
    assert np.max(np.abs(full[:, 2 * degree])) > 1
    # Sampling only degree+1 points would alias T_(2d) into the constant mode.
    assert np.max(np.abs(actual[:, 0] - full[:, 0] - full[:, 2 * degree])) > 1


def test_interval_midpoints_are_proposal_choices_and_report_has_no_verdict():
    result = owner._construct_zero_initial_cycle_candidate(
        ((I(1, 3),),) * 3,
        ((0,),) * 3,
        ((0,),) * 3,
        sigma=I(0, Q(1, 2)),
        duration=I(Q(1, 4), Q(3, 4)),
        degree=2,
    )
    assert result.sigma_center == Q(1, 4)
    assert result.duration_center == Q(1, 2)
    assert result.total_seconds >= result.factorization_seconds >= 0
    assert result.total_seconds >= result.feedback_seconds >= 0
    assert not hasattr(result, "certified")
    assert not hasattr(result, "retention")
    assert (
        "no_convergence_error_bound_canonical_admission_or_retention_verdict"
        in result.scope
    )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
@pytest.mark.parametrize("field", ["form", "phase", "edge", "sigma", "duration"])
def test_invalid_primitives_reject_before_numerical_work(monkeypatch, value, field):
    arguments = dict(
        source_form_coefficients=((0,),) * 3,
        source_phase_coefficients=((0,),) * 3,
        edge_coefficient_polynomials=((0,),) * 3,
        sigma=0,
        duration=1,
        degree=2,
    )
    keys = dict(
        form="source_form_coefficients",
        phase="source_phase_coefficients",
        edge="edge_coefficient_polynomials",
    )
    arguments[keys.get(field, field)] = (
        ((0,), (0,), (value,)) if field in keys else value
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid primitive reached binary64 proposal construction")

    monkeypatch.setattr(owner, "_center_table", forbidden)
    monkeypatch.setattr(owner, "_represented_midpoint", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._construct_zero_initial_cycle_candidate(**arguments)


@pytest.mark.parametrize("degree", [True, 2.0, 0, 97])
def test_degree_policy_rejects_invalid_values(degree):
    with pytest.raises(ValueError, match="degree"):
        owner._construct_zero_initial_cycle_candidate(
            ((0,),) * 3, ((0,),) * 3, ((0,),) * 3, sigma=0, duration=1, degree=degree
        )


def test_midpoint_overflow_is_rejected_before_modal_arithmetic():
    with pytest.raises(ValueError, match="representable"):
        owner._construct_zero_initial_cycle_candidate(
            ((Q(10) ** 400,),) * 3,
            ((0,),) * 3,
            ((0,),) * 3,
            sigma=0,
            duration=1,
            degree=2,
        )


def test_row_shape_declared_degree_and_whole_scalar_intervals_are_admitted():
    base = dict(
        source_form_coefficients=((0,),) * 3,
        source_phase_coefficients=((0,),) * 3,
        edge_coefficient_polynomials=((0,),) * 3,
        sigma=0,
        duration=1,
        degree=2,
    )
    for key, value in (
        ("source_phase_coefficients", ((0,),) * 4),
        ("edge_coefficient_polynomials", ((0, 0, 0, 0),) * 3),
        ("sigma", I(0, 2)),
        ("duration", I(0, 1)),
        ("duration", 17),
    ):
        with pytest.raises(ValueError):
            owner._construct_zero_initial_cycle_candidate(**(base | {key: value}))


def test_twenty_feedback_updates_are_fixed_even_for_the_zero_proposal(monkeypatch):
    actual_feedback = owner._feedback_coefficients
    updates = []

    def count_update(phase, edge_values, degree):
        updates.append(degree)
        return actual_feedback(phase, edge_values, degree)

    monkeypatch.setattr(owner, "_feedback_coefficients", count_update)
    result = owner._construct_zero_initial_cycle_candidate(
        ((0,),) * 4, ((0,),) * 4, ((0,),) * 4, sigma=Q(1, 4), duration=1, degree=3
    )
    assert updates == [3] * 20
    assert result.feedback_iterations == len(updates)
    assert all(
        value == 0
        for row in result.form_coefficients + result.phase_coefficients
        for value in row
    )


def test_projected_fixed_point_still_requires_independent_continuous_defect():
    # B_1 T_1=0 after dropping the quadratic antiderivative. All twenty
    # updates therefore leave the zero polynomial fixed, although its true
    # equation defect contains the entire nonzero linear source.
    source, zero = ((0, 1),) * 3, ((0,),) * 3
    candidate = owner._construct_zero_initial_cycle_candidate(
        source, zero, zero, sigma=0, duration=1, degree=1
    )
    assert all(
        value == 0
        for row in candidate.form_coefficients + candidate.phase_coefficients
        for value in row
    )
    bounds = _bound_cycle_variational_panel(
        candidate.form_coefficients,
        candidate.phase_coefficients,
        source,
        zero,
        zero,
        sigma=0,
        duration=1,
    )
    assert bounds.candidate_norm_upper == 0
    assert 3 <= bounds.polynomial_defect_norm_upper**2 < 4
    assert bounds.conditional_integrated_defect_upper > 1


@pytest.mark.parametrize("field", ["form", "phase", "edge", "sigma", "duration"])
@pytest.mark.parametrize("endpoints", [(True, 1), (0, float("nan")), (2, 1)])
def test_forged_interval_endpoints_reject_before_float_conversion(
    field, endpoints, monkeypatch
):
    value = object.__new__(I)
    object.__setattr__(value, "lo", endpoints[0])
    object.__setattr__(value, "hi", endpoints[1])
    arguments = dict(
        source_form_coefficients=((0,),) * 3,
        source_phase_coefficients=((0,),) * 3,
        edge_coefficient_polynomials=((0,),) * 3,
        sigma=0,
        duration=1,
        degree=2,
    )
    keys = dict(
        form="source_form_coefficients",
        phase="source_phase_coefficients",
        edge="edge_coefficient_polynomials",
    )
    arguments[keys.get(field, field)] = (
        ((0,), (0,), (value,)) if field in keys else value
    )

    def forbidden(*args, **kwargs):
        pytest.fail("forged primitive reached midpoint materialization")

    monkeypatch.setattr(owner, "_represented_midpoint", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._construct_zero_initial_cycle_candidate(**arguments)


def test_represented_loss_must_remain_inside_its_open_upper_boundary():
    with pytest.raises(ValueError, match="numerical domain"):
        owner._construct_zero_initial_cycle_candidate(
            ((0,),) * 3,
            ((0,),) * 3,
            ((0,),) * 3,
            sigma=2 - Q(1, 2**110),
            duration=1,
            degree=2,
        )


def test_finite_midpoints_do_not_authorize_overflowing_proposal_arithmetic():
    with pytest.raises(ArithmeticError):
        owner._construct_zero_initial_cycle_candidate(
            ((Q(10) ** 308,),) * 3,
            ((0,),) * 3,
            ((0,),) * 3,
            sigma=0,
            duration=16,
            degree=2,
        )


def test_candidate_detaches_inputs_and_returns_immutable_exact_coefficients():
    source = [[1], [1], [1]]
    result = owner._construct_zero_initial_cycle_candidate(
        source, ((0,),) * 3, ((0,),) * 3, sigma=0, duration=1, degree=2
    )
    before = result.form_coefficients
    source[0][0] = True
    assert result.form_coefficients == before
    assert isinstance(before, tuple) and all(isinstance(row, tuple) for row in before)
    with pytest.raises(FrozenInstanceError):
        result.feedback_iterations = 0
    assert all(
        sum(((-1) ** index * value for index, value in enumerate(row)), Q(0)) == 0
        for row in before
    )


def _endpoints(report, *, initial=False):
    return tuple(
        sum(
            (
                (-1 if initial and index % 2 else 1) * value
                for index, value in enumerate(row)
            ),
            Q(0),
        )
        for row in report.form_coefficients + report.phase_coefficients
    )


def _carried_arguments(size=5, degree=8):
    return dict(
        source_form_coefficients=((0,),) * size,
        source_phase_coefficients=((0,),) * size,
        edge_coefficient_polynomials=((0,),) * size,
        initial_form=tuple(Q(index - 2, 7) for index in range(size)),
        initial_phase=tuple(Q(2 - index, 11) for index in range(size)),
        sigma=Q(1, 8),
        duration=Q(1, 4),
        degree=degree,
    )


@pytest.mark.parametrize("size,sigma", [(5, 0.0), (6, 0.2)])
def test_nonzero_incoming_rows_match_independent_dense_polynomial_equation(size, sigma):
    degree = 6
    rng = np.random.default_rng(734 + size)
    source_form = rng.integers(-5, 6, size=(size, degree + 1)) / 64
    source_phase = rng.integers(-7, 8, size=(size, degree + 1)) / 64
    edges = np.zeros((size, degree + 1))
    edges[:, :3] = rng.integers(-2, 3, size=(size, 3)) / 128
    initial_form = rng.integers(-4, 5, size=size) / 16
    initial_phase = rng.integers(-6, 7, size=size) / 16
    result = owner._construct_cycle_candidate(
        source_form,
        source_phase,
        edges,
        initial_form=initial_form,
        initial_phase=initial_phase,
        sigma=sigma,
        duration=0.25,
        degree=degree,
    )
    expected = _dense_integrated_solution(
        source_form, source_phase, edges, sigma, 0.25, initial_form, initial_phase
    )
    np.testing.assert_allclose(_rows(result), expected, rtol=2e-12, atol=2e-14)
    assert _endpoints(result, initial=True) == tuple(initial_form) + tuple(
        initial_phase
    )


def _dense_homogeneous_generator(size, sigma):
    incidence = np.eye(size) - np.roll(np.eye(size), 1, axis=1)
    laplacian = incidence.T @ incidence / 2
    return np.block(
        [
            [-float(sigma) * laplacian, -laplacian],
            [laplacian, np.zeros_like(laplacian)],
        ]
    )


def test_nonzero_homogeneous_carry_matches_actual_damped_two_row_flow():
    arguments = _carried_arguments(degree=14)
    result = owner._construct_cycle_candidate(**arguments)
    generator = _dense_homogeneous_generator(result.node_count, arguments["sigma"])
    initial = np.array(
        arguments["initial_form"] + arguments["initial_phase"], dtype=float
    )
    expected = expm(float(arguments["duration"]) * generator) @ initial
    np.testing.assert_allclose(
        np.array(_endpoints(result), dtype=float), expected, atol=2e-15
    )
    assert (
        _endpoints(result, initial=True)
        == arguments["initial_form"] + arguments["initial_phase"]
    )


def test_nonzero_mean_carry_and_constant_forcing_have_independent_affine_solution():
    size, duration = 5, Q(1, 2)
    incoming_form, incoming_phase = Q(1, 3), Q(1, 2**150)
    result = owner._construct_cycle_candidate(
        ((2,),) * size,
        ((-3,),) * size,
        tuple((Q(index + 1, 100), Q(-1, 200)) for index in range(size)),
        initial_form=(incoming_form,) * size,
        initial_phase=(incoming_phase,) * size,
        sigma=Q(1, 8),
        duration=duration,
        degree=8,
    )
    assert (
        _endpoints(result, initial=True)
        == (incoming_form,) * size + (incoming_phase,) * size
    )
    assert result.initial_form == (incoming_form,) * size
    assert result.initial_phase == (incoming_phase,) * size
    expected = np.zeros((2 * size, 9))
    expected[:size, 0] = float(incoming_form - duration)
    expected[:size, 1] = -float(duration)
    expected[size:, 0] = float(incoming_phase + 3 * duration / 2)
    expected[size:, 1] = float(3 * duration / 2)
    np.testing.assert_allclose(_rows(result), expected, rtol=0, atol=2e-15)
    assert result.method == "carried_cycle_chebyshev_modal_binary64_proposal_v1"
    assert not hasattr(result, "certified")


def test_two_panels_carry_exact_rational_endpoint_without_reset_or_free_recentring():
    arguments = _carried_arguments(size=5, degree=12)
    first = owner._construct_cycle_candidate(**arguments)
    endpoint = _endpoints(first)
    second = owner._construct_cycle_candidate(
        **(arguments | {"initial_form": endpoint[:5], "initial_phase": endpoint[5:]})
    )
    assert _endpoints(second, initial=True) == endpoint
    assert any(endpoint)
    generator = _dense_homogeneous_generator(5, arguments["sigma"])
    initial = np.array(
        arguments["initial_form"] + arguments["initial_phase"], dtype=float
    )
    expected = expm(2 * float(arguments["duration"]) * generator) @ initial
    np.testing.assert_allclose(
        np.array(_endpoints(second), dtype=float),
        expected,
        rtol=0,
        atol=3e-15,
    )
    # The continuous validator still recomputes each complete residual.
    for candidate in (first, second):
        bounds = _bound_cycle_variational_panel(
            candidate.form_coefficients,
            candidate.phase_coefficients,
            arguments["source_form_coefficients"],
            arguments["source_phase_coefficients"],
            arguments["edge_coefficient_polynomials"],
            sigma=arguments["sigma"],
            duration=arguments["duration"],
        )
        assert bounds.polynomial_defect_norm_upper < Q(1, 10**12)
    # Merely overlapping interval endpoint observations would not prove this.
    changed = list(endpoint[:5])
    changed[0] += Q(1, 2**160)
    shifted = owner._construct_cycle_candidate(
        **(arguments | {"initial_form": changed, "initial_phase": endpoint[5:]})
    )
    difference = _endpoints(shifted, initial=True)[0] - endpoint[0]
    assert difference == Q(1, 2**160)


@pytest.mark.parametrize("field", ["initial_form", "initial_phase"])
@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1", I(0), I(0, 1)])
def test_late_invalid_incoming_coordinate_rejects_before_materialization(
    field, bad, monkeypatch
):
    arguments = _carried_arguments()
    arguments[field] = (0, 0, 0, 0, bad)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid incoming coordinate reached proposal materialization")

    monkeypatch.setattr(owner, "_represented_midpoint", forbidden)
    monkeypatch.setattr(owner, "_center_table", forbidden)
    monkeypatch.setattr(owner, "_represented_value", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._construct_cycle_candidate(**arguments)


@pytest.mark.parametrize("bad", [(), (0,) * 4, {0: 1}, {1, 2}, "12345", iter((0,) * 5)])
def test_incoming_rows_require_complete_ordered_point_state(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._construct_cycle_candidate(
            **(_carried_arguments() | {"initial_form": bad})
        )


@pytest.mark.parametrize("bad", [Q(10) ** 400, Q(1, 2**1100)])
def test_unrepresentable_incoming_point_rejects_without_zero_coercion(bad):
    with pytest.raises(ValueError, match="representable"):
        owner._construct_cycle_candidate(
            **(_carried_arguments() | {"initial_phase": (0, 0, 0, 0, bad)})
        )


def test_zero_initial_adapter_shares_generic_solver_and_preserves_legacy_scope(
    monkeypatch,
):
    original = owner._construct_cycle_candidate
    observed = []

    def record(*args, **kwargs):
        observed.append((kwargs["initial_form"], kwargs["initial_phase"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_construct_cycle_candidate", record)
    result = owner._construct_zero_initial_cycle_candidate(
        ((1,),) * 3, ((0,),) * 3, ((0,),) * 3, sigma=0, duration=1, degree=2
    )
    assert observed == [((0, 0, 0), (0, 0, 0))]
    assert result.method == "zero_initial_cycle_chebyshev_modal_binary64_proposal_v1"
    assert (
        "zero_initial_endpoint_enforced_after_exact_represented_coefficient_admission"
        in result.scope
    )


def test_incoming_points_are_detached_from_mutable_callers():
    arguments = _carried_arguments()
    incoming = list(arguments["initial_form"])
    result = owner._construct_cycle_candidate(
        **(arguments | {"initial_form": incoming})
    )
    incoming[-1] = True
    assert result.initial_form == arguments["initial_form"]
    assert (
        _endpoints(result, initial=True)
        == arguments["initial_form"] + arguments["initial_phase"]
    )
