"""Response-free joined return, carried means, work and comparator controls."""

import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.physics import _sine_class_repeated_interaction as owner


@pytest.fixture(scope="module", autouse=True)
def no_response_coefficient_acquisition_or_worker():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("joined return arithmetic must not evaluate scientific producers")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                ("_class_cubic_coefficients", "_time_coefficients"),
            ),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (
                relational_sine_class_comparison_readout,
                ("bound_sine_class_comparison_readout",),
            ),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _arguments():
    # A supplied illustrative interval, not read from any frozen evidence.
    return dict(
        nominal_contrast_lower=Q(-20, 10**30), nominal_contrast_upper=Q(-19, 10**30)
    )


@pytest.fixture(scope="module")
def report():
    return owner._bound_repeated_interaction(**_arguments())


def _graph():
    edges = tuple(
        (9 * component + j, 9 * component + (j + 1) % 9)
        for component in range(3)
        for j in range(9)
    ) + ((4, 13), (13, 22))
    matrix = [[Q(0) for _ in range(27)] for _ in range(27)]
    for i, j in edges:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return edges, tuple(tuple(row) for row in matrix)


def _mv(matrix, vector):
    return tuple(sum((x * y for x, y in zip(row, vector)), Q(0)) for row in matrix)


def _dot(left, right):
    return sum((x * y for x, y in zip(left, right)), Q(0))


def _norm2(vector, degrees):
    return sum((d * x**2 for d, x in zip(degrees, vector)), Q(0))


def test_private_two_scalar_interface_and_explicit_conditional_scope(report):
    parameters = signature(owner._bound_repeated_interaction).parameters
    assert set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    assert "mixed_mean_compatibility_is_an_independent_required_premise" in report.scope
    assert (
        "first_acquired_word_and_initial_dwell_establish_entry_separately"
        in report.scope
    )
    assert report.conditional_repeatability_sufficient
    assert report.status == "conditional_repeatability_sufficient"
    assert not hasattr(report, "acquisition_certified")
    assert not hasattr(report, "mixed_mean_compatibility_certified")


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize(
    "bad",
    (True, np.bool_(False), float("nan"), float("inf"), 1j, "0", Decimal("1e-400")),
)
def test_invalid_primitive_is_rejected_before_geometry(key, bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_repeated_geometry", lambda: pytest.fail("late admission")
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bound_repeated_interaction(**(_arguments() | {key: bad}))


def test_order_is_admitted_before_geometry(monkeypatch):
    monkeypatch.setattr(
        owner, "_repeated_geometry", lambda: pytest.fail("late admission")
    )
    with pytest.raises(ValueError, match="ordered"):
        owner._bound_repeated_interaction(
            nominal_contrast_lower=1, nominal_contrast_upper=0
        )


def test_full_support_weighted_spectrum_and_path_bound(report):
    edges, laplacian = _graph()
    degrees = tuple(int(laplacian[i][i]) for i in range(27))
    geometry = report.geometry
    assert geometry.laplacian == laplacian
    assert geometry.degrees == degrees
    assert sum(degrees) == 58 and degrees[4] == degrees[22] == 3 and degrees[13] == 4
    adjacency = {i: set() for i in range(27)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    distances = {0: 0}
    queue = [0]
    for i in queue:
        for j in adjacency[i]:
            if j not in distances:
                distances[j] = distances[i] + 1
                queue.append(j)
    assert distances[18] == geometry.diameter == 10
    assert report.normalized_gap_lower_bound == Q(4, sum(degrees) * distances[18])
    for i in range(27):
        assert sum(laplacian[i]) == 0
        for j in range(27):
            centered = Q(degrees[i] * int(i == j)) - Q(degrees[i] * degrees[j], 58)
            assert geometry.centered_degree_metric[i][j] == centered
            assert (
                geometry.normalized_gap_slack[i][j] == laplacian[i][j] - centered / 145
            )
            assert (
                geometry.normalized_upper_slack[i][j]
                == 2 * degrees[i] * int(i == j) - laplacian[i][j]
            )
    inverse_root = np.diag(1 / np.sqrt(degrees))
    normalized = inverse_root @ np.array(laplacian, dtype=float) @ inverse_root
    eigenvalues = np.linalg.eigvalsh(normalized)
    assert abs(eigenvalues[0]) < 1e-14
    assert eigenvalues[1] > float(report.normalized_gap_lower_bound)
    assert eigenvalues[-1] < 2


def test_weighted_centering_preserves_actual_means_and_bounds_quotient(report):
    d = report.geometry.degrees
    vector = tuple(Q((i * 7) % 13 - 6, 100) for i in range(27))
    arithmetic_mean = sum(vector, Q(0)) / 27
    degree_mean = _dot(d, vector) / 58
    p0 = tuple(value - arithmetic_mean for value in vector)
    pd = tuple(value - degree_mean for value in vector)
    assert arithmetic_mean != degree_mean
    assert _dot(d, pd) == 0 and sum(p0, Q(0)) == 0
    assert _norm2(pd, d) <= _norm2(p0, d) <= 4 * _dot(p0, p0)
    assert _dot(p0, p0) <= _dot(pd, pd) <= _norm2(pd, d) / 2
    # P_D is not a Euclidean orthogonal projection: replacing it by P_0 is
    # justified only by these inequalities, not by identifying the two means.
    assert _dot(pd, pd) > _dot(p0, p0)


def test_complete_weighted_two_rows_cancel_storage_cross_terms_without_commutation(
    report,
):
    edges, laplacian = _graph()
    d = report.geometry.degrees
    phase_hessian = [[Q(0) for _ in range(27)] for _ in range(27)]
    for index, (i, j) in enumerate(edges):
        cosine = Q(1, 20) + Q(index, 100)
        for row, column, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            phase_hessian[row][column] += sign * cosine
    a = tuple(tuple(value / d[i] for value in row) for i, row in enumerate(laplacian))
    c = tuple(
        tuple(value / d[i] for value in row) for i, row in enumerate(phase_hessian)
    )
    raw_x = tuple(Q((i * 7) % 11 - 5, 100) for i in range(27))
    raw_y = tuple(Q((i * 3) % 17 - 8, 100) for i in range(27))
    x = tuple(value - _dot(d, raw_x) / 58 for value in raw_x)
    y = tuple(value - _dot(d, raw_y) / 58 for value in raw_y)
    assert _mv(a, _mv(c, y)) != _mv(c, _mv(a, y))
    lx, hy = _mv(laplacian, x), _mv(phase_hessian, y)
    gamma = Q(1, 3100)
    dx = tuple(-(u + gamma * v) / degree for u, v, degree in zip(lx, hy, d))
    dy = tuple(gamma * u / degree for u, degree in zip(lx, d))
    assert _dot(d, dx) == _dot(d, dy) == 0
    assert any(value != 0 for value in dy)
    assert _dot(lx, dx) + _dot(hy, dy) == -sum(
        u**2 / degree for u, degree in zip(lx, d)
    )
    # Omitting the phase row would retain a nonzero spurious cross term.
    assert _dot(lx, dx) != -sum(u**2 / degree for u, degree in zip(lx, d))


def test_exact_joined_decay_constants_and_subgrid_half_radius(report):
    lam, cap, cosine = Q(1, 145), Q(2), Q(1, 20)
    eta_low, eta_high, beta = Q(1, 11000000), Q(1, 9000000), lam / 4
    mu = eta_low * cosine * lam**2
    amin = mu / 2 + lam**2 / 16
    aplus = eta_high * cap**2 / 2 + beta * cap / 2 + beta**2
    rate = min(4 * (lam - beta) / 3, beta * mu / aplus)
    v0 = Q(3, 4) * eta_high * cap / 36 + aplus / (36 * lam)
    assert mu == report.phase_stiffness_lower_bound == Q(1, 4625500000000)
    assert amin == report.lyapunov_position_lower_coefficient
    assert aplus == report.lyapunov_position_upper_coefficient == Q(6537091, 3784500000)
    assert rate == report.lyapunov_decay_rate == Q(9, 41706640580000)
    assert v0 == report.lyapunov_initial_upper_bound == Q(130741907, 18792000000)
    assert report.relaxation_duration == 256 * 5 * 10**12
    assert rate > Q(1, 5 * 10**12) and report.dwell_exponent_margin > 0
    decay = Q(1, 2**256)
    assert report.decay_upper_bound == decay
    assert report.returned_lyapunov_upper_bound == v0 * decay
    form = 4 * v0 * decay / (eta_low * lam)
    phase = 2 * v0 * decay / amin
    assert report.returned_form_d_norm_squared_upper_bound == form
    assert report.returned_phase_d_norm_squared_upper_bound == phase
    assert 0 < form < Q(551, 10**72) < report.epsilon**2 / 4 < Q(1, 2**128)
    assert 0 < phase < Q(582, 10**76) < report.epsilon**2 / 4
    assert report.nonlinear_return_sufficient and report.tangent_return_sufficient


def test_elementary_pi_sine_bounds_supply_the_declared_gamma_and_cosine(report):
    # Independent alternating arctangent bounds in Machin's identity. No
    # interval-grid exponential, target solver or trajectory is evaluated.
    pi_lower = 16 * (Q(1, 5) - Q(1, 3 * 5**3)) - Q(4, 239)
    pi_upper = 16 * (Q(1, 5) - Q(1, 3 * 5**3) + Q(1, 5 * 5**5)) - 4 * (
        Q(1, 239) - Q(1, 3 * 239**3)
    )
    assert Q(31, 10) < pi_lower < pi_upper < Q(22, 7)
    assert report.eta_lower_bound < 1 / (1023 * pi_upper) ** 2
    assert 1 / (1023 * pi_lower) ** 2 < report.eta_upper_bound
    assert 1 / (1023 * pi_lower) < report.gamma_upper_bound
    root_two_upper = Q(17, 12)
    assert root_two_upper**2 > 2
    acute_margin = Q(31, 10) / 18 - root_two_upper * report.radius
    assert acute_margin == Q(13, 240)
    assert acute_margin - acute_margin**3 / 6 > report.cosine_lower_bound


def test_recurrent_work_and_identity_use_carried_pressure_without_reattachment(report):
    a, eps, g = Q(7, 10000), Q(1, 10**32), Q(1, 3000)
    assert report.initial_excess_storage_upper_bound == 2 * eps**2
    assert report.storage_barrier == Q(1, 388800)
    assert report.recurring_contact_work == 0
    for history, (first, second) in zip(
        report.histories, ((0, 0), (a, 0), (0, a), (a, a))
    ):
        pre_x = (first + eps + 2 * g * eps) / (1 - 2 * g**2)
        defect = eps + 2 * g * eps + 2 * g**2 * pre_x
        pressure = -6 * defect, Q(9, 8) * first + 6 * defect
        assert history.pre_second_laplacian_bounds == pressure
        assert history.first_work_bounds == (
            Q(3, 2) * first**2 - 6 * first * eps,
            Q(3, 2) * first**2 + 6 * first * eps,
        )
        assert history.second_work_bounds == tuple(
            Q(3, 2) * second**2 + second * value for value in pressure
        )
        x = (first + second + eps + 4 * g * eps) / (1 - 8 * g**2)
        y = eps + 4 * g * x
        assert history.after_second_radius_squared_upper_bound == 27 * (x**2 + y**2)
        assert history.final_form_d_norm_squared_upper_bound == 58 * x**2
        assert history.final_phase_d_norm_squared_upper_bound == 58 * y**2
        assert (
            history.after_second_excess_storage_upper_bound
            == 2 * eps**2 + history.first_work_bounds[1] + history.second_work_bounds[1]
        )
        assert history.first_work_bounds[1] < Q(736, 10**9)
        assert history.second_work_bounds[1] < Q(1287, 10**9)
        assert (
            history.after_second_excess_storage_upper_bound
            < Q(2022, 10**9)
            < report.storage_barrier
        )
        assert (
            history.after_second_radius_squared_upper_bound
            < Q(5293, 10**8)
            < report.radius**2
        )
        assert history.identity_sufficient and history.work_within_allowances
        assert history.tangent_endpoint_within_return_budget


def test_full_state_jump_work_and_mean_increment_independent_of_common_offset(report):
    _, laplacian = _graph()
    d = report.geometry.degrees
    before = tuple(Q((5 * i) % 19 - 9, 10**9) + Q(17, 3) for i in range(27))
    amplitude = report.first_probe_amplitude
    after = tuple(value + amplitude * int(i == 4) for i, value in enumerate(before))
    before_energy = _dot(before, _mv(laplacian, before)) / 2
    after_energy = _dot(after, _mv(laplacian, after)) / 2
    assert (
        after_energy - before_energy
        == amplitude * _mv(laplacian, before)[4] + Q(3, 2) * amplitude**2
    )
    mean_change = _dot(d, after) / 58 - _dot(d, before) / 58
    assert mean_change == Q(3, 58) * amplitude
    assert report.histories[1].form_mean_increment == mean_change
    assert all(h.phase_mean_increment == 0 for h in report.histories)
    assert (
        _dot(
            report.mixed_coefficients,
            tuple(h.form_mean_increment for h in report.histories),
        )
        == 0
    )
    for repetition in (0, 1, 2, 10**20):
        means = tuple(
            Q(17, 3) + repetition * h.form_mean_increment for h in report.histories
        )
        assert _dot(report.mixed_coefficients, means) == 0


def test_relative_return_alone_does_not_restore_mixed_mean_compatibility(report):
    d, laplacian = report.geometry.degrees, report.geometry.laplacian
    # Add one uniform form only to class1's neither branch. It has zero
    # relative radius/storage and is stationary, but changes D by exactly1.
    branch_forms = (tuple(Q(1) for _ in range(27)),) + (
        tuple(Q(0) for _ in range(27)),
    ) * 7
    outputs = []
    for branch in branch_forms:
        mean = _dot(d, branch) / 58
        centered = tuple(value - mean for value in branch)
        assert _norm2(centered, d) == 0
        assert _mv(laplacian, branch) == (Q(0),) * 27
        outputs.append(branch[22])
    coefficients = report.mixed_coefficients + tuple(
        -value for value in report.mixed_coefficients
    )
    assert _dot(coefficients, outputs) == 1
    assert 1 > -report.nominal_contrast_bounds[0]


def test_each_model_keeps_its_own_eight_branch_source_and_reading_allowance(report):
    source = 8 * report.epsilon / (1 - 4 * Q(1, 3000))
    assert report.nonlinear_source_error_upper_bound == source
    assert report.tangent_source_error_upper_bound == source
    lo, hi = report.nominal_contrast_bounds
    delta = report.readout_error_bound
    assert report.true_contrast_bounds == (lo - source, hi + source)
    assert report.recorded_contrast_bounds == (
        lo - source - 8 * delta,
        hi + source + 8 * delta,
    )
    assert report.recorded_tangent_bounds == (-source - 8 * delta, source + 8 * delta)
    assert report.separation_margin == -hi - 2 * source - 16 * delta
    assert (
        report.separation_margin
        == report.recorded_tangent_bounds[0] - report.recorded_contrast_bounds[1]
    )
    assert report.separation_sufficient


@pytest.mark.parametrize("side", (-1, 0, 1))
def test_strict_separation_boundary_keeps_arbitrarily_small_exact_margin(report, side):
    threshold = (
        -2 * report.nonlinear_source_error_upper_bound - 16 * report.readout_error_bound
    )
    upper = threshold - side * Q(1, 2**600)
    result = owner._bound_repeated_interaction(
        nominal_contrast_lower=upper - Q(1, 10**30), nominal_contrast_upper=upper
    )
    assert result.separation_margin == side * Q(1, 2**600)
    assert result.separation_sufficient is (side > 0)
    assert result.conditional_repeatability_sufficient is (side > 0)
    assert result.nonlinear_return_sufficient and result.tangent_return_sufficient
    assert result.all_work_within_allowances
    assert result.status == (
        "conditional_repeatability_sufficient" if side > 0 else "bounds_only"
    )


def test_nonnegative_or_broad_nominal_interval_does_not_fabricate_separation():
    result = owner._bound_repeated_interaction(
        nominal_contrast_lower=-1, nominal_contrast_upper=1
    )
    assert result.status == "bounds_only" and not result.separation_sufficient
    assert result.unmet_sufficient_requirements == (
        "strict_recorded_model_separation_not_certified",
    )
    assert (
        result.all_histories_trapped_conditionally
        and result.nonlinear_return_sufficient
    )


def test_boolean_spectral_approval_is_not_substituted_for_fixed_graph_failure(
    monkeypatch,
):
    monkeypatch.setattr(owner, "exact_symmetric_semidefinite", lambda matrix: False)
    with pytest.raises(ArithmeticError, match="spectral"):
        owner._bound_repeated_interaction(**_arguments())
