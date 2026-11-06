"""Independent Cayley-source, acute capture and transient-winding controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.physics import _sine_preparation_mode_selection as previous_owner
from tnfr.physics import _sine_preparation_path_selection as owner
from tnfr.physics import relational_sine_formed_classes as sine_owner

ERROR, TIME, M = Q(1, 10**10), Q(200), 512
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
CORRIDORS = ((Q(0), Q(1, 8)), (Q(3, 4), Q(4, 5)), (Q(63, 64), Q(1)))
EDGES = tuple((j, (j + 1) % 9) for j in range(9))
IDENTITY = tuple(tuple(Q(int(i == j)) for j in range(9)) for i in range(9))


def _multiply(left, right):
    return tuple(
        tuple(sum((left[i][k] * right[k][j] for k in range(9)), Q(0)) for j in range(9))
        for i in range(9)
    )


def _apply(matrix, vector):
    return tuple(sum((x * y for x, y in zip(row, vector)), Q(0)) for row in matrix)


def _laplace(vector):
    return tuple(
        2 * vector[i] - vector[(i - 1) % 9] - vector[(i + 1) % 9] for i in range(9)
    )


def _energy(vector):
    return sum(((vector[j] - vector[i]) ** 2 / 2 for i, j in EDGES), Q(0))


def _source():
    ramp = tuple(Q(j - 4) for j in range(9))
    pulse = tuple(Q(int(j == 3) - int(j == 4)) for j in range(9))
    first = tuple((9 * M + Q(1, M)) * h + r for h, r in zip(pulse, ramp))
    second = tuple((9 * M - Q(1, M)) * h + 2 * r for h, r in zip(pulse, ramp))
    u, v = tuple(a + b for a, b in zip(first, second)), tuple(
        b - a for a, b in zip(first, second)
    )
    lu, lv = _laplace(u), _laplace(v)
    square = sum((x * y for x, y in zip(u, lu)), Q(0))
    generator = tuple(
        tuple((v[i] * lu[j] - u[i] * lv[j]) / square for j in range(9))
        for i in range(9)
    )
    return first, second, generator


def _cayley(generator, parameter):
    minus = tuple(
        tuple(IDENTITY[i][j] - parameter * generator[i][j] for j in range(9))
        for i in range(9)
    )
    plus = tuple(
        tuple(IDENTITY[i][j] + parameter * generator[i][j] for j in range(9))
        for i in range(9)
    )
    return _multiply(exact_matrix_inverse(minus), plus)


def _profile(parameter):
    kappa = Q(1, 27 * M**2)
    denominator = 1 + kappa * parameter**2
    a = 1 - kappa * parameter**2 + 2 * kappa * parameter
    b = 2 * parameter - 1 + kappa * parameter**2
    return (9 * M * a - b / M) / denominator, (3 * a + b) / (2 * denominator)


@pytest.fixture(scope="module", autouse=True)
def no_previous_verdict_or_scientific_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("source path borrowed a prior acquisition verdict")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(previous_owner, "_assess_preparation_mode_selection", forbidden)
        patch.setattr(sine_owner, "assess_sine_formed_class_pair", forbidden)
        yield


def _assess(parameter_interval=CORRIDORS[0], **changes):
    inputs = dict(
        parameter_interval=parameter_interval,
        scaled_time=TIME,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
    )
    return owner._assess_preparation_path_selection(**(inputs | changes))


@pytest.fixture(scope="module")
def corridors():
    return tuple(_assess(interval) for interval in CORRIDORS)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_exact_generator_proves_whole_path_isometry_and_mean_preservation(corridors):
    first, second, generator = _source()
    report = corridors[0]
    assert report.path_generator == generator
    square = _multiply(generator, generator)
    assert report.path_generator_squared == square
    assert report.path_kappa == Q(1, 27 * M**2)
    assert _multiply(square, generator) == tuple(
        tuple(-report.path_kappa * x for x in row) for row in generator
    )
    assert all(sum(row, Q(0)) == 0 for row in generator)
    assert all(sum(column, Q(0)) == 0 for column in zip(*generator))
    laplacian = tuple(_laplace(row) for row in IDENTITY)
    lg = _multiply(laplacian, generator)
    assert all(lg[i][j] == -lg[j][i] for i in range(9) for j in range(9))
    maximum_row = max(sum(map(abs, row), Q(0)) for row in generator)
    assert maximum_row < Q(1, 1024)
    assert (1 + maximum_row) / (1 - maximum_row) < Q(1025, 1023)
    for parameter in (Q(0), Q(2, 7), Q(3, 4), Q(1)):
        matrix = _cayley(generator, parameter)
        denominator = 1 + report.path_kappa * parameter**2
        closed = tuple(
            tuple(
                IDENTITY[i][j]
                + 2 * parameter * generator[i][j] / denominator
                + 2 * parameter**2 * square[i][j] / denominator
                for j in range(9)
            )
            for i in range(9)
        )
        assert matrix == closed
        assert _multiply(tuple(zip(*matrix)), _multiply(laplacian, matrix)) == laplacian
        assert _apply(matrix, (Q(1),) * 9) == (Q(1),) * 9
        assert all(sum(column, Q(0)) == 1 for column in zip(*matrix))
        p, q = _profile(parameter)
        profile = tuple(p * (int(j == 3) - int(j == 4)) + q * (j - 4) for j in range(9))
        assert _apply(matrix, first) == profile
    assert _apply(_cayley(generator, Q(1)), first) == second


def test_endpoint_residual_map_is_new_and_exact_storage_is_constant(corridors):
    first, second, generator = _source()
    reflection = corridors[0].source.source_reflection
    end = _cayley(generator, Q(1))
    assert end != reflection
    error = tuple(ERROR * Q(i - 4, 4) for i in range(9))
    assert _apply(end, error) != _apply(reflection, error)
    complete = tuple(SLOPE * x + e for x, e in zip(first, error))
    initial_cost = _energy(complete)
    for parameter in (Q(0), Q(5, 11), Q(1)):
        reached_source = _apply(_cayley(generator, parameter), complete)
        assert sum(reached_source) == 0
        assert _energy(reached_source) == initial_cost
    # The shared zeta gives identical phase storage, independently of s.
    assert (
        corridors[0].common_initial_storage_upper_bound
        == corridors[1].common_initial_storage_upper_bound
        == corridors[2].common_initial_storage_upper_bound
    )


def test_interval_source_and_operator_bounds_retain_correlations(corridors, mp):
    first, second, generator = _source()
    kappa = Q(1, 27 * M**2)
    # q' has a decreasing numerator, positive even at s=1; energy then makes p decrease.
    assert 1 - 2 * kappa - 3 * kappa**2 > 0
    for report, (low, high) in zip(corridors, CORRIDORS):
        assert report.parameter_interval == (low, high)
        p_high, q_low = _profile(low)
        p_low, q_high = _profile(high)
        assert report.p_bounds.contains(p_low) and report.p_bounds.contains(p_high)
        assert report.q_bounds.contains(q_low) and report.q_bounds.contains(q_high)
        for parameter in (low, (low + high) / 2, high):
            matrix = _cayley(generator, parameter)
            assert all(
                report.path_operator_bounds[i][j].contains(matrix[i][j])
                for i in range(9)
                for j in range(9)
            )
            p, q = _profile(parameter)
            gamma = 1 / (1023 * mp.pi)
            angle = gamma * _mp(mp, SLOPE * q)
            correction = gamma * _mp(mp, SLOPE * p) - 2 * mp.pi * M
            _contains(mp, report.ramp_angle_bounds, angle)
            _contains(mp, report.lift_correction_bounds, correction)
        assert report.uniform_form_error_transport_upper_bound == Q(1025, 1023)
        assert report.transported_form_error_bound == Q(1025, 1023) * ERROR


def _ramp_storage(mp, angle):
    return 9 - 8 * mp.cos(angle) - mp.cos(8 * angle)


def test_correlated_ramp_extrema_and_shared_boundaries_have_independent_values(
    corridors, mp
):
    for report in corridors:
        low, high = map(
            lambda q: _mp(mp, q),
            (report.ramp_angle_bounds.lo, report.ramp_angle_bounds.hi),
        )
        candidates = [low, high]
        candidates.extend(
            point
            for point in (2 * mp.pi / 9, 3 * mp.pi / 7, 4 * mp.pi / 9)
            if low <= point <= high
        )
        expected = max(_ramp_storage(mp, point) for point in candidates)
        excess = _mp(mp, report.ramp_storage_bounds.hi) - expected
        assert 0 <= excess < mp.mpf("1e-24")
        for index, winding in enumerate((1, 2)):
            boundary = 9 - 8 * mp.cos((4 * winding - 1) * mp.pi / 16)
            lower = _mp(mp, report.sector_barrier_lower_bounds[index])
            assert 0 <= boundary - lower < mp.mpf("1e-24")
    # The unique interior maximum must be retained, not just endpoints.
    full = _assess((0, 1))
    critical_energy = 9 - 7 * mp.cos(3 * mp.pi / 7)
    assert _mp(mp, full.ramp_storage_bounds.hi) >= critical_energy
    assert full.winding_at_time is None
    assert full.captured_winding is None
    assert not full.forced_winding_two_exit_certified


def test_complete_transit_and_storage_bounds_have_independent_values(corridors, mp):
    gamma, forcing = 1 / (1023 * mp.pi), mp.sqrt(18)
    latent_error = _mp(mp, ERROR)
    form_error = gamma * forcing * mp.mpf(1025) / 1023 * latent_error
    phase_error = forcing * latent_error
    first, _, _ = _source()
    source_cost = (
        sum(
            ((SLOPE * abs(first[j] - first[i]) + 2 * ERROR) ** 2 / 2 for i, j in EDGES),
            Q(0),
        )
        + 18 * ERROR**2
    )
    for report in corridors:
        p, q = _mp(mp, report.p_bounds.hi), _mp(mp, report.q_bounds.hi)
        nominal = gamma * _mp(mp, SLOPE) * mp.sqrt(2 * (2 * p**2 + 60 * q**2))
        radius = mp.exp(-40) * (nominal + form_error) + 5 * gamma**2 * forcing
        form = radius / (gamma * mp.sqrt(2))
        phase = (
            radius + gamma**2 * forcing * 200 + form_error + phase_error
        ) / mp.sqrt(2)
        comparisons = (
            (report.nominal_scaled_initial_norm_upper_bound, nominal),
            (report.scaled_initial_error_norm_upper_bound, form_error),
            (report.initial_phase_error_norm_upper_bound, phase_error),
            (report.scaled_form_radius_upper_bound, radius),
            (report.endpoint_form_norm_upper_bound, form),
            (report.endpoint_phase_error_norm_upper_bound, phase),
            (report.endpoint_phase_edge_error_upper_bound, mp.sqrt(2) * phase),
        )
        for upper, expected in comparisons:
            assert 0 <= _mp(mp, upper) - expected < mp.mpf("1e-24")
        expected_storage = (
            2 * form**2
            + _mp(mp, report.ramp_storage_bounds.hi)
            + 3 * _mp(mp, report.lift_correction_bounds.abs_max) ** 2
            + 6 * phase
        )
        assert (
            0
            <= _mp(mp, report.endpoint_total_storage_upper_bound) - expected_storage
            < mp.mpf("1e-24")
        )
        assert report.common_initial_storage_upper_bound == source_cost
        _contains(mp, report.exponential_decay_bounds, mp.exp(-40))


def test_edge_association_keeps_lift_until_exact_cell_offset_cancellation(
    corridors, mp
):
    canonical_edges = tuple(sorted(tuple(sorted(edge)) for edge in EDGES))
    dipole = tuple(int(i == 3) - int(i == 4) for i in range(9))
    for report in corridors:
        for parameter in report.parameter_interval:
            p, q = _profile(parameter)
            proxy = tuple(
                _mp(mp, SLOPE * (p * h + q * (i - 4))) / (1023 * mp.pi)
                for i, h in enumerate(dipole)
            )
            for index, (i, j) in enumerate(canonical_edges):
                raw = proxy[j] - proxy[i]
                for sign in (-1, 1):
                    perturbed = raw + sign * _mp(
                        mp, report.endpoint_phase_edge_error_upper_bound
                    )
                    _contains(
                        mp,
                        report.endpoint_unwrapped_phase_edge_bounds[index],
                        perturbed,
                    )
                    for k, winding in enumerate((1, 2)):
                        # Closing edge is oriented 0->8, opposite the cycle direction.
                        expected_offset = -M * (dipole[j] - dipole[i]) - (
                            winding if (i, j) == (0, 8) else 0
                        )
                        assert (
                            report.edge_turn_offsets_by_winding[k][index]
                            == expected_offset
                        )
                        _contains(
                            mp,
                            report.phase_edge_bounds_by_winding[k][index],
                            perturbed + 2 * mp.pi * expected_offset,
                        )
            # Test the correlated Taylor bound on the actual nominal phase profile.
            phase_storage = sum(1 - mp.cos(proxy[j] - proxy[i]) for i, j in EDGES)
            angle = _mp(mp, SLOPE * q) / (1023 * mp.pi)
            correction = _mp(mp, SLOPE * p) / (1023 * mp.pi) - 2 * mp.pi * M
            assert phase_storage <= _ramp_storage(mp, angle) + 3 * correction**2


def test_three_uniform_corridors_separate_capture_from_transient_winding(corridors):
    first, transient, second = corridors
    assert first.winding_at_time == first.captured_winding == 1
    assert second.winding_at_time == second.captured_winding == 2
    assert transient.winding_at_time == 2 and transient.captured_winding is None
    assert transient.forced_winding_two_exit_certified
    assert not first.forced_winding_two_exit_certified
    assert not second.forced_winding_two_exit_certified
    assert first.acute_capture_certified_by_winding == (True, False)
    assert second.acute_capture_certified_by_winding == (False, True)
    assert transient.acute_capture_certified_by_winding == (False, False)
    assert first.endpoint_total_storage_upper_bound < Q(2344, 1000)
    assert transient.endpoint_total_storage_upper_bound < Q(7338, 1000)
    assert second.endpoint_total_storage_upper_bound < Q(74383, 10000)
    assert first.capture_margin_bounds[0].lo > Q(46, 10000)
    assert second.capture_margin_bounds[1].lo > Q(1, 1000)
    for report in corridors:
        assert report.source_budget_certified
        assert report.initial_zero_winding_certified
        assert report.common_initial_storage_upper_bound < 10**15


def test_acute_capture_is_not_the_old_small_ball_postcondition(corridors, mp):
    report = corridors[0]
    p, q = _profile(Q(1, 8))
    gamma = 1 / (1023 * mp.pi)
    differences = tuple(
        gamma * _mp(mp, SLOPE * (p * (int(i == 3) - int(i == 4)) + q * (i - 4)))
        - 2 * mp.pi * (M * (int(i == 3) - int(i == 4)) + mp.mpf(i - 4) / 9)
        for i in range(9)
    )
    nominal_distance = mp.sqrt(sum(value**2 for value in differences))
    assert nominal_distance > mp.mpf(".67") > mp.mpf(1) / 12
    assert (
        nominal_distance - _mp(mp, report.endpoint_phase_error_norm_upper_bound)
        > mp.mpf(1) / 12
    )
    assert report.captured_winding == 1


def test_critical_energy_excludes_all_winding_two_limits_including_boundary(
    mp, corridors
):
    minimum = 9 * (1 - mp.cos(4 * mp.pi / 9))
    energies = []
    for obtuse in range(4):
        delta = (4 - obtuse) * mp.pi / (9 - 2 * obtuse)
        gaps = (delta,) * (9 - obtuse) + (mp.pi - delta,) * obtuse
        assert abs(sum(gaps) - 4 * mp.pi) < mp.mpf("1e-95")
        assert max(mp.sin(gap) for gap in gaps) - min(
            mp.sin(gap) for gap in gaps
        ) < mp.mpf("1e-95")
        energy = sum(1 - mp.cos(gap) for gap in gaps)
        energies.append(energy)
        if obtuse:
            assert energy > minimum
    assert abs(energies[0] - minimum) < mp.mpf("1e-95")
    assert all(a < b for a, b in zip(energies, energies[1:]))
    # Zero-current boundary configurations require at least four antipodal edges.
    assert 8 > minimum
    transient = corridors[1]
    lower = _mp(mp, transient.winding_two_critical_storage_lower_bound)
    assert 0 <= minimum - lower < mp.mpf("1e-24")
    expected_margin = lower - _mp(mp, transient.endpoint_total_storage_upper_bound)
    _contains(mp, transient.forced_winding_two_exit_margin_bounds, expected_margin)
    assert minimum - _mp(mp, transient.endpoint_total_storage_upper_bound) > mp.mpf(
        ".099"
    )
    # Initial winding is zero; this actual later winding-two region cannot be retained.
    assert transient.initial_zero_winding_certified and transient.winding_at_time == 2


@pytest.mark.parametrize(
    "changes",
    [{"scaled_time": 0}, {"form_error_bound": 10**8}, {"phase_error_bound": 2}],
)
def test_unresolved_bounds_do_not_acquire_a_mode_or_predict_a_slip(changes):
    report = _assess(**changes)
    assert report.captured_winding is None
    assert not report.forced_winding_two_exit_certified
    assert report.unresolved_conditions


def test_source_policy_equality_does_not_erase_separate_mathematical_capture(
    monkeypatch, corridors
):
    monkeypatch.setattr(
        owner, "_STORAGE_BUDGET", corridors[0].common_initial_storage_upper_bound
    )
    report = _assess()
    assert (
        report.source_budget_margin_bounds.lo
        == report.source_budget_margin_bounds.hi
        == 0
    )
    assert not report.source_budget_certified
    assert (
        report.common_source_storage_budget == report.common_initial_storage_upper_bound
    )
    assert not report.preparation_policy_certified
    assert report.captured_winding == 1
    assert (
        "strict_common_source_storage_budget_not_certified"
        in report.unresolved_conditions
    )


def test_subgrid_parameters_and_errors_retain_exact_primitive_association():
    tiny = Q(1, 2**300)
    report = _assess((tiny, 2 * tiny), form_error_bound=tiny, phase_error_bound=0)
    assert report.parameter_interval == (tiny, 2 * tiny)
    assert report.form_error_bound == tiny
    assert report.transported_form_error_bound == tiny * Q(1025, 1023)
    assert report.preparation_dimension == 8
    assert not report.full_relative_preparation
    assert report.captured_winding == 1


@pytest.mark.parametrize(
    "field", ["scaled_time", "form_error_bound", "phase_error_bound"]
)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_scalar_admission_precedes_path_work(monkeypatch, field, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached the path construction")

    monkeypatch.setattr(owner, "_path_construction", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "interval",
    [
        (),
        (0,),
        (0, 1, 2),
        (True, 1),
        (0, "1"),
        (0, float("nan")),
        (0, float("inf")),
        (Q(-1, 2**300), 1),
        (0, 1 + Q(1, 2**300)),
        (Q(3, 4), Q(1, 4)),
    ],
)
def test_parameter_interval_is_admitted_before_materialization(monkeypatch, interval):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid interval reached proof arithmetic")

    monkeypatch.setattr(owner, "_path_construction", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(interval)


def test_parameter_iterator_is_bounded_without_draining():
    seen = []

    def endpoints():
        for value in range(20):
            seen.append(value)
            yield value

    with pytest.raises((TypeError, ValueError)):
        _assess(endpoints())
    assert len(seen) <= 3


@pytest.mark.parametrize(
    "changes",
    [
        {"scaled_time": -1},
        {"scaled_time": 20481},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
    ],
)
def test_exact_budget_domains_reject(changes):
    with pytest.raises(ValueError):
        _assess(**changes)
