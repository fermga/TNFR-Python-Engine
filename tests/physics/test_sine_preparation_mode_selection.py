"""Independent matched-source and acquired-mode controls on the fixed C9."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_class_constitutive_acquisition as previous_owner
from tnfr.physics import _sine_preparation_mode_selection as owner
from tnfr.physics import relational_sine_formed_classes as sine_owner

ERROR, TIME, RADIUS = Q(1, 10**10), Q(200), Q(1, 12)
M = 512
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
EDGES = tuple((i, (i + 1) % 9) for i in range(9))


def _laplace(values):
    return tuple(
        2 * values[i] - values[(i - 1) % 9] - values[(i + 1) % 9] for i in range(9)
    )


def _product(matrix, vector):
    return tuple(sum((x * y for x, y in zip(row, vector)), Q(0)) for row in matrix)


def _energy(vector):
    return sum(((vector[j] - vector[i]) ** 2 / 2 for i, j in EDGES), Q(0))


def _construction():
    ramp = tuple(Q(i - 4) for i in range(9))
    pulse = tuple(Q(int(i == 3) - int(i == 4)) for i in range(9))
    first = tuple((9 * M + Q(1, M)) * h + r for h, r in zip(pulse, ramp))
    second = tuple((9 * M - Q(1, M)) * h + 2 * r for h, r in zip(pulse, ramp))
    difference = tuple(a - b for a, b in zip(first, second))
    normal = _laplace(difference)
    square = sum((x * y for x, y in zip(difference, normal)), Q(0))
    reflection = tuple(
        tuple(Q(int(i == j)) - 2 * difference[i] * normal[j] / square for j in range(9))
        for i in range(9)
    )
    return first, second, reflection


@pytest.fixture(scope="module", autouse=True)
def no_previous_verdict_or_scientific_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("new preparation attempted to reuse an old formation verdict")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(previous_owner, "_assess_constitutive_class_pair", forbidden)
        patch.setattr(sine_owner, "assess_sine_formed_class_pair", forbidden)
        patch.setattr(sine_owner, "_fresh_class_preparations", forbidden)
        yield


def _assess(**changes):
    inputs = dict(
        scaled_time=TIME, form_error_bound=ERROR, phase_error_bound=ERROR, radius=RADIUS
    )
    return owner._assess_preparation_mode_selection(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _upper_close(mp, upper, expected):
    error = _mp(mp, upper) - expected
    assert 0 <= error < mp.mpf("1e-24") * max(1, abs(expected))


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def test_exact_source_map_preserves_energy_and_means_but_not_the_law(report):
    first, second, reflection = _construction()
    assert report.source.normalized_initial_forms_by_class == (first, second)
    assert report.source.source_reflection == reflection
    assert _energy(first) == _energy(second) == Q(486 * M**2 + 180, 2) + Q(3, M**2)
    assert _product(reflection, first) == second
    assert _product(reflection, (Q(1),) * 9) == (Q(1),) * 9
    assert tuple(map(sum, zip(*reflection))) == (Q(1),) * 9
    # Basis and cross-basis checks reconstruct the whole quadratic form.
    basis = tuple(tuple(Q(int(i == j)) for i in range(9)) for j in range(9))
    for vector in basis:
        image = _product(reflection, vector)
        assert _product(reflection, image) == vector
        assert _energy(image) == _energy(vector)
        for other in basis:
            combined = tuple(x + y for x, y in zip(vector, other))
            assert _energy(_product(reflection, combined)) == _energy(combined)
    assert _laplace(_product(reflection, first)) != _laplace(first)
    assert _product(reflection, _laplace(first)) != _laplace(
        _product(reflection, first)
    )
    row_norms = tuple(sum(map(abs, row), Q(0)) for row in reflection)
    assert report.source.reflection_row_absolute_sums == row_norms
    assert max(row_norms) == Q(1968129, 786433) > 1


def test_observations_match_for_exactly_paired_errors_not_independent_boxes(report, mp):
    first, second, reflection = _construction()
    form_error = tuple(ERROR * Q(i - 4, 4) for i in range(9))
    phase_error = tuple(ERROR * Q((2 * i) % 9 - 4, 4) for i in range(9))
    mapped = _product(reflection, form_error)
    forms = (
        tuple(SLOPE * a + e for a, e in zip(first, form_error)),
        tuple(SLOPE * b + e for b, e in zip(second, mapped)),
    )
    assert sum(phase_error) == sum(form_error) == sum(mapped) == 0
    assert forms[1] == _product(reflection, forms[0])
    assert _energy(forms[0]) == _energy(forms[1])
    assert report.source.initial_forms_by_class == tuple(
        tuple(SLOPE * x for x in row) for row in (first, second)
    )
    assert all(
        abs(e) <= bound
        for e, bound in zip(mapped, report.form_error_bounds_by_class[1])
    )
    potential = sum(
        1 - mp.cos(_mp(mp, phase_error[j] - phase_error[i])) for i, j in EDGES
    )
    assert _mp(mp, _energy(forms[0])) + potential <= _mp(
        mp, report.common_initial_storage_upper_bound
    )
    assert report.source.common_initial_form_storage == SLOPE**2 * _energy(first)
    expected_upper = (
        sum(
            (
                (abs(SLOPE * (first[j] - first[i])) + 2 * ERROR) ** 2 / 2
                for i, j in EDGES
            ),
            Q(0),
        )
        + 18 * ERROR**2
    )
    assert report.common_initial_storage_upper_bound == expected_upper < 320680599854722
    # This admissible independent error destroys exact matching.
    unrelated = tuple(SLOPE * b + e for b, e in zip(second, form_error))
    assert _energy(unrelated) != _energy(forms[0])
    assert report.preparation_dimension == 16
    assert report.full_relative_preparation


def test_continued_lifts_keep_the_same_circular_targets_and_distinct_windings(
    report, mp
):
    for index, winding in enumerate((1, 2)):
        turns = tuple(
            Q(winding * (j - 4), 9) + M * (int(j == 3) - int(j == 4)) for j in range(9)
        )
        assert report.source.target_phase_turns_by_class[index] == turns
        assert sum(turns) == 0
        principal = tuple(
            (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in EDGES
        )
        assert principal == (Q(winding, 9),) * 9
        assert sum(principal) == winding
        for j, turn in enumerate(turns):
            canonical = Q(winding * (j - 4), 9)
            assert (turn - canonical).denominator == 1
        # Target stationarity on all nodes, including the two huge lift jumps.
        for i in range(9):
            current = sum(
                mp.sin(2 * mp.pi * _mp(mp, turns[j] - turns[i]))
                for j in ((i - 1) % 9, (i + 1) % 9)
            )
            assert abs(current) < mp.mpf("1e-90")
    assert report.initial_zero_winding_certified
    assert report.symmetry_inequivalent


def test_initial_phase_rate_distinguishes_the_hidden_form_arrangement(report, mp):
    first, second, reflection = _construction()
    gamma = 1 / (1023 * mp.pi)
    residual = tuple(ERROR * Q((5 * j) % 9 - 4, 4) for j in range(9))
    delta = tuple(
        SLOPE * (a - b) + e - f
        for a, b, e, f in zip(first, second, residual, _product(reflection, residual))
    )
    observed = gamma * _mp(mp, _laplace(delta)[0]) / 2
    nominal = 9 * gamma * _mp(mp, SLOPE) / 2
    bound = (
        2
        * gamma
        * (1 + _mp(mp, max(report.source.reflection_row_absolute_sums)))
        * _mp(mp, ERROR)
    )
    assert abs(observed - nominal) <= bound < nominal


def test_full_family_bounds_match_independent_high_precision(report, mp):
    first, second, reflection = _construction()
    gamma = 1 / (1023 * mp.pi)
    gap, forcing = mp.mpf(1) / 5, mp.sqrt(18)
    time, radius = _mp(mp, TIME), _mp(mp, RADIUS)
    decay, phase_error = mp.exp(-gap * time), forcing * _mp(mp, ERROR)
    _contains(mp, report.gamma_bounds, gamma)
    _contains(mp, report.exponential_decay_bounds, decay)
    for index, source in enumerate((first, second)):
        nominal = tuple(_mp(mp, SLOPE * x) for x in source)
        initial = gamma * mp.sqrt(2 * sum(x**2 for x in nominal))
        errors = tuple(_mp(mp, q) for q in report.form_error_bounds_by_class[index])
        # Mapped row bounds are used only for transit; matching uses the exact R.
        form_error = gamma * mp.sqrt(2 * sum(x**2 for x in errors))
        target = tuple(
            2 * mp.pi * _mp(mp, t)
            for t in report.source.target_phase_turns_by_class[index]
        )
        distance = mp.sqrt(
            sum((gamma * x - theta) ** 2 for x, theta in zip(nominal, target))
        )
        rz = decay * (initial + form_error) + gamma**2 * forcing / gap
        rx = rz / (gamma * mp.sqrt(2))
        phase_remainder = (
            rz + gamma**2 * forcing * time + form_error + phase_error
        ) / mp.sqrt(2)
        phase_norm = distance + phase_remainder
        _upper_close(
            mp, report.nominal_scaled_initial_norm_upper_bounds[index], initial
        )
        _upper_close(
            mp, report.scaled_initial_error_norm_upper_bounds[index], form_error
        )
        _contains(mp, report.proxy_target_distance_bounds[index], distance)
        _upper_close(mp, report.scaled_form_radius_upper_bounds[index], rz)
        _upper_close(mp, report.endpoint_form_norm_upper_bounds[index], rx)
        _upper_close(
            mp, report.endpoint_phase_error_norm_upper_bounds[index], phase_remainder
        )
        _upper_close(
            mp, report.endpoint_target_phase_norm_upper_bounds[index], phase_norm
        )
        _upper_close(
            mp,
            report.endpoint_relative_norm_squared_upper_bounds[index],
            rx**2 + phase_norm**2,
        )
        _upper_close(
            mp,
            report.endpoint_excess_storage_upper_bounds[index],
            2 * (rx**2 + phase_norm**2),
        )
        # Dropping the integer lift changes the source-target distance by thousands.
        canonical = tuple(2 * mp.pi * (index + 1) * (i - 4) / 9 for i in range(9))
        wrong_distance = mp.sqrt(
            sum((gamma * x - theta) ** 2 for x, theta in zip(nominal, canonical))
        )
        assert distance < mp.mpf(".003") and wrong_distance > 4000
    barrier = mp.cos(4 * mp.pi / 9 + mp.sqrt(2) * radius) * radius**2 / 5
    assert 0 <= barrier - _mp(mp, report.barrier_lower_bound) < mp.mpf("1e-25")


def test_strict_full_family_capture_closes_the_information_counterexample(report):
    assert report.status == "certified_matched_observation_mode_selection"
    assert report.unavailable_reasons == ()
    assert report.formation_certified_by_class == (True, True)
    assert report.source.source_pairing_certified
    assert report.matched_observation_mode_distinction_certified
    assert report.source_budget_certified
    assert report.original_time == TIME * Q(1024, 1023)
    assert report.common_initial_storage_upper_bound < 10**15
    assert max(report.endpoint_relative_norm_squared_upper_bounds) < Q(3, 10**5)
    assert max(report.endpoint_excess_storage_upper_bounds) < Q(59, 10**6)
    assert report.barrier_lower_bound > Q(78, 10**6)
    assert min(value.lo for value in report.storage_barrier_margin_bounds) > Q(
        19, 10**6
    )


def test_equal_source_budget_abstains_without_denying_geometric_capture(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner, "_STORAGE_BUDGET", report.common_initial_storage_upper_bound
    )
    result = _assess()
    assert (
        result.source_budget_margin_bounds.lo
        == result.source_budget_margin_bounds.hi
        == 0
    )
    assert not result.source_budget_certified
    assert result.formation_certified_by_class == (True, True)
    assert not result.matched_observation_mode_distinction_certified
    assert result.status == "unavailable"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"scaled_time": 0}, "strict_endpoint_radius_not_certified"),
        ({"radius": 1}, "strict_acute_radius_not_certified"),
        ({"radius": Q(1, 10**6)}, "strict_excess_storage_barrier_not_certified"),
        ({"phase_error_bound": 2}, "whole_initial_set_zero_winding_not_certified"),
        (
            {"form_error_bound": 10**8},
            "strict_common_source_storage_budget_not_certified",
        ),
    ],
)
def test_failed_sufficient_bounds_remain_unavailable(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert any(reason in item for item in result.unavailable_reasons)
    for name, value in changes.items():
        assert getattr(result, name) == value


@pytest.mark.parametrize(
    "form,phase,dimension", [(0, 0, 0), (ERROR, 0, 8), (0, ERROR, 8)]
)
def test_correlated_preparation_dimension_tracks_exact_latent_errors(
    form, phase, dimension
):
    result = _assess(form_error_bound=form, phase_error_bound=phase)
    assert result.preparation_dimension == dimension
    assert not result.full_relative_preparation


def test_subgrid_errors_and_unavailable_zero_barrier_are_not_conflated():
    tiny = Q(1, 2**300)
    result = _assess(form_error_bound=tiny, phase_error_bound=tiny)
    assert result.form_error_bound == result.phase_error_bound == tiny
    assert result.preparation_dimension == 16
    assert all(value > 0 for value in result.scaled_initial_error_norm_upper_bounds)
    assert (
        result.common_initial_storage_upper_bound
        > result.source.common_initial_form_storage
    )
    result = _assess(radius=tiny)
    assert result.barrier_lower_bound == 0
    assert result.status == "unavailable"
    assert result.storage_barrier_certified_by_class == (False, False)


@pytest.mark.parametrize(
    "field", ["scaled_time", "form_error_bound", "phase_error_bound", "radius"]
)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("nan"), float("inf")])
def test_primitive_admission_precedes_proof_arithmetic(monkeypatch, field, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached proof arithmetic")

    monkeypatch.setattr(owner, "_verify_cycle_gap", forbidden)
    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    monkeypatch.setattr(owner, "_paired_source_construction", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"scaled_time": -1},
        {"scaled_time": 20481},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"radius": 0},
    ],
)
def test_exact_domain_boundaries_reject(changes):
    with pytest.raises(ValueError):
        _assess(**changes)
