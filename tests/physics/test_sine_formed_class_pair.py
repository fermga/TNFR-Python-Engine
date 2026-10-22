"""Independent complete-law controls for the fixed two-class C9 admission."""

from fractions import Fraction as Q
from typing import get_type_hints

import mpmath
import pytest

from tnfr.physics import relational_sine_formed_classes as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

TIME, ERROR, RADIUS = Q(100), Q(1, 10**10), Q(1, 12)
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
EDGES = tuple((i, (i + 1) % 9) for i in range(9))


def _assess(**changes):
    inputs = dict(
        scaled_time=TIME,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        radius=RADIUS,
    )
    return owner.assess_sine_formed_class_pair(**(inputs | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _upper_close(mp, bound, expected):
    difference = _mp(mp, bound) - expected
    assert 0 <= difference < mp.mpf("1e-25") * max(1, abs(expected))


def _laplace(values):
    return tuple(
        2 * values[i] - values[(i - 1) % 9] - values[(i + 1) % 9] for i in range(9)
    )


def _potential(mp, phases):
    return sum(1 - mp.cos(phases[j] - phases[i]) for i, j in EDGES)


def _winding(turns):
    period = sum(
        ((turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in EDGES), Q(0)
    )
    assert period.denominator == 1
    return int(period)


def test_sources_have_same_origins_and_distinct_costs_under_one_budget(frozen, mp):
    assert frozen.nodes == tuple(range(9))
    assert set(frozen.edges) == {(min(i, j), max(i, j)) for i, j in EDGES}
    assert frozen.held_capacities == (Q(1),) * 9
    assert frozen.metric_weights == (Q(2),) * 9
    assert frozen.conserved_form_mean == frozen.conserved_lifted_phase_mean == 0
    assert frozen.relative_mean_leaf_dimension == frozen.preparation_dimension == 16
    assert frozen.full_relative_preparation
    assert frozen.residual_constraints == (
        "sum_form_residuals_equals_zero",
        "sum_lifted_phase_residuals_equals_zero",
    )
    assert frozen.initial_phases == (Q(0),) * 9
    errors = tuple(ERROR * Q(i - 4, 4) for i in range(9))
    phases = tuple(_mp(mp, errors[(2 * i) % 9]) for i in range(9))
    assert sum(errors) == 0 and max(map(abs, errors)) == ERROR
    for index, k in enumerate((1, 2)):
        forms = tuple(k * SLOPE * (i - 4) for i in range(9))
        assert frozen.initial_forms_by_class[index] == forms
        assert sum(forms) == 0
        cost = sum(((forms[j] - forms[i]) ** 2 / 2 for i, j in EDGES), Q(0))
        assert cost == 36 * k**2 * SLOPE**2
        assert frozen.initial_form_storage_by_class[index] == cost
        upper = (
            sum(
                (((abs(forms[j] - forms[i]) + 2 * ERROR) ** 2 / 2) for i, j in EDGES),
                Q(0),
            )
            + 18 * ERROR**2
        )
        assert frozen.initial_storage_upper_bounds[index] == upper
        assert upper < frozen.common_source_storage_budget == 10**9
        perturbed = tuple(x + error for x, error in zip(forms, errors))
        actual = _mp(
            mp, sum(((perturbed[j] - perturbed[i]) ** 2 / 2 for i, j in EDGES), Q(0))
        ) + _potential(mp, phases)
        assert actual <= _mp(mp, upper)
    assert (
        frozen.initial_form_storage_by_class[1]
        == 4 * frozen.initial_form_storage_by_class[0]
    )


def test_fine_complete_rows_preserve_both_means_and_account_for_loss(frozen, mp):
    e, w = mp.mpf(1023) / 1024, mp.mpf(1) / 1024
    gamma = w / (mp.pi * e)
    for raw in frozen.initial_forms_by_class:
        forms = tuple(
            _mp(mp, x) + _mp(mp, ERROR) * (i - 4) / 4 for i, x in enumerate(raw)
        )
        phases = tuple(_mp(mp, ERROR) * (((2 * i) % 9) - 4) / 4 for i in range(9))
        laplace = _laplace(forms)
        currents = tuple(
            sum(mp.sin(phases[j] - phases[i]) for j in ((i - 1) % 9, (i + 1) % 9))
            for i in range(9)
        )
        form_rate = tuple(
            -e * lx / 2 + w * s / (2 * mp.pi) for lx, s in zip(laplace, currents)
        )
        phase_rate = tuple(w * lx / (2 * mp.pi) for lx in laplace)
        assert abs(sum(form_rate)) < mp.mpf("1e-90")
        assert abs(sum(phase_rate)) < mp.mpf("1e-90")
        storage_rate = sum(
            lx * dx - s * dt
            for lx, dx, s, dt in zip(laplace, form_rate, currents, phase_rate)
        )
        expected = -e * sum(value**2 for value in laplace) / 2
        assert abs(storage_rate - expected) < mp.mpf("1e-85")
        z = tuple(gamma * x for x in forms)
        for i, lz in enumerate(_laplace(z)):
            assert abs(
                gamma * form_rate[i] / e - (-lz / 2 + gamma**2 * currents[i] / 2)
            ) < mp.mpf("1e-90")
            assert abs(phase_rate[i] / e - lz / 2) < mp.mpf("1e-90")


def test_full_cycle_spectrum_and_symmetry_orbits_are_independent(frozen, mp):
    laplacian = mp.matrix(9)
    for i, j in EDGES:
        laplacian[i, i] += 1
        laplacian[j, j] += 1
        laplacian[i, j] -= 1
        laplacian[j, i] -= 1
    spectrum = tuple(mp.eigsy(laplacian, eigvals_only=True))
    expected = sorted(2 - 2 * mp.cos(2 * mp.pi * j / 9) for j in range(9))
    assert max(abs(a - b) for a, b in zip(spectrum, expected)) < mp.mpf("1e-90")
    assert spectrum[1] / 2 > _mp(mp, frozen.lambda_lower_bound)
    assert spectrum[-1] < _mp(mp, frozen.laplacian_upper_bound)
    for k, target, recorded_orbit in zip(
        (1, 2), frozen.target_phase_turns_by_class, frozen.symmetry_winding_orbits
    ):
        assert sum(target) == 0 and _winding(target) == k
        orbit = set()
        for shift in range(9):
            for orientation in (-1, 1):
                for sign in (-1, 1):
                    image = tuple(
                        sign * target[(orientation * i + shift) % 9] + Q(3, 7)
                        for i in range(9)
                    )
                    orbit.add(_winding(image))
        assert orbit == set(recorded_orbit) == {-k, k}
        for i in range(9):
            current = sum(
                mp.sin(2 * mp.pi * _mp(mp, target[j] - target[i]))
                for j in ((i - 1) % 9, (i + 1) % 9)
            )
            assert abs(current) < mp.mpf("1e-90")
    assert set(frozen.symmetry_winding_orbits[0]).isdisjoint(
        frozen.symmetry_winding_orbits[1]
    )


def test_analytic_transit_and_geometry_bounds_match_independent_high_precision(
    frozen, mp
):
    time, rx, rt, radius = map(lambda q: _mp(mp, q), (TIME, ERROR, ERROR, RADIUS))
    gamma = 1 / (1023 * mp.pi)
    forcing, gap = mp.sqrt(18), mp.mpf(1) / 5
    eta, decay = gamma**2, mp.exp(-gap * time)
    error_x, error_phase = gamma * forcing * rx, forcing * rt
    slope = _mp(mp, SLOPE)
    _contains(mp, frozen.gamma_bounds, gamma)
    _contains(mp, frozen.eta_bounds, eta)
    _contains(mp, frozen.exponential_decay_bounds, decay)
    _contains(mp, frozen.forcing_norm_bounds, forcing)
    for i, k in enumerate((1, 2)):
        v = gamma * k * slope * mp.sqrt(120)
        rz = decay * (v + error_x) + eta * forcing * min(time, 1 / gap)
        rx_end = rz / (gamma * mp.sqrt(2))
        phase_end = (rz + eta * time * forcing + error_x + error_phase) / mp.sqrt(2)
        distance = mp.sqrt(60) * k * abs(gamma * slope - 2 * mp.pi / 9)
        norm_square = rx_end**2 + (distance + phase_end) ** 2
        _upper_close(mp, frozen.nominal_scaled_initial_norm_upper_bounds[i], v)
        _upper_close(mp, frozen.scaled_initial_error_norm_upper_bounds[i], error_x)
        _upper_close(mp, frozen.scaled_form_radius_upper_bounds[i], rz)
        _upper_close(mp, frozen.endpoint_form_norm_upper_bounds[i], rx_end)
        _upper_close(mp, frozen.endpoint_phase_error_norm_upper_bounds[i], phase_end)
        _contains(mp, frozen.proxy_target_distance_bounds[i], distance)
        _upper_close(
            mp, frozen.endpoint_relative_norm_squared_upper_bounds[i], norm_square
        )
        _upper_close(
            mp, frozen.endpoint_excess_storage_upper_bounds[i], 2 * norm_square
        )
        _contains(
            mp,
            frozen.target_phase_storage_bounds_by_class[i],
            9 * (1 - mp.cos(2 * k * mp.pi / 9)),
        )
    angle = 4 * mp.pi / 9 + mp.sqrt(2) * radius
    _contains(mp, frozen.radius_angle_bounds, angle)
    _contains(mp, frozen.acute_radius_margin_bounds, mp.pi / 2 - angle)
    assert 0 < _mp(mp, frozen.coercivity_lower_bound) <= mp.cos(angle) / 5
    assert 0 < _mp(mp, frozen.barrier_lower_bound) <= radius**2 * mp.cos(angle) / 5


def test_frozen_joint_certificate_requires_both_formations_and_budget(frozen):
    assert frozen.status == "certified_two_formed_classes"
    assert frozen.unavailable_reasons == ()
    assert frozen.formation_certified_by_class == (True, True)
    assert frozen.source_budget_certified_by_class == (True, True)
    assert frozen.symmetry_inequivalent
    assert all(margin.lo > 0 for margin in frozen.endpoint_radius_margin_bounds)
    assert all(margin.lo > 0 for margin in frozen.storage_barrier_margin_bounds)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"scaled_time": 0}, "strict_endpoint_radius_not_certified"),
        ({"radius": 1}, "strict_acute_radius_not_certified"),
        ({"radius": Q(1, 10**6)}, "strict_excess_storage_barrier_not_certified"),
        ({"phase_error_bound": 1}, "whole_initial_set_zero_winding_not_certified"),
        ({"form_error_bound": 10000}, "strict_source_storage_budget_not_certified"),
    ],
)
def test_insufficient_bounds_abstain_without_changing_inputs(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert any(reason in item for item in result.unavailable_reasons)
    for name, value in changes.items():
        assert getattr(result, name) == value


def test_small_positive_barrier_rounding_to_zero_cannot_certify():
    result = _assess(radius=Q(1, 2**200), form_error_bound=0, phase_error_bound=0)
    assert result.acute_radius_certified
    assert result.coercivity_lower_bound > 0
    assert result.barrier_lower_bound == 0
    assert result.status == "unavailable"
    assert result.storage_barrier_certified_by_class == (False, False)


@pytest.mark.parametrize(
    "form,phase,dimension", [(0, 0, 0), (0, ERROR, 8), (ERROR, 0, 8)]
)
def test_zero_budgets_keep_their_lower_dimensional_preparation(form, phase, dimension):
    result = _assess(form_error_bound=form, phase_error_bound=phase)
    assert result.preparation_dimension == dimension
    assert not result.full_relative_preparation
    assert result.relative_mean_leaf_dimension == 16


def test_tiny_exact_error_is_retained_before_outward_rounding():
    tiny = Q(1, 2**300)
    result = _assess(form_error_bound=tiny, phase_error_bound=tiny)
    assert result.form_error_bound == result.phase_error_bound == tiny
    assert all(value > 0 for value in result.scaled_initial_error_norm_upper_bounds)
    assert result.preparation_dimension == 16


@pytest.mark.parametrize(
    "field", ["scaled_time", "form_error_bound", "phase_error_bound", "radius"]
)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_original_scalar_admission_precedes_shared_domain(monkeypatch, field, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached domain work")

    monkeypatch.setattr(owner, "_sine_domain", forbidden)
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
def test_domain_boundaries_reject(changes):
    with pytest.raises(ValueError):
        _assess(**changes)


def test_each_invocation_rebuilds_one_domain_for_both_sources(monkeypatch):
    domains, row_domains = [], []
    original_domain, original_rows = (
        owner._sine_domain,
        owner._sine_preparation_from_rows,
    )

    def domain(*args, **kwargs):
        value = original_domain(*args, **kwargs)
        domains.append(value)
        return value

    def rows(domain, **kwargs):
        row_domains.append(domain)
        return original_rows(domain, **kwargs)

    monkeypatch.setattr(owner, "_sine_domain", domain)
    monkeypatch.setattr(owner, "_sine_preparation_from_rows", rows)
    first = _assess(form_error_bound=0)
    second = _assess(form_error_bound=ERROR)
    assert len(domains) == 2 and domains[0] is not domains[1]
    assert len(row_domains) == 4
    assert row_domains[0] is row_domains[1] is domains[0]
    assert row_domains[2] is row_domains[3] is domains[1]
    assert (
        second.initial_storage_upper_bounds[0] > first.initial_storage_upper_bounds[0]
    )
    with pytest.raises(TypeError):
        owner.assess_sine_formed_class_pair(reference_certificate=first)


def test_direct_sdk_and_json_projection_retain_exact_primitive_evidence(
    frozen, tmp_path
):
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-formed-class-pair.v1"
    assert relational_report_to_dict(frozen)["report"] == direct["report"]
    path = tmp_path / "formed-classes.json"
    export_to_json(frozen, path)
    exported = json_loads(path.read_text(encoding="utf-8"))
    assert exported["report"] == direct["report"]
    for name in ("scaled_time", "form_error_bound", "phase_error_bound", "radius"):
        item = direct["report"][name]
        assert Q(item["numerator"], item["denominator"]) == getattr(frozen, name)
    get_type_hints(owner.SineFormedClassPair)
    get_type_hints(owner.assess_sine_formed_class_pair)
