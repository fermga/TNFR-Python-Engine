"""Independent full-field controls for finite protected bridge discrimination.

The only positive scientific budget is the declared two-preparation comparison.
Other inputs exercise admission and uncertainty failure; no trajectory runs.
"""

import json
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_bridge_discrimination import (
    assess_bridge_finite_law_discrimination,
)
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import export_to_json, relational_report_to_dict

LEFT, RIGHT = tuple(range(6)), tuple(range(6, 12))
TARGET = tuple(Q(i % 6, 6) for i in range(12))
BUDGET = dict(
    amplitude=Q(1, 10),
    duration=Q(1, 50),
    preparation_error=Q(1, 10**10),
    observation_error=Q(1, 10**8),
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def source():
    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for i in graph:
        graph.nodes[i].update(EPI=0, theta=0, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _assess(source, **changes):
    arguments = dict(
        left_cycle=LEFT, right_cycle=RIGHT, target_phase_turns=TARGET, **BUDGET
    )
    arguments.update(changes)
    return assess_bridge_finite_law_discrimination(source, **arguments)


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def _field(source, form, phase, eta):
    neighbors = [[] for _ in source.nodes]
    for i, j in source.edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    form_rate = tuple(
        mp.fsum(
            mp.sin(phase[j] - phase[i]) + eta * mp.sin(phase[j] - phase[i]) ** 3
            for j in row
        )
        / len(row)
        for i, row in enumerate(neighbors)
    )
    phase_rate = tuple(
        mp.fsum(form[i] - form[j] for j in row) / len(row)
        for i, row in enumerate(neighbors)
    )
    return form_rate, phase_rate


def test_finite_initial_acceleration_comes_from_all_nodal_rows(source, report):
    with mp.workdps(85):
        target = tuple(2 * mp.pi * _mp(turn) for turn in TARGET)
        p = tuple(mp.mpf(-1 if i < 6 else 1) / 2 for i in range(12))
        a = _mp(BUDGET["amplitude"])
        for eta, center in zip((Q(0), Q(1, 100)), report.ideal_curvature_values):
            for sign in (-1, 1):
                form = tuple(sign * a * value for value in p)
                dx, velocity = _field(source, form, target, _mp(eta))
                assert max(map(abs, dx)) < mp.mpf("1e-77")

                # x'=0 initially, so this directional derivative is the exact
                # nonlinear u'' for finite a, not a derivative in amplitude.
                def bridge_form_rate(t):
                    phase = tuple(
                        value + t * speed for value, speed in zip(target, velocity)
                    )
                    rows, _ = _field(source, form, phase, _mp(eta))
                    return rows[6] - rows[0]

                second = mp.diff(bridge_form_rate, mp.mpf(0))
                assert abs(second + sign * a * _mp(center)) < mp.mpf("1e-76")
                assert abs(
                    -second / (sign * a) - (mp.mpf(2) / 3 + _mp(eta) / 2)
                ) < mp.mpf("1e-76")
                # Opposite-port preparation immediately excites nonuniform
                # fine phase rates; two collective ring variables do not close.
                assert velocity[0] != velocity[1]
                assert velocity[6] != velocity[7]


def test_full_law_reversal_supports_central_difference_without_identifying_trials(
    source,
):
    with mp.workdps(75):
        form = tuple(mp.mpf(i * i - 9) / 71 for i in range(12))
        phase = tuple(mp.mpf(i) / 13 for i in range(12))
        for eta in (mp.mpf(0), mp.mpf(1) / 100):
            dx, dt = _field(source, form, phase, eta)
            reversed_dx, reversed_dt = _field(
                source, tuple(-x for x in form), phase, eta
            )
            assert dx == reversed_dx
            assert reversed_dt == tuple(-value for value in dt)


def test_fixed_budget_bounds_finite_error_and_protects_both_identities(report):
    assert report.discrimination_certified
    assert report.phase_chamber_certified
    assert report.separation_margin_lower_bound > 0
    assert len(report.positive_initial_box) == len(report.negative_initial_box) == 24
    assert report.positive_initial_form == tuple(
        BUDGET["amplitude"] * value for value in report.nodal_form_preparation
    )
    assert report.negative_initial_form == tuple(
        -value for value in report.positive_initial_form
    )
    with mp.workdps(85):
        a, h, rho, sigma = map(
            _mp,
            (
                BUDGET[key]
                for key in (
                    "amplitude",
                    "duration",
                    "preparation_error",
                    "observation_error",
                )
            ),
        )
        growth = mp.exp(2 * h)
        radius = a * growth / 2
        fourth = (
            128 * mp.mpf(43) / 40 * radius**3
            + 192 * mp.mpf(103) / 100 * radius**2
            + 32 * radius
        )
        finite = fourth * h * h / (12 * a)
        preparation = 4 * rho * growth / (a * h * h)
        observation = 2 * sigma / (a * h * h)
        total = finite + preparation + observation
        assert _mp(report.growth_factor_upper_bound) >= growth
        assert _mp(report.ideal_full_coordinate_radius_upper_bound) >= radius
        assert _mp(report.fourth_bridge_derivative_upper_bound) >= fourth
        assert _mp(report.finite_time_error_upper_bound) >= finite
        assert _mp(report.preparation_error_upper_bound) >= preparation
        assert _mp(report.observation_error_upper_bound) >= observation
        assert _mp(report.total_response_error_upper_bound) >= total
        assert _mp(report.phase_chamber_margin_lower_bound) <= mp.pi / 12 - 2 * (
            radius + rho * growth
        )
        # Independent own-law initial energy upper bound for both preparations:
        # only the bridge has nominal form difference, and target phase work
        # cancels over the complete graph.
        energy = a * a / 2 + 2 * a * rho + 52 * rho * rho
        assert energy < mp.cos(5 * mp.pi / 12) * mp.pi**2 / 288
        assert _mp(report.separation_margin_lower_bound) <= mp.mpf(1) / 200 - 2 * total
    sine, cubic = report.curvature_prediction_bounds
    assert sine.hi < cubic.lo
    assert sine.contains(Q(2, 3)) and cubic.contains(Q(2, 3) + Q(1, 200))


def test_two_endpoint_observation_budget_has_opposite_worst_case_signs(report):
    a, h, sigma = (
        BUDGET[key] for key in ("amplitude", "duration", "observation_error")
    )
    curvature = Q(2, 3)
    positive = a - a * curvature * h * h / 2
    negative = -a + a * curvature * h * h / 2
    observed = (2 * a - (positive - sigma) + (negative + sigma)) / (a * h * h)
    assert observed - curvature == report.observation_error_upper_bound


def test_noise_overlap_is_unavailable_not_a_failed_physical_pattern(source):
    noisy = _assess(source, observation_error=Q(1, 10000))
    assert noisy.phase_chamber_certified
    assert not noisy.discrimination_certified
    assert noisy.separation_margin_lower_bound < 0
    assert (
        noisy.curvature_prediction_bounds[0].hi
        >= noisy.curvature_prediction_bounds[1].lo
    )


def test_lost_phase_chamber_is_a_distinct_noncertificate(source):
    wide = _assess(source, amplitude=Q(1, 2))
    assert not wide.phase_chamber_certified
    assert not wide.discrimination_certified
    assert wide.phase_chamber_margin_lower_bound < 0


@pytest.mark.parametrize(
    "key,value",
    (
        ("amplitude", 0),
        ("amplitude", -1),
        ("amplitude", True),
        ("duration", 0),
        ("duration", Q(3, 4)),
        ("duration", False),
        ("preparation_error", -1),
        ("preparation_error", True),
        ("observation_error", float("nan")),
        ("observation_error", True),
    ),
)
def test_invalid_budget_rejects_before_any_prediction(source, key, value):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _assess(source, **{key: value})


def test_ordered_generators_are_consumed_once_for_both_laws(source, report):
    generated = _assess(
        source,
        left_cycle=iter(LEFT),
        right_cycle=iter(RIGHT),
        target_phase_turns=iter(TARGET),
    )
    assert generated.curvature_prediction_bounds == report.curvature_prediction_bounds
    assert generated.positive_initial_form == report.positive_initial_form


def test_derived_source_cache_cannot_change_finite_prediction(source, report):
    poisoned = replace(
        source, storage=I(-999), form_rates=(I(888),) * 12, phase_rates=(I(-888),) * 12
    )
    rebuilt = _assess(poisoned)
    assert rebuilt.curvature_prediction_bounds == report.curvature_prediction_bounds
    assert rebuilt.ideal_curvature_values == report.ideal_curvature_values
    assert (
        rebuilt.phase_chamber_margin_lower_bound
        == report.phase_chamber_margin_lower_bound
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, capacity=(True,) + source.capacity[1:]))


def test_source_snapshot_is_not_relabelled_as_the_finite_target(source, report):
    assert source.phase == (0,) * 12
    assert report.source.phase == source.phase
    assert report.target_phase_turns == TARGET
    assert report.positive_initial_form != source.epi
    assert all(
        bound.contains(value)
        for bound, value in zip(
            report.positive_initial_box[:12], report.positive_initial_form
        )
    )


def test_sdk_projection_retains_exact_budgets_and_both_prediction_intervals(
    report, tmp_path
):
    projected = relational_report_to_dict(report)
    path = tmp_path / "finite-discrimination.json"
    export_to_json(projected, path)
    saved = json.loads(path.read_text())
    assert saved == projected
    assert saved["report"]["amplitude"] == {"numerator": 1, "denominator": 10}
    assert len(saved["report"]["curvature_prediction_bounds"]) == 2
    assert saved["report"]["discrimination_certified"]
