"""THOL history, pressure consumption and child support have distinct scopes."""

from fractions import Fraction

import pytest

from benchmarks.thol_pressure_feedback import (
    NEXT_STEP,
    PREPARATION_STEPS,
    ROUTES,
    run_prepared_birth_case,
    run_recorded_thol_case,
)
from tnfr.config import CORE_DEFAULTS
from tnfr.constants.canonical import COUPLING_GENTLE


@pytest.fixture(scope="module")
def cases():
    return {route: run_recorded_thol_case(route) for route in ROUTES}


@pytest.fixture(scope="module")
def birth():
    return run_prepared_birth_case()


def test_observed_acceleration_uses_three_executed_unequal_physical_samples(cases):
    case = cases["public_held"]
    preparation = case["preparation"]
    observation = case["acceleration"]
    samples = observation["physical_samples"]
    assert preparation["actual_prefix"] == ("IL", "OZ")
    assert PREPARATION_STEPS == preparation["physical_steps"] == (0.125, 0.375)
    assert tuple(time for time, _ in samples) == (0.0, 0.125, 0.5)
    assert samples[-1][1] == case["before"]["epi"][0]
    assert preparation["segment_methods"] == ("euler", "euler")
    assert preparation["clipping_applied"] == (False, False)
    assert len(preparation["boundaries"]) == 3
    assert all(row["pressure_only_refresh"] for row in preparation["boundaries"])
    assert 0 < observation["observed_acceleration"] < case["default_birth_threshold"]
    assert (
        abs(observation["observed_acceleration"] - observation["cached_acceleration"])
        > 0.004
    )
    assert abs(observation["exact_acceleration_arithmetic_residual"]) < 1e-16
    assert not case["cache_poisoned_for_regression"]


@pytest.mark.parametrize("route", ROUTES)
def test_each_admitted_route_uses_the_shared_nodal_consumer(cases, route):
    case = cases[route]
    before = case["before"]
    raw = case["raw_after_operator"]
    integration = case["integration"]
    assert case["grammar_admission"]["allowed"]
    assert case["default_thol_factor"] == COUPLING_GENTLE
    assert (
        case["default_birth_threshold"] == CORE_DEFAULTS["THOL_BIFURCATION_THRESHOLD"]
    )
    assert before["time"] == integration["before"]["time"] == 0.5
    assert integration["after"]["time"] == 0.75
    assert integration["duration"] == NEXT_STEP == 0.25
    assert integration["clip_mode"] == "hard"
    assert integration["held_proposals_strictly_inside_hard_bounds"]
    assert max(map(abs, integration["exact_held_input_euler_residual"])) < 1e-16
    for key in ("epi", "capacity", "phase", "nodes", "edges"):
        assert raw[key] == before[key]
    assert len(raw["nodes"]) == 8
    assert all(not children for children in raw["children"].values())
    expected_history = ("IL", "OZ")
    if route != "baseline":
        expected_history += ("THOL",)
        assert case["exact_operator_pressure_change"] > 0
        assert abs(case["exact_pressure_arithmetic_residual"]) < 1e-16
    assert raw["glyph_history"][0] == expected_history


def test_public_and_staged_refresh_erase_parent_pressure_feedback(cases):
    baseline = cases["baseline"]["integration"]
    held = cases["public_held"]
    for route in ("public_refreshed", "staged_refreshed"):
        case = cases[route]
        assert (
            case["raw_after_operator"]["pressure"]
            == held["raw_after_operator"]["pressure"]
        )
        assert case["after_refresh"]["pressure"] == case["before"]["pressure"]
        assert case["integration"]["after"]["epi"] == baseline["after"]["epi"]
    assert cases["staged_refreshed"]["stage"] == {
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
    }


def test_held_extra_epi_has_exact_pressure_and_rounding_decomposition(cases):
    baseline = cases["baseline"]["integration"]
    held = cases["public_held"]
    integration = held["integration"]
    observed = Fraction.from_float(
        integration["after"]["epi"][0]
    ) - Fraction.from_float(baseline["after"]["epi"][0])
    dt_nu = Fraction.from_float(NEXT_STEP) * Fraction.from_float(
        held["before"]["capacity"][0]
    )
    exact_acceleration_prediction = dt_nu * held["exact_thol_pressure_proposal"]
    operator_rounding = dt_nu * held["exact_pressure_arithmetic_residual"]
    integration_rounding = (
        integration["exact_held_input_euler_residual"][0]
        - baseline["exact_held_input_euler_residual"][0]
    )
    assert observed == (
        exact_acceleration_prediction + operator_rounding + integration_rounding
    )
    assert 0.00025 < observed < 0.000251
    assert integration["after"]["epi"][1:] == baseline["after"]["epi"][1:]


def test_ordinary_selector_consumes_live_acceleration_before_auxiliary_updates(cases):
    public = cases["public_held"]
    selected = cases["selector_runtime"]
    assert (
        selected["raw_after_operator"]["pressure"]
        == public["raw_after_operator"]["pressure"]
    )
    assert (
        selected["integration"]["after"]["epi"] == public["integration"]["after"]["epi"]
    )
    legacy_cached_pressure = (
        selected["before"]["pressure"][0]
        + selected["default_thol_factor"]
        * selected["acceleration"]["cached_acceleration"]
    )
    assert (
        abs(selected["raw_after_operator"]["pressure"][0] - legacy_cached_pressure)
        > 0.0003
    )
    endpoint = selected["whole_route_endpoint"]
    assert endpoint["epi"] == selected["integration"]["after"]["epi"]
    assert endpoint["phase"] != selected["integration"]["after"]["phase"]
    assert endpoint["capacity"] == (1.0,) * 8
    assert endpoint["physical_epi_history"][0][-1] == (0.75, endpoint["epi"][0])
    # Direct integration carries a duration but does not itself append history.
    assert (
        public["whole_route_endpoint"]["physical_epi_history"]
        == public["before"]["physical_epi_history"]
    )


def test_ordinary_selector_ignores_an_explicitly_poisoned_acceleration_cache(cases):
    poisoned = run_recorded_thol_case("selector_runtime", poison_cache=True)
    reference = cases["selector_runtime"]
    assert poisoned["acceleration"]["cached_acceleration"] == 9.0
    assert (
        poisoned["acceleration"]["observed_acceleration"]
        == reference["acceleration"]["observed_acceleration"]
    )
    assert (
        poisoned["raw_after_operator"]["pressure"]
        == reference["raw_after_operator"]["pressure"]
    )
    assert (
        poisoned["integration"]["after"]["epi"]
        == reference["integration"]["after"]["epi"]
    )


def test_prepared_birth_crosses_unmodified_threshold_without_transport_edges(birth):
    before = birth["before"]
    raw = birth["raw_after_operator"]
    assert "not an executed earlier trajectory" in birth["preparation_scope"]
    assert birth["grammar_admission_allowed"]
    assert birth["acceleration"]["observed_acceleration"] == 0.4
    assert birth["default_birth_threshold"] == 0.1
    assert len(birth["children"]) == 1
    assert len(raw["nodes"]) == len(before["nodes"]) + 1
    assert birth["child_degrees"] == (0,)
    assert raw["edges"] == before["edges"]
    for key in ("epi", "capacity", "phase"):
        assert raw[key][:2] == before[key]
    assert raw["pressure"][0] > before["pressure"][0]
    assert raw["children"][0] == birth["children"]
    assert len(birth["sub_epi_records"]) == 1


def test_isolated_birth_does_not_maintain_parent_component_pressure(birth):
    before = birth["before"]
    refreshed = birth["after_refresh"]
    integration = birth["integration"]
    assert refreshed["pressure"][:2] == before["pressure"]
    assert refreshed["pressure"][2:] == (0.0,)
    assert integration["after"]["epi"][2:] == refreshed["epi"][2:]
    assert integration["held_proposals_strictly_inside_hard_bounds"]
    assert max(map(abs, integration["exact_held_input_euler_residual"])) < 1e-16
