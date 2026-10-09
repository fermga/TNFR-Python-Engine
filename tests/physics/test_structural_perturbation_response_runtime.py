"""A matched finite response keeps its original target and actual event effects."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest

from benchmarks import structural_perturbation_response as response
from tnfr.operators.definitions import Expansion
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_pattern,
)
from tnfr.physics.support_transport import SupportTransportSnapshot

F = Fraction


@pytest.fixture(scope="module")
def cases():
    records = tuple(
        response.run_structural_response_case(name) for name in response.CASES
    )
    comparison = response.compare_structural_response(records)
    print(
        "RESPONSE_METRICS",
        {
            "errors": {
                item["case"]: float(item["final_original_pattern"]["error_variance"])
                for item in records
            },
            "conditional_limits": {
                item["case"]: float(
                    item["conditional_fixed_model_limit"]["pattern"]["error_variance"]
                )
                for item in records
            },
            "event_changes": {
                item["case"]: [
                    (event["requested_glyph"], float(event["original_pattern_change"]))
                    for event in item["events"]
                ]
                for item in records
            },
            "classification": comparison["classification"],
            "damage": float(comparison["damage"]),
            "benefit": float(comparison["corrective_benefit"]),
            "gap": float(comparison["remaining_gap"]),
        },
    )
    return dict(zip(response.CASES, records, strict=True))


def _target(record):
    target = record["original_target"]
    return derive_forced_support_balance(
        SupportTransportSnapshot(**target["source"]),
        epi_weight=target["epi_weight"],
        forcing=target["forcing"],
    )


def test_all_branches_retain_the_actual_positive_capacity_pre_sha_checkpoint(cases):
    u, p, f = (cases[name] for name in response.CASES)
    assert u["initial"] == p["initial"] == f["initial"]
    assert u["original_target"] == p["original_target"] == f["original_target"]
    assert p["pre_feedback_materialized"] == f["pre_feedback_materialized"]
    assert p["events"] == f["events"][:2]
    assert u["events"] == []
    for record in cases.values():
        assert record["status"] == "measured"
        initial = record["initial"]["state"]
        checkpoint = record["checkpoint"]
        assert initial == checkpoint["retained_endpoint"]
        assert initial["time"] == 9.5
        assert checkpoint["executed_recent_segment_count"] == 12
        assert checkpoint["recent_duration"] == 3.0
        assert checkpoint["recent_mean_identity_residuals"] == (0,) * 12
        assert all(capacity > 0 for capacity in initial["capacity"])
        assert initial["glyph_history"]["0_sub_0"] == ("UM",)
        assert initial["glyph_history"][0] == ("IL", "OZ", "THOL", "UM")
        for field in ("metric_weights", "relative_profile", "epi_weight", "forcing"):
            assert (
                record["original_target"][field]
                == checkpoint["original_reference"][field]
            )
        assert record["initial"]["node_attributes"]
        assert record["initial"]["random_provenance"]["resolved_base_seed"] == 17
        assert record["initial"]["configured_controls"].get("EDGE_AWARE_ENABLED", True)


def test_actual_default_val_writes_epi_and_capacity_and_has_a_fixed_target_budget(
    cases,
):
    event = cases["P"]["events"][0]
    before, raw = event["before"]["state"], event["raw_after_event"]["state"]
    index = before["nodes"].index(event["target"])
    assert event["requested_glyph"] == "VAL"
    assert (
        event["resolved_factors"]["VAL_scale"]
        == canonical_glyph_factor_defaults()["VAL_scale"]
    )
    assert raw["capacity"][index] > before["capacity"][index]
    assert raw["epi"][index] != before["epi"][index]
    assert raw["phase"] == before["phase"]
    assert raw["edges"] == before["edges"]
    assert all(
        raw["epi"][i] == before["epi"][i]
        for i in range(len(before["nodes"]))
        if i != index
    )
    assert not event["pressure_refresh_after_event"]
    assert event["raw_after_event"] == event["after_refresh"]
    target = cases["P"]["original_target"]
    metric = target["metric_weights"]
    delta = F.from_float(raw["epi"][index]) - F.from_float(before["epi"][index])
    error = event["original_pattern_before"]["relative_error"][index]
    cross = metric[index] * error * delta
    quadratic = metric[index] * (1 - metric[index] / sum(metric)) * delta**2 / 2
    assert event["original_pattern_change"] == cross + quadratic


def test_il_refresh_preserves_actual_epi_and_records_phase_pressure_effects(cases):
    event = cases["P"]["events"][1]
    assert event["requested_glyph"] == "IL"
    assert event["pressure_refresh_after_event"]
    raw, refreshed = event["raw_after_event"]["state"], event["after_refresh"]["state"]
    for field in ("time", "nodes", "epi", "capacity", "phase", "edges"):
        assert raw[field] == refreshed[field]
    assert event["after_forcing_capture"]["stored_pressure_residual"] == (0,) * 9
    assert event["before"]["state"]["epi"] == raw["epi"]
    assert event["original_pattern_change"] == 0
    assert raw["pressure"] != refreshed["pressure"]


@pytest.mark.parametrize("name", ("P", "F"))
def test_all_actual_events_use_admitted_words_and_full_signed_budgets(cases, name):
    record = cases[name]
    for word in record["words"].values():
        assert word["string_validator_passed"] and word["instance_validator_passed"]
        assert word["context"] == {"initial_epi_nonzero": True}
    for event in record["events"]:
        assert event["admission"]["allowed"]
        assert event["status"] == "executed"
        assert (
            event["raw_after_event"]["state"]["glyph_history"][event["target"]][-1]
            == event["requested_glyph"]
        )
        budget = event["full_event_budget"]
        assert (
            budget["before"]["snapshot"] == event["before_forcing_capture"]["snapshot"]
        )
        assert budget["after"]["snapshot"] == event["after_forcing_capture"]["snapshot"]
        assert budget["mean_identity_residual"] == 0
        assert budget["error_identity_residual"] == (0,) * 9
        for label in ("variance", "dirichlet"):
            jump, reset = (
                budget[f"{label}_jump_budget"],
                budget[f"{label}_reset_budget"],
            )
            assert jump["identity_residual"] == reset["identity_residual"] == 0
            assert jump["energy_change"] == jump["cross_term"] + jump["quadratic_term"]
            assert reset["energy_change"] == sum(
                reset[key]
                for key in (
                    "metric_term",
                    "reference_cross_term",
                    "reference_quadratic_term",
                )
            )
            assert (
                budget[f"{label}_change"]
                == jump["energy_change"] + reset["energy_change"]
            )
            assert budget[f"{label}_identity_residual"] == 0


@pytest.mark.parametrize("name", response.CASES)
def test_flow_is_bound_to_actual_endpoints_with_fixed_refreshed_coefficients(
    cases, name
):
    record = cases[name]
    previous = (
        record["events"][-1]["after_refresh"]["state"]
        if record["events"]
        else record["initial"]["state"]
    )
    assert len(record["segments"]) == response.SEGMENT_COUNT == 12
    for segment in record["segments"]:
        assert segment["before"] == previous
        assert segment["duration"] == response.STEP == 0.25
        assert all(segment["frozen_input_checks"].values())
        evidence = segment["executor_evidence"]
        assert evidence["integrator_provenance_certified"]
        assert evidence["resolved_method"] == "euler"
        assert evidence["resolved_substeps"] == 1
        assert evidence["gamma_is_none"]
        assert not evidence["extended_dynamics_requested"]
        assert all(evidence["left_binding"].values())
        assert all(evidence["right_binding"].values())
        budget = segment["regime_step_budget"]
        assert budget["mean_identity_residual"] == 0
        assert budget["relative_recurrence_residual"] == (0,) * 9
        assert budget["relative_energy_budget"]["identity_residual"] == 0
        assert budget["support_budget"]["identity_residual"] == 0
        assert segment["forcing_capture"]["stored_pressure_residual"] == (0,) * 9
        previous = segment["after_refresh"]
        for node, epi in zip(previous["nodes"], previous["epi"], strict=True):
            assert previous["physical_epi_history"][node][-1] == (previous["time"], epi)
    assert previous == record["final_before_closure"]["state"]
    assert previous["time"] == 12.5
    assert record["physical_elapsed_time"] == 3.0


@pytest.mark.parametrize("name", response.CASES)
def test_variance_telescope_never_substitutes_dirichlet_energy(cases, name):
    record = cases[name]
    telescope = record["finite_telescope"]
    h_flow_change = sum(
        segment["regime_step_budget"]["after"]["error_variance"]
        - segment["regime_step_budget"]["before"]["error_variance"]
        for segment in record["segments"]
    )
    b_flow_change = sum(
        segment["regime_step_budget"]["relative_energy_budget"]["energy_change"]
        for segment in record["segments"]
    )
    assert h_flow_change != b_flow_change
    assert telescope["current_regime_flow_change"] == h_flow_change
    for prefix in ("original_pattern", "current_regime"):
        assert telescope[f"{prefix}_identity_residual"] == 0
        assert telescope[f"{prefix}_total_change"] == (
            telescope[f"{prefix}_event_change"] + telescope[f"{prefix}_flow_change"]
        )


@pytest.mark.parametrize("name", response.CASES)
def test_final_and_conditional_limit_readouts_keep_the_original_target(cases, name):
    record = cases[name]
    target = _target(record)
    final = record["final_before_closure"]["state"]
    expected = observe_forced_support_pattern(
        target, nodes=final["nodes"], epi=final["epi"]
    )
    assert record["final_original_pattern"] == asdict(expected)
    current = record["postevent_reference"]
    model_limit = observe_forced_support_pattern(
        target,
        nodes=current["source"]["nodes"],
        epi=current["relative_profile"],
    )
    assert record["conditional_fixed_model_limit"]["pattern"] == asdict(model_limit)
    assert model_limit.error_variance > 0
    assert "exact-real model limit" in record["conditional_fixed_model_limit"]["scope"]
    assert (
        "not an asymptotic assertion"
        in record["conditional_fixed_model_limit"]["scope"]
    )


def test_damage_benefit_gap_use_matched_fixed_target_endpoints(cases):
    comparison = response.compare_structural_response(tuple(cases.values()))
    assert comparison["status"] == "measured"
    assert all(comparison["matching_checks"].values())
    u, p, f = (
        cases[name]["final_original_pattern"]["error_variance"]
        for name in response.CASES
    )
    assert comparison["damage"] == p - u
    assert comparison["corrective_benefit"] == p - f
    assert comparison["remaining_gap"] == f - u
    assert comparison["damage_identity_residual"] == 0
    assert (
        comparison["damage"]
        == comparison["corrective_benefit"] + comparison["remaining_gap"]
    )


def test_terminal_closures_are_separate_from_the_measurement(cases):
    for record in cases.values():
        final = record["final_before_closure"]["state"]
        assert all(value > 0 for value in final["capacity"])
        closures = record["closures_after_measurement"]
        assert tuple(item["target"] for item in closures) == ("0_sub_0", 0)
        for item in closures:
            assert item["admission"]["allowed"]
            assert item["status"] == "executed"
            assert item["after"]["epi"] == final["epi"]
            assert item["after"]["time"] == 12.5
            assert item["after"]["glyph_history"][item["target"]][-1] == "SHA"


@pytest.mark.parametrize(
    "field", ("node_attributes", "configured_controls", "random_provenance")
)
def test_mismatched_feedback_preparation_is_rejected(cases, field):
    records = [dict(cases[name]) for name in response.CASES]
    altered = dict(records[2]["pre_feedback_materialized"])
    altered[field] = {"different": True}
    records[2]["pre_feedback_materialized"] = altered
    with pytest.raises(RuntimeError, match="matched comparison failed"):
        response.compare_structural_response(records)


def test_live_candidate_refusal_does_not_execute_or_refresh(monkeypatch):
    @dataclass
    class Refusal:
        allowed: bool = False

    monkeypatch.setattr(response, "_materialized", lambda graph: {"before": True})
    monkeypatch.setattr(response, "_reference", lambda capture: object())
    monkeypatch.setattr(
        response, "validate_candidate", lambda *args, **kwargs: Refusal()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("refused event must not execute or refresh")

    monkeypatch.setattr(response, "default_compute_delta_nfr", forbidden)
    monkeypatch.setattr(response, "capture_non_epi_forcing", forbidden)
    monkeypatch.setattr(Expansion, "__call__", forbidden)
    capture = object()
    following, event = response._event(
        nx.Graph(),
        "child",
        Expansion(),
        SimpleNamespace(index=1),
        object(),
        capture,
        refresh=True,
    )
    assert following is capture
    assert event["status"] == "controlled_obstruction"
    assert not event["admission"]["allowed"]
    assert "full_event_budget" not in event


def test_obstructed_branch_never_gets_a_recovery_classification():
    records = [
        {"case": name, "status": "controlled_obstruction"} for name in response.CASES
    ]
    assert (
        response.compare_structural_response(records)["status"]
        == "controlled_obstruction"
    )
    with pytest.raises(ValueError, match="ordered U/P/F"):
        response.compare_structural_response(records[::-1])


@pytest.mark.parametrize(
    "rp,rf,classification,fraction",
    (
        (9, 8, "no_damage_in_fixed_observable", None),
        (10, 8, "no_damage_in_fixed_observable", None),
        (12, 13, "no_beneficial_correction", F(-1, 2)),
        (12, 12, "no_beneficial_correction", F(0)),
        (12, 11, "partial_correction", F(1, 2)),
        (12, 10, "gap_closed_in_fixed_observable", F(1)),
        (12, 9, "outperforms_unperturbed_in_fixed_observable", F(3, 2)),
    ),
)
def test_pure_comparison_classification_requires_positive_damage(
    rp,
    rf,
    classification,
    fraction,
):
    records = [
        {
            "case": name,
            "status": "measured",
            "initial": {},
            "original_target": {},
            "pre_feedback_materialized": {},
            "events": [],
            "physical_elapsed_time": 3.0,
            "final_original_pattern": {"error_variance": F(error)},
        }
        for name, error in zip(response.CASES, (10, rp, rf), strict=True)
    ]
    comparison = response.compare_structural_response(records)
    assert comparison["classification"] == classification
    assert comparison["correction_fraction"] == fraction
    assert comparison["damage"] == F(rp - 10)
    assert comparison["corrective_benefit"] == F(rp - rf)
    assert comparison["remaining_gap"] == F(rf - 10)
    assert comparison["damage_identity_residual"] == 0
