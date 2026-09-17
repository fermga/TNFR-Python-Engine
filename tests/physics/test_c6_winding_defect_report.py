"""Detached B16 evidence is checked without executing another trajectory."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_defect_budget as campaign


INPUT = Path(__file__).resolve().parents[2] / "artifacts/research/c6_winding_joint_domain.json"


@pytest.fixture(scope="module")
def source_bytes():
    if not INPUT.is_file():
        pytest.skip("retained local B16 artifact is not available")
    return INPUT.read_bytes()


@pytest.fixture(scope="module")
def parent(source_bytes):
    return json.loads(source_bytes)


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_defect_report(parent)


def _vector(values):
    return tuple(map(F, values))


def _diameter(values):
    return max(values) - min(values)


def _independent_step(case, cycle):
    reference = case["joint_domain_reference"]
    h, nu, we, wp, rho, b, budget_weight = (
        F(reference[name]) for name in (
            "timestep", "capacity", "epi_weight", "phase_weight",
            "nonlinear_oscillation_factor", "forcing_step_factor", "epi_phase_budget_weight",
        )
    )
    before = cycle["before_capture"]["snapshot"]
    post = cycle["post_il_capture"]
    x = _vector(before["epi"])
    y = _vector(cycle["after_capture"]["snapshot"]["epi"])
    z0 = _vector(cycle["before_joint_readout"]["phase_pi_represented"])
    z1 = _vector(cycle["post_il_joint_readout"]["phase_pi_represented"])
    g_epi = tuple((x[(i - 1) % 6] + x[(i + 1) % 6]) / 2 - x[i] for i in range(6))
    g_phase = tuple((z1[(i - 1) % 6] + z1[(i + 1) % 6]) / 2 - z1[i] for i in range(6))
    ideal = tuple(xi + h * nu * (we * ge + wp * gp) for xi, ge, gp in zip(
        x, g_epi, g_phase, strict=True,
    ))
    pressure = _vector(post["snapshot"]["stored_pressure"])
    full_pressure = _vector(post["full_kernel_pressure"])
    phase_kernel = _vector(post["phase_gradient"])
    held = tuple(xi + h * nu * p for xi, p in zip(x, pressure, strict=True))
    phase_effect = tuple(h * nu * wp * (a - b) for a, b in zip(phase_kernel, g_phase, strict=True))
    assembly = tuple(h * nu * (p - we * ge - wp * gp) for p, ge, gp in zip(
        full_pressure, g_epi, phase_kernel, strict=True,
    ))
    stale = tuple(h * nu * (a - b) for a, b in zip(pressure, full_pressure, strict=True))
    integration = tuple(a - b for a, b in zip(y, held, strict=True))
    defect = tuple(a - b for a, b in zip(y, ideal, strict=True))
    epsilon = max(F(0), _diameter(z1) - rho * _diameter(z0))
    return {
        "ideal": ideal, "epi_defect": defect, "phase_excess": epsilon,
        "epi_defect_norm": max(map(abs, defect)),
        "phase_effect": phase_effect, "assembly_effect": assembly,
        "stored_pressure_effect": stale, "integrator_defect": integration,
        "budget_cost": (budget_weight + b) * epsilon + max(map(abs, defect)),
        "mean_change": sum(y, F(0)) / 6 - sum(x, F(0)) / 6,
    }


def test_analysis_is_readonly_and_never_calls_the_runtime_campaign(parent, monkeypatch):
    from benchmarks import c6_winding_joint_domain as producer

    def forbidden(*args, **kwargs):
        raise AssertionError("offline analysis must not execute a runtime trajectory")

    for name in ("run_c6_joint_case", "prepare_c6_phase_response", "_observed_stage", "_advance_forced_support_interval"):
        monkeypatch.setattr(producer, name, forbidden)
    original = deepcopy(parent)
    result = campaign.analyze_c6_defect_report(parent)
    assert parent == original
    assert result["runtime_executed"] is False
    assert tuple(case["mode"] for case in result["cases"]) == ("null", "k1", "k3")
    assert all(tuple(cycle["ordinal"] for cycle in case["cycles"]) == (1, 2) for case in result["cases"])


def test_all_six_steps_match_independent_exact_nodal_endpoints(parent, report):
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        for cycle, result in zip(retained["cycles"], analyzed["cycles"], strict=True):
            expected = _independent_step(retained, cycle)
            observed = result["observation"]
            assert observed["modeled_epi_after"] == expected["ideal"]
            assert observed["endpoint_defect"] == expected["epi_defect"]
            assert observed["phase_oscillation_defect"] == expected["phase_excess"]
            assert observed["endpoint_defect_infinity"] == expected["epi_defect_norm"]
            assert observed["absolute_defect_cost"] == expected["budget_cost"]
            assert observed["mean_defect"] == expected["mean_change"]
            assert observed["mean_identity_residual"] == 0
            assert observed["centered_defect"] == tuple(
                delta - expected["mean_change"] for delta in expected["epi_defect"]
            )
            assert observed["centered_defect_oscillation"] == _diameter(expected["epi_defect"])


def test_actual_signed_pressure_and_integrator_effects_telescope(parent, report):
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        for cycle, result in zip(retained["cycles"], analyzed["cycles"], strict=True):
            expected = _independent_step(retained, cycle)
            budget = result["pressure_budget"]
            assembly = tuple(a + b for a, b in zip(
                expected["assembly_effect"], expected["stored_pressure_effect"], strict=True,
            ))
            assert expected["stored_pressure_effect"] == (0,) * 6
            assert budget["phase_realization_epi_effect"] == expected["phase_effect"]
            assert budget["pressure_assembly_epi_effect"] == assembly
            assert budget["integrator_epi_effect"] == expected["integrator_defect"]
            assert budget["total_epi_defect"] == tuple(a + b + c for a, b, c in zip(
                expected["phase_effect"], assembly, expected["integrator_defect"], strict=True,
            )) == expected["epi_defect"]
            assert budget["identity_residual"] == (0,) * 6


def test_null_failures_remain_visible_and_their_signed_cancellation_is_retained(report):
    null = report["cases"][0]
    assert len(null["cycles"]) == 2
    for step in null["cycles"]:
        observed, budget = step["observation"], step["pressure_budget"]
        assert step["historical_conditional_status"] == "outside_conditional_transition"
        assert observed["phase_oscillation_defect"] > 0
        assert observed["phase_interval_expansion"] > 0
        assert observed["endpoint_defect_infinity"] > 0
        assert observed["mean_defect"] == 0
        # The pressure and held-step rounding almost cancel for this retained null.
        phase_size = max(map(abs, budget["phase_realization_epi_effect"]))
        integrator_size = max(map(abs, budget["integrator_epi_effect"]))
        assert 10 * observed["endpoint_defect_infinity"] < min(phase_size, integrator_size)
        assert observed["phase_box_before"] is True
        assert observed["phase_box_after"] is True


def test_two_cycle_reserve_cost_and_mean_prefixes_follow_the_captured_chain(parent, report):
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        reference = analyzed["reference"]
        rho, weight, b = (reference[name] for name in (
            "nonlinear_oscillation_factor", "epi_phase_budget_weight", "forcing_step_factor",
        ))
        x0 = _vector(retained["initial_capture"]["snapshot"]["epi"])
        z0 = _vector(retained["initial_joint_readout"]["phase_pi_represented"])
        lower0, upper0 = min(x0) - weight * _diameter(z0), max(x0) + weight * _diameter(z0)
        phase = [_diameter(z0)]
        means, costs, lower_losses, upper_losses = ([F(0)] for _ in range(4))
        lower, upper = [lower0], [upper0]
        for cycle in retained["cycles"]:
            step = _independent_step(retained, cycle)
            phase.append(rho * phase[-1] + step["phase_excess"])
            means.append(means[-1] + step["mean_change"])
            costs.append(costs[-1] + step["budget_cost"])
            lower_losses.append(lower_losses[-1] + (weight + b) * step["phase_excess"] - min(step["epi_defect"]))
            upper_losses.append(upper_losses[-1] + (weight + b) * step["phase_excess"] + max(step["epi_defect"]))
            lower.append(lower0 - lower_losses[-1])
            upper.append(upper0 + upper_losses[-1])
            actual = _vector(cycle["after_capture"]["snapshot"]["epi"])
            z = _vector(cycle["after_joint_readout"]["phase_pi_represented"])
            assert min(actual) - weight * _diameter(z) >= lower[-1]
            assert max(actual) + weight * _diameter(z) <= upper[-1]
            assert sum(actual, F(0)) / 6 == sum(x0, F(0)) / 6 + means[-1]
        prefix = analyzed["prefix_budget"]
        for key, values in (
            ("phase_oscillation_envelope", phase), ("mean_defect_prefix", means),
            ("absolute_defect_cost_prefix", costs), ("signed_lower_loss_prefix", lower_losses),
            ("signed_upper_loss_prefix", upper_losses), ("lower_reserve_bounds", lower),
            ("upper_reserve_bounds", upper),
        ):
            assert prefix[key] == tuple(values)
        assert prefix["mean_prefix_bound"] == max(map(abs, means))
        assert prefix["phase_class_preserved"] is True
        assert prefix["epi_reserve_preserved"] is True
        assert len(prefix["observations"]) == 2


def test_finite_extrema_do_not_become_verified_future_error_bounds(report):
    for case in report["cases"]:
        finite = case["uniform_envelope_from_finite_extrema"]
        assert finite["future_hypotheses_verified"] is False
        bound = finite["conditional_envelope"]
        observations = [cycle["observation"] for cycle in case["cycles"]]
        assert bound["phase_defect_bound"] == max(item["phase_oscillation_defect"] for item in observations)
        assert bound["centered_epi_defect_bound"] == max(item["centered_defect_oscillation"] for item in observations)
        assert bound["mean_prefix_bound"] == max(map(abs, case["prefix_budget"]["mean_defect_prefix"]))
        # A per-step absolute mean bound is a different quantity from this signed prefix.
        if case["mode"] == "k3":
            assert bound["mean_prefix_bound"] < sum(abs(item["mean_defect"]) for item in observations)


def _change(data, path, value):
    current = data
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


BAD_FIELDS = (
    (("cases", 0, "cycle_count"), 3),
    (("cases", 0, "cycles", 0, "ordinal"), 2),
    (("cases", 0, "cycles", 0, "before_capture", "snapshot", "nodes", 0), True),
    (("cases", 0, "cycles", 0, "post_il_capture", "normalized_weights", 0, 1), "1/7"),
    (("cases", 0, "cycles", 0, "post_il_capture", "phase", 0), "1/7"),
    (("cases", 0, "cycles", 0, "post_il_joint_readout", "phase_pi_represented", 0), "1/7"),
    (("cases", 0, "cycles", 0, "um", "raw_capture", "snapshot", "epi", 0), "3/4"),
    (("cases", 0, "cycles", 0, "post_il_capture", "snapshot", "epi", 0), "3/4"),
    (("cases", 0, "cycles", 0, "post_il_capture", "snapshot", "capacity", 0), "9/10"),
    (("cases", 0, "cycles", 0, "post_il_capture", "snapshot", "conductance", 0, 2), "2"),
    (("cases", 0, "cycles", 0, "post_il_capture", "snapshot", "support_neighbors", 0), [1, 3, 5]),
    (("cases", 0, "cycles", 0, "flow", "executor_evidence", "captured_left", "epi", 0), 0.75),
    (("cases", 0, "cycles", 0, "flow", "executor_evidence", "captured_right", "pressure", 0), 0.125),
    (("cases", 0, "cycles", 0, "flow", "raw_after_integrator", "epi", 0), 0.75),
    (("cases", 0, "cycles", 0, "flow", "duration"), 0.5),
    (("cases", 0, "cycles", 1, "before_capture", "snapshot", "epi", 0), "3/4"),
    (("cases", 0, "cycles", 0, "post_il_capture", "kernel_pressure_defect", 0), "1/4"),
    (("cases", 0, "cycles", 0, "post_il_capture", "stored_pressure_residual", 0), "1/4"),
    (("cases", 0, "cycles", 0, "um", "before", "state", "edges", 0, 0), True),
    (("cases", 0, "final_before_closure", "state", "time"), 1.0),
    (("cases", 0, "joint_domain_reference", "epi_lower"), "1/100"),
    (("cases", 0, "admission_band", "um_min_capacity"), 2.0),
)


@pytest.mark.parametrize("path,value", BAD_FIELDS)
def test_corrupted_primary_fields_or_retained_bindings_are_rejected(parent, path, value):
    damaged = deepcopy(parent)
    _change(damaged, path, value)
    with pytest.raises((TypeError, ValueError)):
        campaign.analyze_c6_defect_report(damaged)


@pytest.mark.parametrize("mutation", ("case_order", "missing_case", "cycle_order", "missing_cycle", "missing_endpoint"))
def test_missing_or_reordered_evidence_is_not_silently_shortened(parent, mutation):
    damaged = deepcopy(parent)
    if mutation == "case_order":
        damaged["cases"].reverse()
    elif mutation == "missing_case":
        damaged["cases"].pop()
    elif mutation == "cycle_order":
        damaged["cases"][0]["cycles"].reverse()
    elif mutation == "missing_cycle":
        damaged["cases"][0]["cycles"].pop()
    else:
        damaged["cases"][0]["cycles"][0]["flow"]["executor_evidence"].pop("captured_right")
    with pytest.raises((KeyError, TypeError, ValueError)):
        campaign.analyze_c6_defect_report(damaged)


def test_relative_flow_duration_alone_does_not_bind_the_causal_record_clock(parent):
    damaged = deepcopy(parent)
    flow = damaged["cases"][0]["cycles"][0]["flow"]
    for name in ("before", "raw_after_integrator", "after_refresh"):
        flow[name]["time"] += 1
    with pytest.raises(ValueError, match="time|clock"):
        campaign.analyze_c6_defect_report(damaged)


def test_cli_forbids_overwriting_its_input_even_before_decoding(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_bytes(b"unchanged input")
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(source), "--output", str(source)])
    with pytest.raises(ValueError):
        campaign.main()
    assert source.read_bytes() == b"unchanged input"


def test_cli_retains_historical_producer_and_content_hash_with_own_scope(source_bytes, tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    source.write_bytes(source_bytes)
    calls = []
    analysis_sha = "a" * 40

    def provenance(root, scope):
        calls.append(tuple(scope))
        return analysis_sha, False, None

    monkeypatch.setattr(campaign, "current_git_source_provenance", provenance)
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(source), "--output", str(output)])
    campaign.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    historical = json.loads(source_bytes)
    assert result["runtime_executed"] is False
    assert result["input_evidence"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert result["input_evidence"]["producer_manifest"] == historical["manifest"]
    assert result["input_evidence"]["producer_source_scope"] == historical["source_scope"]
    assert result["manifest"]["git_sha"] == analysis_sha
    assert result["manifest"]["result_status"] == "derived"
    assert tuple(result["source_scope"]) == campaign.SOURCE_SCOPE
    assert calls and all(scope == campaign.SOURCE_SCOPE for scope in calls)
    assert source.read_bytes() == source_bytes


def test_cli_rejects_input_that_changes_during_detached_analysis(source_bytes, tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    source.write_bytes(source_bytes)
    actual_analysis = campaign.analyze_c6_defect_report

    def changing_input(data):
        result = actual_analysis(data)
        source.write_bytes(source_bytes + b"\n")
        return result

    monkeypatch.setattr(campaign, "analyze_c6_defect_report", changing_input)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *args: ("a" * 40, False, None))
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(source), "--output", str(output)])
    with pytest.raises(RuntimeError, match="input|evidence"):
        campaign.main()
    assert not output.exists()


def test_cli_rejects_consumer_source_change_without_rewriting_parent(source_bytes, tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    source.write_bytes(source_bytes)
    revisions = iter((("a" * 40, False, None), ("b" * 40, False, None)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *args: next(revisions))
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(source), "--output", str(output)])
    with pytest.raises(RuntimeError, match="source"):
        campaign.main()
    assert source.read_bytes() == source_bytes
    assert not output.exists()
