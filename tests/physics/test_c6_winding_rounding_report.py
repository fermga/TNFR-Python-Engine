"""The retained C6 record binds detached rounding arithmetic without new words."""

import hashlib
import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_rounding_cells as campaign

INPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/research/c6_winding_joint_domain.json"
)


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
    return campaign.analyze_c6_rounding_report(parent)


@pytest.mark.parametrize(
    "value", [True, None, [], "1/3", "NaN", "Infinity", "1e400", f"1/{2**1075}"]
)
def test_nonrepresented_scalar_inputs_are_rejected_before_replay(value):
    with pytest.raises((ValueError, TypeError)):
        campaign._represented_scalar(value)


@pytest.mark.parametrize(
    "value,expected", [("1/2", 0.5), (0, 0.0), (0.05, 0.05), (str(F(0.05)), 0.05)]
)
def test_exact_binary64_numeric_encodings_keep_their_value(value, expected):
    decoded = campaign._represented_scalar(value)
    assert type(decoded) is float
    assert decoded == expected
    assert F(decoded) == F(value)


def test_readonly_analysis_uses_shared_primitive_without_running_a_word(
    parent, monkeypatch
):
    from benchmarks import c6_winding_joint_domain as producer

    def forbidden(*args, **kwargs):
        raise AssertionError("a rounding audit must not execute a graph trajectory")

    for name in (
        "run_c6_joint_case",
        "prepare_c6_phase_response",
        "_observed_stage",
        "_advance_forced_support_interval",
    ):
        monkeypatch.setattr(producer, name, forbidden)
    original = deepcopy(parent)
    primitive, calls = campaign.euler_update, []

    def tracked(epi, dt, rate):
        calls.append(dt)
        return primitive(epi, dt, rate)

    monkeypatch.setattr(campaign, "euler_update", tracked)
    result = campaign.analyze_c6_rounding_report(parent)
    assert parent == original
    assert calls == [0.0625] * 24
    assert result["runtime_executed"] is False
    assert result["detached_record_validation"] is True
    assert result["live_provenance_certified"] is False
    assert result["future_bounds_verified"] is False
    assert result["full_state_fixed_point_certified"] is False


def test_every_captured_endpoint_matches_scalar_and_numpy_substeps(parent, report):
    assert tuple(case["mode"] for case in report["cases"]) == ("null", "k1", "k3")
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        assert tuple(cycle["ordinal"] for cycle in analyzed["cycles"]) == (1, 2)
        for cycle, result in zip(retained["cycles"], analyzed["cycles"], strict=True):
            observed = result["observation"]
            endpoint = tuple(cycle["flow"]["raw_after_integrator"]["epi"])
            assert observed["endpoint"] == endpoint
            assert observed["exact_endpoint"] == tuple(map(F, endpoint))
            assert result["numpy_replay"][-1] == endpoint
            assert len(observed["substeps"]) == 4
            assert all(result["endpoint_bindings"].values())
            assert observed["scaling_defect"] == (0,) * 6
            assert observed["error_identity_residual"] == (0,) * 6
            assert (
                max(map(abs, observed["endpoint_defect"]))
                <= observed["local_endpoint_error_bound"]
            )


def test_each_addition_is_inside_the_exact_nearest_even_output_cell(report):
    for case in report["cases"]:
        for cycle in case["cycles"]:
            for step in cycle["observation"]["substeps"]:
                for value, total, error, cell in zip(
                    step["after"],
                    step["unrounded_sum"],
                    step["addition_error"],
                    step["result_cells"],
                    strict=True,
                ):
                    predecessor = math.nextafter(value, -math.inf)
                    successor = math.nextafter(value, math.inf)
                    lower, upper = (F(value) + F(predecessor)) / 2, (
                        F(value) + F(successor)
                    ) / 2
                    assert cell["lower"] == lower
                    assert cell["upper"] == upper
                    assert lower <= total <= upper
                    assert error == F(value) - total
                    assert cell["contains_exact_input"] is True
                    assert cell["exact_input"] == total
                    if total in (lower, upper):
                        assert cell["even_significand"] is True


def test_local_bound_is_derived_from_the_band_not_the_six_observed_maxima(report):
    expected = F(1, 2**51) + F(1, 2**1073)
    measured = []
    for case in report["cases"]:
        for cycle in case["cycles"]:
            observation = cycle["observation"]
            assert observation["local_endpoint_error_bound"] == expected
            assert observation["addition_error_bound"] == F(1, 2**53)
            assert observation["scaling_error_bound"] == F(1, 2**1075)
            measured.append(max(map(abs, observation["endpoint_defect"])))
    assert len(set(measured)) > 1
    assert max(measured) < expected


def test_pressure_plus_rounding_exactly_explains_all_six_mean_changes(parent, report):
    actual_means = []
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        for cycle, result in zip(retained["cycles"], analyzed["cycles"], strict=True):
            before, after = (
                cycle["flow"]["before"],
                cycle["flow"]["raw_after_integrator"],
            )
            pressure_effect = sum(map(F, before["pressure"]), F(0)) / 24
            actual = (
                sum(
                    (
                        F(a) - F(b)
                        for a, b in zip(after["epi"], before["epi"], strict=True)
                    ),
                    F(0),
                )
                / 6
            )
            mean = result["mean_budget"]
            assert mean["held_pressure_mean_effect"] == pressure_effect
            assert mean["actual_mean_change"] == actual
            assert mean["integrator_rounding_mean_effect"] == actual - pressure_effect
            assert mean["identity_residual"] == 0
            assert mean["exact_mean_preserved"] is (actual == 0)
            actual_means.append(actual)
    assert actual_means == [
        F(0),
        F(0),
        F(1, 3 * 2**52),
        F(0),
        F(1, 2**53),
        -F(1, 3 * 2**53),
    ]


def test_both_null_flows_have_half_epi_stasis_while_operator_phases_move(report):
    for cycle in report["cases"][0]["cycles"]:
        observation, stasis = cycle["observation"], cycle["null_stasis"]
        assert observation["half_epi_pressure_band"] == (-F(1, 2**51), F(1, 2**50))
        assert observation["half_epi_pressure_band_membership"] == (True,) * 6
        assert observation["stasis"] == (True,) * 6
        assert observation["endpoint"] == (0.5,) * 6
        assert stasis["source_epi_is_uniform_half"] is True
        assert stasis["all_pressures_in_half_epi_band"] is True
        assert stasis["all_epi_coordinates_stay_fixed"] is True
        assert stasis["captured_epi_stays_fixed"] is True
        assert stasis["phase_stays_fixed_across_operator_cycle"] is False
        assert stasis["future_pressure_band_verified"] is False
        assert stasis["full_state_fixed_point_certified"] is False
        assert cycle["mean_budget"]["held_pressure_mean_effect"] < 0
        assert cycle["mean_budget"]["integrator_rounding_mean_effect"] > 0


def test_full_parent_continuity_validation_precedes_numeric_replay(parent, monkeypatch):
    damaged = deepcopy(parent)
    damaged["cases"][0]["cycles"][1]["before_capture"]["snapshot"]["epi"][0] = "3/4"

    def forbidden(*args, **kwargs):
        raise AssertionError("unvalidated records reached the numeric observer")

    monkeypatch.setattr(campaign, "observe_binary64_unit_quarter_flow", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_rounding_report(damaged)


@pytest.mark.parametrize("binding", ["scalar_endpoint", "numpy_endpoint", "B17_defect"])
def test_disagreement_between_independently_bound_endpoints_fails_closed(
    parent, monkeypatch, binding
):
    if binding == "scalar_endpoint":
        observe = campaign.observe_binary64_unit_quarter_flow

        def changed(*args, **kwargs):
            result = observe(*args, **kwargs)
            return replace(result, endpoint=(0.75,) * 6)

        monkeypatch.setattr(campaign, "observe_binary64_unit_quarter_flow", changed)
    elif binding == "numpy_endpoint":
        monkeypatch.setattr(campaign, "_numpy_replay", lambda *args: ((0.75,) * 6,) * 4)
    else:
        analyze = campaign.analyze_c6_defect_report

        def changed(data):
            result = analyze(data)
            result["cases"][0]["cycles"][0]["pressure_budget"][
                "integrator_epi_effect"
            ] = (F(1),) * 6
            return result

        monkeypatch.setattr(campaign, "analyze_c6_defect_report", changed)
    with pytest.raises(ValueError, match="endpoint|defect|replay"):
        campaign.analyze_c6_rounding_report(parent)


def test_cli_does_not_overwrite_its_source(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_bytes(b"source")
    monkeypatch.setattr(
        sys, "argv", ["audit", "--input", str(path), "--output", str(path)]
    )
    with pytest.raises(ValueError, match="overwrite"):
        campaign.main()
    assert path.read_bytes() == b"source"


def test_cli_preserves_historical_manifest_under_a_separate_consumer_digest(
    source_bytes, tmp_path, monkeypatch
):
    path, output = tmp_path / "input.json", tmp_path / "output.json"
    path.write_bytes(source_bytes)
    scopes = []

    def provenance(root, scope):
        scopes.append(tuple(scope))
        return "b" * 40, False, None

    monkeypatch.setattr(campaign, "current_git_source_provenance", provenance)
    monkeypatch.setattr(
        sys, "argv", ["audit", "--input", str(path), "--output", str(output)]
    )
    campaign.main()
    result, historical = json.loads(output.read_text()), json.loads(source_bytes)
    assert (
        result["input_evidence"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    )
    assert result["input_evidence"]["producer_manifest"] == historical["manifest"]
    assert (
        result["input_evidence"]["producer_source_scope"] == historical["source_scope"]
    )
    assert result["manifest"]["git_sha"] == "b" * 40
    assert result["manifest"]["result_status"] == "derived"
    assert tuple(result["source_scope"]) == campaign.SOURCE_SCOPE
    assert scopes == [campaign.SOURCE_SCOPE] * 2
    assert path.read_bytes() == source_bytes
    assert result["runtime_executed"] is False
    assert result["live_provenance_certified"] is False


@pytest.mark.parametrize("changed", ["input", "consumer_source"])
def test_publication_rejects_mutation_during_the_offline_audit(
    source_bytes, tmp_path, monkeypatch, changed
):
    path, output = tmp_path / "input.json", tmp_path / "output.json"
    path.write_bytes(source_bytes)
    provenance = iter(
        [
            ("a" * 40, False, None),
            (("b" if changed == "consumer_source" else "a") * 40, False, None),
        ]
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: next(provenance)
    )
    if changed == "input":
        analyze = campaign.analyze_c6_rounding_report

        def changing(data):
            result = analyze(data)
            path.write_bytes(source_bytes + b"\n")
            return result

        monkeypatch.setattr(campaign, "analyze_c6_rounding_report", changing)
    monkeypatch.setattr(
        sys, "argv", ["audit", "--input", str(path), "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="source|evidence"):
        campaign.main()
    assert not output.exists()
