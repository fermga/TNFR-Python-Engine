"""Detached C6 pairing reads all inherited stages without extending the run."""

import hashlib
import json
import math
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_pairing as campaign

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
    return campaign.analyze_c6_pairing_report(parent)


def _stages(retained, analyzed):
    yield retained["initial_capture"], retained["initial_joint_readout"][
        "phase_pi_represented"
    ], analyzed["initial"]
    for cycle, result in zip(retained["cycles"], analyzed["cycles"], strict=True):
        for glyph in ("um", "il"):
            phase = tuple(
                F(value) / F(math.pi)
                for value in cycle[glyph]["phase_after"]["represented_lift"]
            )
            for state, suffix in (("raw_capture", "raw"), ("after_capture", "refresh")):
                yield cycle[glyph][state], phase, result["stages"][f"{glyph}_{suffix}"]
        yield cycle["after_capture"], cycle["after_joint_readout"][
            "phase_pi_represented"
        ], result["stages"]["flow_refresh"]


def _pairs(values):
    return (values[0] + values[3], values[1] + values[4], values[2] + values[5])


def _centered_pairs(values):
    twice_mean = sum(values, F(0)) / 3
    return tuple(value - twice_mean for value in _pairs(values))


def test_inherited_validation_runs_once_and_no_graph_word_runs(parent, monkeypatch):
    from benchmarks import c6_winding_joint_domain as producer

    def forbidden(*args, **kwargs):
        raise AssertionError("pairing audit must not execute a graph trajectory")

    for name in (
        "run_c6_joint_case",
        "prepare_c6_phase_response",
        "_observed_stage",
        "_advance_forced_support_interval",
    ):
        monkeypatch.setattr(producer, name, forbidden)
    actual, calls = campaign.analyze_c6_rounding_report, []

    def tracked(data):
        calls.append(1)
        return actual(data)

    monkeypatch.setattr(campaign, "analyze_c6_rounding_report", tracked)
    original = deepcopy(parent)
    result = campaign.analyze_c6_pairing_report(parent)
    assert calls == [1]
    assert parent == original
    assert result["runtime_executed"] is False
    assert result["live_provenance_certified"] is False
    assert result["production_pairing_invariance_certified"] is False
    assert result["future_mean_bound_verified"] is False


def test_all_33_stage_projections_match_primary_records(parent, report):
    count = 0
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        for capture, phases, stage in _stages(retained, analyzed):
            count += 1
            z, epi = tuple(map(F, phases)), tuple(map(F, capture["snapshot"]["epi"]))
            assert stage["phase_pi_represented"] == z
            assert stage["phase_lift_radians"] == tuple(
                value * F(math.pi) for value in z
            )
            assert stage["phase_pair_sums_pi"] == _pairs(z)
            assert stage["phase_centered_pairs_pi"] == _centered_pairs(z)
            assert stage["epi_pair_sums"] == _pairs(epi)
            assert stage["epi_centered_pairs"] == _centered_pairs(epi)
            assert sum(stage["phase_centered_pairs_pi"]) == 0
            assert sum(stage["epi_centered_pairs"]) == 0
    assert count == 33


def test_all_pressure_pairs_split_exactly_into_five_recorded_terms(parent, report):
    raw_write_is_needed = False
    for retained, analyzed in zip(parent["cases"], report["cases"], strict=True):
        for capture, phases, stage in _stages(retained, analyzed):
            z = tuple(map(F, phases))
            x = tuple(map(F, capture["snapshot"]["epi"]))
            gradient = tuple(map(F, capture["phase_gradient"]))
            sigma = tuple(
                gradient[i] + z[i] - (z[(i - 1) % 6] + z[(i + 1) % 6]) / 2
                for i in range(6)
            )
            we = F(capture["epi_weight"])
            wp = F(dict(capture["normalized_weights"])["phase"])
            terms = {
                "epi_pressure": tuple(
                    -F(3, 2) * we * value for value in _centered_pairs(x)
                ),
                "ideal_phase_pressure": tuple(
                    -F(3, 2) * wp * value for value in _centered_pairs(z)
                ),
                "phase_realization": tuple(wp * value for value in _pairs(sigma)),
                "pressure_assembly": _pairs(
                    tuple(map(F, capture["kernel_pressure_defect"]))
                ),
                "stored_pressure_residual": _pairs(
                    tuple(map(F, capture["stored_pressure_residual"]))
                ),
            }
            pressure_pairs = _pairs(
                tuple(map(F, capture["snapshot"]["stored_pressure"]))
            )
            assert stage["phase_realization_error"] == sigma
            assert stage["phase_gradient_pairs"] == _pairs(gradient)
            assert stage["pressure_pair_terms"] == terms
            assert stage["pressure_pairs"] == pressure_pairs
            assert pressure_pairs == tuple(
                sum((term[i] for term in terms.values()), F(0)) for i in range(3)
            )
            assert stage["pressure_pair_identity_residual"] == (0,) * 3
            assert stage["epi_pair_identity_residual"] == (0,) * 3
            raw_write_is_needed |= any(terms["stored_pressure_residual"])
    assert raw_write_is_needed


def test_stored_base_reflection_and_initial_pressure_obstructions_are_explicit(report):
    assert report["stored_base_reflection"]["pairs"] == ((0, 3), (1, 2), (4, 5))
    assert report["stored_base_reflection"]["residuals"] == (0, -F(1, 2**52), 0)
    assert report["phase_convention"]["represented_pi"] == F(math.pi)
    expected = [-F(271851918094053, 2**100), -F(63295, 2**68), -F(15825, 2**66)]
    for case, residual in zip(report["cases"], expected, strict=True):
        assert case["initial"]["phase_centered_pairs_pi"] == (0,) * 3
        assert case["initial"]["epi_centered_pairs"] == (0,) * 3
        assert case["initial"]["pressure_pairs"] == (0, 0, residual)


def test_first_um_loses_centered_pairing_not_only_common_rotation(report):
    expected_radians = (
        (
            F(2001599834386887, 2**107),
            -F(2001599834386887, 2**108),
            -F(2001599834386887, 2**108),
        ),
        (F(327, 2**64), F(11961, 2**65), -F(12615, 2**65)),
        (F(1115, 3 * 2**63), -F(1115, 3 * 2**64), -F(1115, 3 * 2**64)),
    )
    for case, expected in zip(report["cases"], expected_radians, strict=True):
        actual = case["cycles"][0]["stages"]["um_refresh"]["phase_centered_pairs_pi"]
        assert tuple(value * F(math.pi) for value in actual) == expected
        assert any(actual)


def test_quotient_residuals_remove_mean_and_compose_without_dropping_rounding(
    parent, report
):
    for retained, case in zip(parent["cases"], report["cases"], strict=True):
        t = F(retained["joint_domain_reference"]["coupling_phase_factor"])
        alpha = F(retained["joint_domain_reference"]["coherence_phase_factor"])
        fu, fi = 1 - t, 1 - F(3, 2) * alpha
        assert case["phase_pair_factors"] == {"um": fu, "il": fi, "combined": fu * fi}
        previous = case["initial"]["phase_centered_pairs_pi"]
        for cycle in case["cycles"]:
            q = cycle["phase_pair_quotient"]
            assert q["before"] == previous
            assert q["um_residual"] == tuple(
                a - fu * b for a, b in zip(q["after_um"], previous, strict=True)
            )
            assert q["il_residual"] == tuple(
                a - fi * b for a, b in zip(q["after_il"], q["after_um"], strict=True)
            )
            assert q["combined_residual"] == tuple(
                a - fu * fi * b for a, b in zip(q["after_il"], previous, strict=True)
            )
            assert q["combined_residual"] == tuple(
                fi * a + b
                for a, b in zip(q["um_residual"], q["il_residual"], strict=True)
            )
            assert q["composition_identity_residual"] == (0,) * 3
            previous = cycle["stages"]["flow_refresh"]["phase_centered_pairs_pi"]


def test_retained_k3_pressure_pairing_is_lost_specifically_at_il_refresh(report):
    cycle = report["cases"][2]["cycles"][0]
    assert cycle["stages"]["um_refresh"]["pressure_pairs"] == (0,) * 3
    assert cycle["stages"]["il_raw"]["pressure_pairs"] == (0,) * 3
    assert cycle["stages"]["il_refresh"]["pressure_pairs"] == (-F(15823, 2**67), 0, 0)
    assert cycle["stages"]["il_refresh"]["phase_gradient_pairs"] == (
        -F(5215, 2**65),
        0,
        0,
    )
    assert (
        cycle["stages"]["il_raw"]["phase_centered_pairs_pi"]
        == cycle["stages"]["il_refresh"]["phase_centered_pairs_pi"]
    )
    assert cycle["opposite_pressure_lost_by_il_refresh"] is True


def test_inherited_flow_bindings_keep_the_mean_and_binade_obstructions(report):
    for case in report["cases"]:
        for cycle in case["cycles"]:
            binding = cycle["flow_binding"]
            assert all(binding["endpoint_bindings"].values())
            assert binding["mean_budget"]["identity_residual"] == 0
            assert binding["ends_straddling_half_binade"] is (case["mode"] != "null")
            assert binding["starts_straddling_half_binade"] is (
                case["mode"] != "null" and cycle["ordinal"] == 2
            )


def test_corrupted_stage_is_rejected_by_inherited_validation_before_projection(
    parent, monkeypatch
):
    damaged = deepcopy(parent)
    damaged["cases"][0]["cycles"][0]["um"]["raw_capture"]["snapshot"]["epi"][0] = "3/4"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid inherited records reached pairing analysis")

    monkeypatch.setattr(campaign, "_stage", forbidden)
    with pytest.raises(ValueError):
        campaign.analyze_c6_pairing_report(damaged)


def test_cli_separates_historical_input_from_own_source_provenance(
    source_bytes, tmp_path, monkeypatch
):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_bytes(source_bytes)
    scopes = []

    def provenance(root, scope):
        scopes.append(tuple(scope))
        return "c" * 40, False, None

    monkeypatch.setattr(campaign, "current_git_source_provenance", provenance)
    monkeypatch.setattr(
        sys, "argv", ["pairing", "--input", str(source), "--output", str(output)]
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
    assert result["manifest"]["git_sha"] == "c" * 40
    assert result["manifest"]["result_status"] == "derived"
    assert scopes == [campaign.SOURCE_SCOPE] * 2
    assert source.read_bytes() == source_bytes
    assert result["runtime_executed"] is False


@pytest.mark.parametrize("change", ["overwrite", "input", "source"])
def test_cli_refuses_unsafe_or_changed_publication(
    source_bytes, tmp_path, monkeypatch, change
):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_bytes(source_bytes)
    if change == "overwrite":
        output = source
    versions = iter(
        [
            ("c" * 40, False, None),
            (("d" if change == "source" else "c") * 40, False, None),
        ]
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: next(versions)
    )
    if change == "input":
        analyze = campaign.analyze_c6_pairing_report

        def changed(data):
            result = analyze(data)
            source.write_bytes(source_bytes + b"\n")
            return result

        monkeypatch.setattr(campaign, "analyze_c6_pairing_report", changed)
    monkeypatch.setattr(
        sys, "argv", ["pairing", "--input", str(source), "--output", str(output)]
    )
    with pytest.raises((ValueError, RuntimeError)):
        campaign.main()
    if change == "overwrite":
        assert source.read_bytes() == source_bytes
    else:
        assert not output.exists()
