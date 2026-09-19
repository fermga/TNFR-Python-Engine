"""Retained B43 evidence and hypothetical mean-boundary inputs stay distinct."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_mean_cylinder as campaign

SOURCE = (
    Path(__file__).resolve().parents[2] / "artifacts/research" / campaign.INPUT_NAME
)


@pytest.fixture(scope="module")
def parent():
    if not SOURCE.is_file():
        pytest.skip("the retained original 89-cycle B43 report is unavailable")
    return json.loads(SOURCE.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_winding_mean_cylinder(parent)


def test_original_tail_is_replayed_without_overwriting_its_actual_carry(report, parent):
    source = report["source"]
    assert (
        campaign._payload(source["retained_B43_endpoint"])
        == parent["B43_original_phase_tail_runtime"]["endpoint_before_SHA"]
    )
    assert source["origin_mean"] == F(1, 2) + F(3293, 3 * 2**114)
    assert source["origin_energy"] <= source["energy_bound"]
    assert source["historical_nodal_steps_replayed"] == 356
    assert source["serialized_numerical_chain_verified"] is True
    assert source["live_execution_seal_recreated"] is False
    assert report["new_saved_trajectory_steps"] == 0
    assert report["saved_trajectory_escape_certified"] is False
    assert report["future_runtime_certified"] is False
    assert report["general_correlated_region_excluded"] is False


def test_entire_local_window_has_both_outward_hypothetical_witnesses(report):
    bound = report["B44_mean_cylinder_obstruction"]
    delta, grid = F(1, 2**54), F(1, 2**3222)
    assert bound["mean_lower"] == F(1, 2) - 5 * delta / 6 + grid
    assert bound["mean_upper"] == F(1, 2) + delta / 6 - grid
    assert bound["positive_mean_increment"] == F(3, 2**114)
    assert bound["negative_mean_increment"] == -F(1, 2**114)
    for record in report["B44_hypothetical_boundary_witnesses"].values():
        assert "obstruction" not in record
        for name, sign in (("upper_boundary", 1), ("lower_boundary", -1)):
            witness = record[name]
            endpoint = record["mean_upper"] if sign == 1 else record["mean_lower"]
            assert (
                record["mean_lower"] <= witness["mean_before"] <= record["mean_upper"]
            )
            assert sign * (witness["mean_after"] - endpoint) > 0
            assert witness["energy"] <= report["source"]["energy_bound"]
            assert witness["band_failure"] is False
            assert witness["step"]["nodal_balance_residual"] == (F(0),) * 6
            assert witness["state"] != report["source"]["retained_B43_endpoint"]
    assert "closure" not in bound


def test_complete_analysis_preserves_input_and_serializes_exact_evidence(parent):
    before = deepcopy(parent)
    result = campaign.analyze_c6_winding_mean_cylinder(parent)
    assert parent == before
    json.dumps(campaign._payload(result), allow_nan=False)


def test_affine_restriction_retains_derived_spacings_and_exact_outward_witnesses(
    report,
):
    bound = report["B45_affine_mean_obstruction"]
    base = report["B44_mean_cylinder_obstruction"]
    small = F(1, 2**113)
    assert bound["gradient_value_count"] == 336
    assert bound["coordinate_spacings"] == tuple(
        value * small for value in (1, 1, 2, 4, 8, 4)
    )
    assert bound["coordinate_residues"] == (F(0),) * 6
    assert bound["mean_quantum"] == small / 6
    assert bound["lift_error_squared_bound"] == 462 * small**2
    assert bound["mean_lower"] == base["mean_lower"] + 8 * small
    assert bound["mean_upper"] == base["mean_upper"] - 19 * small
    assert bound["positive_energy_bound"] < report["source"]["energy_bound"]
    assert bound["negative_energy_bound"] < report["source"]["energy_bound"]
    assert "base_obstruction" not in bound
    for record in report["B45_hypothetical_boundary_witnesses"].values():
        for key in ("upper_boundary", "lower_boundary"):
            witness = record[key]
            assert witness["step"]["before"] == witness["state"]
            assert witness["band_failure"] is False
            for state in (witness["state"], witness["step"]["after"]):
                exact = tuple(
                    F(x) + r
                    for x, r in zip(state["epi"], state["remainder"], strict=True)
                )
                assert all(
                    (x / g).denominator == 1
                    for x, g in zip(exact, bound["coordinate_spacings"], strict=True)
                )
            if key == "upper_boundary":
                assert witness["mean_after"] > record["mean_upper"]
            else:
                assert witness["mean_after"] < record["mean_lower"]
    assert report["B44_coordinate_affine_coset_restriction_applied"] is False
    assert report["B45_coordinate_affine_coset_restriction_applied"] is True
    assert report["local_affine_mean_cylinder_invariance_excluded"] is True


def test_public_mean_observers_use_the_same_physics_owners():
    import tnfr.physics as physics
    from tnfr.physics import c6_carried_affine_mean, c6_carried_mean_cylinder

    for owner in (c6_carried_affine_mean, c6_carried_mean_cylinder):
        for name in owner.__all__:
            assert name in physics.__all__
            assert getattr(physics, name) is getattr(owner, name)


@pytest.mark.parametrize(
    "path,value",
    (
        (("cycle_count",), True),
        (("runtime_provenance_certified_at_capture",), False),
        (("future_runtime_certified",), True),
        (("source", "initial_state", "remainder", 0), "1/2"),
        (("source", "initial_phase", 0), -0.0),
        (("source", "random_seed"), 18),
        (("source", "carry_imported_or_reset"), True),
        (("source", "normalized_weights", 0, 1), "1"),
        (("cycles", 0, "ordinal"), 2),
        (("cycles", 0, "coupling", "binding_preserved"), False),
        (("cycles", 0, "coherence", "capacity_after", 0), 0.5),
        (("cycles", 0, "flows", 0, "phase_before", 0), 0.125),
        (("cycles", 0, "flows", 0, "step", "pressure", 0), 0.5),
        (("cycles", 0, "flows", 0, "start_time"), 0.125),
        (("endpoint_before_SHA", "remainder", 0), "0"),
        (("mean_area",), "0"),
        (("tail_entry_closure", "energy_bound"), "0"),
        (("terminal_silence", "capacity_after", 0), 1.0),
    ),
)
def test_numerical_replay_rejects_corrupted_capture_fields(parent, path, value):
    changed = deepcopy(parent)
    entry = changed["B43_original_phase_tail_runtime"]
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.replay_c6_phase_tail_evidence(changed)


@pytest.mark.parametrize(
    "change",
    ("manifest", "scope", "missing_runtime", "truncated_cycles", "truncated_flows"),
)
def test_source_and_completeness_obligations_are_mandatory(parent, change):
    changed = deepcopy(parent)
    runtime = changed["B43_original_phase_tail_runtime"]
    if change == "manifest":
        changed["manifest"]["claim_id"] = "different-claim"
    elif change == "scope":
        changed["source_scope"] = ["src/tnfr"]
    elif change == "missing_runtime":
        changed["B43_original_phase_tail_runtime"] = None
    elif change == "truncated_cycles":
        runtime["cycles"].pop()
    else:
        runtime["cycles"][0]["flows"].pop()
    with pytest.raises(ValueError):
        campaign.replay_c6_phase_tail_evidence(changed)


def test_cli_preserves_direct_input_hash_and_producer_manifest(
    tmp_path, monkeypatch, parent, report
):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    raw = json.dumps(parent).encode()
    source.write_bytes(raw)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_mean_cylinder", lambda _parent: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *_: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    monkeypatch.setattr(
        sys, "argv", ["campaign", "--input", str(source), "--output", str(output)]
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["input_evidence"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["input_evidence"]["producer_manifest"] == parent["manifest"]
    assert result["input_evidence"]["producer_source_scope"] == parent["source_scope"]
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64
    assert source.read_bytes() == raw


@pytest.mark.parametrize("change", ("source", "input"))
def test_cli_rejects_mid_analysis_provenance_changes(
    tmp_path, monkeypatch, parent, report, change
):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(parent))
    versions = iter(
        (
            ("a" * 40, True, "sha256:" + "b" * 64),
            ("a" * 40, True, "sha256:" + ("c" if change == "source" else "b") * 64),
        )
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *_: next(versions)
    )

    def analyze(_parent):
        if change == "input":
            source.write_text("{}")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_mean_cylinder", analyze)
    monkeypatch.setattr(
        sys, "argv", ["campaign", "--input", str(source), "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="source or historical input changed"):
        campaign.main()
    assert not output.exists()


def test_cli_never_overwrites_its_historical_source(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["campaign", "--input", str(source), "--output", str(source)]
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert source.read_text() == "{}"
