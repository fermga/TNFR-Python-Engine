"""Declared snapshot comparison, with explicit runtime and input boundaries."""

from fractions import Fraction as F
import hashlib
import json
import sys

import pytest

from benchmarks import thol_regional_map_symmetry as study
from test_thol_regional_map_symmetry import _fixture


def _record():
    retained = _fixture()
    for witness in retained["admission"]["witnesses"].values():
        for side in ("control", "perturbed"):
            branch = witness[side]
            for i, row in enumerate(branch["rows"]):
                row["control"] = {"EN_mix": "1/2", "neighbor_indices": branch["snapshot"]["support_neighbors"][i]}
                row["configuration"] = {"EPI_MIN": -4.0, "EPI_MAX": 4.0,
                                        "CLIP_MODE": "hard", "GLYPH_FACTORS": {"EN_mix": 0.5}}
    return retained


def test_snapshot_restores_map_symmetry_but_not_captured_field_symmetry():
    result = study.analyze_snapshot_comparison(_record())
    assert len(result["symmetry"].common_group_indices) == 1
    comparison = result["snapshot_comparison"]
    symmetry = comparison["symmetry"]
    assert {key: len(group) for key, group in symmetry.operator_group_indices} == {"A": 2, "J": 2, "U": 2}
    assert len(symmetry.common_group_indices) == 2
    assert len(dict(symmetry.field_group_indices)["cohort_source_rate_b"]) == 1
    for stage, row in comparison["common_group_environment"].items():
        assert row["fixed_input_dimension"] == 2
        assert row["fixed_input_rank"] == 0
        assert row["fixed_input_protected_dimension"] == 1
        assert row["unrestricted_input_rank"] == 1
        assert row["unrestricted_protected_dimension"] == 0
        assert any(comparison["contrasts"][stage]["sequential_mean_contrast_image"])
        assert not any(comparison["contrasts"][stage]["snapshot_mean_contrast_image"])
    assert len(comparison["row_admission"]) == result["coefficient_builder_calls"] == result["kernel_calls"] == 16
    assert result["native_calls"] == result["scalar_evolution_kernel_calls"] == result["new_trajectories"] == 0
    assert result["geometry_evaluations"] == 4
    assert comparison["runtime_admission"]["status"] == "not_certified"
    assert not comparison["runtime_admission"]["sequential_defects_transferred"]
    assert not result["complete_runtime_equivariance_certified"]


def test_same_snapshot_J_is_local_rows_and_U_uses_pre_generated_A():
    retained = _record()
    result = study.analyze_snapshot_comparison(retained)
    matrices = dict(result["snapshot_comparison"]["symmetry"].operators)
    saved = retained["admission"]["witnesses"]["cohort"]["control"]["rows"]
    assert matrices["J"] == tuple(tuple(map(F, row["row"])) for row in saved)
    assert matrices["J"] != dict(result["symmetry"].operators)["S"]
    assert matrices["U"] == tuple(tuple(j-a/4 for j, a in zip(jr, ar, strict=True))
                                  for jr, ar in zip(matrices["J"], matrices["A"], strict=True))


def test_actual_paired_environment_is_retained_instead_of_symmetrized():
    result = study.analyze_snapshot_comparison(_record())
    symmetry = result["snapshot_comparison"]["symmetry"]
    fields = dict(symmetry.fields)
    assert fields["cohort_paired_environment"] == (F(0), F(0), F(1), F(1))
    assert fields["localized_paired_environment"] == (F(0), F(0), F(1, 2), F(1, 2))
    assert len(dict(symmetry.field_group_indices)["localized_paired_environment"]) == 2
    assert result["snapshot_comparison"]["paired_environment_fixed"] == {"cohort": True, "localized": True}
    # The fixture has symmetric environmental differences, unlike the real
    # retained experiment. The result is derived, not hardcoded to identity.


@pytest.mark.parametrize("corruption", ("mix", "configured_mix", "wrong_row", "neighbors", "duplicate", "boolean_index", "interval", "different_interval", "clip", "missing"))
def test_snapshot_row_premises_fail_closed(corruption):
    retained = _record()
    row = retained["admission"]["witnesses"]["localized"]["perturbed"]["rows"][0]
    if corruption == "mix":
        row["control"]["EN_mix"] = "1/3"
    elif corruption == "configured_mix":
        row["configuration"]["GLYPH_FACTORS"]["EN_mix"] = 0.25
    elif corruption == "wrong_row":
        row["control"]["EN_mix"] = "1/4"
        row["configuration"]["GLYPH_FACTORS"]["EN_mix"] = 0.25
    elif corruption == "neighbors":
        row["control"]["neighbor_indices"] = [1, 3]
    elif corruption == "duplicate":
        row["control"]["neighbor_indices"] = [1, 1, 2]
    elif corruption == "boolean_index":
        row["control"]["neighbor_indices"] = [True, 2]
    elif corruption == "interval":
        row["configuration"]["EPI_MAX"] = -4
    elif corruption == "different_interval":
        row["configuration"]["EPI_MAX"] = 5
    elif corruption == "clip":
        row["configuration"]["CLIP_MODE"] = "soft"
    else:
        del row["control"]
    with pytest.raises((ValueError, KeyError)):
        study.analyze_snapshot_comparison(retained)


def test_snapshot_claim_and_real_manifest_serialize_via_cli(tmp_path, monkeypatch):
    source, output = tmp_path / "criterion.json", tmp_path / "comparison.json"
    source.write_text(json.dumps(_record()), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["compare", "--snapshot-comparison", "--input", str(source),
                                      "--expected-sha256", digest, "--output", str(output)])
    study.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["manifest"]["claim_id"] == "O3.a-snapshot-reception-comparison"
    assert result["snapshot_comparison"]["runtime_admission"]["stage_calls"] == 0
    assert result["coefficient_builder_calls"] == 16
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
