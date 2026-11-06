"""Portable retained geometry fixtures; no producer or native kernel runs."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F

import pytest

from benchmarks import thol_regional_input_geometry as study
from tnfr.physics.regional_response import observe_regional_response
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


def _manifest(claim=study.INPUT_CLAIM):
    return CoreExperimentManifest(
        claim_id=claim,
        git_sha="a" * 40,
        source_dirty=False,
        versions={"python": "fixture"},
        graph_construction="Detached five-coordinate synthetic matrices",
        capacity_specification="Declared positive test metric",
        solver="Exact arithmetic fixture",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=("reception",),
        telemetry=("regional geometry",),
        controls=("synthetic fixture",),
        artifacts=("fixture.json",),
    ).to_dict()


def _fixture():
    nodes = ("p0", "p1", "c0", "c1", "c2")
    region, metric = (2, 3, 4), (F(1), F(2), F(3), F(4), F(5))
    s = [[F(i == j) for j in range(5)] for i in range(5)]
    s[2][0], s[2][2] = F(1, 2), F(1, 2)
    a = [[F(0)] * 5 for _ in range(5)]
    a[1][1], a[1][3], a[3][1], a[3][3] = F(1), F(-1), F(-1), F(1)
    s, a = tuple(map(tuple, s)), tuple(map(tuple, a))
    t = tuple(
        tuple(x - F(1, 4) * y for x, y in zip(sr, ar, strict=True))
        for sr, ar in zip(s, a, strict=True)
    )
    common = {"S": s, "A": a, "T": t, "metric_weights": metric, "dt": F(1, 4)}
    stages = {
        label: asdict(
            observe_regional_response(matrix, metric, region, (0, 1, 2, 3, 4))
        )
        for label, matrix in (("reception", s), ("held_pressure_interval", t))
    }
    return study._payload(
        {
            "manifest": _manifest(),
            "nodes": nodes,
            "children": nodes[2:],
            "region_indices": region,
            "common_coefficients": common,
            "admission": {
                "original_reference": {
                    "source": {"nodes": nodes},
                    "metric_weights": metric,
                },
                "children": nodes[2:],
                "common_coefficients": deepcopy(common),
            },
            "witnesses": {
                name: {"stages": deepcopy(stages)} for name in study.WITNESSES
            },
        }
    )


def _write_fixture(tmp_path, retained=None):
    path = tmp_path / "criterion.json"
    path.write_text(
        json.dumps(_fixture() if retained is None else retained), encoding="utf-8"
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _cli(monkeypatch, path, digest, output):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "geometry",
            "--input",
            str(path),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        study, "current_git_source_provenance", lambda *a: ("a" * 40, False, None)
    )


def test_two_maps_once_and_all_witness_geometries_checked(monkeypatch):
    calls = []
    original = study.observe_regional_input_geometry

    def observe(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(study, "observe_regional_input_geometry", observe)
    result = study.analyze_retained(_fixture())
    assert len(calls) == result["geometry_evaluations"] == 2
    reception = result["geometries"]["reception"]
    held = result["geometries"]["held_pressure_interval"]
    assert (reception.rank, reception.protected_dimension) == (1, 1)
    assert (held.rank, held.protected_dimension) == (2, 0)
    assert result["comparison"]["rank_change"] == 1
    assert result["geometry_summaries"]["reception"]["parent_only_rank"] == 1
    assert (
        result["geometry_summaries"]["held_pressure_interval"]["parent_only_rank"] == 2
    )
    assert result["comparison"]["parent_only_rank_change"] == 1
    assert (
        result["comparison"]["input_map_difference"]
        == result["comparison"]["expected_pressure_contribution"]
    )
    assert (
        result["native_calls"]
        == result["kernel_calls"]
        == result["new_trajectories"]
        == 0
    )
    assert not result["input_reachability_certified"]
    assert not result["repeated_invariance_certified"]


@pytest.mark.parametrize("name", study.WITNESSES)
@pytest.mark.parametrize("stage", tuple(label for label, _ in study.STAGES))
@pytest.mark.parametrize(
    "field",
    ("transition", "metric_weights", "region_indices", "centering", "nullspace_images"),
)
def test_each_retained_geometry_component_is_rederived(name, stage, field):
    retained = _fixture()
    row = retained["witnesses"][name]["stages"][stage]
    if field in ("transition", "centering"):
        row[field][0][0] = "99"
    elif field == "metric_weights":
        row[field][0] = "99"
    elif field == "region_indices":
        row[field][0] = 0
    else:
        row[field][0][1][0] = "99"
    with pytest.raises(ValueError, match="mismatch"):
        study.analyze_retained(retained)


@pytest.mark.parametrize(
    "change",
    (
        "held_identity",
        "common_binding",
        "metric_binding",
        "node_binding",
        "cohort_binding",
        "indices_binding",
        "boolean_index",
        "dt",
        "missing_witness",
        "extra_stage",
        "nullspace_label",
    ),
)
def test_inconsistent_ancestry_or_common_maps_rejected(change):
    retained = _fixture()
    common = retained["common_coefficients"]
    if change == "held_identity":
        common["T"][0][0] = "99"
        retained["admission"]["common_coefficients"] = deepcopy(common)
    elif change == "common_binding":
        common["S"][0][0] = "99"
    elif change == "metric_binding":
        retained["admission"]["original_reference"]["metric_weights"][0] = "99"
    elif change == "node_binding":
        retained["admission"]["original_reference"]["source"]["nodes"].reverse()
    elif change == "cohort_binding":
        retained["admission"]["children"].reverse()
    elif change == "indices_binding":
        retained["region_indices"].reverse()
    elif change == "boolean_index":
        retained["region_indices"][0] = True
    elif change == "dt":
        common["dt"] = "1/2"
        retained["admission"]["common_coefficients"] = deepcopy(common)
    elif change == "missing_witness":
        del retained["witnesses"]["localized"]
    elif change == "extra_stage":
        retained["witnesses"]["localized"]["stages"]["unplanned"] = {}
    else:
        retained["witnesses"]["localized"]["stages"]["reception"]["nullspace_images"][
            0
        ][0] = "wrong_label"
    with pytest.raises(ValueError):
        study.analyze_retained(retained)


def test_nullspace_boolean_is_not_treated_as_evidence():
    retained = _fixture()
    for witness in retained["witnesses"].values():
        for stage in witness["stages"].values():
            stage["nullspace_preserved"] = True
    result = study.analyze_retained(retained)
    assert result["geometries"]["held_pressure_interval"].rank == 2


def test_real_manifests_authenticated_input_and_complete_dataclass_output(
    tmp_path, monkeypatch
):
    path, digest = _write_fixture(tmp_path)
    output = tmp_path / "geometry.json"
    _cli(monkeypatch, path, digest, output)

    def forbidden(*args, **kwargs):
        raise AssertionError("historical admission/kernel must not execute")

    monkeypatch.setattr(study.reset, "audit_reset_step", forbidden)
    monkeypatch.setattr(study.reset, "load_evidence", forbidden)
    study.main()
    report = json.loads(output.read_bytes())
    assert (
        report["manifest"]["claim_id"] == "O3.a-regional-environmental-input-geometry"
    )
    CoreExperimentManifest(**report["manifest"]).validate_for_admission()
    assert report["historical_inputs"]["criterion"]["sha256"] == digest
    assert report["geometries"]["reception"]["rank"] == 1
    assert report["geometries"]["held_pressure_interval"]["protected_dimension"] == 0
    assert report["geometries"]["reception"]["protected_readout_rows"]
    assert report["source_scope"] == ["src/tnfr", "benchmarks"]


@pytest.mark.parametrize("damage", ("digest", "claim", "manifest"))
def test_authentication_failure_precedes_arithmetic(tmp_path, monkeypatch, damage):
    retained = _fixture()
    if damage == "claim":
        retained["manifest"]["claim_id"] = "O3.a-other-claim"
    elif damage == "manifest":
        retained["manifest"]["timestep"] = -1
    path, digest = _write_fixture(tmp_path, retained)
    if damage == "digest":
        digest = "0" * 64
    output = tmp_path / "geometry.json"
    _cli(monkeypatch, path, digest, output)

    def forbidden(*args):
        raise AssertionError("no arithmetic before authentication")

    monkeypatch.setattr(study, "analyze_retained", forbidden)
    with pytest.raises(ValueError):
        study.main()
    assert not output.exists()


def test_input_overwrite_refused_before_reading(tmp_path, monkeypatch):
    path, digest = _write_fixture(tmp_path)
    before = path.read_bytes()
    _cli(monkeypatch, path, digest, path)

    def forbidden(*args):
        raise AssertionError("must refuse before reading")

    monkeypatch.setattr(study.reset, "_load", forbidden)
    with pytest.raises(ValueError, match="overwrite"):
        study.main()
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ("input", "source"))
def test_postanalysis_binding_change_prevents_output(tmp_path, monkeypatch, change):
    path, digest = _write_fixture(tmp_path)
    output = tmp_path / "geometry.json"
    _cli(monkeypatch, path, digest, output)
    original = study.analyze_retained

    def analyze(retained):
        result = original(retained)
        if change == "input":
            path.write_text("{}", encoding="utf-8")
        else:
            monkeypatch.setattr(
                study,
                "current_git_source_provenance",
                lambda *a: ("b" * 40, False, None),
            )
        return result

    monkeypatch.setattr(study, "analyze_retained", analyze)
    with pytest.raises(RuntimeError, match="changed"):
        study.main()
    assert not output.exists()
