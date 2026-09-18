"""Portable complete-family replay and exact retained-model realizations."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_forced_epi_realization import _assert_realization
from tests.physics.test_thol_family_closure import retained_fixture  # noqa: F401
from tnfr.research.core_manifests import CoreExperimentManifest


def _write(path, payload):
    raw = (json.dumps(_payload(payload), allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _kwargs(fixture):
    return {
        "expected_family_sha256": fixture["family_hash"],
        "expected_control_sha256": fixture["control_hash"],
        "expected_lineage_sha256": fixture["lineage_hash"],
    }


def _paths(fixture):
    return tuple(fixture[name + "_path"] for name in ("family", "control", "lineage"))


@pytest.fixture(scope="module")
def family_fixture(retained_fixture, tmp_path_factory):  # noqa: F811 - shared pytest fixture
    from benchmarks.thol_family_closure import run_study

    scientific = run_study(
        retained_fixture["control_path"], retained_fixture["lineage_path"],
        expected_control_sha256=retained_fixture["control_hash"],
        expected_lineage_sha256=retained_fixture["lineage_hash"],
    )
    path = tmp_path_factory.mktemp("family_realization") / "family.json"
    metadata = deepcopy(retained_fixture["control"]["manifest"])
    metadata.update(
        claim_id="O1.b-generated-family-affine-state-sufficiency", timestep=None, seed=None,
        solver="Detached exact family geometry; synthetic fixture without a trajectory",
        telemetry=["Exact closure, witnesses, four held models and observer reset"],
        artifacts=[str(path)],
    )
    CoreExperimentManifest(**metadata).validate_for_admission()
    report = {"manifest": metadata, "source_scope": retained_fixture["control"]["source_scope"],
              **scientific, "experimental_status": "No empirical correspondence tested"}
    digest = _write(path, report)
    return {**retained_fixture, "family_path": path, "family_hash": digest,
            "family": json.loads(path.read_bytes()), "scientific": scientific}


@pytest.fixture(scope="module")
def study(family_fixture):
    from benchmarks.thol_family_realization import run_study

    with patch("tnfr.physics.epi_memory.matrix_exponential", side_effect=AssertionError("no exponential")), patch(
        "tnfr.physics.structural_morphism.matrix_exponential", side_effect=AssertionError("no numerical flow"),
    ), patch("benchmarks.thol_lineage_coordination.run_distributed_target_branch",
             side_effect=AssertionError("no trajectory")):
        return run_study(*_paths(family_fixture), **_kwargs(family_fixture))


def test_complete_replayed_family_and_producer_metadata_are_retained(study, family_fixture):
    assert study["partition"] == family_fixture["scientific"]["partition"]
    assert tuple(row["model"] for row in study["models"]) == (
        "original", "original_parents", "born_children", "all_node_um",
    )
    binding = study["retained_family"]
    assert binding["sha256"] == family_fixture["family_hash"]
    assert binding["historical_manifest"] == family_fixture["family"]["manifest"]
    assert binding["historical_source_scope"] == family_fixture["family"]["source_scope"]
    assert binding["historical_producer_preserved"]
    assert study["family_replay"]["whole_scientific_payload_equal"]
    assert set(study["family_replay"]["compared_fields"]) == set(family_fixture["scientific"])
    assert not study["new_trajectories_executed"]
    assert not study["matrix_exponentials_evaluated"]
    assert not study["partition_search_performed"]


def test_every_minimal_realization_matches_independent_exact_row_space(study, family_fixture):
    for row, old in zip(study["models"], family_fixture["scientific"]["models"], strict=True):
        raw = row["realization"]
        result = SimpleNamespace(**{**raw, "level_records": tuple(
            SimpleNamespace(**level) for level in raw["level_records"]
        )})
        _assert_realization(result, family_fixture["references"][row["model"]], study["partition"]["blocks"])
        assert raw["closure"] == old["exact_observation"]
        assert row["reference_sha256"] == old["reference_sha256"]
        assert row["observation_time"] == old["observation_time"] == 1.0
        assert row["observer_reset"] == old["observer_reset"]
        assert row["old_target_readout"] == old["old_target_readout"]
        assert raw["epi"] == old["actual_snapshot"]["epi"]
        assert raw["max_rank_calls"] == study["max_rank_calls_per_model"]
        assert raw["rank_calls"] <= raw["max_rank_calls"]


@pytest.mark.parametrize("mutation", (
    "wrong_producer", "scope_promotion", "empty_source_scope", "extra_scientific_field",
    "missing_scientific_field", "closure_flag", "closure_matrix", "witness", "norm",
    "reset", "snapshot", "old_target", "parentage", "input_hash", "input_producer",
))
def test_matching_bytes_do_not_authorize_tampered_cached_family_calculations(
    family_fixture, tmp_path, mutation,
):
    from benchmarks.thol_family_realization import load_family_evidence

    payload = deepcopy(family_fixture["family"])
    row = payload["models"][0]
    if mutation == "wrong_producer":
        payload["manifest"]["claim_id"] = "O1.b-unrelated"
    elif mutation == "scope_promotion":
        payload["experimental_status"] = "Physical emergence established"
    elif mutation == "empty_source_scope":
        payload["source_scope"] = []
    elif mutation == "extra_scientific_field":
        payload["unsupported_claim"] = True
    elif mutation == "missing_scientific_field":
        del payload["scope"]
    elif mutation == "closure_flag":
        row["exact_observation"]["all_state_affine_closed"] = not row["exact_observation"]["all_state_affine_closed"]
    elif mutation == "closure_matrix":
        row["exact_observation"]["micro_generator"][0][0] = "99"
    elif mutation == "witness":
        row["exact_observation"]["witness"] = {"fabricated": True}
    elif mutation == "norm":
        row["numerical_crosscheck"]["projection_defect_spectral_norm"] += 0.25
    elif mutation == "reset":
        row["observer_reset"]["projection_reweighting"][0] = "99"
    elif mutation == "snapshot":
        row["actual_snapshot"]["capacity"][0] = "99"
    elif mutation == "old_target":
        row["old_target_readout"]["compatibility_energy"] = "99"
    elif mutation == "parentage":
        payload["partition"]["parent_children"][0][1] = 3
    elif mutation == "input_hash":
        payload["retained_controls"]["sha256"] = "0"*64
    else:
        payload["retained_lineage"]["historical_manifest"]["solver"] = "replaced producer"
    path = tmp_path / "changed.json"
    digest = _write(path, payload)
    args = {**_kwargs(family_fixture), "expected_family_sha256": digest}
    with pytest.raises(ValueError):
        load_family_evidence(path, family_fixture["control_path"], family_fixture["lineage_path"], **args)


def test_identical_input_bytes_can_move_without_reassigning_historical_producer(family_fixture, tmp_path):
    from benchmarks.thol_family_realization import load_family_evidence

    control = tmp_path / "moved_control.json"
    lineage = tmp_path / "moved_lineage.json"
    control.write_bytes(family_fixture["control_path"].read_bytes())
    lineage.write_bytes(family_fixture["lineage_path"].read_bytes())
    replay, binding, comparison = load_family_evidence(
        family_fixture["family_path"], control, lineage, **_kwargs(family_fixture),
    )
    assert replay["retained_controls"]["path"] == str(control)
    assert replay["retained_lineage"]["path"] == str(lineage)
    assert binding["historical_manifest"] == family_fixture["family"]["manifest"]
    assert comparison["locator_fields_normalized"] == ("retained_controls.path", "retained_lineage.path")


@pytest.mark.parametrize("which", ("family", "control", "lineage"))
def test_declared_input_hashes_cannot_be_replaced_by_automatic_admission(family_fixture, which):
    from benchmarks.thol_family_realization import load_family_evidence

    hashes = {**_kwargs(family_fixture), "expected_" + which + "_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest"):
        load_family_evidence(*_paths(family_fixture), **hashes)


@pytest.mark.parametrize("maximum", (True, False, 0, -1, 20.0, "20", None, Fraction(20)))
def test_invalid_budget_is_rejected_before_any_file_read(maximum):
    from benchmarks.thol_family_realization import run_study

    with pytest.raises(ValueError, match="max_rank_calls"):
        run_study("missing.json", "missing_control.json", "missing_lineage.json", max_rank_calls=maximum)


def test_incomplete_rank_budget_cannot_return_a_partial_four_model_result(family_fixture):
    from benchmarks.thol_family_realization import run_study

    with pytest.raises(ValueError, match="incomplete"):
        run_study(*_paths(family_fixture), **_kwargs(family_fixture), max_rank_calls=1)


def _cli_args(fixture, output):
    return [
        "thol_family_realization", "--family", str(fixture["family_path"]),
        "--controls", str(fixture["control_path"]), "--lineage", str(fixture["lineage_path"]),
        "--expected-family-sha256", fixture["family_hash"],
        "--expected-control-sha256", fixture["control_hash"],
        "--expected-lineage-sha256", fixture["lineage_hash"], "--output", str(output),
    ]


@pytest.mark.parametrize("input_name", ("family", "control", "lineage"))
def test_cli_cannot_overwrite_any_retained_input(family_fixture, monkeypatch, input_name):
    import benchmarks.thol_family_realization as benchmark

    destination = family_fixture[input_name + "_path"]
    before = destination.read_bytes()
    monkeypatch.setattr("sys.argv", _cli_args(family_fixture, destination))
    with pytest.raises(ValueError, match="overwrite"):
        benchmark.main()
    assert destination.read_bytes() == before


def test_cli_exhaustion_preserves_an_existing_output_instead_of_publishing_partial_data(
    family_fixture, tmp_path, monkeypatch,
):
    import benchmarks.thol_family_realization as benchmark

    output = tmp_path / "existing.json"
    output.write_bytes(b"previous result")
    monkeypatch.setattr("sys.argv", [*_cli_args(family_fixture, output), "--max-rank-calls", "1"])
    with pytest.raises(ValueError, match="incomplete"):
        benchmark.main()
    assert output.read_bytes() == b"previous result"


def test_cli_json_retains_full_exact_algebra_and_the_previous_producer(study, family_fixture, tmp_path, monkeypatch):
    import benchmarks.thol_family_realization as benchmark

    output = tmp_path / "realization.json"
    monkeypatch.setattr(benchmark, "run_study", lambda *args, **kwargs: study)
    monkeypatch.setattr("sys.argv", _cli_args(family_fixture, output))
    benchmark.main()
    payload = json.loads(output.read_bytes())
    CoreExperimentManifest(**payload["manifest"]).validate_for_admission()
    assert payload["manifest"]["timestep"] is payload["manifest"]["seed"] is None
    assert payload["experimental_status"] == "No empirical correspondence tested"
    assert payload["retained_family"]["historical_manifest"] == family_fixture["family"]["manifest"]
    assert len(payload["models"]) == 4
    for row in payload["models"]:
        assert row["realization"]["observation"]
        assert row["realization"]["reduced_source"]
    assert not payload["matrix_exponentials_evaluated"]
