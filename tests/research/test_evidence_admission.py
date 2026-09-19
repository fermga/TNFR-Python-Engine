"""Adversarial metadata and actual-content admission controls."""

import hashlib
import json
from dataclasses import replace

import pytest

from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    EvidenceAdmissionError,
    EvidenceSidecar,
    ExperimentManifest,
    NumericalCertificate,
    certify_within_tolerance,
)
from tnfr.research import evidence_sidecar as owner


def certificate(**changes):
    values = dict(
        quantity="residual",
        value=0.25,
        tolerance=0.5,
        passed=True,
        backward_error=0.001,
        condition_number=2.0,
        precision="float64",
    )
    values.update(changes)
    return NumericalCertificate(**values)


def sidecar(tmp_path, **changes):
    artifact = tmp_path / "response.json"
    artifact.write_bytes(b'{"prediction": 3}\n')
    manifest = CoreExperimentManifest(
        claim_id="O6.fixture",
        git_sha="deadbeef",
        versions={"python": "3.13"},
        graph_construction="declared fixture",
        capacity_specification="positive",
        solver="none: metadata fixture",
        result_status=ClaimStatus.MEASURED,
        telemetry=("response",),
        controls=("changed bytes",),
        artifacts=("response.json",),
        source_dirty=False,
    )
    values = dict(
        manifest=manifest,
        artifact="response.json",
        model="finite fixture",
        norm="L2",
        distance_convention="declared graph",
        clock="instrument_seconds",
        finite_horizon=2.0,
        tail_status="unobserved",
        provenance={
            name: False
            for name in (
                "uses_future_samples",
                "uses_outcome_derived_wiring",
                "fits_on_evaluation_data",
                "uses_evaluation_labels",
                "uses_postselection",
            )
        },
        certificate=certificate(),
        claim_statement="finite residual",
        claim_status="measured",
        scope="metadata and byte integrity only",
        assumptions=("fixed map",),
        outcome="fixture",
        source_imports=("tnfr.research.evidence_sidecar",),
        graph_context={"nodes": [0, 1]},
        state_context={"epi": [0.0, 1.0]},
        numerical_context={"precision": "float64"},
        observation_context={"calibration": {"scale": [1.0, 2.0]}},
        cost_context={"elapsed": 0.0},
        artifact_hashes={
            "response.json": "sha256:"
            + hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
    )
    values.update(changes)
    return EvidenceSidecar(**values)


@pytest.mark.parametrize(
    "value,tolerance,passed",
    [
        (1, 0, True),
        (1, 1, True),
        (-1, 1, True),
        (0.25, 0.5, False),
    ],
)
def test_inconsistent_strict_verdict_cannot_be_admitted(value, tolerance, passed):
    with pytest.raises(ValueError, match="inconsistent"):
        certificate(
            value=value, tolerance=tolerance, passed=passed
        ).validate_for_admission()


@pytest.mark.parametrize(
    "value,tolerance,passed",
    [
        (1, 1, False),
        (-1, 1, False),
        (0, 0, False),
        (0.25, 0.5, True),
    ],
)
def test_failed_or_boundary_certificate_is_valid_evidence(value, tolerance, passed):
    certificate(
        value=value, tolerance=tolerance, passed=passed
    ).validate_for_admission()


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", 1),
        ("passed", "false"),
        ("value", True),
        ("value", "0.25"),
        ("value", float("nan")),
        ("tolerance", float("inf")),
        ("tolerance", -1),
        ("backward_error", True),
        ("condition_number", float("inf")),
        ("quantity", " "),
        ("precision", []),
    ],
)
def test_certificate_rejects_ambiguous_or_nonfinite_fields(field, value):
    with pytest.raises(ValueError):
        certificate(**{field: value}).validate_for_admission()


def test_factory_does_not_erase_boolean_type_before_admission():
    with pytest.raises(ValueError, match="boolean"):
        certify_within_tolerance("residual", True, 2.0)


def test_core_admission_verifies_content_without_arithmetic_fields(tmp_path):
    evidence = sidecar(tmp_path)
    evidence.validate_metadata()
    evidence.validate_for_admission(root_dir=tmp_path)
    payload = evidence.to_dict()
    assert "uses_known_factors" not in payload["manifest"]
    assert "uses_known_factors" not in payload["provenance"]
    assert CoreExperimentManifest.from_dict(payload["manifest"]) == evidence.manifest
    with pytest.raises(TypeError, match="root_dir"):
        evidence.validate_for_admission()


def test_metadata_success_does_not_imply_actual_content_admission(tmp_path):
    evidence = sidecar(tmp_path)
    (tmp_path / "response.json").unlink()
    evidence.validate_metadata()
    with pytest.raises(EvidenceAdmissionError, match="existing file"):
        evidence.validate_for_admission(root_dir=tmp_path)


@pytest.mark.parametrize(
    "horizon", [float("nan"), float("inf"), -float("inf"), -1, True, "2"]
)
def test_nonfinite_or_ambiguous_horizon_is_not_admitted(tmp_path, horizon):
    with pytest.raises(EvidenceAdmissionError, match="finite_horizon"):
        sidecar(tmp_path, finite_horizon=horizon).validate_metadata()


def test_unassessed_horizon_is_distinct_from_nonfinite_horizon(tmp_path):
    sidecar(tmp_path, finite_horizon=None).validate_for_admission(root_dir=tmp_path)


@pytest.mark.parametrize("digest", ["", "sha256:example", "a" * 63, 12])
def test_hash_presence_without_valid_sha256_is_insufficient(tmp_path, digest):
    with pytest.raises(EvidenceAdmissionError, match="SHA-256"):
        sidecar(tmp_path, artifact_hashes={"response.json": digest}).validate_metadata()


def test_same_size_byte_change_invalidates_admission_and_prevents_export(tmp_path):
    evidence = sidecar(tmp_path)
    evidence.validate_for_admission(root_dir=tmp_path)
    (tmp_path / "response.json").write_bytes(b'{"prediction": 4}\n')
    destination = tmp_path / "sidecar.json"
    destination.write_text("old record", encoding="utf-8")
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch"):
        evidence.write_admitted(destination, root_dir=tmp_path)
    assert destination.read_text(encoding="utf-8") == "old record"


def test_every_manifest_artifact_needs_a_verified_digest(tmp_path):
    evidence = sidecar(tmp_path)
    evidence = replace(
        evidence,
        manifest=replace(
            evidence.manifest,
            artifacts=("response.json", "calibration.json"),
        ),
    )
    with pytest.raises(EvidenceAdmissionError, match="every declared artifact"):
        evidence.validate_metadata()


@pytest.mark.parametrize(
    "name", ["../outside.json", "/outside.json", "C:/outside.json", "a/../b", "a//b"]
)
def test_digest_path_cannot_escape_declared_root(tmp_path, name):
    with pytest.raises(EvidenceAdmissionError, match="within root_dir"):
        sidecar(tmp_path, artifact_hashes={name: "a" * 64}).validate_metadata()


def test_symlink_cannot_authorize_bytes_outside_root(tmp_path):
    evidence = sidecar(tmp_path)
    root = tmp_path / "nested"
    root.mkdir()
    try:
        (root / "response.json").symlink_to(tmp_path / "response.json")
    except OSError:
        pytest.skip("creating symlinks is unavailable on this platform")
    with pytest.raises(EvidenceAdmissionError, match="within root_dir"):
        evidence.validate_for_admission(root_dir=root)


def test_context_and_returned_payload_are_detached_and_deeply_frozen(tmp_path):
    source = {"calibration": {"scale": [1.0, 2.0]}}
    evidence = sidecar(tmp_path, observation_context=source)
    source["calibration"]["scale"][0] = 99.0
    assert evidence.observation_context["calibration"]["scale"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        evidence.observation_context["calibration"]["scale"][0] = 9.0
    with pytest.raises(TypeError):
        evidence.observation_context["calibration"]["scale"] = (9.0,)
    payload = evidence.to_dict()
    payload["observation_context"]["calibration"]["scale"][0] = 55.0
    payload["manifest"]["versions"]["python"] = "changed"
    assert evidence.to_dict()["observation_context"] == {
        "calibration": {"scale": [1.0, 2.0]}
    }
    assert evidence.manifest.versions["python"] == "3.13"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_nested_nonfinite_or_opaque_metadata_is_rejected(tmp_path, value):
    with pytest.raises(EvidenceAdmissionError, match="finite JSON"):
        sidecar(tmp_path, observation_context={"calibration": [value]})


def test_core_provenance_cannot_implicitly_default_or_use_arithmetic_answers(tmp_path):
    with pytest.raises(EvidenceAdmissionError, match="every admission field"):
        sidecar(tmp_path, provenance={"uses_known_factors": False}).validate_metadata()
    evidence = sidecar(tmp_path)
    provenance = dict(evidence.provenance)
    provenance["uses_future_samples"] = 0
    with pytest.raises(EvidenceAdmissionError, match="boolean"):
        replace(evidence, provenance=provenance).validate_metadata()


def test_declared_descriptive_usage_is_retained_not_relabelled_as_discovery(tmp_path):
    evidence = sidecar(tmp_path)
    provenance = dict(evidence.provenance)
    provenance["uses_evaluation_labels"] = True
    evidence = replace(evidence, provenance=provenance)
    evidence.validate_for_admission(root_dir=tmp_path)
    assert evidence.to_dict()["provenance"]["uses_evaluation_labels"] is True


def test_core_source_and_claim_metadata_cannot_disagree(tmp_path):
    evidence = sidecar(tmp_path)
    with pytest.raises(EvidenceAdmissionError, match="claim_status"):
        replace(evidence, claim_status="proved").validate_metadata()
    with pytest.raises(EvidenceAdmissionError, match="dirty_source_hash"):
        replace(evidence, dirty_source_hash="sha256:" + "a" * 64).validate_metadata()
    digest = "sha256:" + "a" * 64
    dirty = replace(evidence.manifest, source_dirty=True, dirty_source_hash=digest)
    replace(evidence, manifest=dirty, dirty_source_hash=digest).validate_metadata()


def test_legacy_manifest_is_cloned_and_its_versions_cannot_mutate_evidence(tmp_path):
    evidence = sidecar(tmp_path)
    manifest = ExperimentManifest(
        claim_id="NT.fixture",
        git_sha="deadbeef",
        versions={"python": "3.13"},
        artifacts=("response.json",),
        controls=("fixture",),
    )
    evidence = replace(
        evidence,
        manifest=manifest,
        dirty_source_hash="sha256:" + "a" * 64,
        provenance={
            "uses_known_factors": False,
            "uses_target_labels": False,
            "uses_expected_answers": False,
        },
    )
    manifest.versions["python"] = "changed externally"
    assert evidence.manifest.versions["python"] == "3.13"
    with pytest.raises(TypeError):
        evidence.manifest.versions["python"] = "changed directly"
    evidence.validate_for_admission(root_dir=tmp_path)


def test_export_preserves_payload_schema_and_cannot_overwrite_its_own_evidence(
    tmp_path,
):
    evidence = sidecar(tmp_path)
    output = tmp_path / "nested" / "sidecar.json"
    assert evidence.write_admitted(output, root_dir=tmp_path) == output
    assert json.loads(output.read_text(encoding="utf-8")) == evidence.to_dict()
    assert not list(output.parent.glob("*.tmp"))
    before = (tmp_path / "response.json").read_bytes()
    with pytest.raises(EvidenceAdmissionError, match="overwrite"):
        evidence.write_admitted(tmp_path / "response.json", root_dir=tmp_path)
    assert (tmp_path / "response.json").read_bytes() == before


def test_changed_artifact_during_serialization_does_not_commit(tmp_path, monkeypatch):
    evidence = sidecar(tmp_path)
    original = owner.json_dumps
    calls = 0

    def changing_dump(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:  # metadata validation precedes the export serialization
            (tmp_path / "response.json").write_bytes(b"changed")
        return result

    monkeypatch.setattr(owner, "json_dumps", changing_dump)
    output = tmp_path / "sidecar.json"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch"):
        evidence.write_admitted(output, root_dir=tmp_path)
    assert output.read_text(encoding="utf-8") == "previous"
