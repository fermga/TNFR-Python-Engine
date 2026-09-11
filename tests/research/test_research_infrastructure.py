r"""Tests for research claims, manifests, certificates, and circularity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tnfr.research import (
    CircularityAudit,
    CircularityVerdict,
    Claim,
    ClaimRegistry,
    ClaimStatus,
    CoreExperimentManifest,
    current_git_source_provenance,
    ClaimTransitionError,
    ExperimentManifest,
    EvidenceAdmissionError,
    EvidenceSidecar,
    ManifestValidationError,
    NumericalCertificate,
    certify_within_tolerance,
    input_bit_length,
    is_valid_transition,
)


def test_git_source_provenance_is_scoped_and_content_addressed(
    tmp_path, monkeypatch
):
    source = tmp_path / "src" / "model.py"
    source.parent.mkdir()
    source.write_text("version = 1\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, check, capture_output):
        calls.append((command, cwd, check, capture_output))
        arguments = command[1:]
        if arguments[:2] == ("rev-parse", "HEAD"):
            output = b"deadbeef\n"
        elif arguments[0] == "status":
            output = b"?? src/model.py\0"
        elif arguments[0] == "ls-files":
            output = b"src/model.py\0"
        else:  # pragma: no cover - guards the fixed Git protocol
            raise AssertionError(arguments)
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr("tnfr.research.core_manifests.subprocess.run", fake_run)

    first = current_git_source_provenance(tmp_path, ("src",))
    source.write_text("version = 2\n", encoding="utf-8")
    second = current_git_source_provenance(tmp_path, ("src",))

    assert first[0] == "deadbeef"
    assert first[1] is True
    assert first[2].startswith("sha256:")
    assert first[2] != second[2]
    assert all(call[1] == tmp_path.resolve() for call in calls)


def test_git_source_provenance_rejects_an_ambiguous_path_scalar(tmp_path):
    with pytest.raises(TypeError, match="iterable"):
        current_git_source_provenance(tmp_path, "src")


def _core_manifest(**overrides):
    base = dict(
        claim_id="S16",
        git_sha="deadbeef",
        versions={"tnfr": "0.0.3.5", "python": "3.13.5"},
        graph_construction="weighted path graph with declared node attributes",
        capacity_specification="positive heterogeneous nu_f in [0.5, 2.0]",
        solver="read-only exact finite-dimensional certificate",
        result_status=ClaimStatus.DERIVED,
        seed=20260906,
        telemetry=("C(t)", "Si", "nu_f", "phase", "tetrad"),
        controls=("relabeling", "boundary partition"),
        artifacts=("results/s16.json",),
        source_dirty=False,
    )
    base.update(overrides)
    return CoreExperimentManifest(**base)


def test_core_manifest_is_domain_neutral_and_round_trips():
    manifest = _core_manifest()
    payload = manifest.to_dict()

    assert "uses_known_factors" not in payload
    assert "input_bits" not in payload
    assert CoreExperimentManifest.from_dict(payload) == manifest
    manifest.validate_for_admission()


@pytest.mark.parametrize(
    "overrides",
    [
        {"result_status": "unknown"},
        {"result_status": None},
        {"git_sha": "not-a-sha"},
        {"seed": True},
        {"timestep": True},
        {"timestep": "0.1"},
        {"timestep": 10**1000},
        {"timestep": 0.0},
        {"versions": {"python": 3}},
        {"operator_sequence": "AL"},
        {"telemetry": "C(t)"},
        {"controls": "relabeling"},
        {"artifacts": "results/s16.json"},
        {"source_dirty": "false"},
        {"dirty_source_hash": "sha256:not-a-digest"},
    ],
)
def test_core_manifest_rejects_ambiguous_provenance(overrides):
    with pytest.raises(ManifestValidationError):
        _core_manifest(**overrides)


def test_core_manifest_frozen_contract_includes_versions_mapping():
    manifest = _core_manifest()

    with pytest.raises(TypeError):
        manifest.versions["python"] = "mutated"  # type: ignore[index]
    assert manifest.to_dict()["versions"]["python"] == "3.13.5"


def test_core_manifest_from_dict_normalizes_invalid_status_error():
    payload = _core_manifest().to_dict()
    payload["result_status"] = None

    with pytest.raises(ManifestValidationError, match="result_status"):
        CoreExperimentManifest.from_dict(payload)


def test_core_manifest_requires_dirty_tree_hash_for_admission():
    dirty = _core_manifest(source_dirty=True)
    with pytest.raises(ManifestValidationError, match="dirty_source_hash"):
        dirty.validate_for_admission()

    admitted = _core_manifest(
        source_dirty=True,
        dirty_source_hash="sha256:" + "a" * 64,
    )
    admitted.validate_for_admission()
    assert CoreExperimentManifest.from_dict(admitted.to_dict()) == admitted


# --- claims -------------------------------------------------------------------


def test_claim_status_strengthening_ladder():
    assert is_valid_transition(ClaimStatus.CONJECTURAL, ClaimStatus.MEASURED)
    assert is_valid_transition(ClaimStatus.MEASURED, ClaimStatus.DERIVED)
    assert is_valid_transition(ClaimStatus.DERIVED, ClaimStatus.PROVED)
    assert is_valid_transition(ClaimStatus.CONJECTURAL, ClaimStatus.PROVED)


def test_claim_can_always_be_corrected():
    for status in ClaimStatus:
        if status in (ClaimStatus.NEGATIVE, ClaimStatus.SUPERSEDED):
            continue
        assert is_valid_transition(status, ClaimStatus.NEGATIVE)
        assert is_valid_transition(status, ClaimStatus.SUPERSEDED)


def test_claim_cannot_silently_weaken():
    assert not is_valid_transition(ClaimStatus.PROVED, ClaimStatus.CONJECTURAL)
    assert not is_valid_transition(ClaimStatus.DERIVED, ClaimStatus.MEASURED)
    claim = Claim("NT-X", "example", ClaimStatus.PROVED)
    with pytest.raises(ClaimTransitionError):
        claim.with_status(ClaimStatus.CONJECTURAL)


def test_claim_promotion_requires_justification_and_proof_reference():
    claim = Claim("NT-X", "example", ClaimStatus.MEASURED)
    with pytest.raises(ClaimTransitionError):
        claim.promote(ClaimStatus.DERIVED, justification="", proof_references=())
    promoted = claim.promote(
        ClaimStatus.DERIVED,
        justification="exact finite-dimensional identity",
        proof_references=("theory/example.md",),
    )
    assert promoted.status is ClaimStatus.DERIVED
    assert promoted.references == ("theory/example.md",)


def test_terminal_statuses_do_not_transition():
    assert not is_valid_transition(ClaimStatus.NEGATIVE, ClaimStatus.MEASURED)
    assert not is_valid_transition(ClaimStatus.SUPERSEDED, ClaimStatus.PROVED)


def test_claim_round_trip():
    claim = Claim(
        "NT-C04",
        "eta^2 > 0.9 recovers factors canonically",
        ClaimStatus.SUPERSEDED,
        references=("theory/TNFR_NUMBER_THEORY.md §9.9",),
    )
    assert Claim.from_dict(claim.to_dict()) == claim


def test_claim_registry_transition_supersedes():
    registry = ClaimRegistry()
    registry.add(Claim("NT-C04", "eta^2>0.9 recovery", ClaimStatus.MEASURED))
    registry.transition("NT-C04", ClaimStatus.SUPERSEDED)
    assert registry.get("NT-C04").status is ClaimStatus.SUPERSEDED
    restored = ClaimRegistry.from_dict(registry.to_dict())
    assert restored.get("NT-C04").status is ClaimStatus.SUPERSEDED


def test_claim_registry_rejects_duplicates():
    registry = ClaimRegistry()
    registry.add(Claim("NT-1", "s", ClaimStatus.MEASURED))
    with pytest.raises(ValueError):
        registry.add(Claim("NT-1", "s", ClaimStatus.MEASURED))


# --- manifests ----------------------------------------------------------------


def _manifest(**overrides):
    base = dict(
        claim_id="NT-P02",
        git_sha="deadbeef",
        versions={"tnfr": "0.0.3.5", "python": "3.13.5"},
        seed=20260904,
        operator_sequence=("AL", "UM", "RA", "IL", "SHA"),
        uses_known_factors=False,
        input_bits=8,
        controls=("relabel", "shuffle"),
        artifacts=("results/pulse.json",),
    )
    base.update(overrides)
    return ExperimentManifest(**base)


@pytest.mark.parametrize("missing", ["claim_id", "git_sha"])
def test_manifest_required_fields(missing):
    with pytest.raises(ManifestValidationError):
        _manifest(**{missing: ""})


def test_manifest_requires_versions():
    with pytest.raises(ManifestValidationError):
        _manifest(versions={})


def test_manifest_round_trip():
    manifest = _manifest()
    assert ExperimentManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_known_factor_flag_survives_round_trip():
    manifest = _manifest(uses_known_factors=True)
    assert manifest.uses_known_factors is True
    assert ExperimentManifest.from_dict(manifest.to_dict()).uses_known_factors


def test_strict_manifest_admission_requires_explicit_provenance():
    manifest = _manifest()
    manifest.validate_for_admission()
    data = manifest.to_dict()
    data.pop("uses_known_factors")
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(data, strict=True)
    data = _manifest().to_dict()
    data["uses_known_factors"] = "false"
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(data, strict=True)


def test_strict_manifest_rejects_old_bit_length_wording():
    with pytest.raises(ManifestValidationError):
        _manifest(input_size_model="L = ceil(log2 n) bits").validate_for_admission()


def test_input_bit_length():
    assert input_bit_length(1) == 1
    assert input_bit_length(255) == 8
    assert input_bit_length(256) == 9
    with pytest.raises(ValueError):
        input_bit_length(0)


# --- certificates -------------------------------------------------------------


def test_numerical_certificate_pass_and_fail():
    ok = certify_within_tolerance("residual", 1e-15, 1e-8)
    bad = certify_within_tolerance("residual", 5e-2, 1e-8)
    assert ok.passed and not bad.passed
    assert NumericalCertificate(**{
        k: v for k, v in ok.to_dict().items()
    }) == ok


def test_strict_numerical_admission_requires_error_context():
    with pytest.raises(ValueError):
        certify_within_tolerance("residual", 1e-15, 1e-8).validate_for_admission()
    cert = certify_within_tolerance(
        "residual", 1e-15, 1e-8,
        backward_error=1e-15,
        condition_number=2.0,
    )
    cert.validate_for_admission()
    with pytest.raises(ValueError):
        certify_within_tolerance(
            "residual", 1e-15, 1e-8,
            backward_error=1e-15, condition_number=2.0,
            precision="magic128",
        ).validate_for_admission()


def _sidecar(**overrides):
    values = dict(
        manifest=_manifest(),
        artifact="results/example.json",
        model="fixed linear EPI channel",
        norm="euclidean",
        distance_convention="outgoing weighted shortest path",
        clock="structural_time",
        finite_horizon=4.0,
        tail_status="UNASSESSED_FINITE_WINDOW",
        provenance={
            "uses_known_factors": False,
            "uses_target_labels": False,
            "uses_expected_answers": False,
        },
        claim_statement="finite residual is below declared tolerance",
        claim_status="measured",
        scope="finite four-node fixture",
        assumptions=("fixed graph", "float64"),
        outcome="measured residual",
        source_imports=("tnfr.physics.structural_morphism",),
        dirty_source_hash="sha256:example",
        graph_context={"family": "path", "nodes": 4, "directed": False},
        state_context={"initial_triads": "fixture", "operator_word": []},
        numerical_context={"precision": "float64", "conditioning": 2.0},
        observation_context={"observable": "residual", "aggregation": "max"},
        cost_context={"construction_seconds": 0.0, "solve_seconds": 0.0},
        artifact_hashes={"results/example.json": "sha256:example"},
    )
    values.update(overrides)
    return EvidenceSidecar(**values)


def test_evidence_sidecar_requires_complete_context_and_serializes():
    sidecar = _sidecar(
        certificate=certify_within_tolerance(
            "residual", 1e-15, 1e-8,
            backward_error=1e-15,
            condition_number=2.0,
        ),
    )
    sidecar.validate_for_admission()
    assert sidecar.to_dict()["provenance"]["uses_known_factors"] is False


def test_evidence_sidecar_rejects_missing_provenance_answer():
    sidecar = _sidecar(provenance={"uses_known_factors": False})
    with pytest.raises(EvidenceAdmissionError):
        sidecar.validate_for_admission()


def test_evidence_sidecar_validates_before_export(tmp_path):
    sidecar = _sidecar()
    destination = sidecar.write_admitted(tmp_path / "sidecar.json")
    assert destination.exists()


def test_evidence_sidecar_rejects_missing_context_group():
    with pytest.raises(EvidenceAdmissionError):
        _sidecar(cost_context={}).validate_for_admission()


# --- circularity --------------------------------------------------------------


def test_circularity_structural():
    audit = CircularityAudit()
    assert audit.verdict is CircularityVerdict.STRUCTURAL
    assert audit.permits_discovery_claim


def test_circularity_descriptive_when_ground_truth_used_for_scoring():
    audit = CircularityAudit(factors_used_only_for_scoring=True)
    assert audit.verdict is CircularityVerdict.DESCRIPTIVE
    assert not audit.permits_discovery_claim


def test_circularity_circular_when_construction_requires_answer():
    audit = CircularityAudit(graph_construction_requires_answer=True)
    assert audit.verdict is CircularityVerdict.CIRCULAR
    assert not audit.permits_discovery_claim


def test_circularity_circular_when_features_call_factorization():
    audit = CircularityAudit(uses_factorization_in_features=True)
    assert audit.verdict is CircularityVerdict.CIRCULAR
