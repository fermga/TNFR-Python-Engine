r"""Tests for the C5 research infrastructure (claims, manifests, certificates,
circularity)."""

from __future__ import annotations

import pytest

from tnfr.research import (
    CircularityAudit,
    CircularityVerdict,
    Claim,
    ClaimRegistry,
    ClaimStatus,
    ClaimTransitionError,
    ExperimentManifest,
    ManifestValidationError,
    NumericalCertificate,
    certify_within_tolerance,
    input_bit_length,
    is_valid_transition,
)


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
