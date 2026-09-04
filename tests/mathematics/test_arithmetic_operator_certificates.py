r"""Tests for the R8 arithmetic transformation-to-operator certification.

Two candidates certify to canonical operators with a valid grammar word and
contract-consistent channel/scale (emission at zero → Emission; residue-edge
propagation → Resonance); four are rejected with an explicit reason (CRT
projection, p-adic lift, affine map, power map). No fourteenth operator is
invented.
"""

from __future__ import annotations

import pytest

from tnfr.mathematics import operator_certificates as oc
from tnfr.mathematics.operator_certificates import (
    ArithmeticOperatorCertificate,
    all_certificates,
    certified_mappings,
    certify_emission_at_zero,
    certify_residue_edge_propagation,
    reject_affine_map,
    reject_crt_projection,
    reject_padic_lift,
    reject_power_map,
    rejected_mappings,
    verify_certificate,
)
from tnfr.operators.grammar import validate_sequence
from tnfr.operators.operator_contracts import contract_for


# --------------------------------------------------------------------------- #
# Required test 1: contract residuals (positive certificates)
# --------------------------------------------------------------------------- #
def test_emission_at_zero_certifies_to_emission():
    cert = certify_emission_at_zero()
    assert cert.certified
    assert cert.canonical_operator == "Emission"
    contract = contract_for("Emission")
    assert cert.state_channel == contract.primary_channel.value
    assert cert.scale == contract.scale.value
    assert cert.residuals["epi_direction_violation"] == 0.0
    assert cert.residuals["delta_epi"] > 0.0


def test_residue_propagation_certifies_to_resonance():
    cert = certify_residue_edge_propagation()
    assert cert.certified
    assert cert.canonical_operator == "Resonance"
    contract = contract_for("Resonance")
    assert cert.state_channel == contract.primary_channel.value
    # additive Cayley transport conserves total EPI (identity preserved)
    assert cert.residuals["conservation_defect"] < 1e-9
    assert cert.residuals["neighbors_reached"] > 0.0


def test_positive_certificates_have_u3_or_stabilizer_preconditions():
    prop = certify_residue_edge_propagation()
    assert any("U3" in pre for pre in prop.preconditions)


# --------------------------------------------------------------------------- #
# Required test 2: grammar word validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [certify_emission_at_zero,
                                     certify_residue_edge_propagation])
def test_certified_grammar_words_are_valid(factory):
    cert = factory()
    assert cert.grammar_valid
    assert validate_sequence(list(cert.grammar_word)).passed


def test_rejected_certificates_have_no_grammar_word():
    for cert in rejected_mappings():
        assert cert.grammar_word == ()
        assert cert.grammar_valid is False


# --------------------------------------------------------------------------- #
# Required test 3: negative mapping cases (the boundary result)
# --------------------------------------------------------------------------- #
def test_crt_projection_is_rejected_as_relabeling():
    cert = reject_crt_projection()
    assert cert.rejected
    assert cert.canonical_operator is None
    assert cert.residuals["is_bijection"] == 1.0
    assert cert.residuals["channel_modification"] == 0.0
    assert "relabeling" in cert.rejection_reason


def test_padic_lift_is_rejected_as_unverified_remesh():
    cert = reject_padic_lift()
    assert cert.rejected
    # transport-consistent but REMESH contract unmet (all four conditions)
    assert cert.residuals["projective_commutation"] == 0.0
    assert cert.residuals["remesh_conditions_unmet"] == 4.0
    assert "REMESH" in cert.rejection_reason


def test_affine_map_is_rejected_as_automorphism():
    cert = reject_affine_map()
    assert cert.rejected
    assert cert.residuals["is_bijection"] == 1.0
    assert cert.residuals["channel_modification"] == 0.0
    assert "automorphism" in cert.rejection_reason


def test_power_map_is_rejected_as_endomorphism():
    cert = reject_power_map(3, 13)
    assert cert.rejected
    # x -> x^3 mod 13: image is the 4-element cubic-residue subgroup of 12
    assert cert.residuals["many_to_one_factor"] == 3.0
    assert cert.residuals["image_fraction"] == pytest.approx(4 / 12)
    assert "endomorphism" in cert.rejection_reason


def test_no_fourteenth_operator_is_invented():
    # every certified mapping targets one of the 13 canonical operators
    for cert in certified_mappings():
        contract_for(cert.canonical_operator)  # raises if not canonical


# --------------------------------------------------------------------------- #
# Aggregation and verification
# --------------------------------------------------------------------------- #
def test_audit_has_two_positive_four_negative():
    certs = all_certificates()
    assert len(certs) == 6
    assert len(certified_mappings()) == 2
    assert len(rejected_mappings()) == 4


def test_all_certificates_verify():
    assert all(verify_certificate(c) for c in all_certificates())


def test_verify_rejects_inconsistent_channel():
    cert = certify_emission_at_zero()
    broken = ArithmeticOperatorCertificate(
        transformation=cert.transformation,
        canonical_operator="Emission",
        state_channel="theta",  # wrong channel for Emission (EPI)
        scale=cert.scale,
        preconditions=cert.preconditions,
        postconditions=cert.postconditions,
        grammar_word=cert.grammar_word,
        residuals=cert.residuals,
    )
    assert verify_certificate(broken) is False


def test_verify_requires_reason_for_rejection():
    reasonless = ArithmeticOperatorCertificate(
        transformation="x -> x (mod n)",
        canonical_operator=None,
        state_channel="none",
        scale="none",
        preconditions=(),
        postconditions=(),
        grammar_word=(),
        residuals={},
        rejection_reason="",
    )
    assert verify_certificate(reasonless) is False


def test_module_exports_complete():
    expected = {
        "ArithmeticOperatorCertificate",
        "certify_emission_at_zero",
        "certify_residue_edge_propagation",
        "reject_crt_projection",
        "reject_padic_lift",
        "reject_affine_map",
        "reject_power_map",
        "all_certificates",
        "certified_mappings",
        "rejected_mappings",
        "verify_certificate",
    }
    assert expected <= set(oc.__all__)
