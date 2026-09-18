"""Grammar basis metadata must not become physical execution evidence."""

import importlib
import json
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from tnfr.operators import grammar_canon as gc
from tnfr.sdk.simple import TNFR


def test_compositional_bases_cover_rules_without_changing_historical_schema():
    assert set(gc.GRAMMAR_BASES) == {item.rule_id for item in gc.GRAMMAR_RULES}
    assert tuple(field.name for field in fields(gc.GrammarRule)) == (
        "rule_id",
        "name",
        "physics",
        "operator_sets",
        "invariant",
        "pdf_reference",
    )
    kinds = set()
    for rule in gc.GRAMMAR_RULES:
        bases = gc.grammar_basis(rule.rule_id)
        assert isinstance(bases, tuple)
        assert len(bases) >= 2
        assert any(basis.kind is gc.GrammarBasisKind.POLICY for basis in bases)
        for basis in bases:
            kinds.add(basis.kind)
            assert basis.statement and basis.owner and basis.hypotheses
            assert (
                basis.as_dict()["scope"] == "declarative_basis_not_execution_evidence"
            )
    assert kinds == set(gc.GrammarBasisKind)
    with pytest.raises(KeyError):
        gc.grammar_basis("unknown")


def test_declared_owners_resolve_without_executing_them():
    root = Path(__file__).resolve().parents[2]
    for bases in gc.GRAMMAR_BASES.values():
        for basis in bases:
            if basis.owner.startswith("theory/"):
                assert (root / basis.owner.split("#")[0]).is_file()
                continue
            parts = basis.owner.split(".")
            # Owners may be module-level functions or methods on a class.
            for boundary in range(len(parts) - 1, 0, -1):
                try:
                    owner = importlib.import_module(".".join(parts[:boundary]))
                except ModuleNotFoundError as exc:
                    if exc.name != ".".join(parts[:boundary]):
                        raise
                    continue
                for name in parts[boundary:]:
                    owner = getattr(owner, name)
                assert callable(owner)
                break
            else:
                pytest.fail(f"unresolved declared owner: {basis.owner}")


def test_basis_detaches_nested_input_and_returned_metadata():
    hypotheses = ["finite input"]
    choices = [["limit", "selected value"]]
    basis = gc.GrammarBasis(
        "policy", "A declared gate.", hypotheses, choices, "declared.owner"
    )
    hypotheses.append("injected")
    choices[0][1] = "altered"
    assert basis.hypotheses == ("finite input",)
    assert basis.configured_choices == (("limit", "selected value"),)
    with pytest.raises(FrozenInstanceError):
        basis.statement = "changed"
    with pytest.raises(TypeError):
        gc.GRAMMAR_BASES["U3"] = ()
    payload = basis.as_dict()
    payload["hypotheses"].append("injected")
    payload["configured_choices"]["limit"] = "changed"
    assert basis.as_dict()["configured_choices"] == {"limit": "selected value"}
    assert basis.hypotheses == ("finite input",)


@pytest.mark.parametrize("bad", ["proved", "measured", "conjectural"])
def test_claim_status_is_not_a_basis_kind(bad):
    with pytest.raises(ValueError):
        gc.GrammarBasis(bad, "Claim label", ("premise",), (), "owner")


@pytest.mark.parametrize("choices", [["ab"], [("x", "1"), ("x", "2")]])
def test_malformed_configured_choices_are_rejected(choices):
    with pytest.raises((TypeError, ValueError)):
        gc.GrammarBasis("policy", "Gate", ("premise",), choices, "owner")


LEGACY_ROLES = {
    "AL": ["generator"],
    "EN": [],
    "IL": ["stabilizer"],
    "OZ": ["closure", "destabilizer"],
    "UM": ["coupling/resonance"],
    "RA": ["coupling/resonance"],
    "SHA": ["closure"],
    "VAL": ["destabilizer"],
    "NUL": [],
    "THOL": ["stabilizer", "transformer"],
    "ZHIR": ["destabilizer", "transformer"],
    "NAV": ["generator", "closure"],
    "REMESH": ["generator", "closure"],
}


def test_sdk_preserves_legacy_roles_and_exposes_all_nine_canonical_roles():
    found = set()
    for row in TNFR.operators():
        glyph = row["glyph"]
        assert row["roles"] == LEGACY_ROLES[glyph]
        canonical = gc.operator_role_metadata(glyph)
        for key in ("canonical_roles", "u_rules", "grammar_basis"):
            assert row[key] == canonical[key]
        found.update(row["canonical_roles"])
        assert TNFR.operators(glyph) == row
    assert found == {role.value for role in gc.GrammarRole}
    assert "handler" in TNFR.operators("IL")["canonical_roles"]
    assert "trigger" in TNFR.operators("OZ")["canonical_roles"]
    assert "recursive" in TNFR.operators("REMESH")["canonical_roles"]


@pytest.mark.parametrize(
    "word, valid",
    [
        (["AL", "UM", "IL", "SHA"], True),
        (["IL", "RA"], False),
    ],
)
def test_word_result_never_claims_state_or_trajectory_admission(word, valid):
    result = TNFR.explain_sequence(word)
    assert result["valid"] is valid
    assert result["validation_scope"] == "word_policy"
    assert result["state_checks_assessed"] is False
    assert result["trajectory_checks_assessed"] is False
    assert result["execution_admission_assessed"] is False
    assert "live_operator_preconditions" in result["missing_evidence"]
    assert "aligned_potential_reference" in result["missing_evidence"]
    assert "full_tetrad_trajectory" in result["missing_evidence"]
    assert "full_state_closure" in result["missing_evidence"]
    assert set(result["grammar_basis"]) == set(gc.GRAMMAR_BASES)
    for row in result["roles"]:
        assert row == {
            "name": row["name"],
            "glyph": row["glyph"],
            **gc.operator_role_metadata(row["glyph"]),
        }
    json.dumps(result, allow_nan=False)


def test_changing_declarative_basis_cannot_make_an_invalid_word_admissible(monkeypatch):
    word = ["IL", "RA"]
    before = TNFR.explain_sequence(word)
    claimed = tuple(
        replace(
            item,
            kind=gc.GrammarBasisKind.IDENTITY,
            statement="Caller labels this an identity.",
        )
        for item in gc.grammar_basis("U1a")
    )
    monkeypatch.setattr(
        gc, "GRAMMAR_BASES", MappingProxyType({**gc.GRAMMAR_BASES, "U1a": claimed})
    )
    after = TNFR.explain_sequence(word)
    assert before["valid"] is after["valid"] is False
    assert before["message"] == after["message"]
    assert after["execution_admission_assessed"] is False
    assert before["missing_evidence"] == after["missing_evidence"]


def test_basis_retains_smooth_acceleration_and_calibration_scope():
    acceleration = gc.grammar_basis("U4a")[0]
    assert "no jump at the differentiation point" in acceleration.hypotheses
    assert "differentiable capacity and pressure" in acceleration.hypotheses
    calibration = gc.grammar_basis("U4b")[0]
    assert "scalar relaxation surrogate" in calibration.hypotheses
    assert "64-step cap" in dict(calibration.configured_choices)["implementation_scope"]


def test_u3_u6_retain_full_tetrad_scope_without_claiming_state_closure():
    for rule_id in ("U3", "U6"):
        (tetrad,) = (
            item
            for item in gc.grammar_basis(rule_id)
            if item.owner == "tnfr.metrics.observations.observe_graph_tetrad"
        )
        assert tetrad.kind is gc.GrammarBasisKind.CONTRACT
        assert "Phi_s" in tetrad.statement
        assert "phase gradient/curvature" in tetrad.statement
        assert "nonlocal coherence length xi_C" in tetrad.statement
        assert (
            "separate state-closure evidence for dynamical prediction"
            in tetrad.hypotheses
        )
