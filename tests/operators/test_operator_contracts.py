"""Tests pinning the canonical operator-contract specification.

These tests make :mod:`tnfr.operators.operator_contracts` the single source of
truth for operator contracts: they assert the spec is self-consistent, that the
proactive audit derives its catalog from the spec, and that the public English
names are exposed everywhere a contract is surfaced.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import warnings

import pytest

from tnfr.operators.grammar_types import FUNCTION_TO_GLYPH
from tnfr.operators.operator_contracts import (
    OPERATOR_CONTRACTS,
    ContractContext,
    EffectDirection,
    OperatorContract,
    OperatorScale,
    StateChannel,
    contract_identifiability_certificate,
    contract_for,
    english_name,
    iter_contracts,
    operators_at_scale,
    operators_in_channel,
    verify_contract_consistency,
)
from tnfr.operators.remesh import apply_network_remesh


class TestSpecSelfConsistency:
    """The spec self-check and structural partitions."""

    def test_verify_contract_consistency_passes(self) -> None:
        verify_contract_consistency()

    def test_covers_all_13_operators(self) -> None:
        assert set(OPERATOR_CONTRACTS) == set(FUNCTION_TO_GLYPH)
        assert len(OPERATOR_CONTRACTS) == 13

    def test_channel_partition(self) -> None:
        assert set(operators_in_channel(StateChannel.EPI)) == {
            "Emission",
            "Reception",
            "Resonance",
            "Recursivity",
        }
        assert set(operators_in_channel(StateChannel.NU_F)) == {
            "Silence",
            "Expansion",
            "Contraction",
        }
        assert set(operators_in_channel(StateChannel.THETA)) == {"Coupling", "Mutation"}
        assert set(operators_in_channel(StateChannel.DELTA_NFR)) == {
            "Coherence",
            "Dissonance",
            "SelfOrganization",
            "Transition",
        }

    def test_scale_partition_is_u5_fractality(self) -> None:
        # Exactly one NETWORK-scale operator: REMESH (operational fractality, U5).
        assert set(operators_at_scale(OperatorScale.NETWORK)) == {"Recursivity"}
        assert len(operators_at_scale(OperatorScale.NODE)) == 12

    def test_network_scale_iff_advisory(self) -> None:
        for c in iter_contracts():
            is_network = c.scale is OperatorScale.NETWORK
            assert is_network == (c.context is ContractContext.ADVISORY)

    def test_glyph_matches_grammar_types(self) -> None:
        for name, c in OPERATOR_CONTRACTS.items():
            assert c.glyph == FUNCTION_TO_GLYPH[name].value

    def test_contract_features_leave_silence_and_contraction_ambiguous(self) -> None:
        certificate = contract_identifiability_certificate()
        assert certificate.ambiguous_groups == (("Silence", "Contraction"),)
        assert len(certificate.uniquely_identifiable) == 11
        assert not certificate.all_operators_identifiable

    def test_identifiability_features_cannot_include_names_or_duplicates(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            contract_identifiability_certificate(("english_name",))
        with pytest.raises(ValueError, match="duplicates"):
            contract_identifiability_certificate(("scale", "scale"))


class TestGroundTruthChannels:
    """The canonical channel of each operator matches the ground-truth effect."""

    def test_emission_is_epi_increase(self) -> None:
        c = contract_for("emission")
        assert c.primary_channel is StateChannel.EPI
        assert c.primary_direction is EffectDirection.INCREASE

    def test_resonance_is_epi_identity(self) -> None:
        # RA propagates EPI preserving identity (NOT a magnitude bound).
        c = contract_for("resonance")
        assert c.primary_channel is StateChannel.EPI
        assert c.context is ContractContext.IDENTITY

    def test_expansion_and_contraction_are_nu_f(self) -> None:
        # VAL/NUL act on the νf capacity channel (ground truth: _make_scale_op).
        assert contract_for("expansion").primary_channel is StateChannel.NU_F
        assert contract_for("contraction").primary_channel is StateChannel.NU_F

    def test_self_organization_reorganizes_pressure(self) -> None:
        """THOL's U2 role must not be represented as monotone pressure growth."""
        contract = contract_for("self_organization")
        assert contract.primary_channel is StateChannel.DELTA_NFR
        assert contract.primary_direction is EffectDirection.REORGANIZE

    def test_recursivity_is_epi_at_network_scale(self) -> None:
        # REMESH is an EPI operator (echoes the form) at NETWORK scale (U5).
        c = contract_for("recursivity")
        assert c.primary_channel is StateChannel.EPI
        assert c.scale is OperatorScale.NETWORK

    def test_network_remesh_uses_delayed_epi_history(self) -> None:
        """The network-scale Recursivity path mixes declared EPI history."""
        graph = nx.path_graph(2)
        graph.graph.update(
            REMESH_TAU_GLOBAL=2,
            REMESH_TAU_LOCAL=1,
            REMESH_ALPHA=0.25,
            REMESH_ALPHA_HARD=True,
            EPI_MIN=-100.0,
            EPI_MAX=100.0,
        )
        for node, value in enumerate((1.0, -2.0)):
            graph.nodes[node]["EPI"] = value
        graph.graph["_epi_hist"] = deque([
            {0: 4.0, 1: 8.0},
            {0: 3.0, 1: 6.0},
            {0: 2.0, 1: 5.0},
        ])

        apply_network_remesh(graph)

        alpha = 0.25
        assert graph.nodes[0]["EPI"] == pytest.approx(
            (1.0 - alpha) ** 2 * 1.0
            + alpha * (1.0 - alpha) * 3.0
            + alpha * 4.0
        )
        assert graph.nodes[1]["EPI"] == pytest.approx(
            (1.0 - alpha) ** 2 * -2.0
            + alpha * (1.0 - alpha) * 6.0
            + alpha * 8.0
        )


class TestPublicEnglishNames:
    """The public level exposes English structural-operator names, not glyphs."""

    def test_english_name_resolves_from_glyph_and_name(self) -> None:
        assert english_name("AL") == "Emission"
        assert english_name("emission") == "Emission"
        assert english_name("Emission") == "Emission"

    def test_every_contract_has_a_capitalized_english_name(self) -> None:
        for c in iter_contracts():
            assert c.english_name[0].isupper()
            # The English name is not the glyph code.
            assert c.english_name != c.glyph

    def test_contract_for_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            contract_for("not_an_operator")


class TestAuditDerivesFromSpec:
    """The proactive audit catalog is derived from the spec, not hardcoded."""

    def test_audit_passes_all_13(self) -> None:
        from tnfr.physics.integrity import audit_operator_contracts

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audit = audit_operator_contracts()
        assert audit.all_satisfied
        assert audit.n_operators == 13

    def test_audit_uses_spec_contract_text_and_english_names(self) -> None:
        from tnfr.physics.integrity import audit_operator_contracts

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audit = audit_operator_contracts()
        for r in audit.results:
            spec = OPERATOR_CONTRACTS[r.operator]
            assert r.contract == spec.postcondition
            assert r.context == spec.context.value
            assert r.english_name == spec.english_name


class TestContractIsImmutable:
    """Contracts are frozen dataclasses (cannot drift at runtime)."""

    def test_contract_is_frozen(self) -> None:
        c = contract_for("emission")
        assert isinstance(c, OperatorContract)
        with pytest.raises(Exception):
            c.postcondition = "tampered"  # type: ignore[misc]
